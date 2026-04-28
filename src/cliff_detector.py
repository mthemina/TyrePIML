import torch
import numpy as np
import pandas as pd
from src.model import TyreLSTM
from src.dataset import COMPOUND_MAP, normalize, denormalize, LAP_TIME_MIN, LAP_TIME_MAX, TYRE_LIFE_MIN, TYRE_LIFE_MAX, SECTOR_MIN, SECTOR_MAX
from src.model_router import get_best_model  # Injecting our new smart router
from src.driver_profiles import get_driver_style_encoding 

# A lap is considered a cliff if it's this many seconds slower than the stint average
CLIFF_THRESHOLD = 1.5

def prepare_sequence(model, stint_df, sequence_length=8):
    """
    Convert a stint dataframe into model input sequences.
    Dynamically scales feature dimensions based on model architecture (7, 8, or 9 features).
    """
    from src.track_profiles import get_track_profile
    from src.dataset import (TEMP_MIN, TEMP_MAX, 
                             ABRASIVENESS_MIN, ABRASIVENESS_MAX)
    from src.driver_profiles import get_driver_style_encoding
    
    # Dynamically read what the model expects
    # Works for both LSTM and Transformer
    if hasattr(model, 'lstm'):
        expected_features = model.lstm.input_size
    elif hasattr(model, 'input_projection'):
        expected_features = model.input_projection.in_features
    else:
        expected_features = 9  # default 
    
    has_weather = 'track_temp_avg' in stint_df.columns
    track_temp = float(stint_df['track_temp_avg'].iloc[0]) if has_weather else 35.0
    air_temp = float(stint_df['air_temp_avg'].iloc[0]) if has_weather else 28.0
    
    abrasiveness = 5.0
    if 'Event' in stint_df.columns:
        event = stint_df['Event'].iloc[0]
        profile = get_track_profile(event)
        abrasiveness = profile['abrasiveness']
        
    driver_name = stint_df['Driver'].iloc[0] if 'Driver' in stint_df.columns else None
    driver_encoding = get_driver_style_encoding(driver_name) if driver_name else 0.5
    
    features = []
    for _, row in stint_df.iterrows():
        # 1. The core 7 features every model has (v1 compound models)
        feat = [
            normalize(row['TyreLife'], TYRE_LIFE_MIN, TYRE_LIFE_MAX),
            COMPOUND_MAP[row['Compound']] / 2.0,
            normalize(row['Sector1Time'], SECTOR_MIN, SECTOR_MAX),
            normalize(row['Sector2Time'], SECTOR_MIN, SECTOR_MAX),
            normalize(row['Sector3Time'], SECTOR_MIN, SECTOR_MAX),
            normalize(track_temp, TEMP_MIN, TEMP_MAX),
            normalize(air_temp, TEMP_MIN, TEMP_MAX),
        ]
        
        # 2. The 8th feature (Track Abrasiveness - added in later models)
        if expected_features >= 8:
            feat.append(normalize(abrasiveness, ABRASIVENESS_MIN, ABRASIVENESS_MAX))
            
        # 3. The 9th feature (Driver Profile - added in v2 models)
        if expected_features >= 9:
            if 'DriverEncoded' in stint_df.columns:
                feat.append(row['DriverEncoded'])
            else:
                feat.append(driver_encoding)
                
        # Absolute failsafe: slice the array to exactly match the model's expected size
        feat = feat[:expected_features]
                
        features.append(feat)
    
    return np.array(features, dtype=np.float32) 

def predict_future_laps(model, current_sequence, n_future=20):
    """Autoregressively predict the next n_future lap times."""
    model.eval()
    predictions = []
    sequence = current_sequence.copy()
    
    # Ensure hardware acceleration compatibility
    device = next(model.parameters()).device
    
    with torch.no_grad():
        for _ in range(n_future):
            seq_len = 8
            x = torch.tensor(current_seq[-seq_len:]).unsqueeze(0).to(device)  # type: ignore
            pred = model(x).item()
            predictions.append(pred)
            
            # Increment tyre life by 1 normalized unit (safely keeps the 9 dimensions)
            next_tyre_life = sequence[-1][0] + (1.0 / (TYRE_LIFE_MAX - TYRE_LIFE_MIN))
            next_features = sequence[-1].copy()
            next_features[0] = next_tyre_life
            sequence = np.vstack([sequence, next_features])
    
    return predictions


def detect_cliff_with_confidence(base_model, stint_df, n_future=20, n_samples=10, track_name=None):
    """
    Routes to the best model, then runs Monte Carlo dropout to find the cliff.
    """
    # 1. SMART ROUTING
    compound = stint_df['Compound'].iloc[-1]
    if track_name is None and 'Event' in stint_df.columns:
        track_name = stint_df['Event'].iloc[0]
    elif track_name is None:
        track_name = "Generic"
        
    try:
        model, _ = get_best_model(track_name, compound)
    except Exception as e:
        print(f"Router fallback in cliff detector: {e}")
        model = base_model
        
    # 2. SEQUENCE PREP (Now using dynamic sizing)
    model.train()  # Keep MC dropout active
    sequence = prepare_sequence(model, stint_df)
    
    current_lap = int(stint_df['TyreLife'].max())
    current_avg = stint_df['LapTime'].mean()
    
    cliff_laps = []
    all_predictions = []
    
    with torch.no_grad():
        for sample in range(n_samples):
            future_preds = predict_future_laps(model, sequence, n_future)
            future_seconds = [denormalize(p, LAP_TIME_MIN, LAP_TIME_MAX) for p in future_preds]
            all_predictions.append(future_seconds)
            
            for i, lap_time in enumerate(future_seconds):
                if lap_time > current_avg + CLIFF_THRESHOLD:
                    cliff_laps.append(current_lap + i + 1)
                    break
    
    model.eval() 
    
    if not cliff_laps:
        return None, None, None, None
    
    cliff_mean = np.mean(cliff_laps)
    cliff_low = int(np.percentile(cliff_laps, 25))
    cliff_high = int(np.percentile(cliff_laps, 75))
    
    return round(cliff_mean), cliff_low, cliff_high, all_predictions


def calculate_pit_delta(current_lap, pit_lap, future_seconds, avg_lap_time, pit_loss=22.0):
    laps_remaining = len(future_seconds) - (pit_lap - current_lap)
    if laps_remaining <= 0:
        return None
    
    degradation_cost = sum(
        future_seconds[i] - avg_lap_time 
        for i in range(pit_lap - current_lap, len(future_seconds))
    )
    
    net_delta = pit_loss - degradation_cost
    return round(net_delta, 3)


def find_optimal_pit_window(base_model, stint_df, race_laps_remaining=30, pit_loss=22.0, track_name=None):
    """Evaluate possible pit laps using the smartest available model."""
    compound = stint_df['Compound'].iloc[-1]
    if track_name is None and 'Event' in stint_df.columns:
        track_name = stint_df['Event'].iloc[0]
    elif track_name is None:
        track_name = "Generic"
        
    try:
        model, _ = get_best_model(track_name, compound)
    except Exception:
        model = base_model

    sequence = prepare_sequence(model, stint_df)
    current_lap = int(stint_df['TyreLife'].max())
    current_avg = stint_df['LapTime'].mean()
    
    future_preds = predict_future_laps(model, sequence, race_laps_remaining)
    future_seconds = [denormalize(p, LAP_TIME_MIN, LAP_TIME_MAX) for p in future_preds]
    
    results = []
    for pit_lap in range(current_lap + 1, current_lap + race_laps_remaining):
        delta = calculate_pit_delta(current_lap, pit_lap, future_seconds, current_avg, pit_loss)
        if delta is not None:
            results.append({
                'pit_lap': pit_lap,
                'net_delta': delta,
                'recommendation': 'PIT' if delta < 0 else 'STAY'
            })
    
    results_df = pd.DataFrame(results)
    optimal = results_df.loc[results_df['net_delta'].idxmin()]
    
    return optimal, results_df 