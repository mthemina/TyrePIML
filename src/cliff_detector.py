import torch
import numpy as np
import pandas as pd
from src.model import TyreLSTM
from src.dataset import COMPOUND_MAP, normalize, denormalize, LAP_TIME_MIN, LAP_TIME_MAX, TYRE_LIFE_MIN, TYRE_LIFE_MAX, SECTOR_MIN, SECTOR_MAX

# A lap is considered a cliff if it's this many seconds slower than the stint average
CLIFF_THRESHOLD = 1.5

def prepare_sequence(stint_df, sequence_length=5):
    """Convert a stint dataframe into model input sequences."""
    features = []
    for _, row in stint_df.iterrows():
        features.append([
            normalize(row['TyreLife'], TYRE_LIFE_MIN, TYRE_LIFE_MAX),
            COMPOUND_MAP[row['Compound']] / 2.0,
            normalize(row['Sector1Time'], SECTOR_MIN, SECTOR_MAX),
            normalize(row['Sector2Time'], SECTOR_MIN, SECTOR_MAX),
            normalize(row['Sector3Time'], SECTOR_MIN, SECTOR_MAX),
        ])
    return np.array(features, dtype=np.float32)


def predict_future_laps(model, current_sequence, n_future=20):
    """
    Autoregressively predict the next n_future lap times.
    Each prediction is fed back as input for the next prediction.
    """
    model.eval()
    predictions = []
    sequence = current_sequence.copy()
    
    with torch.no_grad():
        for _ in range(n_future):
            x = torch.tensor(sequence[-5:]).unsqueeze(0)
            pred = model(x).item()
            predictions.append(pred)
            
            # Build next lap's features using predicted lap time
            # Increment tyre life by 1 normalized unit
            next_tyre_life = sequence[-1][0] + (1.0 / (TYRE_LIFE_MAX - TYRE_LIFE_MIN))
            next_features = sequence[-1].copy()
            next_features[0] = next_tyre_life
            sequence = np.vstack([sequence, next_features])
    
    return predictions


def detect_cliff(model, stint_df, n_future=20):
    """
    Given a driver's current stint, predict when the performance cliff will occur.
    Returns the predicted cliff lap number and confidence range.
    """
    sequence = prepare_sequence(stint_df)
    
    if len(sequence) < 5:
        return None, None, None
    
    # Predict future laps
    future_preds = predict_future_laps(model, sequence, n_future)
    
    # Convert back to seconds
    future_seconds = [denormalize(p, LAP_TIME_MIN, LAP_TIME_MAX) for p in future_preds]
    
    # Baseline — average of current stint
    current_avg = stint_df['LapTime'].mean()
    
    # Find first lap that exceeds cliff threshold
    cliff_lap = None
    current_lap = int(stint_df['TyreLife'].max())
    
    for i, lap_time in enumerate(future_seconds):
        if lap_time > current_avg + CLIFF_THRESHOLD:
            cliff_lap = current_lap + i + 1
            break
    
    return cliff_lap, future_seconds, current_avg

def detect_cliff_with_confidence(model, stint_df, n_future=20, n_samples=10):
    """
    Run cliff detection multiple times with dropout enabled (Monte Carlo dropout)
    to get a confidence range on the cliff prediction.
    
    Monte Carlo dropout = run the model multiple times with random neurons
    dropped each time, giving slightly different predictions each run.
    The spread of those predictions is our confidence interval.
    """
    # Enable dropout during inference for uncertainty estimation
    model.train()  # train mode keeps dropout active
    
    sequence = prepare_sequence(stint_df)
    current_lap = int(stint_df['TyreLife'].max())
    current_avg = stint_df['LapTime'].mean()
    
    cliff_laps = []
    all_predictions = []
    
    with torch.no_grad():
        for sample in range(n_samples):
            future_preds = predict_future_laps(model, sequence, n_future)
            future_seconds = [denormalize(p, LAP_TIME_MIN, LAP_TIME_MAX) 
                            for p in future_preds]
            all_predictions.append(future_seconds)
            
            # Find cliff lap for this sample
            for i, lap_time in enumerate(future_seconds):
                if lap_time > current_avg + CLIFF_THRESHOLD:
                    cliff_laps.append(current_lap + i + 1)
                    break
    
    model.eval()  # back to eval mode
    
    if not cliff_laps:
        return None, None, None, None
    
    cliff_mean = np.mean(cliff_laps)
    cliff_std = np.std(cliff_laps)
    cliff_low = int(np.percentile(cliff_laps, 25))
    cliff_high = int(np.percentile(cliff_laps, 75))
    
    return round(cliff_mean), cliff_low, cliff_high, all_predictions

def calculate_pit_delta(current_lap, pit_lap, future_seconds, avg_lap_time, 
                        pit_loss=22.0):
    """
    Calculate the net time delta of pitting on a given lap vs staying out.
    
    pit_loss: time lost in the pit lane (typically 20-22 seconds at most tracks)
    
    Returns positive number = pitting costs time (bad)
    Returns negative number = pitting saves time (good)
    """
    laps_remaining = len(future_seconds) - (pit_lap - current_lap)
    
    if laps_remaining <= 0:
        return None
    
    # Time lost staying out on degrading tyres
    degradation_cost = sum(
        future_seconds[i] - avg_lap_time 
        for i in range(pit_lap - current_lap, len(future_seconds))
    )
    
    # Net delta = pit stop loss - degradation cost avoided
    net_delta = pit_loss - degradation_cost
    
    return round(net_delta, 3)


def find_optimal_pit_window(model, stint_df, race_laps_remaining=30, pit_loss=22.0):
    """
    Evaluate every possible pit lap and return the optimal one.
    """
    sequence = prepare_sequence(stint_df)
    current_lap = int(stint_df['TyreLife'].max())
    current_avg = stint_df['LapTime'].mean()
    
    future_preds = predict_future_laps(model, sequence, race_laps_remaining)
    future_seconds = [denormalize(p, LAP_TIME_MIN, LAP_TIME_MAX) 
                     for p in future_preds]
    
    results = []
    
    for pit_lap in range(current_lap + 1, current_lap + race_laps_remaining):
        delta = calculate_pit_delta(
            current_lap, pit_lap, future_seconds, current_avg, pit_loss
        )
        if delta is not None:
            results.append({
                'pit_lap': pit_lap,
                'net_delta': delta,
                'recommendation': 'PIT' if delta < 0 else 'STAY'
            })
    
    results_df = pd.DataFrame(results)
    optimal = results_df.loc[results_df['net_delta'].idxmin()]
    
    return optimal, results_df


if __name__ == '__main__':
    model = TyreLSTM()
    model.load_state_dict(torch.load('models/tyre_lstm_piml_v1.pt'))
    
    df = pd.read_csv('data/2023_Monza.csv')
    ver_stint1 = df[(df['Driver'] == 'VER') & (df['Stint'] == 1)].reset_index(drop=True)
    
    # Cliff detection with confidence
    cliff_mean, cliff_low, cliff_high, all_preds = detect_cliff_with_confidence(
        model, ver_stint1
    )
    print(f"Predicted cliff lap: {cliff_mean}")
    print(f"Confidence range: lap {cliff_low} — lap {cliff_high}")
    print(f"Actual cliff lap: ~20 (from race data)")
    
    # Pit window optimization
    print("\n--- Pit Window Analysis ---")
    optimal, results_df = find_optimal_pit_window(
        model, ver_stint1, race_laps_remaining=25, pit_loss=22.0
    )
    
    print(f"\nOptimal pit lap: {int(optimal['pit_lap'])}")
    print(f"Expected net delta: {optimal['net_delta']:.3f}s")
    print(f"\nFull pit window:")
    print(results_df.to_string(index=False))