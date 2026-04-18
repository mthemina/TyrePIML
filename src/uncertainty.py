import torch
import numpy as np
from src.model import TyreLSTM
from src.cliff_detector import prepare_sequence, predict_future_laps
from src.dataset import denormalize, LAP_TIME_MIN, LAP_TIME_MAX


def enable_dropout(model):
    """Enable dropout layers during inference for Monte Carlo sampling."""
    for module in model.modules():
        if isinstance(module, torch.nn.Dropout):
            module.train()


def mc_predict(model, sequence, n_future=20, n_samples=30):
    """
    Monte Carlo dropout prediction.
    Runs the model n_samples times with dropout active.
    """
    # Force dropout active
    model.train()
    
    all_predictions = []
    
    with torch.no_grad():
        for _ in range(n_samples):
            preds = []
            current_seq = sequence.copy()
            
            for _ in range(n_future):
                x = torch.tensor(current_seq[-5:]).unsqueeze(0)
                # Run model with dropout active
                pred = model(x)
                # Handle both scalar and tensor outputs
                if pred.dim() == 0:
                    pred_val = pred.item()
                else:
                    pred_val = pred[0].item()
                
                pred_seconds = denormalize(pred_val, LAP_TIME_MIN, LAP_TIME_MAX)
                preds.append(pred_seconds)
                
                # Build next features
                next_features = current_seq[-1].copy()
                next_features[0] = current_seq[-1][0] + (1.0 / 59.0)
                current_seq = np.vstack([current_seq, next_features])
            
            all_predictions.append(preds)
    
    model.eval()
    all_predictions = np.array(all_predictions)
    
    return {
        'mean': np.mean(all_predictions, axis=0),
        'std': np.std(all_predictions, axis=0),
        'p25': np.percentile(all_predictions, 25, axis=0),
        'p75': np.percentile(all_predictions, 75, axis=0),
        'p10': np.percentile(all_predictions, 10, axis=0),
        'p90': np.percentile(all_predictions, 90, axis=0),
        'all_samples': all_predictions
    } 


def predict_with_uncertainty(model, stint_df, n_future=20, n_samples=30):
    """
    Full uncertainty-aware prediction for a driver stint.
    Returns predictions with confidence bands.
    """
    sequence = prepare_sequence(model, stint_df) 
    
    if len(sequence) < 5:
        return None
    
    uncertainty = mc_predict(model, sequence, n_future, n_samples)
    current_lap = int(stint_df['TyreLife'].max())
    future_laps = list(range(current_lap + 1, 
                             current_lap + n_future + 1))
    
    return {
        'laps': future_laps,
        'mean': uncertainty['mean'].tolist(),
        'std': uncertainty['std'].tolist(),
        'p25': uncertainty['p25'].tolist(),
        'p75': uncertainty['p75'].tolist(),
        'p10': uncertainty['p10'].tolist(),
        'p90': uncertainty['p90'].tolist(),
    }


def predict_cliff_with_uncertainty(model, stint_df, 
                                    cliff_threshold=1.5,
                                    n_future=20, n_samples=30):
    """
    Predict performance cliff with full uncertainty quantification.
    Returns cliff lap distribution across all MC samples.
    """
    sequence = prepare_sequence(model, stint_df) 
    current_lap = int(stint_df['TyreLife'].max())
    current_avg = stint_df['LapTime'].mean()
    
    model.eval()
    enable_dropout(model)
    
    cliff_laps = []
    
    with torch.no_grad():
        for _ in range(n_samples):
            preds = predict_future_laps(model, sequence, n_future)
            preds_seconds = [denormalize(p, LAP_TIME_MIN, LAP_TIME_MAX) 
                           for p in preds]
            
            cliff_found = False
            for i, lap_time in enumerate(preds_seconds):
                if lap_time > current_avg + cliff_threshold:
                    cliff_laps.append(current_lap + i + 1)
                    cliff_found = True
                    break
            
            if not cliff_found:
                cliff_laps.append(current_lap + n_future)
    
    cliff_laps = np.array(cliff_laps)
    
    return {
        'mean': round(float(np.mean(cliff_laps))),
        'std': round(float(np.std(cliff_laps)), 2),
        'p10': int(np.percentile(cliff_laps, 10)),
        'p25': int(np.percentile(cliff_laps, 25)),
        'p75': int(np.percentile(cliff_laps, 75)),
        'p90': int(np.percentile(cliff_laps, 90)),
        'earliest': int(cliff_laps.min()),
        'latest': int(cliff_laps.max()),
    }


if __name__ == '__main__':
    import pandas as pd
    
    model = TyreLSTM(input_size=8, hidden_size=128, num_layers=2)
    model.load_state_dict(torch.load('models/tyre_lstm_piml_v2.pt'))
    
    df = pd.read_csv('data/2023_Monza.csv')
    ver = df[(df['Driver'] == 'VER') & 
             (df['Stint'] == 1)].reset_index(drop=True)
    
    print("=== Uncertainty Quantification — VER Monza 2023 ===\n")
    
    # Lap time predictions with uncertainty
    result = predict_with_uncertainty(model, ver, n_future=15, n_samples=30)
    
    print(f"{'Lap':>4} {'Mean':>8} {'Std':>6} {'P25':>8} {'P75':>8} {'Band':>8}")
    print("-" * 50)
    if result is None:
        print("Not enough telemetry data to calculate uncertainty for this stint.")
    else:
        for i in range(len(result['laps'])):
            band = result['p75'][i] - result['p25'][i]
            print(f"{result['laps'][i]:>4} "
                  f"{result['mean'][i]:>8.3f} "
                  f"{result['std'][i]:>6.3f} "
                  f"{result['p25'][i]:>8.3f} "
                  f"{result['p75'][i]:>8.3f} "
                  f"{band:>8.3f}") 
    
    print("\n=== Cliff Prediction with Uncertainty ===\n")
    cliff = predict_cliff_with_uncertainty(model, ver, n_samples=30)
    print(f"Cliff lap mean:     {cliff['mean']}")
    print(f"Cliff lap std:      {cliff['std']} laps")
    print(f"Earliest possible:  lap {cliff['earliest']}")
    print(f"Latest possible:    lap {cliff['latest']}")
    print(f"80% confidence:     lap {cliff['p10']} — lap {cliff['p90']}")
    print(f"50% confidence:     lap {cliff['p25']} — lap {cliff['p75']}") 