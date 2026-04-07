import torch
from torch.utils.data import DataLoader, random_split
from src.dataset import TyreDataset, denormalize, LAP_TIME_MIN, LAP_TIME_MAX
from src.model import TyreLSTM
import numpy as np

def evaluate():
    # Load dataset
    dataset = TyreDataset()
    
    # Same split as training — same random seed so val set is identical
    torch.manual_seed(42)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    _, val_set = random_split(dataset, [train_size, val_size])
    val_loader = DataLoader(val_set, batch_size=32, shuffle=False)
    
    # Load saved model
    model = TyreLSTM()
    model.load_state_dict(torch.load('models/tyre_lstm_v1.pt'))
    model.eval()
    
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        for x, y in val_loader:
            predictions = model(x)
            all_predictions.extend(predictions.numpy())
            all_targets.extend(y.numpy())
    
    # Convert back to seconds
    pred_seconds = [denormalize(p, LAP_TIME_MIN, LAP_TIME_MAX) for p in all_predictions]
    true_seconds = [denormalize(t, LAP_TIME_MIN, LAP_TIME_MAX) for t in all_targets]
    
    # Calculate MAE — average error in seconds
    mae = np.mean(np.abs(np.array(pred_seconds) - np.array(true_seconds)))
    
    print(f"Validation MAE: {mae:.3f} seconds")
    print(f"That means on average the model is off by {mae:.3f}s per lap")
    
    return pred_seconds, true_seconds

def plot_predictions(pred_seconds, true_seconds):
    import matplotlib.pyplot as plt
    import numpy as np

    # Just plot the first 100 validation laps for clarity
    n = 100
    x = np.arange(n)

    plt.figure(figsize=(12, 5))
    plt.plot(x, true_seconds[:n], label='Actual', color='black', linewidth=1.5)
    plt.plot(x, pred_seconds[:n], label='Predicted', color='red', 
             linewidth=1.5, linestyle='--')

    plt.xlabel('Lap sequence (validation set)')
    plt.ylabel('Lap Time (seconds)')
    plt.title('Baseline LSTM — Predicted vs Actual Lap Times')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('results/baseline_predictions.png')
    print("Plot saved to results/baseline_predictions.png") 

def save_results(mae):
    import json
    import os
    
    results = {
        'model': 'Baseline LSTM',
        'mae_seconds': round(float(mae), 4),
        'val_sequences': 666,
        'epochs': 20,
        'notes': 'Baseline before physics constraints'
    }
    
    with open('results/baseline_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print("Results saved to results/baseline_results.json") 

if __name__ == '__main__':
    pred_seconds, true_seconds = evaluate()
    plot_predictions(pred_seconds, true_seconds)
    mae = np.mean(np.abs(np.array(pred_seconds) - np.array(true_seconds)))
    save_results(mae) 