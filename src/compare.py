import torch
from torch.utils.data import DataLoader, random_split
from src.dataset import TyreDataset, denormalize, LAP_TIME_MIN, LAP_TIME_MAX
from src.model import TyreLSTM
from src.violations import count_violations
import numpy as np

def evaluate_model(model_path, dataset, val_set):
    """Run full evaluation on a model and return all metrics."""
    
    val_loader = DataLoader(val_set, batch_size=32, shuffle=False)
    
    model = TyreLSTM()
    model.load_state_dict(torch.load(model_path))
    model.eval()
    
    all_predictions = []
    all_targets = []
    all_tyre_lives = []
    
    with torch.no_grad():
        for x, y in val_loader:
            predictions = model(x)
            tyre_lives = x[:, -1, 0]
            all_predictions.extend(predictions.numpy())
            all_targets.extend(y.numpy())
            all_tyre_lives.extend(tyre_lives.numpy())
    
    # Convert to seconds
    pred_seconds = [denormalize(p, LAP_TIME_MIN, LAP_TIME_MAX) for p in all_predictions]
    true_seconds = [denormalize(t, LAP_TIME_MIN, LAP_TIME_MAX) for t in all_targets]
    
    # Calculate metrics
    mae = np.mean(np.abs(np.array(pred_seconds) - np.array(true_seconds)))
    rmse = np.sqrt(np.mean((np.array(pred_seconds) - np.array(true_seconds))**2))
    violation_rate = count_violations(all_predictions, all_tyre_lives)
    
    return {
        'mae': round(float(mae), 4),
        'rmse': round(float(rmse), 4),
        'violation_rate': round(float(violation_rate), 2),
        'predictions': pred_seconds,
        'targets': true_seconds
    }

def save_results_table(baseline, piml):
    import csv
    
    rows = [
        ['Metric', 'Baseline LSTM', 'PIML LSTM'],
        ['MAE (seconds)', baseline['mae'], piml['mae']],
        ['RMSE (seconds)', baseline['rmse'], piml['rmse']],
        ['Violation Rate (%)', baseline['violation_rate'], piml['violation_rate']],
    ]
    
    with open('results/comparison_table.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerows(rows)
    
    print("\nResults table saved to results/comparison_table.csv")

def plot_comparison(baseline, piml):
    import matplotlib.pyplot as plt
    import numpy as np
    
    n = 100
    x = np.arange(n)
    
    plt.figure(figsize=(12, 6))
    
    plt.plot(x, baseline['targets'][:n], 
             label='Actual', color='black', linewidth=2)
    plt.plot(x, baseline['predictions'][:n], 
             label='Baseline LSTM', color='red', 
             linewidth=1.5, linestyle='--', alpha=0.8)
    plt.plot(x, piml['predictions'][:n], 
             label='PIML LSTM', color='blue', 
             linewidth=1.5, linestyle='--', alpha=0.8)
    
    plt.xlabel('Lap sequence (validation set)')
    plt.ylabel('Lap Time (seconds)')
    plt.title('Baseline vs PIML — Predicted vs Actual Lap Times')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('results/comparison_plot.png')
    print("Comparison plot saved to results/comparison_plot.png") 

if __name__ == '__main__':
    dataset = TyreDataset()
    torch.manual_seed(42)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    _, val_set = random_split(dataset, [train_size, val_size])
    
    print("Evaluating baseline model...")
    baseline = evaluate_model('models/tyre_lstm_v1.pt', dataset, val_set)
    
    print("Evaluating PIML model...")
    piml = evaluate_model('models/tyre_lstm_piml_v1.pt', dataset, val_set)
    
    print("\n--- Results ---")
    print(f"{'Metric':<25} {'Baseline':>10} {'PIML':>10}")
    print("-" * 45)
    print(f"{'MAE (seconds)':<25} {baseline['mae']:>10} {piml['mae']:>10}")
    print(f"{'RMSE (seconds)':<25} {baseline['rmse']:>10} {piml['rmse']:>10}")
    print(f"{'Violation Rate (%)':<25} {baseline['violation_rate']:>10} {piml['violation_rate']:>10}")
    
    save_results_table(baseline, piml) 
    plot_comparison(baseline, piml) 