import torch
from torch.utils.data import DataLoader, random_split
from src.dataset import TyreDataset
from src.model import TyreLSTM
import numpy as np

def count_violations(predictions, tyre_lives):
    """
    Count what percentage of consecutive prediction pairs
    violate the monotonic degradation rule.
    
    A violation is when tyre life increases but predicted lap time decreases.
    """
    violations = 0
    total = 0
    
    for i in range(1, len(predictions)):
        tyre_increased = tyre_lives[i] > tyre_lives[i-1]
        lap_time_decreased = predictions[i] < predictions[i-1]
        
        if tyre_increased:
            total += 1
            if lap_time_decreased:
                violations += 1
    
    if total == 0:
        return 0.0
    
    return (violations / total) * 100  # return as percentage


def evaluate_violations(model_path):
    """Load a model and measure its physics violation rate."""
    
    dataset = TyreDataset()
    
    torch.manual_seed(42)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    _, val_set = random_split(dataset, [train_size, val_size])
    val_loader = DataLoader(val_set, batch_size=32, shuffle=False)
    
    model = TyreLSTM()
    model.load_state_dict(torch.load(model_path))
    model.eval()
    
    all_predictions = []
    all_tyre_lives = []
    
    with torch.no_grad():
        for x, y in val_loader:
            predictions = model(x)
            tyre_lives = x[:, -1, 0]  # tyre life of last lap in sequence
            all_predictions.extend(predictions.numpy())
            all_tyre_lives.extend(tyre_lives.numpy())
    
    violation_rate = count_violations(all_predictions, all_tyre_lives)
    return violation_rate


if __name__ == '__main__':
    baseline_violations = evaluate_violations('models/tyre_lstm_v1.pt')
    piml_violations = evaluate_violations('models/tyre_lstm_piml_v1.pt')
    
    print(f"Baseline LSTM violation rate:  {baseline_violations:.1f}%")
    print(f"PIML LSTM violation rate:      {piml_violations:.1f}%")
    print(f"Reduction:                     {baseline_violations - piml_violations:.1f}%") 