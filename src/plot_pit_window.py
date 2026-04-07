import torch
import pandas as pd
import matplotlib.pyplot as plt
from src.model import TyreLSTM
from src.cliff_detector import (detect_cliff_with_confidence, 
                                 find_optimal_pit_window,
                                 predict_future_laps,
                                 prepare_sequence)
from src.dataset import denormalize, LAP_TIME_MIN, LAP_TIME_MAX

def plot_pit_window(driver, year, race, stint=1, race_laps_remaining=25):
    model = TyreLSTM()
    model.load_state_dict(torch.load('models/tyre_lstm_piml_v1.pt'))
    
    df = pd.read_csv(f'data/{year}_{race}.csv')
    stint_df = df[(df['Driver'] == driver) & 
                  (df['Stint'] == stint)].reset_index(drop=True)
    
    # Get predictions
    sequence = prepare_sequence(stint_df)
    current_lap = int(stint_df['TyreLife'].max())
    future_preds = predict_future_laps(model, sequence, race_laps_remaining)
    future_seconds = [denormalize(p, LAP_TIME_MIN, LAP_TIME_MAX) 
                     for p in future_preds]
    
    # Get optimal pit
    optimal, results_df = find_optimal_pit_window(
        model, stint_df, race_laps_remaining, pit_loss=22.0
    )
    optimal_lap = int(optimal['pit_lap'])
    
    # Get cliff
    cliff_mean, cliff_low, cliff_high, _ = detect_cliff_with_confidence(
        model, stint_df
    )
    
    # Plot
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
    
    # Top plot — lap time predictions
    actual_laps = list(range(2, current_lap + 1))
    future_laps = list(range(current_lap + 1, 
                             current_lap + race_laps_remaining + 1))
    
    ax1.plot(actual_laps, stint_df['LapTime'], 
             color='black', linewidth=2, label='Actual')
    ax1.plot(future_laps, future_seconds, 
             color='blue', linewidth=2, linestyle='--', label='Predicted')
    ax1.axvline(x=optimal_lap, color='green', linewidth=2, 
                linestyle=':', label=f'Optimal pit: lap {optimal_lap}')
    if cliff_mean:
        ax1.axvline(x=cliff_mean, color='red', linewidth=2, 
                    linestyle=':', label=f'Predicted cliff: lap {cliff_mean}')
    ax1.set_xlabel('Tyre Life (laps)')
    ax1.set_ylabel('Lap Time (seconds)')
    ax1.set_title(f'{driver} — {year} {race} Stint {stint} — Tyre Strategy Analysis')
    ax1.legend()
    ax1.grid(True)
    
    # Bottom plot — pit window delta
    pit_laps = results_df['pit_lap'].tolist()
    deltas = results_df['net_delta'].tolist()
    colors = ['green' if d < 0 else 'red' for d in deltas]
    
    ax2.bar(pit_laps, deltas, color=colors, alpha=0.7)
    ax2.axhline(y=0, color='black', linewidth=1)
    ax2.axvline(x=optimal_lap, color='green', linewidth=2, 
                linestyle=':', label=f'Optimal pit: lap {optimal_lap}')
    ax2.set_xlabel('Pit Lap')
    ax2.set_ylabel('Net Time Delta (seconds)')
    ax2.set_title('Pit Window — Green = Gain, Red = Loss')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'results/{driver}_{year}_{race}_strategy.png')
    print(f"Strategy plot saved!")
    print(f"Optimal pit lap: {optimal_lap}")
    print(f"Predicted cliff: lap {cliff_mean}")

if __name__ == '__main__':
    plot_pit_window('VER', 2023, 'Monza', stint=1, race_laps_remaining=25)