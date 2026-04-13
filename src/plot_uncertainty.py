import torch
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from src.model import TyreLSTM
from src.uncertainty import predict_with_uncertainty, predict_cliff_with_uncertainty
from src.cliff_detector import prepare_sequence


def plot_uncertainty_bands(driver, year, race, stint=1, n_future=20):
    """Plot degradation forecast with uncertainty bands."""
    
    model = TyreLSTM(input_size=8, hidden_size=128, num_layers=2)
    model.load_state_dict(torch.load('models/tyre_lstm_piml_v2.pt'))
    
    df = pd.read_csv(f'data/{year}_{race}.csv')
    stint_df = df[
        (df['Driver'] == driver) & 
        (df['Stint'] == stint)
    ].reset_index(drop=True)
    
    # Get uncertainty predictions
    result = predict_with_uncertainty(model, stint_df, 
                                      n_future=n_future, n_samples=50)
    cliff = predict_cliff_with_uncertainty(model, stint_df, n_samples=50)
    
    if result is None:
        print("Not enough data")
        return
    
    fig, ax = plt.subplots(figsize=(12, 6), facecolor='#1E1E1E')
    ax.set_facecolor('#1E1E1E')
    ax.tick_params(colors='#F0EDE4')
    ax.xaxis.label.set_color('#F0EDE4')
    ax.yaxis.label.set_color('#F0EDE4')
    ax.title.set_color('#C9A84C')
    for spine in ax.spines.values():
        spine.set_edgecolor('#3a3a3a')
    
    # Actual laps
    actual_laps = stint_df['TyreLife'].tolist()
    ax.plot(actual_laps, stint_df['LapTime'],
            color='#F0EDE4', linewidth=2, label='Actual', zorder=5)
    
    future_laps = result['laps']
    mean = result['mean']
    p10 = result['p10']
    p90 = result['p90']
    p25 = result['p25']
    p75 = result['p75']
    
    # Mean prediction
    ax.plot(future_laps, mean,
            color='#6B7C3F', linewidth=2, 
            linestyle='--', label='Predicted mean', zorder=5)
    
    # 80% confidence band (p10-p90)
    ax.fill_between(future_laps, p10, p90,
                   alpha=0.2, color='#6B7C3F', label='80% confidence')
    
    # 50% confidence band (p25-p75)
    ax.fill_between(future_laps, p25, p75,
                   alpha=0.4, color='#6B7C3F', label='50% confidence')
    
    # Cliff prediction
    ax.axvline(x=cliff['mean'], color='#8B2020', linewidth=2,
              linestyle=':', label=f"Cliff: lap {cliff['mean']}", zorder=6)
    ax.axvspan(cliff['p25'], cliff['p75'],
              alpha=0.15, color='#8B2020', label='Cliff 50% range')
    
    ax.set_xlabel('Tyre Life (laps)')
    ax.set_ylabel('Lap Time (seconds)')
    ax.set_title(f'{driver} — {year} {race} — Degradation Forecast with Uncertainty')
    ax.legend(fontsize=8, facecolor='#2a2a2a', labelcolor='#F0EDE4')
    ax.grid(True, alpha=0.15)
    
    plt.tight_layout()
    filename = f'results/{driver}_{year}_{race}_uncertainty.png'
    plt.savefig(filename, dpi=150, facecolor='#1E1E1E')
    print(f"Saved {filename}")
    print(f"\nCliff prediction: lap {cliff['mean']} ± {cliff['std']} laps")
    print(f"50% confidence: lap {cliff['p25']} — lap {cliff['p75']}")
    print(f"80% confidence: lap {cliff['p10']} — lap {cliff['p90']}")


if __name__ == '__main__':
    plot_uncertainty_bands('VER', 2023, 'Monza', stint=1, n_future=20) 