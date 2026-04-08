import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

COMPOUND_COLORS = {
    'SOFT': '#FF3333',
    'MEDIUM': '#FFD700',
    'HARD': '#333333',
    'INTER': '#39B54A',
    'WET': '#0067FF'
}

def plot_strategy_map(race_file, race_title):
    """
    Plot the classic F1 strategy map — every driver's stints
    as coloured horizontal bars.
    """
    df = pd.read_csv(race_file)
    
    # Get all drivers sorted by finishing position (best avg lap time)
    driver_avg = df.groupby('Driver')['LapTime'].mean().sort_values()
    drivers = driver_avg.index.tolist()
    
    fig, ax = plt.subplots(figsize=(14, 8))
    
    for i, driver in enumerate(drivers):
        driver_df = df[df['Driver'] == driver]
        
        for stint in driver_df['Stint'].unique():
            stint_df = driver_df[driver_df['Stint'] == stint]
            
            if len(stint_df) == 0:
                continue
            
            compound = stint_df['Compound'].iloc[0]
            start_lap = stint_df['LapNumber'].min()
            end_lap = stint_df['LapNumber'].max()
            color = COMPOUND_COLORS.get(compound, 'gray')
            
            # Draw stint bar
            ax.barh(i, end_lap - start_lap, left=start_lap,
                   height=0.6, color=color, alpha=0.85,
                   edgecolor='white', linewidth=0.5)
            
            # Add compound label in the middle of the bar
            mid_lap = (start_lap + end_lap) / 2
            ax.text(mid_lap, i, compound[0],  # S, M, or H
                   ha='center', va='center',
                   fontsize=7, fontweight='bold',
                   color='white' if compound != 'MEDIUM' else 'black')
    
    # Styling
    ax.set_yticks(range(len(drivers)))
    ax.set_yticklabels(drivers, fontsize=9)
    ax.set_xlabel('Lap Number', fontsize=11)
    ax.set_title(race_title, fontsize=13, fontweight='bold')
    ax.grid(True, axis='x', alpha=0.3)
    ax.invert_yaxis()
    
    # Legend
    patches = [mpatches.Patch(color=v, label=k) 
               for k, v in COMPOUND_COLORS.items() 
               if k in df['Compound'].values]
    ax.legend(handles=patches, loc='lower right', fontsize=9)
    
    plt.tight_layout()
    filename = f"results/strategy_map_{race_title.replace(' ', '_')}.png"
    plt.savefig(filename, dpi=150)
    print(f"Strategy map saved to {filename}")


if __name__ == '__main__':
    plot_strategy_map('data/2023_Monza.csv', '2023 Italian Grand Prix')
    plot_strategy_map('data/2023_Silverstone.csv', '2023 British Grand Prix')
    plot_strategy_map('data/2023_Spa.csv', '2023 Belgian Grand Prix') 