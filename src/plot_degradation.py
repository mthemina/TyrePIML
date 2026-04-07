import pandas as pd
import matplotlib.pyplot as plt

# Load Monza 2023
df = pd.read_csv('data/2023_Monza.csv')

# Pick one driver — Verstappen
ver = df[df['Driver'] == 'VER']

# Plot lap time vs tyre life
plt.figure(figsize=(10, 5))

for stint in ver['Stint'].unique():
    stint_data = ver[ver['Stint'] == stint]
    compound = stint_data['Compound'].iloc[0]
    plt.plot(stint_data['TyreLife'], stint_data['LapTime'], 
             marker='o', label=f"Stint {int(stint)} — {compound}")

plt.xlabel('Tyre Life (laps)')
plt.ylabel('Lap Time (seconds)')
plt.title('Verstappen — Tyre Degradation — Monza 2023')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('results/ver_monza_2023.png')
print("Plot saved to results/ver_monza_2023.png")