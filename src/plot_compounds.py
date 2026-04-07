import pandas as pd
import matplotlib.pyplot as plt
import glob

# Load all races
all_data = []
for file in glob.glob('data/*.csv'):
    df = pd.read_csv(file)
    all_data.append(df)

df = pd.concat(all_data, ignore_index=True)

# For each compound, plot average lap time vs tyre life
plt.figure(figsize=(10, 5))

colors = {'SOFT': 'red', 'MEDIUM': 'yellow', 'HARD': 'black'}

for compound in ['SOFT', 'MEDIUM', 'HARD']:
    subset = df[df['Compound'] == compound]
    avg = subset.groupby('TyreLife')['LapTime'].mean()
    plt.plot(avg.index, avg.values, 
             marker='o', markersize=3,
             color=colors[compound], label=compound)

plt.xlabel('Tyre Life (laps)')
plt.ylabel('Average Lap Time (seconds)')
plt.title('Average Degradation by Compound — All Races')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('results/compound_comparison.png')
print("Plot saved to results/compound_comparison.png")