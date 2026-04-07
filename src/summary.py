import pandas as pd
import os
import glob

# Find all race CSV files
files = glob.glob('data/*.csv')

print(f"{'Race':<25} {'Laps':>6} {'Drivers':>8} {'Compounds'}")
print("-" * 60)

for file in sorted(files):
    df = pd.read_csv(file)
    race = os.path.basename(file).replace('.csv', '')
    laps = len(df)
    drivers = df['Driver'].nunique()
    compounds = ', '.join(sorted(df['Compound'].unique()))
    print(f"{race:<25} {laps:>6} {drivers:>8}     {compounds}")

print(f"\nTotal laps across all races: {sum(len(pd.read_csv(f)) for f in files)}")