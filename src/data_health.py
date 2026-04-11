import pandas as pd
import glob
import os
import json

def generate_health_report():
    """Generate a comprehensive report on the downloaded dataset."""
    
    files = sorted(glob.glob('data/*.csv'))
    
    if not files:
        print("No CSV files found in data/")
        return
    
    total_laps = 0
    total_races = 0
    seasons = {}
    compounds = {'SOFT': 0, 'MEDIUM': 0, 'HARD': 0}
    tracks = {}
    weather_coverage = 0
    
    print(f"\n{'='*60}")
    print(f"TyrePIML Dataset Health Report")
    print(f"{'='*60}\n")
    
    for file in files:
        df = pd.read_csv(file)
        name = os.path.basename(file).replace('.csv', '')
        
        # Extract year
        year = int(name.split('_')[0])
        event = '_'.join(name.split('_')[1:])
        
        laps = len(df)
        total_laps += laps
        total_races += 1
        
        # Season breakdown
        if year not in seasons:
            seasons[year] = {'races': 0, 'laps': 0}
        seasons[year]['races'] += 1
        seasons[year]['laps'] += laps
        
        # Compound breakdown
        for compound in compounds:
            if compound in df['Compound'].values:
                compounds[compound] += df[df['Compound'] == compound].shape[0]
        
        # Track breakdown
        if event not in tracks:
            tracks[event] = {'races': 0, 'laps': 0}
        tracks[event]['races'] += 1
        tracks[event]['laps'] += laps
        
        # Weather coverage
        if 'track_temp_avg' in df.columns:
            weather_coverage += 1
    
    # Print summary
    print(f"Total races:     {total_races}")
    print(f"Total laps:      {total_laps:,}")
    print(f"Weather data:    {weather_coverage}/{total_races} races")
    print(f"\n--- By Season ---")
    for year in sorted(seasons.keys()):
        s = seasons[year]
        print(f"  {year}: {s['races']:>3} races, {s['laps']:>6,} laps")
    
    print(f"\n--- By Compound ---")
    for compound, count in compounds.items():
        pct = count / total_laps * 100 if total_laps > 0 else 0
        print(f"  {compound:<8}: {count:>6,} laps ({pct:.1f}%)")
    
    print(f"\n--- Top 10 Tracks by Lap Count ---")
    top_tracks = sorted(tracks.items(), 
                       key=lambda x: x[1]['laps'], reverse=True)[:10]
    for track, data in top_tracks:
        print(f"  {track:<35}: {data['laps']:>5,} laps ({data['races']} races)")
    
    # Save report
    report = {
        'total_races': total_races,
        'total_laps': total_laps,
        'weather_coverage': weather_coverage,
        'seasons': seasons,
        'compounds': compounds,
    }
    
    with open('results/data_health.json', 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"\nReport saved to results/data_health.json")
    print(f"{'='*60}\n")

if __name__ == '__main__':
    generate_health_report() 