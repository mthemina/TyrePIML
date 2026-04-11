import fastf1
import pandas as pd
import os
import json
import time

fastf1.Cache.enable_cache('data')

# All seasons to download
SEASONS = list(range(2018, 2025))

# Compounds we care about
VALID_COMPOUNDS = ['SOFT', 'MEDIUM', 'HARD']

# Tracks to skip — street circuits with unusual degradation patterns
# we'll add them back later with special handling
SKIP_EVENTS = ['Monaco Grand Prix', 'Singapore Grand Prix']

def get_race_schedule(year):
    """Get all conventional races for a season."""
    schedule = fastf1.get_event_schedule(year)
    races = schedule[schedule['EventFormat'] == 'conventional'].copy()
    return races[['RoundNumber', 'EventName', 'Location']].reset_index(drop=True)

def load_race_session(year, round_number, event_name):
    """Load a single race session safely."""
    try:
        session = fastf1.get_session(year, round_number, 'R')
        session.load(telemetry=False, weather=True, messages=False)
        return session
    except Exception as e:
        print(f"  ERROR loading {year} R{round_number} {event_name}: {e}")
        return None

def extract_lap_data(session):
    """Extract lap and tyre data from a session."""
    laps = session.laps
    
    columns = [
        'Driver', 'LapNumber', 'LapTime',
        'Compound', 'TyreLife', 'Stint',
        'Sector1Time', 'Sector2Time', 'Sector3Time',
    ]
    
    # Check all columns exist
    missing = [c for c in columns if c not in laps.columns]
    if missing:
        print(f"  Missing columns: {missing}")
        return None
    
    df = laps[columns].copy()
    
    # Convert times to seconds
    for col in ['LapTime', 'Sector1Time', 'Sector2Time', 'Sector3Time']:
        df[col] = df[col].dt.total_seconds()
    
    return df

def extract_weather_data(session):
    """Extract average weather conditions for the session."""
    try:
        weather = session.weather_data
        if weather is None or len(weather) == 0:
            return {}
        return {
            'air_temp_avg': round(weather['AirTemp'].mean(), 1),
            'track_temp_avg': round(weather['TrackTemp'].mean(), 1),
            'humidity_avg': round(weather['Humidity'].mean(), 1),
            'rainfall': bool(weather['Rainfall'].any())
        }
    except Exception:
        return {}

def clean_lap_data(df):
    """Clean and filter lap data."""
    df = df.dropna()
    df = df[df['TyreLife'] > 1]
    df = df[df['Compound'].isin(VALID_COMPOUNDS)]
    
    # Remove unrealistically slow laps
    fastest = df['LapTime'].min()
    df = df[df['LapTime'] < fastest * 1.20]
    
    # Remove unrealistically fast laps (data errors)
    df = df[df['LapTime'] > fastest * 0.95]
    
    return df.reset_index(drop=True)

def get_output_path(year, event_name):
    """Get the CSV output path for a race."""
    safe_name = event_name.replace(' ', '_').replace('/', '_')
    return f"data/{year}_{safe_name}.csv"

def race_already_downloaded(year, event_name):
    """Check if we already have this race's data."""
    return os.path.exists(get_output_path(year, event_name))

def run_full_pipeline(seasons=SEASONS, max_races=None):
    """
    Download and process all races across all seasons.
    Skips races already downloaded.
    """
    os.makedirs('data', exist_ok=True)
    
    total_downloaded = 0
    total_skipped = 0
    total_failed = 0
    manifest = []
    
    for year in seasons:
        print(f"\n{'='*50}")
        print(f"Season {year}")
        print(f"{'='*50}")
        
        schedule = get_race_schedule(year)
        
        for _, race in schedule.iterrows():
            round_num = race['RoundNumber']
            event_name = race['EventName']
            
            # Skip certain events
            if any(skip in event_name for skip in SKIP_EVENTS):
                print(f"  Skipping {event_name} (excluded circuit)")
                continue
            
            # Skip if already downloaded
            if race_already_downloaded(year, event_name):
                print(f"  ✓ {year} {event_name} already exists")
                total_skipped += 1
                manifest.append({
                    'year': year,
                    'event': event_name,
                    'status': 'cached'
                })
                continue
            
            print(f"  Downloading {year} R{round_num} {event_name}...")
            
            # Load session
            session = load_race_session(year, round_num, event_name)
            if session is None:
                total_failed += 1
                continue
            
            # Extract data
            df = extract_lap_data(session)
            if df is None or len(df) == 0:
                print(f"  No lap data for {event_name}")
                total_failed += 1
                continue
            
            # Add weather
            weather = extract_weather_data(session)
            for key, val in weather.items():
                df[key] = val
            
            # Add metadata
            df['Year'] = year
            df['Event'] = event_name
            df['Round'] = round_num
            
            # Clean
            df = clean_lap_data(df)
            
            if len(df) < 50:
                print(f"  Too few laps after cleaning: {len(df)}")
                total_failed += 1
                continue
            
            # Save
            output_path = get_output_path(year, event_name)
            df.to_csv(output_path, index=False)
            print(f"  ✓ Saved {output_path} — {len(df)} laps")
            
            manifest.append({
                'year': year,
                'event': event_name,
                'laps': len(df),
                'status': 'downloaded',
                'weather': weather
            })
            
            total_downloaded += 1
            
            # Stop if max_races reached (for testing)
            if max_races and total_downloaded >= max_races:
                print(f"\nReached max_races={max_races}, stopping.")
                break
            
            # Small delay to be respectful to the API
            time.sleep(1)
        
        if max_races and total_downloaded >= max_races:
            break
    
    # Save manifest
    with open('data/manifest.json', 'w') as f:
        json.dump(manifest, f, indent=2)
    
    print(f"\n{'='*50}")
    print(f"Pipeline complete!")
    print(f"Downloaded: {total_downloaded}")
    print(f"Skipped (cached): {total_skipped}")
    print(f"Failed: {total_failed}")
    print(f"{'='*50}")

if __name__ == '__main__':
    run_full_pipeline() 