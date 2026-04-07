import fastf1
import pandas as pd
import os

# Enable cache so we don't re-download data every time
fastf1.Cache.enable_cache('data')

# The races we'll use for training and testing
RACES = [
    (2023, 'Monza'),
    (2023, 'Silverstone'),
    (2023, 'Spa'),
    (2022, 'Monza'),
    (2022, 'Silverstone'),
]

def load_race(year, race_name):
    """Load a single race session and return cleaned lap data."""
    print(f"Loading {year} {race_name}...")
    session = fastf1.get_session(year, race_name, 'R')
    session.load(telemetry=False, weather=False, messages=False)
    return session

def extract_lap_data(session):
    """Extract relevant columns from a session."""
    laps = session.laps
    
    # The columns we care about
    columns = [
        'Driver',
        'LapNumber',
        'LapTime',
        'Compound',
        'TyreLife',
        'Stint',
        'Sector1Time',
        'Sector2Time',
        'Sector3Time',
    ]
    
    df = laps[columns].copy()
    
    # Convert lap and sector times from timedelta to seconds (easier for the model)
    df['LapTime'] = df['LapTime'].dt.total_seconds()
    df['Sector1Time'] = df['Sector1Time'].dt.total_seconds()
    df['Sector2Time'] = df['Sector2Time'].dt.total_seconds()
    df['Sector3Time'] = df['Sector3Time'].dt.total_seconds()
    
    return df

def clean_lap_data(df):
    """Remove laps that would corrupt model training."""
    
    # Remove laps with any missing values
    df = df.dropna()
    
    # Remove pit laps (tyre life resets to 1 on out lap)
    df = df[df['TyreLife'] > 1]
    
    # Remove unrealistically slow laps (safety car, red flag, pit in/out)
    # Any lap more than 20% slower than the fastest lap is suspicious
    fastest = df['LapTime'].min()
    df = df[df['LapTime'] < fastest * 1.20]
    
    # Remove unknown compounds (no physical meaning)
    df = df[df['Compound'].isin(['SOFT', 'MEDIUM', 'HARD'])]
    
    return df.reset_index(drop=True) 

def save_race_data(df, year, race_name):
    """Save cleaned race data as a CSV file."""
    
    # Create a clean filename e.g. 2023_Monza.csv
    filename = f"{year}_{race_name.replace(' ', '_')}.csv"
    filepath = os.path.join('data', filename)
    
    df.to_csv(filepath, index=False)
    print(f"Saved {filepath} — {len(df)} laps")


def run_pipeline():
    """Run the full pipeline for all races."""
    for year, race_name in RACES:
        session = load_race(year, race_name)
        df = extract_lap_data(session)
        df = clean_lap_data(df)
        save_race_data(df, year, race_name)
    
    print("\nAll races saved!")


# Run when script is called directly
if __name__ == '__main__':
    run_pipeline() 