import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
import pandas as pd
import numpy as np
import glob
import os
from src.model import TyreLSTM
from src.piml_loss import ThermalPIMLLoss
from src.dataset import (normalize, denormalize, COMPOUND_MAP,
                         LAP_TIME_MIN, LAP_TIME_MAX, TYRE_LIFE_MIN,
                         TYRE_LIFE_MAX, SECTOR_MIN, SECTOR_MAX,
                         TEMP_MIN, TEMP_MAX, ABRASIVENESS_MIN, ABRASIVENESS_MAX)
from src.track_profiles import get_track_profile


class TrackDataset(Dataset):
    """Dataset for a single track across all available seasons."""
    
    def __init__(self, track_name, data_path='data/', sequence_length=5):
        self.sequence_length = sequence_length
        self.sequences = []
        self.targets = []
        self.track_name = track_name
        
        # Find all CSV files for this track
        all_files = glob.glob(f'{data_path}*.csv')
        track_files = [f for f in all_files 
                      if track_name.lower() in f.lower()]
        
        if not track_files:
            raise ValueError(f"No data found for track: {track_name}")
        
        profile = get_track_profile(track_name)
        abrasiveness = profile['abrasiveness']
        
        for file in sorted(track_files):
            df = pd.read_csv(file)
            self._process_race(df, abrasiveness)
        
        print(f"  {track_name}: {len(self.sequences)} sequences "
              f"from {len(track_files)} races")
    
    def _process_race(self, df, abrasiveness):
        has_weather = 'track_temp_avg' in df.columns
        
        for driver in df['Driver'].unique():
            driver_df = df[df['Driver'] == driver]
            
            for stint in driver_df['Stint'].unique():
                stint_df = driver_df[
                    driver_df['Stint'] == stint
                ].reset_index(drop=True)
                
                if len(stint_df) < self.sequence_length + 1:
                    continue
                
                track_temp = float(stint_df['track_temp_avg'].iloc[0]) \
                            if has_weather else 35.0
                air_temp = float(stint_df['air_temp_avg'].iloc[0]) \
                          if has_weather else 28.0
                
                features = []
                for _, row in stint_df.iterrows():
                    features.append([
                        normalize(row['TyreLife'], TYRE_LIFE_MIN, TYRE_LIFE_MAX),
                        COMPOUND_MAP.get(row['Compound'], 1) / 2.0,
                        normalize(row['Sector1Time'], SECTOR_MIN, SECTOR_MAX),
                        normalize(row['Sector2Time'], SECTOR_MIN, SECTOR_MAX),
                        normalize(row['Sector3Time'], SECTOR_MIN, SECTOR_MAX),
                        normalize(track_temp, TEMP_MIN, TEMP_MAX),
                        normalize(air_temp, TEMP_MIN, TEMP_MAX),
                        normalize(abrasiveness, ABRASIVENESS_MIN, ABRASIVENESS_MAX),
                    ])
                
                features = np.array(features, dtype=np.float32)
                
                for i in range(len(features) - self.sequence_length):
                    self.sequences.append(features[i:i + self.sequence_length])
                    self.targets.append(
                        normalize(
                            stint_df.iloc[i + self.sequence_length]['LapTime'],
                            LAP_TIME_MIN, LAP_TIME_MAX
                        )
                    )
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        x = torch.tensor(self.sequences[idx])
        y = torch.tensor(self.targets[idx], dtype=torch.float32)
        return x, y


def train_track_model(track_name, epochs=25, min_sequences=200):
    """Train a PIML model specifically for one track."""
    try:
        dataset = TrackDataset(track_name)
    except ValueError as e:
        print(f"  Skipping {track_name}: {e}")
        return None
    
    if len(dataset) < min_sequences:
        print(f"  Skipping {track_name}: only {len(dataset)} sequences")
        return None
    
    torch.manual_seed(42)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_set, val_set = random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_set, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=32, shuffle=False)
    
    model = TyreLSTM(input_size=8, hidden_size=64, num_layers=2)
    criterion = ThermalPIMLLoss(lambda_physics=0.1, lambda_thermal=0.05)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    best_val = float('inf')
    
    for epoch in range(epochs):
        model.train()
        for x, y in train_loader:
            optimizer.zero_grad()
            predictions = model(x)
            tyre_lives = x[:, -1, 0]
            track_temps = x[:, -1, 5]
            abrasiveness = x[:, -1, 7]
            loss, _, _, _ = criterion(
                predictions, y, tyre_lives, track_temps, abrasiveness
            )
            loss.backward()
            optimizer.step()
        
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for x, y in val_loader:
                predictions = model(x)
                tyre_lives = x[:, -1, 0]
                track_temps = x[:, -1, 5]
                abrasiveness = x[:, -1, 7]
                _, pred_loss, _, _ = criterion(
                    predictions, y, tyre_lives, track_temps, abrasiveness
                )
                val_loss += pred_loss.item()
        
        avg_val = val_loss / len(val_loader)
        if avg_val < best_val:
            best_val = avg_val
    
    print(f"  Best val loss: {best_val:.4f}")
    return model, best_val


def train_all_track_models():
    """Train and save models for all tracks with enough data."""
    os.makedirs('models/tracks', exist_ok=True)
    
    # Get unique track names from data files
    all_files = glob.glob('data/*.csv')
    track_names = set()
    for f in all_files:
        name = os.path.basename(f).replace('.csv', '')
        track = '_'.join(name.split('_')[1:])
        track_names.add(track)
    
    results = {}
    print(f"Training models for {len(track_names)} tracks...\n")
    
    for track in sorted(track_names):
        print(f"Track: {track}")
        result = train_track_model(track)
        if result:
            model, val_loss = result
            safe_name = track.replace(' ', '_')
            path = f'models/tracks/{safe_name}.pt'
            torch.save(model.state_dict(), path)
            results[track] = {'path': path, 'val_loss': round(val_loss, 4)}
    
    # Save track model registry
    import json
    with open('models/tracks/registry.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nTrained {len(results)} track models")
    print(f"Registry saved to models/tracks/registry.json")
    return results


if __name__ == '__main__':
    train_all_track_models() 