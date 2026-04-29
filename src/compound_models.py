import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
import pandas as pd
import numpy as np
import glob
import os
from src.model import TyreLSTM
from src.dataset import (COMPOUND_MAP, normalize, denormalize,
                         LAP_TIME_MIN, LAP_TIME_MAX, TYRE_LIFE_MIN,
                         TYRE_LIFE_MAX, SECTOR_MIN, SECTOR_MAX,
                         TEMP_MIN, TEMP_MAX, ABRASIVENESS_MIN, ABRASIVENESS_MAX)
from src.track_profiles import get_track_profile
from src.piml_loss import PIMLLoss
from src.transformer_model import TyreTransformer 


class CompoundDataset(Dataset):
    """Dataset filtered to a single compound."""
    
    def __init__(self, compound, data_path='data/', sequence_length=5):
        self.compound = compound
        self.sequence_length = sequence_length
        self.sequences = []
        self.targets = []
        
        files = glob.glob(f'{data_path}*.csv')
        
        for file in sorted(files):
            df = pd.read_csv(file)
            
            # Filter to this compound only
            df = df[df['Compound'] == compound]
            if len(df) == 0:
                continue
            
            event_name = '_'.join(
                os.path.basename(file).replace('.csv','').split('_')[1:]
            )
            profile = get_track_profile(event_name)
            abrasiveness = profile['abrasiveness']
            
            self._process_race(df, abrasiveness)
        
        print(f"  {compound}: {len(self.sequences)} sequences")
    
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
                    from src.thermal_model import calculate_thermal_energy
                    thermal = calculate_thermal_energy(
                        lap_time=row['LapTime'],
                        sector1=row['Sector1Time'],
                        sector2=row['Sector2Time'],
                        sector3=row['Sector3Time'],
                        track_temp=track_temp,
                        compound=row['Compound'],
                        abrasiveness=abrasiveness,
                        tyre_life=row['TyreLife']
                    )
                    features.append([
                        normalize(row['TyreLife'], TYRE_LIFE_MIN, TYRE_LIFE_MAX),
                        normalize(row['Sector1Time'], SECTOR_MIN, SECTOR_MAX),
                        normalize(row['Sector2Time'], SECTOR_MIN, SECTOR_MAX),
                        normalize(row['Sector3Time'], SECTOR_MIN, SECTOR_MAX),
                        normalize(track_temp, TEMP_MIN, TEMP_MAX),
                        normalize(air_temp, TEMP_MIN, TEMP_MAX),
                        normalize(abrasiveness, ABRASIVENESS_MIN, ABRASIVENESS_MAX),
                        thermal,
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


def train_compound_model(compound, epochs=25):
    """Train a PIML model specifically for one compound."""
    print(f"\nTraining {compound} model...")
    
    dataset = CompoundDataset(compound)
    
    if len(dataset) < 100:
        print(f"  Not enough data for {compound}, skipping")
        return None
    
    torch.manual_seed(42)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_set, val_set = random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_set, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=32, shuffle=False)
    
    # Input size is 8 (no compound feature — we have separate models)
    model = TyreTransformer(input_size=8, d_model=64, nhead=4, num_layers=2) 
    criterion = PIMLLoss(lambda_physics=0.1)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for x, y in train_loader:
            optimizer.zero_grad()
            predictions = model(x)
            tyre_lives = x[:, -1, 0]
            loss, pred_loss, _ = criterion(predictions, y, tyre_lives)
            loss.backward()
            optimizer.step()
            train_loss += pred_loss.item()
        
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for x, y in val_loader:
                predictions = model(x)
                tyre_lives = x[:, -1, 0]
                _, pred_loss, _ = criterion(predictions, y, tyre_lives)
                val_loss += pred_loss.item()
        
        if (epoch + 1) % 5 == 0:
            avg_train = train_loss / len(train_loader)
            avg_val = val_loss / len(val_loader)
            print(f"  Epoch {epoch+1:2d}/{epochs} — "
                  f"Train: {avg_train:.4f} Val: {avg_val:.4f}")
    
    return model


def train_all_compound_models():
    """Train and save models for all three compounds."""
    os.makedirs('models', exist_ok=True)
    
    results = {}
    for compound in ['SOFT', 'MEDIUM', 'HARD']:
        model = train_compound_model(compound)
        if model:
            path = f'models/tyre_transformer_{compound.lower()}_v1.pt' 
            torch.save(model.state_dict(), path)
            print(f"  Saved {path}")
            results[compound] = path
    
    return results


if __name__ == '__main__':
    print("Training compound-specific PIML models...\n")
    results = train_all_compound_models()
    print(f"\nAll compound models saved: {results}") 