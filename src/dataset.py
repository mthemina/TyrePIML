import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
import glob

# Convert compound name to a number the model can understand
COMPOUND_MAP = {'SOFT': 0, 'MEDIUM': 1, 'HARD': 2}

# Normalization values calculated from our dataset
# Min and max lap times across all races
LAP_TIME_MIN = 75.0
LAP_TIME_MAX = 100.0

SECTOR_MIN = 20.0
SECTOR_MAX = 45.0

TYRE_LIFE_MIN = 1.0
TYRE_LIFE_MAX = 55.0

def normalize(value, min_val, max_val):
    """Scale a value to 0-1 range."""
    return (value - min_val) / (max_val - min_val)

def denormalize(value, min_val, max_val):
    """Convert a 0-1 value back to original scale."""
    return value * (max_val - min_val) + min_val

class TyreDataset(Dataset):
    def __init__(self, data_path='data/', sequence_length=5):
        """
        sequence_length: how many consecutive laps the model sees at once
        Think of it as the model looking at the last 5 laps to predict the next one
        """
        self.sequence_length = sequence_length
        self.sequences = []
        self.targets = []
        
        # Load all race CSV files
        files = glob.glob(f'{data_path}*.csv')
        
        for file in files:
            df = pd.read_csv(file)
            self._process_race(df)
        
        print(f"Dataset ready: {len(self.sequences)} sequences")
    
    def _process_race(self, df):
        """Split each driver's stint into sequences."""
        
        for driver in df['Driver'].unique():
            driver_df = df[df['Driver'] == driver]
            
            for stint in driver_df['Stint'].unique():
                stint_df = driver_df[driver_df['Stint'] == stint].reset_index(drop=True)
                
                # Need at least sequence_length + 1 laps to make a sequence
                if len(stint_df) < self.sequence_length + 1:
                    continue
                
                # Build input features for each lap
                features = []
                for _, row in stint_df.iterrows():
                    features.append([
                    normalize(row['TyreLife'], TYRE_LIFE_MIN, TYRE_LIFE_MAX),
                    COMPOUND_MAP[row['Compound']] / 2.0,  # scale 0,1,2 to 0, 0.5, 1
                    normalize(row['Sector1Time'], SECTOR_MIN, SECTOR_MAX),
                    normalize(row['Sector2Time'], SECTOR_MIN, SECTOR_MAX),
                    normalize(row['Sector3Time'], SECTOR_MIN, SECTOR_MAX),
])
                
                features = np.array(features, dtype=np.float32)
                
                # Slide a window of sequence_length across the stint
                for i in range(len(features) - self.sequence_length):
                    self.sequences.append(features[i:i + self.sequence_length])
                    self.targets.append(normalize(stint_df.iloc[i + self.sequence_length]['LapTime'], LAP_TIME_MIN, LAP_TIME_MAX))
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        x = torch.tensor(self.sequences[idx])
        y = torch.tensor(self.targets[idx], dtype=torch.float32)
        return x, y