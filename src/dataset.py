import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
import glob
import os
import json

# Compound encoding
COMPOUND_MAP = {'SOFT': 0, 'MEDIUM': 1, 'HARD': 2}

# Normalization ranges
LAP_TIME_MIN = 60.0
LAP_TIME_MAX = 120.0

SECTOR_MIN = 15.0
SECTOR_MAX = 55.0

TYRE_LIFE_MIN = 1.0
TYRE_LIFE_MAX = 60.0

TEMP_MIN = 15.0
TEMP_MAX = 60.0

ABRASIVENESS_MIN = 1.0
ABRASIVENESS_MAX = 10.0


def normalize(value, min_val, max_val):
    return (value - min_val) / (max_val - min_val)


def denormalize(value, min_val, max_val):
    return value * (max_val - min_val) + min_val


class TyreDataset(Dataset):
    def __init__(self, data_path='data/', sequence_length=5,
                 use_weather=True, use_track=True):
        self.sequence_length = sequence_length
        self.use_weather = use_weather
        self.use_track = use_track
        self.sequences = []
        self.targets = []

        try:
            from src.track_profiles import get_track_profile
            self.get_track_profile = get_track_profile
        except ImportError:
            self.get_track_profile = None

        files = glob.glob(f'{data_path}*.csv')

        if not files:
            raise ValueError(f"No CSV files found in {data_path}")

        for file in sorted(files):
            df = pd.read_csv(file)

            event_name = os.path.basename(file).replace('.csv', '')
            event_name = '_'.join(event_name.split('_')[1:])

            abrasiveness = 5.0
            if self.get_track_profile:
                profile = self.get_track_profile(event_name)
                abrasiveness = profile['abrasiveness']

            self._process_race(df, abrasiveness)

        print(f"Dataset ready: {len(self.sequences)} sequences from {len(files)} races")

    def _get_driver_encoding(self, driver):
        """Get normalized driver style encoding 0-1."""
        try:
            if not hasattr(self, '_driver_profiles'):
                with open('results/driver_profiles.json', 'r') as f:
                    self._driver_profiles = json.load(f)

            profile = self._driver_profiles.get(driver, {})
            rate = profile.get('overall_rate', None)
            if rate is None:
                return 0.5
            rate_clamped = max(-0.1, min(0.1, rate))
            return round((rate_clamped + 0.1) / 0.2, 3)
        except FileNotFoundError:
            return 0.5

    def _process_race(self, df, abrasiveness=5.0):
        """Split each driver's stint into sequences."""

        has_weather = all(col in df.columns
                         for col in ['track_temp_avg', 'air_temp_avg'])

        for driver in df['Driver'].unique():
            driver_df = df[df['Driver'] == driver]
            driver_encoding = self._get_driver_encoding(driver)

            for stint in driver_df['Stint'].unique():
                stint_df = driver_df[
                    driver_df['Stint'] == stint
                ].reset_index(drop=True)

                if len(stint_df) < self.sequence_length + 1:
                    continue

                track_temp = 35.0
                air_temp = 28.0
                if has_weather and self.use_weather:
                    track_temp = float(stint_df['track_temp_avg'].iloc[0]) \
                                if 'track_temp_avg' in stint_df.columns else 35.0
                    air_temp = float(stint_df['air_temp_avg'].iloc[0]) \
                              if 'air_temp_avg' in stint_df.columns else 28.0

                features = []
                for _, row in stint_df.iterrows():
                    feat = [
                        normalize(row['TyreLife'], TYRE_LIFE_MIN, TYRE_LIFE_MAX),
                        COMPOUND_MAP.get(row['Compound'], 1) / 2.0,
                        normalize(row['Sector1Time'], SECTOR_MIN, SECTOR_MAX),
                        normalize(row['Sector2Time'], SECTOR_MIN, SECTOR_MAX),
                        normalize(row['Sector3Time'], SECTOR_MIN, SECTOR_MAX),
                    ]

                    if self.use_weather:
                        feat.append(normalize(track_temp, TEMP_MIN, TEMP_MAX))
                        feat.append(normalize(air_temp, TEMP_MIN, TEMP_MAX))

                    if self.use_track:
                        feat.append(normalize(abrasiveness,
                                            ABRASIVENESS_MIN, ABRASIVENESS_MAX))

                    feat.append(driver_encoding)
                    features.append(feat)

                features = np.array(features, dtype=np.float32)

                for i in range(len(features) - self.sequence_length):
                    self.sequences.append(features[i:i + self.sequence_length])
                    self.targets.append(
                        normalize(
                            stint_df.iloc[i + self.sequence_length]['LapTime'],
                            LAP_TIME_MIN, LAP_TIME_MAX
                        )
                    )

    def get_input_size(self):
        """Return number of features per lap."""
        base = 5
        if self.use_weather:
            base += 2
        if self.use_track:
            base += 1
        base += 1  # driver encoding
        return base

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        x = torch.tensor(self.sequences[idx])
        y = torch.tensor(self.targets[idx], dtype=torch.float32)
        return x, y 