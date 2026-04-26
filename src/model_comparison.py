import os

import torch
import pandas as pd
import numpy as np
import glob
from torch.utils.data import DataLoader
from src.dataset import TyreDataset, denormalize, LAP_TIME_MIN, LAP_TIME_MAX
from src.model import TyreLSTM
from src.compound_models import CompoundDataset
from src.track_models import TrackDataset

def evaluate_model(model, dataloader, input_size):
    """Evaluate a model on a dataloader and return MAE in seconds."""
    model.eval()
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for x, y in dataloader:
            # Handle input size mismatch
            if x.shape[2] != input_size:
                x = x[:, :, :input_size]
            preds = model(x)
            all_preds.extend(preds.numpy())
            all_targets.extend(y.numpy())
    
    pred_sec = [denormalize(p, LAP_TIME_MIN, LAP_TIME_MAX) for p in all_preds]
    true_sec = [denormalize(t, LAP_TIME_MIN, LAP_TIME_MAX) for t in all_targets]
    mae = np.mean(np.abs(np.array(pred_sec) - np.array(true_sec)))
    return round(float(mae), 4)

from src.transformer_model import TyreTransformer
transformer = TyreTransformer(input_size=9, d_model=64, nhead=4, num_layers=2)
transformer.load_state_dict(torch.load('models/tyre_transformer_v1.pt')) 

def run_comparison():
    print("Loading models...")
    
    # Generic model
    generic = TyreLSTM(input_size=9, hidden_size=128, num_layers=2)
    generic.load_state_dict(torch.load('models/tyre_lstm_piml_v2_train2022.pt')) 
    
    # Compound models
    compound_mdls = {}
    for c in ['SOFT', 'MEDIUM', 'HARD']:
        m = TyreLSTM(input_size=7, hidden_size=64, num_layers=2)
        m.load_state_dict(torch.load(f'models/tyre_lstm_{c.lower()}_v1.pt'))
        compound_mdls[c] = m
    
    # Track models
    import json, os
    with open('models/tracks/registry.json') as f:
        registry = json.load(f)
    track_mdls = {}
    for track, info in registry.items():
        if os.path.exists(info['path']):
            m = TyreLSTM(input_size=8, hidden_size=64, num_layers=2)
            m.load_state_dict(torch.load(info['path']))
            track_mdls[track] = m
    
    print(f"Generic model: 1")
    print(f"Compound models: {len(compound_mdls)}")
    print(f"Track models: {len(track_mdls)}")
    
    # Only test on races the generic model has NOT seen — 2023 and 2024
    all_files = glob.glob('data/*.csv')
    test_races = [
        os.path.basename(f).replace('.csv', '') 
        for f in all_files 
        if os.path.basename(f).startswith('2023_') or 
           os.path.basename(f).startswith('2024_')
    ]
    print(f"Testing on {len(test_races)} held-out races (2023-2024)")
    
    print(f"\n{'Race':<25} {'LSTM':>8} {'Transformer':>12} {'Compound':>10} {'Track':>10} {'Best':>14}")
    print("-" * 82) 
    
    results = []
    
    for race in test_races:
        filepath = f'data/{race}.csv'
        if not os.path.exists(filepath):
            continue
        
        df = pd.read_csv(filepath)
        
        # Generic MAE
        from src.dataset import TyreDataset
        import tempfile, shutil
        tmpdir = tempfile.mkdtemp()
        shutil.copy(filepath, f"{tmpdir}/{race}.csv")
        
        try:
            ds = TyreDataset(data_path=f"{tmpdir}/", sequence_length=5)
            loader = DataLoader(ds, batch_size=32)
            generic_mae = evaluate_model(generic, loader, 9)
        except:
            generic_mae = None

        try:
            ds_t = TyreDataset(data_path=f"{tmpdir}/", sequence_length=5)
            loader_t = DataLoader(ds_t, batch_size=32)
            transformer_mae = evaluate_model(transformer, loader_t, 9)
        except:
            transformer_mae = None

        
        # Compound MAE — average across compounds present
        compound_maes = []
        for compound in df['Compound'].unique():
            if compound not in compound_mdls:
                continue
            try:
                ds = CompoundDataset(compound, data_path=f"{tmpdir}/")
                if len(ds) < 10:
                    continue
                loader = DataLoader(ds, batch_size=32)
                mae = evaluate_model(compound_mdls[compound], loader, 7)
                compound_maes.append(mae)
            except:
                pass
        compound_mae = round(float(np.mean(compound_maes)), 4) if compound_maes else None
        
        # Track MAE
        track_name = '_'.join(race.split('_')[1:])
        track_mae = None
        for key in track_mdls:
            if key.lower() in track_name.lower() or track_name.lower() in key.lower():
                try:
                    ds = TrackDataset(track_name, data_path=f"{tmpdir}/")
                    if len(ds) < 10:
                        continue
                    loader = DataLoader(ds, batch_size=32)
                    track_mae = evaluate_model(track_mdls[key], loader, 8)
                except:
                    pass
                break
        
        shutil.rmtree(tmpdir)
        
        maes = {
            'LSTM': generic_mae,
            'Transformer': transformer_mae,
            'Compound': compound_mae,
            'Track': track_mae
        }
        best_name = min((k for k in maes if maes[k]), key=lambda k: maes[k]) # type: ignore
        best_val = maes[best_name]

        print(f"{race:<25} "
              f"{str(generic_mae):>8} "
              f"{str(transformer_mae):>12} "
              f"{str(compound_mae):>10} "
              f"{str(track_mae):>10} "
              f"{f'{best_name}({best_val})':>14}")

        results.append({
            'race': race,
            'generic_mae': generic_mae,
            'transformer_mae': transformer_mae,
            'compound_mae': compound_mae,
            'track_mae': track_mae,
            'best': best_name
        }) 
    
    # Summary
    trans_avg = np.mean([r['transformer_mae'] for r in results if r.get('transformer_mae')])
    gen_avg = np.mean([r['generic_mae'] for r in results if r.get('generic_mae')])
    cmp_avg = np.mean([r['compound_mae'] for r in results if r.get('compound_mae')])
    trk_avg = np.mean([r['track_mae'] for r in results if r.get('track_mae')])

    print("-" * 82)
    print(f"{'AVERAGE':<25} {gen_avg:>8.4f} {trans_avg:>12.4f} {cmp_avg:>10.4f} {trk_avg:>10.4f}")
    print(f"\nTransformer vs LSTM: {round((gen_avg - trans_avg) / gen_avg * 100, 1)}% improvement")
    print(f"Track models vs LSTM: {round((gen_avg - trk_avg) / gen_avg * 100, 1)}% difference") 

if __name__ == '__main__':
    run_comparison() 