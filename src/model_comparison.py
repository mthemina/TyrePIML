import sys
import os

# 1. Force Python to recognize the project root FIRST
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 2. THEN do the rest of the imports
import torch
import numpy as np
import pandas as pd
from src.model import TyreLSTM 
from src.cliff_detector import prepare_sequence
from src.dataset import denormalize, LAP_TIME_MIN, LAP_TIME_MAX 

def load_eval_model(path, device):
    """Safely load a model dynamically to test MAE."""
    if not os.path.exists(path):
        return None
    
    state_dict = torch.load(path, map_location=device)
    
    if 'lstm.weight_ih_l0' in state_dict:
        weight_shape = state_dict['lstm.weight_ih_l0'].shape
        in_size = weight_shape[1]
        hid_size = weight_shape[0] // 4
        layers = sum(1 for k in state_dict.keys() if 'weight_ih_l' in k)
    else:
        return None
        
    model = TyreLSTM(input_size=in_size, hidden_size=hid_size, num_layers=layers)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model

def evaluate_models(race_file, target_driver, target_stint):
    """Run head-to-head comparison on a specific stint."""
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Running evaluation on {device.type.upper()}...")

    # Load Data
    df = pd.read_csv(race_file)
    
    # Intelligently parse the track name directly from the filename
    # e.g., 'data/2023_Monza.csv' -> '2023_Monza' -> 'Monza'
    base_name = os.path.basename(race_file).replace('.csv', '')
    track_name = base_name.split('_', 1)[1] if '_' in base_name else base_name 
    stint_df = df[(df['Driver'] == target_driver) & (df['Stint'] == target_stint)].reset_index(drop=True)
    
    if len(stint_df) < 5:
        print("Not enough laps in stint for evaluation.")
        return
        
    compound = stint_df['Compound'].iloc[-1].lower()
    
    # Model Paths
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    generic_path = os.path.join(base_dir, "models", "tyre_lstm_piml_v2.pt")
    compound_path = os.path.join(base_dir, "models", f"tyre_lstm_{compound}_v1.pt")
    track_path = os.path.join(base_dir, "models", "tracks", f"{track_name.replace(' ', '_')}.pt")
    
    # Load Models
    models = {
        "Generic (118 Races)": load_eval_model(generic_path, device),
        f"Compound ({compound.upper()})": load_eval_model(compound_path, device),
        f"Track ({track_name})": load_eval_model(track_path, device)
    }
    
    actual_lap_times = stint_df['LapTime'].values
    
    print(f"\n--- MAE EVALUATION: {track_name} | {target_driver} | Compound: {compound.upper()} ---")
    print(f"{'Model Tier':<25} | {'MAE (Seconds)':<15} | {'Improvement vs Generic'}")
    print("-" * 70)
    
    baseline_mae = None
    
    for tier_name, model in models.items():
        if model is None:
            print(f"{tier_name:<25} | {'NOT FOUND':<15} | N/A")
            continue
            
        # Get dynamic sequence (safely handles 8 vs 9 features)
        sequence = prepare_sequence(model, stint_df)
        
        predictions = []
        with torch.no_grad():
            # NEW: Sliding window approach to preserve the 3D tensor shape (Batch, Seq_Len, Features)
            for i in range(len(sequence)):
                # Grab up to 5 laps of historical context
                start_idx = max(0, i - 4)
                seq_window = sequence[start_idx:i+1]
                
                # Shape becomes (1, Window_Length, Features)
                x = torch.tensor(seq_window).unsqueeze(0).to(device)
                
                pred_norm = model(x).item()
                pred_sec = denormalize(pred_norm, LAP_TIME_MIN, LAP_TIME_MAX)
                predictions.append(pred_sec)
                
        # Calculate Mean Absolute Error
        mae = np.mean(np.abs(np.array(predictions) - actual_lap_times))
        
        if baseline_mae is None:
            baseline_mae = mae
            improvement = "Baseline"
        else:
            pct_improvement = ((baseline_mae - mae) / baseline_mae) * 100
            improvement = f"+{pct_improvement:.1f}%" if pct_improvement > 0 else f"{pct_improvement:.1f}%"
            
        print(f"{tier_name:<25} | {mae:.3f}s{'':<10} | {improvement}") 

if __name__ == '__main__':
    # Adjust this to a CSV file you know exists in your data folder
    evaluate_models('data/2023_Monza.csv', target_driver='VER', target_stint=1) 