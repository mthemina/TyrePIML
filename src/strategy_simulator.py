import torch
import pandas as pd
import numpy as np
from src.model import TyreLSTM
from src.cliff_detector import prepare_sequence, predict_future_laps
from src.dataset import denormalize, LAP_TIME_MIN, LAP_TIME_MAX

# Average pit lane time loss per track (seconds)
PIT_LOSS = {
    'Monza': 22.0,
    'Silverstone': 24.0,
    'Spa': 23.5,
    'default': 23.0
}

def get_future_lap_times(model, stint_df, n_future=20):
    """Get predicted future lap times for a driver."""
    sequence = prepare_sequence(stint_df)
    future_preds = predict_future_laps(model, sequence, n_future)
    return [denormalize(p, LAP_TIME_MIN, LAP_TIME_MAX) for p in future_preds]


def simulate_undercut(model, driver1_df, driver2_df, 
                      current_lap, gap_seconds, track='default', n_future=20):
    """
    Simulate whether driver1 should undercut driver2.
    
    driver1: the car behind (considering the undercut)
    driver2: the car ahead (being undercut)
    gap_seconds: current gap between driver1 and driver2 (positive = driver1 is behind)
    
    Returns: whether undercut works and by how much
    """
    pit_loss = PIT_LOSS.get(track, PIT_LOSS['default'])
    
    # Get future lap times for both drivers
    d1_future = get_future_lap_times(model, driver1_df, n_future)
    d2_future = get_future_lap_times(model, driver2_df, n_future)
    
    # Simulate gap evolution if driver1 pits now
    # Driver1 loses pit_loss seconds immediately
    gap_after_pit = gap_seconds + pit_loss
    
    # Fresh tyre advantage — driver1 will be faster after pitting
    # Estimate fresh tyre pace as current average minus degradation offset
    d1_avg = driver1_df['LapTime'].mean()
    d2_avg = driver2_df['LapTime'].mean()
    
    # Simulate lap by lap gap evolution after undercut
    gap_evolution = [gap_after_pit]
    cumulative_gap = gap_after_pit
    
    for i in range(min(n_future, len(d1_future), len(d2_future))):
        # Driver1 on fresh tyres vs driver2 on old tyres
        lap_delta = d2_future[i] - d1_future[i]
        cumulative_gap -= lap_delta
        gap_evolution.append(round(cumulative_gap, 3))
    
    # Find when/if driver1 gets ahead
    undercut_works = cumulative_gap < 0
    laps_to_overtake = None
    
    for i, gap in enumerate(gap_evolution):
        if gap < 0:
            laps_to_overtake = i
            break
    
    return {
        'undercut_works': undercut_works,
        'final_gap': round(cumulative_gap, 3),
        'laps_to_overtake': laps_to_overtake,
        'gap_evolution': gap_evolution,
        'initial_gap': gap_seconds,
        'pit_loss': pit_loss
    }


def simulate_overcut(model, driver1_df, driver2_df,
                     current_lap, gap_seconds, track='default', n_future=20):
    """
    Simulate whether driver1 should overcut driver2.
    
    driver1: the car behind (staying out longer)
    driver2: the car ahead (pitting now)
    """
    pit_loss = PIT_LOSS.get(track, PIT_LOSS['default'])
    
    d1_future = get_future_lap_times(model, driver1_df, n_future)
    d2_future = get_future_lap_times(model, driver2_df, n_future)
    
    # Driver2 pits this lap — loses pit_loss seconds
    # Driver1 stays out — gains pit_loss seconds on driver2 temporarily
    gap_after_d2_pit = gap_seconds - pit_loss
    
    # Simulate gap evolution
    gap_evolution = [gap_after_d2_pit]
    cumulative_gap = gap_after_d2_pit
    
    for i in range(min(n_future, len(d1_future), len(d2_future))):
        # Driver1 on old tyres vs driver2 on fresh tyres
        lap_delta = d1_future[i] - d2_future[i]
        cumulative_gap += lap_delta
        gap_evolution.append(round(cumulative_gap, 3))
    
    overcut_works = cumulative_gap < 0
    laps_to_overtake = None
    
    for i, gap in enumerate(gap_evolution):
        if gap < 0:
            laps_to_overtake = i
            break
    
    return {
        'overcut_works': overcut_works,
        'final_gap': round(cumulative_gap, 3),
        'laps_to_overtake': laps_to_overtake,
        'gap_evolution': gap_evolution,
        'initial_gap': gap_seconds,
        'pit_loss': pit_loss
    } 

if __name__ == '__main__':
    model = TyreLSTM()
    model.load_state_dict(torch.load('models/tyre_lstm_piml_v1.pt'))
    
    df = pd.read_csv('data/2023_Monza.csv')
    
    # Verstappen vs Sainz — Monza 2023
    # At lap 15, Sainz was running ahead of Verstappen
    ver = df[(df['Driver'] == 'VER') & (df['Stint'] == 1)].reset_index(drop=True)
    sai = df[(df['Driver'] == 'SAI') & (df['Stint'] == 1)].reset_index(drop=True)
    
    print("=== Undercut Simulation ===")
    print("VER considering undercut on SAI at lap 15")
    print("Assumed gap: SAI is 2.5 seconds ahead of VER\n")
    
    undercut = simulate_undercut(
        model, ver, sai,
        current_lap=15,
        gap_seconds=2.5,
        track='Monza',
        n_future=15
    )
    
    print(f"Undercut works: {undercut['undercut_works']}")
    print(f"Final gap after {len(undercut['gap_evolution'])} laps: {undercut['final_gap']}s")
    if undercut['laps_to_overtake']:
        print(f"VER gets ahead after: {undercut['laps_to_overtake']} laps")
    print(f"\nGap evolution (positive = SAI ahead, negative = VER ahead):")
    for i, gap in enumerate(undercut['gap_evolution']):
        bar = '█' * int(abs(gap))
        side = 'SAI' if gap > 0 else 'VER'
        print(f"  Lap +{i:2d}: {gap:+.2f}s  {side} ahead") 
    
    print("\n=== Overcut Simulation ===")
    print("VER staying out while SAI pits\n")
    
    overcut = simulate_overcut(
        model, ver, sai,
        current_lap=15,
        gap_seconds=2.5,
        track='Monza',
        n_future=15
    )
    
    print(f"Overcut works: {overcut['overcut_works']}")
    print(f"Final gap: {overcut['final_gap']}s") 