import sys
import os

# 1. Force Python to recognize the project root FIRST
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 2. THEN do the rest of the imports
import torch
import pandas as pd
import numpy as np
from src.model import TyreLSTM # NOTE: Ensure this matches your filename (model vs models)
from src.cliff_detector import prepare_sequence, predict_future_laps
from src.dataset import denormalize, LAP_TIME_MIN, LAP_TIME_MAX

# Base pit lane time loss per track (seconds)
PIT_LOSS_BASE = {
    'Monza': 22.0,
    'Silverstone': 24.0,
    'Spa': 23.5,
    'default': 23.0
}

def get_future_lap_times(model, stint_df, n_future=20):
    """Get predicted future lap times for a driver."""
    sequence = prepare_sequence(model, stint_df) # Updated to use dynamic prepare_sequence
    future_preds = predict_future_laps(model, sequence, n_future)
    return [denormalize(p, LAP_TIME_MIN, LAP_TIME_MAX) for p in future_preds]

def calculate_pit_loss(track, simulate_variance=False):
    """
    Returns the pit loss for a given track.
    If simulate_variance is True, applies a probabilistic distribution 
    to model the risk of slow pit stops.
    """
    base_loss = PIT_LOSS_BASE.get(track, PIT_LOSS_BASE['default'])
    
    if not simulate_variance:
        return base_loss
        
    # F1 Pit Stop Distribution:
    # 85% chance of normal stop (variance +/- 0.3s)
    # 10% chance of slow stop (variance +1.0s to +2.5s)
    # 5% chance of botched stop (variance +3.0s to +8.0s)
    
    rand_val = np.random.random()
    if rand_val < 0.85:
        return base_loss + np.random.normal(0, 0.15)
    elif rand_val < 0.95:
        return base_loss + np.random.uniform(1.0, 2.5)
    else:
        return base_loss + np.random.uniform(3.0, 8.0)

def simulate_undercut(model, driver1_df, driver2_df, 
                      current_lap, gap_seconds, track='default', n_future=20, 
                      apply_dirty_air=True, simulate_pit_variance=False):
    """
    Simulate whether driver1 should undercut driver2.
    """
    pit_loss = calculate_pit_loss(track, simulate_variance=simulate_pit_variance)
    
    d1_future = get_future_lap_times(model, driver1_df, n_future)
    d2_future = get_future_lap_times(model, driver2_df, n_future)
    
    gap_after_pit = gap_seconds + pit_loss
    
    gap_evolution = [gap_after_pit]
    cumulative_gap = gap_after_pit
    
    for i in range(min(n_future, len(d1_future), len(d2_future))):
        d1_pace = d1_future[i]
        d2_pace = d2_future[i]
        
        # DIRTY AIR PENALTY LOGIC
        # If D1 is within 1.5s behind D2, D1's pace suffers
        if apply_dirty_air and 0 < cumulative_gap <= 1.5:
             # Scale penalty based on closeness (closer = more dirty air)
             penalty = 0.3 * (1.5 - cumulative_gap) / 1.5
             d1_pace += penalty 
             
        # If D2 is within 1.5s behind D1 (D1 has passed them)
        if apply_dirty_air and -1.5 <= cumulative_gap < 0:
             penalty = 0.3 * (1.5 - abs(cumulative_gap)) / 1.5
             d2_pace += penalty

        lap_delta = d2_pace - d1_pace
        cumulative_gap -= lap_delta
        gap_evolution.append(round(cumulative_gap, 3))
    
    undercut_works = cumulative_gap < 0
    laps_to_overtake = next((i for i, gap in enumerate(gap_evolution) if gap < 0), None)
    
    return {
        'undercut_works': undercut_works,
        'final_gap': round(cumulative_gap, 3),
        'laps_to_overtake': laps_to_overtake,
        'gap_evolution': gap_evolution,
        'initial_gap': gap_seconds,
        'pit_loss': pit_loss
    }


def simulate_overcut(model, driver1_df, driver2_df,
                     current_lap, gap_seconds, track='default', n_future=20,
                     apply_dirty_air=True, simulate_pit_variance=False):
    """
    Simulate whether driver1 should overcut driver2.
    """
    pit_loss = calculate_pit_loss(track, simulate_variance=simulate_pit_variance)
    
    d1_future = get_future_lap_times(model, driver1_df, n_future)
    d2_future = get_future_lap_times(model, driver2_df, n_future)
    
    gap_after_d2_pit = gap_seconds - pit_loss
    
    gap_evolution = [gap_after_d2_pit]
    cumulative_gap = gap_after_d2_pit
    
    for i in range(min(n_future, len(d1_future), len(d2_future))):
        d1_pace = d1_future[i]
        d2_pace = d2_future[i]
        
        # DIRTY AIR PENALTY LOGIC
        if apply_dirty_air and 0 < cumulative_gap <= 1.5:
             penalty = 0.3 * (1.5 - cumulative_gap) / 1.5
             d1_pace += penalty 
             
        if apply_dirty_air and -1.5 <= cumulative_gap < 0:
             penalty = 0.3 * (1.5 - abs(cumulative_gap)) / 1.5
             d2_pace += penalty
             
        lap_delta = d1_pace - d2_pace
        cumulative_gap += lap_delta
        gap_evolution.append(round(cumulative_gap, 3))
    
    overcut_works = cumulative_gap < 0
    laps_to_overtake = next((i for i, gap in enumerate(gap_evolution) if gap < 0), None)
    
    return {
        'overcut_works': overcut_works,
        'final_gap': round(cumulative_gap, 3),
        'laps_to_overtake': laps_to_overtake,
        'gap_evolution': gap_evolution,
        'initial_gap': gap_seconds,
        'pit_loss': pit_loss
    }

def simulate_stint_extension(model, stint_df, current_lap, optimal_pit_lap, extended_pit_lap, track='default'):
    """
    Calculates the exact race time penalty for ignoring the optimal pit window 
    and staying out longer on degrading tyres.
    """
    pit_loss = calculate_pit_loss(track, simulate_variance=False) # Keep clean for baseline comparison
    
    # We need to simulate far enough into the future to cover the extended stint
    laps_ahead = (extended_pit_lap - current_lap) + 5 
    future_preds = get_future_lap_times(model, stint_df, laps_ahead)
    
    current_avg = stint_df['LapTime'].mean()
    
    # Scenario A: Pit on Optimal Lap
    # Cost = Pit loss + degradation suffered before the optimal lap
    optimal_deg_cost = sum(future_preds[i] - current_avg for i in range(optimal_pit_lap - current_lap))
    time_scenario_a = pit_loss + optimal_deg_cost
    
    # Scenario B: Pit on Extended Lap
    # Cost = Pit loss + massive degradation suffered by driving over the cliff
    extended_deg_cost = sum(future_preds[i] - current_avg for i in range(extended_pit_lap - current_lap))
    time_scenario_b = pit_loss + extended_deg_cost
    
    time_penalty = time_scenario_b - time_scenario_a
    
    return {
        'optimal_lap': optimal_pit_lap,
        'extended_lap': extended_pit_lap,
        'time_penalty_seconds': round(time_penalty, 3),
        'recommendation': "PULL THEM IN NOW" if time_penalty > 2.0 else "MARGINAL" if time_penalty > 0 else "STAY OUT"
    } 

def evaluate_safety_car_opportunity(model, stint_df, current_lap, optimal_pit_lap, track='default', sc_type='VSC'):
    """
    Calculates if a driver should dive into the pits under a VSC/SC,
    even if they haven't reached their optimal pit window yet.
    """
    # Base pit loss under green flag
    normal_pit_loss = calculate_pit_loss(track, simulate_variance=False)
    
    # VSC saves about 40% of the pit loss relative to the field. SC saves about 50%.
    discount = 0.60 if sc_type == 'VSC' else 0.50
    cheap_pit_loss = normal_pit_loss * discount
    
    # We need future predictions to see what happens if we stay out
    laps_ahead = (optimal_pit_lap - current_lap) + 5
    future_preds = get_future_lap_times(model, stint_df, laps_ahead)
    current_avg = stint_df['LapTime'].mean()
    
    # Scenario A: Ignore VSC, pit later on the optimal lap under Green Flag
    # Cost = Normal Pit Loss + Degradation suffered until optimal lap
    optimal_deg_cost = sum(future_preds[i] - current_avg for i in range(optimal_pit_lap - current_lap))
    time_scenario_a = normal_pit_loss + optimal_deg_cost
    
    # Scenario B: Pit NOW under VSC
    # Cost = Cheap Pit Loss + Zero degradation cost (since we pit immediately)
    time_scenario_b = cheap_pit_loss + 0 
    
    net_savings = time_scenario_a - time_scenario_b
    
    return {
        'sc_type': sc_type,
        'normal_pit_loss': round(normal_pit_loss, 2),
        'cheap_pit_loss': round(cheap_pit_loss, 2),
        'net_savings_seconds': round(net_savings, 3),
        'recommendation': "BOX BOX BOX" if net_savings > 0 else "STAY OUT"
    } 

if __name__ == '__main__':
    from src.model_router import get_best_model
    df = pd.read_csv('data/2023_Monza.csv')
    
    # Load optimal model dynamically 
    model, tier = get_best_model('Monza', 'MEDIUM')
    print(f"Loaded: {tier}")
    
    ver = df[(df['Driver'] == 'VER') & (df['Stint'] == 1)].reset_index(drop=True)
    sai = df[(df['Driver'] == 'SAI') & (df['Stint'] == 1)].reset_index(drop=True)
    
    print("\n=== Undercut Simulation (Deterministic vs Chaos) ===")
    
    res_clean = simulate_undercut(model, ver, sai, 15, 2.5, 'Monza', 15, apply_dirty_air=False)
    res_chaos = simulate_undercut(model, ver, sai, 15, 2.5, 'Monza', 15, apply_dirty_air=True, simulate_pit_variance=True)
    
    print(f"Clean Air Final Gap:   {res_clean['final_gap']:+.3f}s")
    print(f"Dirty Air & Pit Risk:  {res_chaos['final_gap']:+.3f}s (Pit Time: {res_chaos['pit_loss']:.2f}s)") 
    print("\n=== Stint Extension Simulator ===")
    print("Pit Wall: 'Max, optimal pit is Lap 20.'")
    print("Verstappen: 'Tyres are good, let's go to Lap 24.'\n")
    
    extension_risk = simulate_stint_extension(
        model, ver, 
        current_lap=15, 
        optimal_pit_lap=20, 
        extended_pit_lap=24, 
        track='Monza'
    )
    
    print(f"Cost of ignoring optimal window: +{extension_risk['time_penalty_seconds']}s")
    print(f"Pit Wall Decision: {extension_risk['recommendation']}") 
    print("\n=== Safety Car Trigger ===")
    print("Lap 15: Yellow flags in Sector 2. VSC Deployed.")
    print("Verstappen's optimal green-flag pit is Lap 20. Do we box him now?\n")
    
    sc_opp = evaluate_safety_car_opportunity(
        model, ver, 
        current_lap=15, 
        optimal_pit_lap=20, 
        track='Monza',
        sc_type='VSC'
    )
    
    print(f"Normal Pit Loss: {sc_opp['normal_pit_loss']}s")
    print(f"VSC Pit Loss:    {sc_opp['cheap_pit_loss']}s")
    print(f"Net time saved by pitting now: {sc_opp['net_savings_seconds']:+.3f}s")
    print(f"Pit Wall Decision: {sc_opp['recommendation']}") 