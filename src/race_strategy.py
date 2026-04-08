import torch
import pandas as pd
import numpy as np
from src.model import TyreLSTM
from src.cliff_detector import prepare_sequence, predict_future_laps, detect_cliff_with_confidence
from src.strategy_simulator import simulate_undercut, simulate_overcut, get_future_lap_times
from src.dataset import denormalize, LAP_TIME_MIN, LAP_TIME_MAX

def analyze_all_drivers(model, df, at_lap, n_future=20):
    """
    Analyze tyre state and predicted degradation for all drivers at a given lap.
    Returns a ranked dataframe of driver tyre states.
    """
    results = []
    
    for driver in df['Driver'].unique():
        driver_df = df[df['Driver'] == driver].copy()
        
        # Get their current stint at this lap
        current_laps = driver_df[driver_df['LapNumber'] <= at_lap]
        if len(current_laps) == 0:
            continue
            
        # Get current stint
        current_stint = current_laps['Stint'].iloc[-1]
        stint_df = current_laps[current_laps['Stint'] == current_stint].reset_index(drop=True)
        
        if len(stint_df) < 5:
            continue
        
        # Get tyre info
        current_tyre_life = int(stint_df['TyreLife'].iloc[-1])
        compound = stint_df['Compound'].iloc[-1]
        avg_lap_time = stint_df['LapTime'].mean()
        
        # Predict cliff
        cliff_lap, _, _, _ = detect_cliff_with_confidence(model, stint_df) 
        laps_to_cliff = (cliff_lap - current_tyre_life) if cliff_lap else None
        
        results.append({
            'Driver': driver,
            'Compound': compound,
            'TyreLife': current_tyre_life,
            'AvgLapTime': round(avg_lap_time, 3),
            'PredictedCliff': cliff_lap,
            'LapsToCliff': laps_to_cliff,
            'Urgency': 'HIGH' if laps_to_cliff and laps_to_cliff <= 3 else
                      'MEDIUM' if laps_to_cliff and laps_to_cliff <= 8 else 'LOW'
        })
    
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values('TyreLife', ascending=False)
    return results_df.reset_index(drop=True)

def predict_rival_response(model, df, our_driver, at_lap, our_pit_lap):
    """
    Given our planned pit lap, predict how our closest rivals will respond.
    Returns recommended response for each rival.
    """
    analysis = analyze_all_drivers(model, df, at_lap)
    our_data = analysis[analysis['Driver'] == our_driver]
    
    if len(our_data) == 0:
        return None
    
    our_cliff = our_data['PredictedCliff'].iloc[0]
    responses = []
    
    for _, rival in analysis.iterrows():
        if rival['Driver'] == our_driver:
            continue
        
        rival_cliff = rival['PredictedCliff']
        
        # Can they cover our undercut?
        if pd.isna(rival_cliff):
            can_cover = False
        else:
            # They can cover if their cliff is after our pit lap
            can_cover = rival_cliff > our_pit_lap
        
        # What should they do?
        if can_cover and rival['TyreLife'] < our_data['TyreLife'].iloc[0]:
            response = 'COVER — pit same lap'
        elif not can_cover:
            response = 'CANNOT COVER — tyres too old'
        else:
            response = 'STAY OUT — tyres still alive'
        
        responses.append({
            'Rival': rival['Driver'],
            'RivalCliff': rival_cliff,
            'RivalUrgency': rival['Urgency'],
            'CanCover': can_cover,
            'ExpectedResponse': response
        })
    
    return pd.DataFrame(responses) 

def simulate_race_positions(model, df, at_lap, laps_remaining=20):
    """
    Simulate how positions change lap by lap based on tyre deltas.
    Projects finishing order if everyone stays out vs pits optimally.
    """
    analysis = analyze_all_drivers(model, df, at_lap)
    
    # Get current lap times as position proxy
    # Lower avg lap time = further ahead (simplified)
    drivers = analysis['Driver'].tolist()
    current_times = {row['Driver']: row['AvgLapTime'] 
                    for _, row in analysis.iterrows()}
    
    # Simulate stay out scenario
    stay_out_costs = {}
    pit_optimal_costs = {}
    
    for _, row in analysis.iterrows():
        driver = row['Driver']
        driver_df = df[(df['Driver'] == driver)].copy()
        current_laps = driver_df[driver_df['LapNumber'] <= at_lap]
        if len(current_laps) == 0:
            continue
        current_stint = current_laps['Stint'].iloc[-1]
        stint_df = current_laps[
            current_laps['Stint'] == current_stint
        ].reset_index(drop=True)
        
        if len(stint_df) < 5:
            continue
        
        # Get future lap times
        future = get_future_lap_times(model, stint_df, laps_remaining)
        
        # Stay out — total time on degrading tyres
        stay_out_costs[driver] = sum(future[:laps_remaining])
        
        # Pit optimally — pit at cliff, assume fresh tyre pace after
        cliff = row['PredictedCliff']
        if cliff and not pd.isna(cliff):
            laps_before_pit = max(0, int(cliff) - int(row['TyreLife']))
            laps_after_pit = laps_remaining - laps_before_pit
            fresh_pace = current_times[driver] - 0.5  # fresh tyre is ~0.5s faster
            pit_cost = (sum(future[:laps_before_pit]) + 
                       23.0 +  # pit loss
                       fresh_pace * max(0, laps_after_pit))
        else:
            pit_cost = stay_out_costs[driver]
        
        pit_optimal_costs[driver] = pit_cost
    
    # Rank by total time
    stay_out_order = sorted(stay_out_costs.items(), key=lambda x: x[1])
    pit_optimal_order = sorted(pit_optimal_costs.items(), key=lambda x: x[1])
    
    print(f"{'Pos':<5} {'Stay Out':<10} {'Pit Optimal':<12}")
    print("-" * 30)
    for i in range(min(len(stay_out_order), len(pit_optimal_order))):
        so_driver = stay_out_order[i][0]
        po_driver = pit_optimal_order[i][0]
        print(f"{i+1:<5} {so_driver:<10} {po_driver:<12}")
    
    return stay_out_order, pit_optimal_order 

if __name__ == '__main__':
    model = TyreLSTM()
    model.load_state_dict(torch.load('models/tyre_lstm_piml_v1.pt'))
    
    df = pd.read_csv('data/2023_Monza.csv')
    
    print("=== Multi-Driver Tyre Analysis at Lap 15 ===\n")
    analysis = analyze_all_drivers(model, df, at_lap=15)
    print(analysis.to_string(index=False))
    
    print("\n=== Rival Response if VER pits lap 18 ===\n")
    responses = predict_rival_response(model, df, 'VER', at_lap=15, our_pit_lap=18)
    print(responses.to_string(index=False)) 
    print("\n=== Projected Finishing Order at Lap 15 ===")
    print("(Stay Out vs Pit Optimally)\n")
    simulate_race_positions(model, df, at_lap=15, laps_remaining=20) 