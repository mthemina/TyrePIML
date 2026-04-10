from flask import Flask, jsonify, request, render_template
import torch
import pandas as pd
import sys
import os

# Make sure src is importable
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.model import TyreLSTM
from src.race_strategy import analyze_all_drivers, predict_rival_response
from src.cliff_detector import detect_cliff_with_confidence
from src.strategy_simulator import simulate_undercut, simulate_overcut
from src.plot_pit_window import plot_pit_window

app = Flask(__name__)

# Load model once at startup
model = TyreLSTM()
model.load_state_dict(torch.load('models/tyre_lstm_piml_v1.pt'))
model.eval()

# Available races
RACES = {
    '2023_Monza': 'data/2023_Monza.csv',
    '2023_Silverstone': 'data/2023_Silverstone.csv',
    '2023_Spa': 'data/2023_Spa.csv',
    '2022_Monza': 'data/2022_Monza.csv',
    '2022_Silverstone': 'data/2022_Silverstone.csv',
}

@app.route('/')
def index():
    return render_template('index.html', races=list(RACES.keys()))

@app.route('/api/drivers/<race_key>')
def get_drivers(race_key):
    """Return list of drivers for a given race."""
    if race_key not in RACES:
        return jsonify({'error': 'Race not found'}), 404
    df = pd.read_csv(RACES[race_key])
    drivers = sorted(df['Driver'].unique().tolist())
    return jsonify({'drivers': drivers})

@app.route('/api/analyze', methods=['POST'])
def analyze():
    """Full strategy analysis for a race at a given lap."""
    data = request.json
    race_key = data.get('race')
    at_lap = int(data.get('lap', 15))
    
    if race_key not in RACES:
        return jsonify({'error': 'Race not found'}), 404
    
    df = pd.read_csv(RACES[race_key])
    
    # Multi driver analysis
    analysis = analyze_all_drivers(model, df, at_lap=at_lap)
    analysis = analysis.fillna('N/A')
    
    return jsonify({
        'race': race_key,
        'lap': at_lap,
        'drivers': analysis.to_dict(orient='records')
    })

@app.route('/api/strategy', methods=['POST'])
def strategy():
    """Pit window and rival response for a specific driver."""
    data = request.json
    race_key = data.get('race')
    driver = data.get('driver')
    at_lap = int(data.get('lap', 15))
    pit_lap = int(data.get('pit_lap', at_lap + 3))
    
    if race_key not in RACES:
        return jsonify({'error': 'Race not found'}), 404
    
    df = pd.read_csv(RACES[race_key])
    
    # Rival response
    responses = predict_rival_response(model, df, driver, at_lap, pit_lap)
    if responses is None:
        return jsonify({'error': 'Driver not found'}), 404
    
    responses = responses.fillna('N/A')
    
    return jsonify({
        'driver': driver,
        'planned_pit_lap': pit_lap,
        'rival_responses': responses.to_dict(orient='records')
    })

import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for Flask
import matplotlib.pyplot as plt
import io
import base64
from src.cliff_detector import detect_cliff_with_confidence, predict_future_laps, prepare_sequence
from src.dataset import denormalize, LAP_TIME_MIN, LAP_TIME_MAX

@app.route('/api/pit_chart', methods=['POST'])
def pit_chart():
    """Generate pit window chart for a driver and return as base64 image."""
    data = request.json
    race_key = data.get('race')
    driver = data.get('driver')
    at_lap = int(data.get('lap', 15))
    
    if race_key not in RACES:
        return jsonify({'error': 'Race not found'}), 404
    
    df = pd.read_csv(RACES[race_key])
    driver_laps = df[df['Driver'] == driver]
    current_laps = driver_laps[driver_laps['LapNumber'] <= at_lap]
    
    if len(current_laps) == 0:
        return jsonify({'error': 'Driver not found'}), 404
    
    current_stint = current_laps['Stint'].iloc[-1]
    stint_df = current_laps[
        current_laps['Stint'] == current_stint
    ].reset_index(drop=True)
    
    if len(stint_df) < 5:
        return jsonify({'error': 'Not enough data'}), 400
    
    # Generate predictions
    n_future = 25
    sequence = prepare_sequence(stint_df)
    current_lap_num = int(stint_df['TyreLife'].max())
    future_preds = predict_future_laps(model, sequence, n_future)
    future_seconds = [denormalize(p, LAP_TIME_MIN, LAP_TIME_MAX) 
                     for p in future_preds]
    
    # Cliff and pit window
    cliff_mean, _, _, _ = detect_cliff_with_confidence(model, stint_df)
    current_avg = stint_df['LapTime'].mean()
    
    # Calculate pit deltas
    pit_laps = list(range(current_lap_num + 1, current_lap_num + n_future))
    deltas = []
    for pit_lap in pit_laps:
        laps_before = pit_lap - current_lap_num
        laps_after = n_future - laps_before
        if laps_after <= 0:
            break
        degradation_cost = sum(
            future_seconds[i] - current_avg 
            for i in range(laps_before, min(n_future, laps_before + laps_after))
        )
        delta = round(22.0 - degradation_cost, 3)
        deltas.append(delta)
    
    pit_laps = pit_laps[:len(deltas)]
    optimal_pit = pit_laps[deltas.index(min(deltas))] if deltas else None
    
    # Plot
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), 
                                    facecolor='#1E1E1E')
    
    for ax in [ax1, ax2]:
        ax.set_facecolor('#1E1E1E')
        ax.tick_params(colors='#F0EDE4')
        ax.xaxis.label.set_color('#F0EDE4')
        ax.yaxis.label.set_color('#F0EDE4')
        ax.title.set_color('#C9A84C')
        for spine in ax.spines.values():
            spine.set_edgecolor('#3a3a3a')
    
    # Top — lap time predictions
    actual_laps = stint_df['TyreLife'].tolist()
    future_laps = list(range(current_lap_num + 1, current_lap_num + n_future + 1))
    
    ax1.plot(actual_laps, stint_df['LapTime'], 
             color='#F0EDE4', linewidth=2, label='Actual')
    ax1.plot(future_laps, future_seconds, 
             color='#6B7C3F', linewidth=2, linestyle='--', label='Predicted')
    if cliff_mean:
        ax1.axvline(x=cliff_mean, color='#8B2020', linewidth=1.5,
                    linestyle=':', label=f'Cliff: lap {cliff_mean}')
    if optimal_pit:
        ax1.axvline(x=optimal_pit, color='#C9A84C', linewidth=1.5,
                    linestyle=':', label=f'Optimal pit: lap {optimal_pit}')
    ax1.set_ylabel('Lap Time (s)')
    ax1.set_title(f'{driver} — Degradation Forecast')
    ax1.legend(fontsize=8, facecolor='#2a2a2a', labelcolor='#F0EDE4')
    ax1.grid(True, alpha=0.15)
    
    # Bottom — pit delta bars
    colors = ['#6B7C3F' if d < 0 else '#8B2020' for d in deltas]
    ax2.bar(pit_laps, deltas, color=colors, alpha=0.85)
    ax2.axhline(y=0, color='#F0EDE4', linewidth=0.8)
    if optimal_pit:
        ax2.axvline(x=optimal_pit, color='#C9A84C', linewidth=1.5, linestyle=':')
    ax2.set_xlabel('Pit Lap')
    ax2.set_ylabel('Net Delta (s)')
    ax2.set_title('Pit Window — Green = Gain')
    ax2.grid(True, alpha=0.15)
    
    plt.tight_layout()
    
    # Convert to base64
    buf = io.BytesIO()
    plt.savefig(buf, format='png', facecolor='#1E1E1E', bbox_inches='tight')
    buf.seek(0)
    img_base64 = base64.b64encode(buf.read()).decode('utf-8')
    plt.close()
    
    return jsonify({'chart': img_base64, 'optimal_pit': optimal_pit}) 

@app.route('/api/strategy_map', methods=['POST'])
def strategy_map():
    """Generate strategy map for all drivers and return as base64 image."""
    data = request.json
    race_key = data.get('race')
    
    if race_key not in RACES:
        return jsonify({'error': 'Race not found'}), 404
    
    df = pd.read_csv(RACES[race_key])
    
    COMPOUND_COLORS = {
        'SOFT': '#7a1f1f',
        'MEDIUM': '#C9A84C',
        'HARD': '#3a3a3a'
    }
    
    driver_avg = df.groupby('Driver')['LapTime'].mean().sort_values()
    drivers = driver_avg.index.tolist()
    
    fig, ax = plt.subplots(figsize=(12, 7), facecolor='#1E1E1E')
    ax.set_facecolor('#1E1E1E')
    ax.tick_params(colors='#F0EDE4')
    ax.xaxis.label.set_color('#F0EDE4')
    ax.title.set_color('#C9A84C')
    for spine in ax.spines.values():
        spine.set_edgecolor('#3a3a3a')
    
    for i, driver in enumerate(drivers):
        driver_df = df[df['Driver'] == driver]
        for stint in driver_df['Stint'].unique():
            stint_df = driver_df[driver_df['Stint'] == stint]
            if len(stint_df) == 0:
                continue
            compound = stint_df['Compound'].iloc[0]
            start_lap = stint_df['LapNumber'].min()
            end_lap = stint_df['LapNumber'].max()
            color = COMPOUND_COLORS.get(compound, '#555')
            ax.barh(i, end_lap - start_lap, left=start_lap,
                   height=0.6, color=color, alpha=0.9,
                   edgecolor='#1E1E1E', linewidth=0.8)
            mid_lap = (start_lap + end_lap) / 2
            text_color = '#0F0F0F' if compound == 'MEDIUM' else '#F0EDE4'
            ax.text(mid_lap, i, compound[0],
                   ha='center', va='center',
                   fontsize=7, fontweight='bold', color=text_color)
    
    ax.set_yticks(range(len(drivers)))
    ax.set_yticklabels(drivers, fontsize=8, color='#F0EDE4')
    ax.set_xlabel('Lap Number')
    ax.set_title(f'Strategy Map — {race_key.replace("_", " ")}')
    ax.grid(True, axis='x', alpha=0.15)
    ax.invert_yaxis()
    
    import matplotlib.patches as mpatches
    patches = [mpatches.Patch(color=v, label=k) 
               for k, v in COMPOUND_COLORS.items() 
               if k in df['Compound'].values]
    ax.legend(handles=patches, loc='lower right', fontsize=8,
             facecolor='#2a2a2a', labelcolor='#F0EDE4')
    
    plt.tight_layout()
    
    buf = io.BytesIO()
    plt.savefig(buf, format='png', facecolor='#1E1E1E', bbox_inches='tight')
    buf.seek(0)
    img_base64 = base64.b64encode(buf.read()).decode('utf-8')
    plt.close()
    
    return jsonify({'chart': img_base64}) 

@app.route('/api/undercut', methods=['POST'])
def undercut_analysis():
    """Simulate undercut and overcut for focus driver vs all rivals."""
    data = request.json
    race_key = data.get('race')
    driver = data.get('driver')
    at_lap = int(data.get('lap', 15))
    gap_seconds = float(data.get('gap', 2.5))
    track = race_key.split('_')[1] if '_' in race_key else 'default'

    if race_key not in RACES:
        return jsonify({'error': 'Race not found'}), 404

    df = pd.read_csv(RACES[race_key])
    
    # Get focus driver stint
    driver_laps = df[df['Driver'] == driver]
    current_laps = driver_laps[driver_laps['LapNumber'] <= at_lap]
    if len(current_laps) == 0:
        return jsonify({'error': 'Driver not found'}), 404

    current_stint = current_laps['Stint'].iloc[-1]
    driver_stint = current_laps[
        current_laps['Stint'] == current_stint
    ].reset_index(drop=True)

    results = []

    for rival in df['Driver'].unique():
        if rival == driver:
            continue

        rival_laps = df[df['Driver'] == rival]
        rival_current = rival_laps[rival_laps['LapNumber'] <= at_lap]
        if len(rival_current) == 0:
            continue

        rival_stint_num = rival_current['Stint'].iloc[-1]
        rival_stint = rival_current[
            rival_current['Stint'] == rival_stint_num
        ].reset_index(drop=True)

        if len(rival_stint) < 5 or len(driver_stint) < 5:
            continue

        from src.strategy_simulator import simulate_undercut, simulate_overcut
        
        undercut = simulate_undercut(
            model, driver_stint, rival_stint,
            at_lap, gap_seconds, track
        )
        overcut = simulate_overcut(
            model, driver_stint, rival_stint,
            at_lap, gap_seconds, track
        )

        results.append({
            'Rival': rival,
            'UnderCutWorks': bool(undercut['undercut_works']),
            'UnderCutFinalGap': undercut['final_gap'],
            'OverCutWorks': bool(overcut['overcut_works']),
            'OverCutFinalGap': overcut['final_gap'],
            'Recommendation': 'UNDERCUT' if undercut['undercut_works'] 
                            else 'OVERCUT' if overcut['overcut_works'] 
                            else 'STAY'
        })

    return jsonify({'results': results}) 

if __name__ == '__main__':
    app.run(debug=True, port=5000) 