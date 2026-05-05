import gevent.monkey
gevent.monkey.patch_all() 

import warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="gevent.monkey") 

from flask import Flask, jsonify, request, render_template
from flask_socketio import SocketIO, emit
import torch
import pandas as pd
import sys
import os
import numpy as np 
import json 
from functools import lru_cache 

# Make sure src is importable
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.uncertainty import predict_with_uncertainty, predict_cliff_with_uncertainty  
from src.model import TyreLSTM
from src.race_strategy import analyze_all_drivers, predict_rival_response
from src.cliff_detector import detect_cliff_with_confidence
from src.strategy_simulator import simulate_undercut, simulate_overcut
from src.plot_pit_window import plot_pit_window

app = Flask(__name__)
app.config['SECRET_KEY'] = 'pitwall_super_secret'
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='gevent') # <-- CHANGED HERE 

# Load models at startup
print("Loading models...")

from src.transformer_model import TyreTransformer

# Primary model — Transformer
main_model = TyreTransformer(input_size=10, d_model=64, nhead=4, num_layers=2)
main_model.load_state_dict(torch.load('models/tyre_transformer_v1.pt'))
main_model.eval()
model = main_model 
print("  Loaded Transformer v1 as primary model")

# LSTM fallback
lstm_fallback = TyreLSTM(input_size=9, hidden_size=128, num_layers=2)
lstm_fallback.load_state_dict(torch.load('models/tyre_lstm_piml_v2.pt'))
lstm_fallback.eval()
print("  Loaded LSTM v2 as fallback")

# Transformer compound models
compound_models = {}
for compound in ['soft', 'medium', 'hard']:
    path = f'models/tyre_transformer_{compound}_v1.pt'
    if os.path.exists(path):
        m = TyreTransformer(input_size=8, d_model=64, nhead=4, num_layers=2)
        m.load_state_dict(torch.load(path))
        m.eval()
        compound_models[compound.upper()] = m
        print(f"  Loaded Transformer {compound} model")
    else:
        # Fall back to LSTM compound model
        path = f'models/tyre_lstm_{compound}_v1.pt'
        if os.path.exists(path):
            m = TyreLSTM(input_size=8, hidden_size=64, num_layers=2)
            m.load_state_dict(torch.load(path))
            m.eval()
            compound_models[compound.upper()] = m
            print(f"  Loaded LSTM {compound} model (fallback)") 

print(f"Models ready. Compound models: {list(compound_models.keys())}")

# Track-specific models
track_models = {}
track_registry_path = 'models/tracks/registry.json'
if os.path.exists(track_registry_path):
    with open(track_registry_path, 'r') as f:
        track_registry = json.load(f)
    
    for track_name, info in track_registry.items():
        if os.path.exists(info['path']):
            arch = info.get('arch', 'lstm')
            if arch == 'transformer':
                m = TyreTransformer(input_size=9, d_model=64, nhead=4, num_layers=2)
            else:
                m = TyreLSTM(input_size=9, hidden_size=64, num_layers=2)
            m.load_state_dict(torch.load(info['path']))
            m.eval()
            track_models[track_name] = m 
    
    print(f"  Loaded {len(track_models)} track models")

def get_model_for_driver(stint_df):
    """Return the best model for a driver's current compound."""
    compound = stint_df['Compound'].iloc[-1] if len(stint_df) > 0 else 'MEDIUM'
    if compound in compound_models:
        return compound_models[compound], 7
    return main_model, 9

def get_best_model(race_key, stint_df):
    """
    Smart model router.
    Priority: Transformer compound → track-specific → generic Transformer → LSTM fallback
    """
    compound = stint_df['Compound'].iloc[-1] if len(stint_df) > 0 else 'MEDIUM'
    track_name = '_'.join(race_key.split('_')[1:]) if '_' in race_key else race_key

    # 1. Transformer compound model
    if compound in compound_models:
        return compound_models[compound], 7, f"Transformer — {compound} Compound"

    # 2. Track-specific model
    for key in track_models:
        if key.lower() in track_name.lower() or track_name.lower() in key.lower():
            return track_models[key], 8, f"Track — {key}"

    # 3. Generic Transformer
    return main_model, 9, "Transformer — Full Dataset"  

# Available races
RACES = {}

# Auto-discover all CSV files in data/
import glob
for filepath in sorted(glob.glob('data/*.csv')):
    filename = os.path.basename(filepath).replace('.csv', '')
    RACES[filename] = filepath

print(f"Loaded {len(RACES)} races") 

# --- TASK 4: 60-FPS MEMORY CACHE ---
@lru_cache(maxsize=10)
def load_race_data(race_key):
    """Prevents the server from reading the hard drive 5 times per tick."""
    return pd.read_csv(RACES[race_key])

@app.route('/')
def index():
    return render_template('index.html', races=list(RACES.keys()))

@app.route('/api/drivers/<race_key>')
def get_drivers(race_key):
    """Return list of drivers for a given race."""
    if race_key not in RACES:
        return jsonify({'error': 'Race not found'}), 404
    
    # 60-FPS CACHE INTEGRATION
    df = load_race_data(race_key)
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
    
    # 60-FPS CACHE INTEGRATION
    df = load_race_data(race_key)
    
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
    
    # Grab the 2026 toggle flag from the frontend
    is_2026 = data.get('is_2026', False)
    
    if race_key not in RACES:
        return jsonify({'error': 'Race not found'}), 404
    
    # 60-FPS CACHE INTEGRATION
    df = load_race_data(race_key)
    
    # 1. Fetch Legacy Rival response (gives us the base dataframe)
    responses = predict_rival_response(model, df, driver, at_lap, pit_lap)
    if responses is None:
        return jsonify({'error': 'Driver not found'}), 404
        
    # --- 2. MULTI-AGENT GAME THEORY INJECTION (2026 MODE) ---
    if is_2026:
        try:
            from src.rival_logic import evaluate_2026_rival
            
            # Get focus driver's current tyre age
            driver_laps = df[(df['Driver'] == driver) & (df['LapNumber'] <= at_lap)]
            undercutter_tyre_age = driver_laps['TyreLife'].iloc[-1] if len(driver_laps) > 0 else 10
            
            soc_list = []
            
            # Iterate through the rivals and apply the Nash Equilibrium matrix
            for index, row in responses.iterrows():
                rival = row['Rival']
                
                # Get rival's current tyre age
                rival_laps = df[(df['Driver'] == rival) & (df['LapNumber'] <= at_lap)]
                rival_tyre_age = rival_laps['TyreLife'].iloc[-1] if len(rival_laps) > 0 else 10
                
                # Proxy undercut threat based on legacy urgency for the API feed
                urgency = row.get('RivalUrgency', 'LOW')
                threat_seconds = 2.5 if urgency == 'HIGH' else 1.0
                
                # Run the Bayesian logic
                nash_result = evaluate_2026_rival(
                    driver=driver,
                    rival=rival,
                    undercut_threat_seconds=threat_seconds,
                    rival_tyre_age=rival_tyre_age,
                    undercutter_tyre_age=undercutter_tyre_age,
                    current_lap=at_lap,
                    track_name=race_key
                )
                
                # Overwrite the legacy response with 2026 intelligence
                responses.at[index, 'CanCover'] = nash_result['Can_Cover']
                responses.at[index, 'ExpectedResponse'] = nash_result['Expected_Response']
                soc_list.append(nash_result['Estimated_SoC'])
                
            # Append the new Battery Data to the payload
            responses['Estimated_SoC'] = soc_list
            
        except ImportError:
            print("Warning: rival_logic.py not found. Falling back to legacy strategy.")
            responses['Estimated_SoC'] = "N/A"
    else:
        # If in 2025 legacy mode, fill SoC with N/A
        responses['Estimated_SoC'] = "N/A"
    
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
    """Generate pit window chart with uncertainty bands."""
    data = request.json
    race_key = data.get('race')
    driver = data.get('driver')
    at_lap = int(data.get('lap', 15))
    
    if race_key not in RACES:
        return jsonify({'error': 'Race not found'}), 404
    
    # 60-FPS CACHE INTEGRATION
    df = load_race_data(race_key)
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
    
    # Get model and uncertainty predictions using the dynamic Model Router
    # Smart model routing — track → compound → generic
    best_model, _, model_tier = get_best_model(race_key, stint_df)
    
    n_future = 25 
    
    # Uncertainty-aware predictions
    uncertainty = predict_with_uncertainty(
        best_model, stint_df, n_future=n_future, n_samples=30
    )
    cliff = predict_cliff_with_uncertainty(
        best_model, stint_df, n_samples=30
    )
    
    if uncertainty is None:
        return jsonify({'error': 'Prediction failed'}), 500
    
    current_lap_num = int(stint_df['TyreLife'].max())
    current_avg = stint_df['LapTime'].mean()
    future_laps = uncertainty['laps']
    mean_preds = uncertainty['mean']
    p25 = uncertainty['p25']
    p75 = uncertainty['p75']
    p10 = uncertainty['p10']
    p90 = uncertainty['p90']
    
    # Calculate pit deltas using mean predictions
    pit_laps = list(range(current_lap_num + 1, current_lap_num + n_future))
    deltas = []
    for pit_lap in pit_laps:
        laps_before = pit_lap - current_lap_num
        laps_after = n_future - laps_before
        if laps_after <= 0:
            break
        degradation_cost = sum(
            mean_preds[i] - current_avg
            for i in range(laps_before, min(n_future, laps_before + laps_after))
        )
        delta = round(22.0 - degradation_cost, 3)
        deltas.append(delta)
    
    pit_laps = pit_laps[:len(deltas)]
    optimal_pit = pit_laps[deltas.index(min(deltas))] if deltas else None
    
    # Plot
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7),
                                    facecolor='#1E1E1E')
    
    for ax in [ax1, ax2]:
        ax.set_facecolor('#1E1E1E')
        ax.tick_params(colors='#F0EDE4')
        ax.xaxis.label.set_color('#F0EDE4')
        ax.yaxis.label.set_color('#F0EDE4')
        ax.title.set_color('#C9A84C')
        for spine in ax.spines.values():
            spine.set_edgecolor('#3a3a3a')
    
    # Top — actual + prediction with uncertainty bands
    actual_laps = stint_df['TyreLife'].tolist()
    ax1.plot(actual_laps, stint_df['LapTime'],
             color='#F0EDE4', linewidth=2, label='Actual', zorder=5)
    ax1.plot(future_laps, mean_preds,
             color='#6B7C3F', linewidth=2,
             linestyle='--', label='Predicted mean', zorder=5)
    ax1.fill_between(future_laps, p10, p90,
                    alpha=0.2, color='#6B7C3F', label='80% confidence')
    ax1.fill_between(future_laps, p25, p75,
                    alpha=0.4, color='#6B7C3F', label='50% confidence')
    
    if cliff:
        ax1.axvline(x=cliff['mean'], color='#8B2020', linewidth=1.5,
                    linestyle=':', label=f"Cliff: lap {cliff['mean']}")
        if cliff['p25'] != cliff['p75']:
            ax1.axvspan(cliff['p25'], cliff['p75'],
                       alpha=0.15, color='#8B2020')
    if optimal_pit:
        ax1.axvline(x=optimal_pit, color='#C9A84C', linewidth=1.5,
                    linestyle=':', label=f'Optimal pit: lap {optimal_pit}')
    
    ax1.set_ylabel('Lap Time (s)')
    ax1.set_title(f'{driver} — Degradation Forecast with Uncertainty')
    ax1.legend(fontsize=7, facecolor='#2a2a2a', labelcolor='#F0EDE4')
    ax1.grid(True, alpha=0.15)
    
    # Bottom — pit delta bars
    colors = ['#6B7C3F' if d < 0 else '#8B2020' for d in deltas]
    ax2.bar(pit_laps, deltas, color=colors, alpha=0.85)
    ax2.axhline(y=0, color='#F0EDE4', linewidth=0.8)
    if optimal_pit:
        ax2.axvline(x=optimal_pit, color='#C9A84C', linewidth=1.5,
                    linestyle=':')
    ax2.set_xlabel('Pit Lap')
    ax2.set_ylabel('Net Delta (s)')
    ax2.set_title('Pit Window — Green = Gain')
    ax2.grid(True, alpha=0.15)
    
    plt.tight_layout()
    
    buf = io.BytesIO()
    plt.savefig(buf, format='png', facecolor='#1E1E1E', bbox_inches='tight')
    buf.seek(0)
    img_base64 = base64.b64encode(buf.read()).decode('utf-8')
    plt.close()
    
    return jsonify({
        'chart': img_base64,
        'optimal_pit': optimal_pit,
        'cliff_lap': cliff['mean'] if cliff else None,
        'cliff_std': cliff['std'] if cliff else None,
        'model_tier': model_tier
    }) 

@app.route('/api/strategy_map', methods=['POST'])
def strategy_map():
    """Generate strategy map for all drivers and return as base64 image."""
    data = request.json
    race_key = data.get('race')
    
    if race_key not in RACES:
        return jsonify({'error': 'Race not found'}), 404
    
    # 60-FPS CACHE INTEGRATION
    df = load_race_data(race_key)
    
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
    
    # 2026 REGULATION FLAG
    is_2026 = data.get('is_2026', False)

    if race_key not in RACES:
        return jsonify({'error': 'Race not found'}), 404

    # 60-FPS CACHE INTEGRATION
    df = load_race_data(race_key)
    
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
        
        # Pass the 2026 flag down into the physics simulator!
        undercut = simulate_undercut(
            model, driver_stint, rival_stint,
            at_lap, gap_seconds, track,
            is_2026=is_2026
        )
        overcut = simulate_overcut(
            model, driver_stint, rival_stint,
            at_lap, gap_seconds, track,
            is_2026=is_2026
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

@app.route('/api/driver_profile/<driver>')
def driver_profile(driver):
    """Return driver tyre management profile."""
    import json
    try:
        with open('results/driver_profiles.json', 'r') as f:
            profiles = json.load(f)
        
        if driver not in profiles:
            return jsonify({'error': 'Driver not found'}), 404
        
        profile = profiles[driver]
        
        # Get field averages for comparison
        all_rates = [p['overall_rate'] for p in profiles.values() 
                    if p['overall_rate'] is not None]
        field_avg = round(float(np.median(all_rates)), 4)
        
        return jsonify({
            'driver': driver,
            'overall_rate': profile['overall_rate'],
            'overall_style': profile['overall_style'],
            'compounds': profile['compounds'],
            'field_avg': field_avg,
            'percentile': round(
                sum(1 for r in all_rates 
                    if r <= profile['overall_rate']) / len(all_rates) * 100
            )
        })
    except FileNotFoundError:
        return jsonify({'error': 'Profiles not built yet'}), 404 

# --- TASK 1: WEBSOCKET TIMING BEAM (LIVE PUSH) ---
active_sessions = {}

@socketio.on('connect')
def handle_connect():
    print("Pit wall client connected to live telemetry stream.")

@socketio.on('disconnect')
def handle_disconnect():
    # Use getattr to appease Pylance's strict type checking
    sid = getattr(request, 'sid', None)
    if sid and sid in active_sessions:
        active_sessions[sid] = False

@socketio.on('toggle_live_timing')
def toggle_live_timing(data):
    sid = getattr(request, 'sid', None)
    is_playing = data.get('is_playing')
    
    if sid:
        if is_playing:
            active_sessions[sid] = True
            # Spawn the background thread for this specific user
            socketio.start_background_task(timing_beam_loop, sid, data)
        else:
            active_sessions[sid] = False

def timing_beam_loop(sid, data):
    """Simulates the hardware timing beam triggering at the start/finish line."""
    # BULLETPROOFING: If JS sends NaN/None, default to Lap 1 and Max 100
    try:
        lap = int(data.get('lap') or 1)
        max_laps = int(data.get('max_laps') or 100)
    except (ValueError, TypeError):
        lap = 1
        max_laps = 100
    
    while active_sessions.get(sid) and lap < max_laps:
        socketio.sleep(3) 
        
        if not active_sessions.get(sid):
            break
            
        lap += 1
        socketio.emit('timing_beam_trigger', {'new_lap': lap}, to=sid) 

@app.route('/api/calendar')
def calendar():
    """Return upcoming F1 races."""
    from src.live_race import get_calendar_for_api
    try:
        races = get_calendar_for_api(n=3)
        return jsonify({'races': races})
    except Exception as e:
        return jsonify({'races': [], 'error': str(e)}) 

@app.route('/api/stint_timeline', methods=['POST'])
def stint_timeline():
    """Generate detailed stint timeline for focus driver."""
    data = request.json
    race_key = data.get('race')
    driver = data.get('driver')

    if race_key not in RACES:
        return jsonify({'error': 'Race not found'}), 404

    df = load_race_data(race_key)
    driver_df = df[df['Driver'] == driver]

    if len(driver_df) == 0:
        return jsonify({'error': 'Driver not found'}), 404

    stints = []
    for stint_num in sorted(driver_df['Stint'].unique()):
        stint_df = driver_df[driver_df['Stint'] == stint_num].reset_index(drop=True)
        if len(stint_df) == 0:
            continue

        compound = stint_df['Compound'].iloc[0]
        lap_times = stint_df['LapTime'].tolist()
        tyre_lives = stint_df['TyreLife'].tolist()
        avg_time = round(float(stint_df['LapTime'].mean()), 3)
        min_time = round(float(stint_df['LapTime'].min()), 3)
        max_time = round(float(stint_df['LapTime'].max()), 3)
        deg_rate = round(float(
            (stint_df['LapTime'].iloc[-1] - stint_df['LapTime'].iloc[0]) /
            max(1, len(stint_df))
        ), 4)

        stints.append({
            'stint': int(stint_num),
            'compound': compound,
            'laps': len(stint_df),
            'tyre_life_start': int(tyre_lives[0]),
            'tyre_life_end': int(tyre_lives[-1]),
            'avg_lap_time': avg_time,
            'min_lap_time': min_time,
            'max_lap_time': max_time,
            'degradation_rate': deg_rate,
            'lap_times': [round(t, 3) for t in lap_times],
            'tyre_lives': [int(t) for t in tyre_lives],
        })

    return jsonify({'driver': driver, 'stints': stints}) 

@app.route('/api/strategic_summary', methods=['POST'])
def strategic_summary():
    """Generate a plain-English strategic situation summary."""
    data = request.json
    race_key = data.get('race')
    driver = data.get('driver')
    at_lap = int(data.get('lap', 15))
    pit_lap = int(data.get('pit_lap', 18))

    if race_key not in RACES:
        return jsonify({'error': 'Race not found'}), 404

    df = load_race_data(race_key)
    driver_df = df[df['Driver'] == driver]
    current_laps = driver_df[driver_df['LapNumber'] <= at_lap]

    if len(current_laps) == 0:
        return jsonify({'error': 'No data'}), 404

    current_stint = current_laps['Stint'].iloc[-1]
    stint_df = current_laps[
        current_laps['Stint'] == current_stint
    ].reset_index(drop=True)

    if len(stint_df) < 5:
        return jsonify({'summary': 'Insufficient data for analysis.'}), 200

    # Get all the data points
    compound = stint_df['Compound'].iloc[-1]
    tyre_age = int(stint_df['TyreLife'].iloc[-1])
    avg_lap = round(float(stint_df['LapTime'].mean()), 3)

    # Cliff prediction
    best_model = None
    try:
        best_model, _, model_tier = get_best_model(race_key, stint_df)
        cliff_lap, cliff_low, cliff_high, _ = detect_cliff_with_confidence(
            best_model, stint_df
        )
    except Exception:
        cliff_lap = None
        cliff_low = None
        cliff_high = None
        model_tier = "Generic"

    # Driver profile
    import json as _json
    try:
        with open('results/driver_profiles.json') as f:
            profiles = _json.load(f)
        driver_style = profiles.get(driver, {}).get('overall_style', 'AVERAGE')
        driver_rate = profiles.get(driver, {}).get('overall_rate', 0.0)
    except Exception:
        driver_style = 'AVERAGE'
        driver_rate = 0.0

    # Rival count
    analysis = analyze_all_drivers(best_model, df, at_lap)
    high_urgency = len(analysis[analysis['Urgency'] == 'HIGH'])
    cannot_cover = len(analysis[analysis['LapsToCliff'].isna()])

    # Build summary
    laps_to_cliff = (cliff_lap - tyre_age) if cliff_lap else None

    urgency_word = "CRITICAL" if laps_to_cliff and laps_to_cliff <= 2 else \
                   "HIGH" if laps_to_cliff and laps_to_cliff <= 5 else \
                   "MODERATE" if laps_to_cliff and laps_to_cliff <= 10 else "LOW"

    cliff_str = f"lap {cliff_lap} (±{cliff_high - cliff_low if cliff_high and cliff_low else '?'} laps)" \
                if cliff_lap else "beyond prediction window"

    style_impact = "will accelerate degradation" if driver_style in ['HARD', 'VERY HARD'] else \
                   "should extend tyre life" if driver_style in ['GENTLE', 'VERY GENTLE'] else \
                   "is average for this compound"

    summary = (
        f"SITUATION [{urgency_word}] — Lap {at_lap} | {driver} | {compound} Age {tyre_age}\n\n"
        f"Performance cliff predicted at {cliff_str}. "
        f"At {avg_lap:.1f}s average, this {compound} is "
        f"{'degrading faster than expected' if driver_rate > 0.01 else 'holding well'}. "
        f"{driver}'s driving style ({driver_style}) {style_impact}.\n\n"
        f"FIELD: {high_urgency} rivals at HIGH urgency. "
        f"Strategic window: pit lap {pit_lap} "
        f"{'is within the optimal window' if cliff_lap and pit_lap <= cliff_lap else 'may be too late'}.\n\n"
        f"MODEL: {model_tier}"
    )

    return jsonify({'summary': summary, 'urgency': urgency_word})

@app.route('/api/gap_tracker', methods=['POST'])
def gap_tracker():
    """Track gaps to cars ahead and behind with degradation-adjusted projections."""
    data = request.json
    race_key = data.get('race')
    driver = data.get('driver')
    at_lap = int(data.get('lap', 15))

    if race_key not in RACES:
        return jsonify({'error': 'Race not found'}), 404

    df = load_race_data(race_key)

    # Get all drivers average lap times at this point
    driver_times = {}
    for d in df['Driver'].unique():
        d_laps = df[(df['Driver'] == d) & (df['LapNumber'] <= at_lap)]
        if len(d_laps) >= 3:
            driver_times[d] = float(d_laps['LapTime'].tail(5).mean())

    if driver not in driver_times:
        return jsonify({'error': 'Driver not found'}), 404

    # Sort by lap time (proxy for position — faster = further ahead)
    sorted_drivers = sorted(driver_times.items(), key=lambda x: x[1])
    driver_pos = next((i for i, (d, _) in enumerate(sorted_drivers) if d == driver), None)

    if driver_pos is None:
        return jsonify({'error': 'Position not found'}), 404

    our_pace = driver_times[driver]

    # Car ahead and behind
    ahead = sorted_drivers[driver_pos - 1] if driver_pos > 0 else None
    behind = sorted_drivers[driver_pos + 1] if driver_pos < len(sorted_drivers) - 1 else None

    # Project gap evolution over next 10 laps using degradation rates
    def project_gap(rival_driver, rival_pace, n_laps=10):
        our_df = df[(df['Driver'] == driver) & (df['LapNumber'] <= at_lap)]
        rival_df = df[(df['Driver'] == rival_driver) & (df['LapNumber'] <= at_lap)]

        if len(our_df) < 5 or len(rival_df) < 5:
            return []

        # Simple degradation rate from last 5 laps
        our_rate = float(our_df['LapTime'].diff().tail(5).mean())
        rival_rate = float(rival_df['LapTime'].diff().tail(5).mean())

        gap = round(abs(rival_pace - our_pace), 3)
        evolution = [gap]

        for lap in range(1, n_laps + 1):
            our_time = our_pace + our_rate * lap
            rival_time = rival_pace + rival_rate * lap
            new_gap = round(rival_time - our_time, 3)
            evolution.append(new_gap)

        return evolution

    result = {
        'driver': driver,
        'position': driver_pos + 1,
        'our_pace': round(our_pace, 3),
        'ahead': None,
        'behind': None,
    }

    if ahead:
        result['ahead'] = {
            'driver': ahead[0],
            'current_gap': round(ahead[1] - our_pace, 3),
            'gap_evolution': project_gap(ahead[0], ahead[1]),
            'closing': (ahead[1] - our_pace) > 0
        }

    if behind:
        result['behind'] = {
            'driver': behind[0],
            'current_gap': round(our_pace - behind[1], 3),
            'gap_evolution': project_gap(behind[0], behind[1]),
            'under_threat': (our_pace - behind[1]) < 0.3
        }

    return jsonify(result) 

@app.route('/api/recommendation', methods=['POST'])
def strategy_recommendation():
    """
    Concrete strategic recommendation — the single most valuable output.
    PIT NOW / PIT IN N LAPS / STAY OUT / COVER / DEFEND
    """
    data = request.json
    race_key = data.get('race')
    driver = data.get('driver')
    at_lap = int(data.get('lap', 15))
    pit_lap = int(data.get('pit_lap', 18))

    if race_key not in RACES:
        return jsonify({'error': 'Race not found'}), 404

    df = load_race_data(race_key)
    driver_df = df[df['Driver'] == driver]
    current_laps = driver_df[driver_df['LapNumber'] <= at_lap]

    if len(current_laps) == 0:
        return jsonify({'error': 'No data'}), 404

    current_stint = current_laps['Stint'].iloc[-1]
    stint_df = current_laps[
        current_laps['Stint'] == current_stint
    ].reset_index(drop=True)

    if len(stint_df) < 5:
        return jsonify({'recommendation': 'INSUFFICIENT DATA',
                       'confidence': 0, 'reasoning': []}), 200

    compound = stint_df['Compound'].iloc[-1]
    tyre_age = int(stint_df['TyreLife'].iloc[-1])

    # Get cliff prediction
    try:
        best_model, _, model_tier = get_best_model(race_key, stint_df)
        cliff_lap, cliff_low, cliff_high, _ = detect_cliff_with_confidence(
            best_model, stint_df
        )
    except Exception:
        cliff_lap = None
        cliff_low = None
        cliff_high = None

    # Get rival analysis
    try:
        analysis = analyze_all_drivers(best_model, df, at_lap) # type: ignore
        high_urgency_rivals = analysis[
            (analysis['Urgency'] == 'HIGH') &
            (analysis['Driver'] != driver)
        ]
        rivals_more_urgent = len(high_urgency_rivals)
    except Exception:
        rivals_more_urgent = 0

    # Decision logic
    laps_to_cliff = (cliff_lap - tyre_age) if cliff_lap else 99
    reasoning = []
    confidence = 0

    if laps_to_cliff <= 1:
        recommendation = "🔴 PIT NOW"
        confidence = 95
        reasoning.append(f"Performance cliff imminent — tyre age {tyre_age}, cliff at lap {cliff_lap}")
        reasoning.append(f"Every lap you stay out risks a catastrophic lap time loss")

    elif laps_to_cliff <= 3:
        recommendation = f"🟡 PIT IN {laps_to_cliff} LAPS"
        confidence = 85
        reasoning.append(f"Cliff predicted at lap {cliff_lap} (±{(cliff_high or 0) - (cliff_low or 0)} laps)")
        reasoning.append(f"Window is closing — prepare pit crew now")
        if rivals_more_urgent > 2:
            reasoning.append(f"{rivals_more_urgent} rivals also approaching cliff — pit stop traffic risk")

    elif rivals_more_urgent >= 3 and laps_to_cliff > 5:
        recommendation = "🟢 STAY OUT — RIVALS MORE URGENT"
        confidence = 75
        reasoning.append(f"Your cliff is {laps_to_cliff} laps away")
        reasoning.append(f"{rivals_more_urgent} rivals at HIGH urgency — let them pit first")
        reasoning.append("Track position gain possible if you extend")

    elif laps_to_cliff <= 6:
        recommendation = f"🟠 PREPARE PIT — LAP {cliff_lap}"
        confidence = 70
        reasoning.append(f"Approaching cliff window — {laps_to_cliff} laps remaining")
        reasoning.append(f"Optimal pit window opens now")
        if cliff_high and cliff_low:
            reasoning.append(f"Cliff confidence range: lap {cliff_low}–{cliff_high}")

    else:
        recommendation = "🟢 STAY OUT"
        confidence = 80
        reasoning.append(f"Tyre has {laps_to_cliff} laps before cliff")
        reasoning.append(f"{compound} compound performing within expected range")
        if rivals_more_urgent > 0:
            reasoning.append(f"Monitor {rivals_more_urgent} rivals approaching their windows")

    return jsonify({
        'recommendation': recommendation,
        'confidence': confidence,
        'reasoning': reasoning,
        'cliff_lap': cliff_lap,
        'laps_to_cliff': laps_to_cliff if laps_to_cliff < 99 else None,
        'rivals_urgent': rivals_more_urgent,
        'model_tier': model_tier if 'model_tier' in dir() else 'Generic' # type: ignore
    })

@app.route('/api/compound_comparison', methods=['POST'])
def compound_comparison():
    """
    Compare predicted degradation curves for all three compounds
    at the current track. Answers: what if we switch compound now?
    """
    data = request.json
    race_key = data.get('race')
    driver = data.get('driver')
    at_lap = int(data.get('lap', 15))

    if race_key not in RACES:
        return jsonify({'error': 'Race not found'}), 404

    df = load_race_data(race_key)
    driver_df = df[df['Driver'] == driver]
    current_laps = driver_df[driver_df['LapNumber'] <= at_lap]

    if len(current_laps) == 0:
        return jsonify({'error': 'No data'}), 404

    current_stint = current_laps['Stint'].iloc[-1]
    stint_df = current_laps[
        current_laps['Stint'] == current_stint
    ].reset_index(drop=True)

    if len(stint_df) < 5:
        return jsonify({'error': 'Not enough data'}), 400

    n_future = 25
    results = {}

    for compound in ['SOFT', 'MEDIUM', 'HARD']:
        try:
            comp_model = compound_models.get(compound, main_model)
            # Build a synthetic stint starting fresh on this compound
            synthetic = stint_df.copy()
            synthetic['Compound'] = compound
            synthetic['TyreLife'] = range(1, len(synthetic) + 1)

            from src.cliff_detector import prepare_sequence, predict_future_laps
            from src.dataset import denormalize, LAP_TIME_MIN, LAP_TIME_MAX
            from src.thermal_model import calculate_thermal_energy

            seq = prepare_sequence(comp_model, synthetic)
            preds = predict_future_laps(comp_model, seq, n_future)
            pred_seconds = [
                round(denormalize(p, LAP_TIME_MIN, LAP_TIME_MAX), 3)
                for p in preds
            ]

            # Find cliff for this compound
            avg = float(synthetic['LapTime'].mean())
            from src.cliff_detector import get_cliff_threshold
            threshold = get_cliff_threshold(compound)
            cliff = None
            for i, t in enumerate(pred_seconds):
                if t > avg + threshold:
                    cliff = i + 1
                    break

            results[compound] = {
                'predictions': pred_seconds,
                'cliff_lap': cliff,
                'avg_fresh_pace': round(avg, 3)
            }
        except Exception as e:
            results[compound] = {'error': str(e)}

    current_lap_num = int(stint_df['TyreLife'].max())
    future_laps = list(range(current_lap_num + 1,
                             current_lap_num + n_future + 1))

    return jsonify({
        'future_laps': future_laps,
        'compounds': results,
        'current_compound': stint_df['Compound'].iloc[-1],
        'current_tyre_age': current_lap_num
    }) 

@app.route('/api/compound_comparison_chart', methods=['POST'])
def compound_comparison_chart():
    """Generate compound comparison chart as base64 image."""
    data = request.json
    race_key = data.get('race')
    driver = data.get('driver')
    at_lap = int(data.get('lap', 15))

    # Get comparison data
    from flask import current_app
    with current_app.test_request_context():
        pass

    df = load_race_data(race_key)
    driver_df = df[df['Driver'] == driver]
    current_laps = driver_df[driver_df['LapNumber'] <= at_lap]

    if len(current_laps) == 0:
        return jsonify({'error': 'No data'}), 404

    current_stint = current_laps['Stint'].iloc[-1]
    stint_df = current_laps[
        current_laps['Stint'] == current_stint
    ].reset_index(drop=True)

    if len(stint_df) < 5:
        return jsonify({'error': 'Not enough data'}), 400

    n_future = 25
    current_lap_num = int(stint_df['TyreLife'].max())
    future_laps = list(range(current_lap_num + 1,
                             current_lap_num + n_future + 1))

    compound_colors = {
        'SOFT': '#8B2020',
        'MEDIUM': '#C9A84C',
        'HARD': '#555555'
    }

    fig, ax = plt.subplots(figsize=(10, 5), facecolor='#1E1E1E')
    ax.set_facecolor('#1E1E1E')
    ax.tick_params(colors='#F0EDE4')
    ax.xaxis.label.set_color('#F0EDE4')
    ax.yaxis.label.set_color('#F0EDE4')
    ax.title.set_color('#C9A84C')
    for spine in ax.spines.values():
        spine.set_edgecolor('#3a3a3a')

    # Plot actual laps
    actual_laps = stint_df['TyreLife'].tolist()
    ax.plot(actual_laps, stint_df['LapTime'],
            color='#F0EDE4', linewidth=2, label='Actual', zorder=5)

    summary_data = {}

    for compound in ['SOFT', 'MEDIUM', 'HARD']:
        try:
            comp_model = compound_models.get(compound, main_model)
            synthetic = stint_df.copy()
            synthetic['Compound'] = compound
            synthetic['TyreLife'] = range(1, len(synthetic) + 1)

            from src.cliff_detector import (prepare_sequence,
                                            predict_future_laps,
                                            get_cliff_threshold)
            from src.dataset import denormalize, LAP_TIME_MIN, LAP_TIME_MAX

            seq = prepare_sequence(comp_model, synthetic)
            preds = predict_future_laps(comp_model, seq, n_future)
            pred_seconds = [denormalize(p, LAP_TIME_MIN, LAP_TIME_MAX)
                           for p in preds]

            avg = float(stint_df['LapTime'].mean())
            threshold = get_cliff_threshold(compound)
            cliff = None
            for i, t in enumerate(pred_seconds):
                if t > avg + threshold:
                    cliff = current_lap_num + i + 1
                    break

            color = compound_colors[compound]
            label = f"{compound}"
            if cliff:
                label += f" (cliff ~lap {cliff})"

            ax.plot(future_laps[:len(pred_seconds)], pred_seconds,
                   color=color, linewidth=2, linestyle='--',
                   label=label, alpha=0.9)

            if cliff:
                ax.axvline(x=cliff, color=color, linewidth=1,
                          linestyle=':', alpha=0.6)

            summary_data[compound] = {
                'cliff': cliff,
                'avg_pace': round(float(np.mean(pred_seconds[:10])), 3)
            }

        except Exception:
            pass

    ax.set_xlabel('Tyre Life (laps)')
    ax.set_ylabel('Lap Time (s)')
    ax.set_title(f'{driver} — Compound Switch Analysis from Lap {at_lap}')
    ax.legend(fontsize=8, facecolor='#2a2a2a', labelcolor='#F0EDE4')
    ax.grid(True, alpha=0.15)
    plt.tight_layout()

    buf = io.BytesIO()
    plt.savefig(buf, format='png', facecolor='#1E1E1E', bbox_inches='tight')
    buf.seek(0)
    img_base64 = base64.b64encode(buf.read()).decode('utf-8')
    plt.close()

    return jsonify({'chart': img_base64, 'summary': summary_data}) 

if __name__ == '__main__':
    # We now run socketio.run instead of app.run to enable WebSockets
    socketio.run(app, debug=True, use_reloader=False, port=5000)  