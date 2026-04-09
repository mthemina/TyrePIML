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

if __name__ == '__main__':
    app.run(debug=True, port=5000) 