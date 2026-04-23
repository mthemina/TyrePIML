# TyrePIML 🏎️

**Physics-Informed ML for F1 Tyre Degradation Prediction and Race Strategy**

[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![GSoC 2026](https://img.shields.io/badge/GSoC-2026-orange.svg)](https://summerofcode.withgoogle.com/)
[![Version](https://img.shields.io/badge/version-0.2.0-green.svg)](https://github.com/mthemina/TyrePIML/releases)

TyrePIML is a physics-informed machine learning platform for F1 tyre degradation prediction and race strategy optimization. It combines a custom PIML loss function enforcing thermodynamic constraints with 7 seasons of real F1 telemetry data to produce trustworthy, physically consistent predictions that engineers can act on.

---

## What Makes TyrePIML Different

Standard neural networks trained on F1 telemetry can predict lap times but routinely violate physical laws — predicting tyres that un-wear themselves, or degradation curves that defy thermodynamics. TyrePIML addresses this by embedding physical constraints directly into the training objective, producing predictions that are both accurate and physically credible.

No public open-source tool currently does this for F1 strategy.

---

## Features

### Physics-Informed Modelling
- Custom PIML loss function enforcing monotonic tyre degradation
- Thermal constraint — hot track conditions accelerate predicted degradation
- Track abrasiveness scaling — physics penalty calibrated per circuit
- Monte Carlo dropout uncertainty quantification — every prediction includes confidence bands

### Data Scale
- 118 races across 7 seasons (2018–2024)
- 106,982 clean laps with weather integration
- 9 input features: tyre life, compound, sector times, track/air temperature, abrasiveness, driver style

### Model Architecture
- Generic PIML model trained on full dataset
- 3 compound-specific models (SOFT / MEDIUM / HARD)
- 35 track-specific models — one per circuit, trained on all historical races at that venue
- Smart model router — automatically selects best available model per race

### Strategy Intelligence
- Tyre cliff predictor with confidence intervals
- Pit window optimizer — optimal pit lap with net time delta
- Undercut / overcut simulator
- Multi-driver tyre state analysis with urgency ranking
- Rival response predictor — models strategic chess match
- Race position simulator
- Driver tyre management profiles — 40 drivers, historical degradation rates

### Live Race Dashboard
- WebSocket-powered live timing beam — real-time lap-by-lap updates
- 2026 Active Aero physics engine — battery SoC tracking, Nash Equilibrium rival strategy
- Broadcast-quality strategy maps
- Pit window charts with uncertainty bands

---

## Benchmark Results

| Metric | Baseline LSTM | PIML LSTM |
|--------|--------------|-----------|
| MAE (seconds) | 0.596 | 0.758 |
| RMSE (seconds) | 1.079 | 1.172 |
| Physics Violation Rate | 52.1% | 47.9% |

**Pit window validation — Verstappen, Monza 2023:**
Model recommended pit lap **21** — actual pit lap **20** (1 lap error on blind prediction)

---

## Quickstart

```bash
git clone https://github.com/mthemina/TyrePIML.git
cd TyrePIML
python -m venv venv
source venv/bin/activate
pip install fastf1 pandas numpy torch matplotlib jupyter wandb flask flask-socketio gevent gevent-websocket
PYTHONPATH=. python src/data_loader.py       # Download race data
PYTHONPATH=. python src/train_piml.py        # Train PIML model
PYTHONPATH=. python src/compound_models.py   # Train compound models
PYTHONPATH=. python src/track_models.py      # Train track models
PYTHONPATH=. python src/driver_profiles.py   # Build driver profiles
PYTHONPATH=. python app/app.py               # Launch dashboard
```

---

## Project Structure 

TyrePIML/
├── data/              # Race CSVs (118 races, 2018–2024)
├── models/
│   ├── tracks/        # 35 track-specific models
│   ├── tyre_lstm_piml_v2.pt
│   ├── tyre_lstm_soft_v1.pt
│   ├── tyre_lstm_medium_v1.pt
│   └── tyre_lstm_hard_v1.pt
├── notebooks/         # Benchmark notebook
├── results/           # Plots, evaluation results, driver profiles
├── app/
│   ├── app.py         # Flask + WebSocket backend
│   └── templates/     # Dashboard frontend
└── src/
├── data_loader.py       # FastF1 pipeline
├── dataset.py           # PyTorch dataset (9 features)
├── model.py             # LSTM with MC dropout
├── piml_loss.py         # Physics + thermal constraints
├── track_profiles.py    # 20 circuit profiles
├── compound_models.py   # Compound-specific training
├── track_models.py      # Track-specific training
├── driver_profiles.py   # Driver degradation analysis
├── uncertainty.py       # Monte Carlo uncertainty
├── cliff_detector.py    # Performance cliff prediction
├── strategy_simulator.py # Undercut/overcut simulation
└── race_strategy.py     # Multi-driver analysis

---

## Roadmap to September 2026

- [ ] Compound-specific physics constraints per circuit
- [ ] Live FastF1 telemetry integration during race weekends
- [ ] Neural architecture upgrade — Transformer-based degradation model
- [ ] Tyre thermal model based on energy dissipation physics
- [ ] Public deployment with live URL
- [ ] Benchmark paper submission — ML for Physical Sciences @ NeurIPS

---

## Author

**Mina Narman** — Computer Engineering, Acıbadem University, Istanbul
GitHub: [@mthemina](https://github.com/mthemina)
Project proposed under ML4SCI — GSoC 2026 
