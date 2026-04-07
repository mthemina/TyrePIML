# TyrePIML 🏎️

**Physics-Informed Machine Learning for F1 Tyre Degradation Prediction and Race Strategy**

[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![GSoC 2026](https://img.shields.io/badge/GSoC-2026-orange.svg)](https://summerofcode.withgoogle.com/)

TyrePIML applies physics-informed machine learning to the problem of F1 tyre 
degradation prediction. Unlike standard neural networks, the model enforces 
physical constraints — tyres cannot un-wear themselves — producing predictions 
that engineers can trust.

Built on real F1 telemetry data via the [FastF1](https://github.com/theOehrly/Fast-F1) library.

---

## Features

- **Tyre degradation prediction** — LSTM model trained on lap-by-lap telemetry
- **Physics constraints** — custom loss function enforcing monotonic tyre wear
- **Cliff detector** — predicts the exact lap a tyre will fall off the performance cliff
- **Pit window optimizer** — finds the optimal pit lap with net time delta calculations
- **Strategy visualization** — two-panel race strategy charts

---

## Results

| Metric | Baseline LSTM | PIML LSTM |
|--------|--------------|-----------|
| MAE (seconds) | 0.596 | 0.758 |
| RMSE (seconds) | 1.079 | 1.172 |
| Physics Violation Rate | 52.1% | 47.9% |

**Pit window validation — Verstappen, Monza 2023:**
- Model recommended pit lap: **21**
- Actual pit lap: **20**
- Off by 1 lap on a blind prediction using only public data

---

## Quickstart
```bash
git clone https://github.com/mthemina/TyrePIML.git
cd TyrePIML
python -m venv venv
source venv/bin/activate
pip install fastf1 pandas numpy torch matplotlib jupyter wandb
PYTHONPATH=. python src/data_loader.py
PYTHONPATH=. python src/train_piml.py
PYTHONPATH=. python src/plot_pit_window.py
```

---

## Project Structure 
TyrePIML/
├── data/          # Race CSVs (FastF1 telemetry)
├── models/        # Trained model weights
├── notebooks/     # Benchmark notebook
├── results/       # Plots and evaluation results
└── src/
├── data_loader.py      # FastF1 pipeline
├── dataset.py          # PyTorch dataset
├── model.py            # LSTM architecture
├── piml_loss.py        # Physics-informed loss
├── train.py            # Baseline training
├── train_piml.py       # PIML training
├── evaluate.py         # Evaluation metrics
├── cliff_detector.py   # Tyre cliff prediction
└── compare.py          # Model comparison 
---

## Roadmap

- [ ] Undercut/overcut simulator
- [ ] Multi-driver strategy comparison  
- [ ] Weather impact on degradation
- [ ] Live race mode
- [ ] Web dashboard

---

## Author

**Mina Narman** — Computer Engineering, Acıbadem University, Istanbul  
GitHub: [@mthemina](https://github.com/mthemina)  
Project proposed under ML4SCI — GSoC 2026 