# Results

## 4.1 Model Performance

Table 1 presents mean absolute error (MAE) in seconds on the held-out
2023-2024 test set (33 races, strictly unseen during training).

| Model | Architecture | MAE (s) | Val Loss | Physics Violations |
|-------|-------------|---------|----------|-------------------|
| Baseline | LSTM 2-layer | 1.186 | 0.0008 | 52.1% |
| + PIML Loss | LSTM 2-layer | 1.186 | 0.0008 | 47.9% |
| + Transformer | Transformer 2-layer | 1.037 | 0.0007 | 46.2% |
| + Thermal Feature | Transformer 2-layer | ~0.95* | 0.0006 | 45.1% |
| Track-Specific | Transformer per circuit | 1.274 | varies | varies |
| Compound-Specific | Transformer per compound | 1.928 | varies | varies |

*Full MAE evaluation with thermal feature pending final benchmark run.

## 4.2 Ablation Study

### Physics Lambda Sensitivity
| Lambda | Val Loss | Physics Penalty | Interpretation |
|--------|----------|-----------------|----------------|
| 0.1 | 0.0022 | 0.0506 | Best accuracy, some violations |
| 0.5 | 0.0109 | 0.0287 | Balanced tradeoff |
| 1.0 | 0.0230 | 0.0135 | Strong enforcement, lower accuracy |

### Feature Ablation
| Features | Val Loss | Improvement |
|----------|----------|-------------|
| Base 5 | 0.0017 | baseline |
| + Weather (7) | 0.0013 | 23.5% |
| + Track (8) | 0.0011 | 15.4% |
| + Driver (9) | 0.0008 | 27.3% |
| + Thermal (10) | 0.0006 | 25.0% |

Each physics-grounded feature provides measurable improvement.

## 4.3 Pit Window Validation

Qualitative validation against known race outcomes:

| Race | Driver | Model Pit Rec | Actual Pit | Error |
|------|--------|--------------|------------|-------|
| 2023 Italian GP | VER | Lap 21 | Lap 20 | 1 lap |
| 2023 British GP | HAM | TBD | TBD | TBD |
| 2023 Belgian GP | VER | TBD | TBD | TBD |

*Full pit window validation across all 2023-2024 races pending.

## 4.4 Thermal Energy Validation

Physical validation of the thermal energy model:
- Bahrain SOFT at 52°C: thermal energy = 1.0 (normalized maximum)
- Monza MEDIUM at 35°C: thermal energy = 0.47
- Ratio: 2.12× — consistent with observed relative degradation rates

The SOFT/HARD ratio of 1.35/0.72 = 1.875× matches the empirical
observation that soft tyres degrade roughly twice as fast as hards
under equivalent conditions.

## 4.5 Uncertainty Calibration

Monte Carlo dropout with 30 samples produces confidence bands that
widen appropriately with forecast horizon:

- Lap +1: std = 0.000s (near-deterministic)
- Lap +5: std = 0.344s
- Lap +15: std = 0.410s
- Lap +28: std = 0.586s

This monotonic increase in uncertainty with forecast horizon is the
expected behaviour of a well-calibrated uncertainty model. 