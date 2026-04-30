# TyrePIML: Physics-Informed Machine Learning for F1 Tyre Degradation Prediction

## Target Venue
ML for Physical Sciences Workshop — NeurIPS 2026

---

## Abstract
- TyrePIML applies physics-informed ML to F1 tyre degradation prediction
- Key contribution: custom loss function enforcing thermodynamic constraints
- Transformer architecture outperforms LSTM by 12.5% on held-out races
- Thermal energy feature reduces val loss from 0.0007 to 0.0006
- Open-source tool with live race dashboard

---

## 1. Introduction
- F1 strategy decisions depend on accurate tyre degradation prediction
- Current approaches: empirical models, no uncertainty quantification
- Problem: standard neural networks violate physical laws (tyres un-wearing)
- Our approach: embed physics constraints directly into training objective
- Contributions: PIML loss, Transformer backbone, thermal energy feature, open benchmark

---

## 2. Related Work
- Physics-informed neural networks (PINNs) — Raissi et al. 2019
- Tyre modelling in motorsport — Pacejka magic formula
- Sequence modelling for time series — LSTM, Transformer
- Uncertainty quantification in ML — Monte Carlo dropout
- Gap: no published PIML approach for F1 tyre strategy

---

## 3. Method
### 3.1 Data Pipeline
- FastF1 telemetry — 118 races, 2018-2024, 106,982 laps
- 10 input features: tyre life, compound, sector times, track/air temp, abrasiveness, driver style, thermal energy
- Weather integration, track profiling, driver degradation encoding

### 3.2 Physics-Informed Loss Function
- Monotonic degradation constraint — tyres cannot un-wear
- Track abrasiveness scaling — penalty calibrated per circuit
- Thermal constraint — hot conditions accelerate degradation
- Lambda tuning study — accuracy vs constraint enforcement tradeoff

### 3.3 Thermal Energy Model
- Three dissipation mechanisms: hysteretic deformation, sliding friction, viscoelastic damping
- Compound sensitivity: SOFT 1.35x, MEDIUM 1.0x, HARD 0.72x
- Track thermal load factors from abrasiveness profiles
- Bahrain SOFT 2.12x more stressed than Monza MEDIUM — physically validated

### 3.4 Model Architecture
- Transformer encoder with positional encoding
- Multi-head self-attention captures lap interdependencies
- MC dropout for calibrated uncertainty quantification
- Model hierarchy: generic → compound-specific → track-specific

---

## 4. Experiments
### 4.1 Evaluation Protocol
- Train: 2018-2022 (85 races)
- Test: 2023-2024 (33 races) — strictly held out
- Metrics: MAE in seconds, physics violation rate, cliff prediction accuracy

### 4.2 Ablation Study
- Baseline LSTM vs PIML LSTM — violation rate 52% → 48%
- LSTM vs Transformer — 12.5% MAE improvement
- Transformer vs Transformer+Thermal — val loss 0.0007 → 0.0006
- Lambda sensitivity — accuracy vs constraint tradeoff

### 4.3 Pit Window Validation
- Verstappen Monza 2023 — model recommends lap 21, actual lap 20
- Validation across all 2023-2024 races

---

## 5. Results
- Table 1: Model comparison (LSTM / Transformer / Compound / Track)
- Table 2: Physics violation rates across model tiers
- Figure 1: Uncertainty bands — degradation forecast with 50%/80% confidence
- Figure 2: Model progression — val loss reduction with physics features
- Figure 3: Pit window visualization — green/red delta bars

---

## 6. Conclusion
- Physics constraints produce trustworthy predictions engineers can act on
- Transformer architecture is the right backbone for lap sequence modelling
- Thermal energy as a learned physics feature improves generalization
- Open-source benchmark enables reproducible F1 strategy research
- Future work: live race integration, compound-specific physics, paper submission

---

## Timeline
- May-June 2026: deepen physics models, expand benchmark
- July 2026: live race integration, dashboard polish
- August 2026: deploy, write full paper draft
- September 19 2026: public launch + paper submission 