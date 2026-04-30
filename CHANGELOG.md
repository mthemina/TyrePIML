# TyrePIML Changelog

## v0.3.0 (In Progress)
### Added
- Tyre thermal energy physics model — 3-mechanism energy dissipation
- Thermal energy as 10th model feature
- Transformer architecture — 12.5% improvement over LSTM
- 35 track-specific Transformer models with early stopping
- 3 compound-specific Transformer models (SOFT/MEDIUM/HARD)
- Driver style encoding — 40 drivers profiled
- Monte Carlo dropout uncertainty quantification
- 50%/80% confidence bands on all predictions
- WebSocket live timing beam
- 2026 Active Aero physics engine with battery SoC tracking
- Smart model router — track → compound → generic priority

### Changed
- Input features: 5 → 9 → 10 (weather, track, driver, thermal)
- Primary architecture: LSTM → Transformer
- Val loss: 0.0008 → 0.0007 → 0.0006

---

## v0.2.0 (April 2026)
### Added
- Full dataset: 118 races, 2018-2024, 106,982 laps
- Weather data integration (air temp, track temp, humidity)
- 20 circuit track profiles with abrasiveness ratings
- Driver degradation profiles for 40 drivers
- Compound-specific models (SOFT/MEDIUM/HARD)
- Track-specific models (35 circuits)
- Uncertainty quantification with MC dropout
- Live race dashboard with WebSocket streaming
- Strategy map, undercut/overcut simulator
- Driver profile card in dashboard

### Changed
- Dataset: 5 races → 118 races
- Training sequences: 3,328 → 81,274

---

## v0.1.0 (April 2026)
### Added
- Initial LSTM baseline model
- PIML loss function with monotonic degradation constraint
- FastF1 data pipeline (5 races)
- Benchmark notebook
- Pit window optimizer
- Cliff detector
- Flask dashboard (basic)
- Physics violation rate metric
- Baseline MAE: 0.596s, violation rate: 52.1% 