# Introduction

Formula 1 race strategy is decided in seconds. When a tyre begins its 
performance cliff — the sudden, non-linear degradation that can cost five 
seconds per lap — the window to respond is narrow and the cost of 
misjudgement is irreversible. Strategy engineers on the pit wall must 
predict this cliff before it happens, not after, using only the lap times 
and telemetry available to them in real time.

This is, at its core, a sequence prediction problem under physical 
constraints. Tyres degrade according to thermodynamic laws: cumulative 
energy dissipation is monotonically non-decreasing, rubber temperature 
accelerates molecular breakdown, and harder compounds resist degradation 
longer than softer ones. These constraints are not hypotheses — they are 
physical facts. Yet standard machine learning approaches applied to 
lap time prediction ignore them entirely, producing models that 
occasionally predict physically impossible outcomes: tyres that 
appear to recover performance mid-stint, or degradation rates that 
invert with temperature. Such predictions cannot be trusted by an 
engineer making a decision worth millions of points.

We present TyrePIML, a physics-informed machine learning framework 
for F1 tyre degradation prediction. Our approach makes three 
contributions:

**Physics-informed training.** We introduce a custom loss function 
that penalises predictions violating thermodynamic constraints, 
scaled by circuit abrasiveness and track temperature. This produces 
models whose predictions are not only accurate but physically credible.

**Thermal energy feature engineering.** We model cumulative tyre 
thermal energy dissipation from first principles — lateral load, 
longitudinal braking force, and temperature-dependent rubber 
degradation — and encode this as a learned input feature. This 
bridges the gap between data-driven and physics-based approaches.

**Transformer architecture for lap sequences.** We replace the 
standard LSTM backbone with a Transformer encoder, demonstrating 
a 12.5% improvement in mean absolute error on strictly held-out 
2023-2024 races compared to the LSTM baseline.

TyrePIML is trained on 118 races across seven seasons (2018-2024), 
comprising 106,982 clean laps of official F1 telemetry. We release 
all code, trained models, and benchmarks as open-source software, 
establishing the first public baseline for physics-informed F1 
tyre degradation modelling.

Our validation on the 2023 Italian Grand Prix demonstrates the 
practical value of this approach: TyrePIML predicted Max Verstappen's 
pit stop at lap 21, one lap from the actual stop at lap 20, using 
only publicly available telemetry and no access to team-internal data. 