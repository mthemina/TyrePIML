# TyrePIML — Methods

## 3.1 Data Pipeline

We collect lap-by-lap telemetry from 118 Formula 1 races spanning seven 
seasons (2018–2024) using the FastF1 Python library, which provides access 
to the official F1 timing feed. After filtering safety car laps, formation 
laps, and laps with missing sector times, our dataset comprises 106,982 
clean laps across 35 circuits.

Each lap is represented by a 10-dimensional feature vector:

| Feature | Description |
|---------|-------------|
| Tyre Life | Normalized age of current tyre (laps) |
| Compound | SOFT=0, MEDIUM=0.5, HARD=1.0 |
| Sector 1-3 Times | Normalized sector times (seconds) |
| Track Temperature | Normalized ambient track temperature |
| Air Temperature | Normalized air temperature |
| Track Abrasiveness | Circuit abrasion rating (1-10, normalized) |
| Driver Style | Historical degradation rate encoding (0-1) |
| Thermal Energy | Cumulative tyre energy dissipation (0-1) |

Weather data is sourced from the FastF1 weather channel and joined to each 
lap by session timestamp.

## 3.2 Physics-Informed Loss Function

Standard supervised learning minimizes prediction error alone:

$$\mathcal{L}_{base} = \frac{1}{N}\sum_{i=1}^{N}(\hat{y}_i - y_i)^2$$

TyrePIML augments this with two physics constraints.

**Monotonic Degradation Constraint.** A tyre cannot un-wear itself — 
cumulative energy dissipation is strictly non-decreasing. We penalize 
predictions where lap time decreases as tyre life increases:

$$\mathcal{L}_{physics} = \frac{1}{N-1}\sum_{i=1}^{N-1} 
\max(0, -(\hat{y}_{i+1} - \hat{y}_i)) \cdot \mathbf{1}[\tau_{i+1} > \tau_i]$$

where $\tau_i$ is the tyre life at lap $i$.

**Thermal Constraint.** Higher track temperatures accelerate rubber 
degradation. We penalize flat degradation predictions when track temperature 
exceeds 40°C normalized threshold $T^*$:

$$\mathcal{L}_{thermal} = \max(0, \bar{T} - T^*) \cdot 
\frac{1}{N-1}\sum_{i=1}^{N-1}\max(0, -(\hat{y}_{i+1} - \hat{y}_i))$$

**Track Abrasiveness Scaling.** The physics penalty is scaled by circuit 
abrasiveness $\alpha \in [1,10]$, reflecting that violations at Bahrain 
(α=8) are physically less plausible than at Monza (α=3):

$$w_\alpha = 0.5 + \frac{\alpha - 1}{9} \times 1.5$$

The combined loss is:

$$\mathcal{L}_{total} = \mathcal{L}_{base} + 
\lambda_p \cdot w_\alpha \cdot \mathcal{L}_{physics} + 
\lambda_t \cdot \mathcal{L}_{thermal}$$

We set $\lambda_p = 0.1$ and $\lambda_t = 0.05$ based on ablation study.

## 3.3 Thermal Energy Model

We model cumulative tyre thermal energy dissipation $Q$ from three 
physical mechanisms:

$$Q = k_1 F_{lat}^2 + k_2 F_{lon}^2 + k_3 v^2 T_{track}$$

where $F_{lat}$ and $F_{lon}$ are estimated lateral and longitudinal tyre 
loads from sector time analysis, $v$ is a speed proxy derived from lap time, 
and $T_{track}$ is track temperature. Constants $k_1=0.0012$, $k_2=0.0008$, 
$k_3=0.0003$ are calibrated from the dataset.

Compound sensitivity factors reflect rubber hardness:
SOFT: 1.35×, MEDIUM: 1.0×, HARD: 0.72×.

This model predicts that a SOFT tyre at Bahrain in 52°C conditions 
accumulates 2.12× more thermal energy per lap than a MEDIUM at Monza in 
35°C — consistent with observed degradation patterns in the data.

## 3.4 Transformer Architecture

We replace the standard LSTM backbone with a Transformer encoder, 
motivated by its superior ability to model direct dependencies between 
any two laps in the sequence regardless of distance.

Given an input sequence $X \in \mathbb{R}^{L \times d_{in}}$ of $L=8$ 
consecutive laps each with $d_{in}=10$ features, we apply:

1. **Input projection**: $H_0 = XW_{proj} + b_{proj}$, $H_0 \in \mathbb{R}^{L \times d_{model}}$
2. **Positional encoding**: sinusoidal, injected into $H_0$
3. **Transformer encoder**: 2 layers, $d_{model}=64$, 4 attention heads
4. **Output head**: MLP(64→32→1) applied to final timestep

Monte Carlo dropout ($p=0.3$) is applied after the Transformer output 
during both training and inference, enabling calibrated uncertainty 
quantification via 30-sample averaging.

The Transformer achieves validation loss 0.0006 versus the LSTM baseline 
of 0.0008 — a 25% improvement — on our held-out 2023-2024 test set.

## 3.5 Model Hierarchy

TyrePIML employs a three-tier model selection strategy:

1. **Compound-specific Transformer** (primary): trained on all laps of a 
   single compound (SOFT/MEDIUM/HARD) across all seasons
2. **Track-specific Transformer** (secondary): trained on all historical 
   races at the target circuit
3. **Generic Transformer** (fallback): trained on the full dataset

A smart router selects the highest-priority available model for each 
driver-stint combination at inference time. 