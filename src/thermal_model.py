import numpy as np

"""
Tyre Thermal Energy Model for TyrePIML

Physical basis:
- Tyres generate heat through three mechanisms:
  1. Hysteretic deformation (rubber flexing under load)
  2. Sliding friction (micro-slip at contact patch)
  3. Viscoelastic dissipation (internal damping)

- Energy dissipated per lap (Q) approximates:
  Q = k1 * F_lateral^2 + k2 * F_longitudinal^2 + k3 * v^2 * T_track

- As cumulative energy Q increases, rubber degrades:
  - Rubber molecules break and reform under repeated stress
  - Grain structure breaks down → blistering or graining
  - Surface temperature rises → thermal degradation cliff

This model estimates Q from available lap data (sector times, track profile)
and uses it as an additional physics-grounded feature.
"""

# Physical constants (approximate, calibrated from F1 data)
K_LATERAL = 0.0012      # lateral force coefficient
K_LONGITUDINAL = 0.0008 # longitudinal force coefficient  
K_THERMAL = 0.0003      # thermal dissipation coefficient

# Compound thermal sensitivity (relative to MEDIUM baseline)
COMPOUND_THERMAL = {
    'SOFT':   1.35,  # softer rubber → more heat generation
    'MEDIUM': 1.00,  # baseline
    'HARD':   0.72,  # harder rubber → less heat, slower buildup
}

# Track thermal load factors (from track_profiles abrasiveness)
TRACK_THERMAL_FACTOR = {
    1: 0.6,   # very smooth (Las Vegas)
    2: 0.7,
    3: 0.8,   # smooth (Monza, Abu Dhabi)
    4: 0.9,
    5: 1.0,   # baseline (most tracks)
    6: 1.1,
    7: 1.2,   # abrasive (Spanish, Dutch)
    8: 1.35,  # very abrasive (Bahrain, Qatar)
    9: 1.5,
    10: 1.7,  # extreme
}


def estimate_lateral_load(sector1, sector2, sector3):
    """
    Estimate lateral tyre load from sector times.
    Slower sector times relative to baseline → more cornering → more lateral load.
    Uses sector time ratios as proxy for corner intensity.
    """
    total = sector1 + sector2 + sector3
    # Cornering-heavy sectors have higher time fraction
    s1_fraction = sector1 / total
    s2_fraction = sector2 / total
    s3_fraction = sector3 / total
    
    # Variance in sector fractions indicates mixed high-speed/slow corners
    variance = np.var([s1_fraction, s2_fraction, s3_fraction])
    lateral_load = 100 + variance * 5000  # normalized units
    return lateral_load


def estimate_longitudinal_load(lap_time, sector1, sector2, sector3):
    """
    Estimate braking/acceleration load from lap characteristics.
    More stop-go circuits → higher longitudinal load.
    """
    avg_sector = lap_time / 3
    # High deviation from average sector = more braking zones
    deviations = [abs(s - avg_sector) for s in [sector1, sector2, sector3]]
    longitudinal_load = 80 + np.mean(deviations) * 10
    return longitudinal_load


def calculate_thermal_energy(
    lap_time, sector1, sector2, sector3,
    track_temp, compound, abrasiveness,
    tyre_life
):
    """
    Calculate cumulative thermal energy dissipated in the tyre at this lap.
    
    Returns a normalized energy value (0-1 scale) that represents
    how much the tyre has thermally degraded.
    
    Higher value = more energy dissipated = closer to cliff.
    """
    # Load estimates
    F_lat = estimate_lateral_load(sector1, sector2, sector3)
    F_lon = estimate_longitudinal_load(lap_time, sector1, sector2, sector3)
    
    # Average speed proxy (lower lap time = higher speed = more thermal load)
    speed_proxy = 300.0 / lap_time  # normalized speed
    
    # Track thermal factor
    abrasiveness_int = max(1, min(10, int(round(abrasiveness))))
    track_factor = TRACK_THERMAL_FACTOR.get(abrasiveness_int, 1.0)
    
    # Compound sensitivity
    compound_factor = COMPOUND_THERMAL.get(compound, 1.0)
    
    # Temperature amplification (hotter track = faster degradation)
    temp_factor = 1.0 + max(0, (track_temp - 30.0)) * 0.02
    
    # Energy per lap
    energy_per_lap = (
        K_LATERAL * F_lat**2 +
        K_LONGITUDINAL * F_lon**2 +
        K_THERMAL * speed_proxy**2 * track_temp
    ) * track_factor * compound_factor * temp_factor
    
    # Cumulative energy (increases with tyre life)
    cumulative_energy = energy_per_lap * tyre_life
    
    # Normalize to 0-1 scale (cliff typically occurs around 0.7-0.8)
    normalized = min(1.0, cumulative_energy / 500.0)
    
    return round(float(normalized), 4)


def add_thermal_energy_to_df(df, abrasiveness=5.0):
    """
    Add thermal energy column to a lap dataframe.
    """
    track_temp = df['track_temp_avg'].mean() if 'track_temp_avg' in df.columns else 35.0
    
    thermal_energies = []
    for _, row in df.iterrows():
        energy = calculate_thermal_energy(
            lap_time=row['LapTime'],
            sector1=row['Sector1Time'],
            sector2=row['Sector2Time'],
            sector3=row['Sector3Time'],
            track_temp=track_temp,
            compound=row['Compound'],
            abrasiveness=abrasiveness,
            tyre_life=row['TyreLife']
        )
        thermal_energies.append(energy)
    
    df = df.copy()
    df['ThermalEnergy'] = thermal_energies
    return df


if __name__ == '__main__':
    # Test on a sample lap
    energy = calculate_thermal_energy(
        lap_time=86.5,
        sector1=28.2,
        sector2=29.5,
        sector3=28.8,
        track_temp=35.0,
        compound='MEDIUM',
        abrasiveness=3,  # Monza
        tyre_life=15
    )
    print(f"Monza MEDIUM tyre, lap 15, 35°C track: thermal energy = {energy}")
    
    energy_hot = calculate_thermal_energy(
        lap_time=94.5,
        sector1=31.2,
        sector2=32.5,
        sector3=30.8,
        track_temp=52.0,
        compound='SOFT',
        abrasiveness=8,  # Bahrain
        tyre_life=15
    )
    print(f"Bahrain SOFT tyre, lap 15, 52°C track: thermal energy = {energy_hot}")
    
    print(f"\nBahrain SOFT is {round(energy_hot/energy, 2)}x more thermally stressed than Monza MEDIUM") 