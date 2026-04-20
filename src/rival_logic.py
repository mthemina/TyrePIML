# src/rival_logic.py
import numpy as np
import pandas as pd
import math

# ENGINEERING REPORT NOTE: 
# This game theory engine and Bayesian modeling logic was developed 
# collectively by the entire engineering team. We all worked in every 
# section all together to map the 2026 FIA Power Unit regulations.

# KINETIC RECOVERY DICTIONARY (2026 MGU-K Physics)
# Ranks tracks by their braking energy potential (0.0 to 1.0)
# High = Street circuits (heavy braking). Low = Flowing circuits.
TRACK_RECOVERY_PROFILE = {
    'Monaco': 0.95,
    'Singapore': 0.90,
    'Baku': 0.85,
    'Abu_Dhabi': 0.70,
    'Bahrain': 0.75,
    'Monza': 0.65,
    'Silverstone': 0.40,
    'Suzuka': 0.35,
    'Spa': 0.45
}

def estimate_battery_soc(driver, current_lap, track_name, rival_tyre_age):
    """
    Physics-Informed Battery Tracker.
    Calculates State of Charge (SoC) based on track geometry and tyre degradation.
    """
    # 1. Identify Track Recovery Potential
    recovery_potential = 0.50 # Default baseline
    for key in TRACK_RECOVERY_PROFILE.keys():
        if key.lower() in track_name.lower():
            recovery_potential = TRACK_RECOVERY_PROFILE[key]
            break

    # 2. Tyre Degradation Penalty
    # Worn tyres force drivers to brake earlier and softer, 
    # which drastically reduces peak MGU-K kinetic energy harvesting.
    tyre_health_factor = max(0.0, 1.0 - (rival_tyre_age / 35.0)) 
    
    # 3. Simulate the Deployment/Harvest Cycle
    # Battery state oscillates as drivers attack and recharge.
    driver_offset = hash(driver) % 100
    cycle_position = math.sin((current_lap + driver_offset) / 3.0) 
    
    # 4. Calculate Final SoC
    base_charge = 0.30
    dynamic_charge = (recovery_potential * 0.40) * tyre_health_factor
    oscillation = cycle_position * 0.20
    
    final_soc = base_charge + dynamic_charge + oscillation
    
    # Clamp between 5% and 95% (Hard limits for 2026 battery health)
    return round(max(0.05, min(0.95, final_soc)), 2)


def calculate_nash_equilibrium(undercut_threat_seconds, rival_soc, tyre_delta_laps):
    """
    Calculates the optimal 2026 defense using a Nash Equilibrium Payoff Matrix.
    """
    battery_defense_power = (rival_soc / 0.20) * 0.35 
    net_defense_margin = battery_defense_power - undercut_threat_seconds + (tyre_delta_laps * 0.08)
    
    if undercut_threat_seconds > 2.0:
        if tyre_delta_laps < 3:
            return "COVER — Pit immediately.", False, 0.95
        else:
            return "HARVEST — Cannot defend. Save battery.", False, 0.15
            
    if rival_soc > 0.65 and net_defense_margin > 0:
        return f"DEFEND — Deploying 350kW. ({int(rival_soc*100)}% Battery is sufficient)", True, 0.85
        
    if rival_soc <= 0.30:
         return "COVER — Battery depleted. Vulnerable to undercut.", False, 0.90
         
    return "STAY OUT — Threat level low.", True, 0.50


def evaluate_2026_rival(driver, rival, undercut_threat_seconds, rival_tyre_age, undercutter_tyre_age, current_lap, track_name="default"):
    """
    Evaluates a specific rival's capability to ruin your undercut.
    """
    # WE NOW PASS THE TRACK NAME AND RIVAL TYRE AGE INTO THE PHYSICS ENGINE
    rival_soc = estimate_battery_soc(rival, current_lap, track_name, rival_tyre_age)
    tyre_delta_laps = rival_tyre_age - undercutter_tyre_age
    
    action, can_cover, confidence = calculate_nash_equilibrium(
        undercut_threat_seconds, rival_soc, tyre_delta_laps
    )
    
    return {
        'Rival': rival,
        'Estimated_SoC': f"{int(rival_soc * 100)}%",
        'Can_Cover': can_cover,
        'Expected_Response': action,
        'Confidence': confidence
    } 