# Track profiles — physical characteristics that affect tyre degradation
# abrasiveness: 1-10 scale (10 = most abrasive, destroys tyres fastest)
# downforce: 1-10 scale (10 = maximum downforce = more tyre load)
# pit_loss: seconds lost in pit lane
# avg_lap_time: approximate reference lap time in seconds
# tyre_sensitive: whether strategy is heavily tyre-dependent

TRACK_PROFILES = {
    'Bahrain Grand Prix': {
        'circuit': 'Bahrain International Circuit',
        'abrasiveness': 8,
        'downforce': 6,
        'pit_loss': 22.0,
        'avg_lap_time': 95.0,
        'tyre_sensitive': True,
        'characteristics': 'High abrasion, hot and sandy, heavy on rears'
    },
    'Saudi Arabian Grand Prix': {
        'circuit': 'Jeddah Corniche Circuit',
        'abrasiveness': 3,
        'downforce': 5,
        'pit_loss': 24.0,
        'avg_lap_time': 90.0,
        'tyre_sensitive': False,
        'characteristics': 'Low degradation, street circuit, high speed'
    },
    'Australian Grand Prix': {
        'circuit': 'Albert Park',
        'abrasiveness': 4,
        'downforce': 6,
        'pit_loss': 23.0,
        'avg_lap_time': 85.0,
        'tyre_sensitive': False,
        'characteristics': 'Smooth surface, moderate degradation'
    },
    'Spanish Grand Prix': {
        'circuit': 'Circuit de Barcelona-Catalunya',
        'abrasiveness': 7,
        'downforce': 8,
        'pit_loss': 23.5,
        'avg_lap_time': 82.0,
        'tyre_sensitive': True,
        'characteristics': 'High downforce, heavy on fronts, classic two-stopper'
    },
    'Canadian Grand Prix': {
        'circuit': 'Circuit Gilles Villeneuve',
        'abrasiveness': 3,
        'downforce': 4,
        'pit_loss': 23.0,
        'avg_lap_time': 75.0,
        'tyre_sensitive': False,
        'characteristics': 'Low degradation, stop-go circuit, heavy braking'
    },
    'Austrian Grand Prix': {
        'circuit': 'Red Bull Ring',
        'abrasiveness': 5,
        'downforce': 5,
        'pit_loss': 21.0,
        'avg_lap_time': 67.0,
        'tyre_sensitive': True,
        'characteristics': 'Short lap, moderate degradation, kerb usage critical'
    },
    'British Grand Prix': {
        'circuit': 'Silverstone',
        'abrasiveness': 6,
        'downforce': 8,
        'pit_loss': 24.0,
        'avg_lap_time': 92.0,
        'tyre_sensitive': True,
        'characteristics': 'High speed, heavy on fronts, weather unpredictable'
    },
    'Hungarian Grand Prix': {
        'circuit': 'Hungaroring',
        'abrasiveness': 5,
        'downforce': 9,
        'pit_loss': 22.0,
        'avg_lap_time': 80.0,
        'tyre_sensitive': True,
        'characteristics': 'Maximum downforce, heavy on rears, undercut crucial'
    },
    'Belgian Grand Prix': {
        'circuit': 'Spa-Francorchamps',
        'abrasiveness': 4,
        'downforce': 4,
        'pit_loss': 23.5,
        'avg_lap_time': 107.0,
        'tyre_sensitive': False,
        'characteristics': 'Low downforce, weather critical, long lap'
    },
    'Dutch Grand Prix': {
        'circuit': 'Zandvoort',
        'abrasiveness': 7,
        'downforce': 8,
        'pit_loss': 21.0,
        'avg_lap_time': 73.0,
        'tyre_sensitive': True,
        'characteristics': 'High abrasion, banked corners, two-stopper likely'
    },
    'Italian Grand Prix': {
        'circuit': 'Monza',
        'abrasiveness': 3,
        'downforce': 2,
        'pit_loss': 22.0,
        'avg_lap_time': 85.0,
        'tyre_sensitive': False,
        'characteristics': 'Lowest downforce, low degradation, slipstream critical'
    },
    'Japanese Grand Prix': {
        'circuit': 'Suzuka',
        'abrasiveness': 5,
        'downforce': 7,
        'pit_loss': 23.0,
        'avg_lap_time': 93.0,
        'tyre_sensitive': True,
        'characteristics': 'Technical circuit, heavy on rears, figure-8 layout'
    },
    'United States Grand Prix': {
        'circuit': 'Circuit of the Americas',
        'abrasiveness': 6,
        'downforce': 7,
        'pit_loss': 23.5,
        'avg_lap_time': 97.0,
        'tyre_sensitive': True,
        'characteristics': 'Bumpy surface, high energy input, two-stopper'
    },
    'Mexican Grand Prix': {
        'circuit': 'Autodromo Hermanos Rodriguez',
        'abrasiveness': 4,
        'downforce': 8,
        'pit_loss': 22.5,
        'avg_lap_time': 79.0,
        'tyre_sensitive': False,
        'characteristics': 'High altitude, low air density, unusual tyre behaviour'
    },
    'Brazilian Grand Prix': {
        'circuit': 'Interlagos',
        'abrasiveness': 5,
        'downforce': 6,
        'pit_loss': 22.0,
        'avg_lap_time': 73.0,
        'tyre_sensitive': True,
        'characteristics': 'Bumpy surface, weather unpredictable, safety cars common'
    },
    'Abu Dhabi Grand Prix': {
        'circuit': 'Yas Marina',
        'abrasiveness': 3,
        'downforce': 6,
        'pit_loss': 23.0,
        'avg_lap_time': 88.0,
        'tyre_sensitive': False,
        'characteristics': 'Smooth surface, low degradation, season finale'
    },
    'Azerbaijan Grand Prix': {
        'circuit': 'Baku City Circuit',
        'abrasiveness': 3,
        'downforce': 3,
        'pit_loss': 25.0,
        'avg_lap_time': 105.0,
        'tyre_sensitive': False,
        'characteristics': 'Street circuit, long straight, safety car prone'
    },
    'Miami Grand Prix': {
        'circuit': 'Miami International Autodrome',
        'abrasiveness': 5,
        'downforce': 6,
        'pit_loss': 23.0,
        'avg_lap_time': 91.0,
        'tyre_sensitive': True,
        'characteristics': 'New surface, moderate degradation, hot conditions'
    },
    'Las Vegas Grand Prix': {
        'circuit': 'Las Vegas Strip Circuit',
        'abrasiveness': 2,
        'downforce': 4,
        'pit_loss': 24.0,
        'avg_lap_time': 95.0,
        'tyre_sensitive': False,
        'characteristics': 'Cold night race, low degradation, unusual conditions'
    },
    'Qatar Grand Prix': {
        'circuit': 'Losail International Circuit',
        'abrasiveness': 8,
        'downforce': 7,
        'pit_loss': 22.0,
        'avg_lap_time': 85.0,
        'tyre_sensitive': True,
        'characteristics': 'Very high degradation, blistering common, night race'
    },
}

def get_track_profile(event_name):
    """Get track profile for a given event name."""
    # Try exact match first
    if event_name in TRACK_PROFILES:
        return TRACK_PROFILES[event_name]
    
    # Try partial match
    for key in TRACK_PROFILES:
        if key.lower() in event_name.lower() or event_name.lower() in key.lower():
            return TRACK_PROFILES[key]
    
    # Return default profile
    return {
        'circuit': event_name,
        'abrasiveness': 5,
        'downforce': 5,
        'pit_loss': 23.0,
        'avg_lap_time': 90.0,
        'tyre_sensitive': True,
        'characteristics': 'Unknown circuit'
    }

def get_degradation_multiplier(event_name):
    """
    Return a degradation multiplier based on track abrasiveness.
    Used to scale physics constraints per circuit.
    """
    profile = get_track_profile(event_name)
    # Scale 1-10 abrasiveness to 0.5-2.0 multiplier
    return 0.5 + (profile['abrasiveness'] - 1) * (1.5 / 9)


if __name__ == '__main__':
    # Print all track profiles as a summary table
    print(f"{'Event':<35} {'Abrasion':>8} {'Downforce':>9} {'Pit Loss':>8} {'Tyre Sensitive':>14}")
    print("-" * 80)
    for event, profile in sorted(TRACK_PROFILES.items()):
        print(f"{event:<35} {profile['abrasiveness']:>8} "
              f"{profile['downforce']:>9} {profile['pit_loss']:>8} "
              f"{'Yes' if profile['tyre_sensitive'] else 'No':>14}") 