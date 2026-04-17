import pandas as pd
import numpy as np
import glob
import json
import os


def calculate_driver_degradation_rates(data_path='data/'):
    """
    Calculate each driver's historical tyre degradation rate
    per compound across all races.
    
    Degradation rate = average lap time increase per lap of tyre age
    Lower rate = gentler on tyres
    Higher rate = harder on tyres
    """
    all_files = glob.glob(f'{data_path}*.csv')
    all_data = []
    
    for file in all_files:
        df = pd.read_csv(file)
        all_data.append(df)
    
    if not all_data:
        return {}
    
    df = pd.concat(all_data, ignore_index=True)
    
    profiles = {}
    
    for driver in df['Driver'].unique():
        driver_df = df[df['Driver'] == driver]
        driver_profile = {'compound_rates': {}, 'overall_rate': None}
        
        compound_rates = []
        
        for compound in ['SOFT', 'MEDIUM', 'HARD']:
            compound_df = driver_df[driver_df['Compound'] == compound]
            
            if len(compound_df) < 20:
                continue
            
            # Calculate degradation rate per stint
            stint_rates = []
            
            # Pylance-safe grouping
            group_cols = ['Event', 'Stint'] if 'Event' in compound_df.columns else ['Stint']
            
            for _, stint_df in compound_df.groupby(group_cols):
                if len(stint_df) < 5:
                    continue
                
                stint_df = stint_df.sort_values('TyreLife') 
                
                # Linear regression slope = degradation rate
                x = stint_df['TyreLife'].values
                y = stint_df['LapTime'].values
                
                if len(x) < 3:
                    continue
                
                # Simple linear regression
                x_mean = x.mean()
                y_mean = y.mean()
                slope = np.sum((x - x_mean) * (y - y_mean)) / \
                        np.sum((x - x_mean) ** 2 + 1e-8)
                
                if -1.0 < slope < 2.0:  # filter outliers
                    stint_rates.append(slope)
            
            if stint_rates:
                avg_rate = np.median(stint_rates)
                driver_profile['compound_rates'][compound] = round(float(avg_rate), 4)
                compound_rates.append(avg_rate)
        
        if compound_rates:
            driver_profile['overall_rate'] = round(float(np.median(compound_rates)), 4)
        
        profiles[driver] = driver_profile
    
    return profiles


def get_field_average_rates(profiles):
    """Calculate field average degradation rates per compound."""
    compound_all = {'SOFT': [], 'MEDIUM': [], 'HARD': []}
    
    for driver, profile in profiles.items():
        for compound, rate in profile['compound_rates'].items():
            compound_all[compound].append(rate)
    
    return {
        compound: round(float(np.median(rates)), 4)
        for compound, rates in compound_all.items()
        if rates
    }


def classify_driver_style(rate, field_avg):
    """
    Classify driver tyre management style relative to field average.
    Negative rate = lap times improving with age (unusual, could be track evolution)
    Low positive rate = gentle on tyres
    High positive rate = hard on tyres
    """
    if rate is None or field_avg is None:
        return 'UNKNOWN'
    
    diff = rate - field_avg
    
    if diff < -0.02:
        return 'VERY GENTLE'
    elif diff < -0.005:
        return 'GENTLE'
    elif diff < 0.005:
        return 'AVERAGE'
    elif diff < 0.02:
        return 'HARD'
    else:
        return 'VERY HARD'


def build_driver_profiles():
    """Build and save complete driver profiles."""
    print("Calculating driver degradation profiles...")
    profiles = calculate_driver_degradation_rates()
    field_avg = get_field_average_rates(profiles)
    
    print(f"Field average degradation rates: {field_avg}")
    
    # Add style classification
    enriched = {}
    for driver, profile in profiles.items():
        overall_style = classify_driver_style(
            profile['overall_rate'],
            np.mean(list(field_avg.values()))
        )
        
        compound_styles = {}
        for compound, rate in profile['compound_rates'].items():
            compound_styles[compound] = {
                'rate': rate,
                'style': classify_driver_style(rate, field_avg.get(compound))
            }
        
        enriched[driver] = {
            'overall_rate': profile['overall_rate'],
            'overall_style': overall_style,
            'compounds': compound_styles
        }
    
    # Sort by overall rate
    sorted_profiles = dict(
        sorted(enriched.items(), 
               key=lambda x: x[1]['overall_rate'] or 999)
    )
    
    # Save
    os.makedirs('results', exist_ok=True)
    with open('results/driver_profiles.json', 'w') as f:
        json.dump(sorted_profiles, f, indent=2)
    
    print(f"\nDriver profiles saved for {len(sorted_profiles)} drivers")
    return sorted_profiles, field_avg

def get_driver_style_encoding(driver, profiles=None):
    """
    Return a normalized driver style encoding for use as model feature.
    0.0 = very gentle, 0.5 = average, 1.0 = very hard on tyres
    """
    if profiles is None:
        try:
            with open('results/driver_profiles.json', 'r') as f:
                profiles = json.load(f)
        except FileNotFoundError:
            return 0.5  # default to average
    
    if driver not in profiles:
        return 0.5
    
    rate = profiles[driver]['overall_rate']
    if rate is None:
        return 0.5
    
    # Clamp to reasonable range and normalize
    rate_clamped = max(-0.1, min(0.1, rate))
    normalized = (rate_clamped + 0.1) / 0.2
    return round(float(normalized), 3)


def compare_drivers(driver1, driver2, compound='MEDIUM', profiles=None):
    """Compare two drivers' tyre management on a specific compound."""
    if profiles is None:
        with open('results/driver_profiles.json', 'r') as f:
            profiles = json.load(f)
    
    d1 = profiles.get(driver1, {})
    d2 = profiles.get(driver2, {})
    
    d1_rate = d1.get('compounds', {}).get(compound, {}).get('rate', None)
    d2_rate = d2.get('compounds', {}).get(compound, {}).get('rate', None)
    
    print(f"\n{compound} tyre comparison:")
    print(f"  {driver1}: {d1_rate} ({d1.get('compounds',{}).get(compound,{}).get('style','N/A')})")
    print(f"  {driver2}: {d2_rate} ({d2.get('compounds',{}).get(compound,{}).get('style','N/A')})")
    
    if d1_rate and d2_rate:
        diff = d2_rate - d1_rate
        harder = driver2 if diff > 0 else driver1
        print(f"  {harder} is harder on {compound} tyres "
              f"by {abs(diff):.4f}s/lap") 

if __name__ == '__main__':
    profiles, field_avg = build_driver_profiles()
    
    print(f"\n{'Driver':<6} {'Overall Rate':>12} {'Style':<12} "
          f"{'SOFT':>8} {'MEDIUM':>8} {'HARD':>8}")
    print("-" * 65)
    
    for driver, profile in list(profiles.items())[:20]:
        soft = profile['compounds'].get('SOFT', {}).get('rate', 'N/A')
        med = profile['compounds'].get('MEDIUM', {}).get('rate', 'N/A')
        hard = profile['compounds'].get('HARD', {}).get('rate', 'N/A')
        
        print(f"{driver:<6} {profile['overall_rate']:>12.4f} "
              f"{profile['overall_style']:<12} "
              f"{str(soft):>8} {str(med):>8} {str(hard):>8}") 