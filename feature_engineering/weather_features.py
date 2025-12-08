"""
PMFBY v2.0 - Advanced Weather Feature Engineering
=================================================
Computes 15 scientifically-derived weather features from NASA POWER data.

Features:
- W1: Rainfall CV (distribution)
- W2: Dry Spell Count
- W3: Rainfall Anomaly
- W4: Growing Degree Days (GDD)
- W5: Heat Stress Intensity
- W6: VPD (Vapor Pressure Deficit)
- W7-W8: Critical Stage Rainfall
- W9: Humidity Variability
- W10: Evapotranspiration
- W11: Water Balance
- W12: Max Hot Streak
- W13: Night Temperature Mean
- W14: Diurnal Temperature Range
- W15: Pre-Monsoon Rainfall
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional
from datetime import datetime

# Climatological averages for Maharashtra (long-term means)
CLIMATE_NORMALS = {
    'Maharashtra': {
        'rain_avg': 850,  # mm (June-November)
        'rain_std': 180,
        'temp_avg': 27.5,
        'temp_std': 2.5
    }
}


def compute_gdd(tmax: np.ndarray, tmin: np.ndarray, 
                t_base: float = 10.0, t_upper: float = 35.0) -> float:
    """
    Compute Growing Degree Days (GDD).
    
    Formula: GDD = sum(max(0, T_avg - T_base))
    where T_avg is capped at T_upper
    
    Args:
        tmax: Daily maximum temperature array
        tmin: Daily minimum temperature array
        t_base: Base temperature for crop growth (default 10°C)
        t_upper: Upper temperature limit (default 35°C)
    
    Returns:
        Total GDD for the season
    """
    t_avg = (tmax + tmin) / 2
    t_avg = np.clip(t_avg, t_base, t_upper)
    gdd = np.sum(np.maximum(0, t_avg - t_base))
    return float(gdd)


def compute_vpd(tmax: np.ndarray, tmin: np.ndarray, rh: np.ndarray) -> np.ndarray:
    """
    Compute Vapor Pressure Deficit (VPD).
    
    Formula: VPD = e_s - e_a
    where e_s = saturation vapor pressure
          e_a = actual vapor pressure
    
    Args:
        tmax: Daily maximum temperature
        tmin: Daily minimum temperature
        rh: Daily relative humidity (%)
    
    Returns:
        Daily VPD array (kPa)
    """
    t_avg = (tmax + tmin) / 2
    # Saturation vapor pressure (Tetens formula)
    e_s = 0.6108 * np.exp(17.27 * t_avg / (t_avg + 237.3))
    # Actual vapor pressure
    e_a = e_s * (rh / 100)
    vpd = e_s - e_a
    return vpd


def compute_et0(tmax: np.ndarray, tmin: np.ndarray, rh: np.ndarray,
                solar_rad: Optional[np.ndarray] = None, lat: float = 19.0) -> np.ndarray:
    """
    Compute Reference Evapotranspiration (ET0) using simplified Hargreaves method.
    
    Formula: ET0 = 0.0023 * Ra * (T_mean + 17.8) * sqrt(T_max - T_min)
    
    Args:
        tmax: Daily maximum temperature
        tmin: Daily minimum temperature
        rh: Daily relative humidity
        solar_rad: Daily solar radiation (optional)
        lat: Latitude (for Ra calculation)
    
    Returns:
        Daily ET0 array (mm/day)
    """
    t_mean = (tmax + tmin) / 2
    t_range = tmax - tmin
    t_range = np.maximum(t_range, 0.1)  # Avoid negative/zero
    
    # Approximate extraterrestrial radiation (Ra) in mm/day
    # Simplified: Ra ≈ 10-15 mm/day for Maharashtra latitude
    ra = 12.0  # mm/day average
    
    et0 = 0.0023 * ra * (t_mean + 17.8) * np.sqrt(t_range)
    return np.maximum(et0, 0)


def count_dry_spells(rain: np.ndarray, threshold: float = 1.0, min_days: int = 7) -> int:
    """
    Count number of dry spells (consecutive days with rain < threshold).
    
    Args:
        rain: Daily rainfall array
        threshold: Rainfall threshold for "dry" day (mm)
        min_days: Minimum consecutive days to count as dry spell
    
    Returns:
        Number of dry spells
    """
    is_dry = rain < threshold
    dry_spells = 0
    consecutive = 0
    
    for dry in is_dry:
        if dry:
            consecutive += 1
        else:
            if consecutive >= min_days:
                dry_spells += 1
            consecutive = 0
    
    # Check last spell
    if consecutive >= min_days:
        dry_spells += 1
    
    return dry_spells


def max_consecutive(condition: np.ndarray) -> int:
    """Find maximum consecutive True values."""
    max_streak = 0
    current = 0
    
    for val in condition:
        if val:
            current += 1
            max_streak = max(max_streak, current)
        else:
            current = 0
    
    return max_streak


def compute_weather_features(daily_df: pd.DataFrame, 
                              state: str = 'Maharashtra') -> Dict[str, float]:
    """
    Compute all 15 advanced weather features from daily weather data.
    
    Args:
        daily_df: DataFrame with columns:
            - PRECTOTCORR: Daily rainfall (mm)
            - T2M: Mean temperature (°C)
            - T2M_MAX: Max temperature (°C)
            - T2M_MIN: Min temperature (°C)
            - RH2M: Relative humidity (%)
            - date: Date index
        state: State for climatological normals
    
    Returns:
        Dictionary with 15 weather features
    """
    # Extract arrays
    rain = daily_df['PRECTOTCORR'].values
    tmax = daily_df['T2M_MAX'].values
    tmin = daily_df['T2M_MIN'].values
    rh = daily_df['RH2M'].values
    
    # Get dates for monthly filtering
    if isinstance(daily_df.index, pd.DatetimeIndex):
        months = daily_df.index.month
    else:
        months = pd.to_datetime(daily_df['date']).dt.month.values if 'date' in daily_df else np.ones(len(rain)) * 7
    
    # Climate normals
    clim = CLIMATE_NORMALS.get(state, CLIMATE_NORMALS['Maharashtra'])
    
    # Compute derived quantities
    vpd = compute_vpd(tmax, tmin, rh)
    et0 = compute_et0(tmax, tmin, rh)
    
    # Monthly masks
    jun_jul_mask = (months >= 6) & (months <= 7)
    aug_sep_mask = (months >= 8) & (months <= 9)
    apr_may_mask = (months >= 4) & (months <= 5)
    oct_mask = months == 10
    
    features = {
        # W1: Rainfall Distribution (CV)
        'rain_cv': float(rain.std() / rain.mean()) if rain.mean() > 0 else 0,
        
        # W2: Dry Spell Count
        'dry_spell_count': count_dry_spells(rain, threshold=1.0, min_days=7),
        
        # W3: Rainfall Anomaly (z-score vs climatology)
        'rain_anomaly': float((rain.sum() - clim['rain_avg']) / clim['rain_std']),
        
        # W4: Growing Degree Days
        'gdd': compute_gdd(tmax, tmin, t_base=10, t_upper=35),
        
        # W5: Heat Stress Intensity (cumulative degrees above 35°C)
        'heat_stress_intensity': float(np.sum(np.maximum(0, tmax - 35))),
        
        # W6: Mean VPD
        'vpd_mean': float(np.mean(vpd)),
        
        # W7: Critical Stage Rainfall (June-July) - Vegetative
        'rain_jun_jul': float(rain[jun_jul_mask].sum()) if jun_jul_mask.any() else 0,
        
        # W8: Critical Stage Rainfall (Aug-Sep) - Flowering
        'rain_aug_sep': float(rain[aug_sep_mask].sum()) if aug_sep_mask.any() else 0,
        
        # W9: Humidity Variability
        'humidity_cv': float(rh.std()),
        
        # W10: Total Evapotranspiration
        'et_total': float(et0.sum()),
        
        # W11: Water Balance (Rain - ET)
        'water_balance': float(rain.sum() - et0.sum()),
        
        # W12: Maximum Consecutive Hot Days
        'max_hot_streak': max_consecutive(tmax > 35),
        
        # W13: Night Temperature Mean
        'night_temp_mean': float(tmin.mean()),
        
        # W14: Diurnal Temperature Range
        'diurnal_range': float((tmax - tmin).mean()),
        
        # W15: Pre-Monsoon Rainfall (Apr-May)
        'rain_premonsoon': float(rain[apr_may_mask].sum()) if apr_may_mask.any() else 0,
    }
    
    # Additional derived features
    features['rain_total'] = float(rain.sum())
    features['rain_days'] = int((rain > 1).sum())
    features['temp_mean'] = float((tmax + tmin).mean() / 2)
    features['temp_max'] = float(tmax.max())
    features['heat_days'] = int((tmax > 35).sum())
    features['humidity_mean'] = float(rh.mean())
    
    return features


def fetch_and_compute_weather(lat: float, lon: float, year: int) -> Dict[str, float]:
    """
    Fetch weather from NASA POWER and compute all features.
    
    Args:
        lat: Latitude
        lon: Longitude
        year: Year to fetch
    
    Returns:
        Dictionary with all weather features
    """
    import requests
    
    # Fetch from NASA POWER
    url = "https://power.larc.nasa.gov/api/temporal/daily/point"
    params = {
        'start': f"{year}0401",  # Start from April for pre-monsoon
        'end': f"{year}1130",
        'latitude': lat,
        'longitude': lon,
        'community': 'AG',
        'parameters': 'PRECTOTCORR,T2M,T2M_MAX,T2M_MIN,RH2M,ALLSKY_SFC_SW_DWN',
        'format': 'JSON'
    }
    
    response = requests.get(url, params=params, timeout=60)
    data = response.json()
    params_data = data.get('properties', {}).get('parameter', {})
    
    # Convert to DataFrame
    dates = list(params_data.get('PRECTOTCORR', {}).keys())
    df = pd.DataFrame({
        'date': pd.to_datetime(dates, format='%Y%m%d'),
        'PRECTOTCORR': list(params_data.get('PRECTOTCORR', {}).values()),
        'T2M': list(params_data.get('T2M', {}).values()),
        'T2M_MAX': list(params_data.get('T2M_MAX', {}).values()),
        'T2M_MIN': list(params_data.get('T2M_MIN', {}).values()),
        'RH2M': list(params_data.get('RH2M', {}).values()),
    })
    df = df.set_index('date')
    
    # Clean missing values (-999)
    df = df.replace(-999, np.nan).fillna(method='ffill').fillna(method='bfill')
    
    # Compute features
    features = compute_weather_features(df)
    
    return features


# ===============================================
# EXAMPLE USAGE
# ===============================================
if __name__ == "__main__":
    # Test with Ahmednagar coordinates
    lat, lon = 19.071591, 74.774179
    year = 2024
    
    print("=" * 70)
    print("ADVANCED WEATHER FEATURES (15 Variables)")
    print("=" * 70)
    print(f"Location: {lat}N, {lon}E")
    print(f"Year: {year}")
    print()
    
    features = fetch_and_compute_weather(lat, lon, year)
    
    print("WEATHER FEATURES:")
    print("-" * 50)
    for key, value in features.items():
        if isinstance(value, float):
            print(f"  {key:25s}: {value:10.2f}")
        else:
            print(f"  {key:25s}: {value:10d}")
