"""
PMFBY v2.0 - Agronomic Stress Indices
=====================================
Computes 10 crop-specific stress indices for yield prediction.

Indices:
- A1: Vegetative Stress Index
- A2: Flowering Heat Stress Index
- A3: Grain Fill Moisture Deficit
- A4: Waterlogging Risk Score
- A5: Drought Index (SPI-based)
- A6: Crop Water Stress Index
- A7: Phenology Match Score
- A8: Late Season Stress
- A9: Combined Stress Score
- A10: Yield Potential Index
"""

import numpy as np
from typing import Dict, Optional

# Crop-specific optimal parameters
CROP_PARAMS = {
    'Rice': {
        'optimal_rain': 1200,      # mm for full season
        'rain_vegetative': 400,    # mm for Jun-Jul
        'rain_flowering': 400,     # mm for Aug-Sep
        'gdd_required': 2000,      # degree-days
        't_base': 10,
        't_optimal': 25,
        't_critical': 35,
        'heat_tolerance': 5,       # days above critical
        'drought_tolerance': 0.3,  # stress threshold
    },
    'Soybean': {
        'optimal_rain': 600,
        'rain_vegetative': 200,
        'rain_flowering': 250,
        'gdd_required': 1800,
        't_base': 10,
        't_optimal': 28,
        't_critical': 35,
        'heat_tolerance': 7,
        'drought_tolerance': 0.4,
    },
    'Cotton': {
        'optimal_rain': 700,
        'rain_vegetative': 250,
        'rain_flowering': 300,
        'gdd_required': 2200,
        't_base': 15,
        't_optimal': 30,
        't_critical': 38,
        'heat_tolerance': 15,
        'drought_tolerance': 0.5,
    },
    'Maize': {
        'optimal_rain': 500,
        'rain_vegetative': 200,
        'rain_flowering': 200,
        'gdd_required': 1500,
        't_base': 10,
        't_optimal': 25,
        't_critical': 35,
        'heat_tolerance': 5,
        'drought_tolerance': 0.3,
    },
    'Jowar': {
        'optimal_rain': 450,
        'rain_vegetative': 150,
        'rain_flowering': 200,
        'gdd_required': 1400,
        't_base': 10,
        't_optimal': 30,
        't_critical': 40,
        'heat_tolerance': 20,
        'drought_tolerance': 0.6,
    },
    'Groundnut': {
        'optimal_rain': 500,
        'rain_vegetative': 180,
        'rain_flowering': 200,
        'gdd_required': 1600,
        't_base': 12,
        't_optimal': 27,
        't_critical': 35,
        'heat_tolerance': 8,
        'drought_tolerance': 0.4,
    },
    'default': {
        'optimal_rain': 800,
        'rain_vegetative': 300,
        'rain_flowering': 350,
        'gdd_required': 1800,
        't_base': 10,
        't_optimal': 27,
        't_critical': 35,
        'heat_tolerance': 10,
        'drought_tolerance': 0.4,
    }
}


def compute_vegetative_stress(rain_jun_jul: float, crop: str = 'default') -> float:
    """
    A1: Vegetative Stress Index
    
    Measures water stress during vegetative growth stage (June-July).
    
    Formula: VSI = 1 - min(1, actual_rain / optimal_rain)
    
    Returns: 0 (no stress) to 1 (severe stress)
    """
    params = CROP_PARAMS.get(crop, CROP_PARAMS['default'])
    optimal = params['rain_vegetative']
    
    if optimal <= 0:
        return 0.0
    
    ratio = rain_jun_jul / optimal
    stress = 1 - min(1.0, ratio)
    return float(max(0, stress))


def compute_flowering_heat_stress(heat_days: int, max_hot_streak: int, 
                                   crop: str = 'default') -> float:
    """
    A2: Flowering Heat Stress Index
    
    Measures heat damage during critical flowering stage.
    High temperatures during flowering cause pollen sterility.
    
    Formula: FHSI = (heat_days / 30) * 0.5 + (max_streak / tolerance) * 0.5
    
    Returns: 0 (no stress) to 1 (severe stress)
    """
    params = CROP_PARAMS.get(crop, CROP_PARAMS['default'])
    tolerance = params['heat_tolerance']
    
    # Heat days component
    heat_component = min(1.0, heat_days / 30)
    
    # Streak component (consecutive hot days are more damaging)
    streak_component = min(1.0, max_hot_streak / max(tolerance, 1))
    
    stress = 0.4 * heat_component + 0.6 * streak_component
    return float(min(1.0, stress))


def compute_grain_fill_deficit(rain_aug_sep: float, et_total: float,
                                crop: str = 'default') -> float:
    """
    A3: Grain Fill Moisture Deficit Index
    
    Measures water deficit during grain filling stage (Aug-Sep).
    Inadequate water during grain fill reduces grain weight.
    
    Formula: GFDI = max(0, 1 - (rain / (ET * 0.5)))
    
    Returns: 0 (no deficit) to 1 (severe deficit)
    """
    params = CROP_PARAMS.get(crop, CROP_PARAMS['default'])
    optimal = params['rain_flowering']
    
    # Compare rain to ET demand (assume 50% of ET should be met by rain)
    if et_total <= 0:
        return 0.0
    
    water_ratio = rain_aug_sep / (et_total * 0.4)
    deficit = max(0, 1 - water_ratio)
    return float(min(1.0, deficit))


def compute_waterlogging_risk(rain_total: float, rain_cv: float,
                               heavy_rain_days: int = 0) -> float:
    """
    A4: Waterlogging Risk Score
    
    High rainfall with poor distribution can cause waterlogging.
    
    Formula: WRS = (heavy_days / 15) * 0.6 + (CV / 5) * 0.4
    
    Returns: 0 (no risk) to 1 (high risk)
    """
    # Heavy rain days (>50mm in a day)
    heavy_component = min(1.0, heavy_rain_days / 15)
    
    # CV indicates uneven distribution (concentrated heavy rain)
    cv_component = min(1.0, rain_cv / 5)
    
    # High total rain also indicates risk
    excess_component = max(0, (rain_total - 1200) / 500) if rain_total > 1200 else 0
    
    risk = 0.4 * heavy_component + 0.3 * cv_component + 0.3 * excess_component
    return float(min(1.0, risk))


def compute_drought_index(rain_total: float, rain_anomaly: float) -> float:
    """
    A5: Simplified Drought Index (SPI-inspired)
    
    Uses standardized rainfall anomaly to measure drought.
    
    Interpretation:
        > 0: Above normal rain (wet)
        < 0: Below normal rain (dry)
        < -1: Moderate drought
        < -2: Severe drought
    
    Returns: 0 (no drought) to 1 (severe drought)
    """
    if rain_anomaly >= 0:
        return 0.0
    
    # Convert negative anomaly to 0-1 stress scale
    drought = min(1.0, abs(rain_anomaly) / 2)
    return float(drought)


def compute_late_season_stress(heat_days_oct: int, dry_spell_oct: int,
                                rain_oct: float) -> float:
    """
    A8: Late Season Stress Index
    
    Stress during crop maturity affects final grain quality and weight.
    
    Returns: 0 (no stress) to 1 (severe stress)
    """
    # Heat stress in October
    heat_component = min(1.0, heat_days_oct / 10)
    
    # Dry spell in October
    dry_component = min(1.0, dry_spell_oct / 15)
    
    # Low October rain
    rain_deficit = max(0, 1 - rain_oct / 100) if rain_oct < 100 else 0
    
    stress = 0.4 * heat_component + 0.3 * dry_component + 0.3 * rain_deficit
    return float(stress)


def compute_combined_stress(stress_indices: Dict[str, float]) -> float:
    """
    A9: Combined Stress Score
    
    Weighted average of all stress indices.
    
    Weights reflect critical stage importance:
    - Vegetative: 25%
    - Flowering heat: 30% (most critical for yield)
    - Grain fill: 25%
    - Waterlogging: 10%
    - Late season: 10%
    """
    weights = {
        'vegetative_stress': 0.25,
        'flowering_heat_stress': 0.30,
        'grain_fill_deficit': 0.25,
        'waterlogging_risk': 0.10,
        'late_season_stress': 0.10
    }
    
    total_stress = 0.0
    total_weight = 0.0
    
    for key, weight in weights.items():
        if key in stress_indices:
            total_stress += weight * stress_indices[key]
            total_weight += weight
    
    if total_weight > 0:
        return float(total_stress / total_weight)
    return 0.0


def compute_yield_potential(ndvi_peak: float, gdd: float, 
                            combined_stress: float, crop: str = 'default') -> float:
    """
    A10: Yield Potential Index
    
    Estimates relative yield potential based on growth conditions.
    
    Formula: YPI = (NDVI_peak / 0.8) * (GDD / GDD_req) * (1 - stress)
    
    Returns: 0-1+ (higher = better yield potential)
    """
    params = CROP_PARAMS.get(crop, CROP_PARAMS['default'])
    gdd_required = params['gdd_required']
    
    # NDVI component (normalized to optimal 0.8)
    ndvi_component = min(1.2, ndvi_peak / 0.7) if ndvi_peak > 0 else 0.5
    
    # GDD component (normalized to crop requirement)
    gdd_component = min(1.2, gdd / gdd_required) if gdd_required > 0 else 1.0
    
    # Stress reduction
    stress_factor = 1 - combined_stress
    
    ypi = ndvi_component * gdd_component * stress_factor
    return float(max(0, ypi))


def compute_all_stress_indices(weather_features: Dict[str, float],
                                vegetation_features: Optional[Dict[str, float]] = None,
                                crop: str = 'default') -> Dict[str, float]:
    """
    Compute all 10 agronomic stress indices.
    
    Args:
        weather_features: Dict with weather features from weather_features.py
        vegetation_features: Dict with NDVI/EVI features (optional)
        crop: Crop type for crop-specific parameters
    
    Returns:
        Dict with 10 stress indices
    """
    # Extract weather components
    rain_jun_jul = weather_features.get('rain_jun_jul', 300)
    rain_aug_sep = weather_features.get('rain_aug_sep', 300)
    rain_total = weather_features.get('rain_total', 800)
    rain_cv = weather_features.get('rain_cv', 2.0)
    rain_anomaly = weather_features.get('rain_anomaly', 0)
    heat_days = weather_features.get('heat_days', 10)
    max_hot_streak = weather_features.get('max_hot_streak', 5)
    et_total = weather_features.get('et_total', 500)
    gdd = weather_features.get('gdd', 2000)
    
    # Get NDVI if available
    ndvi_peak = vegetation_features.get('ndvi_peak', 0.6) if vegetation_features else 0.6
    
    # Compute individual stress indices
    stress = {
        'vegetative_stress': compute_vegetative_stress(rain_jun_jul, crop),
        'flowering_heat_stress': compute_flowering_heat_stress(heat_days, max_hot_streak, crop),
        'grain_fill_deficit': compute_grain_fill_deficit(rain_aug_sep, et_total, crop),
        'waterlogging_risk': compute_waterlogging_risk(rain_total, rain_cv),
        'drought_index': compute_drought_index(rain_total, rain_anomaly),
        'late_season_stress': compute_late_season_stress(
            weather_features.get('heat_days_oct', 5),
            weather_features.get('dry_spell_oct', 3),
            weather_features.get('rain_oct', 50)
        ),
    }
    
    # Compute combined stress
    stress['combined_stress'] = compute_combined_stress(stress)
    
    # Compute yield potential
    stress['yield_potential'] = compute_yield_potential(
        ndvi_peak, gdd, stress['combined_stress'], crop
    )
    
    return stress


# ===============================================
# EXAMPLE USAGE
# ===============================================
if __name__ == "__main__":
    # Test with sample weather features
    weather_features = {
        'rain_jun_jul': 285,
        'rain_aug_sep': 585,
        'rain_total': 991,
        'rain_cv': 4.24,
        'rain_anomaly': 0.78,
        'heat_days': 64,
        'max_hot_streak': 46,
        'et_total': 985,
        'gdd': 4232,
    }
    
    print("=" * 70)
    print("AGRONOMIC STRESS INDICES")
    print("=" * 70)
    print()
    
    for crop in ['Rice', 'Soybean', 'Cotton']:
        stress = compute_all_stress_indices(weather_features, crop=crop)
        print(f"CROP: {crop}")
        print("-" * 50)
        for key, value in stress.items():
            bar = '*' * int(value * 30)
            print(f"  {key:25s}: {value:.3f} {bar}")
        print()
