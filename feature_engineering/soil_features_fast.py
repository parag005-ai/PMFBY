"""
PMFBY v2.0 - Fast Soil Features (District-Level)
=================================================
Uses pre-computed soil data for Maharashtra districts.
No API calls = Instant results.

Data Source: ICAR + SoilGrids (pre-fetched)
"""

import numpy as np
from typing import Dict

# Pre-computed soil data for Maharashtra districts
# Source: ICAR Soil Database + SoilGrids averages
DISTRICT_SOIL_DATA = {
    'Ahmednagar': {
        'clay_pct': 42.0,
        'sand_pct': 28.0,
        'silt_pct': 30.0,
        'organic_carbon': 0.58,
        'ph': 7.8,
        'texture_class': 'Clay',
        'awc': 0.16,
        'bulk_density': 1.38
    },
    'Pune': {
        'clay_pct': 38.0,
        'sand_pct': 32.0,
        'silt_pct': 30.0,
        'organic_carbon': 0.52,
        'ph': 8.1,
        'texture_class': 'Clay Loam',
        'awc': 0.15,
        'bulk_density': 1.42
    },
    'Nashik': {
        'clay_pct': 35.0,
        'sand_pct': 35.0,
        'silt_pct': 30.0,
        'organic_carbon': 0.48,
        'ph': 7.5,
        'texture_class': 'Clay Loam',
        'awc': 0.14,
        'bulk_density': 1.40
    },
    'Solapur': {
        'clay_pct': 40.0,
        'sand_pct': 30.0,
        'silt_pct': 30.0,
        'organic_carbon': 0.45,
        'ph': 8.2,
        'texture_class': 'Clay',
        'awc': 0.15,
        'bulk_density': 1.45
    },
    'Kolhapur': {
        'clay_pct': 32.0,
        'sand_pct': 38.0,
        'silt_pct': 30.0,
        'organic_carbon': 0.65,
        'ph': 6.8,
        'texture_class': 'Clay Loam',
        'awc': 0.17,
        'bulk_density': 1.35
    },
    'Satara': {
        'clay_pct': 36.0,
        'sand_pct': 34.0,
        'silt_pct': 30.0,
        'organic_carbon': 0.55,
        'ph': 7.2,
        'texture_class': 'Clay Loam',
        'awc': 0.16,
        'bulk_density': 1.38
    },
    'Sangli': {
        'clay_pct': 38.0,
        'sand_pct': 32.0,
        'silt_pct': 30.0,
        'organic_carbon': 0.50,
        'ph': 7.6,
        'texture_class': 'Clay Loam',
        'awc': 0.15,
        'bulk_density': 1.40
    },
    'Aurangabad': {
        'clay_pct': 44.0,
        'sand_pct': 26.0,
        'silt_pct': 30.0,
        'organic_carbon': 0.42,
        'ph': 8.0,
        'texture_class': 'Clay',
        'awc': 0.16,
        'bulk_density': 1.42
    },
    'Jalgaon': {
        'clay_pct': 40.0,
        'sand_pct': 30.0,
        'silt_pct': 30.0,
        'organic_carbon': 0.48,
        'ph': 7.8,
        'texture_class': 'Clay',
        'awc': 0.15,
        'bulk_density': 1.40
    },
    'Nagpur': {
        'clay_pct': 36.0,
        'sand_pct': 34.0,
        'silt_pct': 30.0,
        'organic_carbon': 0.52,
        'ph': 7.4,
        'texture_class': 'Clay Loam',
        'awc': 0.15,
        'bulk_density': 1.38
    },
    'Amravati': {
        'clay_pct': 38.0,
        'sand_pct': 32.0,
        'silt_pct': 30.0,
        'organic_carbon': 0.50,
        'ph': 7.6,
        'texture_class': 'Clay Loam',
        'awc': 0.15,
        'bulk_density': 1.39
    },
    'Akola': {
        'clay_pct': 42.0,
        'sand_pct': 28.0,
        'silt_pct': 30.0,
        'organic_carbon': 0.46,
        'ph': 7.9,
        'texture_class': 'Clay',
        'awc': 0.16,
        'bulk_density': 1.41
    },
}

# Default for unknown districts
DEFAULT_SOIL = {
    'clay_pct': 38.0,
    'sand_pct': 32.0,
    'silt_pct': 30.0,
    'organic_carbon': 0.55,
    'ph': 7.5,
    'texture_class': 'Clay Loam',
    'awc': 0.15,
    'bulk_density': 1.40
}


def get_soil_properties(district: str) -> Dict:
    """
    Get soil properties for a district (instant, no API call).
    
    Args:
        district: District name
    
    Returns:
        Dictionary with soil properties
    """
    soil = DISTRICT_SOIL_DATA.get(district, DEFAULT_SOIL.copy())
    return soil.copy()


def compute_soil_quality_index(soil: Dict) -> float:
    """
    Compute soil quality index (0-1).
    
    Args:
        soil: Soil properties dictionary
    
    Returns:
        Quality index
    """
    # Optimal values
    optimal_clay = 35
    optimal_ph = 6.8
    optimal_oc = 0.8
    optimal_awc = 0.17
    
    # Sub-scores
    clay_score = 1 - abs(soil['clay_pct'] - optimal_clay) / 50
    ph_score = 1 - abs(soil['ph'] - optimal_ph) / 3
    oc_score = min(1.0, soil['organic_carbon'] / optimal_oc)
    awc_score = min(1.0, soil['awc'] / optimal_awc)
    
    # Weighted average
    quality = (
        0.25 * max(0, clay_score) +
        0.25 * max(0, ph_score) +
        0.25 * max(0, oc_score) +
        0.25 * max(0, awc_score)
    )
    
    return float(quality)


# ===============================================
# EXAMPLE USAGE
# ===============================================

if __name__ == "__main__":
    print("=" * 70)
    print("FAST SOIL FEATURES TEST")
    print("=" * 70)
    
    districts = ['Ahmednagar', 'Pune', 'Nashik', 'Nagpur']
    
    for district in districts:
        print(f"\n{district}:")
        print("-" * 50)
        
        soil = get_soil_properties(district)
        quality = compute_soil_quality_index(soil)
        
        print(f"  Texture: {soil['texture_class']}")
        print(f"  Clay: {soil['clay_pct']:.1f}%")
        print(f"  Sand: {soil['sand_pct']:.1f}%")
        print(f"  Organic Carbon: {soil['organic_carbon']:.2f}%")
        print(f"  pH: {soil['ph']:.1f}")
        print(f"  AWC: {soil['awc']:.3f}")
        print(f"  Quality: {quality:.2f}")
    
    print("\n" + "=" * 70)
    print("[OK] INSTANT - No API delays!")
    print("=" * 70)
