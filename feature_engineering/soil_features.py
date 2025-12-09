"""
PMFBY v2.0 - Soil Feature Engineering
======================================
Fetches soil properties from SoilGrids API (250m resolution).

Features:
- Clay content (%)
- Sand content (%)
- Organic carbon (%)
- pH
- Available Water Capacity (AWC)
- Soil texture classification

Data Source: ISRIC SoilGrids v2.0
Resolution: 250m
Coverage: Global
"""

import requests
import numpy as np
from typing import Dict, Optional
import warnings
warnings.filterwarnings('ignore')

# Default soil properties for Maharashtra
DEFAULT_SOIL = {
    'clay_pct': 35.0,
    'sand_pct': 30.0,
    'silt_pct': 35.0,
    'organic_carbon': 0.6,
    'ph': 7.5,
    'texture_class': 'Clay Loam',
    'awc': 0.15,
    'bulk_density': 1.35
}


def fetch_soil_properties(lat: float, lon: float, use_cache: bool = True) -> Dict:
    """
    Fetch soil properties from SoilGrids API.
    
    Args:
        lat: Latitude
        lon: Longitude
        use_cache: Use cached data if available
    
    Returns:
        Dictionary with soil properties
    """
    try:
        # SoilGrids API endpoint
        url = "https://rest.isric.org/soilgrids/v2.0/properties/query"
        
        params = {
            'lon': lon,
            'lat': lat,
            'property': ['clay', 'sand', 'soc', 'phh2o', 'bdod'],
            'depth': '0-30cm',
            'value': 'mean'
        }
        
        print(f"    Fetching soil data from SoilGrids...")
        response = requests.get(url, params=params, timeout=30)
        
        if response.status_code != 200:
            print(f"    [WARN] SoilGrids API returned {response.status_code}")
            return get_default_soil()
        
        data = response.json()
        
        # Extract properties
        properties = data.get('properties', {}).get('layers', [])
        
        if not properties:
            print("    [WARN] No soil data returned. Using defaults.")
            return get_default_soil()
        
        # Parse values (SoilGrids returns in specific units)
        clay = None
        sand = None
        soc = None
        ph = None
        bdod = None
        
        for layer in properties:
            prop_name = layer.get('name')
            depths = layer.get('depths', [])
            
            if depths:
                value = depths[0].get('values', {}).get('mean')
                
                if prop_name == 'clay':
                    clay = value / 10  # Convert g/kg to %
                elif prop_name == 'sand':
                    sand = value / 10
                elif prop_name == 'soc':
                    soc = value / 10  # Convert dg/kg to %
                elif prop_name == 'phh2o':
                    ph = value / 10  # pH is stored * 10
                elif prop_name == 'bdod':
                    bdod = value / 100  # Convert cg/cm³ to g/cm³
        
        # Validate data
        if clay is None or sand is None:
            print("    [WARN] Incomplete soil data. Using defaults.")
            return get_default_soil()
        
        # Compute derived properties
        silt = max(0, 100 - clay - sand)
        
        # Soil texture classification
        texture = classify_texture(clay, sand, silt)
        
        # Available Water Capacity
        awc = compute_awc(clay, sand, soc if soc else 0.6)
        
        soil_data = {
            'clay_pct': float(clay),
            'sand_pct': float(sand),
            'silt_pct': float(silt),
            'organic_carbon': float(soc) if soc else 0.6,
            'ph': float(ph) if ph else 7.5,
            'texture_class': texture,
            'awc': float(awc),
            'bulk_density': float(bdod) if bdod else 1.35
        }
        
        print(f"    [OK] Soil data fetched: {texture}, pH={soil_data['ph']:.1f}")
        return soil_data
    
    except requests.exceptions.Timeout:
        print("    [WARN] SoilGrids API timeout. Using defaults.")
        return get_default_soil()
    
    except Exception as e:
        print(f"    [WARN] Soil fetch failed: {e}. Using defaults.")
        return get_default_soil()


def classify_texture(clay: float, sand: float, silt: float) -> str:
    """
    Classify soil texture using USDA soil texture triangle.
    
    Args:
        clay: Clay percentage
        sand: Sand percentage
        silt: Silt percentage
    
    Returns:
        Soil texture class name
    """
    # USDA Soil Texture Triangle classification
    if clay >= 40:
        if sand <= 45:
            return 'Clay'
        else:
            return 'Sandy Clay'
    
    elif clay >= 35:
        if sand >= 45:
            return 'Sandy Clay Loam'
        else:
            return 'Clay Loam'
    
    elif clay >= 27:
        if sand <= 20:
            return 'Silty Clay Loam'
        elif sand <= 45:
            return 'Clay Loam'
        else:
            return 'Sandy Clay Loam'
    
    elif clay >= 20:
        if sand >= 45:
            return 'Sandy Loam'
        elif silt >= 50:
            return 'Silt Loam'
        else:
            return 'Loam'
    
    elif clay >= 7:
        if sand >= 52:
            return 'Sandy Loam'
        elif silt >= 50:
            return 'Silt Loam'
        else:
            return 'Loam'
    
    else:
        if sand >= 85:
            return 'Sand'
        elif sand >= 70:
            return 'Loamy Sand'
        else:
            return 'Sandy Loam'


def compute_awc(clay: float, sand: float, organic_carbon: float) -> float:
    """
    Compute Available Water Capacity (AWC) using Saxton equation.
    
    AWC = Field Capacity - Permanent Wilting Point
    
    Args:
        clay: Clay percentage
        sand: Sand percentage
        organic_carbon: Organic carbon percentage
    
    Returns:
        AWC in cm³/cm³ (volumetric)
    """
    # Simplified Saxton and Rawls (2006) equations
    
    # Field Capacity (θ at -33 kPa)
    fc = (
        0.2576 
        - 0.002 * sand 
        + 0.0036 * clay 
        + 0.0299 * organic_carbon
    )
    
    # Permanent Wilting Point (θ at -1500 kPa)
    pwp = (
        0.026 
        + 0.005 * clay 
        + 0.0158 * organic_carbon
    )
    
    # AWC
    awc = fc - pwp
    
    # Ensure positive and reasonable
    awc = max(0.05, min(0.30, awc))
    
    return float(awc)


def get_default_soil() -> Dict:
    """
    Return default soil properties for Maharashtra.
    
    Based on typical Medium Black Soil characteristics.
    """
    return DEFAULT_SOIL.copy()


def compute_soil_quality_index(soil: Dict) -> float:
    """
    Compute overall soil quality index (0-1).
    
    Higher values indicate better soil for crop production.
    
    Args:
        soil: Soil properties dictionary
    
    Returns:
        Soil quality index (0-1)
    """
    # Optimal ranges for crop production
    optimal_clay = 30  # %
    optimal_ph = 6.5
    optimal_oc = 1.0  # %
    optimal_awc = 0.18
    
    # Compute sub-scores
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
    print("SOIL FEATURE ENGINEERING TEST")
    print("=" * 70)
    
    # Test locations
    locations = [
        (19.071591, 74.774179, "Ahmednagar"),
        (18.52, 73.86, "Pune"),
        (21.15, 79.09, "Nagpur")
    ]
    
    for lat, lon, name in locations:
        print(f"\n{name} ({lat}, {lon}):")
        print("-" * 50)
        
        soil = fetch_soil_properties(lat, lon)
        quality = compute_soil_quality_index(soil)
        
        print(f"  Texture: {soil['texture_class']}")
        print(f"  Clay: {soil['clay_pct']:.1f}%")
        print(f"  Sand: {soil['sand_pct']:.1f}%")
        print(f"  Silt: {soil['silt_pct']:.1f}%")
        print(f"  Organic Carbon: {soil['organic_carbon']:.2f}%")
        print(f"  pH: {soil['ph']:.1f}")
        print(f"  AWC: {soil['awc']:.3f} cm³/cm³")
        print(f"  Bulk Density: {soil['bulk_density']:.2f} g/cm³")
        print(f"  Quality Index: {quality:.2f}")
    
    print("\n" + "=" * 70)
    print("TEST COMPLETE")
    print("=" * 70)
