"""
PMFBY v2.0 - Satellite Feature Engineering
===========================================
Fetches vegetation indices from Google Earth Engine.

Features:
- NDVI (Normalized Difference Vegetation Index)
- EVI (Enhanced Vegetation Index)
- NDWI (Normalized Difference Water Index)
- LSWI (Land Surface Water Index)

Data Sources:
- Sentinel-2 (10m resolution, optical)
- MODIS (250m resolution, 16-day composite)

Usage:
    from satellite_features import fetch_satellite_features
    
    features = fetch_satellite_features(
        lat=19.07, 
        lon=74.77, 
        year=2024,
        buffer_m=500
    )
"""

import ee
import numpy as np
from typing import Dict, Optional, List
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Initialize Earth Engine (requires authentication)
try:
    ee.Initialize()
    GEE_AVAILABLE = True
except Exception as e:
    print(f"[WARN] Google Earth Engine not initialized: {e}")
    print("       Run: earthengine authenticate")
    GEE_AVAILABLE = False


def fetch_sentinel2_indices(lat: float, lon: float, start_date: str, end_date: str, 
                            buffer_m: int = 500) -> Dict:
    """
    Fetch vegetation indices from Sentinel-2.
    
    Args:
        lat: Latitude
        lon: Longitude
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)
        buffer_m: Buffer radius in meters
    
    Returns:
        Dictionary with NDVI, EVI, NDWI time series
    """
    if not GEE_AVAILABLE:
        return get_default_satellite_features()
    
    try:
        # Define area of interest
        point = ee.Geometry.Point([lon, lat])
        aoi = point.buffer(buffer_m)
        
        # Sentinel-2 Surface Reflectance collection
        s2 = ee.ImageCollection('COPERNICUS/S2_SR') \
            .filterBounds(aoi) \
            .filterDate(start_date, end_date) \
            .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 30))
        
        # Function to compute indices
        def add_indices(image):
            # NDVI = (NIR - Red) / (NIR + Red)
            ndvi = image.normalizedDifference(['B8', 'B4']).rename('NDVI')
            
            # EVI = 2.5 * (NIR - Red) / (NIR + 6*Red - 7.5*Blue + 1)
            evi = image.expression(
                '2.5 * ((NIR - RED) / (NIR + 6 * RED - 7.5 * BLUE + 1))',
                {
                    'NIR': image.select('B8'),
                    'RED': image.select('B4'),
                    'BLUE': image.select('B2')
                }
            ).rename('EVI')
            
            # NDWI = (NIR - SWIR) / (NIR + SWIR)
            ndwi = image.normalizedDifference(['B8', 'B11']).rename('NDWI')
            
            # LSWI = (NIR - SWIR1) / (NIR + SWIR1)
            lswi = image.normalizedDifference(['B8', 'B11']).rename('LSWI')
            
            return image.addBands([ndvi, evi, ndwi, lswi])
        
        # Apply indices
        s2_indices = s2.map(add_indices)
        
        # Extract time series
        def extract_values(image):
            # Reduce to mean over AOI
            stats = image.select(['NDVI', 'EVI', 'NDWI', 'LSWI']) \
                .reduceRegion(
                    reducer=ee.Reducer.mean(),
                    geometry=aoi,
                    scale=10,
                    maxPixels=1e9
                )
            
            return ee.Feature(None, {
                'date': image.date().format('YYYY-MM-dd'),
                'ndvi': stats.get('NDVI'),
                'evi': stats.get('EVI'),
                'ndwi': stats.get('NDWI'),
                'lswi': stats.get('LSWI')
            })
        
        # Get time series
        time_series = s2_indices.map(extract_values).getInfo()
        
        # Parse results
        features = time_series.get('features', [])
        
        if not features:
            print(f"    [WARN] No Sentinel-2 data for {start_date} to {end_date}")
            return get_default_satellite_features()
        
        # Extract values
        ndvi_values = [f['properties']['ndvi'] for f in features if f['properties'].get('ndvi')]
        evi_values = [f['properties']['evi'] for f in features if f['properties'].get('evi')]
        ndwi_values = [f['properties']['ndwi'] for f in features if f['properties'].get('ndwi')]
        lswi_values = [f['properties']['lswi'] for f in features if f['properties'].get('lswi')]
        
        if not ndvi_values:
            return get_default_satellite_features()
        
        # Compute statistics
        result = {
            'ndvi_mean': float(np.mean(ndvi_values)),
            'ndvi_max': float(np.max(ndvi_values)),
            'ndvi_min': float(np.min(ndvi_values)),
            'ndvi_std': float(np.std(ndvi_values)),
            'ndvi_peak': float(np.percentile(ndvi_values, 90)),
            'ndvi_auc': float(np.trapz(ndvi_values)),  # Area under curve
            
            'evi_mean': float(np.mean(evi_values)) if evi_values else 0.4,
            'evi_peak': float(np.max(evi_values)) if evi_values else 0.5,
            
            'ndwi_mean': float(np.mean(ndwi_values)) if ndwi_values else 0.2,
            'lswi_mean': float(np.mean(lswi_values)) if lswi_values else 0.2,
            
            'n_observations': len(ndvi_values),
            'source': 'Sentinel-2'
        }
        
        return result
    
    except Exception as e:
        print(f"    [ERROR] Sentinel-2 fetch failed: {e}")
        return get_default_satellite_features()


def fetch_modis_ndvi(lat: float, lon: float, start_date: str, end_date: str) -> Dict:
    """
    Fetch NDVI from MODIS (faster, lower resolution).
    
    Args:
        lat: Latitude
        lon: Longitude
        start_date: Start date
        end_date: End date
    
    Returns:
        NDVI statistics
    """
    if not GEE_AVAILABLE:
        return get_default_satellite_features()
    
    try:
        point = ee.Geometry.Point([lon, lat])
        
        # MODIS NDVI (250m, 16-day)
        modis = ee.ImageCollection('MODIS/006/MOD13Q1') \
            .filterBounds(point) \
            .filterDate(start_date, end_date) \
            .select('NDVI')
        
        def extract_ndvi(image):
            # MODIS NDVI is scaled by 10000
            ndvi = image.select('NDVI').multiply(0.0001)
            
            value = ndvi.reduceRegion(
                reducer=ee.Reducer.mean(),
                geometry=point,
                scale=250
            ).get('NDVI')
            
            return ee.Feature(None, {
                'date': image.date().format('YYYY-MM-dd'),
                'ndvi': value
            })
        
        time_series = modis.map(extract_ndvi).getInfo()
        features = time_series.get('features', [])
        
        ndvi_values = [f['properties']['ndvi'] for f in features 
                      if f['properties'].get('ndvi') is not None]
        
        if not ndvi_values:
            return get_default_satellite_features()
        
        return {
            'ndvi_mean': float(np.mean(ndvi_values)),
            'ndvi_max': float(np.max(ndvi_values)),
            'ndvi_peak': float(np.percentile(ndvi_values, 90)),
            'ndvi_auc': float(np.trapz(ndvi_values)),
            'n_observations': len(ndvi_values),
            'source': 'MODIS'
        }
    
    except Exception as e:
        print(f"    [ERROR] MODIS fetch failed: {e}")
        return get_default_satellite_features()


def fetch_satellite_features(lat: float, lon: float, year: int, 
                             season: str = 'Kharif',
                             use_sentinel: bool = True) -> Dict:
    """
    Main function to fetch satellite features.
    
    Args:
        lat: Latitude
        lon: Longitude
        year: Year
        season: Season (Kharif/Rabi/Summer)
        use_sentinel: Use Sentinel-2 (True) or MODIS (False)
    
    Returns:
        Satellite features dictionary
    """
    # Determine date range based on season
    if season.lower() == 'kharif':
        start_date = f"{year}-06-01"
        end_date = f"{year}-11-30"
    elif season.lower() == 'rabi':
        start_date = f"{year}-11-01"
        end_date = f"{year+1}-03-31"
    else:  # Summer
        start_date = f"{year}-03-01"
        end_date = f"{year}-06-30"
    
    print(f"    Fetching satellite data ({start_date} to {end_date})...")
    
    if use_sentinel:
        features = fetch_sentinel2_indices(lat, lon, start_date, end_date)
    else:
        features = fetch_modis_ndvi(lat, lon, start_date, end_date)
    
    print(f"    [OK] Satellite data: {features['source']}, "
          f"NDVI={features['ndvi_mean']:.2f}, "
          f"Obs={features['n_observations']}")
    
    return features


def get_default_satellite_features() -> Dict:
    """
    Return default satellite features (fallback).
    
    Based on typical Kharif crop values for Maharashtra.
    """
    return {
        'ndvi_mean': 0.55,
        'ndvi_max': 0.70,
        'ndvi_min': 0.35,
        'ndvi_std': 0.12,
        'ndvi_peak': 0.68,
        'ndvi_auc': 30.0,
        'evi_mean': 0.42,
        'evi_peak': 0.52,
        'ndwi_mean': 0.20,
        'lswi_mean': 0.18,
        'n_observations': 0,
        'source': 'Default'
    }


# ===============================================
# EXAMPLE USAGE
# ===============================================

if __name__ == "__main__":
    print("=" * 70)
    print("SATELLITE FEATURE ENGINEERING TEST")
    print("=" * 70)
    
    # Test location
    lat, lon = 19.071591, 74.774179
    year = 2024
    season = 'Kharif'
    
    print(f"\nLocation: {lat}N, {lon}E")
    print(f"Season: {season} {year}")
    print()
    
    if GEE_AVAILABLE:
        # Try Sentinel-2
        print("[1] Fetching Sentinel-2 data...")
        s2_features = fetch_satellite_features(
            lat, lon, year, season, use_sentinel=True
        )
        
        print("\nSentinel-2 Features:")
        print("-" * 50)
        for key, value in s2_features.items():
            if isinstance(value, float):
                print(f"  {key:20s}: {value:.4f}")
            else:
                print(f"  {key:20s}: {value}")
        
        # Try MODIS (faster)
        print("\n[2] Fetching MODIS data...")
        modis_features = fetch_satellite_features(
            lat, lon, year, season, use_sentinel=False
        )
        
        print("\nMODIS Features:")
        print("-" * 50)
        for key, value in modis_features.items():
            if isinstance(value, float):
                print(f"  {key:20s}: {value:.4f}")
            else:
                print(f"  {key:20s}: {value}")
    else:
        print("[WARN] GEE not available. Using defaults.")
        features = get_default_satellite_features()
        
        print("\nDefault Features:")
        print("-" * 50)
        for key, value in features.items():
            if isinstance(value, float):
                print(f"  {key:20s}: {value:.4f}")
            else:
                print(f"  {key:20s}: {value}")
    
    print("\n" + "=" * 70)
    print("TEST COMPLETE")
    print("=" * 70)
