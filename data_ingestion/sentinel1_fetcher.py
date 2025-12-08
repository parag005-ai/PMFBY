"""
PMFBY Yield Prediction Engine
Sentinel-1 SAR Data Fetcher Module

Automates ingestion of Sentinel-1 SAR (Synthetic Aperture Radar) data
for biomass estimation and cloud-free monitoring.
"""

import ee
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Sentinel1Fetcher:
    """
    Sentinel-1 SAR data fetcher for PMFBY yield prediction.
    
    SAR data advantages:
    - Cloud-penetrating (works in all weather)
    - Sensitive to crop structure and moisture
    - VH/VV ratio correlates with biomass
    """
    
    def __init__(self, project_id: Optional[str] = None):
        """Initialize Sentinel-1 fetcher."""
        self.initialized = False
        self.project_id = project_id
        self._initialize_gee()
        
    def _initialize_gee(self):
        """Initialize Google Earth Engine."""
        try:
            if self.project_id:
                ee.Initialize(project=self.project_id)
            else:
                ee.Initialize()
            self.initialized = True
            logger.info("GEE initialized for Sentinel-1")
        except Exception as e:
            logger.warning(f"GEE initialization failed: {e}")
            self.initialized = False
    
    def _preprocess_sar(self, image: ee.Image) -> ee.Image:
        """
        Preprocess SAR image.
        - Apply speckle filtering
        - Convert to dB scale
        - Add derived indices
        """
        # Speckle filter (focal mean)
        vv_filtered = image.select('VV').focal_mean(radius=30, units='meters')
        vh_filtered = image.select('VH').focal_mean(radius=30, units='meters')
        
        # VH/VV ratio (biomass proxy)
        vh_vv_ratio = vh_filtered.divide(vv_filtered).rename('VH_VV_ratio')
        
        # RVI (Radar Vegetation Index)
        # RVI = 4 * VH / (VV + VH)
        rvi = vh_filtered.multiply(4).divide(
            vv_filtered.add(vh_filtered)
        ).rename('RVI')
        
        return (image
                .addBands(vv_filtered.rename('VV_filtered'))
                .addBands(vh_filtered.rename('VH_filtered'))
                .addBands(vh_vv_ratio)
                .addBands(rvi))
    
    def fetch_time_series(
        self,
        geometry: Dict,
        start_date: str,
        end_date: str,
        orbit: str = "DESCENDING"
    ) -> pd.DataFrame:
        """
        Fetch Sentinel-1 SAR time series.
        
        Args:
            geometry: GeoJSON geometry (Polygon)
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            orbit: Orbit direction (ASCENDING or DESCENDING)
            
        Returns:
            DataFrame with SAR backscatter values
        """
        if not self.initialized:
            logger.warning("GEE not initialized. Returning synthetic SAR data.")
            return self._generate_synthetic_sar(start_date, end_date)
        
        try:
            ee_geometry = ee.Geometry(geometry)
            
            # Load Sentinel-1 collection
            collection = (ee.ImageCollection('COPERNICUS/S1_GRD')
                         .filterBounds(ee_geometry)
                         .filterDate(start_date, end_date)
                         .filter(ee.Filter.eq('instrumentMode', 'IW'))
                         .filter(ee.Filter.eq('orbitProperties_pass', orbit))
                         .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VV'))
                         .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VH')))
            
            # Apply preprocessing
            processed = collection.map(self._preprocess_sar)
            
            # Extract mean values
            def extract_values(image):
                date = ee.Date(image.get('system:time_start'))
                stats = image.select(['VV', 'VH', 'VH_VV_ratio', 'RVI']).reduceRegion(
                    reducer=ee.Reducer.mean(),
                    geometry=ee_geometry,
                    scale=10,
                    maxPixels=1e9
                )
                return ee.Feature(None, stats).set('date', date.format('YYYY-MM-dd'))
            
            features = processed.map(extract_values)
            data = features.getInfo()['features']
            
            # Convert to DataFrame
            records = []
            for feature in data:
                props = feature['properties']
                records.append({
                    'date': props.get('date'),
                    'vv': props.get('VV'),
                    'vh': props.get('VH'),
                    'vh_vv_ratio': props.get('VH_VV_ratio'),
                    'rvi': props.get('RVI')
                })
            
            df = pd.DataFrame(records)
            if not df.empty:
                df['date'] = pd.to_datetime(df['date'])
                df = df.sort_values('date').reset_index(drop=True)
                
            logger.info(f"Fetched {len(df)} Sentinel-1 observations")
            return df
            
        except Exception as e:
            logger.error(f"Error fetching Sentinel-1 data: {e}")
            return self._generate_synthetic_sar(start_date, end_date)
    
    def _generate_synthetic_sar(
        self,
        start_date: str,
        end_date: str
    ) -> pd.DataFrame:
        """Generate synthetic SAR time series for testing."""
        start = pd.to_datetime(start_date)
        end = pd.to_datetime(end_date)
        dates = pd.date_range(start, end, freq='12D')  # 12-day revisit
        
        n = len(dates)
        days = np.arange(n) * 12
        
        # SAR backscatter pattern (increases with biomass)
        peak_day = 80
        
        # VV backscatter (-15 to -8 dB range)
        vv_base = -12
        vv_increase = 4 * np.sin(np.pi * np.clip(days, 0, peak_day*2) / (peak_day*2))
        vv = vv_base + vv_increase + np.random.normal(0, 0.5, n)
        
        # VH backscatter (-22 to -14 dB range)
        vh_base = -18
        vh_increase = 4 * np.sin(np.pi * np.clip(days, 0, peak_day*2) / (peak_day*2))
        vh = vh_base + vh_increase + np.random.normal(0, 0.5, n)
        
        # Derived indices
        vh_vv_ratio = vh - vv  # In dB domain
        rvi = 4 * np.power(10, vh/10) / (np.power(10, vv/10) + np.power(10, vh/10))
        
        df = pd.DataFrame({
            'date': dates,
            'vv': vv,
            'vh': vh,
            'vh_vv_ratio': vh_vv_ratio,
            'rvi': np.clip(rvi, 0, 1)
        })
        
        logger.info(f"Generated {len(df)} synthetic SAR observations")
        return df


def main():
    """Test the Sentinel-1 fetcher."""
    fetcher = Sentinel1Fetcher()
    
    test_geometry = {
        "type": "Polygon",
        "coordinates": [[[76.95, 29.68], [76.98, 29.68],
                        [76.98, 29.71], [76.95, 29.71],
                        [76.95, 29.68]]]
    }
    
    df = fetcher.fetch_time_series(
        geometry=test_geometry,
        start_date="2024-06-01",
        end_date="2024-11-30"
    )
    
    print("\n=== Sentinel-1 SAR Time Series ===")
    print(df.head(20))
    print(f"\nTotal observations: {len(df)}")
    print(f"VV range: {df['vv'].min():.2f} to {df['vv'].max():.2f} dB")
    print(f"VH range: {df['vh'].min():.2f} to {df['vh'].max():.2f} dB")
    
    return df


if __name__ == "__main__":
    main()
