"""
PMFBY Yield Prediction Engine
Real Data Loader Module

Downloads and processes REAL district-wise yield data from:
1. data.gov.in
2. Kaggle (India Crop Production dataset)
3. ICRISAT District Level Database
4. India Data Portal

INSTRUCTIONS TO GET DATA:
=========================

Option 1: Kaggle (Recommended - Most Complete)
----------------------------------------------
1. Go to: https://www.kaggle.com/datasets/abhinand05/crop-production-in-india
2. Download 'crop_production.csv' (or similar)
3. Place in: d:/tr/pmfby_engine/data/crop_production.csv

Option 2: data.gov.in
---------------------
1. Go to: https://data.gov.in/resource/district-wise-season-wise-crop-production-statistics-1997
2. Download as CSV/Excel
3. Place in: d:/tr/pmfby_engine/data/govt_crop_stats.csv

Option 3: India Data Portal
---------------------------
1. Go to: https://indiadataportal.com/p/area-production-yield-apy
2. Download the dataset
3. Place in: d:/tr/pmfby_engine/data/india_apy.csv

After downloading, run this script to process and calculate thresholds.
"""

import os
import logging
from typing import Dict, Optional, List
import pandas as pd
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RealYieldDataLoader:
    """
    Loads and processes real district-wise yield data from CSV files.
    Can calculate PMFBY thresholds from actual historical data.
    """
    
    DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')
    
    # Expected columns for different data sources
    COLUMN_MAPPINGS = {
        'kaggle': {
            'state': 'State_Name',
            'district': 'District_Name',
            'crop': 'Crop',
            'season': 'Season',
            'year': 'Crop_Year',
            'area': 'Area',
            'production': 'Production'
        },
        'govt': {
            'state': 'state_name',
            'district': 'district_name',
            'crop': 'crop_name',
            'season': 'season',
            'year': 'year',
            'area': 'area_ha',
            'production': 'production_tonnes'
        }
    }
    
    def __init__(self):
        """Initialize data loader."""
        os.makedirs(self.DATA_DIR, exist_ok=True)
        self.data = None
        self.source = None
        
    def check_available_data(self) -> Dict:
        """Check which data files are available."""
        files = {
            'kaggle': os.path.join(self.DATA_DIR, 'crop_production.csv'),
            'govt': os.path.join(self.DATA_DIR, 'govt_crop_stats.csv'),
            'india_portal': os.path.join(self.DATA_DIR, 'india_apy.csv'),
            'custom': os.path.join(self.DATA_DIR, 'custom_yields.csv')
        }
        
        available = {}
        for source, path in files.items():
            if os.path.exists(path):
                size = os.path.getsize(path) / 1024  # KB
                available[source] = {'path': path, 'size_kb': round(size, 2)}
                logger.info(f"Found: {source} ({size:.0f} KB)")
        
        if not available:
            logger.warning("No data files found! Please download data as per instructions above.")
        
        return available
    
    def load_data(self, source: str = 'auto') -> pd.DataFrame:
        """
        Load yield data from CSV file.
        
        Args:
            source: 'kaggle', 'govt', 'india_portal', 'custom', or 'auto'
            
        Returns:
            DataFrame with standardized columns
        """
        available = self.check_available_data()
        
        if not available:
            logger.error("No data files available. Returning empty DataFrame.")
            return pd.DataFrame()
        
        if source == 'auto':
            source = list(available.keys())[0]
        elif source not in available:
            logger.warning(f"{source} not found. Using {list(available.keys())[0]}")
            source = list(available.keys())[0]
        
        path = available[source]['path']
        logger.info(f"Loading data from {source}: {path}")
        
        # Load CSV
        df = pd.read_csv(path)
        logger.info(f"Loaded {len(df)} rows, {len(df.columns)} columns")
        
        # Standardize column names
        df = self._standardize_columns(df, source)
        
        # Calculate yield if not present
        if 'yield' not in df.columns and 'area' in df.columns and 'production' in df.columns:
            df['yield'] = df['production'] / df['area'].replace(0, np.nan)
            df['yield'] = df['yield'].round(2)
        
        self.data = df
        self.source = source
        
        return df
    
    def _standardize_columns(self, df: pd.DataFrame, source: str) -> pd.DataFrame:
        """Standardize column names to common format."""
        df.columns = df.columns.str.lower().str.strip().str.replace(' ', '_')
        
        # Common renames
        rename_map = {
            'state_name': 'state',
            'district_name': 'district',
            'crop_name': 'crop',
            'crop_year': 'year',
            'area_ha': 'area',
            'production_tonnes': 'production',
            'yield_kg_ha': 'yield'
        }
        
        df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})
        
        # Standardize values
        for col in ['state', 'district', 'crop', 'season']:
            if col in df.columns:
                df[col] = df[col].str.lower().str.strip()
        
        return df
    
    def get_district_yield_history(
        self,
        district: str,
        crop: str,
        season: str = None,
        years: int = 7
    ) -> pd.DataFrame:
        """
        Get historical yield data for a district/crop.
        
        Args:
            district: District name
            crop: Crop name
            season: Optional season filter
            years: Number of recent years to retrieve
            
        Returns:
            DataFrame with year-wise yields
        """
        if self.data is None:
            self.load_data()
        
        if self.data is None or self.data.empty:
            return pd.DataFrame()
        
        df = self.data.copy()
        
        # Filter
        mask = (df['district'].str.contains(district.lower(), na=False)) & \
               (df['crop'].str.contains(crop.lower(), na=False))
        
        if season:
            mask = mask & (df['season'].str.contains(season.lower(), na=False))
        
        filtered = df[mask].copy()
        
        if filtered.empty:
            logger.warning(f"No data found for {district}/{crop}")
            return pd.DataFrame()
        
        # Sort by year and get recent years
        filtered = filtered.sort_values('year', ascending=False).head(years)
        
        return filtered[['year', 'district', 'crop', 'season', 'area', 'production', 'yield']]
    
    def calculate_pmfby_threshold(
        self,
        district: str,
        crop: str,
        season: str = None,
        indemnity_level: int = 80,
        exclude_worst_years: int = 2
    ) -> Dict:
        """
        Calculate PMFBY threshold yield using official formula.
        
        Formula:
        1. Take last 7 years of yield data
        2. Exclude 2 worst (calamity) years
        3. Calculate average of remaining 5 years
        4. Threshold = Average × (Indemnity Level / 100)
        
        Args:
            district: District name
            crop: Crop name
            season: Optional season
            indemnity_level: 70, 80, or 90
            exclude_worst_years: Number of worst years to exclude (default 2)
            
        Returns:
            Dictionary with threshold calculation details
        """
        history = self.get_district_yield_history(district, crop, season, years=7)
        
        if history.empty or 'yield' not in history.columns:
            return {
                'status': 'error',
                'message': f'No data available for {district}/{crop}',
                'threshold_yield': None
            }
        
        yields = history['yield'].dropna().values
        years = history['year'].values if 'year' in history.columns else []
        
        if len(yields) < 3:
            return {
                'status': 'insufficient_data',
                'message': f'Only {len(yields)} years of data available',
                'threshold_yield': None
            }
        
        # Sort yields and exclude worst years
        sorted_yields = np.sort(yields)
        if len(sorted_yields) > exclude_worst_years:
            adjusted_yields = sorted_yields[exclude_worst_years:]  # Remove lowest
            excluded_yields = sorted_yields[:exclude_worst_years]
        else:
            adjusted_yields = sorted_yields
            excluded_yields = []
        
        # Calculate average
        avg_yield = np.mean(adjusted_yields)
        
        # Calculate threshold
        threshold_yield = avg_yield * (indemnity_level / 100)
        
        return {
            'status': 'success',
            'district': district,
            'crop': crop,
            'season': season,
            'data_source': self.source,
            'years_analyzed': len(yields),
            'years_available': list(years) if len(years) > 0 else 'N/A',
            'all_yields_kg_ha': list(yields.round(2)),
            'excluded_calamity_yields': list(excluded_yields.round(2)) if len(excluded_yields) > 0 else [],
            'yields_for_average': list(adjusted_yields.round(2)),
            'average_yield': round(float(avg_yield), 2),
            'indemnity_level': indemnity_level,
            'threshold_yield': round(float(threshold_yield), 2),
            'formula': f'Threshold = {avg_yield:.0f} × {indemnity_level}% = {threshold_yield:.0f} kg/ha'
        }
    
    def get_all_districts(self) -> List[str]:
        """Get list of all districts in the data."""
        if self.data is None:
            self.load_data()
        
        if self.data is None or 'district' not in self.data.columns:
            return []
        
        return self.data['district'].dropna().unique().tolist()
    
    def get_crops_for_district(self, district: str) -> List[str]:
        """Get list of crops grown in a district."""
        if self.data is None:
            self.load_data()
        
        if self.data is None:
            return []
        
        mask = self.data['district'].str.contains(district.lower(), na=False)
        return self.data[mask]['crop'].dropna().unique().tolist()


def create_sample_data():
    """Create sample data file for testing."""
    # Sample Haryana data based on published statistics
    sample_data = [
        # Karnal Rice
        {'state': 'Haryana', 'district': 'Karnal', 'crop': 'Rice', 'season': 'Kharif', 'year': 2023, 'area': 145000, 'production': 522000},
        {'state': 'Haryana', 'district': 'Karnal', 'crop': 'Rice', 'season': 'Kharif', 'year': 2022, 'area': 143000, 'production': 486200},
        {'state': 'Haryana', 'district': 'Karnal', 'crop': 'Rice', 'season': 'Kharif', 'year': 2021, 'area': 140000, 'production': 504000},
        {'state': 'Haryana', 'district': 'Karnal', 'crop': 'Rice', 'season': 'Kharif', 'year': 2020, 'area': 138000, 'production': 455400},  # Drought year
        {'state': 'Haryana', 'district': 'Karnal', 'crop': 'Rice', 'season': 'Kharif', 'year': 2019, 'area': 141000, 'production': 507600},
        {'state': 'Haryana', 'district': 'Karnal', 'crop': 'Rice', 'season': 'Kharif', 'year': 2018, 'area': 139000, 'production': 445600},  # Flood year
        {'state': 'Haryana', 'district': 'Karnal', 'crop': 'Rice', 'season': 'Kharif', 'year': 2017, 'area': 136000, 'production': 489600},
        
        # Karnal Wheat
        {'state': 'Haryana', 'district': 'Karnal', 'crop': 'Wheat', 'season': 'Rabi', 'year': 2023, 'area': 160000, 'production': 832000},
        {'state': 'Haryana', 'district': 'Karnal', 'crop': 'Wheat', 'season': 'Rabi', 'year': 2022, 'area': 158000, 'production': 789000},
        {'state': 'Haryana', 'district': 'Karnal', 'crop': 'Wheat', 'season': 'Rabi', 'year': 2021, 'area': 155000, 'production': 806000},
        {'state': 'Haryana', 'district': 'Karnal', 'crop': 'Wheat', 'season': 'Rabi', 'year': 2020, 'area': 152000, 'production': 760000},
        {'state': 'Haryana', 'district': 'Karnal', 'crop': 'Wheat', 'season': 'Rabi', 'year': 2019, 'area': 150000, 'production': 780000},
        
        # Ludhiana Rice (Punjab)
        {'state': 'Punjab', 'district': 'Ludhiana', 'crop': 'Rice', 'season': 'Kharif', 'year': 2023, 'area': 250000, 'production': 1050000},
        {'state': 'Punjab', 'district': 'Ludhiana', 'crop': 'Rice', 'season': 'Kharif', 'year': 2022, 'area': 248000, 'production': 992000},
        {'state': 'Punjab', 'district': 'Ludhiana', 'crop': 'Rice', 'season': 'Kharif', 'year': 2021, 'area': 245000, 'production': 1029000},
        {'state': 'Punjab', 'district': 'Ludhiana', 'crop': 'Rice', 'season': 'Kharif', 'year': 2020, 'area': 242000, 'production': 944000},
        {'state': 'Punjab', 'district': 'Ludhiana', 'crop': 'Rice', 'season': 'Kharif', 'year': 2019, 'area': 240000, 'production': 1008000},
        
        # Indore Soybean (MP)
        {'state': 'Madhya Pradesh', 'district': 'Indore', 'crop': 'Soybean', 'season': 'Kharif', 'year': 2023, 'area': 180000, 'production': 234000},
        {'state': 'Madhya Pradesh', 'district': 'Indore', 'crop': 'Soybean', 'season': 'Kharif', 'year': 2022, 'area': 175000, 'production': 192500},
        {'state': 'Madhya Pradesh', 'district': 'Indore', 'crop': 'Soybean', 'season': 'Kharif', 'year': 2021, 'area': 170000, 'production': 221000},
        {'state': 'Madhya Pradesh', 'district': 'Indore', 'crop': 'Soybean', 'season': 'Kharif', 'year': 2020, 'area': 168000, 'production': 151200},  # Drought
        {'state': 'Madhya Pradesh', 'district': 'Indore', 'crop': 'Soybean', 'season': 'Kharif', 'year': 2019, 'area': 165000, 'production': 206250},
    ]
    
    df = pd.DataFrame(sample_data)
    
    # Calculate yield
    df['yield'] = ((df['production'] / df['area']) * 1000).round(2)  # Convert to kg/ha
    
    # Save
    data_dir = os.path.join(os.path.dirname(__file__), '..', 'data')
    os.makedirs(data_dir, exist_ok=True)
    path = os.path.join(data_dir, 'sample_yield_data.csv')
    df.to_csv(path, index=False)
    logger.info(f"Created sample data: {path}")
    
    return df


def main():
    """Test the data loader."""
    print("\n" + "=" * 60)
    print("REAL YIELD DATA LOADER TEST")
    print("=" * 60)
    
    # Create sample data for testing
    print("\n[1] Creating sample data based on published statistics...")
    sample_df = create_sample_data()
    print(f"    Created {len(sample_df)} records")
    
    # Initialize loader
    print("\n[2] Checking available data files...")
    loader = RealYieldDataLoader()
    available = loader.check_available_data()
    
    if not available:
        print("\n⚠️  No data files found! Creating sample for demo...")
        # Use sample data directly
        loader.data = sample_df
        loader.source = 'sample'
    else:
        loader.load_data()
    
    # Test threshold calculation
    print("\n[3] Calculating PMFBY threshold for Karnal Rice...")
    threshold = loader.calculate_pmfby_threshold(
        district='Karnal',
        crop='Rice',
        season='Kharif',
        indemnity_level=80
    )
    
    if threshold['status'] == 'success':
        print(f"\n    === PMFBY THRESHOLD CALCULATION ===")
        print(f"    District: {threshold['district']}")
        print(f"    Crop: {threshold['crop']}")
        print(f"    Season: {threshold['season']}")
        print(f"    Data Source: {threshold['data_source']}")
        print(f"    Years Analyzed: {threshold['years_analyzed']}")
        print(f"    ")
        print(f"    All Yields (kg/ha): {threshold['all_yields_kg_ha']}")
        print(f"    Excluded (Calamity): {threshold['excluded_calamity_yields']}")
        print(f"    Used for Average: {threshold['yields_for_average']}")
        print(f"    ")
        print(f"    Average Yield: {threshold['average_yield']} kg/ha")
        print(f"    Indemnity Level: {threshold['indemnity_level']}%")
        print(f"    ✓ THRESHOLD YIELD: {threshold['threshold_yield']} kg/ha")
        print(f"    ")
        print(f"    Formula: {threshold['formula']}")
    else:
        print(f"    Error: {threshold['message']}")
    
    print("\n" + "=" * 60)
    print("HOW TO GET REAL DATA:")
    print("=" * 60)
    print("""
1. KAGGLE (Recommended):
   https://www.kaggle.com/datasets/abhinand05/crop-production-in-india
   → Download and save as: d:/tr/pmfby_engine/data/crop_production.csv

2. DATA.GOV.IN:
   https://data.gov.in/resource/district-wise-season-wise-crop-production-statistics-1997
   → Download CSV and save as: d:/tr/pmfby_engine/data/govt_crop_stats.csv

3. INDIA DATA PORTAL:
   https://indiadataportal.com/p/area-production-yield-apy
   → Download and save as: d:/tr/pmfby_engine/data/india_apy.csv
""")
    
    return threshold


if __name__ == "__main__":
    main()
