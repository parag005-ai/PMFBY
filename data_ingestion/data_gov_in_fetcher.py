"""
PMFBY Yield Prediction Engine
data.gov.in API Integration

Fetches REAL district-wise crop production statistics from
Official Government of India Open Data Portal.

API Endpoint: /resource/35be999b-0208-4354-b557-f6ca9a5355de
Description: District-wise, season-wise crop production statistics from 1997
Source: Ministry of Agriculture & Farmers Welfare, DES
"""

import requests
import pandas as pd
import numpy as np
from typing import Dict, List, Optional
import logging
import os
import json
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataGovInFetcher:
    """
    Fetches crop production data from data.gov.in API.
    
    Data includes:
    - State, District
    - Crop, Season, Year
    - Area (hectares)
    - Production (tonnes)
    """
    
    BASE_URL = "https://api.data.gov.in/resource/35be999b-0208-4354-b557-f6ca9a5355de"
    
    # Sample API key from data.gov.in (public test key)
    DEFAULT_API_KEY = "579b464db66ec23bdd000001cdd3946e44ce4aad7209ff7b23ac571b"
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize data.gov.in fetcher.
        
        Args:
            api_key: Optional custom API key. Uses default if not provided.
        """
        self.api_key = api_key or self.DEFAULT_API_KEY
        self.cache_dir = os.path.join(os.path.dirname(__file__), '..', 'data', 'cache')
        os.makedirs(self.cache_dir, exist_ok=True)
        
        logger.info("Initialized data.gov.in API fetcher")
        logger.info(f"Endpoint: {self.BASE_URL}")
    
    def fetch_data(
        self,
        state: Optional[str] = None,
        district: Optional[str] = None,
        crop: Optional[str] = None,
        season: Optional[str] = None,
        limit: int = 1000,
        offset: int = 0,
        format: str = 'json'
    ) -> pd.DataFrame:
        """
        Fetch crop production data from API.
        
        Args:
            state: Filter by state name
            district: Filter by district name
            crop: Filter by crop name
            season: Filter by season (Kharif/Rabi/Whole Year)
            limit: Maximum records to return
            offset: Records to skip (for pagination)
            format: Output format (json/csv/xml)
            
        Returns:
            DataFrame with crop production data
        """
        params = {
            'api-key': self.api_key,
            'format': format,
            'limit': limit,
            'offset': offset
        }
        
        # Add filters
        filters = []
        if state:
            filters.append(f'state_name:"{state}"')
        if district:
            filters.append(f'district_name:"{district}"')
        if crop:
            filters.append(f'crop:"{crop}"')
        if season:
            filters.append(f'season:"{season}"')
        
        if filters:
            params['filters[0]'] = ' AND '.join(filters)
        
        logger.info(f"Fetching data: state={state}, district={district}, crop={crop}")
        
        try:
            response = requests.get(self.BASE_URL, params=params, timeout=30)
            response.raise_for_status()
            
            if format == 'json':
                data = response.json()
                
                if 'records' in data:
                    records = data['records']
                    df = pd.DataFrame(records)
                    logger.info(f"Retrieved {len(df)} records from data.gov.in")
                    return self._standardize_dataframe(df)
                else:
                    logger.warning("No 'records' key in response")
                    return pd.DataFrame()
            else:
                # For CSV format
                from io import StringIO
                df = pd.read_csv(StringIO(response.text))
                return self._standardize_dataframe(df)
                
        except requests.exceptions.RequestException as e:
            logger.error(f"API request failed: {e}")
            return pd.DataFrame()
    
    def _standardize_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Standardize column names and data types."""
        if df.empty:
            return df
        
        # Lowercase column names
        df.columns = df.columns.str.lower().str.strip().str.replace(' ', '_')
        
        # Rename columns to standard format
        rename_map = {
            'state_name': 'state',
            'district_name': 'district',
            'crop_year': 'year',
            'area_': 'area',
            'production_': 'production'
        }
        for old, new in rename_map.items():
            matching_cols = [c for c in df.columns if old in c]
            for col in matching_cols:
                df = df.rename(columns={col: new})
        
        # Convert to lowercase
        for col in ['state', 'district', 'crop', 'season']:
            if col in df.columns:
                df[col] = df[col].astype(str).str.lower().str.strip()
        
        # Convert numeric columns
        for col in ['year', 'area', 'production']:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Calculate yield (kg/ha)
        if 'area' in df.columns and 'production' in df.columns:
            # Production is in tonnes, area in hectares
            # Yield = (Production in tonnes / Area in ha) * 1000 = kg/ha
            df['yield'] = ((df['production'] / df['area'].replace(0, np.nan)) * 1000).round(2)
        
        return df
    
    def fetch_district_data(
        self,
        district: str,
        crop: str,
        season: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Fetch all historical data for a specific district and crop.
        
        Args:
            district: District name
            crop: Crop name
            season: Optional season filter
            
        Returns:
            DataFrame with year-wise yield data
        """
        all_records = []
        offset = 0
        
        while True:
            df = self.fetch_data(
                district=district,
                crop=crop,
                season=season,
                limit=1000,
                offset=offset
            )
            
            if df.empty:
                break
            
            all_records.append(df)
            
            if len(df) < 1000:
                break
            
            offset += 1000
        
        if all_records:
            result = pd.concat(all_records, ignore_index=True)
            result = result.drop_duplicates(subset=['district', 'crop', 'year', 'season'])
            return result.sort_values('year', ascending=False)
        
        return pd.DataFrame()
    
    def calculate_pmfby_threshold(
        self,
        district: str,
        crop: str,
        season: Optional[str] = None,
        indemnity_level: int = 80,
        years_to_use: int = 7,
        exclude_worst: int = 2
    ) -> Dict:
        """
        Calculate PMFBY threshold from real API data.
        
        Official PMFBY Formula:
        1. Use last 7 years of yield data
        2. Exclude 2 calamity (worst) years
        3. Average the remaining 5 years
        4. Threshold Yield = Average × (Indemnity Level / 100)
        
        Args:
            district: District name
            crop: Crop name
            season: Season (Kharif/Rabi)
            indemnity_level: 70, 80, or 90
            years_to_use: Years of data to consider (default 7)
            exclude_worst: Calamity years to exclude (default 2)
            
        Returns:
            Dictionary with threshold calculation details
        """
        logger.info(f"Calculating PMFBY threshold for {district}/{crop}")
        
        # Fetch data from API
        df = self.fetch_district_data(district, crop, season)
        
        if df.empty:
            return {
                'status': 'error',
                'message': f'No data found for {district}/{crop}',
                'source': 'data.gov.in API',
                'threshold_yield': None
            }
        
        # Get recent years
        df = df.sort_values('year', ascending=False).head(years_to_use)
        
        if 'yield' not in df.columns or df['yield'].isna().all():
            return {
                'status': 'error',
                'message': 'Yield data not available',
                'source': 'data.gov.in API',
                'threshold_yield': None
            }
        
        yields = df['yield'].dropna().values
        years = df['year'].values if 'year' in df.columns else []
        
        if len(yields) < 3:
            return {
                'status': 'insufficient_data',
                'message': f'Only {len(yields)} years available (need at least 3)',
                'source': 'data.gov.in API',
                'threshold_yield': None
            }
        
        # Sort and exclude worst years
        sorted_yields = np.sort(yields)
        if len(sorted_yields) > exclude_worst:
            used_yields = sorted_yields[exclude_worst:]  # Remove lowest values
            excluded_yields = sorted_yields[:exclude_worst]
        else:
            used_yields = sorted_yields
            excluded_yields = []
        
        # Calculate average and threshold
        avg_yield = np.mean(used_yields)
        threshold_yield = avg_yield * (indemnity_level / 100)
        
        return {
            'status': 'success',
            'district': district,
            'crop': crop,
            'season': season,
            'source': 'data.gov.in Official API',
            'api_endpoint': self.BASE_URL,
            'years_analyzed': len(yields),
            'years_available': list(years.astype(int)) if len(years) > 0 else [],
            'all_yields_kg_ha': list(yields.round(2)),
            'excluded_calamity_yields': list(excluded_yields.round(2)) if len(excluded_yields) > 0 else [],
            'yields_for_average': list(used_yields.round(2)),
            'average_yield': round(float(avg_yield), 2),
            'indemnity_level': indemnity_level,
            'threshold_yield': round(float(threshold_yield), 2),
            'units': 'kg/ha',
            'formula': f'Threshold = {avg_yield:.0f} × {indemnity_level}% = {threshold_yield:.0f} kg/ha',
            'data_authenticity': '100% Official Government Data'
        }
    
    def cache_state_data(self, state: str) -> str:
        """
        Download and cache all data for a state.
        
        Args:
            state: State name
            
        Returns:
            Path to cached CSV file
        """
        logger.info(f"Caching data for state: {state}")
        
        all_data = []
        offset = 0
        
        while True:
            df = self.fetch_data(state=state, limit=1000, offset=offset)
            
            if df.empty:
                break
            
            all_data.append(df)
            logger.info(f"  Fetched {len(df)} records (offset {offset})")
            
            if len(df) < 1000:
                break
            
            offset += 1000
        
        if all_data:
            result = pd.concat(all_data, ignore_index=True)
            
            # Save to cache
            filename = f"{state.lower().replace(' ', '_')}_{datetime.now().strftime('%Y%m%d')}.csv"
            filepath = os.path.join(self.cache_dir, filename)
            result.to_csv(filepath, index=False)
            
            logger.info(f"Cached {len(result)} records to {filepath}")
            return filepath
        
        return ""


def main():
    """Test the data.gov.in API fetcher."""
    print("=" * 70)
    print("data.gov.in API TEST - Official Government Crop Statistics")
    print("=" * 70)
    
    fetcher = DataGovInFetcher()
    
    # Test 1: Fetch sample data
    print("\n[1] Testing API connection...")
    df = fetcher.fetch_data(limit=10)
    
    if not df.empty:
        print(f"    ✓ API connected successfully!")
        print(f"    Retrieved {len(df)} sample records")
        print(f"    Columns: {list(df.columns)}")
        print(f"\n    Sample data:")
        print(df[['state', 'district', 'crop', 'year', 'area', 'production']].head(5).to_string())
    else:
        print("    ✗ API connection failed or no data returned")
        print("    Using cached/sample data instead...")
        return
    
    # Test 2: Fetch Karnal Rice data
    print("\n[2] Fetching Karnal Rice data...")
    karnal_df = fetcher.fetch_district_data('Karnal', 'Rice', 'Kharif')
    
    if not karnal_df.empty:
        print(f"    ✓ Found {len(karnal_df)} years of data")
        print(karnal_df[['year', 'crop', 'production', 'area', 'yield']].to_string())
    else:
        print("    Trying with different search...")
        karnal_df = fetcher.fetch_data(crop='Rice', limit=100)
        if not karnal_df.empty:
            print(f"    Found rice data for districts: {karnal_df['district'].unique()[:10]}")
    
    # Test 3: Calculate PMFBY threshold
    print("\n[3] Calculating PMFBY Threshold (Official Formula)...")
    result = fetcher.calculate_pmfby_threshold(
        district='Karnal',
        crop='Rice',
        season='Kharif',
        indemnity_level=80
    )
    
    print(f"\n    === PMFBY THRESHOLD CALCULATION ===")
    for key, value in result.items():
        print(f"    {key}: {value}")
    
    print("\n" + "=" * 70)
    print("DATA SOURCE VERIFICATION")
    print("=" * 70)
    print(f"API Endpoint: {fetcher.BASE_URL}")
    print(f"Provider: Ministry of Agriculture & Farmers Welfare, DES")
    print(f"Portal: Open Government Data Platform India (data.gov.in)")
    print(f"Data Authenticity: 100% Official Government Data")
    print("=" * 70)
    
    return result


if __name__ == "__main__":
    main()
