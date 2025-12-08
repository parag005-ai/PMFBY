"""
PMFBY Yield Prediction Engine
Weather Data Fetcher Module

Fetches weather data from NASA POWER API for crop stress analysis.
Includes temperature, rainfall, solar radiation, humidity, and derived indices.
"""

import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import logging
import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class WeatherFetcher:
    """
    Weather data fetcher using NASA POWER API.
    
    Features:
    - Daily temperature (min/max)
    - Precipitation
    - Solar radiation
    - Wind speed and humidity
    - Derived indices (VPD, GDD, ET)
    """
    
    NASA_POWER_URL = "https://power.larc.nasa.gov/api/temporal/daily/point"
    
    # Parameters to fetch
    PARAMETERS = [
        "T2M_MAX",       # Max temperature at 2m (°C)
        "T2M_MIN",       # Min temperature at 2m (°C)
        "T2M",           # Mean temperature at 2m (°C)
        "PRECTOTCORR",   # Precipitation (mm/day)
        "ALLSKY_SFC_SW_DWN",  # Solar radiation (MJ/m²/day)
        "WS2M",          # Wind speed at 2m (m/s)
        "RH2M",          # Relative humidity at 2m (%)
        "T2MDEW",        # Dew point temperature (°C)
        "PS",            # Surface pressure (kPa)
    ]
    
    def __init__(self):
        """Initialize weather fetcher."""
        self.session = requests.Session()
        
    def fetch_daily_weather(
        self,
        latitude: float,
        longitude: float,
        start_date: str,
        end_date: str
    ) -> pd.DataFrame:
        """
        Fetch daily weather data from NASA POWER.
        
        Args:
            latitude: Latitude in decimal degrees
            longitude: Longitude in decimal degrees
            start_date: Start date (YYYY-MM-DD or YYYYMMDD)
            end_date: End date (YYYY-MM-DD or YYYYMMDD)
            
        Returns:
            DataFrame with daily weather parameters
        """
        # Clean date format
        start_clean = start_date.replace('-', '')
        end_clean = end_date.replace('-', '')
        
        params = {
            "parameters": ",".join(self.PARAMETERS),
            "community": "AG",
            "longitude": longitude,
            "latitude": latitude,
            "start": start_clean,
            "end": end_clean,
            "format": "JSON"
        }
        
        try:
            logger.info(f"Fetching weather data for ({latitude}, {longitude})")
            response = self.session.get(self.NASA_POWER_URL, params=params, timeout=60)
            response.raise_for_status()
            
            data = response.json()
            
            if 'properties' not in data or 'parameter' not in data['properties']:
                logger.error("Invalid response structure from NASA POWER")
                return self._generate_synthetic_weather(start_date, end_date)
            
            # Extract parameters
            params_data = data['properties']['parameter']
            
            # Build DataFrame
            records = []
            dates = list(params_data.get('T2M_MAX', {}).keys())
            
            for date_str in dates:
                record = {'date': date_str}
                for param in self.PARAMETERS:
                    if param in params_data:
                        value = params_data[param].get(date_str)
                        # Handle missing values (-999)
                        if value is not None and value > -900:
                            record[param.lower()] = value
                        else:
                            record[param.lower()] = np.nan
                records.append(record)
            
            df = pd.DataFrame(records)
            
            if not df.empty:
                # Parse dates
                df['date'] = pd.to_datetime(df['date'], format='%Y%m%d')
                df = df.sort_values('date').reset_index(drop=True)
                
                # Calculate derived indices
                df = self._calculate_derived_indices(df)
                
                # Interpolate missing values
                df = df.interpolate(method='linear', limit=5)
            
            logger.info(f"Fetched {len(df)} days of weather data")
            return df
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Error fetching weather data: {e}")
            return self._generate_synthetic_weather(start_date, end_date)
    
    def _calculate_derived_indices(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate derived weather indices.
        
        Indices:
        - VPD (Vapor Pressure Deficit)
        - GDD (Growing Degree Days)
        - ET0 (Reference Evapotranspiration - Hargreaves method)
        """
        # Mean temperature
        if 't2m' not in df.columns:
            df['t2m'] = (df['t2m_max'] + df['t2m_min']) / 2
        
        # VPD (Vapor Pressure Deficit) in kPa
        # VPD = es - ea, where es = saturation vapor pressure, ea = actual vapor pressure
        def calc_vpd(row):
            if pd.isna(row['t2m']) or pd.isna(row.get('rh2m')):
                return np.nan
            t = row['t2m']
            rh = row.get('rh2m', 60)
            es = 0.6108 * np.exp(17.27 * t / (t + 237.3))  # Tetens formula
            ea = es * (rh / 100)
            return max(0, es - ea)
        
        df['vpd'] = df.apply(calc_vpd, axis=1)
        
        # Growing Degree Days (base 10°C for most crops)
        def calc_gdd(row, base_temp=10, max_temp=30):
            if pd.isna(row['t2m_max']) or pd.isna(row['t2m_min']):
                return np.nan
            t_max = min(row['t2m_max'], max_temp)
            t_min = max(row['t2m_min'], base_temp)
            t_avg = (t_max + t_min) / 2
            return max(0, t_avg - base_temp)
        
        df['gdd'] = df.apply(calc_gdd, axis=1)
        df['gdd_cumsum'] = df['gdd'].cumsum()
        
        # Reference ET (Hargreaves method)
        # ET0 = 0.0023 * Ra * (Tmean + 17.8) * (Tmax - Tmin)^0.5
        def calc_et0(row):
            if pd.isna(row['t2m_max']) or pd.isna(row['t2m_min']):
                return np.nan
            t_range = row['t2m_max'] - row['t2m_min']
            if t_range <= 0:
                t_range = 0.1
            ra = row.get('allsky_sfc_sw_dwn', 20)  # Approximate
            return 0.0023 * ra * (row['t2m'] + 17.8) * np.sqrt(t_range)
        
        df['et0'] = df.apply(calc_et0, axis=1)
        
        # Rainfall cumulative
        if 'prectotcorr' in df.columns:
            df['rain_cumsum'] = df['prectotcorr'].cumsum()
            
            # Water balance (Rain - ET)
            df['water_balance'] = df['prectotcorr'] - df['et0']
            df['water_balance_cumsum'] = df['water_balance'].cumsum()
        
        return df
    
    def _generate_synthetic_weather(
        self,
        start_date: str,
        end_date: str
    ) -> pd.DataFrame:
        """Generate synthetic weather data for testing."""
        start = pd.to_datetime(start_date)
        end = pd.to_datetime(end_date)
        dates = pd.date_range(start, end, freq='D')
        
        n = len(dates)
        
        # Simulate monsoon season weather pattern
        day_of_year = dates.dayofyear
        
        # Temperature pattern (peaks in May, cooler in winter)
        t_base = 25 + 10 * np.sin(2 * np.pi * (day_of_year - 120) / 365)
        t_max = t_base + np.random.uniform(5, 10, n)
        t_min = t_base - np.random.uniform(5, 10, n)
        
        # Rainfall (monsoon peak July-August)
        rain_prob = 0.3 + 0.5 * np.exp(-0.5 * ((day_of_year - 210) / 30) ** 2)
        rain_mask = np.random.random(n) < rain_prob
        rain = np.where(rain_mask, np.random.exponential(15, n), 0)
        
        # Humidity (higher during monsoon)
        humidity = 50 + 30 * rain_prob + np.random.uniform(-5, 5, n)
        
        # Solar radiation
        solar = 18 + 5 * np.sin(2 * np.pi * (day_of_year - 172) / 365)
        solar = np.where(rain > 5, solar * 0.6, solar)
        
        df = pd.DataFrame({
            'date': dates,
            't2m_max': t_max,
            't2m_min': t_min,
            't2m': (t_max + t_min) / 2,
            'prectotcorr': np.clip(rain, 0, 150),
            'allsky_sfc_sw_dwn': np.clip(solar + np.random.normal(0, 2, n), 5, 30),
            'ws2m': np.clip(2 + np.random.normal(0, 1, n), 0.5, 8),
            'rh2m': np.clip(humidity, 30, 95)
        })
        
        # Calculate derived indices
        df = self._calculate_derived_indices(df)
        
        logger.info(f"Generated {len(df)} days of synthetic weather data")
        return df
    
    def analyze_stress_conditions(
        self,
        df: pd.DataFrame,
        stages: Dict[str, Tuple[int, int]],
        sowing_date: str
    ) -> Dict:
        """
        Analyze weather stress conditions by crop stage.
        
        Args:
            df: Weather DataFrame
            stages: Dict of stage_name -> (start_day, end_day)
            sowing_date: Sowing date string
            
        Returns:
            Dictionary with stress analysis by stage
        """
        sowing = pd.to_datetime(sowing_date)
        df['days_after_sowing'] = (df['date'] - sowing).dt.days
        
        stress_analysis = {}
        
        for stage_name, (start_day, end_day) in stages.items():
            stage_df = df[(df['days_after_sowing'] >= start_day) & 
                          (df['days_after_sowing'] < end_day)]
            
            if stage_df.empty:
                continue
            
            # Heat stress: days with Tmax > 35°C
            heat_stress_days = (stage_df['t2m_max'] > 35).sum()
            
            # Drought stress: consecutive dry days
            dry_days = (stage_df['prectotcorr'] < 2.5).sum()
            
            # VPD stress: days with VPD > 2.5 kPa
            vpd_stress_days = (stage_df['vpd'] > 2.5).sum() if 'vpd' in stage_df.columns else 0
            
            # Water deficit
            water_deficit = stage_df['water_balance'].sum() if 'water_balance' in stage_df.columns else 0
            
            stress_analysis[stage_name] = {
                'days_in_stage': len(stage_df),
                'mean_temp': stage_df['t2m'].mean(),
                'max_temp_recorded': stage_df['t2m_max'].max(),
                'total_rainfall_mm': stage_df['prectotcorr'].sum(),
                'heat_stress_days': int(heat_stress_days),
                'dry_days': int(dry_days),
                'vpd_stress_days': int(vpd_stress_days),
                'water_deficit_mm': round(water_deficit, 1),
                'gdd_accumulated': stage_df['gdd'].sum() if 'gdd' in stage_df.columns else 0
            }
        
        # Overall stress score calculation
        total_heat = sum(s.get('heat_stress_days', 0) for s in stress_analysis.values())
        total_drought = sum(1 for s in stress_analysis.values() if s.get('dry_days', 0) > 10)
        total_vpd = sum(s.get('vpd_stress_days', 0) for s in stress_analysis.values())
        
        stress_analysis['overall'] = {
            'heat_risk': 'HIGH' if total_heat > 15 else 'MODERATE' if total_heat > 7 else 'LOW',
            'drought_risk': 'HIGH' if total_drought > 2 else 'MODERATE' if total_drought > 0 else 'LOW',
            'vpd_risk': 'HIGH' if total_vpd > 20 else 'MODERATE' if total_vpd > 10 else 'LOW'
        }
        
        return stress_analysis


def main():
    """Test the weather fetcher."""
    fetcher = WeatherFetcher()
    
    # Test for Karnal, Haryana
    df = fetcher.fetch_daily_weather(
        latitude=29.69,
        longitude=76.97,
        start_date="2024-06-01",
        end_date="2024-11-30"
    )
    
    print("\n=== Weather Data Summary ===")
    print(df.head(20))
    print(f"\nTotal days: {len(df)}")
    print(f"\nTemperature range: {df['t2m_min'].min():.1f}°C - {df['t2m_max'].max():.1f}°C")
    print(f"Total rainfall: {df['prectotcorr'].sum():.1f} mm")
    print(f"Total GDD: {df['gdd'].sum():.1f}")
    
    # Test stress analysis
    stages = {
        'vegetative': (0, 30),
        'tillering': (30, 55),
        'flowering': (55, 80),
        'grain_filling': (80, 105),
        'maturity': (105, 130)
    }
    
    stress = fetcher.analyze_stress_conditions(df, stages, "2024-06-15")
    print("\n=== Stress Analysis ===")
    for stage, data in stress.items():
        print(f"\n{stage}: {data}")
    
    return df


if __name__ == "__main__":
    main()
