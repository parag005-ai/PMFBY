"""
PMFBY Yield Prediction Engine
Feature Extraction Module

Extracts comprehensive features from time series data
for yield prediction model input.
"""

import pandas as pd
import numpy as np
from scipy import stats
from typing import Dict, List, Tuple, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FeatureExtractor:
    """
    Extracts features from satellite and weather time series
    for yield prediction model.
    
    Feature categories:
    1. Spectral (NDVI, EVI, NDWI)
    2. SAR (VV, VH, RVI)
    3. Weather (temperature, rainfall, GDD)
    4. Stress indices
    5. Phenological
    6. Soil (static)
    """
    
    def __init__(self, sequence_length: int = 36):
        """
        Initialize feature extractor.
        
        Args:
            sequence_length: Number of timesteps for sequence features
        """
        self.sequence_length = sequence_length
        
    def extract_all_features(
        self,
        df: pd.DataFrame,
        soil_data: Optional[Dict] = None,
        sowing_date: Optional[str] = None
    ) -> Dict:
        """
        Extract all features from merged time series.
        
        Args:
            df: Merged DataFrame with satellite, weather, stress data
            soil_data: Optional soil properties dictionary
            sowing_date: Optional sowing date for phenology
            
        Returns:
            Dictionary with all extracted features
        """
        features = {}
        
        # Spectral features
        features.update(self._extract_spectral_features(df))
        
        # SAR features
        features.update(self._extract_sar_features(df))
        
        # Weather features
        features.update(self._extract_weather_features(df))
        
        # Stress features
        features.update(self._extract_stress_features(df))
        
        # Phenological features
        if sowing_date:
            features.update(self._extract_phenology_features(df, sowing_date))
        
        # Soil features
        if soil_data:
            features.update(self._extract_soil_features(soil_data))
        
        # Sequence features for LSTM/Transformer
        features['sequence'] = self._extract_sequence_features(df)
        
        logger.info(f"Extracted {len(features)} feature groups")
        return features
    
    def _extract_spectral_features(self, df: pd.DataFrame) -> Dict:
        """Extract vegetation index features."""
        features = {}
        
        for col in ['ndvi', 'ndvi_smooth', 'evi', 'ndwi', 'savi']:
            if col not in df.columns:
                continue
                
            series = df[col].dropna()
            if series.empty:
                continue
            
            base = col.replace('_smooth', '')
            features[f'{base}_peak'] = float(series.max())
            features[f'{base}_min'] = float(series.min())
            features[f'{base}_mean'] = float(series.mean())
            features[f'{base}_std'] = float(series.std())
            features[f'{base}_range'] = float(series.max() - series.min())
            
            # Area under curve (integral)
            features[f'{base}_auc'] = float(np.trapz(series.values, dx=5))
            
            # Trend (linear slope)
            if len(series) > 2:
                slope, _, _, _, _ = stats.linregress(range(len(series)), series.values)
                features[f'{base}_slope'] = float(slope)
            
            # Peak timing (position as fraction of season)
            peak_idx = series.idxmax()
            features[f'{base}_peak_timing'] = float(peak_idx / len(df)) if len(df) > 0 else 0.5
            
            # Rate metrics
            if len(series) > 5:
                # Greenup rate (first half)
                half = len(series) // 2
                greenup = (series.iloc[half] - series.iloc[0]) / half if half > 0 else 0
                features[f'{base}_greenup_rate'] = float(greenup)
                
                # Senescence rate (second half)
                senescence = (series.iloc[-1] - series.iloc[half]) / (len(series) - half) if half < len(series) else 0
                features[f'{base}_senescence_rate'] = float(senescence)
        
        return features
    
    def _extract_sar_features(self, df: pd.DataFrame) -> Dict:
        """Extract SAR backscatter features."""
        features = {}
        
        for col in ['vv', 'vh', 'vh_vv_ratio', 'rvi']:
            if col not in df.columns:
                continue
                
            series = df[col].dropna()
            if series.empty:
                continue
            
            features[f'sar_{col}_mean'] = float(series.mean())
            features[f'sar_{col}_std'] = float(series.std())
            features[f'sar_{col}_max'] = float(series.max())
            features[f'sar_{col}_min'] = float(series.min())
            
            # Temporal coefficient of variation
            if series.mean() != 0:
                features[f'sar_{col}_cv'] = float(series.std() / abs(series.mean()))
        
        # Biomass proxy from VH/VV ratio
        if 'vh_vv_ratio' in df.columns:
            ratio = df['vh_vv_ratio'].dropna()
            if not ratio.empty:
                features['sar_biomass_proxy'] = float(ratio.mean())
        
        return features
    
    def _extract_weather_features(self, df: pd.DataFrame) -> Dict:
        """Extract weather-derived features."""
        features = {}
        
        # Temperature features
        if 't2m_max' in df.columns:
            features['temp_max_mean'] = float(df['t2m_max'].mean())
            features['temp_max_extreme'] = float(df['t2m_max'].max())
            features['heat_days'] = int((df['t2m_max'] > 35).sum())
            features['hot_spell_days'] = int((df['t2m_max'] > 38).sum())
        
        if 't2m_min' in df.columns:
            features['temp_min_mean'] = float(df['t2m_min'].mean())
            features['cold_days'] = int((df['t2m_min'] < 10).sum())
        
        if 't2m' in df.columns:
            features['temp_mean'] = float(df['t2m'].mean())
            features['temp_range'] = float(df['t2m'].max() - df['t2m'].min())
        
        # Rainfall features
        if 'prectotcorr' in df.columns:
            rain = df['prectotcorr'].fillna(0)
            features['rain_total'] = float(rain.sum())
            features['rain_days'] = int((rain > 2.5).sum())
            features['heavy_rain_days'] = int((rain > 50).sum())
            features['dry_days'] = int((rain < 1).sum())
            features['rain_max_daily'] = float(rain.max())
            
            # Dry spell detection
            dry_spell = (rain < 1).astype(int)
            max_dry_spell = 0
            current_spell = 0
            for val in dry_spell:
                if val == 1:
                    current_spell += 1
                    max_dry_spell = max(max_dry_spell, current_spell)
                else:
                    current_spell = 0
            features['max_dry_spell'] = int(max_dry_spell)
        
        # GDD features
        if 'gdd' in df.columns:
            features['gdd_total'] = float(df['gdd'].sum())
            features['gdd_mean'] = float(df['gdd'].mean())
        
        if 'gdd_cumsum' in df.columns:
            features['gdd_cumulative'] = float(df['gdd_cumsum'].iloc[-1])
        
        # VPD features
        if 'vpd' in df.columns:
            features['vpd_mean'] = float(df['vpd'].mean())
            features['vpd_max'] = float(df['vpd'].max())
            features['vpd_stress_days'] = int((df['vpd'] > 2.5).sum())
        
        # ET features
        if 'et0' in df.columns:
            features['et_total'] = float(df['et0'].sum())
            features['et_mean'] = float(df['et0'].mean())
        
        # Water balance
        if 'water_balance_cum' in df.columns:
            features['water_balance_final'] = float(df['water_balance_cum'].iloc[-1])
            features['water_deficit_days'] = int((df['water_balance_cum'] < -30).sum())
        
        return features
    
    def _extract_stress_features(self, df: pd.DataFrame) -> Dict:
        """Extract stress index features."""
        features = {}
        
        stress_cols = ['heat_stress', 'moisture_stress', 'water_balance_stress', 'combined_stress']
        
        for col in stress_cols:
            if col not in df.columns:
                continue
                
            series = df[col].fillna(0)
            features[f'{col}_mean'] = float(series.mean())
            features[f'{col}_max'] = float(series.max())
            features[f'{col}_days_above_50'] = int((series > 0.5).sum())
            features[f'{col}_days_above_70'] = int((series > 0.7).sum())
        
        # Stress severity distribution
        if 'stress_severity' in df.columns:
            severity_counts = df['stress_severity'].value_counts(normalize=True)
            for severity in ['LOW', 'MODERATE', 'HIGH', 'SEVERE']:
                features[f'stress_pct_{severity.lower()}'] = float(severity_counts.get(severity, 0))
        
        return features
    
    def _extract_phenology_features(
        self,
        df: pd.DataFrame,
        sowing_date: str
    ) -> Dict:
        """Extract phenology-related features."""
        features = {}
        
        sowing = pd.to_datetime(sowing_date)
        
        if 'date' in df.columns:
            df = df.copy()
            df['das'] = (pd.to_datetime(df['date']) - sowing).dt.days
            
            features['season_length'] = int(df['das'].max())
            
            # Peak NDVI timing
            if 'ndvi' in df.columns or 'ndvi_smooth' in df.columns:
                ndvi_col = 'ndvi_smooth' if 'ndvi_smooth' in df.columns else 'ndvi'
                peak_idx = df[ndvi_col].idxmax()
                features['peak_ndvi_das'] = int(df.loc[peak_idx, 'das'])
        
        return features
    
    def _extract_soil_features(self, soil_data: Dict) -> Dict:
        """Extract soil features."""
        features = {}
        
        soil_keys = [
            'ph', 'organic_carbon_pct', 'nitrogen_kg_ha',
            'phosphorus_kg_ha', 'potassium_kg_ha', 'water_holding_pct'
        ]
        
        for key in soil_keys:
            if key in soil_data:
                features[f'soil_{key}'] = float(soil_data[key])
        
        # Encoded categorical
        drainage_map = {'good': 3, 'moderate': 2, 'poor': 1, 'excessive': 2}
        if 'drainage' in soil_data:
            features['soil_drainage'] = float(drainage_map.get(soil_data['drainage'], 2))
        
        return features
    
    def _extract_sequence_features(self, df: pd.DataFrame) -> np.ndarray:
        """
        Extract sequence features for LSTM/Transformer input.
        
        Returns array of shape (seq_len, num_features)
        """
        # Select columns for sequence
        sequence_cols = [
            'ndvi', 'evi', 'ndwi',          # Spectral
            'vv', 'vh', 'rvi',               # SAR
            't2m', 'prectotcorr', 'vpd',     # Weather
            'heat_stress', 'moisture_stress', 'combined_stress'  # Stress
        ]
        
        available_cols = [c for c in sequence_cols if c in df.columns]
        
        if not available_cols:
            return np.zeros((self.sequence_length, len(sequence_cols)))
        
        # Resample to fixed sequence length
        df_seq = df[available_cols].copy()
        
        # Handle length mismatch
        if len(df_seq) >= self.sequence_length:
            # Downsample by taking evenly spaced samples
            indices = np.linspace(0, len(df_seq) - 1, self.sequence_length, dtype=int)
            sequence = df_seq.iloc[indices].values
        else:
            # Upsample by interpolation
            x_old = np.linspace(0, 1, len(df_seq))
            x_new = np.linspace(0, 1, self.sequence_length)
            sequence = np.zeros((self.sequence_length, len(available_cols)))
            for i, col in enumerate(available_cols):
                sequence[:, i] = np.interp(x_new, x_old, df_seq[col].fillna(0).values)
        
        # Pad with zeros if not all columns available
        full_sequence = np.zeros((self.sequence_length, len(sequence_cols)))
        for i, col in enumerate(available_cols):
            col_idx = sequence_cols.index(col)
            full_sequence[:, col_idx] = sequence[:, i]
        
        return full_sequence
    
    def features_to_dataframe(self, features: Dict) -> pd.DataFrame:
        """Convert feature dictionary to flat DataFrame row."""
        flat_features = {}
        
        for key, value in features.items():
            if key == 'sequence':
                continue  # Skip sequence (handled separately)
            if isinstance(value, (int, float)):
                flat_features[key] = value
            elif isinstance(value, dict):
                for sub_key, sub_value in value.items():
                    if isinstance(sub_value, (int, float)):
                        flat_features[f'{key}_{sub_key}'] = sub_value
        
        return pd.DataFrame([flat_features])


def main():
    """Test feature extraction."""
    # Create sample merged data
    np.random.seed(42)
    dates = pd.date_range('2024-06-15', periods=30, freq='5D')
    n = len(dates)
    
    df = pd.DataFrame({
        'date': dates,
        'ndvi': 0.3 + 0.5 * np.sin(np.pi * np.arange(n) / 15) + np.random.normal(0, 0.03, n),
        'evi': 0.25 + 0.45 * np.sin(np.pi * np.arange(n) / 15),
        'ndwi': 0.1 + 0.2 * np.sin(np.pi * np.arange(n) / 20),
        'vv': -12 + np.random.normal(0, 1, n),
        'vh': -18 + np.random.normal(0, 1, n),
        'rvi': 0.3 + 0.2 * np.sin(np.pi * np.arange(n) / 15),
        't2m': 28 + 5 * np.sin(np.pi * np.arange(n) / 30),
        't2m_max': 33 + 5 * np.sin(np.pi * np.arange(n) / 30),
        'prectotcorr': np.random.exponential(10, n),
        'vpd': 1.5 + np.random.uniform(0, 1.5, n),
        'gdd': 15 + np.random.uniform(0, 5, n),
        'heat_stress': np.random.uniform(0, 0.5, n),
        'moisture_stress': np.random.uniform(0, 0.4, n),
        'combined_stress': np.random.uniform(0, 0.4, n)
    })
    
    soil_data = {
        'ph': 7.5,
        'organic_carbon_pct': 0.55,
        'nitrogen_kg_ha': 185,
        'phosphorus_kg_ha': 22,
        'potassium_kg_ha': 280,
        'water_holding_pct': 45,
        'drainage': 'good'
    }
    
    # Extract features
    extractor = FeatureExtractor()
    features = extractor.extract_all_features(df, soil_data, '2024-06-15')
    
    print("\n=== Extracted Features ===")
    for key, value in features.items():
        if key != 'sequence':
            print(f"  {key}: {value}")
    
    print(f"\n=== Sequence Shape ===")
    print(f"  {features['sequence'].shape}")
    
    # Convert to DataFrame
    df_features = extractor.features_to_dataframe(features)
    print(f"\n=== Feature DataFrame ===")
    print(f"  Columns: {len(df_features.columns)}")
    print(df_features.head())
    
    return features


if __name__ == "__main__":
    main()
