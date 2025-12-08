"""
PMFBY Yield Prediction Engine
Data Preprocessing Module

Handles cloud masking, gap filling, noise filtering,
and spatial clipping for satellite time series.
"""

import pandas as pd
import numpy as np
from scipy.signal import savgol_filter
from scipy.interpolate import interp1d
from typing import Dict, List, Optional, Tuple
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TimeSeriesPreprocessor:
    """
    Preprocessing pipeline for satellite time series data.
    
    Features:
    - Gap filling (linear, spline interpolation)
    - Noise filtering (Savitzky-Golay)
    - Outlier removal
    - Quality flagging
    """
    
    def __init__(
        self,
        savgol_window: int = 5,
        savgol_order: int = 2,
        outlier_threshold: float = 3.0
    ):
        """
        Initialize preprocessor.
        
        Args:
            savgol_window: Window size for Savitzky-Golay filter (must be odd)
            savgol_order: Polynomial order for Savitzky-Golay filter
            outlier_threshold: Z-score threshold for outlier detection
        """
        self.savgol_window = savgol_window if savgol_window % 2 == 1 else savgol_window + 1
        self.savgol_order = savgol_order
        self.outlier_threshold = outlier_threshold
        
    def preprocess_vegetation_series(
        self,
        df: pd.DataFrame,
        date_col: str = 'date',
        target_cols: List[str] = ['ndvi', 'evi', 'ndwi']
    ) -> pd.DataFrame:
        """
        Full preprocessing pipeline for vegetation indices.
        
        Args:
            df: DataFrame with date and vegetation indices
            date_col: Name of date column
            target_cols: Columns to preprocess
            
        Returns:
            Preprocessed DataFrame with smoothed values
        """
        if df.empty:
            return df
            
        df = df.copy()
        df = df.sort_values(date_col).reset_index(drop=True)
        
        for col in target_cols:
            if col not in df.columns:
                continue
                
            # Step 1: Remove outliers
            df[col] = self._remove_outliers(df[col])
            
            # Step 2: Gap filling
            df[col] = self._fill_gaps(df[col])
            
            # Step 3: Savitzky-Golay smoothing
            df[f'{col}_smooth'] = self._apply_savgol(df[col])
            
        # Add quality flag
        df['quality_flag'] = self._calculate_quality_flag(df, target_cols)
        
        logger.info(f"Preprocessed {len(df)} observations")
        return df
    
    def _remove_outliers(self, series: pd.Series) -> pd.Series:
        """Remove outliers using Z-score method."""
        if series.dropna().empty:
            return series
            
        mean = series.mean()
        std = series.std()
        
        if std == 0:
            return series
            
        z_scores = np.abs((series - mean) / std)
        mask = z_scores > self.outlier_threshold
        
        # Replace outliers with NaN
        result = series.copy()
        result[mask] = np.nan
        
        outliers_removed = mask.sum()
        if outliers_removed > 0:
            logger.debug(f"Removed {outliers_removed} outliers")
            
        return result
    
    def _fill_gaps(
        self,
        series: pd.Series,
        method: str = 'linear',
        max_gap: int = 5
    ) -> pd.Series:
        """
        Fill gaps in time series using interpolation.
        
        Args:
            series: Input series with gaps (NaN)
            method: Interpolation method ('linear', 'spline', 'nearest')
            max_gap: Maximum consecutive NaN to fill
            
        Returns:
            Gap-filled series
        """
        if series.dropna().empty:
            return series
            
        # Limit gap filling to max_gap consecutive values
        result = series.copy()
        
        if method == 'spline':
            # Cubic spline interpolation
            valid_idx = result.dropna().index
            valid_vals = result.dropna().values
            
            if len(valid_idx) < 4:
                method = 'linear'  # Fallback to linear
            else:
                f = interp1d(valid_idx, valid_vals, kind='cubic', 
                            fill_value='extrapolate', bounds_error=False)
                result = pd.Series(f(range(len(result))), index=result.index)
        
        if method in ['linear', 'nearest']:
            result = result.interpolate(method=method, limit=max_gap)
        
        # Fill any remaining NaN at edges
        result = result.fillna(method='ffill').fillna(method='bfill')
        
        return result
    
    def _apply_savgol(self, series: pd.Series) -> pd.Series:
        """
        Apply Savitzky-Golay filter for smoothing.
        
        Preserves peaks and valleys better than moving average.
        """
        if series.dropna().empty or len(series) < self.savgol_window:
            return series
            
        # Handle NaN by filling temporarily
        filled = series.fillna(method='ffill').fillna(method='bfill')
        
        try:
            smoothed = savgol_filter(
                filled.values,
                window_length=min(self.savgol_window, len(filled)),
                polyorder=min(self.savgol_order, self.savgol_window - 1)
            )
            return pd.Series(smoothed, index=series.index)
        except Exception as e:
            logger.warning(f"Savitzky-Golay filter failed: {e}")
            return series
    
    def _calculate_quality_flag(
        self,
        df: pd.DataFrame,
        target_cols: List[str]
    ) -> pd.Series:
        """
        Calculate quality flag for each observation.
        
        Flags:
        0 = Good quality
        1 = Some missing data (interpolated)
        2 = High cloud/missing data
        3 = Poor quality (many gaps)
        """
        flags = np.zeros(len(df))
        
        for col in target_cols:
            if col not in df.columns:
                continue
                
            # Check for originally missing values
            original_null = df[col].isna()
            flags[original_null] += 1
        
        # Normalize to 0-3 scale
        max_missing = len(target_cols)
        flags = np.clip(flags, 0, 3)
        
        return pd.Series(flags.astype(int), index=df.index)
    
    def extract_phenological_features(
        self,
        df: pd.DataFrame,
        ndvi_col: str = 'ndvi_smooth',
        date_col: str = 'date'
    ) -> Dict:
        """
        Extract phenological features from NDVI time series.
        
        Features:
        - Peak NDVI and date
        - Start/end of season
        - Greenup rate
        - Senescence rate
        - Season length
        """
        if df.empty or ndvi_col not in df.columns:
            return {}
        
        ndvi = df[ndvi_col].values
        dates = df[date_col].values
        
        # Peak detection
        peak_idx = np.argmax(ndvi)
        peak_ndvi = ndvi[peak_idx]
        peak_date = dates[peak_idx]
        
        # Threshold for season (20% of max)
        threshold = 0.2 * peak_ndvi + 0.1  # Baseline adjustment
        
        # Start of season (first crossing above threshold)
        above_threshold = ndvi > threshold
        sos_idx = np.argmax(above_threshold) if above_threshold.any() else 0
        
        # End of season (last crossing above threshold after peak)
        post_peak = above_threshold.copy()
        post_peak[:peak_idx] = True  # Keep everything before peak
        eos_idx = len(ndvi) - 1 - np.argmax(post_peak[::-1]) if post_peak.any() else len(ndvi) - 1
        
        # Calculate rates
        if peak_idx > sos_idx:
            greenup_days = (peak_idx - sos_idx) * 5  # Assuming 5-day interval
            greenup_rate = (peak_ndvi - ndvi[sos_idx]) / greenup_days if greenup_days > 0 else 0
        else:
            greenup_rate = 0
            
        if eos_idx > peak_idx:
            senescence_days = (eos_idx - peak_idx) * 5
            senescence_rate = (peak_ndvi - ndvi[eos_idx]) / senescence_days if senescence_days > 0 else 0
        else:
            senescence_rate = 0
        
        # Area under curve (integral)
        auc = np.trapz(ndvi, dx=5)  # 5-day intervals
        
        return {
            'peak_ndvi': float(peak_ndvi),
            'peak_date': pd.Timestamp(peak_date).isoformat() if pd.notna(peak_date) else None,
            'peak_doy': pd.Timestamp(peak_date).dayofyear if pd.notna(peak_date) else None,
            'sos_date': pd.Timestamp(dates[sos_idx]).isoformat() if sos_idx < len(dates) else None,
            'eos_date': pd.Timestamp(dates[eos_idx]).isoformat() if eos_idx < len(dates) else None,
            'season_length_days': int((eos_idx - sos_idx) * 5),
            'greenup_rate': float(greenup_rate),
            'senescence_rate': float(senescence_rate),
            'ndvi_auc': float(auc),
            'ndvi_mean': float(ndvi.mean()),
            'ndvi_std': float(ndvi.std()),
            'ndvi_min': float(ndvi.min()),
            'ndvi_max': float(ndvi.max())
        }


class DataMerger:
    """
    Merges multiple data sources (satellite, weather, soil)
    into a unified dataset for model input.
    """
    
    def merge_timeseries(
        self,
        s2_df: pd.DataFrame,
        s1_df: pd.DataFrame,
        weather_df: pd.DataFrame,
        date_col: str = 'date'
    ) -> pd.DataFrame:
        """
        Merge Sentinel-2, Sentinel-1, and weather time series.
        
        Args:
            s2_df: Sentinel-2 DataFrame (NDVI, EVI, NDWI)
            s1_df: Sentinel-1 DataFrame (VV, VH, RVI)
            weather_df: Weather DataFrame
            date_col: Date column name
            
        Returns:
            Merged DataFrame aligned to S2 dates
        """
        if s2_df.empty:
            return pd.DataFrame()
        
        # Ensure date columns are datetime
        s2_df = s2_df.copy()
        s2_df[date_col] = pd.to_datetime(s2_df[date_col])
        
        # Start with S2 as base (primary observation frequency)
        merged = s2_df.copy()
        
        # Merge S1 data
        if not s1_df.empty:
            s1_df = s1_df.copy()
            s1_df[date_col] = pd.to_datetime(s1_df[date_col])
            
            # Merge on nearest date
            merged = pd.merge_asof(
                merged.sort_values(date_col),
                s1_df.sort_values(date_col),
                on=date_col,
                direction='nearest',
                tolerance=pd.Timedelta('7 days'),
                suffixes=('', '_sar')
            )
        
        # Merge weather data
        if not weather_df.empty:
            weather_df = weather_df.copy()
            weather_df[date_col] = pd.to_datetime(weather_df[date_col])
            
            # For weather, use exact or preceding date
            merged = pd.merge_asof(
                merged.sort_values(date_col),
                weather_df.sort_values(date_col),
                on=date_col,
                direction='backward',
                tolerance=pd.Timedelta('1 day'),
                suffixes=('', '_wx')
            )
        
        merged = merged.sort_values(date_col).reset_index(drop=True)
        logger.info(f"Merged {len(merged)} observations from multiple sources")
        
        return merged
    
    def add_soil_features(
        self,
        df: pd.DataFrame,
        soil_data: Dict
    ) -> pd.DataFrame:
        """
        Add static soil features to time series data.
        
        Args:
            df: Time series DataFrame
            soil_data: Dictionary with soil properties
            
        Returns:
            DataFrame with soil columns added
        """
        if df.empty:
            return df
            
        df = df.copy()
        
        # Add soil properties as constant columns
        soil_features = [
            'ph', 'organic_carbon_pct', 'nitrogen_kg_ha',
            'phosphorus_kg_ha', 'potassium_kg_ha', 'water_holding_pct'
        ]
        
        for feature in soil_features:
            if feature in soil_data:
                df[f'soil_{feature}'] = soil_data[feature]
        
        # Encode drainage as numeric
        drainage_map = {'good': 3, 'moderate': 2, 'poor': 1, 'excessive': 2}
        if 'drainage' in soil_data:
            df['soil_drainage_score'] = drainage_map.get(soil_data['drainage'], 2)
        
        return df


def main():
    """Test the preprocessing module."""
    # Create sample noisy data
    np.random.seed(42)
    dates = pd.date_range('2024-06-01', '2024-11-30', freq='5D')
    n = len(dates)
    
    # Simulate NDVI with noise and gaps
    days = np.arange(n) * 5
    ndvi_true = 0.85 / (1 + np.exp(-0.08 * (days - 35))) - np.maximum(0, (days - 70) * 0.005)
    ndvi_noisy = ndvi_true + np.random.normal(0, 0.05, n)
    
    # Add some gaps
    ndvi_noisy[5:7] = np.nan
    ndvi_noisy[15] = np.nan
    ndvi_noisy[20] = 1.2  # Outlier
    
    df = pd.DataFrame({
        'date': dates,
        'ndvi': ndvi_noisy,
        'evi': ndvi_noisy * 0.9,
        'ndwi': 0.3 * np.sin(np.pi * days / 90)
    })
    
    # Preprocess
    preprocessor = TimeSeriesPreprocessor()
    df_clean = preprocessor.preprocess_vegetation_series(df)
    
    print("\n=== Original vs Preprocessed ===")
    print(df_clean[['date', 'ndvi', 'ndvi_smooth', 'quality_flag']].head(15))
    
    # Extract features
    features = preprocessor.extract_phenological_features(df_clean)
    print("\n=== Phenological Features ===")
    for k, v in features.items():
        print(f"  {k}: {v}")
    
    return df_clean


if __name__ == "__main__":
    main()
