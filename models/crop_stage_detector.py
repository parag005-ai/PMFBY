"""
PMFBY Yield Prediction Engine
Crop Stage Detection Module

Detects crop growth stages from time series patterns
using rule-based + ML hybrid approach.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CropStageDetector:
    """
    Crop growth stage detection using sowing date and phenological patterns.
    
    Stages detected:
    1. Vegetative (emergence to active tillering)
    2. Tillering/Branching
    3. Flowering/Reproductive
    4. Grain filling/Pod development
    5. Maturity/Senescence
    """
    
    # Phenology database (days after sowing)
    CROP_PHENOLOGY = {
        'rice': {
            'vegetative': {'start': 0, 'end': 30, 'ndvi_range': (0.2, 0.5)},
            'tillering': {'start': 30, 'end': 55, 'ndvi_range': (0.5, 0.75)},
            'flowering': {'start': 55, 'end': 80, 'ndvi_range': (0.75, 0.85)},
            'grain_filling': {'start': 80, 'end': 105, 'ndvi_range': (0.7, 0.85)},
            'maturity': {'start': 105, 'end': 130, 'ndvi_range': (0.3, 0.7)}
        },
        'wheat': {
            'vegetative': {'start': 0, 'end': 35, 'ndvi_range': (0.15, 0.45)},
            'tillering': {'start': 35, 'end': 60, 'ndvi_range': (0.45, 0.7)},
            'flowering': {'start': 60, 'end': 90, 'ndvi_range': (0.7, 0.85)},
            'grain_filling': {'start': 90, 'end': 115, 'ndvi_range': (0.65, 0.8)},
            'maturity': {'start': 115, 'end': 140, 'ndvi_range': (0.25, 0.65)}
        },
        'cotton': {
            'vegetative': {'start': 0, 'end': 45, 'ndvi_range': (0.15, 0.4)},
            'squaring': {'start': 45, 'end': 70, 'ndvi_range': (0.4, 0.6)},
            'flowering': {'start': 70, 'end': 110, 'ndvi_range': (0.6, 0.75)},
            'boll_development': {'start': 110, 'end': 150, 'ndvi_range': (0.5, 0.7)},
            'maturity': {'start': 150, 'end': 180, 'ndvi_range': (0.25, 0.5)}
        },
        'soybean': {
            'vegetative': {'start': 0, 'end': 30, 'ndvi_range': (0.2, 0.5)},
            'flowering': {'start': 30, 'end': 55, 'ndvi_range': (0.6, 0.8)},
            'pod_development': {'start': 55, 'end': 80, 'ndvi_range': (0.7, 0.85)},
            'seed_filling': {'start': 80, 'end': 100, 'ndvi_range': (0.6, 0.8)},
            'maturity': {'start': 100, 'end': 110, 'ndvi_range': (0.2, 0.5)}
        },
        'maize': {
            'vegetative': {'start': 0, 'end': 35, 'ndvi_range': (0.2, 0.55)},
            'tasseling': {'start': 35, 'end': 55, 'ndvi_range': (0.6, 0.8)},
            'silking': {'start': 55, 'end': 70, 'ndvi_range': (0.75, 0.9)},
            'grain_filling': {'start': 70, 'end': 100, 'ndvi_range': (0.65, 0.85)},
            'maturity': {'start': 100, 'end': 120, 'ndvi_range': (0.25, 0.6)}
        }
    }
    
    def __init__(self, crop_type: str):
        """
        Initialize stage detector.
        
        Args:
            crop_type: Type of crop (rice, wheat, cotton, soybean, maize)
        """
        self.crop_type = crop_type.lower()
        
        if self.crop_type not in self.CROP_PHENOLOGY:
            logger.warning(f"Unknown crop '{crop_type}'. Defaulting to rice phenology.")
            self.crop_type = 'rice'
        
        self.phenology = self.CROP_PHENOLOGY[self.crop_type]
        
    def get_stage_boundaries(self) -> Dict[str, Tuple[int, int]]:
        """Get stage boundaries as (start_day, end_day) tuples."""
        return {
            stage: (info['start'], info['end'])
            for stage, info in self.phenology.items()
        }
    
    def detect_stage(
        self,
        days_after_sowing: int
    ) -> str:
        """
        Detect crop stage from days after sowing.
        
        Args:
            days_after_sowing: Number of days since sowing
            
        Returns:
            Stage name
        """
        for stage, info in self.phenology.items():
            if info['start'] <= days_after_sowing < info['end']:
                return stage
        
        # Beyond maturity
        if days_after_sowing >= max(info['end'] for info in self.phenology.values()):
            return 'harvested'
        
        return 'pre_sowing'
    
    def add_stage_labels(
        self,
        df: pd.DataFrame,
        sowing_date: str,
        date_col: str = 'date'
    ) -> pd.DataFrame:
        """
        Add stage labels to time series DataFrame.
        
        Args:
            df: DataFrame with date column
            sowing_date: Sowing date string
            date_col: Date column name
            
        Returns:
            DataFrame with stage column added
        """
        df = df.copy()
        sowing = pd.to_datetime(sowing_date)
        df['days_after_sowing'] = (pd.to_datetime(df[date_col]) - sowing).dt.days
        df['crop_stage'] = df['days_after_sowing'].apply(self.detect_stage)
        
        # One-hot encode stages
        stage_names = list(self.phenology.keys())
        for stage in stage_names:
            df[f'stage_{stage}'] = (df['crop_stage'] == stage).astype(int)
        
        return df
    
    def detect_stage_from_ndvi(
        self,
        df: pd.DataFrame,
        ndvi_col: str = 'ndvi',
        sowing_date: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Detect stages using NDVI patterns (hybrid approach).
        
        Uses both timing (if sowing_date provided) and NDVI values.
        """
        df = df.copy()
        
        if sowing_date:
            df = self.add_stage_labels(df, sowing_date)
        
        if ndvi_col not in df.columns:
            return df
        
        # Validate stages with NDVI ranges
        for stage, info in self.phenology.items():
            stage_mask = df['crop_stage'] == stage
            if not stage_mask.any():
                continue
                
            ndvi_low, ndvi_high = info['ndvi_range']
            ndvi_vals = df.loc[stage_mask, ndvi_col]
            
            # Confidence based on NDVI match
            confidence = np.clip(
                1 - np.abs(ndvi_vals - (ndvi_low + ndvi_high) / 2) / 0.5,
                0, 1
            )
            df.loc[stage_mask, 'stage_confidence'] = confidence
        
        return df
    
    def detect_phenological_events(
        self,
        df: pd.DataFrame,
        ndvi_col: str = 'ndvi_smooth'
    ) -> Dict:
        """
        Detect key phenological events from NDVI curve.
        
        Events:
        - Emergence (NDVI starts rising)
        - Peak vegetative (NDVI maximum)
        - Flowering (NDVI derivative peak)
        - Senescence onset (NDVI starts declining)
        """
        if ndvi_col not in df.columns or df.empty:
            return {}
        
        ndvi = df[ndvi_col].values
        dates = df['date'].values if 'date' in df.columns else np.arange(len(ndvi))
        
        # Calculate derivatives
        ndvi_diff = np.gradient(ndvi)
        ndvi_diff2 = np.gradient(ndvi_diff)
        
        events = {}
        
        # Peak NDVI (maximum)
        peak_idx = np.argmax(ndvi)
        events['peak_vegetation'] = {
            'index': int(peak_idx),
            'date': str(dates[peak_idx]) if hasattr(dates[0], 'isoformat') else int(peak_idx),
            'ndvi': float(ndvi[peak_idx])
        }
        
        # Emergence (first significant rise)
        threshold = 0.2 * (ndvi.max() - ndvi.min())
        above_threshold = ndvi > (ndvi.min() + threshold)
        emergence_idx = np.argmax(above_threshold)
        events['emergence'] = {
            'index': int(emergence_idx),
            'date': str(dates[emergence_idx]) if hasattr(dates[0], 'isoformat') else int(emergence_idx),
            'ndvi': float(ndvi[emergence_idx])
        }
        
        # Senescence onset (after peak, where decline starts)
        post_peak = ndvi_diff[peak_idx:]
        if len(post_peak) > 0:
            senescence_offset = np.argmax(post_peak < -0.01)
            senescence_idx = peak_idx + senescence_offset
            if senescence_idx < len(ndvi):
                events['senescence_onset'] = {
                    'index': int(senescence_idx),
                    'date': str(dates[senescence_idx]) if hasattr(dates[0], 'isoformat') else int(senescence_idx),
                    'ndvi': float(ndvi[senescence_idx])
                }
        
        # Estimate flowering (around 70% of peak timing for most crops)
        flowering_idx = int(peak_idx * 0.7)
        events['estimated_flowering'] = {
            'index': int(flowering_idx),
            'date': str(dates[flowering_idx]) if hasattr(dates[0], 'isoformat') else int(flowering_idx),
            'ndvi': float(ndvi[flowering_idx])
        }
        
        return events
    
    def get_critical_period(self) -> Tuple[int, int]:
        """Get the critical period for yield (usually flowering + grain filling)."""
        critical_stages = ['flowering', 'silking', 'grain_filling', 'pod_development', 'seed_filling']
        
        start_day = 999
        end_day = 0
        
        for stage, info in self.phenology.items():
            if stage in critical_stages:
                start_day = min(start_day, info['start'])
                end_day = max(end_day, info['end'])
        
        return (start_day, end_day)


def main():
    """Test crop stage detection."""
    # Create sample data
    dates = pd.date_range('2024-06-15', periods=30, freq='5D')
    ndvi = 0.3 + 0.5 * np.sin(np.pi * np.arange(30) / 15)
    
    df = pd.DataFrame({
        'date': dates,
        'ndvi': ndvi,
        'ndvi_smooth': ndvi
    })
    
    # Test stage detection
    detector = CropStageDetector('rice')
    
    print("\n=== Stage Boundaries ===")
    for stage, bounds in detector.get_stage_boundaries().items():
        print(f"  {stage}: day {bounds[0]} - {bounds[1]}")
    
    # Add stage labels
    df = detector.detect_stage_from_ndvi(df, sowing_date='2024-06-15')
    
    print("\n=== Time Series with Stages ===")
    print(df[['date', 'days_after_sowing', 'crop_stage', 'ndvi']].head(20))
    
    # Detect phenological events
    events = detector.detect_phenological_events(df)
    print("\n=== Phenological Events ===")
    for event, data in events.items():
        print(f"  {event}: {data}")
    
    # Critical period
    critical = detector.get_critical_period()
    print(f"\n=== Critical Period ===")
    print(f"  Days {critical[0]} - {critical[1]} after sowing")
    
    return df


if __name__ == "__main__":
    main()
