"""
PMFBY Yield Prediction Engine
Stress Indices Module

Computes crop stress indices from weather and satellite data:
- NDWI-based moisture stress
- VPD heat stress
- Water balance stress
- Stage-wise stress aggregation
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class StressIndexCalculator:
    """
    Calculates various crop stress indices for yield prediction.
    
    Stress types:
    1. Moisture stress (NDWI-based)
    2. Heat stress (VPD, temperature)
    3. Water balance stress (Rain - ET deficit)
    4. Combined stress index
    """
    
    # Stress thresholds (crop-specific adjustments available)
    THRESHOLDS = {
        'heat': {
            'temp_critical': 35,      # °C
            'temp_severe': 40,        # °C
            'vpd_moderate': 2.0,      # kPa
            'vpd_severe': 3.0         # kPa
        },
        'moisture': {
            'ndwi_stress': -0.1,      # NDWI below this = stressed
            'ndwi_severe': -0.25      # Severe moisture stress
        },
        'water_balance': {
            'deficit_moderate': -30,  # mm cumulative
            'deficit_severe': -60     # mm cumulative
        }
    }
    
    # Critical stage weights (flowering most critical)
    STAGE_WEIGHTS = {
        'vegetative': 0.8,
        'tillering': 0.9,
        'flowering': 1.5,     # Most critical
        'grain_filling': 1.3,
        'maturity': 0.5
    }
    
    def __init__(self, crop_type: str = 'rice'):
        """
        Initialize stress calculator.
        
        Args:
            crop_type: Crop type for threshold adjustments
        """
        self.crop_type = crop_type.lower()
        self._adjust_thresholds()
        
    def _adjust_thresholds(self):
        """Adjust thresholds based on crop type."""
        crop_adjustments = {
            'rice': {'temp_critical': 35, 'ndwi_stress': 0.0},   # Rice likes water
            'wheat': {'temp_critical': 30, 'ndwi_stress': -0.15},
            'cotton': {'temp_critical': 38, 'ndwi_stress': -0.2},  # Cotton tolerates heat
            'soybean': {'temp_critical': 33, 'ndwi_stress': -0.1},
            'maize': {'temp_critical': 35, 'ndwi_stress': -0.15}
        }
        
        if self.crop_type in crop_adjustments:
            for key, value in crop_adjustments[self.crop_type].items():
                if key.startswith('temp'):
                    self.THRESHOLDS['heat'][key] = value
                elif key.startswith('ndwi'):
                    self.THRESHOLDS['moisture'][key] = value
    
    def calculate_heat_stress(
        self,
        df: pd.DataFrame,
        temp_col: str = 't2m_max',
        vpd_col: str = 'vpd'
    ) -> pd.DataFrame:
        """
        Calculate daily heat stress index.
        
        Heat stress factors:
        1. High temperature (>35°C)
        2. High VPD (>2.5 kPa)
        3. Consecutive hot days
        
        Returns:
            DataFrame with heat stress columns
        """
        df = df.copy()
        thresholds = self.THRESHOLDS['heat']
        
        # Temperature-based stress (0-1 scale)
        if temp_col in df.columns:
            df['heat_stress_temp'] = np.clip(
                (df[temp_col] - thresholds['temp_critical']) / 
                (thresholds['temp_severe'] - thresholds['temp_critical']),
                0, 1
            )
        else:
            df['heat_stress_temp'] = 0
        
        # VPD-based stress
        if vpd_col in df.columns:
            df['heat_stress_vpd'] = np.clip(
                (df[vpd_col] - thresholds['vpd_moderate']) /
                (thresholds['vpd_severe'] - thresholds['vpd_moderate']),
                0, 1
            )
        else:
            df['heat_stress_vpd'] = 0
        
        # Combined heat stress
        df['heat_stress'] = np.maximum(
            df['heat_stress_temp'],
            df['heat_stress_vpd']
        )
        
        # Consecutive hot days (rolling sum)
        df['hot_day_flag'] = (df['heat_stress'] > 0.3).astype(int)
        df['consecutive_hot_days'] = df['hot_day_flag'].rolling(
            window=5, min_periods=1
        ).sum()
        
        return df
    
    def calculate_moisture_stress(
        self,
        df: pd.DataFrame,
        ndwi_col: str = 'ndwi'
    ) -> pd.DataFrame:
        """
        Calculate moisture stress from NDWI.
        
        NDWI interpretation:
        - Positive: Good moisture
        - Near zero: Moderate stress
        - Negative: High stress
        """
        df = df.copy()
        thresholds = self.THRESHOLDS['moisture']
        
        if ndwi_col in df.columns:
            # Normalize NDWI to stress scale (lower NDWI = higher stress)
            df['moisture_stress'] = np.clip(
                (thresholds['ndwi_stress'] - df[ndwi_col]) /
                (thresholds['ndwi_stress'] - thresholds['ndwi_severe']),
                0, 1
            )
            
            # Smoothed moisture stress
            df['moisture_stress_smooth'] = df['moisture_stress'].rolling(
                window=3, min_periods=1
            ).mean()
        else:
            df['moisture_stress'] = 0
            df['moisture_stress_smooth'] = 0
        
        return df
    
    def calculate_water_balance_stress(
        self,
        df: pd.DataFrame,
        rain_col: str = 'prectotcorr',
        et_col: str = 'et0'
    ) -> pd.DataFrame:
        """
        Calculate water balance stress (Rain - ET deficit).
        
        Cumulative water deficit indicates drought stress.
        """
        df = df.copy()
        thresholds = self.THRESHOLDS['water_balance']
        
        if rain_col in df.columns and et_col in df.columns:
            # Daily water balance
            df['water_balance'] = df[rain_col] - df[et_col]
            
            # Cumulative water balance
            df['water_balance_cum'] = df['water_balance'].cumsum()
            
            # Stress based on cumulative deficit
            df['water_balance_stress'] = np.clip(
                (thresholds['deficit_moderate'] - df['water_balance_cum']) /
                (thresholds['deficit_moderate'] - thresholds['deficit_severe']),
                0, 1
            )
        else:
            df['water_balance'] = 0
            df['water_balance_cum'] = 0
            df['water_balance_stress'] = 0
        
        return df
    
    def calculate_combined_stress(
        self,
        df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Calculate combined stress index from all stress types.
        
        Combined stress = weighted average of individual stresses.
        """
        df = df.copy()
        
        stress_cols = ['heat_stress', 'moisture_stress', 'water_balance_stress']
        weights = [0.35, 0.35, 0.30]  # Weights for each stress type
        
        available_cols = [c for c in stress_cols if c in df.columns]
        
        if available_cols:
            # Calculate weighted average
            stress_sum = np.zeros(len(df))
            weight_sum = 0
            
            for col, weight in zip(stress_cols, weights):
                if col in df.columns:
                    stress_sum += df[col].fillna(0).values * weight
                    weight_sum += weight
            
            df['combined_stress'] = stress_sum / weight_sum if weight_sum > 0 else 0
            
            # Stress severity category
            df['stress_severity'] = pd.cut(
                df['combined_stress'],
                bins=[-0.01, 0.2, 0.4, 0.6, 1.01],
                labels=['LOW', 'MODERATE', 'HIGH', 'SEVERE']
            )
        else:
            df['combined_stress'] = 0
            df['stress_severity'] = 'LOW'
        
        return df
    
    def calculate_stagewise_stress(
        self,
        df: pd.DataFrame,
        sowing_date: str,
        stages: Dict[str, Tuple[int, int]],
        date_col: str = 'date'
    ) -> Dict:
        """
        Aggregate stress by crop growth stage.
        
        Args:
            df: DataFrame with stress indices
            sowing_date: Sowing date string
            stages: Dict of stage_name -> (start_day, end_day)
            date_col: Date column name
            
        Returns:
            Dictionary with stage-wise stress summary
        """
        df = df.copy()
        sowing = pd.to_datetime(sowing_date)
        df['days_after_sowing'] = (pd.to_datetime(df[date_col]) - sowing).dt.days
        
        stagewise_stress = {}
        
        for stage_name, (start_day, end_day) in stages.items():
            stage_df = df[(df['days_after_sowing'] >= start_day) & 
                          (df['days_after_sowing'] < end_day)]
            
            if stage_df.empty:
                continue
            
            # Get stage weight
            weight = self.STAGE_WEIGHTS.get(stage_name, 1.0)
            
            # Calculate weighted stress for this stage
            heat_stress = stage_df['heat_stress'].mean() if 'heat_stress' in stage_df else 0
            moisture_stress = stage_df['moisture_stress'].mean() if 'moisture_stress' in stage_df else 0
            water_stress = stage_df['water_balance_stress'].mean() if 'water_balance_stress' in stage_df else 0
            combined_stress = stage_df['combined_stress'].mean() if 'combined_stress' in stage_df else 0
            
            # Days with high stress
            high_stress_days = (stage_df['combined_stress'] > 0.5).sum() if 'combined_stress' in stage_df else 0
            
            stagewise_stress[stage_name] = {
                'period_days': len(stage_df),
                'weight': weight,
                'heat_stress_mean': round(float(heat_stress), 3),
                'moisture_stress_mean': round(float(moisture_stress), 3),
                'water_balance_stress_mean': round(float(water_stress), 3),
                'combined_stress_mean': round(float(combined_stress), 3),
                'weighted_stress': round(float(combined_stress * weight), 3),
                'high_stress_days': int(high_stress_days),
                'max_stress': round(float(stage_df['combined_stress'].max()) if 'combined_stress' in stage_df else 0, 3)
            }
        
        # Overall weighted stress
        total_weighted = sum(s['weighted_stress'] for s in stagewise_stress.values())
        total_weight = sum(s['weight'] for s in stagewise_stress.values())
        
        stagewise_stress['overall'] = {
            'weighted_average_stress': round(total_weighted / total_weight, 3) if total_weight > 0 else 0,
            'total_high_stress_days': sum(s['high_stress_days'] for s in stagewise_stress.values() if 'high_stress_days' in s),
            'most_stressed_stage': max(
                [(k, v['weighted_stress']) for k, v in stagewise_stress.items() if k != 'overall'],
                key=lambda x: x[1],
                default=('none', 0)
            )[0]
        }
        
        return stagewise_stress
    
    def generate_stress_explanation(
        self,
        stagewise_stress: Dict,
        threshold: float = 0.3
    ) -> str:
        """
        Generate human-readable stress explanation for insurance report.
        
        Args:
            stagewise_stress: Stage-wise stress dictionary
            threshold: Stress threshold for reporting
            
        Returns:
            Explanation string
        """
        explanations = []
        
        for stage, data in stagewise_stress.items():
            if stage == 'overall':
                continue
            
            if data['weighted_stress'] > threshold:
                # Determine dominant stress type
                if data['heat_stress_mean'] >= data['moisture_stress_mean'] and \
                   data['heat_stress_mean'] >= data['water_balance_stress_mean']:
                    stress_type = "heat stress"
                elif data['moisture_stress_mean'] >= data['water_balance_stress_mean']:
                    stress_type = "moisture stress"
                else:
                    stress_type = "water deficit"
                
                severity = "severe" if data['weighted_stress'] > 0.6 else "moderate"
                explanations.append(
                    f"{severity.capitalize()} {stress_type} during {stage} stage "
                    f"({data['high_stress_days']} high-stress days)"
                )
        
        if not explanations:
            return "No significant stress events detected during the crop season."
        
        overall = stagewise_stress.get('overall', {})
        main_stage = overall.get('most_stressed_stage', 'unknown')
        
        prefix = f"Yield reduction primarily due to stress during {main_stage}. "
        return prefix + "; ".join(explanations) + "."


def main():
    """Test stress index calculation."""
    # Create sample data
    np.random.seed(42)
    dates = pd.date_range('2024-06-15', periods=130, freq='D')
    
    df = pd.DataFrame({
        'date': dates,
        't2m_max': 30 + 8 * np.sin(np.pi * np.arange(130) / 60) + np.random.normal(0, 2, 130),
        'vpd': 1.5 + np.random.uniform(0, 2, 130),
        'ndwi': 0.2 - 0.3 * np.sin(np.pi * np.arange(130) / 80) + np.random.normal(0, 0.05, 130),
        'prectotcorr': np.random.exponential(8, 130) * (np.random.random(130) > 0.5),
        'et0': 4 + np.random.uniform(0, 2, 130)
    })
    
    # Calculate stresses
    calculator = StressIndexCalculator(crop_type='rice')
    df = calculator.calculate_heat_stress(df)
    df = calculator.calculate_moisture_stress(df)
    df = calculator.calculate_water_balance_stress(df)
    df = calculator.calculate_combined_stress(df)
    
    print("\n=== Stress Indices Sample ===")
    print(df[['date', 'heat_stress', 'moisture_stress', 'water_balance_stress', 
              'combined_stress', 'stress_severity']].head(20))
    
    # Stage-wise stress
    stages = {
        'vegetative': (0, 30),
        'tillering': (30, 55),
        'flowering': (55, 80),
        'grain_filling': (80, 105),
        'maturity': (105, 130)
    }
    
    stagewise = calculator.calculate_stagewise_stress(df, '2024-06-15', stages)
    print("\n=== Stage-wise Stress ===")
    for stage, data in stagewise.items():
        print(f"{stage}: {data}")
    
    # Generate explanation
    explanation = calculator.generate_stress_explanation(stagewise)
    print(f"\n=== Stress Explanation ===\n{explanation}")
    
    return df


if __name__ == "__main__":
    main()
