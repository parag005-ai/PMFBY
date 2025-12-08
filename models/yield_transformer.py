"""
PMFBY Yield Prediction Engine
Transformer-based Yield Prediction Model

Time-series yield prediction using Transformer architecture
with multi-head attention for temporal patterns.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
import logging
import json
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Check for PyTorch availability
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import Dataset, DataLoader
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    logger.warning("PyTorch not available. Using fallback empirical model.")


if TORCH_AVAILABLE:
    class PositionalEncoding(nn.Module):
        """Sinusoidal positional encoding for sequence inputs."""
        
        def __init__(self, d_model: int, max_len: int = 100, dropout: float = 0.1):
            super().__init__()
            self.dropout = nn.Dropout(p=dropout)
            
            # Create positional encoding matrix
            pe = torch.zeros(max_len, d_model)
            position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
            div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
            
            pe[:, 0::2] = torch.sin(position * div_term)
            pe[:, 1::2] = torch.cos(position * div_term)
            pe = pe.unsqueeze(0)  # [1, max_len, d_model]
            
            self.register_buffer('pe', pe)
        
        def forward(self, x):
            """Add positional encoding to input."""
            x = x + self.pe[:, :x.size(1), :]
            return self.dropout(x)

    class YieldTransformerEncoder(nn.Module):
        """
        Transformer Encoder for yield prediction.
        
        Input: [batch, seq_len, features]
        Output: [batch, 4] -> (yield_pred, yield_low, yield_high, confidence)
        """
        
        def __init__(
            self,
            input_dim: int = 12,
            d_model: int = 256,
            nhead: int = 8,
            num_layers: int = 4,
            dim_feedforward: int = 512,
            dropout: float = 0.2,
            max_len: int = 50
        ):
            super().__init__()
            
            self.d_model = d_model
            
            # Input projection
            self.input_proj = nn.Linear(input_dim, d_model)
            
            # Positional encoding
            self.pos_encoder = PositionalEncoding(d_model, max_len, dropout)
            
            # Transformer encoder layers
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                batch_first=True
            )
            self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers)
            
            # Global average pooling + output heads
            self.fc1 = nn.Linear(d_model, 128)
            self.fc2 = nn.Linear(128, 64)
            
            # Output heads
            self.yield_head = nn.Linear(64, 1)  # Predicted yield
            self.quantile_head = nn.Linear(64, 2)  # Low (10%), High (90%)
            self.confidence_head = nn.Linear(64, 1)  # Confidence score
            
            self.dropout = nn.Dropout(dropout)
            
        def forward(self, x, return_attention=False):
            """
            Forward pass.
            
            Args:
                x: Input tensor [batch, seq_len, features]
                
            Returns:
                Dictionary with yield predictions
            """
            # Input projection
            x = self.input_proj(x)
            
            # Add positional encoding
            x = self.pos_encoder(x)
            
            # Transformer encoding
            x = self.transformer_encoder(x)
            
            # Global average pooling
            x = x.mean(dim=1)  # [batch, d_model]
            
            # Dense layers
            x = F.relu(self.fc1(x))
            x = self.dropout(x)
            x = F.relu(self.fc2(x))
            
            # Output heads
            yield_pred = self.yield_head(x).squeeze(-1)
            quantiles = self.quantile_head(x)
            confidence = torch.sigmoid(self.confidence_head(x)).squeeze(-1)
            
            return {
                'yield_pred': yield_pred,
                'yield_low_10': quantiles[:, 0],
                'yield_high_90': quantiles[:, 1],
                'confidence': confidence
            }

    class YieldDataset(Dataset):
        """Dataset for yield prediction training."""
        
        def __init__(self, sequences: np.ndarray, targets: np.ndarray):
            self.sequences = torch.FloatTensor(sequences)
            self.targets = torch.FloatTensor(targets)
        
        def __len__(self):
            return len(self.sequences)
        
        def __getitem__(self, idx):
            return self.sequences[idx], self.targets[idx]


class YieldPredictor:
    """
    Main yield prediction class with Transformer or fallback model.
    
    Uses Transformer if PyTorch available, else empirical model.
    """
    
    # India-calibrated crop parameters
    CROP_PARAMS = {
        'rice': {
            'base_yield': 2800,
            'max_yield': 5500,
            'ndvi_coefficient': 4500,
            'stress_sensitivity': 0.35,
            'critical_gdd': 1800
        },
        'wheat': {
            'base_yield': 2500,
            'max_yield': 5800,
            'ndvi_coefficient': 5200,
            'stress_sensitivity': 0.30,
            'critical_gdd': 1600
        },
        'cotton': {
            'base_yield': 400,
            'max_yield': 800,
            'ndvi_coefficient': 600,
            'stress_sensitivity': 0.25,
            'critical_gdd': 2200
        },
        'soybean': {
            'base_yield': 900,
            'max_yield': 2000,
            'ndvi_coefficient': 1800,
            'stress_sensitivity': 0.30,
            'critical_gdd': 1400
        },
        'maize': {
            'base_yield': 2200,
            'max_yield': 5000,
            'ndvi_coefficient': 4800,
            'stress_sensitivity': 0.32,
            'critical_gdd': 1500
        }
    }
    
    def __init__(
        self,
        crop_type: str,
        model_path: Optional[str] = None,
        use_transformer: bool = True
    ):
        """
        Initialize yield predictor.
        
        Args:
            crop_type: Crop type (rice, wheat, cotton, soybean, maize)
            model_path: Path to pre-trained model weights
            use_transformer: Whether to use Transformer (if available)
        """
        self.crop_type = crop_type.lower()
        self.params = self.CROP_PARAMS.get(self.crop_type, self.CROP_PARAMS['rice'])
        
        self.use_transformer = use_transformer and TORCH_AVAILABLE
        self.model = None
        
        if self.use_transformer:
            self._init_transformer(model_path)
        else:
            logger.info(f"Using empirical yield model for {self.crop_type}")
    
    def _init_transformer(self, model_path: Optional[str] = None):
        """Initialize Transformer model."""
        self.model = YieldTransformerEncoder(
            input_dim=12,
            d_model=256,
            nhead=8,
            num_layers=4,
            dropout=0.2
        )
        
        if model_path and os.path.exists(model_path):
            try:
                self.model.load_state_dict(torch.load(model_path))
                logger.info(f"Loaded model weights from {model_path}")
            except Exception as e:
                logger.warning(f"Could not load model weights: {e}")
        
        self.model.eval()
        logger.info("Initialized Transformer yield prediction model")
    
    def predict(
        self,
        features: Dict,
        mc_dropout: bool = True,
        n_samples: int = 50
    ) -> Dict:
        """
        Predict yield from features.
        
        Args:
            features: Dictionary with extracted features
            mc_dropout: Use Monte Carlo dropout for uncertainty
            n_samples: Number of MC samples
            
        Returns:
            Dictionary with predictions and uncertainty
        """
        if self.use_transformer and 'sequence' in features:
            return self._predict_transformer(features, mc_dropout, n_samples)
        else:
            return self._predict_empirical(features)
    
    def _predict_transformer(
        self,
        features: Dict,
        mc_dropout: bool,
        n_samples: int
    ) -> Dict:
        """Predict using Transformer model."""
        sequence = features['sequence']
        
        # Ensure correct shape [1, seq_len, features]
        if sequence.ndim == 2:
            sequence = sequence[np.newaxis, :]
        
        x = torch.FloatTensor(sequence)
        
        if mc_dropout:
            # Monte Carlo dropout for uncertainty
            self.model.train()  # Enable dropout
            predictions = []
            
            with torch.no_grad():
                for _ in range(n_samples):
                    output = self.model(x)
                    predictions.append({
                        'yield': output['yield_pred'].item(),
                        'low': output['yield_low_10'].item(),
                        'high': output['yield_high_90'].item()
                    })
            
            self.model.eval()
            
            yields = [p['yield'] for p in predictions]
            yield_pred = np.mean(yields)
            yield_std = np.std(yields)
            yield_low = np.percentile(yields, 10)
            yield_high = np.percentile(yields, 90)
            confidence = 1 - (yield_std / max(yield_pred, 1))
            
        else:
            self.model.eval()
            with torch.no_grad():
                output = self.model(x)
            
            yield_pred = output['yield_pred'].item()
            yield_low = output['yield_low_10'].item()
            yield_high = output['yield_high_90'].item()
            confidence = output['confidence'].item()
        
        # Post-process to ensure reasonable bounds
        yield_pred = np.clip(yield_pred, 0, self.params['max_yield'] * 1.2)
        yield_low = np.clip(yield_low, 0, yield_pred)
        yield_high = np.clip(yield_high, yield_pred, self.params['max_yield'] * 1.3)
        
        return {
            'yield_pred': round(yield_pred, 2),
            'yield_low_10': round(yield_low, 2),
            'yield_high_90': round(yield_high, 2),
            'confidence_score': round(np.clip(confidence, 0, 1), 3),
            'model_type': 'transformer',
            'units': 'kg/ha'
        }
    
    def _predict_empirical(self, features: Dict) -> Dict:
        """Predict using empirical LAI-NDVI model."""
        params = self.params
        
        # Extract key features
        ndvi_peak = features.get('ndvi_peak', 0.7)
        ndvi_mean = features.get('ndvi_mean', 0.5)
        ndvi_auc = features.get('ndvi_auc', 10)
        gdd_total = features.get('gdd_total', params['critical_gdd'])
        stress_mean = features.get('combined_stress_mean', 0.2)
        soil_score = features.get('soil_organic_carbon_pct', 0.5) / 0.75 * 100
        
        # Base yield from NDVI
        ndvi_factor = (ndvi_peak + ndvi_mean) / 2
        base_yield = params['base_yield'] + params['ndvi_coefficient'] * ndvi_factor
        
        # GDD adjustment
        gdd_ratio = min(1.0, gdd_total / params['critical_gdd'])
        base_yield *= (0.7 + 0.3 * gdd_ratio)
        
        # Stress penalty
        stress_penalty = 1 - (stress_mean * params['stress_sensitivity'])
        base_yield *= max(0.5, stress_penalty)
        
        # Soil adjustment
        soil_factor = 0.9 + 0.1 * (soil_score / 100)
        base_yield *= soil_factor
        
        # Clip to reasonable range
        yield_pred = np.clip(base_yield, params['base_yield'] * 0.3, params['max_yield'])
        
        # Uncertainty estimation
        uncertainty = 0.15 + 0.1 * stress_mean  # 15-25% uncertainty
        yield_low = yield_pred * (1 - uncertainty)
        yield_high = yield_pred * (1 + uncertainty)
        
        # Confidence based on data quality
        confidence = 0.7 - 0.2 * stress_mean
        
        return {
            'yield_pred': round(yield_pred, 2),
            'yield_low_10': round(yield_low, 2),
            'yield_high_90': round(yield_high, 2),
            'confidence_score': round(np.clip(confidence, 0.3, 0.9), 3),
            'model_type': 'empirical',
            'units': 'kg/ha'
        }
    
    def calculate_pmfby_loss(
        self,
        yield_pred: float,
        threshold_yield: float
    ) -> Dict:
        """
        Calculate PMFBY loss percentage.
        
        Args:
            yield_pred: Predicted yield (kg/ha)
            threshold_yield: Threshold yield for insurance (kg/ha)
            
        Returns:
            Dictionary with loss calculation
        """
        if threshold_yield <= 0:
            return {'error': 'Invalid threshold yield'}
        
        shortfall = max(0, threshold_yield - yield_pred)
        loss_pct = (shortfall / threshold_yield) * 100
        
        # Claim trigger thresholds
        claim_trigger = loss_pct >= 33  # 33% trigger for most states
        
        return {
            'threshold_yield': round(threshold_yield, 2),
            'predicted_yield': round(yield_pred, 2),
            'shortfall_kg_ha': round(shortfall, 2),
            'loss_percentage': round(loss_pct, 2),
            'claim_trigger': claim_trigger,
            'trigger_threshold': 33.0
        }


def main():
    """Test yield prediction."""
    # Sample features
    features = {
        'ndvi_peak': 0.78,
        'ndvi_mean': 0.55,
        'ndvi_auc': 12.5,
        'gdd_total': 1650,
        'combined_stress_mean': 0.25,
        'soil_organic_carbon_pct': 0.52,
        'sequence': np.random.randn(36, 12)  # Random sequence for testing
    }
    
    # Test for different crops
    for crop in ['rice', 'wheat', 'cotton']:
        print(f"\n=== {crop.upper()} Yield Prediction ===")
        
        predictor = YieldPredictor(crop, use_transformer=False)  # Use empirical for testing
        result = predictor.predict(features)
        
        print(f"  Predicted: {result['yield_pred']} {result['units']}")
        print(f"  Range: {result['yield_low_10']} - {result['yield_high_90']} {result['units']}")
        print(f"  Confidence: {result['confidence_score']}")
        print(f"  Model: {result['model_type']}")
        
        # PMFBY loss calculation
        threshold = predictor.params['base_yield'] * 1.2
        loss = predictor.calculate_pmfby_loss(result['yield_pred'], threshold)
        print(f"\n  PMFBY Analysis:")
        print(f"    Threshold: {loss['threshold_yield']} kg/ha")
        print(f"    Loss: {loss['loss_percentage']}%")
        print(f"    Claim Trigger: {loss['claim_trigger']}")
    
    return features


if __name__ == "__main__":
    main()
