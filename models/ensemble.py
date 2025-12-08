"""
PMFBY v2.0 - Ensemble Yield Prediction Model
=============================================
Combines Random Forest, XGBoost, and LightGBM for robust predictions
with uncertainty estimation.

Models:
- Random Forest (300 trees)
- XGBoost (500 rounds)
- LightGBM (500 rounds)

Features:
- Weighted ensemble prediction
- Uncertainty from model disagreement
- Confidence intervals from RF tree variance
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional, List
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import pickle
import os
import warnings
warnings.filterwarnings('ignore')

# Try importing XGBoost and LightGBM
try:
    from xgboost import XGBRegressor
    HAS_XGB = True
except ImportError:
    HAS_XGB = False
    print("Warning: XGBoost not installed. Using RF only.")

try:
    from lightgbm import LGBMRegressor
    HAS_LGB = True
except ImportError:
    HAS_LGB = False
    print("Warning: LightGBM not installed. Using RF only.")


class EnsembleYieldPredictor:
    """
    Ensemble yield prediction model with uncertainty estimation.
    
    Uses weighted average of Random Forest, XGBoost, and LightGBM.
    Provides prediction intervals based on model disagreement and RF tree variance.
    """
    
    def __init__(self, 
                 rf_params: Optional[Dict] = None,
                 xgb_params: Optional[Dict] = None,
                 lgb_params: Optional[Dict] = None,
                 weights: Optional[List[float]] = None):
        """
        Initialize ensemble with configurable parameters.
        
        Args:
            rf_params: Random Forest hyperparameters
            xgb_params: XGBoost hyperparameters
            lgb_params: LightGBM hyperparameters
            weights: Ensemble weights [RF, XGB, LGB]
        """
        # Default RF parameters
        self.rf_params = rf_params or {
            'n_estimators': 300,
            'max_depth': 20,
            'min_samples_split': 5,
            'min_samples_leaf': 2,
            'max_features': 'sqrt',
            'n_jobs': -1,
            'random_state': 42
        }
        
        # Default XGBoost parameters
        self.xgb_params = xgb_params or {
            'n_estimators': 500,
            'max_depth': 8,
            'learning_rate': 0.05,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'reg_alpha': 0.1,
            'reg_lambda': 1.0,
            'random_state': 42,
            'n_jobs': -1
        }
        
        # Default LightGBM parameters
        self.lgb_params = lgb_params or {
            'n_estimators': 500,
            'max_depth': 10,
            'learning_rate': 0.05,
            'num_leaves': 31,
            'feature_fraction': 0.8,
            'bagging_fraction': 0.8,
            'reg_alpha': 0.1,
            'reg_lambda': 1.0,
            'random_state': 42,
            'n_jobs': -1,
            'verbose': -1
        }
        
        # Ensemble weights
        self.weights = weights or [0.40, 0.35, 0.25]  # RF, XGB, LGB
        
        # Initialize models
        self.rf = RandomForestRegressor(**self.rf_params)
        self.xgb = XGBRegressor(**self.xgb_params) if HAS_XGB else None
        self.lgb = LGBMRegressor(**self.lgb_params) if HAS_LGB else None
        
        # Metrics storage
        self.metrics = {}
        self.feature_importance = None
        self.is_fitted = False
    
    def fit(self, X: np.ndarray, y: np.ndarray, 
            feature_names: Optional[List[str]] = None,
            verbose: bool = True) -> 'EnsembleYieldPredictor':
        """
        Train all ensemble models.
        
        Args:
            X: Feature matrix (n_samples, n_features)
            y: Target vector (n_samples,)
            feature_names: Optional list of feature names
            verbose: Print training progress
        
        Returns:
            self
        """
        if verbose:
            print("Training Ensemble Model...")
            print(f"  Samples: {X.shape[0]}, Features: {X.shape[1]}")
        
        # Train Random Forest
        if verbose:
            print("\n  [1/3] Training Random Forest (300 trees)...")
        self.rf.fit(X, y)
        
        # Train XGBoost
        if self.xgb is not None:
            if verbose:
                print("  [2/3] Training XGBoost (500 rounds)...")
            self.xgb.fit(X, y)
        
        # Train LightGBM
        if self.lgb is not None:
            if verbose:
                print("  [3/3] Training LightGBM (500 rounds)...")
            self.lgb.fit(X, y)
        
        # Compute feature importance (average across models)
        self._compute_feature_importance(feature_names)
        
        self.is_fitted = True
        
        if verbose:
            print("\n  [OK] Ensemble training complete!")
        
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions using weighted ensemble.
        
        Args:
            X: Feature matrix
        
        Returns:
            Predictions array
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")
        
        # Get predictions from each model
        pred_rf = self.rf.predict(X)
        
        predictions = [pred_rf]
        weights = [self.weights[0]]
        
        if self.xgb is not None:
            pred_xgb = self.xgb.predict(X)
            predictions.append(pred_xgb)
            weights.append(self.weights[1])
        
        if self.lgb is not None:
            pred_lgb = self.lgb.predict(X)
            predictions.append(pred_lgb)
            weights.append(self.weights[2])
        
        # Normalize weights
        weights = np.array(weights) / sum(weights)
        
        # Weighted average
        ensemble_pred = np.zeros_like(pred_rf)
        for pred, weight in zip(predictions, weights):
            ensemble_pred += weight * pred
        
        return ensemble_pred
    
    def predict_with_uncertainty(self, X: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Make predictions with uncertainty estimation.
        
        Uncertainty is estimated from:
        1. Model disagreement (std across RF, XGB, LGB)
        2. RF tree variance (std across 300 trees)
        
        Args:
            X: Feature matrix
        
        Returns:
            Dict with:
                - prediction: Point estimates
                - uncertainty: Standard deviation
                - confidence_low: 95% CI lower bound
                - confidence_high: 95% CI upper bound
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")
        
        # Get predictions from each model
        pred_rf = self.rf.predict(X)
        
        all_predictions = [pred_rf]
        
        if self.xgb is not None:
            all_predictions.append(self.xgb.predict(X))
        
        if self.lgb is not None:
            all_predictions.append(self.lgb.predict(X))
        
        # Stack predictions
        predictions_stack = np.vstack(all_predictions)
        
        # Ensemble prediction (weighted mean)
        weights = np.array(self.weights[:len(all_predictions)])
        weights = weights / weights.sum()
        ensemble_pred = np.average(predictions_stack, axis=0, weights=weights)
        
        # Uncertainty 1: Model disagreement
        model_std = predictions_stack.std(axis=0)
        
        # Uncertainty 2: RF tree variance
        tree_predictions = np.array([tree.predict(X) for tree in self.rf.estimators_])
        rf_std = tree_predictions.std(axis=0)
        
        # Combined uncertainty (average of both)
        total_uncertainty = (model_std + rf_std) / 2
        
        # 95% Confidence intervals
        z_score = 1.96
        conf_low = ensemble_pred - z_score * total_uncertainty
        conf_high = ensemble_pred + z_score * total_uncertainty
        
        return {
            'prediction': ensemble_pred,
            'uncertainty': total_uncertainty,
            'confidence_low': np.maximum(0, conf_low),
            'confidence_high': conf_high,
            'model_std': model_std,
            'rf_tree_std': rf_std
        }
    
    def evaluate(self, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        """
        Evaluate model performance.
        
        Args:
            X: Feature matrix
            y: True values
        
        Returns:
            Dict with R2, MAE, RMSE, MAPE
        """
        y_pred = self.predict(X)
        
        metrics = {
            'r2': r2_score(y, y_pred),
            'mae': mean_absolute_error(y, y_pred),
            'rmse': np.sqrt(mean_squared_error(y, y_pred)),
            'mape': np.mean(np.abs((y - y_pred) / y)) * 100,
            'bias': float((y_pred - y).mean())
        }
        
        self.metrics = metrics
        return metrics
    
    def _compute_feature_importance(self, feature_names: Optional[List[str]] = None):
        """Compute averaged feature importance across models."""
        n_features = self.rf.n_features_in_
        
        if feature_names is None:
            feature_names = [f'feature_{i}' for i in range(n_features)]
        
        # RF importance
        importance = self.rf.feature_importances_.copy()
        count = 1
        
        # XGBoost importance
        if self.xgb is not None:
            importance += self.xgb.feature_importances_
            count += 1
        
        # LightGBM importance
        if self.lgb is not None:
            importance += self.lgb.feature_importances_
            count += 1
        
        # Average
        importance = importance / count
        
        # Create DataFrame
        self.feature_importance = pd.DataFrame({
            'feature': feature_names,
            'importance': importance
        }).sort_values('importance', ascending=False)
    
    def save(self, filepath: str):
        """Save model to file."""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        model_data = {
            'rf': self.rf,
            'xgb': self.xgb,
            'lgb': self.lgb,
            'weights': self.weights,
            'metrics': self.metrics,
            'feature_importance': self.feature_importance,
            'is_fitted': self.is_fitted
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        
        print(f"Model saved to: {filepath}")
    
    @classmethod
    def load(cls, filepath: str) -> 'EnsembleYieldPredictor':
        """Load model from file."""
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        instance = cls()
        instance.rf = model_data['rf']
        instance.xgb = model_data['xgb']
        instance.lgb = model_data['lgb']
        instance.weights = model_data['weights']
        instance.metrics = model_data['metrics']
        instance.feature_importance = model_data['feature_importance']
        instance.is_fitted = model_data['is_fitted']
        
        return instance


# ===============================================
# TRAINING PIPELINE
# ===============================================

def train_ensemble_model(X: np.ndarray, y: np.ndarray,
                          feature_names: List[str],
                          test_size: float = 0.2,
                          save_path: str = 'models/trained/ensemble_v2.pkl') -> EnsembleYieldPredictor:
    """
    Complete training pipeline for ensemble model.
    
    Args:
        X: Feature matrix
        y: Target vector
        feature_names: List of feature names
        test_size: Train/test split ratio
        save_path: Path to save model
    
    Returns:
        Trained EnsembleYieldPredictor
    """
    print("=" * 70)
    print("ENSEMBLE MODEL TRAINING PIPELINE")
    print("=" * 70)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42
    )
    
    print(f"\nData Split:")
    print(f"  Training: {len(X_train)} samples")
    print(f"  Testing: {len(X_test)} samples")
    
    # Create and train model
    model = EnsembleYieldPredictor()
    model.fit(X_train, y_train, feature_names=feature_names)
    
    # Evaluate
    print("\n" + "=" * 70)
    print("MODEL EVALUATION")
    print("=" * 70)
    
    # Training metrics
    train_metrics = model.evaluate(X_train, y_train)
    print("\nTraining Metrics:")
    for key, value in train_metrics.items():
        print(f"  {key:10s}: {value:.4f}")
    
    # Test metrics
    test_metrics = model.evaluate(X_test, y_test)
    print("\nTest Metrics:")
    for key, value in test_metrics.items():
        print(f"  {key:10s}: {value:.4f}")
    
    # Feature importance
    print("\nTop 10 Feature Importance:")
    print(model.feature_importance.head(10).to_string(index=False))
    
    # Save model
    model.save(save_path)
    
    return model


# ===============================================
# EXAMPLE USAGE
# ===============================================

if __name__ == "__main__":
    # Create synthetic data for testing
    np.random.seed(42)
    n_samples = 1000
    n_features = 25
    
    X = np.random.randn(n_samples, n_features)
    y = 1500 + 200 * X[:, 0] - 100 * X[:, 1] + 50 * np.random.randn(n_samples)
    
    feature_names = [f'feature_{i}' for i in range(n_features)]
    
    # Train model
    model = train_ensemble_model(X, y, feature_names)
    
    # Test prediction with uncertainty
    print("\n" + "=" * 70)
    print("PREDICTION WITH UNCERTAINTY")
    print("=" * 70)
    
    X_new = np.random.randn(5, n_features)
    result = model.predict_with_uncertainty(X_new)
    
    print("\nSample Predictions:")
    for i in range(5):
        print(f"  Sample {i+1}: {result['prediction'][i]:.0f} kg/ha "
              f"[{result['confidence_low'][i]:.0f}, {result['confidence_high'][i]:.0f}] "
              f"(uncertainty: {result['uncertainty'][i]:.0f})")
