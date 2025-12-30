"""Baseline Logistic Regression model with class weighting"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from typing import Dict, Any, Optional
import joblib
from pathlib import Path
from src.utils.logger import get_logger

logger = get_logger(__name__)


class BaselineModel:
    """
    Baseline Logistic Regression model with proper scaling and class weighting.
    Designed for rare-event prediction with <1% failure rate.
    """
    
    def __init__(
        self,
        class_weight: Optional[Dict[int, float]] = None,
        random_state: int = 42,
        max_iter: int = 1000
    ):
        """
        Initialize baseline model.
        
        Args:
            class_weight: Dictionary mapping class labels to weights.
                         If None, uses 'balanced' for automatic weighting.
            random_state: Random seed for reproducibility
            max_iter: Maximum iterations for convergence
        """
        self.random_state = random_state
        self.max_iter = max_iter
        
        # Use balanced class weights if not specified
        if class_weight is None:
            class_weight = 'balanced'
        
        # Create pipeline: scaling + logistic regression
        self.model = Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', LogisticRegression(
                class_weight=class_weight,
                random_state=random_state,
                max_iter=max_iter,
                solver='lbfgs'  # Good for small-medium datasets
            ))
        ])
        
        self.feature_names_ = None
        self.is_fitted = False
    
    def fit(self, X: pd.DataFrame, y: pd.Series) -> 'BaselineModel':
        """
        Train the baseline model.
        
        Args:
            X: Feature matrix (DataFrame)
            y: Target vector (Series)
            
        Returns:
            Self for method chaining
        """
        logger.info("Training baseline Logistic Regression model")
        logger.info(f"Training samples: {len(X)}, Features: {X.shape[1]}")
        logger.info(f"Class distribution: {y.value_counts().to_dict()}")
        
        # Store feature names for consistency
        self.feature_names_ = list(X.columns)
        
        # Fit model
        self.model.fit(X, y)
        self.is_fitted = True
        
        logger.info("Baseline model training completed")
        
        return self
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict class labels.
        
        Args:
            X: Feature matrix
            
        Returns:
            Predicted class labels
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        return self.model.predict(X)
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict class probabilities.
        
        Args:
            X: Feature matrix
            
        Returns:
            Predicted probabilities (shape: [n_samples, 2])
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        return self.model.predict_proba(X)
    
    def get_params(self) -> Dict[str, Any]:
        """Get model parameters."""
        return {
            'random_state': self.random_state,
            'max_iter': self.max_iter,
            'class_weight': self.model.named_steps['classifier'].class_weight
        }
    
    def save(self, filepath: Path) -> None:
        """
        Save model to disk.
        
        Args:
            filepath: Path to save model
        """
        filepath.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({
            'model': self.model,
            'feature_names': self.feature_names_,
            'params': self.get_params()
        }, filepath)
        logger.info(f"Baseline model saved to: {filepath}")
    
    @classmethod
    def load(cls, filepath: Path) -> 'BaselineModel':
        """
        Load model from disk.
        
        Args:
            filepath: Path to saved model
            
        Returns:
            Loaded BaselineModel instance
        """
        data = joblib.load(filepath)
        instance = cls(
            class_weight=data['params']['class_weight'],
            random_state=data['params']['random_state'],
            max_iter=data['params']['max_iter']
        )
        instance.model = data['model']
        instance.feature_names_ = data['feature_names']
        instance.is_fitted = True
        logger.info(f"Baseline model loaded from: {filepath}")
        return instance

