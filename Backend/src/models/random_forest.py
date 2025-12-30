"""Random Forest model with class imbalance handling"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
from typing import Dict, Any, Optional
import joblib
from pathlib import Path
from src.utils.logger import get_logger

logger = get_logger(__name__)


class RandomForestModel:
    """
    Random Forest model with class imbalance handling.
    Supports both class_weight and SMOTE for handling rare events.
    """
    
    def __init__(
        self,
        class_weight: Optional[Dict[int, float]] = None,
        random_state: int = 42,
        n_estimators: int = 100,
        max_depth: Optional[int] = None,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1
    ):
        """
        Initialize Random Forest model.
        
        Args:
            class_weight: Dictionary mapping class labels to weights.
                         If None, uses 'balanced' for automatic weighting.
            random_state: Random seed for reproducibility
            n_estimators: Number of trees in the forest
            max_depth: Maximum depth of trees
            min_samples_split: Minimum samples required to split a node
            min_samples_leaf: Minimum samples required at a leaf node
        """
        self.random_state = random_state
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        
        # Use balanced class weights if not specified
        if class_weight is None:
            class_weight = 'balanced'
        
        self.model = RandomForestClassifier(
            class_weight=class_weight,
            random_state=random_state,
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            n_jobs=-1  # Use all available cores
        )
        
        self.feature_names_ = None
        self.is_fitted = False
        self.best_params_ = None
    
    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        optimize: bool = False,
        cv: int = 5,
        n_iter: int = 20,
        scoring: str = 'f1'
    ) -> 'RandomForestModel':
        """
        Train the Random Forest model.
        
        Args:
            X: Feature matrix
            y: Target vector
            optimize: Whether to perform hyperparameter optimization
            cv: Number of cross-validation folds
            n_iter: Number of parameter settings sampled for optimization
            scoring: Scoring metric for optimization
            
        Returns:
            Self for method chaining
        """
        logger.info("Training Random Forest model")
        logger.info(f"Training samples: {len(X)}, Features: {X.shape[1]}")
        logger.info(f"Class distribution: {y.value_counts().to_dict()}")
        
        self.feature_names_ = list(X.columns)
        
        if optimize:
            logger.info("Performing hyperparameter optimization...")
            
            # Parameter grid for RandomizedSearchCV
            param_distributions = {
                'n_estimators': [50, 100, 200, 300],
                'max_depth': [5, 10, 15, 20, None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4],
                'class_weight': ['balanced', {0: 1, 1: 10}, {0: 1, 1: 50}, {0: 1, 1: 100}]
            }
            
            # Create base model for search
            base_model = RandomForestClassifier(
                random_state=self.random_state,
                n_jobs=-1
            )
            
            # Randomized search
            search = RandomizedSearchCV(
                base_model,
                param_distributions,
                n_iter=n_iter,
                cv=cv,
                scoring=scoring,
                random_state=self.random_state,
                n_jobs=-1,
                verbose=1
            )
            
            search.fit(X, y)
            
            # Use best model
            self.model = search.best_estimator_
            self.best_params_ = search.best_params_
            
            logger.info(f"Best parameters: {self.best_params_}")
            logger.info(f"Best CV score ({scoring}): {search.best_score_:.4f}")
        else:
            # Fit with default parameters
            self.model.fit(X, y)
        
        self.is_fitted = True
        logger.info("Random Forest model training completed")
        
        return self
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predict class labels."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        return self.model.predict(X)
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Predict class probabilities."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        return self.model.predict_proba(X)
    
    def get_params(self) -> Dict[str, Any]:
        """Get model parameters."""
        params = {
            'random_state': self.random_state,
            'n_estimators': self.n_estimators,
            'max_depth': self.max_depth,
            'min_samples_split': self.min_samples_split,
            'min_samples_leaf': self.min_samples_leaf
        }
        if self.best_params_:
            params['best_params'] = self.best_params_
        return params
    
    def save(self, filepath: Path) -> None:
        """Save model to disk."""
        filepath.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({
            'model': self.model,
            'feature_names': self.feature_names_,
            'params': self.get_params()
        }, filepath)
        logger.info(f"Random Forest model saved to: {filepath}")
    
    @classmethod
    def load(cls, filepath: Path) -> 'RandomForestModel':
        """Load model from disk."""
        data = joblib.load(filepath)
        params = data['params']
        instance = cls(
            random_state=params['random_state'],
            n_estimators=params['n_estimators'],
            max_depth=params['max_depth'],
            min_samples_split=params['min_samples_split'],
            min_samples_leaf=params['min_samples_leaf']
        )
        instance.model = data['model']
        instance.feature_names_ = data['feature_names']
        instance.best_params_ = params.get('best_params')
        instance.is_fitted = True
        logger.info(f"Random Forest model loaded from: {filepath}")
        return instance

