"""XGBoost model with hyperparameter optimization for rare-event prediction"""

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import RandomizedSearchCV
from typing import Dict, Any, Optional
import joblib
from pathlib import Path
from src.utils.logger import get_logger

logger = get_logger(__name__)


class XGBoostModel:
    """
    XGBoost model optimized for rare-event prediction.
    Primary model with hyperparameter optimization using RandomizedSearchCV.
    Handles class imbalance through scale_pos_weight and class weighting.
    """
    
    def __init__(
        self,
        random_state: int = 42,
        scale_pos_weight: Optional[float] = None,
        max_depth: int = 6,
        learning_rate: float = 0.1,
        n_estimators: int = 100,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8
    ):
        """
        Initialize XGBoost model.
        
        Args:
            random_state: Random seed for reproducibility
            scale_pos_weight: Weight for positive class (handles imbalance).
                            If None, will be calculated from data.
            max_depth: Maximum tree depth
            learning_rate: Learning rate (eta)
            n_estimators: Number of boosting rounds
            subsample: Subsample ratio of training instances
            colsample_bytree: Subsample ratio of columns when constructing each tree
        """
        self.random_state = random_state
        self.scale_pos_weight = scale_pos_weight
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.n_estimators = n_estimators
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        
        self.model = None
        self.feature_names_ = None
        self.is_fitted = False
        self.best_params_ = None
    
    def _create_model(self, scale_pos_weight: Optional[float] = None) -> xgb.XGBClassifier:
        """Create XGBoost classifier with specified parameters."""
        return xgb.XGBClassifier(
            scale_pos_weight=scale_pos_weight or self.scale_pos_weight,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            n_estimators=self.n_estimators,
            subsample=self.subsample,
            colsample_bytree=self.colsample_bytree,
            random_state=self.random_state,
            eval_metric='logloss',
            n_jobs=-1
        )
    
    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        optimize: bool = True,
        cv: int = 5,
        n_iter: int = 50,
        scoring: str = 'f1'
    ) -> 'XGBoostModel':
        """
        Train the XGBoost model with optional hyperparameter optimization.
        
        Args:
            X: Feature matrix
            y: Target vector
            optimize: Whether to perform hyperparameter optimization
            cv: Number of cross-validation folds
            n_iter: Number of parameter settings sampled for optimization
            scoring: Scoring metric for optimization ('f1', 'recall', 'roc_auc')
            
        Returns:
            Self for method chaining
        """
        logger.info("Training XGBoost model")
        logger.info(f"Training samples: {len(X)}, Features: {X.shape[1]}")
        
        class_counts = y.value_counts()
        logger.info(f"Class distribution: {class_counts.to_dict()}")
        
        # Sanitize feature names for XGBoost (remove invalid characters: [, ], <, >)
        X_clean = X.copy()
        invalid_chars = ['[', ']', '<', '>']
        original_names = list(X_clean.columns)
        cleaned_names = []
        
        for name in original_names:
            cleaned = name
            for char in invalid_chars:
                cleaned = cleaned.replace(char, '_')
            cleaned_names.append(cleaned)
        
        # Only rename if there were changes
        if cleaned_names != original_names:
            X_clean.columns = cleaned_names
            logger.info(f"Cleaned {sum(1 for a, b in zip(original_names, cleaned_names) if a != b)} feature names for XGBoost compatibility")
        
        # Calculate scale_pos_weight if not provided
        if self.scale_pos_weight is None:
            # For rare events: weight = negative_samples / positive_samples
            if len(class_counts) == 2:
                self.scale_pos_weight = class_counts[0] / class_counts[1]
                logger.info(f"Auto-calculated scale_pos_weight: {self.scale_pos_weight:.2f}")
            else:
                self.scale_pos_weight = 1.0
        
        self.feature_names_ = list(X_clean.columns)
        
        if optimize:
            logger.info("Performing hyperparameter optimization...")
            
            # Parameter grid for RandomizedSearchCV
            # Focused on parameters that matter most for rare-event prediction
            param_distributions = {
                'max_depth': [3, 4, 5, 6, 7, 8],
                'learning_rate': [0.01, 0.05, 0.1, 0.15, 0.2],
                'n_estimators': [100, 200, 300, 500],
                'subsample': [0.6, 0.7, 0.8, 0.9, 1.0],
                'colsample_bytree': [0.6, 0.7, 0.8, 0.9, 1.0],
                'min_child_weight': [1, 3, 5],
                'gamma': [0, 0.1, 0.2, 0.3],
                'scale_pos_weight': [
                    self.scale_pos_weight * 0.5,
                    self.scale_pos_weight,
                    self.scale_pos_weight * 1.5,
                    self.scale_pos_weight * 2.0
                ]
            }
            
            # Create base model for search
            base_model = self._create_model(scale_pos_weight=self.scale_pos_weight)
            
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
            
            search.fit(X_clean, y)
            
            # Use best model
            self.model = search.best_estimator_
            self.best_params_ = search.best_params_
            
            logger.info(f"Best parameters: {self.best_params_}")
            logger.info(f"Best CV score ({scoring}): {search.best_score_:.4f}")
        else:
            # Fit with default/specified parameters
            self.model = self._create_model(scale_pos_weight=self.scale_pos_weight)
            self.model.fit(X_clean, y)
        
        self.is_fitted = True
        logger.info("XGBoost model training completed")
        
        return self
    
    def _sanitize_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """Sanitize feature names to match training data."""
        X_clean = X.copy()
        if self.feature_names_:
            # Map original names to cleaned names if needed
            invalid_chars = ['[', ']', '<', '>']
            name_mapping = {}
            for orig_name in X.columns:
                cleaned = orig_name
                for char in invalid_chars:
                    cleaned = cleaned.replace(char, '_')
                name_mapping[orig_name] = cleaned
            
            # Rename columns to match training
            if name_mapping and any(k != v for k, v in name_mapping.items()):
                X_clean = X_clean.rename(columns=name_mapping)
        return X_clean
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predict class labels."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        X_clean = self._sanitize_features(X)
        return self.model.predict(X_clean)
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Predict class probabilities."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        X_clean = self._sanitize_features(X)
        return self.model.predict_proba(X_clean)
    
    def get_feature_importance(self) -> pd.DataFrame:
        """
        Get feature importance scores.
        
        Returns:
            DataFrame with feature names and importance scores
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before getting feature importance")
        
        importances = self.model.feature_importances_
        return pd.DataFrame({
            'feature': self.feature_names_,
            'importance': importances
        }).sort_values('importance', ascending=False)
    
    def get_params(self) -> Dict[str, Any]:
        """Get model parameters."""
        params = {
            'random_state': self.random_state,
            'scale_pos_weight': self.scale_pos_weight,
            'max_depth': self.max_depth,
            'learning_rate': self.learning_rate,
            'n_estimators': self.n_estimators,
            'subsample': self.subsample,
            'colsample_bytree': self.colsample_bytree
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
        logger.info(f"XGBoost model saved to: {filepath}")
    
    @classmethod
    def load(cls, filepath: Path) -> 'XGBoostModel':
        """Load model from disk."""
        data = joblib.load(filepath)
        params = data['params']
        instance = cls(
            random_state=params['random_state'],
            scale_pos_weight=params['scale_pos_weight'],
            max_depth=params['max_depth'],
            learning_rate=params['learning_rate'],
            n_estimators=params['n_estimators'],
            subsample=params['subsample'],
            colsample_bytree=params['colsample_bytree']
        )
        instance.model = data['model']
        instance.feature_names_ = data['feature_names']
        instance.best_params_ = params.get('best_params')
        instance.is_fitted = True
        logger.info(f"XGBoost model loaded from: {filepath}")
        return instance

