"""Utility for loading trained models and test data"""

import pandas as pd
import joblib
from pathlib import Path
from typing import Tuple, Optional, Dict, Any
from src.models.xgboost_model import XGBoostModel
from src.models.random_forest import RandomForestModel
from src.models.baseline import BaselineModel
from src.config.settings import Settings
from src.utils.logger import get_logger

logger = get_logger(__name__)


def find_latest_model_dir(artifacts_dir: Optional[Path] = None) -> Optional[Path]:
    """
    Find the latest model directory in artifacts folder.
    
    Args:
        artifacts_dir: Base artifacts directory (defaults to Settings.ARTIFACTS_DIR)
        
    Returns:
        Path to latest model directory or None if not found
    """
    if artifacts_dir is None:
        artifacts_dir = Settings.ARTIFACTS_DIR
    
    # Look for directories matching models_YYYYMMDD_HHMMSS pattern
    model_dirs = sorted(
        [d for d in artifacts_dir.iterdir() 
         if d.is_dir() and d.name.startswith('models_')],
        reverse=True
    )
    
    if not model_dirs:
        logger.warning(f"No model directories found in {artifacts_dir}")
        return None
    
    latest_dir = model_dirs[0]
    logger.info(f"Found latest model directory: {latest_dir}")
    return latest_dir


def load_model(
    model_type: str = "xgboost",
    model_dir: Optional[Path] = None
) -> Tuple[Any, Path]:
    """
    Load a trained model from disk.
    
    Args:
        model_type: Type of model to load ('xgboost', 'random_forest', 'baseline')
        model_dir: Directory containing model files (if None, uses latest)
        
    Returns:
        Tuple of (loaded model, model directory path)
    """
    if model_dir is None:
        model_dir = find_latest_model_dir()
        if model_dir is None:
            raise FileNotFoundError("No model directory found. Please train models first.")
    
    model_type = model_type.lower()
    
    if model_type == "xgboost":
        model_path = model_dir / "xgboost_model.joblib"
        if not model_path.exists():
            raise FileNotFoundError(f"XGBoost model not found at {model_path}")
        model = XGBoostModel.load(model_path)
        logger.info(f"Loaded XGBoost model from {model_path}")
        
    elif model_type == "random_forest":
        model_path = model_dir / "random_forest_model.joblib"
        if not model_path.exists():
            raise FileNotFoundError(f"Random Forest model not found at {model_path}")
        model = RandomForestModel.load(model_path)
        logger.info(f"Loaded Random Forest model from {model_path}")
        
    elif model_type == "baseline":
        model_path = model_dir / "baseline_model.joblib"
        if not model_path.exists():
            raise FileNotFoundError(f"Baseline model not found at {model_path}")
        model = BaselineModel.load(model_path)
        logger.info(f"Loaded Baseline model from {model_path}")
        
    else:
        raise ValueError(f"Unknown model type: {model_type}. Use 'xgboost', 'random_forest', or 'baseline'")
    
    return model, model_dir


def load_test_data(
    artifacts_dir: Optional[Path] = None,
    test_file: Optional[Path] = None
) -> pd.DataFrame:
    """
    Load test dataset for explainability.
    
    Args:
        artifacts_dir: Base artifacts directory
        test_file: Specific test file path (if None, finds latest)
        
    Returns:
        Test DataFrame
    """
    if artifacts_dir is None:
        artifacts_dir = Settings.ARTIFACTS_DIR
    
    if test_file is None:
        # Look for test parquet files
        test_files = sorted(
            [f for f in artifacts_dir.glob("test_*.parquet")],
            reverse=True
        )
        
        if not test_files:
            raise FileNotFoundError(f"No test data files found in {artifacts_dir}")
        
        test_file = test_files[0]
        logger.info(f"Using latest test file: {test_file}")
    
    if not test_file.exists():
        raise FileNotFoundError(f"Test file not found: {test_file}")
    
    df = pd.read_parquet(test_file)
    logger.info(f"Loaded test data: {len(df)} rows, {len(df.columns)} columns")
    
    return df


def prepare_explainability_data(
    test_df: pd.DataFrame,
    model: Any,
    target_col: str = Settings.TARGET_COL
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Prepare data for explainability by extracting features and target.
    
    Args:
        test_df: Test DataFrame with all columns
        model: Trained model with feature_names_ attribute
        target_col: Name of target column
        
    Returns:
        Tuple of (X_test, y_test)
    """
    # Get feature columns (exclude metadata and target)
    exclude_cols = [
        Settings.TIMESTAMP_COL,
        Settings.MACHINE_ID_COL,
        target_col,
        'Machine failure'  # Original failure column if present
    ]
    
    feature_cols = [col for col in test_df.columns if col not in exclude_cols]
    
    # Ensure feature order matches model
    if hasattr(model, 'feature_names_') and model.feature_names_:
        # Sanitize feature names to match model expectations
        model_features = set(model.feature_names_)
        available_features = set(feature_cols)
        
        # Find matching features
        matching_features = []
        for feat in model.feature_names_:
            # Try exact match first
            if feat in available_features:
                matching_features.append(feat)
            else:
                # Try to find sanitized version
                for avail_feat in available_features:
                    # Check if sanitized version matches
                    sanitized = avail_feat.replace('[', '_').replace(']', '_').replace('<', '_').replace('>', '_')
                    if sanitized == feat:
                        matching_features.append(avail_feat)
                        break
        
        if len(matching_features) != len(model.feature_names_):
            logger.warning(
                f"Feature mismatch: Model expects {len(model.feature_names_)} features, "
                f"found {len(matching_features)} matching features"
            )
        
        feature_cols = matching_features
    
    X_test = test_df[feature_cols].copy()
    y_test = test_df[target_col] if target_col in test_df.columns else None
    
    # Fill any missing values
    X_test = X_test.fillna(0)
    
    logger.info(f"Prepared explainability data: {len(X_test)} samples, {len(X_test.columns)} features")
    
    if y_test is not None:
        logger.info(f"Target distribution: {y_test.value_counts().to_dict()}")
    
    return X_test, y_test

