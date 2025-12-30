"""Training orchestration for all models"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
from datetime import datetime
import json

from src.models.baseline import BaselineModel
from src.models.random_forest import RandomForestModel
from src.models.xgboost_model import XGBoostModel
from src.models.evaluate import (
    evaluate_model,
    compare_models,
    save_evaluation_results
)
from src.config.settings import Settings
from src.utils.logger import get_logger

logger = get_logger(__name__)


def prepare_features(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    target_col: str = Settings.TARGET_COL,
    exclude_cols: Optional[list] = None
) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """
    Prepare features and targets from train/test DataFrames.
    
    Args:
        train_df: Training DataFrame
        test_df: Test DataFrame
        target_col: Name of target column
        exclude_cols: Additional columns to exclude from features
        
    Returns:
        Tuple of (X_train, y_train, X_test, y_test)
    """
    logger.info("Preparing features and targets")
    
    if exclude_cols is None:
        exclude_cols = []
    
    # Standard columns to exclude
    standard_exclude = [
        Settings.TIMESTAMP_COL,
        Settings.MACHINE_ID_COL,
        target_col,
        'failure',  # Original failure column (if exists)
        'Machine failure',  # Alternative failure column name
        'Product ID',  # Common ID column
        'UDI', 'udi',  # Unique device identifier
        'id', 'ID',  # Generic ID columns
        'Type',  # Machine type (categorical)
        'TWF', 'HDF', 'PWF', 'OSF', 'RNF'  # Failure type indicators
    ]
    
    all_exclude = set(standard_exclude + exclude_cols)
    
    # Get feature columns - only numeric columns that are not excluded
    feature_cols = [
        col for col in train_df.columns 
        if col not in all_exclude 
        and pd.api.types.is_numeric_dtype(train_df[col])
    ]
    
    # Log any non-numeric columns that were excluded
    non_numeric_cols = [
        col for col in train_df.columns 
        if col not in all_exclude 
        and not pd.api.types.is_numeric_dtype(train_df[col])
    ]
    
    if non_numeric_cols:
        logger.info(f"Excluding {len(non_numeric_cols)} non-numeric columns: {non_numeric_cols[:10]}")
    
    logger.info(f"Using {len(feature_cols)} numeric features")
    logger.info(f"Excluded columns: {sorted(all_exclude)}")
    
    # Extract features and targets
    X_train = train_df[feature_cols].copy()
    y_train = train_df[target_col].copy()
    
    X_test = test_df[feature_cols].copy()
    y_test = test_df[target_col].copy()
    
    # Check for missing values
    if X_train.isnull().any().any():
        logger.warning("Training features contain missing values - filling with 0")
        X_train = X_train.fillna(0)
    
    if X_test.isnull().any().any():
        logger.warning("Test features contain missing values - filling with 0")
        X_test = X_test.fillna(0)
    
    # Log class distribution
    train_dist = y_train.value_counts().to_dict()
    test_dist = y_test.value_counts().to_dict()
    logger.info(f"Training class distribution: {train_dist}")
    logger.info(f"Test class distribution: {test_dist}")
    
    # Validate that we have positive cases
    train_positive = train_dist.get(1, 0)
    test_positive = test_dist.get(1, 0)
    
    if train_positive == 0:
        error_msg = (
            f"\n{'='*80}\n"
            f"ERROR: No positive cases (failures) found in training data!\n"
            f"{'='*80}\n"
            f"Training class distribution: {train_dist}\n"
            f"Test class distribution: {test_dist}\n\n"
            f"This means the target column '{target_col}' contains only zeros.\n"
            f"Possible causes:\n"
            f"  1. The failure column name is incorrect\n"
            f"  2. The failure column doesn't contain any failure events\n"
            f"  3. The failure column format is not recognized (should be 0/1 or True/False)\n\n"
            f"To fix:\n"
            f"  1. Check your CSV file for the actual failure column name\n"
            f"  2. Verify the failure column contains failure events (non-zero values)\n"
            f"  3. Update the 'failure_col' parameter in run_week1_week2.py\n"
            f"{'='*80}\n"
        )
        logger.error(error_msg)
        raise ValueError("Cannot train models: No positive cases in training data. "
                        "Please check your failure column and data.")
    
    if train_positive < 10:
        logger.warning(f"Very few positive cases in training data: {train_positive}. "
                      f"Model performance may be limited.")
    
    return X_train, y_train, X_test, y_test


def train_baseline(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    artifacts_dir: Path
) -> Tuple[BaselineModel, Dict[str, Any]]:
    """
    Train baseline Logistic Regression model.
    
    Args:
        X_train: Training features
        y_train: Training targets
        X_test: Test features
        y_test: Test targets
        artifacts_dir: Directory to save artifacts
        
    Returns:
        Tuple of (trained model, evaluation results)
    """
    logger.info("\n" + "="*80)
    logger.info("Training Baseline Model (Logistic Regression)")
    logger.info("="*80)
    
    # Train model
    model = BaselineModel(random_state=42)
    model.fit(X_train, y_train)
    
    # Evaluate
    results = evaluate_model(model, X_test, y_test, "Baseline (Logistic Regression)")
    
    # Save model and results
    model_path = artifacts_dir / "baseline_model.joblib"
    model.save(model_path)
    
    results_path = artifacts_dir / "baseline_results.json"
    save_evaluation_results(results, results_path, "baseline")
    
    return model, results


def train_random_forest(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    artifacts_dir: Path,
    optimize: bool = True
) -> Tuple[RandomForestModel, Dict[str, Any]]:
    """
    Train Random Forest model.
    
    Args:
        X_train: Training features
        y_train: Training targets
        X_test: Test features
        y_test: Test targets
        artifacts_dir: Directory to save artifacts
        optimize: Whether to perform hyperparameter optimization
        
    Returns:
        Tuple of (trained model, evaluation results)
    """
    logger.info("\n" + "="*80)
    logger.info("Training Random Forest Model")
    logger.info("="*80)
    
    # Train model
    model = RandomForestModel(random_state=42)
    model.fit(X_train, y_train, optimize=optimize, scoring='f1')
    
    # Evaluate
    results = evaluate_model(model, X_test, y_test, "Random Forest")
    
    # Save model and results
    model_path = artifacts_dir / "random_forest_model.joblib"
    model.save(model_path)
    
    results_path = artifacts_dir / "random_forest_results.json"
    save_evaluation_results(results, results_path, "random_forest")
    
    return model, results


def train_xgboost(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    artifacts_dir: Path,
    optimize: bool = True
) -> Tuple[XGBoostModel, Dict[str, Any]]:
    """
    Train XGBoost model (primary model).
    
    Args:
        X_train: Training features
        y_train: Training targets
        X_test: Test features
        y_test: Test targets
        artifacts_dir: Directory to save artifacts
        optimize: Whether to perform hyperparameter optimization
        
    Returns:
        Tuple of (trained model, evaluation results)
    """
    logger.info("\n" + "="*80)
    logger.info("Training XGBoost Model (Primary Model)")
    logger.info("="*80)
    
    # Train model
    model = XGBoostModel(random_state=42)
    model.fit(X_train, y_train, optimize=optimize, scoring='f1')
    
    # Evaluate
    results = evaluate_model(model, X_test, y_test, "XGBoost")
    
    # Save model and results
    model_path = artifacts_dir / "xgboost_model.joblib"
    model.save(model_path)
    
    results_path = artifacts_dir / "xgboost_results.json"
    save_evaluation_results(results, results_path, "xgboost")
    
    # Save feature importance
    feature_importance = model.get_feature_importance()
    importance_path = artifacts_dir / "xgboost_feature_importance.csv"
    feature_importance.to_csv(importance_path, index=False)
    logger.info(f"Feature importance saved to: {importance_path}")
    
    return model, results


def train_all_models(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    artifacts_dir: Optional[Path] = None,
    optimize: bool = True,
    target_col: str = Settings.TARGET_COL
) -> Dict[str, Any]:
    """
    Train all models and compare performance.
    
    Args:
        train_df: Training DataFrame
        test_df: Test DataFrame
        artifacts_dir: Directory to save artifacts (default: Settings.ARTIFACTS_DIR)
        optimize: Whether to perform hyperparameter optimization
        target_col: Name of target column
        
    Returns:
        Dictionary containing all models and results
    """
    if artifacts_dir is None:
        artifacts_dir = Settings.ARTIFACTS_DIR
    
    # Create versioned artifacts directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    versioned_dir = artifacts_dir / f"models_{timestamp}"
    versioned_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"\n{'='*80}")
    logger.info("FactoryGuard AI - Week 2: Model Training")
    logger.info(f"{'='*80}")
    logger.info(f"Artifacts directory: {versioned_dir}")
    
    # Prepare features
    X_train, y_train, X_test, y_test = prepare_features(
        train_df, test_df, target_col=target_col
    )
    
    # Train all models
    baseline_model, baseline_results = train_baseline(
        X_train, y_train, X_test, y_test, versioned_dir
    )
    
    rf_model, rf_results = train_random_forest(
        X_train, y_train, X_test, y_test, versioned_dir, optimize=optimize
    )
    
    xgb_model, xgb_results = train_xgboost(
        X_train, y_train, X_test, y_test, versioned_dir, optimize=optimize
    )
    
    # Compare models
    all_results = {
        'Baseline': baseline_results,
        'Random Forest': rf_results,
        'XGBoost': xgb_results
    }
    
    comparison_df = compare_models(all_results, primary_metric='recall')
    
    # Save comparison
    comparison_path = versioned_dir / "model_comparison.csv"
    comparison_df.to_csv(comparison_path, index=False)
    logger.info(f"Model comparison saved to: {comparison_path}")
    
    # Save training config
    config = {
        'timestamp': timestamp,
        'optimize': optimize,
        'target_col': target_col,
        'n_features': len(X_train.columns),
        'n_train_samples': len(X_train),
        'n_test_samples': len(X_test),
        'train_class_distribution': y_train.value_counts().to_dict(),
        'test_class_distribution': y_test.value_counts().to_dict()
    }
    
    config_path = versioned_dir / "training_config.json"
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2, default=str)
    
    logger.info(f"\n{'='*80}")
    logger.info("Training Completed Successfully")
    logger.info(f"{'='*80}")
    logger.info(f"All artifacts saved to: {versioned_dir}")
    
    return {
        'models': {
            'baseline': baseline_model,
            'random_forest': rf_model,
            'xgboost': xgb_model
        },
        'results': all_results,
        'comparison': comparison_df,
        'artifacts_dir': versioned_dir
    }

