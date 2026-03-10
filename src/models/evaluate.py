"""Model evaluation metrics for rare-event prediction"""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    recall_score,
    precision_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_auc_score,
    roc_curve
)
from typing import Dict, Any, Tuple, Optional
import json
from pathlib import Path
from src.utils.logger import get_logger

logger = get_logger(__name__)


def calculate_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_pred_proba: Optional[np.ndarray] = None
) -> Dict[str, float]:
    """
    Calculate comprehensive evaluation metrics for rare-event prediction.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        y_pred_proba: Predicted probabilities (optional, for ROC-AUC)
        
    Returns:
        Dictionary of metrics
    """
    metrics = {
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'f1': f1_score(y_true, y_pred, zero_division=0)
    }
    
    # Add ROC-AUC if probabilities are available
    if y_pred_proba is not None:
        try:
            metrics['roc_auc'] = roc_auc_score(y_true, y_pred_proba[:, 1])
        except ValueError:
            logger.warning("Could not calculate ROC-AUC (possibly only one class present)")
            metrics['roc_auc'] = 0.0
    
    return metrics


def get_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray
) -> np.ndarray:
    """
    Get confusion matrix.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        
    Returns:
        Confusion matrix (2x2 array)
    """
    return confusion_matrix(y_true, y_pred)


def evaluate_model(
    model: Any,
    X: pd.DataFrame,
    y: pd.Series,
    model_name: str = "Model"
) -> Dict[str, Any]:
    """
    Comprehensive model evaluation.
    
    Args:
        model: Trained model with predict() and predict_proba() methods
        X: Feature matrix
        y: True labels
        model_name: Name of the model for logging
        
    Returns:
        Dictionary containing all evaluation results
    """
    logger.info(f"\n{'='*80}")
    logger.info(f"Evaluating {model_name}")
    logger.info(f"{'='*80}")
    
    # Predictions
    y_pred = model.predict(X)
    y_pred_proba = model.predict_proba(X)
    
    # Calculate metrics
    metrics = calculate_metrics(y, y_pred, y_pred_proba)
    
    # Confusion matrix
    cm = get_confusion_matrix(y, y_pred)
    
    # Classification report
    report = classification_report(y, y_pred, output_dict=True)
    
    # Log results
    logger.info(f"\nMetrics:")
    logger.info(f"  Recall:    {metrics['recall']:.4f}")
    logger.info(f"  Precision: {metrics['precision']:.4f}")
    logger.info(f"  F1-Score:  {metrics['f1']:.4f}")
    if 'roc_auc' in metrics:
        logger.info(f"  ROC-AUC:   {metrics['roc_auc']:.4f}")
    
    logger.info(f"\nConfusion Matrix:")
    logger.info(f"  True Negatives:  {cm[0, 0]}")
    logger.info(f"  False Positives: {cm[0, 1]}")
    logger.info(f"  False Negatives: {cm[1, 0]}")
    logger.info(f"  True Positives:  {cm[1, 1]}")
    
    logger.info(f"\nClassification Report:")
    logger.info(f"  Class 0 - Precision: {report['0']['precision']:.4f}, "
                f"Recall: {report['0']['recall']:.4f}, "
                f"F1: {report['0']['f1-score']:.4f}")
    logger.info(f"  Class 1 - Precision: {report['1']['precision']:.4f}, "
                f"Recall: {report['1']['recall']:.4f}, "
                f"F1: {report['1']['f1-score']:.4f}")
    
    # Why Recall is prioritized for rare events
    logger.info(f"\nWhy Recall is Prioritized:")
    logger.info(f"  - Rare events (<1% failure rate) require detecting ALL failures")
    logger.info(f"  - False Negatives (missed failures) are costly in predictive maintenance")
    logger.info(f"  - Recall measures: True Positives / (True Positives + False Negatives)")
    logger.info(f"  - High Recall = Fewer missed failures = Better safety and cost savings")
    
    return {
        'metrics': metrics,
        'confusion_matrix': cm.tolist(),
        'classification_report': report,
        'predictions': y_pred.tolist(),
        'probabilities': y_pred_proba[:, 1].tolist() if y_pred_proba is not None else None
    }


def compare_models(
    results: Dict[str, Dict[str, Any]],
    primary_metric: str = 'recall'
) -> pd.DataFrame:
    """
    Compare multiple models on key metrics.
    
    Args:
        results: Dictionary mapping model names to evaluation results
        primary_metric: Primary metric for comparison (default: 'recall')
        
    Returns:
        DataFrame comparing all models
    """
    comparison_data = []
    
    for model_name, result in results.items():
        metrics = result['metrics']
        comparison_data.append({
            'Model': model_name,
            'Recall': metrics['recall'],
            'Precision': metrics['precision'],
            'F1-Score': metrics['f1'],
            'ROC-AUC': metrics.get('roc_auc', 0.0)
        })
    
    comparison_df = pd.DataFrame(comparison_data)
    
    # Map lowercase metric names to DataFrame column names
    metric_map = {
        'recall': 'Recall',
        'precision': 'Precision',
        'f1': 'F1-Score',
        'f1-score': 'F1-Score',
        'roc_auc': 'ROC-AUC',
        'roc-auc': 'ROC-AUC'
    }
    
    sort_column = metric_map.get(primary_metric.lower(), primary_metric.capitalize())
    if sort_column not in comparison_df.columns:
        # Fallback to 'Recall' if mapping fails
        sort_column = 'Recall'
        logger.warning(f"Could not map '{primary_metric}' to DataFrame column. Using 'Recall' instead.")
    
    comparison_df = comparison_df.sort_values(sort_column, ascending=False)
    
    logger.info(f"\n{'='*80}")
    logger.info("Model Comparison")
    logger.info(f"{'='*80}")
    logger.info(f"\n{comparison_df.to_string(index=False)}")
    
    return comparison_df


def save_evaluation_results(
    results: Dict[str, Any],
    filepath: Path,
    model_name: str = "model"
) -> None:
    """
    Save evaluation results to JSON file.
    
    Args:
        results: Evaluation results dictionary
        filepath: Path to save results
        model_name: Name of the model
    """
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    # Prepare data for JSON serialization
    save_data = {
        'model_name': model_name,
        'metrics': results['metrics'],
        'confusion_matrix': results['confusion_matrix'],
        'classification_report': results['classification_report']
    }
    
    with open(filepath, 'w') as f:
        json.dump(save_data, f, indent=2)
    
    logger.info(f"Evaluation results saved to: {filepath}")


def load_evaluation_results(filepath: Path) -> Dict[str, Any]:
    """
    Load evaluation results from JSON file.
    
    Args:
        filepath: Path to saved results
        
    Returns:
        Evaluation results dictionary
    """
    with open(filepath, 'r') as f:
        return json.load(f)

