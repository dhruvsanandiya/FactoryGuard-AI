"""Visualization functions for SHAP explanations"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap
from pathlib import Path
from typing import Optional, Tuple, List
from src.utils.logger import get_logger

logger = get_logger(__name__)


def plot_shap_summary(
    shap_values: np.ndarray,
    X: pd.DataFrame,
    feature_names: List[str],
    output_path: Optional[Path] = None,
    max_display: int = 20,
    show: bool = False
) -> None:
    """
    Create SHAP summary plot showing global feature importance.
    
    Args:
        shap_values: SHAP values array
        X: Feature data
        feature_names: List of feature names
        output_path: Path to save plot
        max_display: Maximum number of features to display
        show: Whether to display plot
    """
    logger.info("Creating SHAP summary plot...")
    
    # Handle binary classification - extract positive class
    if len(shap_values.shape) == 3:
        if shap_values.shape[2] == 2:
            shap_values = shap_values[:, :, 1]  # Positive class
        elif shap_values.shape[2] == 1:
            shap_values = shap_values[:, :, 0]  # Single class
    
    # Create SHAP summary plot
    plt.figure(figsize=(10, 8))
    shap.summary_plot(
        shap_values,
        X.values,
        feature_names=feature_names,
        max_display=max_display,
        show=False
    )
    plt.tight_layout()
    
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        logger.info(f"SHAP summary plot saved to {output_path}")
    
    if show:
        plt.show()
    else:
        plt.close()


def plot_feature_importance_bar(
    feature_importance: pd.DataFrame,
    output_path: Optional[Path] = None,
    top_n: int = 20,
    show: bool = False
) -> None:
    """
    Create bar plot of feature importance.
    
    Args:
        feature_importance: DataFrame with 'feature' and 'importance' columns
        output_path: Path to save plot
        top_n: Number of top features to display
        show: Whether to display plot
    """
    logger.info("Creating feature importance bar plot...")
    
    # Get top N features
    top_features = feature_importance.head(top_n)
    
    plt.figure(figsize=(10, max(6, top_n * 0.3)))
    plt.barh(range(len(top_features)), top_features['importance'].values)
    plt.yticks(range(len(top_features)), top_features['feature'].values)
    plt.xlabel('SHAP Importance (Mean |SHAP Value|)')
    plt.title(f'Top {top_n} Most Important Features')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        logger.info(f"Feature importance plot saved to {output_path}")
    
    if show:
        plt.show()
    else:
        plt.close()


def plot_force_plot(
    shap_values: np.ndarray,
    X: pd.DataFrame,
    feature_names: List[str],
    instance_idx: int,
    expected_value: float,
    output_path: Optional[Path] = None,
    show: bool = False
) -> None:
    """
    Create SHAP force plot for a single instance.
    
    Args:
        shap_values: SHAP values array
        X: Feature data
        feature_names: List of feature names
        instance_idx: Index of instance to explain
        expected_value: Expected value from SHAP explainer
        output_path: Path to save plot (HTML format)
        show: Whether to display plot
    """
    logger.info(f"Creating force plot for instance {instance_idx}...")
    
    # Handle binary classification - extract positive class
    if len(shap_values.shape) == 3:
        if shap_values.shape[2] == 2:
            shap_values = shap_values[:, :, 1]  # Positive class
        elif shap_values.shape[2] == 1:
            shap_values = shap_values[:, :, 0]  # Single class
    
    # Get instance data
    instance_shap = shap_values[instance_idx]
    instance_features = X.iloc[instance_idx].values
    
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        # Save as HTML for interactive plot
        if output_path.suffix == '.html':
            try:
                force_plot = shap.force_plot(
                    expected_value,
                    instance_shap,
                    instance_features,
                    feature_names=feature_names,
                    show=False,
                    matplotlib=False
                )
                # Save HTML - method varies by SHAP version
                if hasattr(force_plot, 'save_html'):
                    force_plot.save_html(str(output_path))
                else:
                    # For newer SHAP versions
                    with open(output_path, 'w', encoding='utf-8') as f:
                        f.write(force_plot._repr_html_())
                logger.info(f"Force plot (HTML) saved to {output_path}")
            except Exception as e:
                logger.warning(f"Could not save HTML force plot: {e}. Saving as image instead.")
                # Fallback to image
                plt.figure(figsize=(12, 4))
                shap.force_plot(
                    expected_value,
                    instance_shap,
                    instance_features,
                    feature_names=feature_names,
                    matplotlib=True,
                    show=False
                )
                output_path = output_path.with_suffix('.png')
                plt.savefig(output_path, dpi=300, bbox_inches='tight')
                plt.close()
                logger.info(f"Force plot saved as image to {output_path}")
        else:
            # Save as image using matplotlib
            plt.figure(figsize=(12, 4))
            shap.force_plot(
                expected_value,
                instance_shap,
                instance_features,
                feature_names=feature_names,
                matplotlib=True,
                show=False
            )
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()
            logger.info(f"Force plot saved to {output_path}")
    elif show:
        # Display plot
        shap.force_plot(
            expected_value,
            instance_shap,
            instance_features,
            feature_names=feature_names,
            matplotlib=True,
            show=True
        )


def plot_waterfall(
    shap_values: np.ndarray,
    X: pd.DataFrame,
    feature_names: List[str],
    instance_idx: int,
    expected_value: float,
    output_path: Optional[Path] = None,
    show: bool = False
) -> None:
    """
    Create SHAP waterfall plot for a single instance.
    
    Args:
        shap_values: SHAP values array
        X: Feature data
        feature_names: List of feature names
        instance_idx: Index of instance to explain
        expected_value: Expected value from SHAP explainer
        output_path: Path to save plot
        show: Whether to display plot
    """
    logger.info(f"Creating waterfall plot for instance {instance_idx}...")
    
    # Handle binary classification - extract positive class
    if len(shap_values.shape) == 3:
        if shap_values.shape[2] == 2:
            shap_values = shap_values[:, :, 1]  # Positive class
        elif shap_values.shape[2] == 1:
            shap_values = shap_values[:, :, 0]  # Single class
    
    # Get instance data
    instance_shap = shap_values[instance_idx]
    instance_features = X.iloc[instance_idx]
    
    # Create Explanation object for waterfall plot
    explanation = shap.Explanation(
        values=instance_shap,
        base_values=expected_value,
        data=instance_features.values,
        feature_names=feature_names
    )
    
    plt.figure(figsize=(10, 8))
    shap.waterfall_plot(explanation, show=False)
    plt.tight_layout()
    
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        logger.info(f"Waterfall plot saved to {output_path}")
    
    if show:
        plt.show()
    else:
        plt.close()


def plot_dependence(
    shap_values: np.ndarray,
    X: pd.DataFrame,
    feature_names: List[str],
    feature_idx: int,
    output_path: Optional[Path] = None,
    show: bool = False
) -> None:
    """
    Create SHAP dependence plot for a specific feature.
    
    Args:
        shap_values: SHAP values array
        X: Feature data
        feature_names: List of feature names
        feature_idx: Index of feature to plot
        output_path: Path to save plot
        show: Whether to display plot
    """
    feature_name = feature_names[feature_idx]
    logger.info(f"Creating dependence plot for feature: {feature_name}...")
    
    # Handle binary classification - extract positive class
    if len(shap_values.shape) == 3:
        if shap_values.shape[2] == 2:
            shap_values = shap_values[:, :, 1]  # Positive class
        elif shap_values.shape[2] == 1:
            shap_values = shap_values[:, :, 0]  # Single class
    
    plt.figure(figsize=(10, 6))
    shap.dependence_plot(
        feature_idx,
        shap_values,
        X.values,
        feature_names=feature_names,
        show=False
    )
    plt.tight_layout()
    
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        logger.info(f"Dependence plot saved to {output_path}")
    
    if show:
        plt.show()
    else:
        plt.close()

