"""Explainability module for FactoryGuard AI"""

from src.explainability.shap_explainer import SHAPExplainer
from src.explainability.plots import (
    plot_shap_summary,
    plot_feature_importance_bar,
    plot_force_plot,
    plot_waterfall,
    plot_dependence
)
from src.explainability.insights import InsightGenerator

__all__ = [
    'SHAPExplainer',
    'plot_shap_summary',
    'plot_feature_importance_bar',
    'plot_force_plot',
    'plot_waterfall',
    'plot_dependence',
    'InsightGenerator'
]

