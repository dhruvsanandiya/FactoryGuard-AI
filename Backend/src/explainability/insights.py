"""Human-readable insights generator from SHAP explanations"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from src.utils.logger import get_logger

logger = get_logger(__name__)


class InsightGenerator:
    """
    Generates human-readable explanations from SHAP values.
    Converts technical SHAP output into actionable insights for engineers.
    """
    
    def __init__(
        self,
        feature_names: List[str],
        feature_thresholds: Optional[Dict[str, Dict[str, float]]] = None
    ):
        """
        Initialize insight generator.
        
        Args:
            feature_names: List of feature names
            feature_thresholds: Optional dictionary mapping feature names to safe ranges
                              Format: {'feature_name': {'min': value, 'max': value}}
        """
        self.feature_names = feature_names
        self.feature_thresholds = feature_thresholds or {}
    
    def _parse_feature_name(self, feature_name: str) -> Dict[str, str]:
        """
        Parse feature name to extract sensor type, operation, and window.
        
        Examples:
            'temperature_lag_1' -> {'sensor': 'temperature', 'op': 'lag', 'window': '1'}
            'pressure_rolling_mean_4h' -> {'sensor': 'pressure', 'op': 'rolling_mean', 'window': '4h'}
        """
        parts = feature_name.split('_')
        
        # Common sensor names
        sensors = ['temperature', 'pressure', 'vibration', 'rpm', 'voltage', 'current']
        sensor = None
        for s in sensors:
            if s in feature_name.lower():
                sensor = s
                break
        
        # Operations
        operations = {
            'lag': 'lag',
            'rolling_mean': 'rolling_mean',
            'rolling_std': 'rolling_std',
            'ema': 'ema'
        }
        
        op = None
        for op_key, op_value in operations.items():
            if op_key in feature_name.lower():
                op = op_value
                break
        
        # Extract window/timeframe
        window = None
        if 'lag' in feature_name.lower():
            # Extract lag number
            for part in parts:
                if part.isdigit():
                    window = f"{part} time step(s) ago"
                    break
        elif 'rolling' in feature_name.lower():
            # Extract time window (1h, 4h, 8h)
            for part in parts:
                if 'h' in part.lower():
                    window = part
                    break
        elif 'ema' in feature_name.lower():
            # Extract alpha
            for part in parts:
                try:
                    alpha = float(part)
                    window = f"EMA (alpha={alpha})"
                    break
                except ValueError:
                    pass
        
        return {
            'sensor': sensor or 'unknown',
            'operation': op or 'raw',
            'window': window or 'current'
        }
    
    def _get_feature_interpretation(
        self,
        feature_name: str,
        feature_value: float,
        shap_value: float
    ) -> str:
        """
        Generate human-readable interpretation for a single feature.
        
        Args:
            feature_name: Name of the feature
            feature_value: Actual feature value
            shap_value: SHAP value (contribution to prediction)
            
        Returns:
            Human-readable explanation string
        """
        parsed = self._parse_feature_name(feature_name)
        sensor = parsed['sensor']
        operation = parsed['operation']
        window = parsed['window']
        
        # Determine direction
        if shap_value > 0:
            direction = "increased"
            risk_impact = "increased failure risk"
        else:
            direction = "decreased"
            risk_impact = "decreased failure risk"
        
        # Check against thresholds if available
        threshold_info = ""
        if feature_name in self.feature_thresholds:
            thresholds = self.feature_thresholds[feature_name]
            if 'min' in thresholds and feature_value < thresholds['min']:
                threshold_info = f" (below safe minimum of {thresholds['min']})"
            elif 'max' in thresholds and feature_value > thresholds['max']:
                threshold_info = f" (exceeding safe maximum of {thresholds['max']})"
        
        # Build explanation
        if operation == 'lag':
            explanation = (
                f"{sensor.capitalize()} from {window} {direction} "
                f"failure risk (value: {feature_value:.2f}{threshold_info})"
            )
        elif operation == 'rolling_mean':
            explanation = (
                f"{sensor.capitalize()} average over {window} {direction} "
                f"failure risk (mean: {feature_value:.2f}{threshold_info})"
            )
        elif operation == 'rolling_std':
            explanation = (
                f"{sensor.capitalize()} variability over {window} {direction} "
                f"failure risk (std: {feature_value:.2f}{threshold_info})"
            )
        elif operation == 'ema':
            explanation = (
                f"{sensor.capitalize()} trend ({window}) {direction} "
                f"failure risk (value: {feature_value:.2f}{threshold_info})"
            )
        else:
            explanation = (
                f"{sensor.capitalize()} {direction} failure risk "
                f"(value: {feature_value:.2f}{threshold_info})"
            )
        
        return explanation
    
    def explain_prediction(
        self,
        shap_values: np.ndarray,
        X: pd.DataFrame,
        instance_idx: int,
        prediction: float,
        top_n: int = 5
    ) -> Dict[str, any]:
        """
        Generate human-readable explanation for a single prediction.
        
        Args:
            shap_values: SHAP values array
            X: Feature data
            instance_idx: Index of instance to explain
            prediction: Model prediction probability
            top_n: Number of top contributing features to include
            
        Returns:
            Dictionary with explanation components
        """
        # Extract positive class SHAP values if needed
        if len(shap_values.shape) == 3:
            if shap_values.shape[2] == 2:
                shap_values = shap_values[:, :, 1]  # Positive class
            elif shap_values.shape[2] == 1:
                shap_values = shap_values[:, :, 0]  # Single class
        
        instance_shap = shap_values[instance_idx]
        instance_features = X.iloc[instance_idx]
        
        # Get top contributing features
        abs_shap = np.abs(instance_shap)
        top_indices = np.argsort(abs_shap)[-top_n:][::-1]
        
        # Separate positive and negative contributions
        positive_factors = []
        negative_factors = []
        
        for idx in top_indices:
            feature_name = self.feature_names[idx]
            feature_value = instance_features.iloc[idx]
            shap_value = instance_shap[idx]
            
            interpretation = self._get_feature_interpretation(
                feature_name,
                feature_value,
                shap_value
            )
            
            if shap_value > 0:
                positive_factors.append({
                    'feature': feature_name,
                    'value': float(feature_value),
                    'shap_value': float(shap_value),
                    'explanation': interpretation
                })
            else:
                negative_factors.append({
                    'feature': feature_name,
                    'value': float(feature_value),
                    'shap_value': float(shap_value),
                    'explanation': interpretation
                })
        
        # Generate summary
        risk_level = "HIGH" if prediction > 0.7 else "MEDIUM" if prediction > 0.3 else "LOW"
        
        summary = f"Failure risk is {risk_level} (probability: {prediction:.1%}).\n\n"
        
        if positive_factors:
            summary += "Failure risk increased due to:\n"
            for i, factor in enumerate(positive_factors, 1):
                summary += f"  {i}. {factor['explanation']}\n"
        
        if negative_factors:
            summary += "\nFailure risk decreased due to:\n"
            for i, factor in enumerate(negative_factors, 1):
                summary += f"  {i}. {factor['explanation']}\n"
        
        result = {
            'instance_idx': instance_idx,
            'prediction': float(prediction),
            'risk_level': risk_level,
            'summary': summary,
            'positive_factors': positive_factors,
            'negative_factors': negative_factors,
            'top_features': [
                {
                    'feature': self.feature_names[idx],
                    'value': float(instance_features.iloc[idx]),
                    'shap_value': float(instance_shap[idx])
                }
                for idx in top_indices
            ]
        }
        
        # Validation assertions
        assert 0 <= prediction <= 1, f"Prediction must be between 0 and 1, got {prediction}"
        assert len(positive_factors) + len(negative_factors) > 0, "At least one contributing factor must be identified"
        assert len(result['top_features']) == top_n, f"Expected {top_n} top features, got {len(result['top_features'])}"
        
        # Validate that top features match sorted order
        top_shap_values = [f['shap_value'] for f in result['top_features']]
        assert top_shap_values == sorted(top_shap_values, key=abs, reverse=True), "Top features must be sorted by absolute SHAP value"
        
        return result
    
    def explain_high_risk_machines(
        self,
        shap_values: np.ndarray,
        X: pd.DataFrame,
        predictions: np.ndarray,
        top_k: int = 5
    ) -> List[Dict[str, any]]:
        """
        Explain predictions for high-risk machines.
        
        Args:
            shap_values: SHAP values array
            X: Feature data
            predictions: Model predictions (probabilities)
            top_k: Number of high-risk machines to explain
            
        Returns:
            List of explanation dictionaries
        """
        # Get top K high-risk instances
        high_risk_indices = np.argsort(predictions)[-top_k:][::-1]
        
        explanations = []
        for idx in high_risk_indices:
            explanation = self.explain_prediction(
                shap_values,
                X,
                idx,
                predictions[idx]
            )
            explanations.append(explanation)
        
        return explanations
    
    def generate_report(
        self,
        shap_values: np.ndarray,
        X: pd.DataFrame,
        predictions: np.ndarray,
        y_true: Optional[pd.Series] = None,
        output_path: Optional[str] = None
    ) -> str:
        """
        Generate comprehensive human-readable report.
        
        Args:
            shap_values: SHAP values array
            X: Feature data
            predictions: Model predictions
            y_true: True labels (optional, for validation)
            output_path: Optional path to save report
            
        Returns:
            Report text
        """
        logger.info("Generating human-readable insights report...")
        
        report_lines = []
        report_lines.append("=" * 80)
        report_lines.append("FactoryGuard AI - Model Interpretability Report")
        report_lines.append("=" * 80)
        report_lines.append("")
        
        # Overall statistics
        high_risk_count = (predictions > 0.7).sum()
        medium_risk_count = ((predictions > 0.3) & (predictions <= 0.7)).sum()
        low_risk_count = (predictions <= 0.3).sum()
        
        report_lines.append("Overall Risk Distribution:")
        report_lines.append(f"  High Risk (>70%):   {high_risk_count} machines ({high_risk_count/len(predictions)*100:.1f}%)")
        report_lines.append(f"  Medium Risk (30-70%): {medium_risk_count} machines ({medium_risk_count/len(predictions)*100:.1f}%)")
        report_lines.append(f"  Low Risk (<30%):    {low_risk_count} machines ({low_risk_count/len(predictions)*100:.1f}%)")
        report_lines.append("")
        
        # Validation if true labels available
        if y_true is not None:
            true_positives = ((predictions > 0.5) & (y_true == 1)).sum()
            false_positives = ((predictions > 0.5) & (y_true == 0)).sum()
            false_negatives = ((predictions <= 0.5) & (y_true == 1)).sum()
            
            report_lines.append("Prediction Validation:")
            report_lines.append(f"  True Positives:  {true_positives}")
            report_lines.append(f"  False Positives: {false_positives}")
            report_lines.append(f"  False Negatives: {false_negatives}")
            report_lines.append("")
        
        # Top high-risk machines
        report_lines.append("=" * 80)
        report_lines.append("Top 5 High-Risk Machines - Detailed Explanations")
        report_lines.append("=" * 80)
        report_lines.append("")
        
        high_risk_explanations = self.explain_high_risk_machines(
            shap_values,
            X,
            predictions,
            top_k=5
        )
        
        for i, explanation in enumerate(high_risk_explanations, 1):
            report_lines.append(f"Machine #{i} (Instance {explanation['instance_idx']}):")
            report_lines.append(f"  Risk Level: {explanation['risk_level']}")
            report_lines.append(f"  Failure Probability: {explanation['prediction']:.1%}")
            report_lines.append("")
            report_lines.append(explanation['summary'])
            report_lines.append("")
            report_lines.append("-" * 80)
            report_lines.append("")
        
        report_text = "\n".join(report_lines)
        
        if output_path:
            with open(output_path, 'w') as f:
                f.write(report_text)
            logger.info(f"Report saved to {output_path}")
        
        return report_text

