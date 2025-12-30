"""Model inference engine with feature transformation and SHAP explanations"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path
import time
from src.utils.logger import get_logger
from src.explainability.shap_explainer import SHAPExplainer
from src.explainability.insights import InsightGenerator

logger = get_logger(__name__)


class ModelInference:
    """
    Model inference engine with feature transformation and SHAP explanations.
    Handles model loading, feature engineering, prediction, and explainability.
    """
    
    def __init__(
        self,
        model_path: Optional[Path] = None,
        model_type: str = "xgboost",
        background_data_path: Optional[Path] = None
    ):
        """
        Initialize inference engine.
        
        Args:
            model_path: Path to model file (if None, uses latest)
            model_type: Type of model ('xgboost', 'random_forest', 'baseline')
            background_data_path: Path to background data for SHAP (if None, uses test data)
        """
        self.model_type = model_type
        self.model = None
        self.model_path = None
        self.feature_names = None
        self.shap_explainer = None
        self.insight_generator = None
        self.background_data = None
        self._load_start_time = None
        self._load_duration = None
        
        # Load model
        self._load_model(model_path, model_type)
        
        # Load background data for SHAP
        self._load_background_data(background_data_path)
        
        # Initialize SHAP explainer
        self._initialize_shap()
        
        logger.info("Model inference engine initialized successfully")
    
    def _load_model(self, model_path: Optional[Path], model_type: str) -> None:
        """Load model from disk and validate schema"""
        self._load_start_time = time.time()
        
        try:
            from src.utils.model_loader import load_model
            self.model, self.model_path = load_model(model_type=model_type, model_dir=model_path)
            
            # Get feature names
            if hasattr(self.model, 'feature_names_') and self.model.feature_names_:
                self.feature_names = self.model.feature_names_
            else:
                raise ValueError("Model does not have feature_names_ attribute")
            
            # Validate model has required methods
            if not hasattr(self.model, 'predict_proba'):
                raise ValueError("Model must have predict_proba method")
            
            self._load_duration = time.time() - self._load_start_time
            logger.info(f"Model loaded in {self._load_duration:.3f}s")
            logger.info(f"Model has {len(self.feature_names)} features")
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
    
    def _load_background_data(self, background_data_path: Optional[Path]) -> None:
        """Load background data for SHAP explanations"""
        try:
            from src.utils.model_loader import load_test_data, prepare_explainability_data
            if background_data_path:
                test_df = pd.read_parquet(background_data_path)
            else:
                test_df = load_test_data()
            
            # Prepare features
            X_background, _ = prepare_explainability_data(test_df, self.model)
            
            # Sample for efficiency (SHAP can be slow with large datasets)
            if len(X_background) > 100:
                X_background = X_background.sample(n=100, random_state=42)
                logger.info(f"Sampled {len(X_background)} rows for SHAP background")
            
            self.background_data = X_background
            logger.info(f"Loaded {len(X_background)} background samples for SHAP")
            
        except Exception as e:
            logger.warning(f"Could not load background data for SHAP: {e}")
            logger.warning("SHAP explanations will be disabled")
            self.background_data = None
    
    def _initialize_shap(self) -> None:
        """Initialize SHAP explainer"""
        if self.background_data is None:
            logger.warning("SHAP explainer not initialized - no background data")
            return
        
        try:
            self.shap_explainer = SHAPExplainer(
                model=self.model,
                X_background=self.background_data
            )
            
            # Initialize insight generator
            self.insight_generator = InsightGenerator(
                feature_names=self.feature_names
            )
            
            logger.info("SHAP explainer initialized")
            
        except Exception as e:
            logger.warning(f"Could not initialize SHAP explainer: {e}")
            self.shap_explainer = None
            self.insight_generator = None
    
    def _transform_features(
        self,
        machine_id: str,
        temperature: float,
        pressure: float,
        vibration: float
    ) -> pd.DataFrame:
        """
        Transform raw sensor inputs to model features.
        
        This is a simplified transformation that creates basic features.
        In production, you would need historical data for lag/rolling features.
        For now, we create what we can from current values.
        
        Args:
            machine_id: Machine identifier
            temperature: Temperature reading
            pressure: Pressure reading
            vibration: Vibration reading
            
        Returns:
            DataFrame with features matching model schema
        """
        # Create base features from raw inputs
        # Note: This is simplified - real implementation would need historical data
        # for lag and rolling features
        
        features = {}
        
        # Raw sensor values
        features['temperature'] = temperature
        features['pressure'] = pressure
        features['vibration'] = vibration
        
        # Create simplified feature set matching model expectations
        # We'll need to map to actual feature names from the model
        # For now, create a DataFrame and align with model features
        
        # Start with raw values
        base_features = {
            'temperature': temperature,
            'pressure': pressure,
            'vibration': vibration
        }
        
        # Create feature dictionary matching model schema
        # Map raw sensor inputs to model features
        feature_dict = {}
        
        for feat_name in self.feature_names:
            feat_lower = feat_name.lower()
            
            # Match raw sensor features (exact or close match)
            if feat_lower == 'temperature' or (feat_lower.startswith('temperature') and 
                                               'lag' not in feat_lower and 
                                               'rolling' not in feat_lower and 
                                               'ema' not in feat_lower):
                feature_dict[feat_name] = temperature
            elif feat_lower == 'pressure' or (feat_lower.startswith('pressure') and 
                                              'lag' not in feat_lower and 
                                              'rolling' not in feat_lower and 
                                              'ema' not in feat_lower):
                feature_dict[feat_name] = pressure
            elif feat_lower == 'vibration' or (feat_lower.startswith('vibration') and 
                                               'lag' not in feat_lower and 
                                               'rolling' not in feat_lower and 
                                               'ema' not in feat_lower):
                feature_dict[feat_name] = vibration
            # For lag features, use current value as approximation (production needs history)
            elif 'lag' in feat_lower:
                if 'temperature' in feat_lower:
                    feature_dict[feat_name] = temperature  # Approximate with current
                elif 'pressure' in feat_lower:
                    feature_dict[feat_name] = pressure
                elif 'vibration' in feat_lower:
                    feature_dict[feat_name] = vibration
                else:
                    feature_dict[feat_name] = 0.0
            # For rolling/EMA features, use current value (production needs history)
            elif 'rolling' in feat_lower or 'ema' in feat_lower:
                if 'temperature' in feat_lower:
                    feature_dict[feat_name] = temperature
                elif 'pressure' in feat_lower:
                    feature_dict[feat_name] = pressure
                elif 'vibration' in feat_lower:
                    feature_dict[feat_name] = vibration
                else:
                    feature_dict[feat_name] = 0.0
            else:
                # Unknown feature - set to 0
                feature_dict[feat_name] = 0.0
        
        # Create DataFrame with correct feature order
        df = pd.DataFrame([feature_dict])
        
        # Ensure all model features are present
        for feat in self.feature_names:
            if feat not in df.columns:
                df[feat] = 0.0
        
        # Reorder to match model feature order
        df = df[self.feature_names]
        
        return df
    
    def predict(
        self,
        machine_id: str,
        temperature: float,
        pressure: float,
        vibration: float,
        include_explanations: bool = True
    ) -> Dict[str, Any]:
        """
        Make prediction with optional SHAP explanations.
        
        Args:
            machine_id: Machine identifier
            temperature: Temperature reading
            pressure: Pressure reading
            vibration: Vibration reading
            include_explanations: Whether to include SHAP explanations
            
        Returns:
            Dictionary with prediction results
        """
        start_time = time.time()
        
        try:
            # Transform features
            X = self._transform_features(machine_id, temperature, pressure, vibration)
            
            # Make prediction
            proba = self.model.predict_proba(X)[0]
            failure_probability = float(proba[1])  # Positive class probability
            
            # Determine risk level
            if failure_probability >= 0.7:
                risk_level = "HIGH"
            elif failure_probability >= 0.3:
                risk_level = "MEDIUM"
            else:
                risk_level = "LOW"
            
            # Get SHAP explanations if requested and available
            top_risk_factors = []
            shap_explanations = None
            
            if include_explanations and self.shap_explainer is not None:
                try:
                    # Compute SHAP values for this instance
                    shap_values, _ = self.shap_explainer.explain_instance(X)
                    
                    # Get top contributing factors
                    if self.insight_generator:
                        explanation = self.insight_generator.explain_prediction(
                            shap_values=np.array([shap_values]),
                            X=X,
                            instance_idx=0,
                            prediction=failure_probability,
                            top_n=5
                        )
                        
                        # Extract top risk factors
                        top_risk_factors = [
                            factor['explanation'] 
                            for factor in explanation['positive_factors'][:3]
                        ]
                        
                        # Create detailed explanations
                        shap_explanations = [
                            {
                                'feature': factor['feature'],
                                'value': factor['value'],
                                'shap_value': factor['shap_value'],
                                'explanation': factor['explanation']
                            }
                            for factor in explanation['top_features'][:5]
                        ]
                    else:
                        # Fallback: use SHAP values directly
                        abs_shap = np.abs(shap_values[0])
                        top_indices = np.argsort(abs_shap)[-5:][::-1]
                        
                        for idx in top_indices:
                            feat_name = self.feature_names[idx]
                            shap_val = shap_values[0][idx]
                            feat_val = X.iloc[0, idx]
                            
                            # Simple interpretation
                            if shap_val > 0:
                                direction = "increased"
                            else:
                                direction = "decreased"
                            
                            interpretation = f"{feat_name} {direction} failure risk (value: {feat_val:.2f})"
                            top_risk_factors.append(interpretation)
                            
                except Exception as e:
                    logger.warning(f"Could not generate SHAP explanations: {e}")
                    # Fallback to simple risk factors based on input values
                    if temperature > 80:
                        top_risk_factors.append("High temperature")
                    if pressure > 1.5:
                        top_risk_factors.append("Elevated pressure")
                    if vibration > 0.015:
                        top_risk_factors.append("High vibration")
            
            # If no explanations generated, create simple ones
            if not top_risk_factors:
                if failure_probability > 0.5:
                    top_risk_factors = [
                        "High temperature" if temperature > 80 else "Temperature within range",
                        "Pressure instability" if abs(pressure - 1.0) > 0.5 else "Pressure stable",
                        "Vibration trend" if vibration > 0.015 else "Vibration normal"
                    ]
            
            inference_time = time.time() - start_time
            
            result = {
                'failure_probability': failure_probability,
                'risk_level': risk_level,
                'top_risk_factors': top_risk_factors[:3],  # Top 3
                'shap_explanations': shap_explanations,
                'inference_time_ms': inference_time * 1000
            }
            
            logger.info(
                f"Prediction completed in {inference_time*1000:.2f}ms - "
                f"Probability: {failure_probability:.2f}, Risk: {risk_level}"
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            raise
    
    def validate_schema(self) -> bool:
        """Validate that model schema is correct"""
        if self.model is None:
            return False
        if self.feature_names is None or len(self.feature_names) == 0:
            return False
        return True
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information"""
        return {
            'model_type': self.model_type,
            'model_path': str(self.model_path) if self.model_path else None,
            'num_features': len(self.feature_names) if self.feature_names else 0,
            'feature_names': self.feature_names[:10] if self.feature_names else [],  # First 10
            'shap_enabled': self.shap_explainer is not None,
            'load_time_ms': self._load_duration * 1000 if self._load_duration else None
        }

