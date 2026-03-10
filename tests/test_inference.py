"""Unit tests for inference engine"""

import pytest
import numpy as np
import pandas as pd
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from src.api.inference import ModelInference


class TestModelInference:
    """Test cases for ModelInference class"""
    
    @pytest.fixture
    def mock_model(self):
        """Create a mock model"""
        model = Mock()
        model.feature_names_ = [
            'temperature', 'pressure', 'vibration',
            'temperature_lag_1', 'pressure_rolling_mean_1h'
        ]
        model.predict_proba = Mock(return_value=np.array([[0.2, 0.8]]))
        model.model = Mock()  # For SHAP
        return model
    
    @pytest.fixture
    def mock_background_data(self):
        """Create mock background data"""
        return pd.DataFrame({
            'temperature': [70, 75, 80],
            'pressure': [1.0, 1.2, 1.5],
            'vibration': [0.01, 0.015, 0.02],
            'temperature_lag_1': [68, 72, 78],
            'pressure_rolling_mean_1h': [0.9, 1.1, 1.3]
        })
    
    @patch('src.api.inference.load_model')
    @patch('src.api.inference.load_test_data')
    @patch('src.api.inference.prepare_explainability_data')
    def test_model_loading(self, mock_prep, mock_load_test, mock_load_model, mock_model):
        """Test model loading"""
        mock_load_model.return_value = (mock_model, Path('/fake/path'))
        mock_load_test.return_value = pd.DataFrame({'col1': [1, 2, 3]})
        mock_prep.return_value = (pd.DataFrame({'temp': [1, 2]}), pd.Series([0, 1]))
        
        inference = ModelInference()
        
        assert inference.model is not None
        assert inference.feature_names == mock_model.feature_names_
        assert len(inference.feature_names) == 5
    
    @patch('src.api.inference.load_model')
    @patch('src.api.inference.load_test_data')
    @patch('src.api.inference.prepare_explainability_data')
    def test_feature_transformation(self, mock_prep, mock_load_test, mock_load_model, mock_model):
        """Test feature transformation"""
        mock_load_model.return_value = (mock_model, Path('/fake/path'))
        mock_load_test.return_value = pd.DataFrame({'col1': [1, 2, 3]})
        mock_prep.return_value = (pd.DataFrame({'temp': [1, 2]}), pd.Series([0, 1]))
        
        inference = ModelInference()
        
        X = inference._transform_features(
            machine_id='M_001',
            temperature=82.5,
            pressure=1.9,
            vibration=0.02
        )
        
        assert isinstance(X, pd.DataFrame)
        assert len(X) == 1
        assert len(X.columns) == len(mock_model.feature_names_)
        assert all(col in X.columns for col in mock_model.feature_names_)
    
    @patch('src.api.inference.load_model')
    @patch('src.api.inference.load_test_data')
    @patch('src.api.inference.prepare_explainability_data')
    @patch('src.api.inference.SHAPExplainer')
    def test_prediction(self, mock_shap, mock_prep, mock_load_test, mock_load_model, mock_model, mock_background_data):
        """Test prediction"""
        mock_load_model.return_value = (mock_model, Path('/fake/path'))
        mock_load_test.return_value = pd.DataFrame({'col1': [1, 2, 3]})
        mock_prep.return_value = (mock_background_data, pd.Series([0, 1, 0]))
        
        # Mock SHAP explainer
        mock_shap_instance = Mock()
        mock_shap_instance.explain_instance = Mock(return_value=(
            np.array([[0.1, 0.2, 0.3, 0.4, 0.5]]),
            np.array([[82.5, 1.9, 0.02, 80.0, 1.7]])
        ))
        mock_shap.return_value = mock_shap_instance
        
        inference = ModelInference()
        
        result = inference.predict(
            machine_id='M_001',
            temperature=82.5,
            pressure=1.9,
            vibration=0.02,
            include_explanations=False
        )
        
        assert 'failure_probability' in result
        assert 'risk_level' in result
        assert 'top_risk_factors' in result
        assert 0 <= result['failure_probability'] <= 1
        assert result['risk_level'] in ['LOW', 'MEDIUM', 'HIGH']
    
    @patch('src.api.inference.load_model')
    @patch('src.api.inference.load_test_data')
    @patch('src.api.inference.prepare_explainability_data')
    def test_risk_level_classification(self, mock_prep, mock_load_test, mock_load_model, mock_model):
        """Test risk level classification"""
        mock_load_model.return_value = (mock_model, Path('/fake/path'))
        mock_load_test.return_value = pd.DataFrame({'col1': [1, 2, 3]})
        mock_prep.return_value = (pd.DataFrame({'temp': [1, 2]}), pd.Series([0, 1]))
        
        # Test HIGH risk
        mock_model.predict_proba = Mock(return_value=np.array([[0.1, 0.9]]))
        inference = ModelInference()
        result = inference.predict('M_001', 85.0, 2.0, 0.03, include_explanations=False)
        assert result['risk_level'] == 'HIGH'
        
        # Test MEDIUM risk
        mock_model.predict_proba = Mock(return_value=np.array([[0.4, 0.6]]))
        inference = ModelInference()
        result = inference.predict('M_001', 75.0, 1.5, 0.015, include_explanations=False)
        assert result['risk_level'] == 'MEDIUM'
        
        # Test LOW risk
        mock_model.predict_proba = Mock(return_value=np.array([[0.8, 0.2]]))
        inference = ModelInference()
        result = inference.predict('M_001', 70.0, 1.0, 0.01, include_explanations=False)
        assert result['risk_level'] == 'LOW'
    
    @patch('src.api.inference.load_model')
    @patch('src.api.inference.load_test_data')
    @patch('src.api.inference.prepare_explainability_data')
    def test_schema_validation(self, mock_prep, mock_load_test, mock_load_model, mock_model):
        """Test schema validation"""
        mock_load_model.return_value = (mock_model, Path('/fake/path'))
        mock_load_test.return_value = pd.DataFrame({'col1': [1, 2, 3]})
        mock_prep.return_value = (pd.DataFrame({'temp': [1, 2]}), pd.Series([0, 1]))
        
        inference = ModelInference()
        
        assert inference.validate_schema() == True
        
        # Test with invalid model
        inference.model = None
        assert inference.validate_schema() == False


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

