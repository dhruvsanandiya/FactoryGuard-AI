"""Integration tests for API endpoints"""

import pytest
from flask import Flask
from unittest.mock import patch, MagicMock
from src.api.app import create_app
from src.api.routes import inference_engine, init_inference_engine


class TestAPIEndpoints:
    """Test cases for API endpoints"""
    
    @pytest.fixture
    def app(self):
        """Create test Flask app"""
        with patch('src.api.routes.init_inference_engine'):
            app = create_app()
            app.config['TESTING'] = True
            return app
    
    @pytest.fixture
    def client(self, app):
        """Create test client"""
        return app.test_client()
    
    @pytest.fixture
    def mock_inference_engine(self):
        """Create mock inference engine"""
        mock_engine = MagicMock()
        mock_engine.predict = MagicMock(return_value={
            'failure_probability': 0.86,
            'risk_level': 'HIGH',
            'top_risk_factors': ['High temperature', 'Pressure instability'],
            'shap_explanations': None,
            'inference_time_ms': 15.5
        })
        mock_engine.get_model_info = MagicMock(return_value={
            'model_type': 'xgboost',
            'num_features': 50,
            'shap_enabled': True
        })
        return mock_engine
    
    def test_health_check_healthy(self, client, mock_inference_engine):
        """Test health check when model is loaded"""
        with patch('src.api.routes.inference_engine', mock_inference_engine):
            response = client.get('/api/v1/health')
            assert response.status_code == 200
            data = response.get_json()
            assert data['status'] == 'healthy'
            assert data['model_loaded'] == True
    
    def test_health_check_unhealthy(self, client):
        """Test health check when model is not loaded"""
        with patch('src.api.routes.inference_engine', None):
            response = client.get('/api/v1/health')
            assert response.status_code == 503
            data = response.get_json()
            assert data['status'] == 'unhealthy'
            assert data['model_loaded'] == False
    
    def test_predict_success(self, client, mock_inference_engine):
        """Test successful prediction"""
        with patch('src.api.routes.inference_engine', mock_inference_engine):
            response = client.post(
                '/api/v1/predict',
                json={
                    'machine_id': 'M_204',
                    'temperature': 82.4,
                    'pressure': 1.9,
                    'vibration': 0.02
                },
                content_type='application/json'
            )
            assert response.status_code == 200
            data = response.get_json()
            assert 'failure_probability' in data
            assert 'risk_level' in data
            assert 'top_risk_factors' in data
            assert data['risk_level'] == 'HIGH'
    
    def test_predict_invalid_json(self, client):
        """Test prediction with invalid JSON"""
        response = client.post(
            '/api/v1/predict',
            data='not json',
            content_type='text/plain'
        )
        assert response.status_code == 400
    
    def test_predict_missing_fields(self, client):
        """Test prediction with missing required fields"""
        response = client.post(
            '/api/v1/predict',
            json={
                'machine_id': 'M_204',
                'temperature': 82.4
                # Missing pressure and vibration
            },
            content_type='application/json'
        )
        assert response.status_code == 400
    
    def test_predict_invalid_values(self, client):
        """Test prediction with invalid sensor values"""
        response = client.post(
            '/api/v1/predict',
            json={
                'machine_id': 'M_204',
                'temperature': 'not a number',
                'pressure': 1.9,
                'vibration': 0.02
            },
            content_type='application/json'
        )
        assert response.status_code == 400
    
    def test_model_info(self, client, mock_inference_engine):
        """Test model info endpoint"""
        with patch('src.api.routes.inference_engine', mock_inference_engine):
            response = client.get('/api/v1/model/info')
            assert response.status_code == 200
            data = response.get_json()
            assert 'model_type' in data
            assert 'num_features' in data


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

