"""Flask API routes for FactoryGuard AI"""

from flask import Blueprint, request, jsonify
from typing import Dict, Any
import time
from src.api.schemas import PredictionRequest, PredictionResponse, HealthResponse
from src.api.inference import ModelInference
from src.utils.logger import get_logger

logger = get_logger(__name__)

# Global inference engine (loaded at startup)
inference_engine: ModelInference = None


def init_inference_engine(model_path: str = None, model_type: str = "xgboost") -> None:
    """Initialize the global inference engine"""
    global inference_engine
    from pathlib import Path
    
    try:
        model_path_obj = Path(model_path) if model_path else None
        inference_engine = ModelInference(
            model_path=model_path_obj,
            model_type=model_type
        )
        logger.info("Inference engine initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize inference engine: {e}")
        raise


# Create Blueprint
api_bp = Blueprint('api', __name__)


@api_bp.route('/', methods=['GET'])
def root() -> Dict[str, Any]:
    """Root endpoint with API information"""
    return jsonify({
        'name': 'FactoryGuard AI API',
        'version': '1.0.0',
        'description': 'Production REST API for real-time machine failure prediction',
        'endpoints': {
            'health': '/api/v1/health',
            'predict': '/api/v1/predict',
            'model_info': '/api/v1/model/info'
        },
        'documentation': 'See API_README.md for complete documentation'
    }), 200


@api_bp.route('/health', methods=['GET'])
def health_check() -> Dict[str, Any]:
    """Health check endpoint"""
    try:
        if inference_engine is None:
            return jsonify({
                'status': 'unhealthy',
                'model_loaded': False,
                'error': 'Inference engine not initialized'
            }), 503
        
        model_info = inference_engine.get_model_info()
        
        return jsonify({
            'status': 'healthy',
            'model_loaded': True,
            'model_type': model_info['model_type'],
            'shap_enabled': model_info['shap_enabled']
        }), 200
        
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return jsonify({
            'status': 'unhealthy',
            'model_loaded': False,
            'error': str(e)
        }), 503


@api_bp.route('/predict', methods=['POST'])
def predict() -> Dict[str, Any]:
    """
    Prediction endpoint.
    
    Accepts JSON with machine_id, temperature, pressure, vibration.
    Returns failure probability, risk level, and top risk factors.
    """
    request_start_time = time.time()
    
    try:
        # Validate request
        if not request.is_json:
            return jsonify({
                'error': 'Content-Type must be application/json'
            }), 400
        
        data = request.get_json()
        
        # Validate using Pydantic schema
        try:
            pred_request = PredictionRequest(**data)
        except Exception as e:
            return jsonify({
                'error': 'Invalid request data',
                'details': str(e)
            }), 400
        
        # Check inference engine is loaded
        if inference_engine is None:
            return jsonify({
                'error': 'Model not loaded. Service unavailable.'
            }), 503
        
        # Make prediction
        result = inference_engine.predict(
            machine_id=pred_request.machine_id,
            temperature=pred_request.temperature,
            pressure=pred_request.pressure,
            vibration=pred_request.vibration,
            include_explanations=True
        )
        
        # Build response
        response_data = {
            'failure_probability': result['failure_probability'],
            'risk_level': result['risk_level'],
            'top_risk_factors': result['top_risk_factors']
        }
        
        # Add SHAP explanations if available
        if result.get('shap_explanations'):
            response_data['shap_explanations'] = result['shap_explanations']
        
        # Log request timing
        total_time = (time.time() - request_start_time) * 1000
        inference_time = result.get('inference_time_ms', 0)
        
        logger.info(
            f"Prediction request completed - "
            f"Total: {total_time:.2f}ms, Inference: {inference_time:.2f}ms, "
            f"Risk: {result['risk_level']}"
        )
        
        # Validate latency target (<50ms)
        if total_time > 50:
            logger.warning(f"Request latency {total_time:.2f}ms exceeds 50ms target")
        
        return jsonify(response_data), 200
        
    except ValueError as e:
        logger.error(f"Validation error: {e}")
        return jsonify({
            'error': 'Invalid input',
            'details': str(e)
        }), 400
        
    except Exception as e:
        logger.error(f"Prediction error: {e}", exc_info=True)
        return jsonify({
            'error': 'Internal server error',
            'details': str(e) if logger.level <= 10 else 'An error occurred'
        }), 500


@api_bp.route('/model/info', methods=['GET'])
def model_info() -> Dict[str, Any]:
    """Get model information"""
    try:
        if inference_engine is None:
            return jsonify({
                'error': 'Model not loaded'
            }), 503
        
        info = inference_engine.get_model_info()
        return jsonify(info), 200
        
    except Exception as e:
        logger.error(f"Error getting model info: {e}")
        return jsonify({
            'error': str(e)
        }), 500

