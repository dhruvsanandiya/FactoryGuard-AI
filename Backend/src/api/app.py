"""Flask application factory for FactoryGuard AI API"""

from flask import Flask
from flask_cors import CORS
import os
from pathlib import Path
from src.api.routes import api_bp, init_inference_engine
from src.config.settings import Settings
from src.utils.logger import get_logger

logger = get_logger(__name__)


def create_app(
    model_path: str = None,
    model_type: str = "xgboost",
    config: dict = None
) -> Flask:
    """
    Create and configure Flask application.
    
    Args:
        model_path: Path to model directory (if None, uses latest)
        model_type: Type of model ('xgboost', 'random_forest', 'baseline')
        config: Optional configuration dictionary
        
    Returns:
        Configured Flask application
    """
    app = Flask(__name__)
    
    # Load configuration
    if config:
        app.config.update(config)
    else:
        app.config.update({
            'DEBUG': os.getenv('FLASK_DEBUG', 'False').lower() == 'true',
            'TESTING': os.getenv('FLASK_TESTING', 'False').lower() == 'true'
        })
    
    # Enable CORS
    CORS(app)
    
    # Register blueprints
    app.register_blueprint(api_bp, url_prefix='/api/v1')
    
    # Root endpoint with API information
    @app.route('/', methods=['GET'])
    def root():
        """Root endpoint with API information"""
        from flask import jsonify
        return jsonify({
            'name': 'FactoryGuard AI API',
            'version': '1.0.0',
            'description': 'Production REST API for real-time machine failure prediction',
            'status': 'running',
            'endpoints': {
                'root': '/',
                'health': '/api/v1/health',
                'predict': '/api/v1/predict',
                'model_info': '/api/v1/model/info'
            },
            'documentation': 'See API_README.md for complete documentation',
            'example_request': {
                'url': '/api/v1/predict',
                'method': 'POST',
                'body': {
                    'machine_id': 'M_204',
                    'temperature': 82.4,
                    'pressure': 1.9,
                    'vibration': 0.02
                }
            }
        }), 200
    
    # Favicon handler to prevent 404 errors
    @app.route('/favicon.ico', methods=['GET'])
    def favicon():
        """Favicon endpoint - return 204 No Content"""
        from flask import Response
        return Response(status=204)
    
    # Initialize model at startup (before first request)
    # For Flask 2.2+, we initialize immediately
    try:
        logger.info("Initializing model at startup...")
        init_inference_engine(model_path=model_path, model_type=model_type)
        logger.info("Model initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize model: {e}")
        # Don't raise - allow app to start, but health check will fail
    
    @app.before_request
    def ensure_model_loaded():
        """Ensure model is loaded before handling requests"""
        from src.api.routes import inference_engine
        if inference_engine is None and request.endpoint not in ['api.health_check', None]:
            # Try to initialize if not already done
            try:
                init_inference_engine(model_path=model_path, model_type=model_type)
            except Exception as e:
                logger.error(f"Model initialization failed: {e}")
    
    # Error handlers
    @app.errorhandler(404)
    def not_found(error):
        return {'error': 'Endpoint not found'}, 404
    
    @app.errorhandler(500)
    def internal_error(error):
        logger.error(f"Internal server error: {error}")
        return {'error': 'Internal server error'}, 500
    
    logger.info("Flask application created")
    
    return app


def run_app(
    host: str = "0.0.0.0",
    port: int = 5000,
    debug: bool = False,
    model_path: str = None,
    model_type: str = "xgboost"
) -> None:
    """
    Run the Flask application.
    
    Args:
        host: Host to bind to
        port: Port to bind to
        debug: Enable debug mode
        model_path: Path to model directory
        model_type: Type of model to load
    """
    app = create_app(model_path=model_path, model_type=model_type)
    
    # Initialize model before running
    try:
        logger.info("Initializing model...")
        init_inference_engine(model_path=model_path, model_type=model_type)
        logger.info("Model loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise
    
    logger.info(f"Starting FactoryGuard AI API on {host}:{port}")
    app.run(host=host, port=port, debug=debug)


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='FactoryGuard AI REST API')
    parser.add_argument('--host', type=str, default='0.0.0.0', help='Host to bind to')
    parser.add_argument('--port', type=int, default=5000, help='Port to bind to')
    parser.add_argument('--model-path', type=str, default=None, help='Path to model directory')
    parser.add_argument('--model-type', type=str, default='xgboost', choices=['xgboost', 'random_forest', 'baseline'],
                       help='Type of model to load')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode')
    
    args = parser.parse_args()
    
    run_app(
        host=args.host,
        port=args.port,
        debug=args.debug,
        model_path=args.model_path,
        model_type=args.model_type
    )

