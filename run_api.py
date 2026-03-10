"""Convenience script to run FactoryGuard AI API"""

import argparse
from pathlib import Path
from src.api.app import run_app
from src.config.settings import Settings
from src.utils.logger import get_logger

logger = get_logger(__name__)


def main():
    """Main entry point for API server"""
    parser = argparse.ArgumentParser(
        description='FactoryGuard AI - Production REST API',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage (uses latest model)
  python run_api.py

  # Custom model path
  python run_api.py --model-path data/artifacts/models_20251231_013620

  # Custom port
  python run_api.py --port 8080

  # Debug mode
  python run_api.py --debug
        """
    )
    
    parser.add_argument(
        '--host',
        type=str,
        default='0.0.0.0',
        help='Host to bind to (default: 0.0.0.0)'
    )
    
    parser.add_argument(
        '--port',
        type=int,
        default=5000,
        help='Port to bind to (default: 5000)'
    )
    
    parser.add_argument(
        '--model-path',
        type=str,
        default=None,
        help='Path to model directory (default: latest in artifacts/)'
    )
    
    parser.add_argument(
        '--model-type',
        type=str,
        default='xgboost',
        choices=['xgboost', 'random_forest', 'baseline'],
        help='Type of model to load (default: xgboost)'
    )
    
    parser.add_argument(
        '--debug',
        action='store_true',
        help='Enable debug mode'
    )
    
    args = parser.parse_args()
    
    # Validate model path if provided
    if args.model_path:
        model_path = Path(args.model_path)
        if not model_path.exists():
            logger.error(f"Model path does not exist: {model_path}")
            return 1
        if not model_path.is_dir():
            logger.error(f"Model path is not a directory: {model_path}")
            return 1
    
    logger.info("=" * 60)
    logger.info("FactoryGuard AI - Production REST API")
    logger.info("=" * 60)
    logger.info(f"Host: {args.host}")
    logger.info(f"Port: {args.port}")
    logger.info(f"Model Type: {args.model_type}")
    logger.info(f"Model Path: {args.model_path or 'Latest in artifacts/'}")
    logger.info(f"Debug: {args.debug}")
    logger.info("=" * 60)
    
    try:
        run_app(
            host=args.host,
            port=args.port,
            debug=args.debug,
            model_path=args.model_path,
            model_type=args.model_type
        )
    except KeyboardInterrupt:
        logger.info("Shutting down API server...")
        return 0
    except Exception as e:
        logger.error(f"Failed to start API server: {e}")
        return 1


if __name__ == '__main__':
    exit(main())

