"""Configuration settings for data pipeline"""

from pathlib import Path
from typing import Dict, Any
import os


class Settings:
    """Application settings and configuration"""
    
    # Project paths
    PROJECT_ROOT = Path(__file__).parent.parent.parent
    DATA_DIR = PROJECT_ROOT / "data"
    RAW_DATA_DIR = DATA_DIR / "raw"
    PROCESSED_DATA_DIR = DATA_DIR / "processed"
    ARTIFACTS_DIR = DATA_DIR / "artifacts"
    OUTPUT_DIR = DATA_DIR / "output"
    
    # Data processing parameters
    TIMESTAMP_COL = "timestamp"
    MACHINE_ID_COL = "machine_id"
    TARGET_COL = "failure_within_24h"
    
    # Feature engineering parameters
    LAG_WINDOWS = [1, 2]  # t-1, t-2
    ROLLING_WINDOWS = {
        "1h": "1h",
        "4h": "4h",
        "8h": "8h"
    }
    EMA_ALPHAS = [0.3, 0.5, 0.7]
    
    # Target creation
    PREDICTION_HORIZON_HOURS = 24
    
    # Train/test split
    TEST_SIZE = 0.2  # 20% for testing (time-based)
    
    # Data validation (adjust based on your sensor ranges)
    MIN_SENSOR_VALUE = -10000.0  # Very permissive minimum
    MAX_SENSOR_VALUE = 100000.0   # Very permissive maximum (adjust per sensor type)
    
    # Logging
    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
    LOG_DIR = PROJECT_ROOT / "logs"
    
    @classmethod
    def ensure_directories(cls) -> None:
        """Create necessary directories if they don't exist"""
        for dir_path in [
            cls.DATA_DIR,
            cls.RAW_DATA_DIR,
            cls.PROCESSED_DATA_DIR,
            cls.ARTIFACTS_DIR,
            cls.OUTPUT_DIR,
            cls.LOG_DIR
        ]:
            dir_path.mkdir(parents=True, exist_ok=True)
    
    @classmethod
    def get_feature_config(cls) -> Dict[str, Any]:
        """Get feature engineering configuration"""
        return {
            "lag_windows": cls.LAG_WINDOWS,
            "rolling_windows": cls.ROLLING_WINDOWS,
            "ema_alphas": cls.EMA_ALPHAS,
            "prediction_horizon_hours": cls.PREDICTION_HORIZON_HOURS
        }


# Initialize directories on import
Settings.ensure_directories()

