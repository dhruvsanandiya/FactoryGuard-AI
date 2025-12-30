"""Structured logging utility for FactoryGuard AI"""

import logging
import sys
from pathlib import Path
from datetime import datetime
from typing import Optional
from src.config.settings import Settings


def setup_logger(
    name: str = "factoryguard",
    log_level: str = Settings.LOG_LEVEL,
    log_file: Optional[Path] = None
) -> logging.Logger:
    """
    Setup structured logger with file and console handlers.
    
    Args:
        name: Logger name
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Optional path to log file
        
    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, log_level.upper()))
    
    # Avoid duplicate handlers
    if logger.handlers:
        return logger
    
    # Formatter
    formatter = logging.Formatter(
        fmt="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(getattr(logging, log_level.upper()))
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler (if specified)
    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(getattr(logging, log_level.upper()))
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger


def get_logger(name: str = "factoryguard") -> logging.Logger:
    """
    Get or create logger instance.
    
    Args:
        name: Logger name
        
    Returns:
        Logger instance
    """
    logger = logging.getLogger(name)
    if not logger.handlers:
        log_file = Settings.LOG_DIR / f"{name}_{datetime.now().strftime('%Y%m%d')}.log"
        return setup_logger(name, Settings.LOG_LEVEL, log_file)
    return logger

