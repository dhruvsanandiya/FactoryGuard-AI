"""Data ingestion module for loading and parsing raw IoT sensor logs"""

import pandas as pd
from pathlib import Path
from typing import Optional, List
from src.utils.logger import get_logger
from src.config.settings import Settings

logger = get_logger(__name__)


def load_raw_data(
    file_path: Path,
    timestamp_col: str = Settings.TIMESTAMP_COL,
    machine_id_col: str = Settings.MACHINE_ID_COL,
    parse_dates: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Load raw CSV sensor logs and parse timestamps.
    If timestamp or machine_id columns are missing, creates synthetic ones.
    
    Args:
        file_path: Path to raw CSV file
        timestamp_col: Name of timestamp column
        machine_id_col: Name of machine_id column
        parse_dates: Optional list of date columns to parse
        
    Returns:
        DataFrame with parsed timestamps
    """
    logger.info(f"Loading raw data from {file_path}")
    
    if not file_path.exists():
        raise FileNotFoundError(f"Data file not found: {file_path}")
    
    # First, load without parsing dates to check what columns exist
    try:
        df = pd.read_csv(file_path, low_memory=False)
        logger.info(f"Loaded {len(df)} rows, {len(df.columns)} columns")
        
        # Check if timestamp column exists, if not create synthetic one
        if timestamp_col not in df.columns:
            logger.warning(f"Timestamp column '{timestamp_col}' not found. Creating synthetic timestamps.")
            # Create synthetic timestamps: assume 1-hour intervals starting from a base date
            base_date = pd.Timestamp('2024-01-01 00:00:00')
            df[timestamp_col] = base_date + pd.to_timedelta(df.index, unit='h')
            logger.info(f"Created synthetic timestamps starting from {base_date}")
        else:
            # Parse dates only if column exists
            if parse_dates is None:
                parse_dates = [timestamp_col]
            if timestamp_col in parse_dates:
                if not pd.api.types.is_datetime64_any_dtype(df[timestamp_col]):
                    df[timestamp_col] = pd.to_datetime(df[timestamp_col], errors='coerce')
                    logger.info("Converted timestamp column to datetime")
        
        # Check if machine_id column exists, if not try to infer or create
        if machine_id_col not in df.columns:
            logger.warning(f"Machine ID column '{machine_id_col}' not found.")
            # Try common alternatives
            alternatives = ['Product ID', 'product_id', 'machine_id', 'Machine ID', 'machine', 'id']
            found = False
            for alt in alternatives:
                if alt in df.columns:
                    df[machine_id_col] = df[alt].astype(str)
                    logger.info(f"Using '{alt}' as machine_id")
                    found = True
                    break
            
            if not found:
                # Create synthetic machine_id (all rows same machine if no grouping available)
                df[machine_id_col] = 'machine_1'
                logger.warning("No machine identifier found. Using single machine_id='machine_1'")
        
        return df
    
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        raise


def sort_by_time(
    df: pd.DataFrame,
    timestamp_col: str = Settings.TIMESTAMP_COL,
    machine_id_col: str = Settings.MACHINE_ID_COL
) -> pd.DataFrame:
    """
    Sort data strictly by time per machine_id.
    Preserves time-series ordering.
    
    Args:
        df: Input DataFrame
        timestamp_col: Name of timestamp column
        machine_id_col: Name of machine_id column
        
    Returns:
        Sorted DataFrame
    """
    logger.info("Sorting data by time per machine_id")
    
    if timestamp_col not in df.columns:
        raise ValueError(f"Timestamp column '{timestamp_col}' not found")
    if machine_id_col not in df.columns:
        raise ValueError(f"Machine ID column '{machine_id_col}' not found")
    
    # Sort by machine_id first, then timestamp
    df_sorted = df.sort_values(
        by=[machine_id_col, timestamp_col],
        ascending=[True, True]
    ).reset_index(drop=True)
    
    logger.info(f"Sorted {len(df_sorted)} rows")
    logger.info(f"Unique machines: {df_sorted[machine_id_col].nunique()}")
    
    return df_sorted


def ingest_pipeline(
    file_path: Path,
    timestamp_col: str = Settings.TIMESTAMP_COL,
    machine_id_col: str = Settings.MACHINE_ID_COL
) -> pd.DataFrame:
    """
    Complete ingestion pipeline: load and sort data.
    
    Args:
        file_path: Path to raw CSV file
        timestamp_col: Name of timestamp column
        machine_id_col: Name of machine_id column
        
    Returns:
        Ingested and sorted DataFrame
    """
    logger.info("Starting data ingestion pipeline")
    
    # Load data
    df = load_raw_data(file_path, timestamp_col, machine_id_col)
    
    # Sort by time per machine
    df = sort_by_time(df, timestamp_col, machine_id_col)
    
    # Validate required columns
    if timestamp_col not in df.columns:
        raise ValueError(f"Required column '{timestamp_col}' missing")
    if machine_id_col not in df.columns:
        raise ValueError(f"Required column '{machine_id_col}' missing")
    
    logger.info("Data ingestion pipeline completed successfully")
    logger.info(f"Data shape: {df.shape}")
    logger.info(f"Time range: {df[timestamp_col].min()} to {df[timestamp_col].max()}")
    logger.info(f"Unique machines: {df[machine_id_col].nunique()}")
    
    return df

