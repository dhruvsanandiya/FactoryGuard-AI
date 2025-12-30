"""Train/test split module with time-based splitting and versioned artifacts"""

import pandas as pd
from pathlib import Path
from typing import Tuple, Optional
from datetime import datetime
from src.utils.logger import get_logger
from src.config.settings import Settings

logger = get_logger(__name__)


def time_based_split(
    df: pd.DataFrame,
    test_size: float = Settings.TEST_SIZE,
    timestamp_col: str = Settings.TIMESTAMP_COL,
    machine_id_col: str = Settings.MACHINE_ID_COL
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Time-based train/test split (no random split to prevent leakage).
    Splits at a time threshold to ensure test set is strictly after train set.
    
    Args:
        df: Input DataFrame (must be sorted by time)
        test_size: Proportion of data for test set (0.0 to 1.0)
        timestamp_col: Name of timestamp column
        machine_id_col: Name of machine_id column
        
    Returns:
        Tuple of (train_df, test_df)
    """
    logger.info(f"Performing time-based split (test_size={test_size})")
    
    if timestamp_col not in df.columns:
        raise ValueError(f"Timestamp column '{timestamp_col}' not found")
    
    # Ensure timestamp is datetime
    if not pd.api.types.is_datetime64_any_dtype(df[timestamp_col]):
        df[timestamp_col] = pd.to_datetime(df[timestamp_col])
    
    # Sort by timestamp to ensure proper ordering
    df_sorted = df.sort_values(by=timestamp_col).reset_index(drop=True)
    
    # Calculate split point based on time
    total_rows = len(df_sorted)
    split_idx = int(total_rows * (1 - test_size))
    
    # Get split timestamp threshold
    split_timestamp = df_sorted.iloc[split_idx][timestamp_col]
    
    # Split: train = before threshold, test = at or after threshold
    train_df = df_sorted[df_sorted[timestamp_col] < split_timestamp].copy()
    test_df = df_sorted[df_sorted[timestamp_col] >= split_timestamp].copy()
    
    train_time_range = (
        train_df[timestamp_col].min(),
        train_df[timestamp_col].max()
    )
    test_time_range = (
        test_df[timestamp_col].min(),
        test_df[timestamp_col].max()
    )
    
    logger.info(f"Train set: {len(train_df)} rows ({train_time_range[0]} to {train_time_range[1]})")
    logger.info(f"Test set: {len(test_df)} rows ({test_time_range[0]} to {test_time_range[1]})")
    
    # Validate no overlap
    if len(train_df) > 0 and len(test_df) > 0:
        max_train_time = train_df[timestamp_col].max()
        min_test_time = test_df[timestamp_col].min()
        
        if max_train_time >= min_test_time:
            logger.warning(
                f"Potential time overlap detected: "
                f"max_train={max_train_time}, min_test={min_test_time}"
            )
        else:
            logger.info("Time-based split validated: no overlap")
    
    return train_df, test_df


def save_datasets(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    output_dir: Path = Settings.ARTIFACTS_DIR,
    version: Optional[str] = None,
    format: str = "parquet"
) -> Tuple[Path, Path]:
    """
    Save train and test datasets as versioned artifacts.
    
    Args:
        train_df: Training DataFrame
        test_df: Test DataFrame
        output_dir: Output directory for artifacts
        version: Version string (if None, uses timestamp)
        format: File format ('parquet', 'csv', or 'both')
        
    Returns:
        Tuple of (train_path, test_path)
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if version is None:
        version = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    logger.info(f"Saving datasets with version: {version}")
    
    train_path = None
    test_path = None
    
    if format in ("parquet", "both"):
        train_path = output_dir / f"train_{version}.parquet"
        test_path = output_dir / f"test_{version}.parquet"
        
        train_df.to_parquet(train_path, index=False)
        test_df.to_parquet(test_path, index=False)
        
        logger.info(f"Saved Parquet files: {train_path.name}, {test_path.name}")
    
    if format in ("csv", "both"):
        train_path_csv = output_dir / f"train_{version}.csv"
        test_path_csv = output_dir / f"test_{version}.csv"
        
        train_df.to_csv(train_path_csv, index=False)
        test_df.to_csv(test_path_csv, index=False)
        
        logger.info(f"Saved CSV files: {train_path_csv.name}, {test_path_csv.name}")
        
        if train_path is None:
            train_path = train_path_csv
            test_path = test_path_csv
    
    return train_path, test_path


def split_pipeline(
    df: pd.DataFrame,
    test_size: float = Settings.TEST_SIZE,
    timestamp_col: str = Settings.TIMESTAMP_COL,
    machine_id_col: str = Settings.MACHINE_ID_COL,
    output_dir: Path = Settings.ARTIFACTS_DIR,
    version: Optional[str] = None,
    save: bool = True,
    format: str = "parquet"
) -> Tuple[pd.DataFrame, pd.DataFrame, Optional[Path], Optional[Path]]:
    """
    Complete split pipeline: time-based split and save artifacts.
    
    Args:
        df: Input DataFrame
        test_size: Proportion of data for test set
        timestamp_col: Name of timestamp column
        machine_id_col: Name of machine_id column
        output_dir: Output directory for artifacts
        version: Version string (if None, uses timestamp)
        save: Whether to save datasets to disk
        format: File format ('parquet', 'csv', or 'both')
        
    Returns:
        Tuple of (train_df, test_df, train_path, test_path)
    """
    logger.info("Starting train/test split pipeline")
    
    # Perform time-based split
    train_df, test_df = time_based_split(
        df,
        test_size=test_size,
        timestamp_col=timestamp_col,
        machine_id_col=machine_id_col
    )
    
    train_path = None
    test_path = None
    
    # Save datasets if requested
    if save:
        train_path, test_path = save_datasets(
            train_df,
            test_df,
            output_dir=output_dir,
            version=version,
            format=format
        )
    
    logger.info("Train/test split pipeline completed")
    
    return train_df, test_df, train_path, test_path

