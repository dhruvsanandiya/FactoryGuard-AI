"""Data cleaning module with time-aware interpolation and validation"""

import pandas as pd
import numpy as np
from typing import List, Optional, Dict
from src.utils.logger import get_logger
from src.config.settings import Settings

logger = get_logger(__name__)


def remove_impossible_values(
    df: pd.DataFrame,
    sensor_cols: Optional[List[str]] = None,
    min_value: float = Settings.MIN_SENSOR_VALUE,
    max_value: float = Settings.MAX_SENSOR_VALUE,
    timestamp_col: str = Settings.TIMESTAMP_COL,
    machine_id_col: str = Settings.MACHINE_ID_COL
) -> pd.DataFrame:
    """
    Remove impossible sensor values (outliers beyond reasonable range).
    
    Args:
        df: Input DataFrame
        sensor_cols: List of sensor column names (if None, auto-detect numeric columns)
        min_value: Minimum allowed sensor value
        max_value: Maximum allowed sensor value
        timestamp_col: Name of timestamp column
        machine_id_col: Name of machine_id column
        
    Returns:
        DataFrame with impossible values set to NaN
    """
    logger.info("Removing impossible sensor values")
    
    df_clean = df.copy()
    
    # Auto-detect sensor columns if not provided
    if sensor_cols is None:
        # Exclude ID columns, timestamp, machine_id, and failure indicators
        exclude_cols = {
            timestamp_col, machine_id_col, "UDI", "udi", "id", "ID",
            "Type", "TWF", "HDF", "PWF", "OSF", "RNF"
        }
        sensor_cols = [
            col for col in df.columns
            if col not in exclude_cols and pd.api.types.is_numeric_dtype(df[col])
        ]
    
    if not sensor_cols:
        logger.warning("No sensor columns found for cleaning")
        return df_clean
    
    # Count outliers before removal
    outliers_before = {}
    for col in sensor_cols:
        outliers = ((df_clean[col] < min_value) | (df_clean[col] > max_value)).sum()
        if outliers > 0:
            outliers_before[col] = outliers
    
    # Set impossible values to NaN
    for col in sensor_cols:
        mask = (df_clean[col] < min_value) | (df_clean[col] > max_value)
        n_outliers = mask.sum()
        if n_outliers > 0:
            df_clean.loc[mask, col] = np.nan
            logger.info(f"Removed {n_outliers} impossible values from {col}")
    
    if outliers_before:
        total_outliers = sum(outliers_before.values())
        logger.info(f"Total impossible values removed: {total_outliers}")
    else:
        logger.info("No impossible values found")
    
    return df_clean


def interpolate_missing_values(
    df: pd.DataFrame,
    sensor_cols: Optional[List[str]] = None,
    method: str = "time",
    timestamp_col: str = Settings.TIMESTAMP_COL,
    machine_id_col: str = Settings.MACHINE_ID_COL
) -> pd.DataFrame:
    """
    Time-aware interpolation of missing values per machine.
    
    Args:
        df: Input DataFrame
        sensor_cols: List of sensor column names (if None, auto-detect numeric columns)
        method: Interpolation method ('time', 'linear', etc.)
        timestamp_col: Name of timestamp column
        machine_id_col: Name of machine_id column
        
    Returns:
        DataFrame with interpolated missing values
    """
    logger.info(f"Interpolating missing values using '{method}' method")
    
    df_clean = df.copy()
    
    # Auto-detect sensor columns if not provided
    if sensor_cols is None:
        exclude_cols = {timestamp_col, machine_id_col}
        sensor_cols = [
            col for col in df.columns
            if col not in exclude_cols and pd.api.types.is_numeric_dtype(df[col])
        ]
    
    if not sensor_cols:
        logger.warning("No sensor columns found for interpolation")
        return df_clean
    
    # Count missing values before interpolation
    missing_before = {}
    for col in sensor_cols:
        missing = df_clean[col].isna().sum()
        if missing > 0:
            missing_before[col] = missing
    
    # Set timestamp as index for time-aware interpolation
    original_index = df_clean.index
    
    # Interpolate per machine to avoid cross-machine leakage
    for machine_id in df_clean[machine_id_col].unique():
        machine_mask = df_clean[machine_id_col] == machine_id
        machine_df = df_clean[machine_mask].copy()
        
        if len(machine_df) == 0:
            continue
        
        # Skip interpolation if only one row (need at least 2 points)
        if len(machine_df) < 2:
            continue
        
        # Sort by timestamp for proper interpolation
        machine_df = machine_df.sort_values(timestamp_col).copy()
        
        # Store original index for alignment
        original_indices = machine_df.index.tolist()
        
        # For time-based interpolation, we need DatetimeIndex
        # Create a temporary dataframe with timestamp as index
        try:
            # Ensure timestamp is datetime
            if not pd.api.types.is_datetime64_any_dtype(machine_df[timestamp_col]):
                machine_df[timestamp_col] = pd.to_datetime(machine_df[timestamp_col])
            
            # Create indexed version
            temp_df = machine_df.set_index(timestamp_col)
            
            # Interpolate sensor columns
            for col in sensor_cols:
                if col not in temp_df.columns:
                    continue
                    
                try:
                    # Interpolate using time method
                    interpolated = temp_df[col].interpolate(
                        method=method,
                        limit_direction='both'
                    )
                    
                    # Update the main dataframe
                    df_clean.loc[original_indices, col] = interpolated.values
                except Exception as e:
                    logger.debug(f"Interpolation issue for {col} in machine {machine_id}: {e}")
                    continue
                    
        except Exception as e:
            logger.warning(f"Could not interpolate for machine {machine_id}: {e}")
            continue
    
    # Log interpolation results
    if missing_before:
        for col, n_missing in missing_before.items():
            missing_after = df_clean[col].isna().sum()
            interpolated = n_missing - missing_after
            logger.info(
                f"{col}: {n_missing} missing -> {missing_after} remaining "
                f"({interpolated} interpolated)"
            )
    else:
        logger.info("No missing values found")
    
    return df_clean


def clean_pipeline(
    df: pd.DataFrame,
    sensor_cols: Optional[List[str]] = None,
    timestamp_col: str = Settings.TIMESTAMP_COL,
    machine_id_col: str = Settings.MACHINE_ID_COL,
    min_value: float = Settings.MIN_SENSOR_VALUE,
    max_value: float = Settings.MAX_SENSOR_VALUE
) -> pd.DataFrame:
    """
    Complete cleaning pipeline: remove outliers and interpolate missing values.
    
    Args:
        df: Input DataFrame
        sensor_cols: List of sensor column names (if None, auto-detect)
        timestamp_col: Name of timestamp column
        machine_id_col: Name of machine_id column
        min_value: Minimum allowed sensor value
        max_value: Maximum allowed sensor value
        
    Returns:
        Cleaned DataFrame
    """
    logger.info("Starting data cleaning pipeline")
    
    initial_shape = df.shape
    initial_missing = df.isna().sum().sum()
    
    # Step 1: Remove impossible values
    df_clean = remove_impossible_values(
        df,
        sensor_cols=sensor_cols,
        min_value=min_value,
        max_value=max_value,
        timestamp_col=timestamp_col,
        machine_id_col=machine_id_col
    )
    
    # Step 2: Interpolate missing values (time-aware, per machine)
    df_clean = interpolate_missing_values(
        df_clean,
        sensor_cols=sensor_cols,
        method="time",
        timestamp_col=timestamp_col,
        machine_id_col=machine_id_col
    )
    
    # Final statistics
    final_shape = df_clean.shape
    final_missing = df_clean.isna().sum().sum()
    
    logger.info("Data cleaning pipeline completed")
    logger.info(f"Shape: {initial_shape} -> {final_shape}")
    logger.info(f"Missing values: {initial_missing} -> {final_missing}")
    
    return df_clean

