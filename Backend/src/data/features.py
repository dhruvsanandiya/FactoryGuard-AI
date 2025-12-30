"""Feature engineering module with lag features, rolling stats, EMAs, and target creation"""

import pandas as pd
import numpy as np
from typing import List, Optional, Dict
from src.utils.logger import get_logger
from src.config.settings import Settings

logger = get_logger(__name__)


def create_lag_features(
    df: pd.DataFrame,
    sensor_cols: List[str],
    lag_windows: List[int] = Settings.LAG_WINDOWS,
    timestamp_col: str = Settings.TIMESTAMP_COL,
    machine_id_col: str = Settings.MACHINE_ID_COL
) -> pd.DataFrame:
    """
    Create lag features (t-1, t-2, etc.) per machine.
    Preserves time alignment and prevents cross-machine leakage.
    
    Args:
        df: Input DataFrame (must be sorted by time per machine)
        sensor_cols: List of sensor column names
        lag_windows: List of lag periods (e.g., [1, 2] for t-1, t-2)
        timestamp_col: Name of timestamp column
        machine_id_col: Name of machine_id column
        
    Returns:
        DataFrame with lag features added
    """
    logger.info(f"Creating lag features for windows: {lag_windows}")
    
    df_feat = df.copy()
    
    # Group by machine to prevent cross-machine leakage
    try:
        for machine_id in df_feat[machine_id_col].unique():
            machine_mask = df_feat[machine_id_col] == machine_id
            machine_indices = df_feat[machine_mask].index
            
            for col in sensor_cols:
                if col not in df_feat.columns:
                    continue
                
                for lag in lag_windows:
                    lag_col_name = f"{col}_lag_{lag}"
                    # Shift within machine group only
                    df_feat.loc[machine_indices, lag_col_name] = (
                        df_feat.loc[machine_indices, col].shift(lag)
                    )
    except KeyboardInterrupt:
        logger.warning("Lag feature creation interrupted by user")
        raise
    
    n_lag_features = len(sensor_cols) * len(lag_windows)
    logger.info(f"Created {n_lag_features} lag features")
    
    return df_feat


def create_rolling_features(
    df: pd.DataFrame,
    sensor_cols: List[str],
    rolling_windows: Dict[str, str] = Settings.ROLLING_WINDOWS,
    timestamp_col: str = Settings.TIMESTAMP_COL,
    machine_id_col: str = Settings.MACHINE_ID_COL
) -> pd.DataFrame:
    """
    Create rolling window statistics (mean, std) per machine.
    Time-aware rolling windows prevent leakage.
    
    Args:
        df: Input DataFrame (must be sorted by time per machine)
        sensor_cols: List of sensor column names
        rolling_windows: Dict of window names to pandas time offsets (e.g., {"1h": "1h"})
        timestamp_col: Name of timestamp column
        machine_id_col: Name of machine_id column
        
    Returns:
        DataFrame with rolling features added
    """
    logger.info(f"Creating rolling features for windows: {list(rolling_windows.keys())}")
    
    df_feat = df.copy()
    
    # Ensure timestamp is datetime
    if not pd.api.types.is_datetime64_any_dtype(df_feat[timestamp_col]):
        df_feat[timestamp_col] = pd.to_datetime(df_feat[timestamp_col])
    
    # Set timestamp as index temporarily for time-based rolling
    original_index = df_feat.index
    df_feat = df_feat.set_index(timestamp_col)
    
    # Create rolling features per machine
    try:
        for machine_id in df_feat[machine_id_col].unique():
            machine_mask = df_feat[machine_id_col] == machine_id
            machine_df = df_feat[machine_mask].copy()
            
            if len(machine_df) == 0:
                continue
            
            for col in sensor_cols:
                if col not in machine_df.columns:
                    continue
                
                for window_name, window_offset in rolling_windows.items():
                    # Time-based rolling window (backward-looking only)
                    rolling_mean = machine_df[col].rolling(
                        window=window_offset,
                        min_periods=1
                    ).mean()
                    rolling_std = machine_df[col].rolling(
                        window=window_offset,
                        min_periods=1
                    ).std()
                    
                    # Update main dataframe
                    mean_col_name = f"{col}_rolling_mean_{window_name}"
                    std_col_name = f"{col}_rolling_std_{window_name}"
                    
                    df_feat.loc[machine_mask, mean_col_name] = rolling_mean.values
                    df_feat.loc[machine_mask, std_col_name] = rolling_std.values
    except KeyboardInterrupt:
        logger.warning("Feature engineering interrupted by user")
        # Reset index before returning
        df_feat = df_feat.reset_index()
        raise
    
    # Reset index
    df_feat = df_feat.reset_index()
    
    n_rolling_features = len(sensor_cols) * len(rolling_windows) * 2  # mean + std
    logger.info(f"Created {n_rolling_features} rolling features")
    
    return df_feat


def create_ema_features(
    df: pd.DataFrame,
    sensor_cols: List[str],
    ema_alphas: List[float] = Settings.EMA_ALPHAS,
    timestamp_col: str = Settings.TIMESTAMP_COL,
    machine_id_col: str = Settings.MACHINE_ID_COL
) -> pd.DataFrame:
    """
    Create Exponential Moving Average features per machine.
    Backward-looking only to prevent leakage.
    
    Args:
        df: Input DataFrame (must be sorted by time per machine)
        sensor_cols: List of sensor column names
        ema_alphas: List of smoothing factors (0 < alpha <= 1)
        timestamp_col: Name of timestamp column
        machine_id_col: Name of machine_id column
        
    Returns:
        DataFrame with EMA features added
    """
    logger.info(f"Creating EMA features for alphas: {ema_alphas}")
    
    df_feat = df.copy()
    
    # Create EMA features per machine
    try:
        for machine_id in df_feat[machine_id_col].unique():
            machine_mask = df_feat[machine_id_col] == machine_id
            machine_indices = df_feat[machine_mask].index
            
            for col in sensor_cols:
                if col not in df_feat.columns:
                    continue
                
                for alpha in ema_alphas:
                    ema_col_name = f"{col}_ema_{alpha}"
                    # Calculate EMA within machine group only
                    machine_values = df_feat.loc[machine_indices, col].values
                    ema_values = pd.Series(machine_values).ewm(alpha=alpha, adjust=False).mean()
                    df_feat.loc[machine_indices, ema_col_name] = ema_values.values
    except KeyboardInterrupt:
        logger.warning("EMA feature creation interrupted by user")
        raise
    
    n_ema_features = len(sensor_cols) * len(ema_alphas)
    logger.info(f"Created {n_ema_features} EMA features")
    
    return df_feat


def create_target(
    df: pd.DataFrame,
    failure_col: str = "failure",
    prediction_horizon_hours: int = Settings.PREDICTION_HORIZON_HOURS,
    target_col: str = Settings.TARGET_COL,
    timestamp_col: str = Settings.TIMESTAMP_COL,
    machine_id_col: str = Settings.MACHINE_ID_COL
) -> pd.DataFrame:
    """
    Create binary target: failure within next N hours.
    Uses forward-looking shift to ensure NO leakage.
    
    Args:
        df: Input DataFrame (must be sorted by time per machine)
        failure_col: Name of failure indicator column (binary: 0/1 or True/False)
        prediction_horizon_hours: Hours ahead to predict failure
        target_col: Name of target column to create
        timestamp_col: Name of timestamp column
        machine_id_col: Name of machine_id column
        
    Returns:
        DataFrame with target column added
    """
    logger.info(f"Creating target: failure within {prediction_horizon_hours} hours")
    
    # Try to find failure column with multiple possible names
    if failure_col not in df.columns:
        # Try common alternative names
        possible_names = [
            'Machine failure', 'machine_failure', 'Machine Failure',
            'failure', 'Failure', 'FAILURE',
            'target', 'Target', 'TARGET',
            'label', 'Label', 'LABEL'
        ]
        
        found_col = None
        for name in possible_names:
            if name in df.columns:
                found_col = name
                logger.info(f"Failure column '{failure_col}' not found. Using '{name}' instead.")
                failure_col = name
                break
        
        if found_col is None:
            logger.error(f"Failure column '{failure_col}' not found and no alternatives found.")
            logger.error(f"Available columns: {list(df.columns)}")
            logger.warning("Creating dummy target with all zeros. Models will not train properly.")
            df[target_col] = 0
            return df
    
    df_feat = df.copy()
    
    # Ensure timestamp is datetime
    if not pd.api.types.is_datetime64_any_dtype(df_feat[timestamp_col]):
        df_feat[timestamp_col] = pd.to_datetime(df_feat[timestamp_col])
    
    # Initialize target column
    df_feat[target_col] = 0
    
    # Convert failure column to boolean if needed
    # Handle different formats: 0/1, True/False, 'Yes'/'No', etc.
    if df_feat[failure_col].dtype == 'object':
        # Try to convert string values
        unique_vals = [str(v).lower() for v in df_feat[failure_col].unique()[:10]]
        logger.info(f"Failure column unique values (first 10): {unique_vals}")
        
        # Try common boolean representations
        valid_bool_strings = {'0', '1', 'true', 'false', 'yes', 'no', 'y', 'n', 't', 'f'}
        if all(str(v).lower() in valid_bool_strings for v in df_feat[failure_col].unique()):
            df_feat[failure_col] = df_feat[failure_col].astype(str).str.lower()
            df_feat[failure_col] = df_feat[failure_col].isin(['1', 'true', 'yes', 'y', 't'])
        else:
            logger.warning(f"Failure column contains unexpected values. Attempting numeric conversion...")
            df_feat[failure_col] = pd.to_numeric(df_feat[failure_col], errors='coerce').fillna(0).astype(bool)
    elif df_feat[failure_col].dtype != bool:
        # Convert numeric to boolean (0 = False, non-zero = True)
        df_feat[failure_col] = (df_feat[failure_col] != 0).astype(bool)
    
    # Create target per machine to prevent cross-machine leakage
    try:
        # Check if we have multiple rows per machine
        rows_per_machine = df_feat.groupby(machine_id_col).size()
        single_row_machines = rows_per_machine[rows_per_machine == 1].index.tolist()
        multi_row_machines = rows_per_machine[rows_per_machine > 1].index.tolist()
        
        if len(single_row_machines) > 0:
            logger.info(f"Found {len(single_row_machines)} machines with only 1 row. "
                       f"Using simplified target creation for these machines.")
        
        for machine_id in df_feat[machine_id_col].unique():
            machine_mask = df_feat[machine_id_col] == machine_id
            machine_df = df_feat[machine_mask].copy()
            
            if len(machine_df) == 0:
                continue
            
            # Get failure timestamps for this machine
            failure_times = machine_df[machine_df[failure_col] == True][timestamp_col]
            
            if len(failure_times) == 0:
                continue
            
            # Handle single-row machines differently
            if machine_id in single_row_machines:
                # For single-row machines with failures, set target=1
                # Note: This creates a point-in-time target rather than future prediction
                # In production, you'd want multiple time-series rows per machine
                if len(failure_times) > 0:
                    df_feat.loc[machine_df.index[0], target_col] = 1
                    logger.debug(f"Single-row machine {machine_id}: failure detected, target set to 1")
            else:
                # Multi-row machines: use forward-looking approach
                # For each row, check if failure occurs within prediction horizon
                for idx in machine_df.index:
                    current_time = machine_df.loc[idx, timestamp_col]
                    
                    # Find failures within prediction horizon (forward-looking)
                    horizon_end = current_time + pd.Timedelta(hours=prediction_horizon_hours)
                    failures_in_horizon = failure_times[
                        (failure_times > current_time) & (failure_times <= horizon_end)
                    ]
                    
                    # Set target to 1 if any failure occurs in horizon
                    if len(failures_in_horizon) > 0:
                        df_feat.loc[idx, target_col] = 1
    except KeyboardInterrupt:
        logger.warning("Target creation interrupted by user")
        raise
    
    target_positive = df_feat[target_col].sum()
    target_rate = target_positive / len(df_feat) if len(df_feat) > 0 else 0
    
    logger.info(f"Target created: {target_positive} positive cases ({target_rate:.2%})")
    
    # Check if we have any failures in the original column
    original_failures = df_feat[failure_col].sum() if failure_col in df_feat.columns else 0
    if original_failures > 0 and target_positive == 0:
        logger.warning(
            f"Found {original_failures} failures in '{failure_col}' but created 0 positive targets. "
            f"This might indicate an issue with the prediction horizon or time alignment."
        )
    elif original_failures == 0:
        logger.error(
            f"No failures found in column '{failure_col}'. "
            f"Please verify the column name and that it contains failure events."
        )
    
    return df_feat


def feature_pipeline(
    df: pd.DataFrame,
    sensor_cols: Optional[List[str]] = None,
    failure_col: str = "failure",
    timestamp_col: str = Settings.TIMESTAMP_COL,
    machine_id_col: str = Settings.MACHINE_ID_COL,
    lag_windows: List[int] = Settings.LAG_WINDOWS,
    rolling_windows: Dict[str, str] = Settings.ROLLING_WINDOWS,
    ema_alphas: List[float] = Settings.EMA_ALPHAS,
    prediction_horizon_hours: int = Settings.PREDICTION_HORIZON_HOURS
) -> pd.DataFrame:
    """
    Complete feature engineering pipeline.
    
    Args:
        df: Input DataFrame (must be sorted by time per machine)
        sensor_cols: List of sensor column names (if None, auto-detect)
        failure_col: Name of failure indicator column
        timestamp_col: Name of timestamp column
        machine_id_col: Name of machine_id column
        lag_windows: List of lag periods
        rolling_windows: Dict of rolling window names to time offsets
        ema_alphas: List of EMA smoothing factors
        prediction_horizon_hours: Hours ahead to predict failure
        
    Returns:
        DataFrame with all features and target
    """
    logger.info("Starting feature engineering pipeline")
    
    initial_shape = df.shape
    
    # Auto-detect sensor columns if not provided
    if sensor_cols is None:
        exclude_cols = {
            timestamp_col, machine_id_col, failure_col,
            "UDI", "udi", "id", "ID", "Type", "TWF", "HDF", "PWF", "OSF", "RNF"
        }
        sensor_cols = [
            col for col in df.columns
            if col not in exclude_cols and pd.api.types.is_numeric_dtype(df[col])
        ]
    
    if not sensor_cols:
        raise ValueError("No sensor columns found for feature engineering")
    
    logger.info(f"Using {len(sensor_cols)} sensor columns for feature engineering")
    
    df_feat = df.copy()
    
    # Step 1: Create lag features
    df_feat = create_lag_features(
        df_feat,
        sensor_cols=sensor_cols,
        lag_windows=lag_windows,
        timestamp_col=timestamp_col,
        machine_id_col=machine_id_col
    )
    
    # Step 2: Create rolling features
    df_feat = create_rolling_features(
        df_feat,
        sensor_cols=sensor_cols,
        rolling_windows=rolling_windows,
        timestamp_col=timestamp_col,
        machine_id_col=machine_id_col
    )
    
    # Step 3: Create EMA features
    df_feat = create_ema_features(
        df_feat,
        sensor_cols=sensor_cols,
        ema_alphas=ema_alphas,
        timestamp_col=timestamp_col,
        machine_id_col=machine_id_col
    )
    
    # Step 4: Create target (must be last to avoid leakage)
    df_feat = create_target(
        df_feat,
        failure_col=failure_col,
        prediction_horizon_hours=prediction_horizon_hours,
        timestamp_col=timestamp_col,
        machine_id_col=machine_id_col
    )
    
    final_shape = df_feat.shape
    n_new_features = final_shape[1] - initial_shape[1]
    
    logger.info("Feature engineering pipeline completed")
    logger.info(f"Shape: {initial_shape} -> {final_shape}")
    logger.info(f"Created {n_new_features} new features")
    
    return df_feat

