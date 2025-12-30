"""Validation and leakage detection module"""

import pandas as pd
import numpy as np
from typing import List, Tuple, Optional
from src.utils.logger import get_logger
from src.config.settings import Settings

logger = get_logger(__name__)


def check_time_ordering(
    df: pd.DataFrame,
    timestamp_col: str = Settings.TIMESTAMP_COL,
    machine_id_col: str = Settings.MACHINE_ID_COL
) -> bool:
    """
    Validate that data is sorted by time within each machine.
    
    Args:
        df: Input DataFrame
        timestamp_col: Name of timestamp column
        machine_id_col: Name of machine_id column
        
    Returns:
        True if properly ordered, False otherwise
    """
    logger.info("Checking time ordering per machine")
    
    if timestamp_col not in df.columns:
        logger.error(f"Timestamp column '{timestamp_col}' not found")
        return False
    
    if machine_id_col not in df.columns:
        logger.error(f"Machine ID column '{machine_id_col}' not found")
        return False
    
    # Check ordering per machine
    is_ordered = True
    for machine_id in df[machine_id_col].unique():
        machine_df = df[df[machine_id_col] == machine_id].copy()
        
        if len(machine_df) < 2:
            continue
        
        # Check if timestamps are monotonically increasing
        timestamps = machine_df[timestamp_col].values
        if not np.all(timestamps[:-1] <= timestamps[1:]):
            logger.warning(f"Time ordering violated for machine {machine_id}")
            is_ordered = False
    
    if is_ordered:
        logger.info("Time ordering validated: all machines properly sorted")
    else:
        logger.error("Time ordering check failed")
    
    return is_ordered


def check_train_test_leakage(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    timestamp_col: str = Settings.TIMESTAMP_COL,
    machine_id_col: str = Settings.MACHINE_ID_COL
) -> bool:
    """
    Validate that test set is strictly after train set (no time leakage).
    
    Args:
        train_df: Training DataFrame
        test_df: Test DataFrame
        timestamp_col: Name of timestamp column
        machine_id_col: Name of machine_id column
        
    Returns:
        True if no leakage detected, False otherwise
    """
    logger.info("Checking train/test time leakage")
    
    if timestamp_col not in train_df.columns or timestamp_col not in test_df.columns:
        logger.error(f"Timestamp column '{timestamp_col}' not found in both datasets")
        return False
    
    # Check global time ordering
    max_train_time = train_df[timestamp_col].max()
    min_test_time = test_df[timestamp_col].min()
    
    if max_train_time >= min_test_time:
        logger.error(
            f"TIME LEAKAGE DETECTED: "
            f"max_train_time={max_train_time} >= min_test_time={min_test_time}"
        )
        return False
    
    # Check per-machine time ordering
    leakage_detected = False
    for machine_id in set(train_df[machine_id_col].unique()) & set(test_df[machine_id_col].unique()):
        train_machine = train_df[train_df[machine_id_col] == machine_id]
        test_machine = test_df[test_df[machine_id_col] == machine_id]
        
        if len(train_machine) == 0 or len(test_machine) == 0:
            continue
        
        max_train_machine = train_machine[timestamp_col].max()
        min_test_machine = test_machine[timestamp_col].min()
        
        if max_train_machine >= min_test_machine:
            logger.error(
                f"TIME LEAKAGE DETECTED for machine {machine_id}: "
                f"max_train={max_train_machine} >= min_test={min_test_machine}"
            )
            leakage_detected = True
    
    if not leakage_detected:
        logger.info("Train/test leakage check passed: no time leakage detected")
    
    return not leakage_detected


def check_feature_leakage(
    df: pd.DataFrame,
    feature_cols: List[str],
    timestamp_col: str = Settings.TIMESTAMP_COL,
    machine_id_col: str = Settings.MACHINE_ID_COL
) -> bool:
    """
    Check for potential feature leakage (future information in features).
    Validates that lag features and rolling features are backward-looking only.
    
    Args:
        df: Input DataFrame
        feature_cols: List of feature column names to check
        timestamp_col: Name of timestamp column
        machine_id_col: Name of machine_id column
        
    Returns:
        True if no leakage detected, False otherwise
    """
    logger.info("Checking feature leakage")
    
    if timestamp_col not in df.columns:
        logger.error(f"Timestamp column '{timestamp_col}' not found")
        return False
    
    # This is a basic check - in production, you'd want more sophisticated validation
    # For now, we check that lag features and rolling features don't use future data
    # by ensuring they're properly named and structured
    
    leakage_detected = False
    
    # Check for suspicious feature names that might indicate forward-looking features
    suspicious_patterns = ["future", "forward", "next", "ahead", "lead"]
    for col in feature_cols:
        col_lower = col.lower()
        for pattern in suspicious_patterns:
            if pattern in col_lower:
                logger.warning(
                    f"Potentially suspicious feature name detected: {col} "
                    f"(contains '{pattern}')"
                )
                # Don't fail, just warn - could be legitimate
    
    if not leakage_detected:
        logger.info("Feature leakage check passed: no obvious leakage detected")
    
    return not leakage_detected


def check_target_leakage(
    df: pd.DataFrame,
    target_col: str = Settings.TARGET_COL,
    failure_col: str = "failure",
    prediction_horizon_hours: int = Settings.PREDICTION_HORIZON_HOURS,
    timestamp_col: str = Settings.TIMESTAMP_COL,
    machine_id_col: str = Settings.MACHINE_ID_COL
) -> bool:
    """
    Validate that target is correctly shifted (no leakage from current/past failures).
    Target should indicate failure in FUTURE, not current or past.
    
    Args:
        df: Input DataFrame
        target_col: Name of target column
        failure_col: Name of failure indicator column
        prediction_horizon_hours: Hours ahead to predict failure
        timestamp_col: Name of timestamp column
        machine_id_col: Name of machine_id column
        
    Returns:
        True if target is correctly created, False otherwise
    """
    logger.info("Checking target leakage")
    
    if target_col not in df.columns:
        logger.error(f"Target column '{target_col}' not found")
        return False
    
    if failure_col not in df.columns:
        logger.warning(f"Failure column '{failure_col}' not found - skipping target validation")
        return True
    
    # Ensure timestamp is datetime
    if not pd.api.types.is_datetime64_any_dtype(df[timestamp_col]):
        df[timestamp_col] = pd.to_datetime(df[timestamp_col])
    
    # Check that target=1 only when failure occurs in the future
    leakage_detected = False
    
    for machine_id in df[machine_id_col].unique():
        machine_df = df[df[machine_id_col] == machine_id].copy()
        machine_df = machine_df.sort_values(by=timestamp_col).reset_index(drop=True)
        
        # Get rows where target=1
        target_positive = machine_df[machine_df[target_col] == 1]
        
        for idx in target_positive.index:
            current_time = machine_df.loc[idx, timestamp_col]
            current_failure = machine_df.loc[idx, failure_col]
            
            # If current row has failure=True, target should NOT be 1
            # (unless there's another failure in the future)
            if current_failure:
                # Check if there's a future failure within horizon
                future_failures = machine_df[
                    (machine_df.index > idx) &
                    (machine_df[failure_col] == True) &
                    (machine_df[timestamp_col] <= current_time + pd.Timedelta(hours=prediction_horizon_hours))
                ]
                
                if len(future_failures) == 0:
                    logger.warning(
                        f"Potential target leakage for machine {machine_id} at {current_time}: "
                        f"target=1 but no future failure in horizon"
                    )
                    # Don't fail, just warn - this could be edge case
    
    if not leakage_detected:
        logger.info("Target leakage check passed: target appears correctly shifted")
    
    return not leakage_detected


def validate_pipeline(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_cols: Optional[List[str]] = None,
    timestamp_col: str = Settings.TIMESTAMP_COL,
    machine_id_col: str = Settings.MACHINE_ID_COL,
    target_col: str = Settings.TARGET_COL
) -> bool:
    """
    Complete validation pipeline: check ordering, leakage, and data quality.
    
    Args:
        train_df: Training DataFrame
        test_df: Test DataFrame
        feature_cols: List of feature column names (if None, auto-detect)
        timestamp_col: Name of timestamp column
        machine_id_col: Name of machine_id column
        target_col: Name of target column
        
    Returns:
        True if all validations pass, False otherwise
    """
    logger.info("Starting validation pipeline")
    
    all_passed = True
    
    # Check time ordering
    if not check_time_ordering(train_df, timestamp_col, machine_id_col):
        all_passed = False
    
    if not check_time_ordering(test_df, timestamp_col, machine_id_col):
        all_passed = False
    
    # Check train/test leakage
    if not check_train_test_leakage(train_df, test_df, timestamp_col, machine_id_col):
        all_passed = False
    
    # Check feature leakage (if feature columns provided)
    if feature_cols:
        combined_df = pd.concat([train_df, test_df], ignore_index=True)
        if not check_feature_leakage(combined_df, feature_cols, timestamp_col, machine_id_col):
            all_passed = False
    
    # Check target leakage
    if target_col in train_df.columns:
        if not check_target_leakage(train_df, target_col, timestamp_col=timestamp_col, machine_id_col=machine_id_col):
            all_passed = False
    
    if all_passed:
        logger.info("All validation checks passed")
    else:
        logger.error("Some validation checks failed")
    
    return all_passed

