"""Main data pipeline orchestrating all processing steps"""

from pathlib import Path
from typing import Optional, List, Tuple
import pandas as pd
from src.data.ingest import ingest_pipeline
from src.data.clean import clean_pipeline
from src.data.features import feature_pipeline
from src.data.split import split_pipeline
from src.data.validate import validate_pipeline
from src.utils.logger import get_logger
from src.config.settings import Settings

logger = get_logger(__name__)


def run_pipeline(
    input_file: Path,
    output_dir: Optional[Path] = None,
    sensor_cols: Optional[List[str]] = None,
    failure_col: str = "failure",
    timestamp_col: str = Settings.TIMESTAMP_COL,
    machine_id_col: str = Settings.MACHINE_ID_COL,
    test_size: float = Settings.TEST_SIZE,
    version: Optional[str] = None,
    validate: bool = True
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Run complete data pipeline from raw CSV to train/test datasets.
    
    Args:
        input_file: Path to raw CSV file
        output_dir: Output directory for artifacts (default: Settings.ARTIFACTS_DIR)
        sensor_cols: List of sensor column names (if None, auto-detect)
        failure_col: Name of failure indicator column
        timestamp_col: Name of timestamp column
        machine_id_col: Name of machine_id column
        test_size: Proportion of data for test set
        version: Version string for artifacts (if None, uses timestamp)
        validate: Whether to run validation checks
        
    Returns:
        Tuple of (train_df, test_df)
    """
    logger.info("=" * 80)
    logger.info("Starting FactoryGuard AI Data Pipeline")
    logger.info("=" * 80)
    
    if output_dir is None:
        output_dir = Settings.ARTIFACTS_DIR
    
    # Step 1: Data Ingestion
    logger.info("\n[STEP 1] Data Ingestion")
    logger.info("-" * 80)
    df = ingest_pipeline(
        input_file,
        timestamp_col=timestamp_col,
        machine_id_col=machine_id_col
    )
    
    # Step 2: Data Cleaning
    logger.info("\n[STEP 2] Data Cleaning")
    logger.info("-" * 80)
    df = clean_pipeline(
        df,
        sensor_cols=sensor_cols,
        timestamp_col=timestamp_col,
        machine_id_col=machine_id_col
    )
    
    # Save cleaned data
    cleaned_data_path = Settings.OUTPUT_DIR / "cleaned_data.csv"
    Settings.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(cleaned_data_path, index=False)
    logger.info(f"Cleaned data saved to: {cleaned_data_path}")
    
    # Step 3: Feature Engineering
    logger.info("\n[STEP 3] Feature Engineering")
    logger.info("-" * 80)
    df = feature_pipeline(
        df,
        sensor_cols=sensor_cols,
        failure_col=failure_col,
        timestamp_col=timestamp_col,
        machine_id_col=machine_id_col
    )
    
    # Step 4: Train/Test Split
    logger.info("\n[STEP 4] Train/Test Split")
    logger.info("-" * 80)
    train_df, test_df, train_path, test_path = split_pipeline(
        df,
        test_size=test_size,
        timestamp_col=timestamp_col,
        machine_id_col=machine_id_col,
        output_dir=output_dir,
        version=version,
        save=True
    )
    
    # Step 5: Validation
    if validate:
        logger.info("\n[STEP 5] Validation")
        logger.info("-" * 80)
        
        # Auto-detect feature columns
        if sensor_cols is None:
            exclude_cols = {timestamp_col, machine_id_col, failure_col, Settings.TARGET_COL}
            feature_cols = [
                col for col in train_df.columns
                if col not in exclude_cols
            ]
        else:
            # Include derived features
            feature_cols = [
                col for col in train_df.columns
                if any(sensor in col for sensor in sensor_cols) or col in sensor_cols
            ]
        
        validation_passed = validate_pipeline(
            train_df,
            test_df,
            feature_cols=feature_cols,
            timestamp_col=timestamp_col,
            machine_id_col=machine_id_col,
            target_col=Settings.TARGET_COL
        )
        
        if not validation_passed:
            logger.warning("Validation checks failed - review logs for details")
        else:
            logger.info("All validation checks passed")
    
    # Final summary
    logger.info("\n" + "=" * 80)
    logger.info("Pipeline Completed Successfully")
    logger.info("=" * 80)
    logger.info(f"Train set: {len(train_df)} rows, {len(train_df.columns)} columns")
    logger.info(f"Test set: {len(test_df)} rows, {len(test_df.columns)} columns")
    if train_path:
        logger.info(f"Artifacts saved to: {output_dir}")
    
    return train_df, test_df

