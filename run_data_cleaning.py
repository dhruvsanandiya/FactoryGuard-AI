"""Week 1: Data Pipeline - Run script for data ingestion, cleaning, feature engineering, and validation"""

import sys
from pathlib import Path
from src.data.pipeline import run_pipeline
from src.config.settings import Settings
from src.utils.logger import get_logger

logger = get_logger(__name__)


def main():
    """Main function to run Week 1 data pipeline"""
    
    logger.info("=" * 80)
    logger.info("FactoryGuard AI - Week 1: Data Pipeline")
    logger.info("=" * 80)
    
    # Get input file
    if len(sys.argv) > 1:
        input_file = Path(sys.argv[1])
        # Resolve relative paths relative to project root
        if not input_file.is_absolute():
            input_file = Settings.PROJECT_ROOT / input_file
        input_file = input_file.resolve()
    else:
        # Default: look for data in raw directory
        input_file = Settings.RAW_DATA_DIR / "sensor_data.csv"
    
    if not input_file.exists():
        # Check for common typos and auto-correct if the corrected path exists
        corrected_path = None
        if "row" in str(input_file).lower() and "raw" not in str(input_file).lower():
            corrected_path = Path(str(input_file).replace("row", "raw"))
            if corrected_path.exists():
                logger.info(f"Detected typo in path: {input_file}")
                logger.info(f"Auto-correcting to: {corrected_path}")
                input_file = corrected_path
            else:
                logger.error(f"Input file not found: {input_file}")
                logger.info(f"Did you mean: {corrected_path}?")
                logger.info(f"Please provide a CSV file path or place data in: {Settings.RAW_DATA_DIR}")
                sys.exit(1)
        else:
            logger.error(f"Input file not found: {input_file}")
            logger.info(f"Please provide a CSV file path or place data in: {Settings.RAW_DATA_DIR}")
            sys.exit(1)
    
    logger.info(f"Processing file: {input_file}")
    
    # Run Week 1 pipeline
    try:
        logger.info("\n" + "=" * 80)
        logger.info("STEP 1: Data Ingestion")
        logger.info("=" * 80)
        
        logger.info("\n" + "=" * 80)
        logger.info("STEP 2: Data Cleaning")
        logger.info("=" * 80)
        
        logger.info("\n" + "=" * 80)
        logger.info("STEP 3: Feature Engineering")
        logger.info("=" * 80)
        
        logger.info("\n" + "=" * 80)
        logger.info("STEP 4: Train/Test Split")
        logger.info("=" * 80)
        
        logger.info("\n" + "=" * 80)
        logger.info("STEP 5: Validation")
        logger.info("=" * 80)
        
        train_df, test_df = run_pipeline(
            input_file=input_file,
            output_dir=Settings.ARTIFACTS_DIR,
            sensor_cols=None,  # Auto-detect
            failure_col="Machine failure",  # Adjust if your column name differs
            test_size=0.2,
            validate=True
        )
        
        logger.info("\n" + "=" * 80)
        logger.info("Week 1 Pipeline Completed Successfully!")
        logger.info("=" * 80)
        logger.info(f"Train set: {len(train_df)} rows, {len(train_df.columns)} columns")
        logger.info(f"Test set: {len(test_df)} rows, {len(test_df.columns)} columns")
        logger.info(f"\nOutput files saved to: {Settings.ARTIFACTS_DIR}")
        logger.info("  - train_*.parquet")
        logger.info("  - test_*.parquet")
        logger.info("\nNext step: Run Week 2 model training")
        logger.info("  python run_week2.py")
        
    except Exception as e:
        logger.error(f"Week 1 pipeline failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()

