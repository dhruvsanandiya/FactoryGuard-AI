"""Week 2: Model Training - Run script for training and evaluating models"""

import sys
from pathlib import Path
import pandas as pd
from src.training.train import train_all_models
from src.config.settings import Settings
from src.utils.logger import get_logger

logger = get_logger(__name__)


def find_latest_datasets(artifacts_dir: Path):
    """
    Find the latest train and test parquet files.
    
    Args:
        artifacts_dir: Directory containing parquet files
        
    Returns:
        Tuple of (train_file, test_file) paths
    """
    # Find latest train file
    train_files = sorted(
        [f for f in artifacts_dir.glob("train_*.parquet")],
        reverse=True
    )
    
    # Find latest test file
    test_files = sorted(
        [f for f in artifacts_dir.glob("test_*.parquet")],
        reverse=True
    )
    
    if not train_files:
        raise FileNotFoundError(
            f"No train dataset found in {artifacts_dir}. "
            f"Please run Week 1 first: python run_week1.py"
        )
    
    if not test_files:
        raise FileNotFoundError(
            f"No test dataset found in {artifacts_dir}. "
            f"Please run Week 1 first: python run_week1.py"
        )
    
    train_file = train_files[0]
    test_file = test_files[0]
    
    logger.info(f"Using latest train file: {train_file.name}")
    logger.info(f"Using latest test file: {test_file.name}")
    
    return train_file, test_file


def main():
    """Main function to run Week 2 model training"""
    
    logger.info("=" * 80)
    logger.info("FactoryGuard AI - Week 2: Model Training")
    logger.info("=" * 80)
    
    # Check if custom train/test files are provided
    if len(sys.argv) >= 3:
        train_file = Path(sys.argv[1])
        test_file = Path(sys.argv[2])
        
        if not train_file.is_absolute():
            train_file = Settings.PROJECT_ROOT / train_file
        if not test_file.is_absolute():
            test_file = Settings.PROJECT_ROOT / test_file
        
        train_file = train_file.resolve()
        test_file = test_file.resolve()
        
        if not train_file.exists():
            logger.error(f"Train file not found: {train_file}")
            sys.exit(1)
        if not test_file.exists():
            logger.error(f"Test file not found: {test_file}")
            sys.exit(1)
    else:
        # Find latest datasets from Week 1
        logger.info("\n" + "=" * 80)
        logger.info("STEP 1: Loading Train/Test Datasets from Week 1")
        logger.info("=" * 80)
        
        try:
            train_file, test_file = find_latest_datasets(Settings.ARTIFACTS_DIR)
        except FileNotFoundError as e:
            logger.error(str(e))
            logger.info("\nTo run Week 1:")
            logger.info("  python run_week1.py [path/to/sensor_data.csv]")
            sys.exit(1)
    
    # Load datasets
    try:
        logger.info(f"Loading train dataset: {train_file}")
        train_df = pd.read_parquet(train_file)
        
        logger.info(f"Loading test dataset: {test_file}")
        test_df = pd.read_parquet(test_file)
        
        logger.info(f"Train set: {len(train_df)} rows, {len(train_df.columns)} columns")
        logger.info(f"Test set: {len(test_df)} rows, {len(test_df.columns)} columns")
        
    except Exception as e:
        logger.error(f"Failed to load datasets: {e}", exc_info=True)
        sys.exit(1)
    
    # Check for optimization flag
    optimize = True
    if len(sys.argv) > 3:
        optimize_flag = sys.argv[3].lower()
        if optimize_flag in ['--no-optimize', '--no-opt', '--fast']:
            optimize = False
            logger.info("Hyperparameter optimization disabled (fast mode)")
    
    # Run Week 2 training
    try:
        logger.info("\n" + "=" * 80)
        logger.info("STEP 2: Feature Preparation")
        logger.info("=" * 80)
        
        logger.info("\n" + "=" * 80)
        logger.info("STEP 3: Training Models")
        logger.info("=" * 80)
        logger.info("  - Baseline (Logistic Regression)")
        logger.info("  - Random Forest")
        logger.info("  - XGBoost (Primary)")
        
        # Train all models
        results = train_all_models(
            train_df=train_df,
            test_df=test_df,
            artifacts_dir=Settings.ARTIFACTS_DIR,
            optimize=optimize,
            target_col=Settings.TARGET_COL
        )
        
        logger.info("\n" + "=" * 80)
        logger.info("Week 2 Training Completed Successfully!")
        logger.info("=" * 80)
        
        # Print summary
        comparison = results['comparison']
        logger.info("\nFinal Model Comparison (sorted by Recall):")
        logger.info(f"\n{comparison.to_string(index=False)}")
        
        # Identify best model
        best_model = comparison.iloc[0]['Model']
        best_recall = comparison.iloc[0]['Recall']
        best_f1 = comparison.iloc[0]['F1-Score']
        
        logger.info(f"\nBest Model: {best_model}")
        logger.info(f"  Recall: {best_recall:.4f}")
        logger.info(f"  F1-Score: {best_f1:.4f}")
        
        logger.info(f"\nAll artifacts saved to: {results['artifacts_dir']}")
        logger.info("\nNext step: Run Week 3 explainability analysis")
        logger.info("  python run_week3.py")
        
    except Exception as e:
        logger.error(f"Week 2 training failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()

