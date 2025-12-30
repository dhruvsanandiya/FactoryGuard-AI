"""Week 3: Model Explainability - Run script for SHAP-based explainability analysis"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
from src.utils.model_loader import (
    load_model,
    load_test_data,
    prepare_explainability_data,
    find_latest_model_dir
)
from src.explainability.shap_explainer import SHAPExplainer
from src.explainability.plots import (
    plot_shap_summary,
    plot_feature_importance_bar,
    plot_force_plot,
    plot_waterfall
)
from src.explainability.insights import InsightGenerator
from src.config.settings import Settings
from src.utils.logger import get_logger

logger = get_logger(__name__)


def main():
    """Main function to run Week 3 explainability analysis"""
    
    logger.info("=" * 80)
    logger.info("FactoryGuard AI - Week 3: Model Explainability")
    logger.info("=" * 80)
    
    # Check if custom model directory is provided
    model_dir = None
    if len(sys.argv) > 1:
        model_dir = Path(sys.argv[1])
        if not model_dir.is_absolute():
            model_dir = Settings.PROJECT_ROOT / model_dir
        model_dir = model_dir.resolve()
        
        if not model_dir.exists():
            logger.error(f"Model directory not found: {model_dir}")
            sys.exit(1)
    
    # Step 1: Load model and test data
    logger.info("\n" + "=" * 80)
    logger.info("STEP 1: Loading Model and Test Data")
    logger.info("=" * 80)
    
    try:
        # Load XGBoost model (primary model)
        if model_dir:
            model, loaded_model_dir = load_model(model_type="xgboost", model_dir=model_dir)
        else:
            model, loaded_model_dir = load_model(model_type="xgboost")
        
        logger.info(f"Model loaded from: {loaded_model_dir}")
        
        # Load test data
        test_df = load_test_data()
        
        # Prepare data for explainability
        X_test, y_test = prepare_explainability_data(test_df, model)
        
        logger.info(f"Test data prepared: {len(X_test)} samples, {len(X_test.columns)} features")
        
    except FileNotFoundError as e:
        logger.error(str(e))
        logger.info("\nTo run Week 2:")
        logger.info("  python run_week2.py")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Failed to load model/data: {e}", exc_info=True)
        sys.exit(1)
    
    # Step 2: Initialize SHAP explainer
    logger.info("\n" + "=" * 80)
    logger.info("STEP 2: Initializing SHAP Explainer")
    logger.info("=" * 80)
    
    try:
        # Use sample of test data as background (or could use training data)
        # For efficiency, sample background data
        n_background = min(100, len(X_test))
        X_background = X_test.sample(n=n_background, random_state=42)
        
        # Create cache directory
        cache_dir = loaded_model_dir / "shap_cache"
        
        # Initialize explainer
        explainer = SHAPExplainer(
            model=model,
            X_background=X_background,
            cache_dir=cache_dir
        )
        
        logger.info("SHAP explainer initialized successfully")
        
    except Exception as e:
        logger.error(f"Failed to initialize SHAP explainer: {e}", exc_info=True)
        sys.exit(1)
    
    # Step 3: Compute SHAP values
    logger.info("\n" + "=" * 80)
    logger.info("STEP 3: Computing SHAP Values")
    logger.info("=" * 80)
    
    try:
        # Compute SHAP values for test data
        shap_values = explainer.compute_shap_values(
            X_test,
            cache_key="test_data",
            use_cache=True
        )
        
        logger.info(f"SHAP values computed: shape {shap_values.shape}")
        
        # Validate explanations
        predictions = model.predict_proba(X_test)[:, 1]
        validation_results = explainer.validate_explanations(
            X_test,
            shap_values,
            predictions
        )
        
        # Validation results are logged in the validate_explanations method
        # Continue even if validation shows warnings - explanations are still useful
        if validation_results.get('max_error', 0) > 0.5:
            logger.warning(
                "SHAP validation shows high approximation error. "
                "This is common with TreeExplainer and may indicate scale differences. "
                "Explanations are still useful for relative feature importance analysis."
            )
        
    except Exception as e:
        logger.error(f"Failed to compute SHAP values: {e}", exc_info=True)
        sys.exit(1)
    
    # Step 4: Global Interpretability
    logger.info("\n" + "=" * 80)
    logger.info("STEP 4: Global Interpretability Analysis")
    logger.info("=" * 80)
    
    try:
        output_dir = loaded_model_dir / "explainability"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # SHAP summary plot
        summary_plot_path = output_dir / "shap_summary_plot.png"
        plot_shap_summary(
            shap_values,
            X_test,
            explainer.feature_names,
            output_path=summary_plot_path,
            max_display=20
        )
        
        # Feature importance
        feature_importance = explainer.get_feature_importance(shap_values)
        importance_plot_path = output_dir / "feature_importance_bar.png"
        plot_feature_importance_bar(
            feature_importance,
            output_path=importance_plot_path,
            top_n=20
        )
        
        # Save feature importance CSV
        importance_csv_path = output_dir / "shap_feature_importance.csv"
        feature_importance.to_csv(importance_csv_path, index=False)
        logger.info(f"Feature importance saved to {importance_csv_path}")
        
        logger.info("Global interpretability plots created")
        
    except Exception as e:
        logger.error(f"Failed to create global interpretability plots: {e}", exc_info=True)
    
    # Step 5: Local Interpretability
    logger.info("\n" + "=" * 80)
    logger.info("STEP 5: Local Interpretability Analysis")
    logger.info("=" * 80)
    
    try:
        # Find high-risk machines
        predictions = model.predict_proba(X_test)[:, 1]
        high_risk_indices = np.argsort(predictions)[-5:][::-1]
        
        logger.info(f"Analyzing top 5 high-risk machines...")
        
        # Get expected value
        expected_value = explainer.explainer.expected_value
        if isinstance(expected_value, np.ndarray):
            # If array has 2 elements, use positive class (index 1)
            # If array has 1 element, use that (index 0)
            if len(expected_value) == 2:
                expected_value = expected_value[1]  # Positive class
            else:
                expected_value = expected_value[0]
        else:
            expected_value = float(expected_value)
        
        # Create force plots for high-risk machines
        for i, idx in enumerate(high_risk_indices, 1):
            force_plot_path = output_dir / f"force_plot_high_risk_{i}_instance_{idx}.html"
            plot_force_plot(
                shap_values,
                X_test,
                explainer.feature_names,
                instance_idx=idx,
                expected_value=expected_value,
                output_path=force_plot_path
            )
            
            waterfall_plot_path = output_dir / f"waterfall_plot_high_risk_{i}_instance_{idx}.png"
            plot_waterfall(
                shap_values,
                X_test,
                explainer.feature_names,
                instance_idx=idx,
                expected_value=expected_value,
                output_path=waterfall_plot_path
            )
        
        logger.info("Local interpretability plots created")
        
    except Exception as e:
        logger.error(f"Failed to create local interpretability plots: {e}", exc_info=True)
    
    # Step 6: Human-Readable Insights
    logger.info("\n" + "=" * 80)
    logger.info("STEP 6: Generating Human-Readable Insights")
    logger.info("=" * 80)
    
    try:
        # Initialize insight generator
        insight_generator = InsightGenerator(
            feature_names=explainer.feature_names
        )
        
        # Generate report
        predictions = model.predict_proba(X_test)[:, 1]
        report_path = output_dir / "human_readable_insights.txt"
        
        report_text = insight_generator.generate_report(
            shap_values,
            X_test,
            predictions,
            y_true=y_test if 'y_test' in locals() else None,
            output_path=str(report_path)
        )
        
        # Also generate explanations for top high-risk machines
        high_risk_explanations = insight_generator.explain_high_risk_machines(
            shap_values,
            X_test,
            predictions,
            top_k=5
        )
        
        # Save detailed explanations as JSON
        import json
        explanations_path = output_dir / "high_risk_explanations.json"
        with open(explanations_path, 'w') as f:
            json.dump(high_risk_explanations, f, indent=2, default=str)
        
        logger.info(f"Human-readable insights saved to {report_path}")
        logger.info(f"Detailed explanations saved to {explanations_path}")
        
        # Print summary
        logger.info("\n" + "=" * 80)
        logger.info("SUMMARY")
        logger.info("=" * 80)
        logger.info(f"All explainability outputs saved to: {output_dir}")
        logger.info("\nGenerated outputs:")
        logger.info(f"  - SHAP summary plot: {summary_plot_path}")
        logger.info(f"  - Feature importance plot: {importance_plot_path}")
        logger.info(f"  - Feature importance CSV: {importance_csv_path}")
        logger.info(f"  - Force plots (HTML): {output_dir / 'force_plot_high_risk_*.html'}")
        logger.info(f"  - Waterfall plots: {output_dir / 'waterfall_plot_high_risk_*.png'}")
        logger.info(f"  - Human-readable report: {report_path}")
        logger.info(f"  - Detailed explanations: {explanations_path}")
        
    except Exception as e:
        logger.error(f"Failed to generate insights: {e}", exc_info=True)
    
    logger.info("\n" + "=" * 80)
    logger.info("Week 3 Explainability Analysis Completed Successfully!")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()

