"""
explain_with_shap.py
--------------------

Use SHAP (SHapley Additive exPlanations) to explain model predictions.

SHAP provides:
- Global feature importance (which features matter most overall)
- Local explanations (why a specific prediction was made)
- Interaction effects between features

Outputs:
    explainability_plots/
        shap_summary.png           - Global feature importance
        shap_beeswarm.png          - Feature impact distribution
        shap_bar.png               - Mean absolute SHAP values
        shap_waterfall_*.png       - Individual prediction explanations
        shap_force_*.png           - Force plots for specific predictions
"""

import os
import logging
import json
import joblib

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

MODEL_PATH = os.path.join("models", "xgboost_balanced_model.pkl")
SCALER_PATH = os.path.join("processed_ml", "scaler.pkl")
MAPPING_PATH = os.path.join("processed_ml", "attack_label_mapping.json")
TEST_DATA_PATH = os.path.join("processed_ml", "X_test.parquet")
TEST_LABELS_PATH = os.path.join("processed_ml", "y_multiclass_test.parquet")

OUTPUT_DIR = "explainability_plots"

# Number of samples to use for SHAP analysis (too many = slow)
SHAP_SAMPLE_SIZE = 1000
BACKGROUND_SAMPLE_SIZE = 100  # For TreeExplainer background

# Number of specific predictions to explain in detail
NUM_EXAMPLES = 5

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Load Model and Data
# ---------------------------------------------------------------------------

def load_artifacts():
    """Load model, test data, and label mapping."""
    log.info("Loading trained model from: %s", MODEL_PATH)
    model = joblib.load(MODEL_PATH)
    
    log.info("Loading test data from: %s", TEST_DATA_PATH)
    X_test = pd.read_parquet(TEST_DATA_PATH)
    
    y_test = pd.read_parquet(TEST_LABELS_PATH)
    y_test = y_test.iloc[:, 0].values if isinstance(y_test, pd.DataFrame) else y_test.values
    
    log.info("Loading label mapping from: %s", MAPPING_PATH)
    with open(MAPPING_PATH, 'r') as f:
        name_to_id = json.load(f)
    id_to_name = {int(v): k for k, v in name_to_id.items()}
    
    log.info("Loaded %d test samples with %d features", len(X_test), X_test.shape[1])
    
    return model, X_test, y_test, id_to_name


def sample_data(X_test, y_test, n_samples=SHAP_SAMPLE_SIZE):
    """Sample data for SHAP analysis."""
    if len(X_test) > n_samples:
        log.info("Sampling %d samples for SHAP analysis...", n_samples)
        indices = np.random.choice(len(X_test), size=n_samples, replace=False)
        X_sample = X_test.iloc[indices].reset_index(drop=True)
        y_sample = y_test[indices]
        return X_sample, y_sample
    return X_test, y_test


# ---------------------------------------------------------------------------
# SHAP Analysis
# ---------------------------------------------------------------------------

def create_shap_explainer(model, X_background):
    """Create SHAP TreeExplainer for XGBoost."""
    log.info("Creating SHAP TreeExplainer with %d background samples...", len(X_background))
    
    # Use a small background dataset for faster computation
    if len(X_background) > BACKGROUND_SAMPLE_SIZE:
        bg_indices = np.random.choice(len(X_background), size=BACKGROUND_SAMPLE_SIZE, replace=False)
        X_background = X_background.iloc[bg_indices]
    
    explainer = shap.TreeExplainer(model, data=X_background)
    log.info("SHAP explainer created successfully")
    
    return explainer


def compute_shap_values(explainer, X_sample):
    """Compute SHAP values for samples."""
    log.info("Computing SHAP values for %d samples...", len(X_sample))
    shap_values = explainer.shap_values(X_sample)
    log.info("SHAP values computed successfully")
    
    return shap_values


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

def plot_shap_summary(shap_values, X_sample, output_dir=OUTPUT_DIR):
    """Create SHAP summary plot (beeswarm)."""
    os.makedirs(output_dir, exist_ok=True)
    
    log.info("Creating SHAP summary plot (beeswarm)...")
    plt.figure(figsize=(12, 8))
    shap.summary_plot(shap_values, X_sample, show=False, max_display=20)
    plt.tight_layout()
    
    output_path = os.path.join(output_dir, "shap_beeswarm.png")
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    log.info("Saved SHAP beeswarm plot: %s", output_path)


def plot_shap_bar(shap_values, X_sample, output_dir=OUTPUT_DIR):
    """Create SHAP bar plot (mean absolute importance)."""
    log.info("Creating SHAP bar plot...")
    plt.figure(figsize=(12, 8))
    shap.summary_plot(shap_values, X_sample, plot_type="bar", show=False, max_display=20)
    plt.tight_layout()
    
    output_path = os.path.join(output_dir, "shap_bar.png")
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    log.info("Saved SHAP bar plot: %s", output_path)


def plot_waterfall_examples(explainer, X_sample, y_sample, y_pred, id_to_name, 
                            n_examples=NUM_EXAMPLES, output_dir=OUTPUT_DIR):
    """Create waterfall plots for individual predictions."""
    log.info("Creating waterfall plots for %d example predictions...", n_examples)
    
    # Get SHAP values for examples
    shap_values = explainer(X_sample.iloc[:n_examples])
    
    for i in range(min(n_examples, len(X_sample))):
        true_label = id_to_name.get(y_sample[i], f"Class_{y_sample[i]}")
        pred_label = id_to_name.get(y_pred[i], f"Class_{y_pred[i]}")
        
        plt.figure(figsize=(10, 6))
        shap.plots.waterfall(shap_values[i][:, y_pred[i]], show=False, max_display=15)
        plt.title(f"Prediction #{i+1}: {pred_label} (True: {true_label})", fontsize=12, pad=20)
        plt.tight_layout()
        
        output_path = os.path.join(output_dir, f"shap_waterfall_{i+1}.png")
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        log.info("Saved waterfall plot %d/%d: %s", i+1, n_examples, output_path)


def plot_dependence_plots(shap_values, X_sample, top_features, output_dir=OUTPUT_DIR):
    """Create SHAP dependence plots for top features."""
    log.info("Creating dependence plots for top %d features...", len(top_features))
    
    for i, feature in enumerate(top_features[:5]):  # Top 5 features
        plt.figure(figsize=(10, 6))
        shap.dependence_plot(feature, shap_values, X_sample, show=False)
        plt.tight_layout()
        
        # Sanitize feature name for filename
        safe_feature_name = feature.replace('/', '_').replace('\\', '_').replace(' ', '_')
        output_path = os.path.join(output_dir, f"shap_dependence_{safe_feature_name}.png")
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        log.info("Saved dependence plot for '%s': %s", feature, output_path)


def get_top_features(shap_values, X_sample, n_features=20):
    """Get top N most important features by mean absolute SHAP value."""
    # Calculate mean absolute SHAP values
    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    
    # Get feature names and sort by importance
    feature_importance = pd.DataFrame({
        'feature': X_sample.columns,
        'importance': mean_abs_shap
    }).sort_values('importance', ascending=False)
    
    top_features = feature_importance.head(n_features)['feature'].tolist()
    
    log.info("Top 10 most important features:")
    for i, (idx, row) in enumerate(feature_importance.head(10).iterrows(), 1):
        log.info("  %2d. %-40s  %.4f", i, row['feature'], row['importance'])
    
    return top_features, feature_importance


def save_feature_importance(feature_importance, output_dir=OUTPUT_DIR):
    """Save feature importance to CSV."""
    output_path = os.path.join(output_dir, "shap_feature_importance.csv")
    feature_importance.to_csv(output_path, index=False)
    log.info("Saved feature importance to: %s", output_path)


# ---------------------------------------------------------------------------
# Attack Class Analysis
# ---------------------------------------------------------------------------

def analyze_attack_classes(model, explainer, X_sample, y_sample, id_to_name, output_dir=OUTPUT_DIR):
    """Analyze SHAP values for each attack class."""
    log.info("Analyzing feature importance per attack class...")
    
    predictions = model.predict(X_sample)
    shap_values = explainer.shap_values(X_sample)
    
    class_analysis = {}
    
    for class_id, class_name in id_to_name.items():
        # Get samples of this class
        mask = (y_sample == class_id)
        if not mask.any():
            continue
        
        n_samples = mask.sum()
        class_shap = shap_values[mask]
        
        # Mean absolute SHAP values for this class
        mean_abs_shap = np.abs(class_shap).mean(axis=0)
        
        # Top 10 features for this class
        top_indices = np.argsort(mean_abs_shap)[-10:][::-1]
        top_features = [(X_sample.columns[i], mean_abs_shap[i]) for i in top_indices]
        
        class_analysis[class_name] = {
            'n_samples': int(n_samples),
            'top_features': top_features
        }
        
        log.info("  %s (%d samples): Top feature = %s (%.4f)", 
                 class_name, n_samples, top_features[0][0], top_features[0][1])
    
    # Save analysis
    output_path = os.path.join(output_dir, "shap_class_analysis.txt")
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("="*80 + "\n")
        f.write("  SHAP Analysis by Attack Class\n")
        f.write("="*80 + "\n\n")
        
        for class_name, analysis in sorted(class_analysis.items()):
            f.write(f"\n{class_name} ({analysis['n_samples']} samples)\n")
            f.write("-" * 80 + "\n")
            f.write("Top 10 Most Important Features:\n")
            for i, (feature, importance) in enumerate(analysis['top_features'], 1):
                f.write(f"  {i:2d}. {feature:40s}  {importance:8.4f}\n")
    
    log.info("Saved class analysis to: %s", output_path)


# ---------------------------------------------------------------------------
# Main Pipeline
# ---------------------------------------------------------------------------

def main():
    """Main SHAP analysis pipeline."""
    log.info("="*80)
    log.info("  SHAP Model Explainability Analysis")
    log.info("="*80)
    
    # 1. Load model and data
    model, X_test, y_test, id_to_name = load_artifacts()
    
    # 2. Sample data
    X_sample, y_sample = sample_data(X_test, y_test)
    
    # 3. Get predictions
    log.info("Generating predictions...")
    y_pred = model.predict(X_sample)
    
    # 4. Create SHAP explainer
    explainer = create_shap_explainer(model, X_sample)
    
    # 5. Compute SHAP values
    shap_values = compute_shap_values(explainer, X_sample)
    
    # 6. Get top features
    top_features, feature_importance = get_top_features(shap_values, X_sample)
    save_feature_importance(feature_importance)
    
    # 7. Create visualizations
    plot_shap_summary(shap_values, X_sample)
    plot_shap_bar(shap_values, X_sample)
    plot_waterfall_examples(explainer, X_sample, y_sample, y_pred, id_to_name)
    plot_dependence_plots(shap_values, X_sample, top_features)
    
    # 8. Analyze by attack class
    analyze_attack_classes(model, explainer, X_sample, y_sample, id_to_name)
    
    # 9. Summary
    print("\n" + "="*80)
    print("  SHAP ANALYSIS COMPLETE")
    print("="*80)
    print(f"  Analyzed {len(X_sample)} samples with {X_sample.shape[1]} features")
    print(f"  Output directory: {OUTPUT_DIR}/")
    print(f"  Generated plots:")
    print(f"    - shap_beeswarm.png        (Feature importance distribution)")
    print(f"    - shap_bar.png             (Mean absolute SHAP values)")
    print(f"    - shap_waterfall_*.png     (Individual prediction explanations)")
    print(f"    - shap_dependence_*.png    (Feature interactions)")
    print(f"    - shap_feature_importance.csv")
    print(f"    - shap_class_analysis.txt")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
