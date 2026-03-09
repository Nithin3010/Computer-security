"""
explain_model.py
----------------

Explain the predictions of the trained XGBoost intrusion detection model
using SHAP (SHapley Additive Explanations).

Inputs
------
    models/xgboost_model.pkl
    processed_ml/X_test.parquet
    processed_ml/y_multiclass_test.parquet

Outputs
-------
    explainability_plots/
        shap_summary.png
        shap_feature_importance.png
        shap_dependence_<feature>.png  (top 3 features)
"""

import logging
import os

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

INPUT_DIR    = "processed_ml"
MODEL_PATH   = os.path.join("models", "xgboost_model.pkl")
PLOTS_DIR    = "explainability_plots"
SAMPLE_SIZE  = 10_000
RANDOM_STATE = 42
TOP_FEATURES = 20
TOP_DEPENDENCE = 3

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
# Loaders
# ---------------------------------------------------------------------------

def load_model(path: str = MODEL_PATH):
    log.info("Loading model from: %s", path)
    clf = joblib.load(path)
    log.info("Model loaded: %s", type(clf).__name__)
    return clf


def load_test_data() -> tuple[pd.DataFrame, pd.Series]:
    log.info("Loading test dataset from: %s", INPUT_DIR)

    X_test = pd.read_parquet(os.path.join(INPUT_DIR, "X_test.parquet"))
    y_df   = pd.read_parquet(os.path.join(INPUT_DIR, "y_multiclass_test.parquet"))
    y_test = y_df.iloc[:, 0] if isinstance(y_df, pd.DataFrame) else y_df

    log.info("Test dataset: %d rows, %d features", len(X_test), X_test.shape[1])
    return X_test, y_test


def sample_data(X: pd.DataFrame, n: int = SAMPLE_SIZE) -> pd.DataFrame:
    actual = min(n, len(X))
    log.info("Sampling %d rows for SHAP analysis (dataset size: %d)", actual, len(X))
    return X.sample(actual, random_state=RANDOM_STATE)


# ---------------------------------------------------------------------------
# SHAP computation
# ---------------------------------------------------------------------------

def compute_shap_values(clf, X_sample: pd.DataFrame):
    """Compute SHAP values using TreeExplainer on *X_sample*."""
    log.info("Initialising SHAP TreeExplainer …")
    explainer = shap.TreeExplainer(clf)

    log.info("Computing SHAP values for %d samples …", len(X_sample))
    shap_values = explainer.shap_values(X_sample)
    log.info("SHAP values computed.")
    return explainer, shap_values


def _mean_abs_shap(shap_values, n_features: int) -> np.ndarray:
    """
    Return mean |SHAP| per feature, averaged across all classes when
    shap_values is a list (multi-class) or used directly (2-D array).
    """
    if isinstance(shap_values, list):
        # Multi-class: list of (n_samples, n_features) arrays, one per class
        stacked = np.stack([np.abs(sv) for sv in shap_values], axis=0)
        return stacked.mean(axis=(0, 1))          # shape: (n_features,)
    else:
        # Single 2-D array
        return np.abs(shap_values).mean(axis=0)


def _shap_matrix(shap_values) -> np.ndarray:
    """
    Collapse multi-class SHAP values to a single (n_samples, n_features)
    matrix by taking the mean across classes.  Used for summary plots.
    """
    if isinstance(shap_values, list):
        return np.stack(shap_values, axis=0).mean(axis=0)
    return shap_values


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def plot_summary(shap_values, X_sample: pd.DataFrame, out_dir: str) -> None:
    """Beeswarm summary plot (global feature importance + effect direction)."""
    os.makedirs(out_dir, exist_ok=True)
    shap_matrix = _shap_matrix(shap_values)

    fig, ax = plt.subplots(figsize=(12, 8))
    shap.summary_plot(
        shap_matrix, X_sample,
        max_display=TOP_FEATURES,
        show=False,
        plot_size=None,
    )
    plt.title("SHAP Summary Plot — XGBoost Intrusion Detection", fontsize=13, pad=12)
    plt.tight_layout()
    path = os.path.join(out_dir, "shap_summary.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    log.info("Saved SHAP summary plot → %s", path)


def plot_bar_importance(shap_values, X_sample: pd.DataFrame, out_dir: str) -> None:
    """Bar chart of mean |SHAP| — top 20 features."""
    os.makedirs(out_dir, exist_ok=True)
    mean_abs = _mean_abs_shap(shap_values, X_sample.shape[1])

    indices    = np.argsort(mean_abs)[::-1][:TOP_FEATURES]
    top_names  = [X_sample.columns[i] for i in indices]
    top_scores = mean_abs[indices]

    fig, ax = plt.subplots(figsize=(12, 6))
    bars = ax.barh(top_names[::-1], top_scores[::-1], color="steelblue")
    ax.bar_label(bars, fmt="%.4f", padding=3, fontsize=8)
    ax.set_xlabel("Mean |SHAP value|", fontsize=11)
    ax.set_title(f"Top {TOP_FEATURES} Features — Mean SHAP Importance", fontsize=13)
    ax.margins(x=0.12)
    plt.tight_layout()

    path = os.path.join(out_dir, "shap_feature_importance.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved SHAP feature importance bar plot → %s", path)


def plot_dependence(
    shap_values,
    X_sample: pd.DataFrame,
    out_dir: str,
    top_n: int = TOP_DEPENDENCE,
) -> None:
    """
    SHAP dependence plots for the *top_n* most important features.

    For multi-class models the mean SHAP across all classes is used so a
    single interpretable plot is produced per feature.
    """
    os.makedirs(out_dir, exist_ok=True)
    mean_abs   = _mean_abs_shap(shap_values, X_sample.shape[1])
    shap_matrix = _shap_matrix(shap_values)

    top_indices  = np.argsort(mean_abs)[::-1][:top_n]
    top_features = [X_sample.columns[i] for i in top_indices]

    for feature in top_features:
        fig, ax = plt.subplots(figsize=(10, 5))
        shap.dependence_plot(
            feature,
            shap_matrix,
            X_sample,
            ax=ax,
            show=False,
        )
        ax.set_title(f"SHAP Dependence — {feature}", fontsize=13)
        plt.tight_layout()

        safe_name = feature.replace(" ", "_").replace("/", "_")
        path = os.path.join(out_dir, f"shap_dependence_{safe_name}.png")
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        log.info("Saved SHAP dependence plot → %s", path)


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run_explainability() -> None:
    """End-to-end SHAP explainability pipeline."""

    # 1. Load model ----------------------------------------------------------
    clf = load_model()

    # 2. Load test data ------------------------------------------------------
    X_test, _ = load_test_data()

    # 3. Sample --------------------------------------------------------------
    X_sample = sample_data(X_test)

    # 4. Compute SHAP values -------------------------------------------------
    explainer, shap_values = compute_shap_values(clf, X_sample)

    # 5. Summary plot --------------------------------------------------------
    plot_summary(shap_values, X_sample, PLOTS_DIR)

    # 6. Bar importance plot -------------------------------------------------
    plot_bar_importance(shap_values, X_sample, PLOTS_DIR)

    # 7. Dependence plots (top 3 features) -----------------------------------
    plot_dependence(shap_values, X_sample, PLOTS_DIR)

    # 8. Summary -------------------------------------------------------------
    print("\n" + "=" * 60)
    print("  SHAP Explainability — Summary")
    print("=" * 60)
    print(f"  Model          : {type(clf).__name__}")
    print(f"  Sample size    : {len(X_sample):,}")
    print(f"  Features       : {X_sample.shape[1]}")
    print(f"  Plots saved to : {os.path.abspath(PLOTS_DIR)}")
    print("=" * 60 + "\n")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    run_explainability()
