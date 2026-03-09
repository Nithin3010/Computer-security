"""
evaluate_model.py
-----------------

Evaluate a trained intrusion detection model on the test dataset.

The trained model was produced by train_random_forest.py and saved to:

    models/random_forest_model.pkl

The ML-ready test data was produced by prepare_ml_dataset.py and stored in:

    processed_ml/
"""

import json
import logging
import os

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

INPUT_DIR   = "processed_ml"
MODEL_PATH  = os.path.join("models", "random_forest_model.pkl")
MAPPING_PATH = os.path.join(INPUT_DIR, "attack_label_mapping.json")
PLOTS_DIR   = "model_plots"
RESULTS_DIR = "results"
PREDICTIONS_PATH = os.path.join(RESULTS_DIR, "predictions.parquet")

TOP_N_FEATURES = 20

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
    """Load a trained model from *path*."""
    log.info("Loading model from: %s", path)
    clf = joblib.load(path)
    log.info("Model loaded successfully: %s", type(clf).__name__)
    return clf


def load_test_data() -> tuple[pd.DataFrame, pd.Series]:
    """Load X_test and y_multiclass_test from processed_ml/."""
    log.info("Loading test dataset from: %s", INPUT_DIR)

    X_test = pd.read_parquet(os.path.join(INPUT_DIR, "X_test.parquet"))
    y_df   = pd.read_parquet(os.path.join(INPUT_DIR, "y_multiclass_test.parquet"))
    y_test = y_df.iloc[:, 0] if isinstance(y_df, pd.DataFrame) else y_df

    log.info("Dataset size — test: %d rows, %d features", len(X_test), X_test.shape[1])
    return X_test, y_test


def load_label_mapping(path: str = MAPPING_PATH) -> dict:
    """Return {class_name: class_id} from the saved JSON mapping."""
    if not os.path.exists(path):
        log.warning("Label mapping not found at %s — using integer labels.", path)
        return {}
    with open(path, encoding="utf-8") as fh:
        mapping = json.load(fh)
    log.info("Loaded label mapping: %d classes", len(mapping))
    return mapping


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def compute_metrics(
    y_true: pd.Series,
    y_pred: np.ndarray,
    class_names: list[str] | None,
) -> dict:
    """Compute and log classification metrics, print classification report."""
    acc  = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average="macro", zero_division=0)
    rec  = recall_score(y_true, y_pred, average="macro", zero_division=0)
    f1   = f1_score(y_true, y_pred, average="macro", zero_division=0)

    log.info("Accuracy  : %.4f", acc)
    log.info("Precision : %.4f  (macro)", prec)
    log.info("Recall    : %.4f  (macro)", rec)
    log.info("F1-score  : %.4f  (macro)", f1)

    n_classes = len(np.unique(y_true))
    log.info("Number of classes: %d", n_classes)

    report = classification_report(
        y_true, y_pred,
        target_names=class_names,
        zero_division=0,
    )
    print("\n" + "=" * 60)
    print("  Classification Report — Test Set")
    print("=" * 60)
    print(report)

    return dict(accuracy=acc, precision=prec, recall=rec, f1=f1)


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def _heatmap(
    data: np.ndarray,
    labels: list[str],
    title: str,
    fmt: str,
    ax: plt.Axes,
) -> None:
    sns.heatmap(
        data,
        annot=True,
        fmt=fmt,
        cmap="Blues",
        xticklabels=labels,
        yticklabels=labels,
        ax=ax,
        linewidths=0.4,
    )
    ax.set_title(title, fontsize=13)
    ax.set_xlabel("Predicted label", fontsize=11)
    ax.set_ylabel("True label", fontsize=11)
    ax.tick_params(axis="x", rotation=45)
    ax.tick_params(axis="y", rotation=0)


def plot_confusion_matrices(
    y_true: pd.Series,
    y_pred: np.ndarray,
    class_names: list[str] | None,
    out_dir: str = PLOTS_DIR,
) -> None:
    """Save raw and normalized confusion matrix plots separately."""
    os.makedirs(out_dir, exist_ok=True)
    labels = class_names if class_names else [str(i) for i in sorted(np.unique(y_true))]

    cm      = confusion_matrix(y_true, y_pred)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    # Raw confusion matrix
    fig, ax = plt.subplots(figsize=(14, 10))
    _heatmap(cm, labels, "Confusion Matrix (counts) — Test Set", "d", ax)
    plt.tight_layout()
    path = os.path.join(out_dir, "confusion_matrix.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved confusion matrix → %s", path)

    # Normalized confusion matrix
    fig, ax = plt.subplots(figsize=(14, 10))
    _heatmap(cm_norm, labels, "Confusion Matrix (normalized) — Test Set", ".2f", ax)
    plt.tight_layout()
    path_norm = os.path.join(out_dir, "confusion_matrix_normalized.png")
    fig.savefig(path_norm, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved normalized confusion matrix → %s", path_norm)


def plot_feature_importance(
    clf,
    feature_names: list[str],
    top_n: int = TOP_N_FEATURES,
    out_dir: str = PLOTS_DIR,
) -> None:
    """Save a horizontal bar chart of the top *top_n* feature importances."""
    if not hasattr(clf, "feature_importances_"):
        log.warning("Model does not expose feature_importances_ — skipping plot.")
        return

    os.makedirs(out_dir, exist_ok=True)
    importances = clf.feature_importances_
    indices = np.argsort(importances)[::-1][:top_n]

    top_names  = [feature_names[i] for i in indices]
    top_scores = importances[indices]

    fig, ax = plt.subplots(figsize=(12, 6))
    bars = ax.barh(top_names[::-1], top_scores[::-1], color="steelblue")
    ax.bar_label(bars, fmt="%.4f", padding=3, fontsize=8)
    ax.set_xlabel("Mean Decrease in Impurity", fontsize=11)
    ax.set_title(f"Top {top_n} Feature Importances — Random Forest", fontsize=13)
    ax.margins(x=0.12)

    plt.tight_layout()
    path = os.path.join(out_dir, "feature_importance.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved feature importance plot → %s", path)


# ---------------------------------------------------------------------------
# Prediction output
# ---------------------------------------------------------------------------

def save_predictions(
    y_true: pd.Series,
    y_pred: np.ndarray,
    out_path: str = PREDICTIONS_PATH,
) -> None:
    """Persist true and predicted labels to a parquet file."""
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    pred_df = pd.DataFrame({
        "true_label":      y_true.values,
        "predicted_label": y_pred,
    })
    pred_df.to_parquet(out_path, compression="snappy", index=False)
    log.info("Saved predictions (%d rows) → %s", len(pred_df), out_path)


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run_evaluation() -> None:
    """End-to-end model evaluation pipeline."""

    # 1. Load model ----------------------------------------------------------
    clf = load_model()

    # 2. Load test data ------------------------------------------------------
    X_test, y_test = load_test_data()

    # 3. Load class label mapping --------------------------------------------
    mapping = load_label_mapping()
    if mapping:
        id_to_name  = {v: k for k, v in mapping.items()}
        class_names = [id_to_name[i] for i in sorted(id_to_name)]
    else:
        class_names = None

    # 4. Generate predictions ------------------------------------------------
    log.info("Generating predictions …")
    y_pred = clf.predict(X_test)

    # 5. Evaluate ------------------------------------------------------------
    metrics = compute_metrics(y_test, y_pred, class_names)

    # 6. Confusion matrix plots ----------------------------------------------
    plot_confusion_matrices(y_test, y_pred, class_names)

    # 7. Feature importance plot ---------------------------------------------
    plot_feature_importance(clf, list(X_test.columns))

    # 8. Save predictions ----------------------------------------------------
    save_predictions(y_test, y_pred)

    # 9. Summary -------------------------------------------------------------
    print("\n" + "=" * 60)
    print("  Evaluation Summary — Test Set")
    print("=" * 60)
    print(f"  Accuracy          : {metrics['accuracy']:.4f}")
    print(f"  Precision (macro) : {metrics['precision']:.4f}")
    print(f"  Recall    (macro) : {metrics['recall']:.4f}")
    print(f"  F1-score  (macro) : {metrics['f1']:.4f}")
    print("=" * 60)
    print(f"  Plots saved to    : {os.path.abspath(PLOTS_DIR)}")
    print(f"  Predictions saved : {os.path.abspath(PREDICTIONS_PATH)}")
    print("=" * 60 + "\n")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    run_evaluation()
