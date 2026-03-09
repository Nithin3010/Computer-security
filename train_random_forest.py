"""
train_random_forest.py
----------------------

Train and evaluate a Random Forest model for intrusion detection.

The ML-ready dataset was produced by prepare_ml_dataset.py and stored in:

    processed_ml/

Pipeline Steps
--------------
1. Load training, validation, and test datasets.
2. Flatten label DataFrames to Series if necessary.
3. Train a RandomForestClassifier.
4. Evaluate model on validation set.
5. Evaluate final model on test set.
6. Compute performance metrics.
7. Plot confusion matrix and feature importance.
8. Save trained model.
"""

import os
import json
import logging

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
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
OUTPUT_DIR  = "model_plots"
MODEL_DIR   = "models"
MODEL_PATH  = os.path.join(MODEL_DIR, "random_forest_model.pkl")
MAPPING_PATH = os.path.join(INPUT_DIR, "attack_label_mapping.json")

RF_PARAMS = dict(
    n_estimators=200,
    max_depth=None,
    n_jobs=-1,
    class_weight="balanced",
    random_state=42,
)

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
# Helpers
# ---------------------------------------------------------------------------

def _load_parquet(name: str) -> pd.DataFrame:
    path = os.path.join(INPUT_DIR, f"{name}.parquet")
    df = pd.read_parquet(path)
    log.info("Loaded %-30s  shape=%s", name + ".parquet", df.shape)
    return df


def _to_series(obj: pd.DataFrame | pd.Series) -> pd.Series:
    """Flatten a single-column DataFrame to a Series."""
    if isinstance(obj, pd.DataFrame):
        return obj.iloc[:, 0]
    return obj


def load_datasets() -> tuple[
    pd.DataFrame, pd.DataFrame, pd.DataFrame,
    pd.Series,   pd.Series,   pd.Series,
]:
    log.info("Loading datasets from: %s", INPUT_DIR)

    X_train = _load_parquet("X_train")
    X_val   = _load_parquet("X_val")
    X_test  = _load_parquet("X_test")

    y_train = _to_series(_load_parquet("y_multiclass_train"))
    y_val   = _to_series(_load_parquet("y_multiclass_val"))
    y_test  = _to_series(_load_parquet("y_multiclass_test"))

    log.info(
        "Dataset sizes — train: %d | val: %d | test: %d",
        len(X_train), len(X_val), len(X_test),
    )
    log.info("Number of features: %d", X_train.shape[1])
    return X_train, X_val, X_test, y_train, y_val, y_test


def load_label_mapping() -> dict:
    """Return {class_name: class_id} from the saved JSON mapping."""
    if not os.path.exists(MAPPING_PATH):
        log.warning("Label mapping not found at %s — using integer labels.", MAPPING_PATH)
        return {}
    with open(MAPPING_PATH, encoding="utf-8") as fh:
        mapping = json.load(fh)
    log.info("Loaded label mapping: %d classes", len(mapping))
    return mapping


def train_model(X_train: pd.DataFrame, y_train: pd.Series) -> RandomForestClassifier:
    log.info("Training RandomForestClassifier  params=%s", RF_PARAMS)
    clf = RandomForestClassifier(**RF_PARAMS)
    clf.fit(X_train, y_train)
    log.info("Training complete.")
    return clf


def evaluate(
    clf: RandomForestClassifier,
    X: pd.DataFrame,
    y_true: pd.Series,
    split_name: str,
    class_names: list[str] | None = None,
) -> dict:
    """Compute and log evaluation metrics for a single split."""
    y_pred = clf.predict(X)

    acc  = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average="weighted", zero_division=0)
    rec  = recall_score(y_true, y_pred, average="weighted", zero_division=0)
    f1   = f1_score(y_true, y_pred, average="weighted", zero_division=0)

    log.info("[%s]  accuracy=%.4f  precision=%.4f  recall=%.4f  F1=%.4f",
             split_name, acc, prec, rec, f1)

    report = classification_report(
        y_true, y_pred,
        target_names=class_names,
        zero_division=0,
    )
    print(f"\n{'='*60}")
    print(f"  Classification Report — {split_name}")
    print(f"{'='*60}")
    print(report)

    return dict(
        split=split_name,
        y_pred=y_pred,
        accuracy=acc,
        precision=prec,
        recall=rec,
        f1=f1,
    )


def plot_confusion_matrix(
    y_true: pd.Series,
    y_pred: np.ndarray,
    class_names: list[str] | None,
    split_name: str,
    out_dir: str,
) -> None:
    os.makedirs(out_dir, exist_ok=True)

    cm = confusion_matrix(y_true, y_pred)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    labels = class_names if class_names else [str(i) for i in range(cm.shape[0])]

    fig, axes = plt.subplots(1, 2, figsize=(20, 8))

    for ax, data, title, fmt in zip(
        axes,
        [cm, cm_norm],
        ["Confusion Matrix (counts)", "Confusion Matrix (normalized)"],
        ["d", ".2f"],
    ):
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
        ax.set_title(f"{title}\n({split_name})", fontsize=13)
        ax.set_xlabel("Predicted label", fontsize=11)
        ax.set_ylabel("True label", fontsize=11)
        ax.tick_params(axis="x", rotation=45)
        ax.tick_params(axis="y", rotation=0)

    plt.tight_layout()
    path = os.path.join(out_dir, "confusion_matrix.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved confusion matrix → %s", path)


def plot_feature_importance(
    clf: RandomForestClassifier,
    feature_names: list[str],
    top_n: int,
    out_dir: str,
) -> None:
    os.makedirs(out_dir, exist_ok=True)

    importances = clf.feature_importances_
    indices = np.argsort(importances)[::-1][:top_n]

    top_names   = [feature_names[i] for i in indices]
    top_scores  = importances[indices]

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


def save_model(clf: RandomForestClassifier) -> None:
    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(clf, MODEL_PATH)
    log.info("Saved model → %s", MODEL_PATH)


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run_pipeline() -> None:
    """End-to-end Random Forest training and evaluation pipeline."""

    # 1. Load ----------------------------------------------------------------
    X_train, X_val, X_test, y_train, y_val, y_test = load_datasets()

    # 2. Label names ---------------------------------------------------------
    mapping = load_label_mapping()
    # Invert {class_name: id} → {id: class_name} for ordered label list
    if mapping:
        id_to_name = {v: k for k, v in mapping.items()}
        class_names = [id_to_name[i] for i in sorted(id_to_name)]
    else:
        class_names = None

    # 3. Train ---------------------------------------------------------------
    clf = train_model(X_train, y_train)

    # 4. Validation evaluation -----------------------------------------------
    val_results = evaluate(clf, X_val, y_val, "Validation", class_names)

    # 5. Test evaluation -----------------------------------------------------
    test_results = evaluate(clf, X_test, y_test, "Test", class_names)

    # 6. Plots ---------------------------------------------------------------
    plot_confusion_matrix(
        y_test, test_results["y_pred"],
        class_names, "Test Set", OUTPUT_DIR,
    )
    plot_feature_importance(
        clf, list(X_train.columns), TOP_N_FEATURES, OUTPUT_DIR,
    )

    # 7. Save model ----------------------------------------------------------
    save_model(clf)

    # 8. Summary -------------------------------------------------------------
    print("\n" + "=" * 60)
    print("  Random Forest — Final Summary")
    print("=" * 60)
    for results in (val_results, test_results):
        split = results["split"]
        print(f"  [{split}]")
        print(f"    Accuracy  : {results['accuracy']:.4f}")
        print(f"    Precision : {results['precision']:.4f}")
        print(f"    Recall    : {results['recall']:.4f}")
        print(f"    F1-score  : {results['f1']:.4f}")
    print("=" * 60)
    print(f"  Model saved  : {os.path.abspath(MODEL_PATH)}")
    print(f"  Plots saved  : {os.path.abspath(OUTPUT_DIR)}")
    print("=" * 60 + "\n")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    run_pipeline()
