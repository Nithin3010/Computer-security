"""
compare_models.py
-----------------

Compare the performance of multiple trained intrusion detection models
on the same held-out test dataset.

Models compared
---------------
    Random Forest  →  models/random_forest_model.pkl
    XGBoost        →  models/xgboost_model.pkl

Dataset
-------
    processed_ml/X_test.parquet
    processed_ml/y_multiclass_test.parquet
"""

import logging
import os

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

INPUT_DIR   = "processed_ml"
RESULTS_DIR = "results"
OUTPUT_PATH = os.path.join(RESULTS_DIR, "model_comparison.csv")

MODELS = {
    "Random Forest": os.path.join("models", "random_forest_model.pkl"),
    "XGBoost":       os.path.join("models", "xgboost_model.pkl"),
}

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

def load_test_data() -> tuple[pd.DataFrame, pd.Series]:
    """Load X_test and y_multiclass_test from processed_ml/."""
    log.info("Loading test dataset from: %s", INPUT_DIR)

    X_test = pd.read_parquet(os.path.join(INPUT_DIR, "X_test.parquet"))
    y_df   = pd.read_parquet(os.path.join(INPUT_DIR, "y_multiclass_test.parquet"))
    y_test = y_df.iloc[:, 0] if isinstance(y_df, pd.DataFrame) else y_df

    log.info("Dataset size — %d rows, %d features", len(X_test), X_test.shape[1])
    return X_test, y_test


def load_model(name: str, path: str):
    """Load a single model from *path* and log the result."""
    log.info("Loading model '%s' from: %s", name, path)
    clf = joblib.load(path)
    log.info("Loaded '%s' (%s)", name, type(clf).__name__)
    return clf


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate_model(
    clf,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    model_name: str,
) -> dict:
    """Generate predictions and compute macro metrics for one model."""
    y_pred = clf.predict(X_test)

    acc  = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average="macro", zero_division=0)
    rec  = recall_score(y_test, y_pred, average="macro", zero_division=0)
    f1   = f1_score(y_test, y_pred, average="macro", zero_division=0)

    log.info(
        "[%s]  accuracy=%.4f  precision=%.4f  recall=%.4f  F1=%.4f",
        model_name, acc, prec, rec, f1,
    )

    return dict(model=model_name, accuracy=acc, precision=prec, recall=rec, f1=f1)


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------

def save_comparison(results: list[dict], path: str = OUTPUT_PATH) -> None:
    """Persist the comparison table as a CSV file."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df = pd.DataFrame(results)
    df.to_csv(path, index=False, float_format="%.4f")
    log.info("Saved model comparison → %s", path)


def print_comparison(results: list[dict]) -> None:
    """Print a formatted comparison table to stdout."""
    col_w = max(len(r["model"]) for r in results) + 2

    header = (
        f"{'Model':<{col_w}}"
        f"{'Accuracy':>12}"
        f"{'Precision':>12}"
        f"{'Recall':>12}"
        f"{'F1':>12}"
    )
    sep = "-" * len(header)

    print("\nModel Performance Comparison")
    print(sep)
    print(header)
    print(sep)
    for r in results:
        print(
            f"{r['model']:<{col_w}}"
            f"{r['accuracy']:>12.4f}"
            f"{r['precision']:>12.4f}"
            f"{r['recall']:>12.4f}"
            f"{r['f1']:>12.4f}"
        )
    print(sep)

    # Highlight the best model by F1
    best = max(results, key=lambda r: r["f1"])
    print(f"\n  Best model by F1-score: {best['model']}  (F1 = {best['f1']:.4f})\n")


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run_comparison() -> None:
    """Load all models, evaluate on the shared test set, and report."""

    # 1. Load test data ------------------------------------------------------
    X_test, y_test = load_test_data()

    # 2. Evaluate each model -------------------------------------------------
    results: list[dict] = []
    for name, path in MODELS.items():
        if not os.path.exists(path):
            log.warning("Model file not found, skipping: %s", path)
            continue
        clf = load_model(name, path)
        metrics = evaluate_model(clf, X_test, y_test, name)
        results.append(metrics)

    if not results:
        log.error("No models could be evaluated. Aborting.")
        return

    # 3. Save CSV ------------------------------------------------------------
    save_comparison(results)

    # 4. Print table ---------------------------------------------------------
    print_comparison(results)

    print(f"  Results saved to: {os.path.abspath(OUTPUT_PATH)}\n")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    run_comparison()
