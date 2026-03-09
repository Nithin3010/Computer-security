"""
compare_all_models.py
---------------------

Compare all trained intrusion detection models on the same test dataset.

Models
------
    Random Forest  →  models/random_forest_model.pkl
    XGBoost        →  models/xgboost_model.pkl
    Deep Neural Net →  models/dnn_model.keras

Dataset
-------
    processed_ml/X_test.parquet
    processed_ml/y_multiclass_test.parquet

Outputs
-------
    results/model_comparison.csv
    model_plots/model_comparison.png
"""

import logging
import os
import time

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import tensorflow as tf

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

INPUT_DIR    = "processed_ml"
RESULTS_DIR  = "results"
PLOTS_DIR    = "model_plots"
OUTPUT_CSV   = os.path.join(RESULTS_DIR, "model_comparison.csv")
OUTPUT_PLOT  = os.path.join(PLOTS_DIR,   "model_comparison.png")

BATCH_SIZE   = 1024   # for DNN inference

MODELS = {
    "Random Forest":   ("sklearn",     os.path.join("models", "random_forest_model.pkl")),
    "XGBoost":         ("sklearn",     os.path.join("models", "xgboost_model.pkl")),
    "Deep Neural Net": ("keras",       os.path.join("models", "dnn_model.keras")),
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
# Data loading
# ---------------------------------------------------------------------------

def load_test_data() -> tuple[pd.DataFrame, np.ndarray]:
    log.info("Loading test dataset from: %s", INPUT_DIR)

    X_test = pd.read_parquet(os.path.join(INPUT_DIR, "X_test.parquet"))
    y_df   = pd.read_parquet(os.path.join(INPUT_DIR, "y_multiclass_test.parquet"))
    y_test = (y_df.iloc[:, 0] if isinstance(y_df, pd.DataFrame) else y_df).values

    log.info("Dataset — %d rows, %d features", len(X_test), X_test.shape[1])
    return X_test, y_test


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_model(name: str, kind: str, path: str):
    log.info("Loading '%s' from: %s", name, path)
    if kind == "keras":
        clf = tf.keras.models.load_model(path)
    else:
        clf = joblib.load(path)
    log.info("Loaded '%s' (%s)", name, type(clf).__name__)
    return clf


# ---------------------------------------------------------------------------
# Prediction & timing
# ---------------------------------------------------------------------------

def predict(clf, X: pd.DataFrame, kind: str) -> tuple[np.ndarray, float]:
    """Return (predictions, inference_seconds)."""
    t0 = time.perf_counter()
    if kind == "keras":
        probs  = clf.predict(X.values.astype(np.float32), batch_size=BATCH_SIZE, verbose=0)
        y_pred = np.argmax(probs, axis=1)
    else:
        y_pred = clf.predict(X)
    elapsed = time.perf_counter() - t0
    return y_pred, elapsed


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate_model(
    name: str,
    kind: str,
    path: str,
    X_test: pd.DataFrame,
    y_test: np.ndarray,
) -> dict | None:
    if not os.path.exists(path):
        log.warning("Model file not found, skipping: %s", path)
        return None

    clf = load_model(name, kind, path)
    y_pred, elapsed = predict(clf, X_test, kind)

    acc  = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average="macro", zero_division=0)
    rec  = recall_score(y_test, y_pred, average="macro", zero_division=0)
    f1   = f1_score(y_test, y_pred, average="macro", zero_division=0)

    log.info(
        "[%s]  accuracy=%.4f  prec=%.4f  recall=%.4f  F1=%.4f  inference=%.2fs",
        name, acc, prec, rec, f1, elapsed,
    )

    return dict(
        model=name,
        accuracy=acc,
        precision=prec,
        recall=rec,
        f1=f1,
        inference_time_s=elapsed,
    )


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------

def save_csv(results: list[dict], path: str = OUTPUT_CSV) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df = pd.DataFrame(results)
    df.to_csv(path, index=False, float_format="%.4f")
    log.info("Saved comparison CSV → %s", path)


def print_table(results: list[dict]) -> None:
    col_w = max(len(r["model"]) for r in results) + 2
    header = (
        f"{'Model':<{col_w}}"
        f"{'Accuracy':>12}"
        f"{'Precision':>12}"
        f"{'Recall':>12}"
        f"{'F1':>12}"
        f"{'Inference(s)':>14}"
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
            f"{r['inference_time_s']:>14.2f}"
        )
    print(sep)

    best = max(results, key=lambda r: r["f1"])
    print(f"\n  Best model by F1-score: {best['model']}  (F1 = {best['f1']:.4f})\n")


def save_comparison_plot(results: list[dict], path: str = OUTPUT_PLOT) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)

    models    = [r["model"]    for r in results]
    accuracy  = [r["accuracy"] for r in results]
    f1_scores = [r["f1"]       for r in results]

    x     = np.arange(len(models))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))
    bars_acc = ax.bar(x - width / 2, accuracy,  width, label="Accuracy",  color="steelblue")
    bars_f1  = ax.bar(x + width / 2, f1_scores, width, label="F1 (macro)", color="coral")

    ax.bar_label(bars_acc, fmt="%.4f", padding=3, fontsize=9)
    ax.bar_label(bars_f1,  fmt="%.4f", padding=3, fontsize=9)

    ax.set_xlabel("Model", fontsize=11)
    ax.set_ylabel("Score", fontsize=11)
    ax.set_title("Model Comparison — Accuracy & Macro F1", fontsize=13)
    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=10)
    ax.set_ylim(0, 1.08)
    ax.legend(fontsize=10)
    ax.grid(axis="y", linestyle=":", alpha=0.6)

    plt.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved comparison plot → %s", path)


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run_comparison() -> None:
    """Load all models, evaluate on the shared test set, and report."""

    # 1. Load test data ------------------------------------------------------
    X_test, y_test = load_test_data()

    # 2. Evaluate each model -------------------------------------------------
    results: list[dict] = []
    for name, (kind, path) in MODELS.items():
        result = evaluate_model(name, kind, path, X_test, y_test)
        if result:
            results.append(result)

    if not results:
        log.error("No models could be evaluated. Aborting.")
        return

    # 3. Save CSV ------------------------------------------------------------
    save_csv(results)

    # 4. Print table ---------------------------------------------------------
    print_table(results)

    # 5. Save bar chart ------------------------------------------------------
    save_comparison_plot(results)

    print(f"  Results  : {os.path.abspath(OUTPUT_CSV)}")
    print(f"  Plot     : {os.path.abspath(OUTPUT_PLOT)}\n")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    run_comparison()
