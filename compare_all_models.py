"""
compare_all_models.py
---------------------

Compare all trained intrusion detection models on the same test dataset.

Models
------
    Random Forest  →  models/random_forest_model.pkl
    XGBoost        →  models/xgboost_model.pkl
    Deep Neural Net (PyTorch) →  models/dnn_improved_model.pth

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

import torch
import torch.nn as nn

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
    "Deep Neural Net": ("pytorch",     os.path.join("models", "dnn_improved_model.pth")),
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
# PyTorch Model Architecture
# ---------------------------------------------------------------------------

class ImprovedDNNClassifier(nn.Module):
    """PyTorch DNN classifier with BatchNorm for intrusion detection."""
    
    def __init__(self, input_dim: int, num_classes: int):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3),
            
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.25),
            
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            # No dropout before output layer
            nn.Linear(64, num_classes)
        )
    
    def forward(self, x):
        return self.network(x)


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
    if kind == "pytorch":
        # Load PyTorch model checkpoint
        checkpoint = torch.load(path, map_location=torch.device('cpu'))
        
        # Check if checkpoint contains metadata or just weights
        if 'model_state_dict' in checkpoint:
            # Checkpoint contains metadata
            input_dim = checkpoint['n_features']
            num_classes = checkpoint['n_classes']
            state_dict = checkpoint['model_state_dict']
        else:
            # Direct state dict - infer dimensions
            first_layer_key = 'network.0.weight'
            if first_layer_key in checkpoint:
                input_dim = checkpoint[first_layer_key].shape[1]
                output_layer_keys = [k for k in checkpoint.keys() if 'weight' in k and 'BatchNorm' not in k]
                last_layer_key = max(output_layer_keys, key=lambda x: int(x.split('.')[1]))
                num_classes = checkpoint[last_layer_key].shape[0]
                state_dict = checkpoint
            else:
                # Fallback
                input_dim = 78
                num_classes = 14
                state_dict = checkpoint
        
        clf = ImprovedDNNClassifier(input_dim, num_classes)
        clf.load_state_dict(state_dict)
        clf.eval()
        log.info("Loaded '%s' (PyTorch DNN: %d → %d classes)", name, input_dim, num_classes)
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
    if kind == "pytorch":
        # PyTorch inference
        X_tensor = torch.from_numpy(X.values.astype(np.float32))
        with torch.no_grad():
            outputs = clf(X_tensor)
            y_pred = torch.argmax(outputs, dim=1).numpy()
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

    try:
        clf = load_model(name, kind, path)
        y_pred, elapsed = predict(clf, X_test, kind)
    except Exception as e:
        log.error("Failed to load or predict with '%s': %s", name, str(e))
        log.warning("Skipping corrupted model: %s", path)
        return None

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
