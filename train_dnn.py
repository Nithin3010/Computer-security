"""
train_dnn.py
------------

Train a Deep Neural Network (DNN) for network intrusion detection.

Dataset
-------
    processed_ml/X_train.parquet, X_val.parquet, X_test.parquet
    processed_ml/y_multiclass_train.parquet, y_multiclass_val.parquet,
                 y_multiclass_test.parquet

Architecture
------------
    Input(78)  →  Dense(256, ReLU) + Dropout(0.3)
               →  Dense(128, ReLU) + Dropout(0.3)
               →  Dense(64,  ReLU)
               →  Dense(num_classes, Softmax)

Outputs
-------
    models/dnn_model.keras
    model_plots/dnn_training_loss.png
    model_plots/dnn_training_accuracy.png
"""

import logging
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.utils.class_weight import compute_class_weight

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")   # suppress TF C++ info logs

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

INPUT_DIR   = "processed_ml"
MODEL_DIR   = "models"
PLOTS_DIR   = "model_plots"
MODEL_PATH  = os.path.join(MODEL_DIR, "dnn_model.keras")

BATCH_SIZE   = 1024
EPOCHS       = 30
PATIENCE     = 5
RANDOM_STATE = 42

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
    log.info("Loaded %-35s  shape=%s", name + ".parquet", df.shape)
    return df


def _to_series(obj: pd.DataFrame | pd.Series) -> pd.Series:
    if isinstance(obj, pd.DataFrame):
        return obj.iloc[:, 0]
    return obj


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_datasets() -> tuple[
    np.ndarray, np.ndarray, np.ndarray,
    np.ndarray, np.ndarray, np.ndarray,
]:
    log.info("Loading datasets from: %s", INPUT_DIR)

    X_train = _load_parquet("X_train").values.astype(np.float32)
    X_val   = _load_parquet("X_val").values.astype(np.float32)
    X_test  = _load_parquet("X_test").values.astype(np.float32)

    y_train = _to_series(_load_parquet("y_multiclass_train")).values.astype(np.int32)
    y_val   = _to_series(_load_parquet("y_multiclass_val")).values.astype(np.int32)
    y_test  = _to_series(_load_parquet("y_multiclass_test")).values.astype(np.int32)

    log.info(
        "Dataset sizes — train: %d | val: %d | test: %d",
        len(X_train), len(X_val), len(X_test),
    )
    log.info("Number of features : %d", X_train.shape[1])
    return X_train, X_val, X_test, y_train, y_val, y_test


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

def build_model(n_features: int, n_classes: int) -> keras.Model:
    tf.random.set_seed(RANDOM_STATE)

    model = keras.Sequential(
        [
            keras.Input(shape=(n_features,), name="input"),
            layers.Dense(256, activation="relu", name="dense_1"),
            layers.Dropout(0.3, name="dropout_1"),
            layers.Dense(128, activation="relu", name="dense_2"),
            layers.Dropout(0.3, name="dropout_2"),
            layers.Dense(64, activation="relu", name="dense_3"),
            layers.Dense(n_classes, activation="softmax", name="output"),
        ],
        name="dnn_ids",
    )

    model.compile(
        optimizer=keras.optimizers.Adam(),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    model.summary(print_fn=log.info)
    return model


# ---------------------------------------------------------------------------
# Class weights
# ---------------------------------------------------------------------------

def get_class_weights(y_train: np.ndarray) -> dict:
    classes = np.unique(y_train)
    weights = compute_class_weight(
        class_weight="balanced",
        classes=classes,
        y=y_train,
    )
    cw = {int(cls): float(w) for cls, w in zip(classes, weights)}
    log.info("Class weights computed for %d classes.", len(cw))
    return cw


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_model(
    model: keras.Model,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    class_weights: dict,
) -> keras.callbacks.History:
    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=PATIENCE,
            restore_best_weights=True,
            verbose=1,
        ),
    ]

    log.info(
        "Starting training — epochs=%d, batch_size=%d, patience=%d",
        EPOCHS, BATCH_SIZE, PATIENCE,
    )

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        class_weight=class_weights,
        callbacks=callbacks,
        verbose=1,
    )

    log.info("Training complete — ran %d epochs.", len(history.history["loss"]))
    return history


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate(
    model: keras.Model,
    X: np.ndarray,
    y_true: np.ndarray,
    split_name: str,
) -> dict:
    y_pred = np.argmax(model.predict(X, batch_size=BATCH_SIZE, verbose=0), axis=1)

    acc  = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average="macro", zero_division=0)
    rec  = recall_score(y_true, y_pred, average="macro", zero_division=0)
    f1   = f1_score(y_true, y_pred, average="macro", zero_division=0)

    log.info("[%s]  accuracy=%.4f  precision=%.4f  recall=%.4f  F1=%.4f",
             split_name, acc, prec, rec, f1)

    report = classification_report(y_true, y_pred, zero_division=0)
    print(f"\n{'='*60}")
    print(f"  Classification Report — {split_name}")
    print(f"{'='*60}")
    print(report)

    return dict(accuracy=acc, precision=prec, recall=rec, f1=f1)


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def _plot_metric(
    history: keras.callbacks.History,
    metric: str,
    title: str,
    ylabel: str,
    filename: str,
    out_dir: str,
) -> None:
    os.makedirs(out_dir, exist_ok=True)
    train_vals = history.history[metric]
    val_vals   = history.history[f"val_{metric}"]
    epochs     = range(1, len(train_vals) + 1)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(epochs, train_vals, label="Train")
    ax.plot(epochs, val_vals,   label="Validation", linestyle="--")
    ax.set_title(title, fontsize=13)
    ax.set_xlabel("Epoch", fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.legend()
    ax.grid(True, linestyle=":")
    plt.tight_layout()

    path = os.path.join(out_dir, filename)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved plot → %s", path)


def save_training_plots(history: keras.callbacks.History, out_dir: str = PLOTS_DIR) -> None:
    _plot_metric(
        history, "loss",
        title="DNN Training vs Validation Loss",
        ylabel="Loss",
        filename="dnn_training_loss.png",
        out_dir=out_dir,
    )
    _plot_metric(
        history, "accuracy",
        title="DNN Training vs Validation Accuracy",
        ylabel="Accuracy",
        filename="dnn_training_accuracy.png",
        out_dir=out_dir,
    )


# ---------------------------------------------------------------------------
# Saving
# ---------------------------------------------------------------------------

def save_model(model: keras.Model, path: str = MODEL_PATH) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    model.save(path)
    log.info("Saved model → %s", path)


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run_pipeline() -> None:
    """End-to-end DNN training and evaluation pipeline."""

    # 1. Load ----------------------------------------------------------------
    X_train, X_val, X_test, y_train, y_val, y_test = load_datasets()

    n_features = X_train.shape[1]
    n_classes  = int(np.unique(y_train).size)
    log.info("n_features=%d  n_classes=%d", n_features, n_classes)

    # 2. Class weights -------------------------------------------------------
    class_weights = get_class_weights(y_train)

    # 3. Build model ---------------------------------------------------------
    model = build_model(n_features, n_classes)

    # 4. Train ---------------------------------------------------------------
    history = train_model(model, X_train, y_train, X_val, y_val, class_weights)

    # 5. Evaluate ------------------------------------------------------------
    val_metrics  = evaluate(model, X_val,  y_val,  "Validation")
    test_metrics = evaluate(model, X_test, y_test, "Test")

    # 6. Save plots ----------------------------------------------------------
    save_training_plots(history)

    # 7. Save model ----------------------------------------------------------
    save_model(model)

    # 8. Summary -------------------------------------------------------------
    print("\n" + "=" * 60)
    print("  DNN — Final Summary")
    print("=" * 60)
    for name, m in [("Validation", val_metrics), ("Test", test_metrics)]:
        print(f"  [{name}]")
        print(f"    Accuracy          : {m['accuracy']:.4f}")
        print(f"    Precision (macro) : {m['precision']:.4f}")
        print(f"    Recall    (macro) : {m['recall']:.4f}")
        print(f"    F1-score  (macro) : {m['f1']:.4f}")
    print("=" * 60)
    print(f"  Model saved  : {os.path.abspath(MODEL_PATH)}")
    print(f"  Plots saved  : {os.path.abspath(PLOTS_DIR)}")
    print("=" * 60 + "\n")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    run_pipeline()
