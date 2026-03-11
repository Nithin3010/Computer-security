"""
train_xgboost_balanced.py
-------------------------

Train XGBoost on the BALANCED dataset to improve minority class detection.

This script demonstrates the impact of class balancing on model performance.
"""

import os
import logging
import time
import joblib

import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.metrics import (
    classification_report,
    accuracy_score,
    confusion_matrix,
)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

INPUT_DIR = "processed_ml"
OUTPUT_DIR = "models"

# XGBoost parameters - reduced for memory efficiency
XGBOOST_PARAMS = {
    'n_estimators': 100,  # Reduced from 200
    'max_depth': 10,  # Reduced from 15
    'learning_rate': 0.1,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'tree_method': 'hist',  # Fast training
    'random_state': 42,
    'n_jobs': 4,  # Limit parallelism to save memory
    'eval_metric': 'mlogloss',
}

# Subsample stratified training data to reduce memory usage
BALANCED_SAMPLE_SIZE = 2000000  # Use 2M out of 4.9M for training

MODEL_PATH = os.path.join(OUTPUT_DIR, "xgboost_balanced_model.pkl")

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
# Pipeline
# ---------------------------------------------------------------------------

def load_data():
    """Load balanced training data and original test data."""
    log.info("Loading BALANCED training data...")
    X_train = pd.read_parquet(os.path.join(INPUT_DIR, "X_train_balanced.parquet"))
    y_train = pd.read_parquet(os.path.join(INPUT_DIR, "y_multiclass_train_balanced.parquet"))
    y_train = y_train.iloc[:, 0].values if isinstance(y_train, pd.DataFrame) else y_train.values
    
    log.info("Full balanced set: %d samples", len(X_train))
    
    # Subsample to reduce memory usage
    if len(X_train) > BALANCED_SAMPLE_SIZE:
        log.info("Subsampling to %d samples for memory efficiency...", BALANCED_SAMPLE_SIZE)
        indices = np.random.choice(len(X_train), size=BALANCED_SAMPLE_SIZE, replace=False)
        X_train = X_train.iloc[indices].reset_index(drop=True)
        y_train = y_train[indices]
    
    log.info("Training set: %d samples", len(X_train))
    
    log.info("Loading ORIGINAL test data (to compare performance)...")
    X_test = pd.read_parquet(os.path.join(INPUT_DIR, "X_test.parquet"))
    y_test = pd.read_parquet(os.path.join(INPUT_DIR, "y_multiclass_test.parquet"))
    y_test = y_test.iloc[:, 0].values if isinstance(y_test, pd.DataFrame) else y_test.values
    
    log.info("Test set: %d samples", len(X_test))
    
    return X_train, y_train, X_test, y_test


def train_model(X_train, y_train):
    """Train XGBoost classifier."""
    log.info("Training XGBoost on BALANCED dataset...")
    log.info("Parameters: %s", XGBOOST_PARAMS)
    
    clf = XGBClassifier(**XGBOOST_PARAMS)
    
    t0 = time.time()
    clf.fit(X_train, y_train)
    elapsed = time.time() - t0
    
    log.info("Training completed in %.2f seconds", elapsed)
    return clf


def evaluate_model(clf, X_test, y_test):
    """Evaluate model on test set."""
    log.info("Evaluating model on original (imbalanced) test set...")
    
    y_pred = clf.predict(X_test)
    
    acc = accuracy_score(y_test, y_pred)
    log.info("Test Accuracy: %.4f", acc)
    
    # Detailed classification report
    print("\n" + "="*80)
    print("  Classification Report (Test Set)")
    print("="*80)
    print(classification_report(y_test, y_pred, digits=4))
    
    print("\n" + "="*80)
    print(f"  Overall Test Accuracy: {acc:.4f} ({100*acc:.2f}%)")
    print("="*80 + "\n")
    
    return acc, y_pred


def save_model(clf, path=MODEL_PATH):
    """Save trained model."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(clf, path)
    log.info("Model saved to: %s", path)


def main():
    """Main training pipeline."""
    log.info("="*80)
    log.info("  XGBoost Training on BALANCED Dataset")
    log.info("="*80)
    
    # Load data
    X_train, y_train, X_test, y_test = load_data()
    
    # Train
    clf = train_model(X_train, y_train)
    
    # Evaluate
    acc, y_pred = evaluate_model(clf, X_test, y_test)
    
    # Save
    save_model(clf)
    
    print("\n" + "="*80)
    print("  TRAINING COMPLETE")
    print("="*80)
    print(f"  Model trained on BALANCED data with perfect class distribution")
    print(f"  Test Accuracy: {acc:.4f}")
    print(f"  Model saved: {MODEL_PATH}")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
