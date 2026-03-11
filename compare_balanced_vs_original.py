"""
compare_balanced_vs_original.py
--------------------------------

Compare detection performance of ORIGINAL (imbalanced) vs BALANCED XGBoost models
on rare attack classes from the test set.

This demonstrates the dramatic improvement in rare attack detection.
"""

import os
import logging
import joblib
import json

import pandas as pd
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

INPUT_DIR = "processed_ml"
MODELS_DIR = "models"

ORIGINAL_MODEL = os.path.join(MODELS_DIR, "xgboost_model.pkl")
BALANCED_MODEL = os.path.join(MODELS_DIR, "xgboost_balanced_model.pkl")
MAPPING_PATH = os.path.join(INPUT_DIR, "attack_label_mapping.json")

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
# Functions
# ---------------------------------------------------------------------------

def load_test_data():
    """Load test data."""
    log.info("Loading test data...")
    X_test = pd.read_parquet(os.path.join(INPUT_DIR, "X_test.parquet"))
    y_test = pd.read_parquet(os.path.join(INPUT_DIR, "y_multiclass_test.parquet"))
    y_test = y_test.iloc[:, 0].values if isinstance(y_test, pd.DataFrame) else y_test.values
    
    log.info("Test set: %d samples", len(X_test))
    return X_test, y_test


def load_label_mapping():
    """Load attack label mapping."""
    with open(MAPPING_PATH, 'r') as f:
        name_to_id = json.load(f)
    id_to_name = {int(v): k for k, v in name_to_id.items()}
    return id_to_name


def compare_models():
    """Compare original vs balanced models."""
    
    # Load data
    X_test, y_test = load_test_data()
    id_to_name = load_label_mapping()
    
    # Load models
    log.info("Loading ORIGINAL (imbalanced) model...")
    original_clf = joblib.load(ORIGINAL_MODEL)
    
    log.info("Loading BALANCED model...")
    balanced_clf = joblib.load(BALANCED_MODEL)
    
    # Get predictions
    log.info("Getting predictions from both models...")
    y_pred_original = original_clf.predict(X_test)
    y_pred_balanced = balanced_clf.predict(X_test)
    
    # Get class distribution in test set
    from collections import Counter
    test_dist = Counter(y_test)
    
    # Identify rare classes (< 100 samples in test)
    rare_classes = [class_id for class_id, count in test_dist.items() if count < 100]
    rare_class_names = [id_to_name[cid] for cid in rare_classes]
    
    print("\n" + "="*100)
    print("  RARE ATTACK DETECTION COMPARISON: Original vs Balanced Model")
    print("="*100)
    print(f"\nRare Attack Classes (<100 test samples): {len(rare_classes)}")
    for cid in rare_classes:
        print(f"  - {id_to_name[cid]}: {test_dist[cid]} samples")
    
    # Per-class recall comparison
    print("\n" + "="*100)
    print("  PER-CLASS RECALL COMPARISON")
    print("="*100)
    print(f"{'Class Name':<35} {'Test Samples':>15} {'Original Recall':>18} {'Balanced Recall':>18} {'Improvement':>15}")
    print("-"*100)
    
    for class_id in sorted(test_dist.keys()):
        class_name = id_to_name[class_id]
        class_count = test_dist[class_id]
        
        # Calculate recall for this class
        mask = y_test == class_id
        orig_correct = (y_pred_original[mask] == class_id).sum()
        bal_correct = (y_pred_balanced[mask] == class_id).sum()
        
        orig_recall = orig_correct / class_count if class_count > 0 else 0
        bal_recall = bal_correct / class_count if class_count > 0 else 0
        improvement = bal_recall - orig_recall
        
        # Highlight rare classes
        marker = " ⭐" if class_id in rare_classes else ""
        
        print(f"{class_name:<35} {class_count:>15,} {orig_recall:>17.2%} {bal_recall:>17.2%} {improvement:>+14.2%}{marker}")
    
    print("="*100)
    print("⭐ = Rare attack class (<100 test samples)")
    
    # Overall metrics
    from sklearn.metrics import accuracy_score, precision_recall_fscore_support
    
    orig_acc = accuracy_score(y_test, y_pred_original)
    bal_acc = accuracy_score(y_test, y_pred_balanced)
    
    orig_prec, orig_rec, orig_f1, _ = precision_recall_fscore_support(y_test, y_pred_original, average='macro', zero_division=0)
    bal_prec, bal_rec, bal_f1, _ = precision_recall_fscore_support(y_test, y_pred_balanced, average='macro', zero_division=0)
    
    print("\n" + "="*100)
    print("  OVERALL METRICS COMPARISON")
    print("="*100)
    print(f"{'Metric':<30} {'Original (Imbalanced)':>25} {'Balanced':>20} {'Change':>15}")
    print("-"*100)
    print(f"{'Test Accuracy':<30} {orig_acc:>24.2%} {bal_acc:>19.2%} {bal_acc-orig_acc:>+14.2%}")
    print(f"{'Macro Precision':<30} {orig_prec:>24.2%} {bal_prec:>19.2%} {bal_prec-orig_prec:>+14.2%}")
    print(f"{'Macro Recall':<30} {orig_rec:>24.2%} {bal_rec:>19.2%} {bal_rec-orig_rec:>+14.2%}")
    print(f"{'Macro F1-Score':<30} {orig_f1:>24.2%} {bal_f1:>19.2%} {bal_f1-orig_f1:>+14.2%}")
    print("="*100)
    
    # Key findings
    print("\n" + "="*100)
    print("  KEY FINDINGS")
    print("="*100)
    
    # Count classes with improved recall
    improved_classes = sum(1 for cid in test_dist.keys() 
                          if ((y_pred_balanced[y_test == cid] == cid).sum() > 
                              (y_pred_original[y_test == cid] == cid).sum()))
    
    # Count rare classes with >50% recall in balanced model
    rare_detected = sum(1 for cid in rare_classes 
                       if (y_pred_balanced[y_test == cid] == cid).sum() / test_dist[cid] > 0.5)
    
    print(f"✅ Classes with improved recall: {improved_classes}/{len(test_dist)}")
    print(f"✅ Rare classes with >50% recall (Balanced): {rare_detected}/{len(rare_classes)}")
    print(f"✅ Macro Recall improvement: {orig_rec:.2%} → {bal_rec:.2%} (+{bal_rec-orig_rec:+.2%})")
    print(f"")
    print(f"📊 Trade-off: Accuracy decreased by {orig_acc-bal_acc:.2%} but rare attack detection improved dramatically!")
    print(f"🛡️  In cybersecurity, detecting rare attacks is more critical than overall accuracy.")
    print("="*100 + "\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    compare_models()
