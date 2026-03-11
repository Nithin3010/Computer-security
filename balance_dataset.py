"""
balance_dataset.py
------------------

Apply class balancing techniques to handle severe class imbalance in the IDS dataset.

Techniques Applied:
1. SMOTE (Synthetic Minority Over-sampling Technique) for rare classes
2. Random Under-sampling for majority class (Benign)
3. Hybrid approach combining both

Input:
    processed_ml/X_train.parquet, y_multiclass_train.parquet

Output:
    processed_ml/X_train_balanced.parquet, y_multiclass_train_balanced.parquet
    processed_ml/balancing_report.txt
"""

import os
import logging
import time

import numpy as np
import pandas as pd
from collections import Counter
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.combine import SMOTEENN

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

INPUT_DIR = "processed_ml"
OUTPUT_DIR = "processed_ml"

# Balancing strategy: 'smote', 'undersample', 'hybrid', 'smoteenn'
STRATEGY = 'hybrid'

# Target samples per class (for SMOTE)
# 'auto': balance all to majority class size
# 'minority': balance minority to second largest
# dict: specify exact counts per class
# float (0-1): ratio relative to majority class
SMOTE_STRATEGY = 0.3  # Each minority class will have 30% of majority class samples

# Undersample majority class ratio
# 0.5 means Benign:Attack ratio will be 2:1
UNDERSAMPLE_RATIO = 0.5

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
# Helper Functions
# ---------------------------------------------------------------------------

def load_training_data():
    """Load training features and labels."""
    log.info("Loading training data from: %s", INPUT_DIR)
    
    X_train = pd.read_parquet(os.path.join(INPUT_DIR, "X_train.parquet"))
    y_train = pd.read_parquet(os.path.join(INPUT_DIR, "y_multiclass_train.parquet"))
    y_train = y_train.iloc[:, 0].values if isinstance(y_train, pd.DataFrame) else y_train.values
    
    log.info("Loaded %d training samples with %d features", len(X_train), X_train.shape[1])
    return X_train, y_train


def load_label_mapping():
    """Load attack label mapping."""
    import json
    mapping_path = os.path.join(INPUT_DIR, "attack_label_mapping.json")
    with open(mapping_path, 'r') as f:
        name_to_id = json.load(f)
    id_to_name = {int(v): k for k, v in name_to_id.items()}
    return id_to_name


def print_class_distribution(y, title="Class Distribution", id_to_name=None):
    """Print detailed class distribution."""
    counter = Counter(y)
    total = len(y)
    
    print(f"\n{'='*80}")
    print(f"  {title}")
    print(f"{'='*80}")
    print(f"{'Class ID':<10} {'Class Name':<30} {'Count':>15} {'Percentage':>12}")
    print("-" * 80)
    
    for class_id in sorted(counter.keys()):
        count = counter[class_id]
        percentage = 100.0 * count / total
        class_name = id_to_name.get(class_id, f"Class_{class_id}") if id_to_name else f"Class_{class_id}"
        print(f"{class_id:<10} {class_name:<30} {count:>15,} {percentage:>11.2f}%")
    
    print("-" * 80)
    print(f"{'TOTAL':<10} {'':<30} {total:>15,} {100.0:>11.1f}%")
    print(f"{'='*80}\n")


def apply_smote(X_train, y_train, strategy=SMOTE_STRATEGY):
    """Apply SMOTE to oversample minority classes."""
    log.info("Applying SMOTE with strategy: %s", strategy)
    
    t0 = time.time()
    
    # For multi-class, convert float strategy to dict
    if isinstance(strategy, float):
        counter = Counter(y_train)
        majority_class_count = max(counter.values())
        target_count = int(majority_class_count * strategy)
        
        # Create dict: oversample all minority classes to target_count
        sampling_strategy_dict = {}
        for class_id, count in counter.items():
            if count < target_count:
                sampling_strategy_dict[class_id] = target_count
        
        log.info("Target samples per minority class: %d (%.1f%% of majority)", 
                 target_count, 100.0 * strategy)
        strategy = sampling_strategy_dict
    
    smote = SMOTE(sampling_strategy=strategy, random_state=RANDOM_STATE)
    X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
    elapsed = time.time() - t0
    
    log.info("SMOTE completed in %.2f seconds", elapsed)
    log.info("Original samples: %d → Resampled: %d", len(y_train), len(y_resampled))
    
    return X_resampled, y_resampled


def apply_undersampling(X_train, y_train, strategy=UNDERSAMPLE_RATIO):
    """Apply random undersampling to majority class."""
    log.info("Applying Random Under-sampling with strategy: %s", strategy)
    
    t0 = time.time()
    
    # For multi-class with float, use 'majority' or 'not minority'
    # to automatically undersample the majority class
    if isinstance(strategy, float):
        # Use 'not minority' to keep minority classes and undersample majority
        actual_strategy = 'not minority'
        log.info("Under-sampling majority class (Benign) to balance with attacks")
    else:
        actual_strategy = strategy
    
    rus = RandomUnderSampler(sampling_strategy=actual_strategy, random_state=RANDOM_STATE)
    X_resampled, y_resampled = rus.fit_resample(X_train, y_train)
    elapsed = time.time() - t0
    
    log.info("Under-sampling completed in %.2f seconds", elapsed)
    log.info("Original samples: %d → Resampled: %d", len(y_train), len(y_resampled))
    
    return X_resampled, y_resampled


def apply_hybrid(X_train, y_train, smote_strategy=SMOTE_STRATEGY, under_strategy=UNDERSAMPLE_RATIO):
    """Apply SMOTE first, then undersample majority class."""
    log.info("Applying Hybrid approach (SMOTE + Under-sampling)")
    
    # Step 1: Oversample minority classes with SMOTE
    X_temp, y_temp = apply_smote(X_train, y_train, strategy=smote_strategy)
    
    # Step 2: Undersample majority class
    X_balanced, y_balanced = apply_undersampling(X_temp, y_temp, strategy=under_strategy)
    
    return X_balanced, y_balanced


def apply_smoteenn(X_train, y_train):
    """Apply SMOTEENN (SMOTE + Edited Nearest Neighbors)."""
    log.info("Applying SMOTEENN (this may take several minutes...)")
    
    t0 = time.time()
    smoteenn = SMOTEENN(random_state=RANDOM_STATE)
    X_resampled, y_resampled = smoteenn.fit_resample(X_train, y_train)
    elapsed = time.time() - t0
    
    log.info("SMOTEENN completed in %.2f seconds", elapsed)
    log.info("Original samples: %d → Resampled: %d", len(y_train), len(y_resampled))
    
    return X_resampled, y_resampled


def save_balanced_data(X_balanced, y_balanced, output_dir=OUTPUT_DIR):
    """Save balanced dataset."""
    os.makedirs(output_dir, exist_ok=True)
    
    log.info("Saving balanced dataset to: %s", output_dir)
    
    # Convert to DataFrame for parquet
    X_df = pd.DataFrame(X_balanced) if not isinstance(X_balanced, pd.DataFrame) else X_balanced
    y_df = pd.DataFrame(y_balanced, columns=['label_multiclass'])
    
    X_path = os.path.join(output_dir, "X_train_balanced.parquet")
    y_path = os.path.join(output_dir, "y_multiclass_train_balanced.parquet")
    
    X_df.to_parquet(X_path, compression="snappy", index=False)
    y_df.to_parquet(y_path, compression="snappy", index=False)
    
    log.info("Saved X_train_balanced.parquet: %s", X_df.shape)
    log.info("Saved y_multiclass_train_balanced.parquet: %s", y_df.shape)


def save_report(original_dist, balanced_dist, strategy_name, output_dir=OUTPUT_DIR):
    """Save balancing report."""
    report_path = os.path.join(output_dir, "balancing_report.txt")
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("="*80 + "\n")
        f.write("  CLASS BALANCING REPORT\n")
        f.write("="*80 + "\n\n")
        f.write(f"Strategy Applied: {strategy_name}\n")
        f.write(f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("ORIGINAL DISTRIBUTION:\n")
        f.write("-"*80 + "\n")
        for class_id, count in sorted(original_dist.items()):
            f.write(f"  Class {class_id}: {count:>10,}\n")
        f.write(f"  Total:    {sum(original_dist.values()):>10,}\n\n")
        
        f.write("BALANCED DISTRIBUTION:\n")
        f.write("-"*80 + "\n")
        for class_id, count in sorted(balanced_dist.items()):
            f.write(f"  Class {class_id}: {count:>10,}\n")
        f.write(f"  Total:    {sum(balanced_dist.values()):>10,}\n\n")
        
        f.write("CHANGE SUMMARY:\n")
        f.write("-"*80 + "\n")
        for class_id in sorted(original_dist.keys()):
            orig = original_dist.get(class_id, 0)
            bal = balanced_dist.get(class_id, 0)
            change = bal - orig
            change_pct = 100.0 * change / orig if orig > 0 else 0
            f.write(f"  Class {class_id}: {orig:>10,} -> {bal:>10,} ({change:>+10,}, {change_pct:>+7.1f}%)\n")
        
        f.write("\n" + "="*80 + "\n")
    
    log.info("Saved balancing report: %s", report_path)


# ---------------------------------------------------------------------------
# Main Pipeline
# ---------------------------------------------------------------------------

def run_balancing():
    """Main pipeline for class balancing."""
    
    # 1. Load data
    X_train, y_train = load_training_data()
    id_to_name = load_label_mapping()
    
    # 2. Show original distribution
    print_class_distribution(y_train, "ORIGINAL Class Distribution", id_to_name)
    original_dist = Counter(y_train)
    
    # 3. Apply balancing strategy
    log.info("Applying balancing strategy: %s", STRATEGY)
    
    if STRATEGY == 'smote':
        X_balanced, y_balanced = apply_smote(X_train, y_train)
        strategy_name = f"SMOTE (strategy={SMOTE_STRATEGY})"
    
    elif STRATEGY == 'undersample':
        X_balanced, y_balanced = apply_undersampling(X_train, y_train)
        strategy_name = f"Random Under-sampling (ratio={UNDERSAMPLE_RATIO})"
    
    elif STRATEGY == 'hybrid':
        X_balanced, y_balanced = apply_hybrid(X_train, y_train)
        strategy_name = f"Hybrid (SMOTE {SMOTE_STRATEGY} + Undersample {UNDERSAMPLE_RATIO})"
    
    elif STRATEGY == 'smoteenn':
        X_balanced, y_balanced = apply_smoteenn(X_train, y_train)
        strategy_name = "SMOTEENN"
    
    else:
        raise ValueError(f"Unknown strategy: {STRATEGY}")
    
    # 4. Show balanced distribution
    print_class_distribution(y_balanced, "BALANCED Class Distribution", id_to_name)
    balanced_dist = Counter(y_balanced)
    
    # 5. Calculate improvement metrics
    print("\n" + "="*80)
    print("  BALANCING IMPACT ANALYSIS")
    print("="*80)
    print(f"{'Metric':<40} {'Before':>15} {'After':>15} {'Change':>10}")
    print("-"*80)
    
    total_before = len(y_train)
    total_after = len(y_balanced)
    print(f"{'Total Samples':<40} {total_before:>15,} {total_after:>15,} {total_after-total_before:>+10,}")
    
    # Count rare classes (< 1000 samples originally)
    rare_before = sum(1 for c in original_dist.values() if c < 1000)
    rare_after = sum(1 for c in balanced_dist.values() if c < 1000)
    print(f"{'Rare Classes (< 1000 samples)':<40} {rare_before:>15} {rare_after:>15} {rare_after-rare_before:>+10}")
    
    # Minimum class size
    min_before = min(original_dist.values())
    min_after = min(balanced_dist.values())
    print(f"{'Minimum Class Size':<40} {min_before:>15,} {min_after:>15,} {min_after-min_before:>+10,}")
    
    # Maximum class size
    max_before = max(original_dist.values())
    max_after = max(balanced_dist.values())
    print(f"{'Maximum Class Size':<40} {max_before:>15,} {max_after:>15,} {max_after-max_before:>+10,}")
    
    # Imbalance ratio (max/min)
    ratio_before = max_before / min_before
    ratio_after = max_after / min_after
    print(f"{'Imbalance Ratio (Max/Min)':<40} {ratio_before:>15.1f} {ratio_after:>15.1f} {ratio_after-ratio_before:>+10.1f}")
    
    print("="*80 + "\n")
    
    # 6. Save balanced dataset
    save_balanced_data(X_balanced, y_balanced)
    
    # 7. Save report
    save_report(original_dist, balanced_dist, strategy_name)
    
    # 8. Summary
    print("\n" + "="*80)
    print("  BALANCING COMPLETE")
    print("="*80)
    print(f"  Strategy:  {strategy_name}")
    print(f"  Original:  {total_before:>10,} samples")
    print(f"  Balanced:  {total_after:>10,} samples ({100*total_after/total_before:.1f}% of original)")
    print(f"  Reduction in imbalance: {ratio_before:.1f}x → {ratio_after:.1f}x")
    print("="*80 + "\n")
    
    log.info("Balanced training data ready for model training!")


# ---------------------------------------------------------------------------
# Entry Point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    run_balancing()
