"""
prepare_ml_dataset.py
---------------------

Purpose
-------
Prepare the cleaned IDS dataset for machine learning training.

The input dataset was produced by preprocess.py and stored as:

    processed/cleaned.parquet

It contains:
    • 78 numeric network flow features
    • label_binary  (0 = BENIGN, 1 = ATTACK)
    • attack_type   (string attack class)

This script prepares ML-ready datasets for both:

    1. Binary classification (benign vs attack)
    2. Multi-class attack classification

Output:
    processed_ml/
        X_train.parquet, X_val.parquet, X_test.parquet
        y_binary_train.parquet, y_binary_val.parquet, y_binary_test.parquet
        y_multiclass_train.parquet, y_multiclass_val.parquet, y_multiclass_test.parquet
        attack_label_mapping.json
        scaler.pkl
"""

import os
import json
import logging
import joblib

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

INPUT_PATH   = os.path.join("processed", "cleaned.parquet")
OUTPUT_DIR   = "processed_ml"

LABEL_BINARY    = "label_binary"
LABEL_ATTACK    = "attack_type"

# Sampling configuration (set to None to use full dataset)
# For systems with limited RAM, use 1000000 or 2000000
SAMPLE_SIZE = 2000000  # Use 2M rows instead of 6.75M to reduce memory usage

TRAIN_SIZE = 0.70   # 70 % train
VAL_SIZE   = 0.50   # 50 % of the 30 % remainder → 15 % overall
RANDOM_STATE = 42

SCALER_PATH  = os.path.join(OUTPUT_DIR, "scaler.pkl")
MAPPING_PATH = os.path.join(OUTPUT_DIR, "attack_label_mapping.json")

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
# Pipeline functions
# ---------------------------------------------------------------------------

def load_dataset(path: str = INPUT_PATH) -> pd.DataFrame:
    """Load the cleaned parquet dataset from *path*."""
    log.info("Loading dataset from: %s", path)
    df = pd.read_parquet(path)
    log.info("Loaded %d rows and %d columns.", len(df), df.shape[1])
    
    # Sample data if SAMPLE_SIZE is set (for memory-constrained systems)
    if SAMPLE_SIZE is not None and len(df) > SAMPLE_SIZE:
        log.info("Sampling %d rows from %d total (random sampling) ...", 
                 SAMPLE_SIZE, len(df))
        # Use simple random sampling to avoid memory issues
        df = df.sample(n=SAMPLE_SIZE, random_state=RANDOM_STATE).reset_index(drop=True)
        log.info("After sampling: %d rows", len(df))
    
    return df


def encode_attack_labels(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """
    Encode the *attack_type* column to integer class IDs using LabelEncoder.

    Returns
    -------
    df : DataFrame with a new column ``label_multiclass`` added (int32).
    mapping : dict  {class_name: class_id}
    """
    log.info("Encoding attack type labels …")
    le = LabelEncoder()
    # Removed df.copy() to reduce memory usage - modifying in place
    df["label_multiclass"] = le.fit_transform(df[LABEL_ATTACK]).astype(np.int32)

    mapping: dict = {cls: int(idx) for idx, cls in enumerate(le.classes_)}
    log.info("Found %d attack classes: %s", len(mapping), list(mapping.keys()))
    return df, mapping


def log_class_distribution(series: pd.Series, label: str = "attack_type") -> None:
    """Log the value counts and percentages for *series*."""
    counts = series.value_counts().sort_index()
    total  = len(series)
    log.info("Class distribution for '%s' (total=%d):", label, total)
    for cls, cnt in counts.items():
        log.info("    %-35s  %8d  (%.2f %%)", cls, cnt, 100.0 * cnt / total)


def split_dataset(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Stratified split into train / validation / test sets.

    Strategy
    --------
    First  split : 70 % train, 30 % temp   (stratify on attack_type)
    Second split : temp → 50/50            (stratify on attack_type)
                   → 15 % val, 15 % test overall

    Returns
    -------
    train_df, val_df, test_df
    """
    log.info("Splitting dataset (train=70 %%, val=15 %%, test=15 %%) …")

    train_df, temp_df = train_test_split(
        df,
        train_size=TRAIN_SIZE,
        random_state=RANDOM_STATE,
        stratify=df[LABEL_ATTACK],
    )

    val_df, test_df = train_test_split(
        temp_df,
        train_size=VAL_SIZE,
        random_state=RANDOM_STATE,
        stratify=temp_df[LABEL_ATTACK],
    )

    log.info(
        "Split sizes — train: %d | val: %d | test: %d",
        len(train_df), len(val_df), len(test_df),
    )
    return train_df, val_df, test_df


def clean_features(
    X_train: pd.DataFrame,
    X_val:   pd.DataFrame,
    X_test:  pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Sanitize feature matrices by removing infinite and NaN values.

    Strategy
    --------
    1. Replace +inf / -inf with NaN in all splits.
    2. Compute column medians on *X_train* only (no leakage).
    3. Fill NaN in every split using those training medians.

    Returns
    -------
    X_train_clean, X_val_clean, X_test_clean
    """
    log.info("Cleaning features: replacing inf / NaN values …")

    # Replace infinities with NaN (in-place to save memory)
    X_train.replace([np.inf, -np.inf], np.nan, inplace=True)
    X_val.replace([np.inf, -np.inf], np.nan, inplace=True)
    X_test.replace([np.inf, -np.inf], np.nan, inplace=True)

    # Compute medians from training split only
    train_medians = X_train.median()

    inf_count = X_train.isna().sum().sum()
    log.info("  NaN / inf cells replaced in train: %d", inf_count)

    X_train.fillna(train_medians, inplace=True)
    X_val.fillna(train_medians, inplace=True)
    X_test.fillna(train_medians, inplace=True)

    log.info("Cleaning complete.")
    return X_train, X_val, X_test


def scale_features(
    X_train: pd.DataFrame,
    X_val:   pd.DataFrame,
    X_test:  pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, StandardScaler]:
    """
    Fit a StandardScaler on *X_train*, transform all three splits.

    Scaler is fitted only on training data to prevent data leakage.
    Values are cast to float32 to reduce memory usage.

    Returns
    -------
    X_train_scaled, X_val_scaled, X_test_scaled, fitted_scaler
    """
    log.info("Fitting StandardScaler on training features …")
    scaler = StandardScaler()

    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train).astype(np.float32),
        columns=X_train.columns,
        index=X_train.index,
    )
    X_val_scaled = pd.DataFrame(
        scaler.transform(X_val).astype(np.float32),
        columns=X_val.columns,
        index=X_val.index,
    )
    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test).astype(np.float32),
        columns=X_test.columns,
        index=X_test.index,
    )

    log.info("Scaling complete.")
    return X_train_scaled, X_val_scaled, X_test_scaled, scaler


def save_datasets(
    X_train: pd.DataFrame,
    X_val:   pd.DataFrame,
    X_test:  pd.DataFrame,
    y_binary_train: pd.Series,
    y_binary_val:   pd.Series,
    y_binary_test:  pd.Series,
    y_mc_train: pd.Series,
    y_mc_val:   pd.Series,
    y_mc_test:  pd.Series,
    scaler:  StandardScaler,
    mapping: dict,
    out_dir: str = OUTPUT_DIR,
) -> None:
    """
    Persist all ML-ready artefacts to *out_dir*.

    Saved files
    -----------
    X_train / X_val / X_test             — feature matrices (Parquet, snappy)
    y_binary_*                            — binary labels    (Parquet, snappy)
    y_multiclass_*                        — multiclass labels(Parquet, snappy)
    scaler.pkl                            — fitted StandardScaler
    attack_label_mapping.json             — class-name → class-id mapping
    """
    os.makedirs(out_dir, exist_ok=True)
    log.info("Saving artefacts to: %s", out_dir)

    def _save_parquet(df: pd.DataFrame | pd.Series, name: str) -> None:
        path = os.path.join(out_dir, f"{name}.parquet")
        if isinstance(df, pd.Series):
            df = df.to_frame()
        df.to_parquet(path, compression="snappy", index=False)
        log.info("  Saved %s  (%s)", name + ".parquet", df.shape)

    # Feature matrices
    _save_parquet(X_train, "X_train")
    _save_parquet(X_val,   "X_val")
    _save_parquet(X_test,  "X_test")

    # Binary labels
    _save_parquet(y_binary_train, "y_binary_train")
    _save_parquet(y_binary_val,   "y_binary_val")
    _save_parquet(y_binary_test,  "y_binary_test")

    # Multi-class labels
    _save_parquet(y_mc_train, "y_multiclass_train")
    _save_parquet(y_mc_val,   "y_multiclass_val")
    _save_parquet(y_mc_test,  "y_multiclass_test")

    # Scaler
    joblib.dump(scaler, SCALER_PATH)
    log.info("  Saved scaler → %s", SCALER_PATH)

    # Label mapping
    with open(MAPPING_PATH, "w", encoding="utf-8") as fh:
        json.dump(mapping, fh, indent=4)
    log.info("  Saved label mapping → %s", MAPPING_PATH)


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run_pipeline() -> None:
    """End-to-end ML dataset preparation pipeline."""

    # 1. Load ----------------------------------------------------------------
    df = load_dataset()

    # 2. Log class distribution ----------------------------------------------
    log_class_distribution(df[LABEL_ATTACK])

    # 3. Identify feature columns --------------------------------------------
    non_feature_cols = {LABEL_BINARY, LABEL_ATTACK, "label_multiclass"}
    feature_cols = [c for c in df.columns if c not in non_feature_cols]
    log.info("Number of feature columns: %d", len(feature_cols))

    # 4. Encode multi-class labels -------------------------------------------
    df, mapping = encode_attack_labels(df)

    # 5. Stratified split ----------------------------------------------------
    train_df, val_df, test_df = split_dataset(df)

    # 6. Separate X / y ------------------------------------------------------
    X_train = train_df[feature_cols].astype(np.float32)
    X_val   = val_df[feature_cols].astype(np.float32)
    X_test  = test_df[feature_cols].astype(np.float32)

    y_binary_train = train_df[LABEL_BINARY].astype(np.int32)
    y_binary_val   = val_df[LABEL_BINARY].astype(np.int32)
    y_binary_test  = test_df[LABEL_BINARY].astype(np.int32)

    y_mc_train = train_df["label_multiclass"].astype(np.int32)
    y_mc_val   = val_df["label_multiclass"].astype(np.int32)
    y_mc_test  = test_df["label_multiclass"].astype(np.int32)

    # 7. Clean features (remove inf / NaN) ------------------------------------
    X_train, X_val, X_test = clean_features(X_train, X_val, X_test)

    # 8. Scale features ------------------------------------------------------
    X_train, X_val, X_test, scaler = scale_features(X_train, X_val, X_test)

    # 9. Save ----------------------------------------------------------------
    save_datasets(
        X_train, X_val, X_test,
        y_binary_train, y_binary_val, y_binary_test,
        y_mc_train, y_mc_val, y_mc_test,
        scaler, mapping,
    )

    # 10. Summary ------------------------------------------------------------
    total = len(df)
    print("\n" + "=" * 60)
    print("  ML Dataset Preparation — Summary")
    print("=" * 60)
    print(f"  Total samples   : {total:>10,}")
    print(f"  Feature columns : {len(feature_cols):>10,}")
    print(f"  Train samples   : {len(X_train):>10,}  ({100*len(X_train)/total:.1f} %)")
    print(f"  Val   samples   : {len(X_val):>10,}  ({100*len(X_val)/total:.1f} %)")
    print(f"  Test  samples   : {len(X_test):>10,}  ({100*len(X_test)/total:.1f} %)")
    print()
    print("  Attack class distribution (full dataset):")
    counts = df[LABEL_ATTACK].value_counts().sort_index()
    for cls, cnt in counts.items():
        print(f"    {cls:<35}  {cnt:>8,}  ({100*cnt/total:.2f} %)")
    print("=" * 60)
    print(f"  Outputs saved to: {os.path.abspath(OUTPUT_DIR)}")
    print("=" * 60 + "\n")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    run_pipeline()
