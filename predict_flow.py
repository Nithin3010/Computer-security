"""
predict_flow.py
---------------

Real-time inference script for the intrusion detection system.

Loads the trained XGBoost model and predicts the attack type for one
or more network flows based on the 78 extracted features.

Usage
-----
# Use built-in sample flow (demo mode)
python predict_flow.py

# Predict from a JSON file (single dict or list of dicts)
python predict_flow.py --json path/to/flow.json

# Predict from a CSV file (one row per flow, header row required)
python predict_flow.py --csv path/to/flows.csv
"""

import argparse
import json
import logging
import os
import sys

import joblib
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

MODEL_PATH   = os.path.join("models",        "xgboost_balanced_model.pkl")  # Using BALANCED model
SCALER_PATH  = os.path.join("processed_ml",  "scaler.pkl")
MAPPING_PATH = os.path.join("processed_ml",  "attack_label_mapping.json")

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

def load_artifacts() -> tuple:
    """Load model, scaler, and label mapping. Returns (clf, scaler, id_to_label)."""
    log.info("Loading model   : %s", MODEL_PATH)
    clf = joblib.load(MODEL_PATH)

    log.info("Loading scaler  : %s", SCALER_PATH)
    scaler = joblib.load(SCALER_PATH)

    log.info("Loading mapping : %s", MAPPING_PATH)
    with open(MAPPING_PATH, encoding="utf-8") as fh:
        name_to_id: dict = json.load(fh)

    id_to_label = {int(v): k for k, v in name_to_id.items()}
    log.info("Loaded %d attack classes.", len(id_to_label))
    return clf, scaler, id_to_label


def get_feature_names(scaler) -> list[str]:
    """Return ordered feature names from the fitted scaler."""
    if hasattr(scaler, "feature_names_in_"):
        return list(scaler.feature_names_in_)
    # Fallback: generic names based on number of scaler features
    n = scaler.n_features_in_
    log.warning("Scaler has no feature_names_in_; using generic names f0…f%d.", n - 1)
    return [f"f{i}" for i in range(n)]


# ---------------------------------------------------------------------------
# Input handling
# ---------------------------------------------------------------------------

def _dict_to_df(flow: dict, feature_names: list[str]) -> pd.DataFrame:
    """Convert a single flow dict to a properly ordered single-row DataFrame."""
    missing = [f for f in feature_names if f not in flow]
    if missing:
        log.warning(
            "%d feature(s) missing from input — filling with 0: %s",
            len(missing), missing[:10],
        )
    row = {f: flow.get(f, 0.0) for f in feature_names}
    return pd.DataFrame([row], columns=feature_names)


def load_from_json(path: str, feature_names: list[str]) -> pd.DataFrame:
    """Load one or more flows from a JSON file."""
    with open(path, encoding="utf-8") as fh:
        data = json.load(fh)

    if isinstance(data, dict):
        data = [data]
    if not isinstance(data, list):
        raise ValueError("JSON file must contain a dict or a list of dicts.")

    log.info("Loaded %d flow(s) from JSON: %s", len(data), path)
    return pd.concat([_dict_to_df(d, feature_names) for d in data], ignore_index=True)


def load_from_csv(path: str, feature_names: list[str]) -> pd.DataFrame:
    """Load flows from a CSV file (must have a header row)."""
    df = pd.read_csv(path)
    log.info("Loaded %d flow(s) from CSV: %s", len(df), path)

    # Drop non-feature columns silently, fill missing features with 0
    for f in feature_names:
        if f not in df.columns:
            df[f] = 0.0
    return df[feature_names].reset_index(drop=True)


def make_sample_flow(feature_names: list[str]) -> pd.DataFrame:
    """
    Create a minimal demo flow using zero-filled features.

    In a real deployment this would come from a network capture
    tool (e.g. CICFlowMeter).
    """
    log.info("No input provided — using a zero-filled demo flow.")
    flow = {f: 0.0 for f in feature_names}
    return _dict_to_df(flow, feature_names)


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def preprocess(df: pd.DataFrame, scaler) -> np.ndarray:
    """Replace inf/NaN with 0, then apply the training scaler."""
    df = df.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return scaler.transform(df).astype(np.float32)


def predict(clf, X_scaled: np.ndarray, id_to_label: dict) -> list[dict]:
    """
    Run inference and return a list of result dicts with:
        predicted_id, predicted_label, confidence
    """
    raw_probs = clf.predict_proba(X_scaled)   # shape (n_flows, n_classes)
    pred_ids  = np.argmax(raw_probs, axis=1)
    confidences = raw_probs[np.arange(len(pred_ids)), pred_ids]

    results = []
    for pred_id, conf in zip(pred_ids, confidences):
        label = id_to_label.get(int(pred_id), f"class_{pred_id}")
        results.append(dict(
            predicted_id=int(pred_id),
            predicted_label=label,
            confidence=float(conf),
        ))
    return results


# ---------------------------------------------------------------------------
# Display
# ---------------------------------------------------------------------------

def print_results(results: list[dict]) -> None:
    sep = "-" * 50
    print(f"\n{'='*50}")
    print("  Intrusion Detection — Prediction Results")
    print(f"{'='*50}")
    for i, r in enumerate(results, 1):
        tag = "(BENIGN)" if r["predicted_label"].upper() == "BENIGN" else "(ATTACK)"
        print(f"\n  Flow #{i}")
        print(sep)
        print(f"  Predicted Attack Type : {r['predicted_label']}  {tag}")
        print(f"  Confidence            : {r['confidence']:.4f}  ({r['confidence']*100:.2f} %)")
    print(f"\n{'='*50}\n")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Real-time intrusion detection inference for network flows.",
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--json", metavar="PATH",
        help="Path to a JSON file containing a flow dict or list of dicts.",
    )
    group.add_argument(
        "--csv", metavar="PATH",
        help="Path to a CSV file with a header row (one flow per row).",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run(args: argparse.Namespace) -> None:
    # 1. Load artefacts ------------------------------------------------------
    clf, scaler, id_to_label = load_artifacts()
    feature_names = get_feature_names(scaler)
    log.info("Expecting %d features.", len(feature_names))

    # 2. Load input ----------------------------------------------------------
    if args.json:
        if not os.path.exists(args.json):
            log.error("JSON file not found: %s", args.json)
            sys.exit(1)
        df = load_from_json(args.json, feature_names)
    elif args.csv:
        if not os.path.exists(args.csv):
            log.error("CSV file not found: %s", args.csv)
            sys.exit(1)
        df = load_from_csv(args.csv, feature_names)
    else:
        df = make_sample_flow(feature_names)

    # 3. Preprocess ----------------------------------------------------------
    X_scaled = preprocess(df, scaler)

    # 4. Predict -------------------------------------------------------------
    results = predict(clf, X_scaled, id_to_label)

    # 5. Display -------------------------------------------------------------
    print_results(results)


if __name__ == "__main__":
    run(parse_args())
