# Network Intrusion Detection System (IDS)

A machine learning pipeline for detecting and classifying network intrusion attacks using the **CIC-IDS-2018** dataset.

Models trained: **Random Forest**, **XGBoost**, and a **Deep Neural Network (DNN)**.

---

## Project Structure

```
cs_proj/
├── raw_csv/                        # Raw CIC-IDS-2018 CSV files (not in repo)
├── processed/                      # Cleaned parquet output (generated)
├── processed_ml/                   # ML-ready splits (generated)
│   └── attack_label_mapping.json   # Class name → ID mapping
├── models/                         # Saved trained models (generated)
├── model_plots/                    # Training curves & confusion matrices (generated)
├── explainability_plots/           # SHAP plots (generated)
├── results/                        # Predictions & comparison CSV (generated)
│
├── preprocess.py                   # Step 1: Clean raw CSVs → cleaned.parquet
├── prepare_ml_dataset.py           # Step 2: Split, scale → ML-ready parquets
├── train_random_forest.py          # Step 3a: Train Random Forest
├── train_xgboost.py                # Step 3b: Train XGBoost
├── train_dnn.py                    # Step 3c: Train Deep Neural Network
├── evaluate_model.py               # Step 4: Evaluate Random Forest on test set
├── compare_models.py               # Step 5a: Compare RF vs XGBoost
├── compare_all_models.py           # Step 5b: Compare RF vs XGBoost vs DNN
├── explain_model.py                # Step 6: SHAP explainability for XGBoost
├── predict_flow.py                 # Step 7: Real-time prediction for a single flow
│
├── requirements.txt
└── README.md
```

---

## Requirements

- Python **3.10+**
- The `raw_csv/` folder containing the CIC-IDS-2018 CSV files

---

## Setup

### 1. Clone the repository

```bash
git clone https://github.com/Nithin3010/Computer-security.git
cd Computer-security
```

### 2. Create and activate a virtual environment

**Windows:**
```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

**Linux / macOS:**
```bash
python -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Add the dataset

Place the raw CIC-IDS-2018 CSV files inside the `raw_csv/` folder:

```
raw_csv/
    02-14-2018.csv
    02-15-2018.csv
    02-16-2018.csv
    02-20-2018.csv
    02-21-2018.csv
    02-22-2018.csv
    02-23-2018.csv
    02-28-2018.csv
    03-01-2018.csv
    03-02-2018.csv
```

> The dataset can be downloaded from the [CIC-IDS-2018 official page](https://www.unb.ca/cic/datasets/ids-2018.html).

---

## Running the Pipeline

Run the scripts **in order**:

### Step 1 — Preprocess raw data

Cleans the raw CSVs, removes duplicates, fixes infinite/NaN values, and saves a cleaned parquet file.

```bash
python preprocess.py
```

Output: `processed/cleaned.parquet`

---

### Step 2 — Prepare ML dataset

Encodes labels, performs a stratified 70/15/15 train-val-test split, and scales features.

```bash
python prepare_ml_dataset.py
```

Output: `processed_ml/` (feature matrices, label files, scaler, label mapping)

---

### Step 3 — Train models

Train any or all of the following models (they are independent):

```bash
python train_random_forest.py   # ~5–10 min
python train_xgboost.py         # ~10–20 min
python train_dnn.py             # ~5–15 min (GPU recommended)
```

Output: `models/random_forest_model.pkl`, `models/xgboost_model.pkl`, `models/dnn_model.keras`

---

### Step 4 — Evaluate Random Forest

Full evaluation with classification report, confusion matrices, and feature importance.

```bash
python evaluate_model.py
```

Output: `model_plots/confusion_matrix.png`, `model_plots/confusion_matrix_normalized.png`, `model_plots/feature_importance.png`, `results/predictions.parquet`

---

### Step 5 — Compare models

**RF vs XGBoost only:**
```bash
python compare_models.py
```

**All three models (RF + XGBoost + DNN):**
```bash
python compare_all_models.py
```

Output: `results/model_comparison.csv`, `model_plots/model_comparison.png`

---

### Step 6 — Explain model predictions (SHAP)

Generates SHAP explainability plots for the XGBoost model.

```bash
python explain_model.py
```

Output: `explainability_plots/shap_summary.png`, `shap_feature_importance.png`, `shap_dependence_<feature>.png`

---

### Step 7 — Real-time prediction

Predict the attack type for a single network flow.

```bash
# Demo mode (zero-filled sample)
python predict_flow.py

# From a JSON file
python predict_flow.py --json my_flow.json

# From a CSV file (one flow per row)
python predict_flow.py --csv my_flows.csv
```

---

## Metrics Reported

| Metric | Averaging |
|---|---|
| Accuracy | — |
| Precision | Macro |
| Recall | Macro |
| F1-score | Macro |

---

## Attack Classes

The dataset contains the following attack categories:

| Label | Attack Type |
|---|---|
| BENIGN | Normal traffic |
| Bot | Botnet activity |
| Brute Force | SSH / FTP brute force |
| DDoS | Distributed Denial of Service |
| DoS attacks-GoldenEye | DoS variant |
| DoS attacks-Hulk | DoS variant |
| DoS attacks-SlowHTTPTest | DoS variant |
| DoS attacks-Slowloris | DoS variant |
| Infilteration | Network infiltration |
| SQL Injection | Web attack |
| FTP-BruteForce | FTP brute force |
| SSH-Bruteforce | SSH brute force |
| XSS | Cross-site scripting |
| ... | (varies by dataset version) |

---

## Notes

- Large files (`raw_csv/`, `processed/`, `processed_ml/*.parquet`, `models/`) are excluded from the repository via `.gitignore`.
- Only `processed_ml/attack_label_mapping.json` is committed since it is a small file needed to interpret model outputs.
- All scripts log progress to the console with timestamps.
