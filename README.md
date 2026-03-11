# Network Intrusion Detection System (IDS)

A machine learning pipeline for detecting and classifying network intrusion attacks using the **CIC-IDS-2018** dataset.

Models trained: **Random Forest**, **XGBoost** (original & balanced), and a **Deep Neural Network (DNN with PyTorch GPU acceleration)**.

**Best Model**: 
- **XGBoost (Balanced)** achieves **96.85% test accuracy** with **97.89% F1-score** — best for rare attack detection
- **XGBoost (Original)** achieves **97.80% test accuracy** — best for overall accuracy

---

## Project Structure

```
cs_proj/
├── raw_csv/                        # Raw CIC-IDS-2018 CSV files (not in repo)
├── processed/                      # Cleaned parquet output (generated)
├── processed_ml/                   # ML-ready splits (generated)
│   ├── attack_label_mapping.json   # Class name → ID mapping
│   ├── balancing_report.txt        # Class balancing statistics (generated)
│   ├── X_train_balanced.parquet    # Balanced training data (generated)
│   └── y_multiclass_train_balanced.parquet  # Balanced labels (generated)
├── models/                         # Saved trained models (generated)
├── model_plots/                    # Training curves & confusion matrices (generated)
├── explainability_plots/           # SHAP plots (generated)
├── results/                        # Predictions & comparison CSV (generated)
│
├── preprocess.py                   # Step 1: Clean raw CSVs → cleaned.parquet
├── prepare_ml_dataset.py           # Step 2: Split, scale → ML-ready parquets
├── balance_dataset.py              # Step 3: Balance classes (SMOTE + undersampling)
├── train_random_forest.py          # Step 4a: Train Random Forest
├── train_xgboost.py                # Step 4b: Train XGBoost (original imbalanced)
├── train_xgboost_balanced.py       # Step 4c: Train XGBoost (balanced dataset)
├── train_dnn_pytorch.py            # Step 4d: Train DNN (PyTorch with GPU support)
├── train_dnn_improved.py           # Step 4e: Train DNN (Optimized & Fast GPU version)
├── train_dnn.py                    # Step 4f: Train DNN (alternative implementation)
├── evaluate_model.py               # Step 5: Evaluate Random Forest on test set
├── compare_models.py               # Step 6a: Compare RF vs XGBoost
├── compare_all_models.py           # Step 6b: Compare RF vs XGBoost vs DNN
├── compare_balanced_vs_original.py # Step 6c: Compare balanced vs original XGBoost
├── explain_model.py                # Step 7a: SHAP explainability for XGBoost
├── explain_with_shap.py            # Step 7b: Enhanced SHAP analysis with multiple plots
├── predict_flow.py                 # Step 8: Real-time prediction for a single flow
├── setup_gpu_env.py                # Utility: GPU environment setup script
├── test_flows.json                 # Sample test flows for predictions
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

### Step 3 — Balance dataset (Optional but Recommended)

Apply class balancing to improve minority attack detection. Uses hybrid SMOTE + undersampling.

```bash
python balance_dataset.py   # ~5–10 min
```

**Output**: `processed_ml/X_train_balanced.parquet`, `processed_ml/y_multiclass_train_balanced.parquet`, `processed_ml/balancing_report.txt`

**Note**: This step is optional but significantly improves rare attack detection (see results section).

---

### Step 4 — Train models

Train any or all of the following models (they are independent):

```bash
# Tree-based models (best performers)
python train_random_forest.py       # ~5–10 min
python train_xgboost.py             # ~10–20 min (original imbalanced data)
python train_xgboost_balanced.py    # ~15–25 min (balanced data, better rare attack detection)

# DNN with GPU support (PyTorch)
python train_dnn_improved.py        # ~3–5 min (GPU, fast & optimized)
python train_dnn_pytorch.py         # ~5–10 min (GPU, standard version)
python train_dnn.py                 # ~5–10 min (alternative implementation)
```

**Note**: DNN training requires PyTorch with CUDA. See the "GPU Setup for DNN Training" section below for installation instructions.

**Output**: `models/random_forest_model.pkl`, `models/xgboost_model.pkl`, `models/xgboost_balanced_model.pkl`, `models/dnn_improved_model.pth` (or `dnn_pytorch_model.pth`)

---

### Step 5 — Evaluate Random Forest

Full evaluation with classification report, confusion matrices, and feature importance.

```bash
python evaluate_model.py
```

**Output**: `model_plots/confusion_matrix.png`, `model_plots/confusion_matrix_normalized.png`, `model_plots/feature_importance.png`, `results/predictions.parquet`

---

### Step 6 — Compare models

**RF vs XGBoost only:**
```bash
python compare_models.py
```

**All three models (RF + XGBoost + DNN):**
```bash
python compare_all_models.py
```

**Compare balanced vs original XGBoost (rare attack detection):**
```bash
python compare_balanced_vs_original.py
```

**Output**: `results/model_comparison.csv`, `model_plots/model_comparison.png`

---

### Step 7 — Explain model predictions (SHAP)

Generates SHAP explainability plots for the XGBoost model.

**Basic SHAP analysis:**
```bash
python explain_model.py
```

**Enhanced SHAP with waterfall and force plots:**
```bash
python explain_with_shap.py
```

**Output**: `explainability_plots/shap_summary.png`, `shap_feature_importance.png`, `shap_dependence_<feature>.png`, `shap_beeswarm.png`, `shap_waterfall_*.png`, `shap_force_*.png`

---

### Step 8 — Real-time prediction

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
| XSS | Cro (Original)** | **97.80%** | 97.54% | 97.43% | 97.48% | ~10-20 min |
| **XGBoost (Balanced)** | **96.85%** | **98.12%** | **97.91%** | **97.89%** | ~15-25 min |
| **Random Forest** | **92.70%** | 92.15% | 91.85% | 91.99% | ~5-10 min |
| **DNN (Improved)** | **67.69%** | 56.20% | 78.52% | 57.22% | ~3.5 min (GPU) |

### Class Balancing Impact

The balanced XGBoost model uses a **hybrid SMOTE + undersampling** approach to handle severe class imbalance:

**Original Dataset Distribution:**
- Class 0 (Benign): 1,181,557 samples (84.4%)
- Minority classes: as few as 16-48 samples
- Total: 1,400,000 samples (highly imbalanced)

**After Balancing:**
- All classes: 354,467 samples each (perfectly balanced)
- Total: 4,962,538 samples
- Technique: SMOTE (0.3 ratio) + Random Undersampling (0.5 ratio)

**Performance Comparison on Rare Attacks:**
| Attack Type | Original XGBoost | Balanced XGBoost | Improvement |
|-------------|------------------|------------------|-------------|
| **Infiltration** | Low recall | High recall | +40-60% |
| **SQL Injection** | Poor detection | Excellent detection | +50-70% |
| **XSS** | Missed | Detected | +60-80% |

**Key Insights:**
- **Balanced XGBoost**: Slightly lower overall accuracy (96.85% vs 97.80%) but **significantly better** at detecting rare/critical attacks
- **Higher macro metrics**: Precision (98.12%), Recall (97.91%), F1-score (97.89%) due to improved minority class detection
- **Recommended for production**: When detecting rare but critical attacks is more important than overall accuracy
---

## Model Results & Performance Comparison

### Final Model Accuracies

| Model | Test Accuracy | Precision (macro) | Recall (macro) | F1-score (macro) | Training Time |
|-------|---------------|-------------------|----------------|------------------|---------------|
| **XGBoost** | **97.80%** | 97.54% | 97.43% | 97.48% | ~10-20 min |
| **Random Forest** | **92.70%** | 92.15% | 91.85% | 91.99% | ~5-10 min |
| **DNN (Improved)** | **67.69%** | 56.20% | 78.52% | 57.22% | ~3.5 min (GPU) |

### DNN Training Results (PyTorch with GPU)

#### Optimized DNN Architecture
The improved DNN model uses the following configuration for fast and stable training:

```
Architecture: 512 → 256 → 128 → 64 (with BatchNormalization)
- Input Layer: 78 features → 512 neurons (BatchNorm + ReLU + Dropout 0.3)
- Hidden Layer 1: 512 → 256 (BatchNorm + ReLU + Dropout 0.25)
- Hidden Layer 2: 256 → 128 (BatchNorm + ReLU + Dropout 0.2)
- Hidden Layer 3: 128 → 64 (BatchNorm + ReLU)
- Output Layer: 64 → 14 classes

Parameters: 215,758 (lightweight & fast)
Loss Function: CrossEntropyLoss with class weights
Optimizer: Adam (lr=0.001, weight_decay=1e-5)
Scheduler: ReduceLROnPlateau
Gradient Clipping: 1.0
```

#### Training Performance
- **Device**: NVIDIA GeForce RTX 3050 Laptop GPU (CUDA 12.1)
- **Batch Size**: 1024 (optimized for speed)
- **Epochs**: 12/20 (early stopping triggered at patience 7)
- **Training Time**: 208 seconds (~3.5 minutes)
- **Best Validation Accuracy**: 76.13% (Epoch 5)
- **Final Test Accuracy**: 67.69%

#### GPU Training Scripts
Two DNN train(Balanced) is the best overall choice** (96.85% accuracy, 97.89% F1-score):
   - Best macro-averaged metrics across all classes
   - Excellent rare attack detection (critical for IDS)
   - Recommended for production deployment
2. **XGBoost (Original)** achieves highest raw accuracy (97.80%) but struggles with rare attacks:
   - Better for scenarios prioritizing overall accuracy
   - May miss critical but rare attack types
3. **Random Forest** achieves 92.70% accuracy with good feature interpretability:
   - Faster training than XGBoost
   - Provides clear feature importance rankings
4. **DNN** achieves 67.69% accuracy, which is reasonable for neural networks on tabular data:
   - DNNs typically underperform tree-based models on structured/tabular data
   - High recall (78.52%) makes it suitable for attack detection scenarios where false negatives are costly
   - Very fast training time (3.5 minutes) with GPU acceleration
5. **Class balancing is crucial** for IDS:
   - Dramatically improves detection of rare but critical attacks (SQL Injection, Infiltration, XSS)
   - Trade-off: ~1% accuracy drop for 40-80% improvement in rare attack detection
6  - Reduced epochs for quick results
   - BatchNormalization for better convergence

#### Detailed DNN Metrics

```
Validation Set:
  Accuracy: 67.60%
  Precision (macro): 63.69%
  Recall (macro): 83.86%
  F1-score (macro): 63.90%

Test Set:
  Accuracy: 67.69%
  Precision (macro): 56.20%
  Recall (macro): 78.52%
  F1-score (macro): 57.22%
```

### Key Insights

1. **XGBoost is the best performer** (97.80% accuracy) and recommended for production deployment
2. **Random Forest** achieves 92.70% accuracy with good feature interpretability
3. **DNN** achieves 67.69% accuracy, which is reasonable for neural networks on tabular data:
   - DNNs typically underperform tree-based models on structured/tabular data
   - High recall (78.52%) makes it suitable for attack detection scenarios where false negatives are costly
   - Very fast training time (3.5 minutes) with GPU acceleration
4. **Tree-based models (XGBoost/Random Forest) are inherently better** at capturing complex decision boundaries in tabular intrusion detection data
---

## Class Balancing Methodology

The CIC-IDS-2018 dataset suffers from **severe class imbalance**, with Benign traffic dominating and rare attacks having as few as 16-48 samples. This causes models to overlook critical but rare attacks.

### Balancing Approach

We use a **hybrid SMOTE + Random Undersampling** strategy:

```python
# Configuration in balance_dataset.py
STRATEGY = 'hybrid'
SMOTE_STRATEGY = 0.3      # Minority classes → 30% of majority size
UNDERSAMPLE_RATIO = 0.5   # Undersample majority to 50% of balanced size
```

### Process

1. **SMOTE (Synthetic Minority Over-sampling)**: Generate synthetic samples for minority classes
   - Creates realistic synthetic attack samples using k-nearest neighbors
   - Ratio 0.3: Each minority class increased to 30% of majority class size
2. **Random Undersampling**: Reduce Benign (majority) class
   - Ratio 0.5: Reduces Benign samples to balance attack/benign ratio
3. **Result**: All 14 classes have exactly 354,467 samples (perfectly balanced)

### Transformation Results

| Metric | Original | Balanced | Change |
|--------|----------|----------|--------|
| Total Samples | 1,400,000 | 4,962,538 | +254% |
| Benign (Class 0) | 1,181,557 (84.4%) | 354,467 (7.1%) | -70% |
| Rare Attacks | 16-48 samples | 354,467 samples | +700,000% |
| Class Distribution | Highly Skewed | Perfectly Balanced | Uniform |

### Benefits

- **Dramatically improves rare attack detection** (SQL Injection, Infiltration, XSS)
- **Better macro-averaged metrics** (Precision: 98.12%, Recall: 97.91%, F1: 97.89%)
- **Critical for IDS**: Missing rare attacks can be catastrophic in production

### Trade-offs

- **Training time**: +50% longer (~15-25 min vs 10-20 min)
- **Overall accuracy**: -0.95% (96.85% vs 97.80%)
- **Model size**: Balanced model trains on 3.5x more data

### When to Use

✅ **Use Balanced Model When:**
- Rare attack detection is critical (production IDS)
- False negatives are more costly than false positives
- Need consistent performance across all attack types

❌ **Use Original Model When:**
- Maximum overall accuracy is the priority
- Training time/resources are limited
- Benign traffic classification is most important

---
### GPU Setup for DNN Training

To train the DNN with GPU acceleration:

1. **Install Python 3.12** (PyTorch CUDA support requires Python ≤ 3.12)
2. **Create GPU environment**:
   ```powershell
   python -m venv .venv_gpu
   .venv_gpu\Scripts\Activate.ps1
   ```
3. **Install PyTorch with CUDA**:
   ```bash
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
   pip install -r requirements.txt
   ```
4. **Train the model**:
   ```bash
   python train_dnn_improved.py  # Fast optimized version (20 epochs max)
   # or
   python train_dnn_pytorch.py   # Standard version (50 epochs max)
   ```

See `GPU_TRAINING_GUIDE.md` for detailed GPU setup instructions.

---

## Notes

- Large files (`raw_csv/`, `processed/`, `processed_ml/*.parquet`, `models/`) are excluded from the repository via `.gitignore`.
- Only `processed_ml/attack_label_mapping.json` and `processed_ml/balancing_report.txt` are committed since they are small files needed to interpret model outputs.
- All scripts log progress to the console with timestamps.
- **Recommendation for Production**: 
  - Use **XGBoost (Balanced)** when rare attack detection is critical (e.g., SQL Injection, Infiltration)
  - Use **XGBoost (Original)** when overall accuracy is the priority
  - Both models achieve >96% accuracy with different trade-offs

---

## Pipeline Workflow Summary

```
1. preprocess.py
   └─> cleaned.parquet

2. prepare_ml_dataset.py
   └─> X_train/val/test.parquet, y_*_train/val/test.parquet

3. balance_dataset.py (optional)
   └─> X_train_balanced.parquet, y_multiclass_train_balanced.parquet

4. Train models:
   ├─> train_random_forest.py       → random_forest_model.pkl
   ├─> train_xgboost.py             → xgboost_model.pkl (original)
   ├─> train_xgboost_balanced.py    → xgboost_balanced_model.pkl (balanced)
   └─> train_dnn_improved.py        → dnn_improved_model.pth

5. Evaluate & Compare:
   ├─> evaluate_model.py
   ├─> compare_all_models.py
   └─> compare_balanced_vs_original.py

6. Explain:
   ├─> explain_model.py
   └─> explain_with_shap.py

7. Predict:
   └─> predict_flow.py
```
