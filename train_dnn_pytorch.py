"""
train_dnn_pytorch.py
-------------------

Train a Deep Neural Network (DNN) for network intrusion detection using PyTorch.
Automatically uses GPU if available.

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
    models/dnn_pytorch_model.pth
    model_plots/dnn_training_loss.png
    model_plots/dnn_training_accuracy.png
"""

import logging
import os
import time

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import DataLoader, TensorDataset

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

INPUT_DIR   = "processed_ml"
MODEL_DIR   = "models"
PLOTS_DIR   = "model_plots"
MODEL_PATH  = os.path.join(MODEL_DIR, "dnn_pytorch_model.pth")

# Improved hyperparameters for better accuracy
BATCH_SIZE   = 512      # Smaller batch size for better generalization
EPOCHS       = 50       # More epochs for better convergence
PATIENCE     = 10       # More patience to avoid premature stopping
LEARNING_RATE = 0.0005  # Lower learning rate for finer optimization
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
# Device configuration (GPU/CPU)
# ---------------------------------------------------------------------------

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
log.info(f"Using device: {device}")
if torch.cuda.is_available():
    log.info(f"GPU: {torch.cuda.get_device_name(0)}")
    log.info(f"CUDA Version: {torch.version.cuda}")
    log.info(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

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

    y_train = _to_series(_load_parquet("y_multiclass_train")).values.astype(np.int64)
    y_val   = _to_series(_load_parquet("y_multiclass_val")).values.astype(np.int64)
    y_test  = _to_series(_load_parquet("y_multiclass_test")).values.astype(np.int64)

    log.info(
        "Dataset sizes — train: %d | val: %d | test: %d",
        len(X_train), len(X_val), len(X_test),
    )
    log.info("Number of features : %d", X_train.shape[1])
    return X_train, X_val, X_test, y_train, y_val, y_test

# ---------------------------------------------------------------------------
# Model Definition
# ---------------------------------------------------------------------------

class DNNClassifier(nn.Module):
    def __init__(self, n_features: int, n_classes: int):
        super(DNNClassifier, self).__init__()
        
        # Improved architecture with batch normalization and more capacity
        self.network = nn.Sequential(
            # Input layer with batch norm
            nn.Linear(n_features, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            # Hidden layer 1
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            # Hidden layer 2
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.15),
            
            # Hidden layer 3
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            
            # Output layer
            nn.Linear(64, n_classes)
        )
    
    def forward(self, x):
        return self.network(x)

# ---------------------------------------------------------------------------
# Class weights
# ---------------------------------------------------------------------------

def get_class_weights(y_train: np.ndarray) -> torch.Tensor:
    classes = np.unique(y_train)
    weights = compute_class_weight(
        class_weight="balanced",
        classes=classes,
        y=y_train,
    )
    weights_tensor = torch.tensor(weights, dtype=torch.float32).to(device)
    log.info("Class weights computed for %d classes.", len(weights))
    return weights_tensor

# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    scheduler = None,
) -> dict:
    
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': []
    }
    
    best_val_loss = float('inf')
    patience_counter = 0
    best_model_state = None
    
    log.info(
        "Starting training — epochs=%d, batch_size=%d, patience=%d",
        EPOCHS, BATCH_SIZE, PATIENCE,
    )
    
    start_time = time.time()
    
    for epoch in range(EPOCHS):
        # Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for batch_idx, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
            
            if (batch_idx + 1) % 100 == 0:
                log.info(f"Epoch [{epoch+1}/{EPOCHS}], Step [{batch_idx+1}/{len(train_loader)}], Loss: {loss.item():.4f}")
        
        train_loss = train_loss / len(train_loader)
        train_acc = train_correct / train_total
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
        
        val_loss = val_loss / len(val_loader)
        val_acc = val_correct / val_total
        
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        log.info(f"Epoch [{epoch+1}/{EPOCHS}] - Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
        
        # Update learning rate based on validation loss
        if scheduler:
            scheduler.step(val_loss)
        
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_model_state = model.state_dict().copy()
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                log.info(f"Early stopping triggered after {epoch+1} epochs")
                break
    
    # Restore best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    elapsed_time = time.time() - start_time
    log.info(f"Training complete in {elapsed_time:.1f} seconds — ran {len(history['train_loss'])} epochs.")
    
    return history

# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate(
    model: nn.Module,
    loader: DataLoader,
    split_name: str,
) -> dict:
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.numpy())
    
    y_true = np.array(all_labels)
    y_pred = np.array(all_preds)
    
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

def save_training_plots(history: dict, out_dir: str = PLOTS_DIR) -> None:
    os.makedirs(out_dir, exist_ok=True)
    epochs = range(1, len(history['train_loss']) + 1)
    
    # Loss plot
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(epochs, history['train_loss'], label="Train")
    ax.plot(epochs, history['val_loss'], label="Validation", linestyle="--")
    ax.set_title("DNN Training vs Validation Loss (PyTorch)", fontsize=13)
    ax.set_xlabel("Epoch", fontsize=11)
    ax.set_ylabel("Loss", fontsize=11)
    ax.legend()
    ax.grid(True, linestyle=":")
    plt.tight_layout()
    
    path = os.path.join(out_dir, "dnn_training_loss.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved plot → %s", path)
    
    # Accuracy plot
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(epochs, history['train_acc'], label="Train")
    ax.plot(epochs, history['val_acc'], label="Validation", linestyle="--")
    ax.set_title("DNN Training vs Validation Accuracy (PyTorch)", fontsize=13)
    ax.set_xlabel("Epoch", fontsize=11)
    ax.set_ylabel("Accuracy", fontsize=11)
    ax.legend()
    ax.grid(True, linestyle=":")
    plt.tight_layout()
    
    path = os.path.join(out_dir, "dnn_training_accuracy.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved plot → %s", path)

# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run_pipeline() -> None:
    torch.manual_seed(RANDOM_STATE)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(RANDOM_STATE)
    
    # Load data
    X_train, X_val, X_test, y_train, y_val, y_test = load_datasets()
    
    n_features = X_train.shape[1]
    n_classes = len(np.unique(y_train))
    log.info("Number of classes: %d", n_classes)
    
    # Create DataLoaders
    train_dataset = TensorDataset(
        torch.from_numpy(X_train),
        torch.from_numpy(y_train)
    )
    val_dataset = TensorDataset(
        torch.from_numpy(X_val),
        torch.from_numpy(y_val)
    )
    test_dataset = TensorDataset(
        torch.from_numpy(X_test),
        torch.from_numpy(y_test)
    )
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    # Build model
    model = DNNClassifier(n_features, n_classes).to(device)
    log.info(f"\nModel Architecture:\n{model}")
    
    # Class weights
    class_weights = get_class_weights(y_train)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    
    # Optimizer with weight decay for regularization
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)
    
    # Learning rate scheduler for better convergence
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3, verbose=True
    )
    
    # Train
    history = train_model(model, train_loader, val_loader, criterion, optimizer, scheduler)
    
    # Evaluate
    log.info("\nEvaluating on validation set:")
    val_metrics = evaluate(model, val_loader, "Validation")
    
    log.info("\nEvaluating on test set:")
    test_metrics = evaluate(model, test_loader, "Test")
    
    # Save model
    os.makedirs(MODEL_DIR, exist_ok=True)
    torch.save({
        'model_state_dict': model.state_dict(),
        'n_features': n_features,
        'n_classes': n_classes,
        'test_metrics': test_metrics,
    }, MODEL_PATH)
    log.info("Saved model → %s", MODEL_PATH)
    
    # Save plots
    save_training_plots(history)
    
    # Summary
    print("\n" + "="*60)
    print("  DNN (PyTorch) — Final Summary")
    print("="*60)
    print(f"  Device: {device}")
    print(f"  [Validation]")
    print(f"    Accuracy          : {val_metrics['accuracy']:.4f}")
    print(f"    Precision (macro) : {val_metrics['precision']:.4f}")
    print(f"    Recall    (macro) : {val_metrics['recall']:.4f}")
    print(f"    F1-score  (macro) : {val_metrics['f1']:.4f}")
    print(f"  [Test]")
    print(f"    Accuracy          : {test_metrics['accuracy']:.4f}")
    print(f"    Precision (macro) : {test_metrics['precision']:.4f}")
    print(f"    Recall    (macro) : {test_metrics['recall']:.4f}")
    print(f"    F1-score  (macro) : {test_metrics['f1']:.4f}")
    print("="*60)
    print(f"  Model saved : {os.path.abspath(MODEL_PATH)}")
    print("="*60 + "\n")

# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    run_pipeline()
