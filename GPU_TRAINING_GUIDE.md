# GPU Training Guide for IDS Project

## ✅ Your System

- **GPU:** NVIDIA GeForce RTX 3050 Laptop (4GB VRAM)
- **Driver Version:** 591.44
- **CUDA Version:** 13.1
- **Status:** GPU detected and ready!

## ⚠️ Current Issue

**Python 3.14** is too new - PyTorch doesn't have CUDA-enabled builds for it yet.

Your current setup uses **CPU-only PyTorch**, which is why the DNN trained on CPU (took ~7 minutes).

## 🚀 Solutions to Use GPU

### Option 1: Use Python 3.12 (Recommended)

1. **Install Python 3.12**
   - Download from: https://www.python.org/downloads/
   - Select version 3.12.x

2. **Run the setup script**
   ```batch
   setup_gpu_env.bat
   ```

3. **Activate GPU environment**
   ```batch
   .venv_gpu\Scripts\activate
   ```

4. **Train with GPU**
   ```batch
   python train_dnn_pytorch.py
   ```

### Option 2: Manual Setup with Python 3.12

```batch
# Create new virtual environment with Python 3.12
python3.12 -m venv .venv_gpu

# Activate
.venv_gpu\Scripts\activate

# Install PyTorch with CUDA 12.1
pip install torch --index-url https://download.pytorch.org/whl/cu121

# Install other dependencies
pip install numpy pandas scikit-learn pyarrow joblib matplotlib seaborn xgboost shap

# Train model (will automatically use GPU)
python train_dnn_pytorch.py
```

### Option 3: Continue with CPU (Current Setup)

Your model already trained successfully on CPU:
- Training time: ~7 minutes
- Test accuracy: 57.5%
- Model saved: `models/dnn_pytorch_model.pth`

**To retrain on CPU:**
```batch
python train_dnn_pytorch.py
```

## 📊 Expected GPU Performance

With your RTX 3050:
- **Training time:** ~2-3 minutes (2-3x faster than CPU)
- **Batch processing:** Much faster
- **Memory usage:** ~2-3GB VRAM

## 🔍 Verify GPU is Being Used

After setup, check GPU usage:

```python
import torch
print("CUDA available:", torch.cuda.is_available())
print("GPU:", torch.cuda.get_device_name(0))
```

During training, monitor with:
```batch
nvidia-smi -l 1
```

## 📝 Notes

- The training script `train_dnn_pytorch.py` **automatically detects and uses GPU** if available
- No code changes needed - it switches between CPU/GPU automatically
- Your RTX 3050 (4GB) is sufficient for this dataset
- Current dataset size: 2M samples (memory-optimized)

## ⚡ Why Python 3.14 Doesn't Work

PyTorch typically lags behind new Python releases by several months. CUDA builds for Python 3.14 will likely be available in 2-3 months.

**Current PyTorch CUDA support:**
- ✅ Python 3.12
- ✅ Python 3.11
- ✅ Python 3.10
- ❌ Python 3.14 (not yet)
