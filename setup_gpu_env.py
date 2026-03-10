"""
Setup GPU environment and install PyTorch with CUDA support
"""
import subprocess
import sys
import os

print("="*60)
print("  GPU Environment Setup")
print("="*60)
print()

# Check current Python version
print(f"Current Python: {sys.version}")
print(f"Python executable: {sys.executable}")
print()

# Create virtual environment
if not os.path.exists(".venv_gpu"):
    print("Creating virtual environment with Python 3.12...")
    subprocess.run([sys.executable, "-m", "venv", ".venv_gpu"], check=True)
    print("✓ Virtual environment created")
else:
    print("✓ Virtual environment already exists")

# Determine the pip path
if os.name == 'nt':  # Windows
    pip_path = os.path.join(".venv_gpu", "Scripts", "pip.exe")
    python_path = os.path.join(".venv_gpu", "Scripts", "python.exe")
else:
    pip_path = os.path.join(".venv_gpu", "bin", "pip")
    python_path = os.path.join(".venv_gpu", "bin", "python")

print()
print("Upgrading pip...")
subprocess.run([python_path, "-m", "pip", "install", "--upgrade", "pip"], check=True)
print("✓ Pip upgraded")

print()
print("Installing PyTorch with CUDA 12.1 support...")
print("This may take a few minutes...")
subprocess.run([
    pip_path, "install", "torch", 
    "--index-url", "https://download.pytorch.org/whl/cu121"
], check=True)
print("✓ PyTorch with CUDA installed")

print()
print("Installing other dependencies...")
subprocess.run([
    pip_path, "install",
    "numpy", "pandas", "scikit-learn", "pyarrow",
    "joblib", "matplotlib", "seaborn", "xgboost", "shap"
], check=True)
print("✓ All dependencies installed")

print()
print("="*60)
print("  Setup Complete!")
print("="*60)
print()
print("To activate the GPU environment:")
print("  Windows: .venv_gpu\\Scripts\\activate")
print("  Linux/Mac: source .venv_gpu/bin/activate")
print()
print("Then run: python train_dnn_pytorch.py")
print("="*60)
