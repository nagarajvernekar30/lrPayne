# 🌟 LRPayne — A Neural Network Model for Stellar Spectrum Fitting

**LRPayne** is a Python-based implementation inspired by *The Payne* methodology, designed to train and apply artificial neural networks (ANNs) for generating and fitting stellar spectra.  
It enables efficient forward modeling of stellar spectra and fast estimation of stellar parameters by combining machine learning and traditional spectroscopic techniques.

---

## 🚀 Features

- Train a fully connected neural network to generate stellar spectra from stellar parameters.
- Fit observed spectra using a pre-trained ANN to infer stellar parameters.
- GPU-accelerated with TensorFlow (CUDA/cuDNN compatible).
- Modular, well-documented code with clear workflow separation:
  - `training.py` → train the neural network.
  - `fitting.py` → fit one observed spectrum.
- Ready-to-use environment file for reproducibility (`environment.yml`).

---

## 🧠 Workflow Overview

### 1️⃣ Training the ANN
The ANN learns to predict normalized stellar spectra from a grid of known stellar parameters.

```bash
python training.py
```
Inputs:
- Grid of synthetic spectra → 'training_input/spectra_example.csv'
- Labels corresponding to synthetic spectra → 'training_input/labels_example.csv'
Outputs:
- Trained model → `models/nosnr_<dataset_name>_test.keras`
- Training loss → `loss/nosnr.csv`
- Scaling parameters → `scaling/minmax_<dataset_name>.csv`

### 2️⃣ Fitting a Single Star
Using the trained model, LRPayne fits the observed spectrum of a single target star and compares it to literature parameters.

Edit the target star in the script:

```python
STAR_NAME = "18Sco"
```

Then run:

```bash
python fitting.py
```

Outputs:
- Fitted stellar parameters (printed summary)
- Observed vs. best-fit spectrum → `fitting/<star_name>_fit.png`

---

## 🧩 File Structure

```
LRPayne/
│
├── training.py                  # Train the neural network
├── fitting.py                   # Fit a single star using the ANN
├── environment.yml              # Clean reproducible environment
│
├── training_input/
│   ├── labels_<dataset>.csv
│   └── spectra_<dataset>.csv
│
├── models/                      # Trained model files (.keras)
├── loss/                        # Training loss logs
├── scaling/                     # Scaling parameters for input normalization
├── parameter/                   # True stellar parameters
├── example_spectra/             # Observed spectra for testing
└── fitting/                     # Fit results and plots
```

---

## ⚙️ Installation & Environment Setup

### 🧱 Option 1 — Conda (recommended)
Create a reproducible GPU-ready environment:

```bash
conda env create -f environment.yml
conda activate lrpayne
```

Requirements:
- Python ≥ 3.10  
- NVIDIA driver ≥ 550  
- CUDA 12.5  
- cuDNN 9.1  

### 🧱 Option 2 — Pip (CPU only)

```bash
python3 -m venv lrpayne_env
source lrpayne_env/bin/activate
pip install -r requirements.txt
```

---

## 🧮 Dependencies (core)

- `tensorflow` (GPU)
- `numpy`, `pandas`, `scipy`
- `scikit-learn`, `matplotlib`, `h5py`
- `astropy`, `lmfit`, `emcee`
- `ezpadova`, `uncertainties`

All dependencies are managed automatically via `environment.yml`.

---

## 🧪 Citation

If you use **LRPayne** in your research, please cite:

> *Author Name(s)*, “LRPayne: Neural Network Fitting of Stellar Spectra,” (2025),  
> GitHub Repository: [https://github.com/<your-username>/LRPayne](https://github.com/<your-username>/LRPayne)

---

## 🧑‍💻 Author

Nagaraj Vernekar 

---

## 📜 License

This software is governed by the MIT License: In brief, you can use, distribute, and change this package as you please..
