"""
----------------------
Fit observed stellar spectra using a trained ANN (Payne-like model).

This script:
  1. Loads a trained ANN model.
  2. Defines helper functions to generate synthetic spectra from stellar labels.
  3. Reads observed spectra and performs parameter fitting using curve_fit.
  4. Produces diagnostic plots and saves fitted parameters.

Original author: Nagaraj Vernekar
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import tensorflow as tf


# =============================================================================
# Configuration
# =============================================================================
STAR_NAME = "18Sco"  # 👈 change this to the target star name
MODEL_PATH = "models/pre_trained.keras" #Trained ANN model
SCALING_FILE = "scaling/pre_trained_scaled.csv" # Values used for scaling data during training
STAR_PARAM_FILE = "literature_parameters.csv"   # File with literature values for parameters 
MASK_FILE = "masked_pixels.csv"                 # List of wavelength values not considered for fitting
OUTPUT_PLOT = f"fitting/{STAR_NAME}_fit.png"
VERIFICATION_DIR = "example_spectra"



# =============================================================================
# Helper Functions
# =============================================================================
def leaky_relu(z, alpha=0.3):
    """Leaky ReLU activation."""
    return np.where(z > 0, z, alpha * z)


def sigmoid(z):
    """Sigmoid activation."""
    return 1 / (1 + np.exp(-z))


def load_ann_weights(model_path):
    """Extract weights and biases from the trained ANN model."""
    model = tf.keras.models.load_model(model_path)
    weights = [layer.get_weights() for layer in model.layers if layer.get_weights()]
    print(f"Loaded ANN model from {model_path}")
    return [w for layer in weights for w in layer]  # flatten [(W, b), (W, b), ...]


def generate_spectrum(labels, weights):
    """Compute synthetic spectrum from ANN given scaled stellar parameters."""
    w1, b1, w2, b2, w3, b3, w4, b4 = weights
    x1 = leaky_relu(np.dot(w1.T, labels) + b1)
    x2 = leaky_relu(np.dot(w2.T, x1) + b2)
    x3 = leaky_relu(np.dot(w3.T, x2) + b3)
    return sigmoid(np.dot(w4.T, x3) + b4)


def masking_from_chi(wave, chi, threshold=0.02):
    """Create a mask for wavelengths where |residual| > threshold."""
    mask = np.abs(chi) > threshold
    print(f"Masked {mask.sum()} pixels with |residual| > {threshold}")
    return mask


def masking_from_file(wave, mask_file):
    """Mask wavelengths listed in an existing mask file."""
    mask_df = pd.read_csv(mask_file)
    masked_wave = mask_df["mask"].astype(float).values
    mask = np.isin(np.round(wave, 2), masked_wave)
    print(f"Masked {mask.sum()} points using {mask_file}")
    return mask


# =============================================================================
# Main Fitting Function
# =============================================================================
def fit_single_star(star_name):
    """Fit a single star's observed spectrum using the ANN model."""
    # Load ANN weights and scaling
    weights = load_ann_weights(MODEL_PATH)
    scaling = pd.read_csv(SCALING_FILE)
    x_max, x_min = scaling["x_max"].values, scaling["x_min"].values

    # Load reference stellar parameters
    star_params_df = pd.read_csv(STAR_PARAM_FILE)
    star_row = star_params_df[star_params_df["star"] == star_name]
    if star_row.empty:
        raise ValueError(f"Star '{star_name}' not found in {STAR_PARAM_FILE}")
    true_labels = star_row.iloc[0, 1:].to_numpy(float)

    # Load observed spectrum
    obs_file = os.path.join(VERIFICATION_DIR, f"{star_name}.csv")
    df = pd.read_csv(obs_file, delimiter="\t")
    if df["waveobs"].iloc[0] < 2000:
        df["waveobs"] *= 10
    df = df[(df["waveobs"] >= 4200) & (df["waveobs"] <= 6900)]

    wave = df["waveobs"].to_numpy(float)
    flux = df["flux"].to_numpy(float)

    # Scale parameters
    scaled_labels = (true_labels - x_min) / (x_max - x_min) - 0.5

    # Masking and uncertainty
    if star_name.lower() == "sun":
        pred = generate_spectrum(scaled_labels, weights)
        chi = flux - pred
        mask = masking_from_chi(wave, chi)
        pd.DataFrame({"mask": np.round(wave[mask], 2)}).to_csv(MASK_FILE, index=False)
    else:
        mask = masking_from_file(wave, MASK_FILE)

    obs_err = np.ones_like(wave) * 0.01
    obs_err[mask] = 1e8

    # Curve fitting setup
    bounds = (-0.5 * np.ones_like(scaled_labels), 0.5 * np.ones_like(scaled_labels))

    def model_func(_, *params):
        return generate_spectrum(np.array(params), weights)

    print(f"Starting fit for {star_name} ...")
    popt, pcov = curve_fit(
        model_func,
        xdata=[],
        ydata=flux,
        p0=np.zeros_like(scaled_labels),
        bounds=bounds,
        sigma=obs_err,
        method="trf",
        xtol=1e-10,
        ftol=1e-10,
    )

    # Unscale fitted parameters
    fitted = (popt + 0.5) * (x_max - x_min) + x_min
    fitted_err = np.sqrt(np.diag(pcov)) * (x_max - x_min)

    # Compute best-fit spectrum
    best_fit_flux = generate_spectrum(popt, weights)
    chi_sq = np.sum((flux - best_fit_flux) ** 2)

    print(f"\n✅ Fit complete for {star_name}")
    print(f"Chi-square: {chi_sq:.4f}")
    
    # -------------------------------------------------------------------------
    # PRINT SUMMARY TABLE
    # -------------------------------------------------------------------------
    param_names = ["Teff", "logg", "[Fe/H]", "[α/Fe]", "Vmic"]
    print("\nSummary of fitted parameters:")
    print(f"{'Parameter':<10}{'Literature':>15}{'Fitted':>15}{'Δ (Lit−Fit)':>15}")
    print("-" * 55)
    for i, name in enumerate(param_names):
        diff = true_labels[i] - fitted[i]
        print(f"{name:<10}{true_labels[i]:>15.4f}{fitted[i]:>15.4f}{diff:>15.4f}")
    print("-" * 55)



    # Plot final comparison
    plt.figure(figsize=(12, 6))
    plt.plot(wave, flux, "k", lw=1.0, label="Observed")
    plt.plot(wave, best_fit_flux, "r", lw=1.0, alpha=0.8, label="Best Fit")
    plt.xlabel("Wavelength [Å]")
    plt.ylabel("Flux")
    plt.title(f"{star_name} — Observed vs Best Fit Spectrum")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUTPUT_PLOT, dpi=300)
    plt.show()

    print(f"Saved plot: {OUTPUT_PLOT}")

    return fitted, fitted_err, chi_sq


# =============================================================================
# Run the fit
# =============================================================================
if __name__ == "__main__":
    fit_single_star(STAR_NAME)

