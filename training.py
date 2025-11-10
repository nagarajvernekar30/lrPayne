"""
------------
Train a neural network model that predicts stellar spectra from given stellar parameters.

Original author: Nagaraj Vernekar
Date: 2025-11-06
"""

import os
import time
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow import keras
import tensorflow as tf


# =============================================================================
# Configuration
# =============================================================================
DATASET_NAME = "5000_new"
LABEL_FILE = f"training_input/labels_example.csv"
SPECTRA_FILE = f"training_input/spectra_example.csv"
MODEL_OUTPUT = f"models/model_example.keras"
SCALING_FILE = f"scaling/minmax.csv"
LOSS_OUTPUT = "loss/loss.csv"

SCALER_METHOD = "minmax"  # options: "minmax", "standard"
BATCH_SIZE = 128
EPOCHS = 5000
VALIDATION_SPLIT = 0.2
RANDOM_SEED = 42


# =============================================================================
# Utility Functions
# =============================================================================
def load_data(label_path: str, spectra_path: str):
    """Load labels and spectra from CSV files."""
    labels = pd.read_csv(label_path).round(3)
    print(f"Labels loaded: {labels.shape}")

    spectra = np.loadtxt(spectra_path, delimiter=",")[1:].T
    print(f"Spectra loaded: {spectra.shape}")
    return labels, spectra


def scale_inputs(X_train, X_test, method="minmax"):
    """Scale input parameters using the specified method."""
    if method == "minmax":
        x_max = np.max(X_train, axis=0)
        x_min = np.min(X_train, axis=0)
        X_train_scaled = (X_train - x_min) / (x_max - x_min) - 0.5
        X_test_scaled = (X_test - x_min) / (x_max - x_min) - 0.5

        scaling_df = pd.DataFrame({"x_max": x_max, "x_min": x_min})
        os.makedirs(os.path.dirname(SCALING_FILE), exist_ok=True)
        scaling_df.to_csv(SCALING_FILE, index=False)

        print(f"Scaling parameters saved to {SCALING_FILE}")
        return X_train_scaled, X_test_scaled
    else:
        raise ValueError(f"Unsupported scaler method: {method}")


def build_model(input_dim: int, output_dim: int):
    """Build and compile the ANN model."""
    input_layer = keras.layers.Input(shape=(input_dim,))
    x = keras.layers.Dense(200, kernel_initializer="he_normal")(input_layer)
    x = keras.layers.LeakyReLU(alpha=0.3)(x)

    x = keras.layers.Dense(200, kernel_initializer="he_normal")(x)
    x = keras.layers.LeakyReLU(alpha=0.3)(x)

    x = keras.layers.Dense(200, kernel_initializer="he_normal")(x)
    x = keras.layers.LeakyReLU(alpha=0.3)(x)

    output_layer = keras.layers.Dense(output_dim, activation="sigmoid")(x)
    model = keras.models.Model(inputs=input_layer, outputs=output_layer)

    loss_fn = keras.losses.MeanAbsolutePercentageError()
    optimizer = tf.keras.optimizers.AdamW(learning_rate=0.001)
    model.compile(loss=loss_fn, optimizer=optimizer, metrics=["mse"])

    print(model.summary())
    return model


def plot_training_history(history, output_path):
    """Plot and save the training and validation loss."""
    training_loss = history.history["loss"]
    validation_loss = history.history["val_loss"]

    loss_df = pd.DataFrame({
        "training_loss": training_loss,
        "validation_loss": validation_loss
    })
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    loss_df.to_csv(output_path, index=False)

    plt.figure(figsize=(10, 5))
    plt.plot(training_loss, label="Training")
    plt.plot(validation_loss, label="Validation")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    print(f"Loss history saved to {output_path}")


def plot_sample_predictions(model, X_test, y_test, n_samples=3):
    """Randomly plot predicted vs. original spectra for a few test samples."""
    sample_indices = np.random.randint(0, len(X_test), size=n_samples)
    predicted_spectra = model.predict(X_test[sample_indices])

    fig, axes = plt.subplots(n_samples, 1, figsize=(15, 5 * n_samples))
    for i, idx in enumerate(sample_indices):
        ax = axes[i] if n_samples > 1 else axes
        ax.plot(y_test[idx], label="Original", alpha=0.6)
        ax.plot(predicted_spectra[i], label="Predicted", alpha=0.6)
        ax.set_title(f"Sample {idx}")
        ax.legend()
    plt.tight_layout()
    plt.show()


# =============================================================================
# Main Training Workflow
# =============================================================================
def train_model():
    """Main workflow to train the ANN model."""
    start_time = time.perf_counter()

    # Load data
    labels, spectra = load_data(LABEL_FILE, SPECTRA_FILE)
    X = np.array(labels)
    y = spectra

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=VALIDATION_SPLIT, random_state=RANDOM_SEED, shuffle=True
    )

    print(f"Training samples: {len(X_train)}, Test samples: {len(X_test)}")

    # Scale inputs
    X_train_scaled, X_test_scaled = scale_inputs(X_train, X_test, SCALER_METHOD)

    # Build model
    model = build_model(input_dim=X_train_scaled.shape[1], output_dim=y_train.shape[1])

    # Training
    lr_scheduler = keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=10, min_lr=1e-7, verbose=1)
    early_stopping = keras.callbacks.EarlyStopping(patience=30, min_delta=0.0005, restore_best_weights=True)

    history = model.fit(
        X_train_scaled,
        y_train,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        verbose=1,
        validation_data=(X_test_scaled, y_test),
        callbacks=[early_stopping, lr_scheduler],
    )

    # Save model
    os.makedirs(os.path.dirname(MODEL_OUTPUT), exist_ok=True)
    model.save(MODEL_OUTPUT)
    print(f"Model saved to {MODEL_OUTPUT}")

    # Plot loss
    plot_training_history(history, LOSS_OUTPUT)

    # Plot sample spectra
    plot_sample_predictions(model, X_test_scaled, y_test)

    print(f"Total training time: {time.perf_counter() - start_time:.2f} s")


# =============================================================================
# Run the training
# =============================================================================
if __name__ == "__main__":
    train_model()

