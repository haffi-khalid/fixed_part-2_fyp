# File: train_cnn.py
# Author: Muhammad Haffi Khalid
# Date: April 2025
#
# Purpose:
#     Trains a CNN model on the preprocessed banknote dataset.
#     Supports clean, FGSM, or BIM adversarial training.
#     Outputs model weights and training history in CSV format.

import sys
import os

parent = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent)

import numpy as np
import pandas as pd
from models.model_cnn import build_cnn_model
from adversaries.adversaries_cnn import fgsm_attack, bim_attack

# --- Configuration ---
training_type = "fgsm"  # "clean", "fgsm", or "bim"
epsilon = 0.1
bim_alpha = 0.02
bim_iters = 5
epochs = 100
batch_size = 32
mode = "quantum"

# --- Paths ---
base_data = r"C:\Users\ASDF\Desktop\part-2_fyp\data\cnn"
gen_dir = r"C:\Users\ASDF\Desktop\part-2_fyp\Gen_data\cnn"
weights_dir = r"C:\Users\ASDF\Desktop\part-2_fyp\weights\cnn"
os.makedirs(gen_dir, exist_ok=True)
os.makedirs(weights_dir, exist_ok=True)

# --- Load Data ---
def load_csv(file):
    df = pd.read_csv(file)
    X = df[["variance", "skewness", "curtosis", "entropy"]].values
    y = df["class"].values.astype(np.float32)  # already -1.0 / +1.0
    return X, y

X_train, y_train = load_csv(os.path.join(base_data, "banknote_cnn_quantum_train.csv"))
X_val, y_val = load_csv(os.path.join(base_data, "banknote_cnn_quantum_validation.csv"))
X_test, y_test = load_csv(os.path.join(base_data, "banknote_cnn_quantum_test.csv"))

# --- Reshape for Conv1D: (samples, time_steps, features) = (N, 4, 1) ---
X_train = X_train.reshape(-1, 4, 1)
X_val = X_val.reshape(-1, 4, 1)
X_test = X_test.reshape(-1, 4, 1)

# --- Build Model ---
model = build_cnn_model((4, 1))

# --- Training Loop ---
train_history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": [], "test_loss": [], "test_acc": []}
for epoch in range(1, epochs + 1):
    # Shuffle training data
    indices = np.arange(len(X_train))
    np.random.shuffle(indices)
    X_train, y_train = X_train[indices], y_train[indices]

    # Apply adversarial training
    X_batch, y_batch = [], []
    for x, y in zip(X_train, y_train):
        if training_type == "clean":
            X_batch.append(x)
        elif training_type == "fgsm":
            x_adv = fgsm_attack(model, x, y, epsilon)
            X_batch.append(x_adv.reshape(4, 1))  # Reshape to match (4,1) for Conv1D
        elif training_type == "bim":
            X_batch.append(bim_attack(model, x, y, epsilon, bim_alpha, bim_iters))
        y_batch.append(y)

    # Train on this batch

    X_batch = np.array(X_batch).reshape(-1, 4, 1)
    y_batch = np.array(y_batch).reshape(-1, 1)


    model.fit(np.array(X_batch), np.array(y_batch), batch_size=batch_size, epochs=1, verbose=0)

    # Evaluate
    train_loss, train_acc = model.evaluate(X_train, y_train, verbose=0)
    val_loss, val_acc = model.evaluate(X_val, y_val, verbose=0)
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)

    # Log
    print(f"Epoch {epoch}/{epochs}")
    print(f"Train Loss: {train_loss:.4f} | Acc: {train_acc*100:.2f}%")
    print(f"Val   Loss: {val_loss:.4f} | Acc: {val_acc*100:.2f}%")
    print(f"Test  Loss: {test_loss:.4f} | Acc: {test_acc*100:.2f}%\n")

    # Save metrics
    train_history["train_loss"].append(train_loss)
    train_history["train_acc"].append(train_acc)
    train_history["val_loss"].append(val_loss)
    train_history["val_acc"].append(val_acc)
    train_history["test_loss"].append(test_loss)
    train_history["test_acc"].append(test_acc)

# --- Save Model ---
suffix = "clean" if training_type == "clean" else f"{training_type}_{epsilon}"
model.save(os.path.join(weights_dir, f"cnn_model_{suffix}.h5"))

# --- Save Metrics ---
for key, values in train_history.items():
    df = pd.DataFrame({key: values})
    df.to_csv(os.path.join(gen_dir, f"{key}_{suffix}.csv"), index=False)

import sys
import os

parent = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent)

