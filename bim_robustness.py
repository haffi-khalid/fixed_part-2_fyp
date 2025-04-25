# File: generate_bim_robustness_strong.py
# Author: Muhammad Haffi Khalid
# Date: April 2025
#
# Purpose:
#     Stronger BIM attack evaluation with 10 iterations and smaller alpha.
#     Adds Z-distribution inspection for deeper insight.

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Add project root to path
parent = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(parent)

# Attack hyperparameters (stronger)
epsilons = np.round(np.arange(0.0, 1.0001, 0.05), 2)
bim_iterations = 10      # Increased from 5
bim_alpha = 0.005        # Decreased from 0.02

OUT_DIR = r"C:\Users\ASDF\Desktop\part-2_fyp\Results\bim_stronger"
os.makedirs(OUT_DIR, exist_ok=True)

# Imports: Replace dummy with actual
from models.quantum_model_angle import quantum_classifier as qc_angle
from adversaries.adversaries_angle import bim_attack as bim_angle
from models.quantum_model_amplitude import quantum_classifier_amplitude as qc_amp
from adversaries.adversaries_amplitude import bim_attack_amplitude as bim_amp
from models.quantum_model_reupload import quantum_classifier_reupload as qc_reu
from adversaries.adversaries_reupload import bim_attack_reupload as bim_reu

# Models to evaluate
MODELS = {
    "Angle": {
        "qc": qc_angle,
        "bim": bim_angle,
        "weights": r"C:\Users\ASDF\Desktop\part-2_fyp\weights\angle\angle_10layers_clean.npy",
        "test_csv": r"C:\Users\ASDF\Desktop\part-2_fyp\data\angle\banknote_angle_preprocessed_test.csv"
    },
    "Amplitude": {
        "qc": qc_amp,
        "bim": bim_amp,
        "weights": r"C:\Users\ASDF\Desktop\part-2_fyp\weights\amplitude\amplitude_10layers_clean.npy",
        "test_csv": r"C:\Users\ASDF\Desktop\part-2_fyp\data\amplitude\banknote_amplitude_preprocessed_test.csv"
    },
    "Reupload": {
        "qc": qc_reu,
        "bim": bim_reu,
        "weights": r"C:\Users\ASDF\Desktop\part-2_fyp\weights\Reupload\reupload_10layers_clean.npy",
        "test_csv": r"C:\Users\ASDF\Desktop\part-2_fyp\data\Reupload\banknote_reupload_preprocessed_test.csv"
    }
}

def load_test(path):
    df = pd.read_csv(path)
    X = df[["variance", "skewness", "curtosis", "entropy"]].values
    y = df["class"].apply(lambda v: -1.0 if v == 0 else 1.0).values
    return X, y

combined_acc = {}
combined_loss = {}
combined_z = {}

for name, cfg in MODELS.items():
    print(f"\n>>> Starting {name} model")
    qc = cfg["qc"]
    bim = cfg["bim"]
    weights = np.load(cfg["weights"], allow_pickle=True)
    X_test, y_test = load_test(cfg["test_csv"])

    acc_eps, loss_eps, mean_z_eps = [], [], []

    for eps in epsilons:
        correct = 0
        losses, z_vals = [], []

        for x, y in zip(X_test, y_test):
            x_adv = bim(weights, x.copy(), y, eps, bim_iterations, bim_alpha)
            z = qc(weights, x_adv)
            label = 1.0 if z >= 0 else -1.0
            z_vals.append(z)
            losses.append((z - y)**2)
            if label == y:
                correct += 1

        acc = correct / len(y_test)
        loss = np.mean(losses)
        z_mean = np.mean(z_vals)

        print(f"[{name}] ε={eps:.2f} | acc={acc:.2f} | loss={loss:.3f} | ⟨Z⟩={z_mean:.3f}")
        acc_eps.append(acc)
        loss_eps.append(loss)
        mean_z_eps.append(z_mean)

    # Store metrics
    combined_acc[name] = acc_eps
    combined_loss[name] = loss_eps
    combined_z[name] = mean_z_eps

    # Save individual CSV
    df = pd.DataFrame({
        "epsilon": epsilons,
        "accuracy": acc_eps,
        "loss": loss_eps,
        "mean_z": mean_z_eps
    })
    df.to_csv(os.path.join(OUT_DIR, f"{name.lower()}_bim_strong.csv"), index=False)

# Save combined CSV
combined_df = pd.DataFrame({"epsilon": epsilons})
for name in MODELS:
    combined_df[f"{name.lower()}_accuracy"] = combined_acc[name]
    combined_df[f"{name.lower()}_loss"] = combined_loss[name]
    combined_df[f"{name.lower()}_mean_z"] = combined_z[name]

combined_df.to_csv(os.path.join(OUT_DIR, "combined_bim_stronger.csv"), index=False)

# Plot Accuracy
plt.figure()
for name in MODELS:
    plt.plot(epsilons, combined_acc[name], marker='o', label=name)
plt.title("BIM (Strong) Accuracy vs Epsilon")
plt.xlabel("Epsilon (ε)")
plt.ylabel("Adversarial Test Accuracy")
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(OUT_DIR, "combined_accuracy_strong.png"))
plt.close()

# Plot Loss
plt.figure()
for name in MODELS:
    plt.plot(epsilons, combined_loss[name], marker='o', label=name)
plt.title("BIM (Strong) Loss vs Epsilon")
plt.xlabel("Epsilon (ε)")
plt.ylabel("MSE Loss")
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(OUT_DIR, "combined_loss_strong.png"))
plt.close()

# Optional: Plot Mean Z-values
plt.figure()
for name in MODELS:
    plt.plot(epsilons, combined_z[name], marker='o', label=name)
plt.title("Mean ⟨Z⟩ Prediction vs Epsilon")
plt.xlabel("Epsilon (ε)")
plt.ylabel("Average Prediction Value ⟨Z⟩")
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(OUT_DIR, "combined_mean_z_strong.png"))
plt.close()

print(f"\n[✓] Stronger BIM plots + data saved to: {OUT_DIR}")
65