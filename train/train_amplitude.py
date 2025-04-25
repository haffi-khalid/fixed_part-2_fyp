# File: train_quantum_model_amplitude.py
# Author: Muhammad Haffi Khalid
# Date: [Today's Date]

# Purpose:
#     Training script for the amplitude‐encoded quantum classifier. Supports:
#       - Clean training
#       - Adversarial training with FGSM
#       - Adversarial training with BIM

#     It:
#       1. Loads preprocessed amplitude‐encoded data (train/val/test).
#       2. Applies optional adversarial perturbations to inputs (FGSM or BIM).
#       3. Trains the quantum classifier with the Adam optimizer.
#       4. Logs epoch‐wise metrics (loss & accuracy on train/val/test, plus adversarial test metrics).
#       5. Saves metrics to CSV files under Gen_data/amplitude/{clean,adversarial}/.
#       6. Saves final weights to weights/amplitude/.


import sys
import os

# Add parent directory to sys.path
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)

import time
import logging
import pandas as pd
import numpy as np
import pennylane as qml
from pennylane import numpy as pnp
# ── extras for plots & metrics ─────────────────────────────────────────
from pathlib import Path
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (confusion_matrix, precision_recall_fscore_support,
                             roc_auc_score, roc_curve)
# ───────────────────────────────────────────────────────────────────────


# Configure logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)s:%(message)s')

# Import model and attacks
try:
    from models.quantum_model_amplitude import quantum_classifier_amplitude, n_qubits, p as num_layers
    from adversaries.adversaries_amplitude import fgsm_attack_amplitude, bim_attack_amplitude
    logging.info("Imported amplitude model and adversarial attack functions.")
except Exception as e:
    logging.error("Import error: %s", e)
    raise

# 1. PARAMETERS
training_type = "fsgm"       # "clean", "fgsm", or "bim"
epsilon = 0.1               # perturbation magnitude
bim_iterations = 5          # BIM steps
bim_alpha = 0.03            # BIM step size
num_epochs = 100
learning_rate = 0.05

# 2. DATA PATHS
train_csv = r"C:\Users\ASDF\Desktop\part-2_fyp\data\amplitude\banknote_amplitude_preprocessed_train.csv"
val_csv   = r"C:\Users\ASDF\Desktop\part-2_fyp\data\amplitude\banknote_amplitude_preprocessed_validation.csv"
test_csv  = r"C:\Users\ASDF\Desktop\part-2_fyp\data\amplitude\banknote_amplitude_preprocessed_test.csv"

# Metrics folders
# Metrics folder (requirement ①)
embedding   = "amplitude"
metrics_dir = fr"C:\Users\ASDF\Desktop\part-2_fyp\Gen_data\{embedding}\{training_type.lower()}"
os.makedirs(metrics_dir, exist_ok=True)

logging.info("Metrics directory: %s", metrics_dir)

# Weights folder
weights_dir = r"C:\Users\ASDF\Desktop\part-2_fyp\weights\amplitude"
os.makedirs(weights_dir, exist_ok=True)
logging.info("Weights directory: %s", weights_dir)

# 3. LOAD DATA
def load_data(path):
    df = pd.read_csv(path)
    X = df[["variance", "skewness", "curtosis", "entropy"]].values
    y = df["class"].apply(lambda v: -1.0 if v == 0 else 1.0).values
    return X, y

X_train, y_train = load_data(train_csv)
X_val,   y_val   = load_data(val_csv)
X_test,  y_test  = load_data(test_csv)
logging.info("Data loaded: train=%d, val=%d, test=%d", len(y_train), len(y_val), len(y_test))



# ─────────────────── helpers for plots ────────────────────────────────
def z_to_prob(z):          # ⟨Z⟩→[0,1]
    return (z + 1.0) / 2.0

def save_eval_plots(y_true, y_pred, losses, folder: str, tag="clean"):
    """Confusion-matrix, PR/F1, Δ-loss and ROC"""
    p = Path(folder); p.mkdir(parents=True, exist_ok=True)

    # 1 Confusion matrix ------------------------------------------------
    cm = confusion_matrix(y_true, y_pred, labels=[1, -1])
    plt.figure(figsize=(3,3))
    sns.heatmap(cm, annot=True, cmap="Blues", fmt="d",
                xticklabels=["+1","-1"], yticklabels=["+1","-1"])
    plt.xlabel("Pred"); plt.ylabel("True"); plt.title(f"CM ({tag})")
    plt.savefig(p/f"{tag}_confusion_matrix.png", dpi=300, bbox_inches="tight")
    plt.close()

    # 2 Precision / Recall / F1 table -----------------------------------
    pr, rc, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=[1,-1], zero_division=0, average=None)
    fig, ax = plt.subplots(figsize=(4,1.4)); ax.axis("off")
    cell = [[f"{x:.2f}" for x in row] for row in zip(pr, rc, f1)]
    tbl = ax.table(cellText=cell, rowLabels=["+1","-1"],
                   colLabels=["Prec","Rec","F1"], loc="center")
    tbl.auto_set_font_size(False); tbl.set_fontsize(8)
    plt.title(f"PR/F1 ({tag})")
    plt.savefig(p/f"{tag}_f1_table.png", dpi=300, bbox_inches="tight")
    plt.close()

    # 3 Δ-loss curve -----------------------------------------------------
    if losses.get("train") and losses.get("val"):
        dl = [tr-va for tr,va in zip(losses["train"], losses["val"])]
        plt.figure(); plt.plot(dl); plt.title("Δ loss (train-val)")
        plt.xlabel("epoch"); plt.ylabel("train-val")
        plt.savefig(p/f"{tag}_delta_loss.png", dpi=300, bbox_inches="tight")
        plt.close()

    # 4 ROC / AUROC ------------------------------------------------------
    try:
        y_score = z_to_prob(y_pred)          # already probabilities
        y_bin   = (np.array(y_true)+1)//2
        auc     = roc_auc_score(y_bin, y_score)
        fpr,tpr,_ = roc_curve(y_bin, y_score)
        plt.figure(); plt.plot(fpr,tpr,label=f"AUC={auc:.3f}")
        plt.plot([0,1],[0,1],'--k'); plt.xlabel("FPR"); plt.ylabel("TPR")
        plt.legend(); plt.title(f"ROC ({tag})")
        plt.savefig(p/f"{tag}_roc.png", dpi=300, bbox_inches="tight"); plt.close()
    except ValueError:
        pass
# ───────────────────────────────────────────────────────────────────────









# 4. COST AND ACCURACY
def cost(weights, X, y):
    losses = []
    for xi, yi in zip(X, y):
        losses.append((quantum_classifier_amplitude(weights, xi) - yi)**2)
    return pnp.mean(pnp.array(losses))

def accuracy(weights, X, y):
    correct = 0
    for xi, yi in zip(X, y):
        pred = quantum_classifier_amplitude(weights, xi)
        correct += (1.0 if pred >= 0 else -1.0) == yi
    return correct / len(y)

# 5. INITIALIZE
weights = pnp.array(np.random.uniform(0, 2*np.pi, (num_layers, n_qubits, 3)), requires_grad=True)
opt = qml.AdamOptimizer(stepsize=learning_rate)
logging.info("Initialized weights shape %s", weights.shape)

# 6. TRAINING LOOP
history = {
    "train_loss": [], "train_acc": [],
    "val_loss": [],   "val_acc":   [],
    "test_loss": [],  "test_acc":  [],
    "adv_test_loss": [], "adv_test_acc": []
}

logging.info("Starting %s training for %d epochs", training_type, num_epochs)
for epoch in range(1, num_epochs+1):
    t0 = time.time()
    
    # Training step
    def step_fn(w):
        batch_losses = []
        for xi, yi in zip(X_train, y_train):
            x_in = xi.copy()
            if training_type == "fgsm":
                x_in = fgsm_attack_amplitude(w, x_in, yi, epsilon)
            elif training_type == "bim":
                x_in = bim_attack_amplitude(w, x_in, yi, epsilon, bim_iterations, bim_alpha)
            batch_losses.append((quantum_classifier_amplitude(w, x_in) - yi)**2)
        return pnp.mean(pnp.array(batch_losses))
    
    weights, train_cost = opt.step_and_cost(step_fn, weights)
    
    # Compute metrics
    tr_loss = cost(weights, X_train, y_train)
    tr_acc  = accuracy(weights, X_train, y_train)
    v_loss  = cost(weights, X_val,   y_val)
    v_acc   = accuracy(weights, X_val,   y_val)
    te_loss = cost(weights, X_test,  y_test)
    te_acc  = accuracy(weights, X_test,  y_test)
    
    # Adversarial test metrics
    if training_type in ("fgsm","bim"):
        adv_losses = []
        adv_correct = 0
        for xi, yi in zip(X_test, y_test):
            if training_type == "fgsm":
                xa = fgsm_attack_amplitude(weights, xi.copy(), yi, epsilon)
            else:
                xa = bim_attack_amplitude(weights, xi.copy(), yi, epsilon, bim_iterations, bim_alpha)
            adv_losses.append((quantum_classifier_amplitude(weights, xa)-yi)**2)
            adv_pred = quantum_classifier_amplitude(weights, xa)
            adv_correct += (1.0 if adv_pred>=0 else -1.0)==yi
        adv_te_loss = pnp.mean(pnp.array(adv_losses))
        adv_te_acc  = adv_correct / len(y_test)
    else:
        adv_te_loss = None
        adv_te_acc  = None
    
    dt = time.time()-t0

    # Log and record
    logging.info("Epoch %2d/%d (%.2fs)", epoch, num_epochs, dt)
    logging.info("  Train loss=%.4f acc=%.2f%%", tr_loss, tr_acc*100)
    logging.info("  Val   loss=%.4f acc=%.2f%%", v_loss,  v_acc*100)
    logging.info("  Test  loss=%.4f acc=%.2f%%", te_loss, te_acc*100)
    if adv_te_loss is not None:
        logging.info("  AdvTest loss=%.4f acc=%.2f%%", adv_te_loss, adv_te_acc*100)
    
    history["train_loss"].append(tr_loss)
    history["train_acc"].append(tr_acc)
    history["val_loss"].append(v_loss)
    history["val_acc"].append(v_acc)
    history["test_loss"].append(te_loss)
    history["test_acc"].append(te_acc)
    if adv_te_loss is not None:
        history["adv_test_loss"].append(adv_te_loss)
        history["adv_test_acc"].append(adv_te_acc)

# 7. SAVE METRICS
for key, values in history.items():
    if "adv_" in key and training_type=="clean":
        continue
    pd.DataFrame({key: values}).to_csv(os.path.join(metrics_dir, f"{key}.csv"), index=False)
logging.info("Saved metrics to %s", metrics_dir)





# ------------- PLOTS on clean test ------------------------------------
y_score_clean = np.array([ z_to_prob(quantum_classifier_amplitude(weights, x))
                           for x in X_test ])
y_pred_clean  = np.where(y_score_clean >= 0.5, 1.0, -1.0)
save_eval_plots(
    y_true=y_test,
    y_pred=y_pred_clean,
    losses={"train": history["train_loss"], "val": history["val_loss"]},
    folder=metrics_dir,
    tag="clean"
)

# ------------- PLOTS on adversarial test (if any) ----------------------
if training_type in ("fgsm", "bim"):
    X_adv = []
    for xi, yi in zip(X_test, y_test):
        xa = (fgsm_attack_amplitude(weights, xi.copy(), yi, epsilon)
              if training_type=="fgsm"
              else bim_attack_amplitude(weights, xi.copy(), yi,
                                        epsilon, bim_iterations, bim_alpha))
        X_adv.append(xa)
    y_score_adv = np.array([ z_to_prob(quantum_classifier_amplitude(weights, x))
                             for x in X_adv ])
    y_pred_adv = np.where(y_score_adv >= 0.5, 1.0, -1.0)
    save_eval_plots(
        y_true=y_test, y_pred=y_pred_adv,
        losses={"train": history["train_loss"], "val": history["val_loss"]},
        folder=metrics_dir, tag="adv")








# 8. SAVE WEIGHTS
suffix = "clean" if training_type=="clean" else f"{training_type}_{epsilon}"
fname = f"amplitude_{num_layers}layers_{suffix}.npy"
pnp.save(os.path.join(weights_dir, fname), weights)
logging.info("Saved weights to %s", os.path.join(weights_dir, fname))
