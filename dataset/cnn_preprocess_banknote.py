# File: cnn_preprocess_banknote.py
# Author: Muhammad Haffi Khalid
# Date: April 2025

# Purpose:
#     Preprocess the UCI Banknote Authentication dataset for CNN training.
#     This script supports two preprocessing modes:
#       - 'angle'   → min-max scaling to [0, π]
#       - 'quantum' → row-wise normalization (L2 norm = 1)
#
#     Output:
#       - Three CSVs: train, validation, test
#       - All with 4 normalized features and label ∈ {–1.0, +1.0}

import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, normalize

# ----------------------------
# CONFIGURATION
# ----------------------------
MODE = "quantum"  # Options: "angle" or "quantum"
INPUT_CSV = r"C:\Users\ASDF\Desktop\part-2_fyp\data\data_banknote_authentication.csv"
OUTPUT_DIR = r"C:\Users\ASDF\Desktop\part-2_fyp\data\cnn"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ----------------------------
# LOAD + SHUFFLE
# ----------------------------
df = pd.read_csv(INPUT_CSV, header=None)
df.columns = ["variance", "skewness", "curtosis", "entropy", "class"]
df = df.sample(frac=1.0, random_state=42).reset_index(drop=True)

# ----------------------------
# SPLIT: Train (70%), Val (15%), Test (15%)
# ----------------------------
train_df, test_df = train_test_split(df, test_size=0.3, stratify=df["class"], random_state=42)
val_df, test_df   = train_test_split(test_df, test_size=0.5, stratify=test_df["class"], random_state=42)

# ----------------------------
# PREPROCESSING FUNCTION
# ----------------------------
def preprocess(df, mode):
    X = df[["variance", "skewness", "curtosis", "entropy"]].values
    y = df["class"].apply(lambda v: -1.0 if v == 0 else 1.0).values

    if mode == "angle":
        scaler = MinMaxScaler(feature_range=(0, np.pi))
        X = scaler.fit_transform(X)
    elif mode == "quantum":
        X = normalize(X, norm='l2', axis=1)
    else:
        raise ValueError("Unknown mode. Choose either 'angle' or 'quantum'.")

    df_out = pd.DataFrame(X, columns=["variance", "skewness", "curtosis", "entropy"])
    df_out["class"] = y
    return df_out

# ----------------------------
# APPLY PREPROCESSING
# ----------------------------
train_out = preprocess(train_df, MODE)
val_out   = preprocess(val_df, MODE)
test_out  = preprocess(test_df, MODE)

# ----------------------------
# SAVE FILES
# ----------------------------
suffix = f"cnn_{MODE}"
train_out.to_csv(os.path.join(OUTPUT_DIR, f"banknote_{suffix}_train.csv"), index=False)
val_out.to_csv(os.path.join(OUTPUT_DIR, f"banknote_{suffix}_validation.csv"), index=False)
test_out.to_csv(os.path.join(OUTPUT_DIR, f"banknote_{suffix}_test.csv"), index=False)

print(f"[✓] Saved preprocessed CSVs to: {OUTPUT_DIR}")
