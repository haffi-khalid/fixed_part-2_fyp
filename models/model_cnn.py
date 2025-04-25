# File: model_cnn.py
# Author: Muhammad Haffi Khalid
# Date: April 2025
#
# Purpose:
#     Defines a CNN classifier compatible with quantum-classifier input format
#     (4 input features → reshaped to (4, 1), no padding, ready for transferability).

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, BatchNormalization, Input

def build_cnn_model(input_shape):
    model = Sequential([
        Input(shape=input_shape),
        Conv1D(32, kernel_size=2, activation='relu'),
        BatchNormalization(),
        MaxPooling1D(pool_size=1),
        Flatten(),
        Dense(32, activation='relu'),
        Dense(1, activation='tanh')  # Output in range [-1, +1] to match quantum classifier output
    ])
    model.compile(optimizer='adam', loss='mse', metrics=['accuracy'])
    return model
