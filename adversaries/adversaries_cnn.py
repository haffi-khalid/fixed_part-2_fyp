# File: adversaries_cnn.py
# Author: Muhammad Haffi Khalid
# Date: April 2025
#
# Purpose:
#     Implements FGSM and BIM attacks for CNN model compatible with
#     quantum-classifier format (4 features, normalized, single output logit).

import tensorflow as tf
import numpy as np

def fgsm_attack(model, x, y, epsilon):
    x_var = tf.convert_to_tensor(x.reshape((1, x.shape[0], 1)), dtype=tf.float32)
    y_var = tf.convert_to_tensor([[y]], dtype=tf.float32)

    with tf.GradientTape() as tape:
        tape.watch(x_var)
        pred = model(x_var)
        loss_fn = tf.keras.losses.MeanSquaredError()
        loss = loss_fn(y_var, pred)

    gradient = tape.gradient(loss, x_var)
    signed_grad = tf.sign(gradient).numpy().reshape(-1)

    x_adv = x + epsilon * signed_grad
    x_adv = np.clip(x_adv, 0.0, np.pi)
    return x_adv


def bim_attack(model, x, y_true, epsilon, alpha, num_iters):
    x_adv = tf.convert_to_tensor(x.reshape(1, 4, 1), dtype=tf.float32)
    y_var = tf.convert_to_tensor([[y_true]], dtype=tf.float32)

    for _ in range(num_iters):
        with tf.GradientTape() as tape:
            tape.watch(x_adv)
            pred = model(x_adv)
            loss = tf.keras.losses.mean_squared_error(y_var, pred)

        grad = tape.gradient(loss, x_adv)
        x_adv = x_adv + alpha * tf.sign(grad)
        x_adv = tf.clip_by_value(x_adv, x.reshape(1, 4, 1) - epsilon, x.reshape(1, 4, 1) + epsilon)
        x_adv = tf.clip_by_value(x_adv, 0.0, np.pi)

    return x_adv.numpy().reshape(4, 1)
