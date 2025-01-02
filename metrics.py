import numpy as np


def MAE(y_true, y_pred):
    if y_true.shape != y_pred.shape:
        raise ValueError("Shapes of y_true and y_pred must match.")
    return np.mean(np.abs(y_true - y_pred))


def RMSE(y_true, y_pred):
    if y_true.shape != y_pred.shape:
        raise ValueError("Shapes of y_true and y_pred must match.")
    return np.sqrt(np.mean((y_true - y_pred) ** 2))


def MSE(y_true, y_pred):
    if y_true.shape != y_pred.shape:
        raise ValueError("Shapes of y_true and y_pred must match.")
    return np.mean((y_true - y_pred) ** 2)
