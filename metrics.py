import numpy as np


def MAE(y_true, y_pred):
    return np.mean(np.abs(y_true - y_pred))


def RMSE(y_true, y_pred):
    return np.sqrt(np.mean((y_true - y_pred) ** 2))


def MSE(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)
