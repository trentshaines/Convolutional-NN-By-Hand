import numpy as np


def bce(Y_true, Y_pred):
    return np.mean(-Y_true * np.log(Y_pred) - (1 - Y_true) * np.log(1 - Y_pred))


def dBce_dY(Y_true, Y_pred):
    return ((1 - Y_true) / (1 - Y_pred) - Y_true / Y_pred) / np.size(Y_true)


def cce(Y_true, Y_pred):
    pass  # todo for all digit classification


def dCce_dY(Y_true, Y_pred):
    pass  # todo for all digit classification
