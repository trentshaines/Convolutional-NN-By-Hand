import numpy as np
from Layer import Layer

class Reshape(Layer):
    def __init__(self, X_shape, Y_shape):
        self.X_shape = X_shape
        self.Y_shape = Y_shape

    def forward_prop(self, X):
        return np.reshape(X, self.Y_shape)

    def backward_prop(self, dE_dY, alpha):
        return np.reshape(dE_dY, self.X_shape)