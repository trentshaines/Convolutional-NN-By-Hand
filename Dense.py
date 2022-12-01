import numpy as np
from Layer import Layer


class Dense(Layer):
    def __init__(self, X_size, Y_size):
        self.W = np.random.randn(Y_size, X_size)
        self.B = np.random.randn(Y_size, 1)

    def forward_prop(self, X):
        self.X = X
        return np.dot(self.W, self.X) + self.B

    def backward_prop(self, dE_dY, alpha):
        dE_dW = np.dot(dE_dY, self.X.T)
        dE_dX = np.dot(self.W.T, dE_dY)
        self.W -= alpha * dE_dW
        self.B -= alpha * dE_dY
        return dE_dX
