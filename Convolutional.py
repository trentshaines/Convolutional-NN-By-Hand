import numpy as np
from scipy import signal
from Layer import Layer


class Convolutional(Layer):
    def __init__(self, X_shape, K_size, Y_depth):
        X_depth, X_height, X_width = X_shape
        self.Y_depth = Y_depth
        self.X_shape = X_shape
        self.X_depth = X_depth
        self.Y_shape = (Y_depth, X_height - K_size + 1, X_width - K_size + 1)
        self.K_shape = (Y_depth, X_depth, K_size, K_size)
        self.K = np.random.randn(*self.K_shape)
        self.biases = np.random.randn(*self.Y_shape)

    def forward_prop(self, X):
        self.X = X
        self.Y = np.copy(self.biases)
        for i in range(self.Y_depth):
            for j in range(self.X_depth):
                self.Y[i] += signal.correlate2d(self.X[j], self.K[i, j], "valid")
        return self.Y

    def backward_prop(self, dE_dY, alpha):
        dE_dK = np.zeros(self.K_shape)
        dE_dX = np.zeros(self.X_shape)

        for i in range(self.Y_depth):
            for j in range(self.X_depth):
                dE_dK[i, j] = signal.correlate2d(self.X[j], dE_dY[i], "valid")
                dE_dX[j] += signal.convolve2d(dE_dY[i], self.K[i, j], "full")

        self.K -= alpha * dE_dK
        self.biases -= alpha * dE_dY
        return dE_dX
