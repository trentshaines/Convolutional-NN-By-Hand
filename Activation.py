import numpy as np
from Layer import Layer


class Activation(Layer):
    def __init__(self, A, dA_dX):
        self.A = A
        self.dA_dX = dA_dX

    def forward_prop(self, X):
        self.X = X
        return self.A(self.X)

    def backward_prop(self, dE_dY, alpha):
        return np.multiply(dE_dY, self.dA_dX(self.X))
