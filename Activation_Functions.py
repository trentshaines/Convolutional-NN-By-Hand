import numpy as np
from Activation import Activation

class Sigmoid(Activation):
    def __init__(self):
        def sigmoid(x):
            return 1 / (1 + np.exp(-x))

        def dSigmoid_dX(x):
            s = sigmoid(x)
            return s * (1 - s)

        super().__init__(sigmoid, dSigmoid_dX)
