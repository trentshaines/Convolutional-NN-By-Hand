import numpy as np
from numpy import random
from keras.datasets import mnist
from keras.utils import np_utils
from matplotlib import pyplot

from Dense import Dense
from Convolutional import Convolutional
from Reshape import Reshape
from Activation_Functions import Sigmoid
from Loss_Functions import bce, dBce_dY
from Sequential import Sequential


def one_hot_keep_index(x, y, size):
    zero_index = np.where(y == 0)[0][:size]
    one_index = np.where(y == 1)[0][:size]
    all_indices = np.hstack((zero_index, one_index))
    all_indices = np.random.permutation(all_indices)

    x, y = x[all_indices], y[all_indices]
    x = x.reshape(len(x), 1, 28, 28)
    x = x.astype("float32") / 255
    y = np_utils.to_categorical(y)
    y = y.reshape(len(y), 2, 1)
    return x, y, all_indices


output_dim = 4
(x_train, y_train), (x_test, y_test) = mnist.load_data()
original_x_test = x_test
x_train, y_train, blank = one_hot_keep_index(x_train, y_train, 500)
x_test, y_test, img_indices = one_hot_keep_index(x_test, y_test, (output_dim * output_dim) // 2)

network = Sequential([
    Convolutional((1, 28, 28), 3, 5),
    Sigmoid(),
    Reshape((5, 26, 26), (5 * 26 * 26, 1)),
    Dense(5 * 26 * 26, 100),
    Sigmoid(),
    Dense(100, 2),
    Sigmoid()
])

network.set_loss(bce, dBce_dY)

# train network
network.fit(
    x_train,
    y_train,
    epochs=20,
    alpha=0.1
)

# test - plot images with predictions
i = 0
fig, axs = pyplot.subplots(output_dim, output_dim, sharex=True, sharey=True)
for x, y, index in zip(x_test, y_test, img_indices):
    output = network.predict(x)
    axs[i // output_dim][i % output_dim].imshow(original_x_test[index], cmap=pyplot.get_cmap('gray'))
    axs[i // output_dim][i % output_dim].set_title(f"Predicted: {np.argmax(output)}")
    i += 1

pyplot.tight_layout()
pyplot.show()
