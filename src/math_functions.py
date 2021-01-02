import numpy as np


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoid_derivative(x):
    f = 1 / (1 + np.exp(-x))
    return f * (1 - f)


def relu(x):
    return x * (x > 0)


def relu_derivative(x):
    return (x > 0) * 1


def mse_derivative(groundtruth, prediction):
    return 2 * (prediction - groundtruth) / groundtruth.size
