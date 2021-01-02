import numpy as np


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoid_derivative(x):
    f = 1 / (1 + np.exp(-x))
    return f * (1 - f)


def tanh(x):
    return np.tanh(x)


def tanh_derivative(x):
    return 1 - np.tanh(x) ** 2


def mse_derivative(groundtruth, prediction):
    return 2 * (prediction - groundtruth) / groundtruth.size
