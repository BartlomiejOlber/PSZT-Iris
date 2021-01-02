from src.layer import Layer

import numpy as np
from typing import Callable


class Model(object):
    def __init__(self):
        self._layers = []

    def add(self, layer: Layer):
        self._layers.append(layer)

    def predict(self, features: np.ndarray) -> np.ndarray:
        layer_output = features
        for layer in self._layers:
            layer_output = layer.feedforward(layer_output)
        return layer_output

    def train(self, x: np.ndarray, y: np.ndarray, loss_function: Callable, learning_rate: float, epochs: int):
        for epoch in range(epochs):
            for i in range(len(x)):
                prediction = self.predict(x[i])
                loss = loss_function(y[i], prediction)
                for layer in reversed(self._layers):
                    loss = layer.backpropagation(loss, learning_rate)
