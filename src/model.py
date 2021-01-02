from src.layer import Layer
from src.callback import print_loss

import numpy as np
from typing import Callable
from sklearn.metrics import mean_squared_error


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
            predictions = []
            for i in range(len(x)):
                predictions.append(self.predict(x[i]))
                loss = loss_function(y[i], predictions[i])
                # print(loss)
                for layer in reversed(self._layers):
                    loss = layer.backpropagation(loss, learning_rate)
            self._on_epoch_end(y, predictions, epoch)

    @staticmethod
    def _on_epoch_end(groundtruth, predictions, epoch):
        loss = mean_squared_error(groundtruth, np.reshape(np.array(predictions), groundtruth.shape))
        print_loss(epoch, loss)
