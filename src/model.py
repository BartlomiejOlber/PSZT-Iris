from src.layer import Layer
from src.callback import print_loss

import numpy as np
from typing import Callable
from sklearn.metrics import mean_squared_error
from tqdm import tqdm


class Model(object):
    def __init__(self):
        self._layers = []

    def clear_weights(self):
        for layer in self._layers:
            layer.clear()

    def add(self, layer: Layer):
        self._layers.append(layer)

    def predict(self, features: np.ndarray) -> np.ndarray:
        layer_output = features
        for layer in self._layers:
            layer_output = layer.feedforward(layer_output)
        return layer_output

    def train(self, x: np.ndarray, y: np.ndarray, loss_function: Callable, learning_rate: float, epochs: int,
              print_loss: bool = False):
        epochs_iterator = tqdm(range(epochs)) if not print_loss else range(epochs)
        for epoch in epochs_iterator:
            predictions = []
            for i in range(len(x)):
                prediction = self.predict(x[i])
                predictions.append(prediction)
                loss = loss_function(y[i], prediction)
                # print(loss)
                for layer in reversed(self._layers):
                    loss = layer.backpropagation(loss, learning_rate)
            if print_loss:
                self._on_epoch_end(y, predictions, epoch)

    @staticmethod
    def _on_epoch_end(groundtruth, predictions, epoch):
        loss = mean_squared_error(groundtruth, np.reshape(np.array(predictions), groundtruth.shape))
        print_loss(epoch, loss)
