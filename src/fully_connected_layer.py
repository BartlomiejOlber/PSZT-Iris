from src.layer import Layer
import numpy as np


class FullyConnectedLayer(Layer):
    def __init__(self, input_size: int, output_size: int):
        super().__init__()
        self._input_size = input_size
        self._output_size = output_size
        self._uniform_max_value = 1./np.sqrt(input_size)
        self._weights = np.random.rand(input_size, output_size) * 2 * self._uniform_max_value - self._uniform_max_value
        self._bias = np.random.rand(1, output_size) * 2 * self._uniform_max_value - self._uniform_max_value

    def feedforward(self, input_vector: np.ndarray) -> np.ndarray:
        self._input = input_vector
        self._output = np.dot(self._input, self._weights) + self._bias
        return self._output

    def backpropagation(self, output_error: np.ndarray, learning_rate: float = None) -> np.ndarray:
        input_error = np.dot(output_error, self._weights.transpose())
        weights_error = np.dot(self._input.transpose(), output_error)
        self._weights -= weights_error * learning_rate
        self._bias -= output_error * learning_rate
        return input_error

    def clear(self):
        self._weights = np.random.rand(self._input_size, self._output_size) * 2 * self._uniform_max_value - self._uniform_max_value
        self._bias = np.random.rand(1, self._output_size) * 2 * self._uniform_max_value - self._uniform_max_value
        self._input = None
        self._output = None
