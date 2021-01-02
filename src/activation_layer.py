from src.layer import Layer
from typing import Callable


class ActivationLayer(Layer):
    def __init__(self, activation_function: Callable, function_derivative: Callable):
        super().__init__()
        self._activation_function = activation_function
        self._function_derivative = function_derivative

    def feedforward(self, input_vector):
        self._input = input_vector
        self._output = self._activation_function(self._input)
        return self._output

    def backpropagation(self, output_error, learning_rate=None):
        return output_error * self._function_derivative(self._input)
