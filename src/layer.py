from abc import ABC, abstractmethod


class Layer(ABC):
    def __init__(self):
        self._input = None
        self._output = None
        super().__init__()

    @abstractmethod
    def feedforward(self, input_vector):
        pass

    @abstractmethod
    def backpropagation(self, output_error):
        pass
