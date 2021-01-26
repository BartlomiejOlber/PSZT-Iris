import numpy as np

from src.utils import load_data, one_hot_encode, make_crossvalidation_indices, eval
from src.activation_layer import ActivationLayer
from src.fully_connected_layer import FullyConnectedLayer
from src.model import Model
import src.math_functions as fn
from tests.tests import find_lr, test_architecture


def train_and_validate(crossval_n, features_n):
    x, y = load_data('data/iris.csv', features_n)
    dataset_size = len(y)
    y_encoded = one_hot_encode(y)
    all_indices_folded = make_crossvalidation_indices(dataset_size, crossval_n)
    all_indices = np.arange(dataset_size)
    accuracy = []
    for i, validation_indices in enumerate(all_indices_folded):
        train_indices = np.delete(all_indices, validation_indices)
        train_x, train_y = x[train_indices], y_encoded[train_indices]
        validation_x, validation_y = x[validation_indices], y_encoded[validation_indices]
        model = Model()

        train_x.resize((train_x.shape[0], 1, train_x.shape[1]))

        model.add(FullyConnectedLayer(x.shape[-1], 40))
        # model.add(ActivationLayer(activation_function=fn.sigmoid, function_derivative=fn.sigmoid_derivative))
        model.add(ActivationLayer(activation_function=fn.tanh, function_derivative=fn.tanh_derivative))
        model.add(FullyConnectedLayer(40, 10))
        # model.add(ActivationLayer(activation_function=fn.sigmoid, function_derivative=fn.sigmoid_derivative))
        model.add(ActivationLayer(activation_function=fn.tanh, function_derivative=fn.tanh_derivative))
        model.add(FullyConnectedLayer(10, train_y.shape[-1]))
        model.train(train_x, train_y, fn.mse_derivative, learning_rate=0.005, epochs=600)
        accuracy.append(eval(model, validation_x, validation_y, i, crossval_n))
    print(f"Avg crossvalidation accuracy: {np.array(accuracy).mean()}")


if __name__ == '__main__':
    # train_and_validate(6, 4)
    # find_lr(6, 4)
    test_architecture(6, 4)

