import pandas as pd
import numpy as np
from typing import Tuple

from src.activation_layer import ActivationLayer
from src.fully_connected_layer import FullyConnectedLayer
from src.model import Model
import src.math_functions as fn


def make_crossvalidation_indices(dataset_size, n_folds):
    idx_all_permute = np.random.permutation(dataset_size)
    fold_size = int(dataset_size/n_folds)
    folds = []
    for fold_idx in range(n_folds):
        start_idx = fold_idx*fold_size
        end_idx = (fold_idx+1)*fold_size if fold_idx < n_folds - 1 else dataset_size
        folds.append(idx_all_permute[start_idx:end_idx])
    return folds


def one_hot_encode(labels):
    from sklearn.preprocessing import OneHotEncoder, LabelEncoder
    label_encoder = LabelEncoder()
    integer_encoded = label_encoder.fit_transform(labels)
    onehot_encoder = OneHotEncoder(sparse=False)
    integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
    onehot_encoded = onehot_encoder.fit_transform(integer_encoded)
    return onehot_encoded


def load_data(filepath, n_features) -> Tuple[np.ndarray, np.ndarray]:
    data = pd.read_csv(filepath, header=None)
    x = data.values[:, :n_features]
    y = data.values[:, n_features]
    return x.astype('float64'), y


def foo(crossval_n, features_n):
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

        model.add(FullyConnectedLayer(x.shape[-1], 8))
        # model.add(ActivationLayer(activation_function=fn.sigmoid, function_derivative=fn.sigmoid_derivative))
        model.add(ActivationLayer(activation_function=fn.tanh, function_derivative=fn.tanh_derivative))
        model.add(FullyConnectedLayer(8, 5))
        # model.add(ActivationLayer(activation_function=fn.sigmoid, function_derivative=fn.sigmoid_derivative))
        model.add(ActivationLayer(activation_function=fn.tanh, function_derivative=fn.tanh_derivative))
        model.add(FullyConnectedLayer(5, train_y.shape[-1]))
        model.train(train_x, train_y, fn.mse_derivative, learning_rate=0.1, epochs=100)
        accuracy.append(eval(model, validation_x, validation_y, i, crossval_n))
    print(f"Avg crossvalidation accuracy: {np.array(accuracy).mean()}")


def eval(model, val_x, val_y, val_id, val_n):
    right_count = 0
    for i in range(len(val_x)):
        pred = model.predict(val_x[i])
        print(f"true: {val_y[i]}")
        print(f"true: {pred}")
        if np.argmax(pred) == np.argmax(val_y[i]):
            right_count += 1
    acc_percentage = 100*right_count/len(val_x)
    print(f"crossvalidation: {val_id+1}/{val_n} correct: {right_count} out of {len(val_x)}\n - {acc_percentage} %")
    return acc_percentage


if __name__ == '__main__':
    foo(5, 4)


