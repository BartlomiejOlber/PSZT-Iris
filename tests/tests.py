from src.utils import load_data, one_hot_encode, make_crossvalidation_indices, eval
from src.model import Model
from src.activation_layer import ActivationLayer
from src.fully_connected_layer import FullyConnectedLayer

import src.math_functions as fn

import numpy as np
import pandas as pd

SIGMOID_LR = 0.13
TANH_LR = 0.02
EPOCHS = 400


def find_lr(crossval_n, features_n):
    x, y = load_data('data/iris.csv', features_n)
    dataset_size = len(y)
    y_encoded = one_hot_encode(y)
    all_indices_folded = make_crossvalidation_indices(dataset_size, crossval_n)
    all_indices = np.arange(dataset_size)

    # for layer in np.arange(1,4):
    model = Model()
    model.add(FullyConnectedLayer(x.shape[-1], 12))
    # model.add(ActivationLayer(activation_function=fn.sigmoid, function_derivative=fn.sigmoid_derivative))
    model.add(ActivationLayer(activation_function=fn.tanh, function_derivative=fn.tanh_derivative))
    model.add(FullyConnectedLayer(12, 4))
    # model.add(ActivationLayer(activation_function=fn.sigmoid, function_derivative=fn.sigmoid_derivative))
    model.add(ActivationLayer(activation_function=fn.tanh, function_derivative=fn.tanh_derivative))
    model.add(FullyConnectedLayer(4, 10))
    # model.add(ActivationLayer(activation_function=fn.sigmoid, function_derivative=fn.sigmoid_derivative))
    model.add(ActivationLayer(activation_function=fn.tanh, function_derivative=fn.tanh_derivative))
    model.add(FullyConnectedLayer(10, y_encoded.shape[-1]))
    rows = []
    for lr in np.arange(0.005, 0.07, 0.005):
        for epochs in np.arange(50, 751, 50):
            accuracy = []
            for i, validation_indices in enumerate(all_indices_folded):
                train_indices = np.delete(all_indices, validation_indices)
                train_x, train_y = x[train_indices], y_encoded[train_indices]
                validation_x, validation_y = x[validation_indices], y_encoded[validation_indices]
                train_x.resize((train_x.shape[0], 1, train_x.shape[1]))
                model.train(train_x, train_y, fn.mse_derivative, learning_rate=TANH_LR, epochs=EPOCHS)
                accuracy.append(eval(model, validation_x, validation_y, i, crossval_n))
                model.clear_weights()
            avg_acc = np.array(accuracy).mean()
            print(f"lr: {lr} epochs: {epochs} accuracy: {avg_acc}")
            rows.append((lr, epochs, avg_acc))
    pd.DataFrame(rows).to_csv("results_2tanh.csv", index=False, header=False)


def test_architecture(crossval_n, features_n):
    x, y = load_data('data/iris.csv', features_n)
    dataset_size = len(y)
    y_encoded = one_hot_encode(y)
    all_indices_folded = make_crossvalidation_indices(dataset_size, crossval_n)
    all_indices = np.arange(dataset_size)
    # one layer
    rows = []
    for neurons in np.arange(4, 50, 2):
        model = Model()
        model.add(FullyConnectedLayer(x.shape[-1], neurons))
        model.add(ActivationLayer(activation_function=fn.tanh, function_derivative=fn.tanh_derivative))
        model.add(FullyConnectedLayer(neurons, y_encoded.shape[-1]))
        accuracy = []
        for i, validation_indices in enumerate(all_indices_folded):
            train_indices = np.delete(all_indices, validation_indices)
            train_x, train_y = x[train_indices], y_encoded[train_indices]
            validation_x, validation_y = x[validation_indices], y_encoded[validation_indices]
            train_x.resize((train_x.shape[0], 1, train_x.shape[1]))
            model.train(train_x, train_y, fn.mse_derivative, learning_rate=TANH_LR, epochs=EPOCHS)
            accuracy.append(eval(model, validation_x, validation_y, i, crossval_n))
            model.clear_weights()
        avg_acc = np.array(accuracy).mean()
        print(f"neurons: {neurons} accuracy: {avg_acc}")
        rows.append((neurons, avg_acc))
    pd.DataFrame(rows).to_csv("single_layer_neurons.csv", index=False, header=False)
    # high to low
    rows = []
    for layers in np.arange(1, 4):
        for neurons in np.arange(8, 65, 8):
            model = Model()
            model.add(FullyConnectedLayer(x.shape[-1], neurons))
            model.add(ActivationLayer(activation_function=fn.tanh, function_derivative=fn.tanh_derivative))
            for i in range(layers):
                model.add(FullyConnectedLayer(int(neurons/pow(2, i)), int(neurons/pow(2, i+1))))
                model.add(ActivationLayer(activation_function=fn.tanh, function_derivative=fn.tanh_derivative))
            model.add(FullyConnectedLayer(int(neurons/pow(2, layers)), y_encoded.shape[-1]))
            accuracy = []
            for i, validation_indices in enumerate(all_indices_folded):
                train_indices = np.delete(all_indices, validation_indices)
                train_x, train_y = x[train_indices], y_encoded[train_indices]
                validation_x, validation_y = x[validation_indices], y_encoded[validation_indices]
                train_x.resize((train_x.shape[0], 1, train_x.shape[1]))
                model.train(train_x, train_y, fn.mse_derivative, learning_rate=TANH_LR, epochs=EPOCHS)
                accuracy.append(eval(model, validation_x, validation_y, i, crossval_n))
                model.clear_weights()
            avg_acc = np.array(accuracy).mean()
            print(f"hidden layers: {layers + 1} neurons in first layer: {neurons} accuracy: {avg_acc}")
            rows.append((neurons,layers+1 , avg_acc))
    pd.DataFrame(rows).to_csv("high_to_low.csv", index=False, header=False)
    # low to high
    rows = []
    for layers in np.arange(1, 4):
        for neurons in np.arange(3, 9):
            model = Model()
            model.add(FullyConnectedLayer(x.shape[-1], neurons))
            model.add(ActivationLayer(activation_function=fn.tanh, function_derivative=fn.tanh_derivative))
            for i in range(layers):
                model.add(FullyConnectedLayer(neurons * pow(2, i), neurons * pow(2, i+1)))
                model.add(ActivationLayer(activation_function=fn.tanh, function_derivative=fn.tanh_derivative))
            model.add(FullyConnectedLayer(neurons * pow(2, layers), y_encoded.shape[-1]))
            accuracy = []
            for i, validation_indices in enumerate(all_indices_folded):
                train_indices = np.delete(all_indices, validation_indices)
                train_x, train_y = x[train_indices], y_encoded[train_indices]
                validation_x, validation_y = x[validation_indices], y_encoded[validation_indices]
                train_x.resize((train_x.shape[0], 1, train_x.shape[1]))
                model.train(train_x, train_y, fn.mse_derivative, learning_rate=TANH_LR, epochs=EPOCHS)
                accuracy.append(eval(model, validation_x, validation_y, i, crossval_n))
                model.clear_weights()
            avg_acc = np.array(accuracy).mean()
            print(f"hidden layers: {layers + 1} neurons in first layer: {neurons} accuracy: {avg_acc}")
            rows.append((neurons,layers+1 , avg_acc))
    pd.DataFrame(rows).to_csv("low_to_high.csv", index=False, header=False)
    # encoder-decoder
    rows = []
    for layers in np.arange(1, 3):
        for neurons in np.arange(8, 65, 8):
            model = Model()
            model.add(FullyConnectedLayer(x.shape[-1], neurons))
            model.add(ActivationLayer(activation_function=fn.tanh, function_derivative=fn.tanh_derivative))
            for i in range(layers):
                model.add(FullyConnectedLayer(int(neurons/pow(2, i)), int(neurons/pow(2, i+1))))
                model.add(ActivationLayer(activation_function=fn.tanh, function_derivative=fn.tanh_derivative))
            middle_layer_neurons = int(neurons/pow(2, layers))
            for i in range(layers):
                model.add(FullyConnectedLayer(middle_layer_neurons * pow(2, i), middle_layer_neurons * pow(2, i+1)))
                model.add(ActivationLayer(activation_function=fn.tanh, function_derivative=fn.tanh_derivative))
            model.add(FullyConnectedLayer(middle_layer_neurons * pow(2, layers), y_encoded.shape[-1]))
            accuracy = []
            for i, validation_indices in enumerate(all_indices_folded):
                train_indices = np.delete(all_indices, validation_indices)
                train_x, train_y = x[train_indices], y_encoded[train_indices]
                validation_x, validation_y = x[validation_indices], y_encoded[validation_indices]
                train_x.resize((train_x.shape[0], 1, train_x.shape[1]))
                model.train(train_x, train_y, fn.mse_derivative, learning_rate=TANH_LR, epochs=EPOCHS)
                accuracy.append(eval(model, validation_x, validation_y, i, crossval_n))
                model.clear_weights()
            avg_acc = np.array(accuracy).mean()
            print(f"hidden layers: {2 * layers + 1} neurons in first layer: {neurons} accuracy: {avg_acc}")
            rows.append((neurons,layers+1, avg_acc))
    pd.DataFrame(rows).to_csv("encoder_decoder.csv", index=False, header=False)


