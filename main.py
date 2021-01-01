import pandas as pd
import numpy as np


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


def load_data(filepath, n_features):
    data = pd.read_csv(filepath, header=None)
    x = data.values[:, :n_features]
    y = data.values[:, n_features]
    return x, y


def foo():
    x, y = load_data('data/iris.csv', 4)
    dataset_size = len(y)
    y_encoded = one_hot_encode(y)
    all_indices_folded = make_crossvalidation_indices(dataset_size, 4)
    all_indices = np.arange(dataset_size)

    for validation_indices in all_indices_folded:
        train_indices = np.delete(all_indices, validation_indices)
        train_x, train_y = x[train_indices], y_encoded[train_indices]
        validation_x, validation_y = x[validation_indices], y_encoded[validation_indices]


if __name__ == '__main__':
    foo()


