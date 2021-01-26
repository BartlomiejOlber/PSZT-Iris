import numpy as np
import pandas as pd
from typing import Tuple


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


def eval(model, val_x, val_y, val_id, val_n):
    right_count = 0
    for i in range(len(val_x)):
        pred = model.predict(val_x[i])
        # print(f"true: {val_y[i]}")
        # print(f"true: {pred}")
        if np.argmax(pred) == np.argmax(val_y[i]):
            right_count += 1
    acc_percentage = 100*right_count/len(val_x)
    print(f"\n crossvalidation: {val_id+1}/{val_n} correct: {right_count} out of {len(val_x)} - {acc_percentage} % ")
    return acc_percentage
