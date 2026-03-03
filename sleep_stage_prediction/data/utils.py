from enum import Enum
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]


class Architecture(Enum):
    FEDERATED_CROSS_DEVICE = 1 
    FEDERATED_CROSS_SILO = 2
    CENTRALIZED = 3



def save_data_array(file: Path , arr):
    file.parent.mkdir(parents=True, exist_ok=True)
    np.save(file, arr)
    

def _split_dataset(X, y, test_size=0.2, rng=None, shuffle=True):
    dataset_len = X.shape[0]
    idx = np.arange(dataset_len)
    if shuffle and rng:
        idx = rng.permutation(idx)

    X_train = X[idx[int(dataset_len * test_size) :]]

    y_train = y[idx[int(dataset_len * test_size) :]]

    X_test = X[idx[: int(dataset_len * test_size)]]
    y_test = y[idx[: int(dataset_len * test_size)]]

    return X_train, X_test, y_train, y_test


def _centralize_data(X, y, dataset, rng, test_size=0.2):
    X = np.concat(X)
    y = np.concat(y)
    X_train, X_test, y_train, y_test = _split_dataset(X, y, test_size, rng)
    folder = PROJECT_ROOT / "data" / "processed" / dataset
    save_data_array(
        folder / "server" / "train_data",
        np.permute_dims(X_train, axes=(0, 2, 1)).astype("float32"),
    )
    save_data_array(
        folder / "server" / "test_data",
        np.permute_dims(X_test, axes=(0, 2, 1)).astype("float32"),
    )
    save_data_array(
        folder / "server" / "train_target",
        y_train,
    )
    save_data_array(
        folder / "server" / "test_target",
        y_test,
    )


def _federate_data(X, y, dataset, rng, test_size=0.2):
    split_data = [
        _split_dataset(
            X[i],
            y[i],
            test_size,
            rng,
        )
        for i in range(len(X))
    ]
    folder = PROJECT_ROOT / "data" / "processed" / dataset
    for idx, (x_train, x_test, y_train, y_test) in enumerate(split_data):
        save_data_array(
            folder / f"client_{idx}" / "train_data",
            np.permute_dims(x_train, axes=(0, 2, 1)).astype("float32"),
        )
        save_data_array(folder / f"client_{idx}" / "train_target", y_train)
        save_data_array(
            folder / f"client_{idx}" / "test_data",
            np.permute_dims(x_test, axes=(0, 2, 1)).astype("float32"),
        )
        save_data_array(folder / f"client_{idx}" / "test_target", y_test)
