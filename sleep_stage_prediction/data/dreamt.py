from pathlib import Path

import numpy as np
import pandas as pd

from .utils import federate_data

COLS_TO_DROP = [
    "IBI",
    "TIMESTAMP",
    "Obstructive_Apnea",
    "Central_Apnea",
    "Hypopnea",
    "Multiple_Events",
]

PROJECT_ROOT = Path(__file__).resolve().parents[2]

# TODO: remove workflow if not used
def load_dreamt(nb_patients, frequency=64, seed=42):
    # TODO: a dataclass to encapsulate rng etc
    dataset_name = "dreamt"
    rng = np.random.default_rng(seed=seed)
    signals, labels = _load_dreamt(
        nb_patients,
        rng,
        frequency,
    )
    signals_preprocessed, labels_preprocessed = _preprocess_dreamt(signals, labels)
    split_data = federate_data(signals_preprocessed, labels_preprocessed, dataset_name, rng)
    X_train = []
    X_test = []
    y_train = []
    y_test = []
    for client_data in split_data:
        X_train.append(client_data[0])
        X_test.append(client_data[1])
        y_train.append(client_data[2])
        y_test.append(client_data[3])

    X_train = np.concat(X_train, axis=0)
    y_train = np.concat(y_train, axis=0)

    return X_train, X_test, y_train, y_test


def _load_dreamt(
    nb_patients,
    rng,
    frequency=64,
):
    # TODO: download dataset if not present
    folder = PROJECT_ROOT / "data" / "raw" / "dreamt" / f"data_{frequency}Hz"
    if not folder.exists():
        raise FileNotFoundError("Data Folder does not exist")

    patient_file_list = [f for f in folder.iterdir() if f.is_file()]
    rng.shuffle(patient_file_list)
    labels = []
    signals = []
    nb_patients = min(100, nb_patients)
    for _ in range(nb_patients):
        patient_file = patient_file_list.pop()
        df = pd.read_csv(patient_file)
        df["Sleep_Stage"] = df["Sleep_Stage"].replace("P", "W")
        df = df.drop(columns=COLS_TO_DROP)
        df = df[df["Sleep_Stage"] != "Missing"]
        labels.append(df["Sleep_Stage"].to_numpy())
        signals.append(df.drop(columns=["Sleep_Stage"]).to_numpy())

    return signals, labels


def _preprocess_dreamt(signals, labels, signal_len=64):
    signals_preprocessed = []
    labels_preprocessed = []
    for X_p, y_p in zip(signals, labels):
        signals_preprocessed.append(X_p[:-1].reshape(-1, signal_len, 7))
        labels_preprocessed.append(y_p[:-1].reshape(-1, signal_len)[:, 0])

    classes_n = np.unique(np.concat(labels_preprocessed, axis=0))
    label_encoder = {val: idx for idx, val in enumerate(classes_n)}
    labels_preprocessed_encoded = [
        np.array([label_encoder[val] for val in patient_labels])
        for patient_labels in labels_preprocessed
    ]

    return signals_preprocessed, labels_preprocessed_encoded
