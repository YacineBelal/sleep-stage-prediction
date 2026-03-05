import fire
import numpy as np
import torch
import torch.nn as nn
from torchvision.transforms import Compose, Lambda

from data import DreamtDataset, Workflow, load_dreamt
from models import ConvolutionnalClassifier, train_model


def main(
    nb_patients=1,
    workflow=Workflow.CENTRALIZED,
    frequency=64,
    epochs=10,
    batch_size=128,
    lr=0.001,
    momentum=0.9,
    seed=42,
):
    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    split_data = load_dreamt(nb_patients, mode=workflow, frequency=frequency, seed=seed)
    print(workflow)
    if workflow == Workflow.CENTRALIZED:
        X_train, X_test, y_train, y_test = split_data
    elif workflow == Workflow.FEDERATED_CROSS_DEVICE:
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
        X_test = np.concat(X_train, axis=0)
        y_train = np.concat(y_train, axis=0)
        y_test = np.concat(y_test, axis=0)
    else:
        raise TypeError(f"{workflow} is not a defined workflow.")

    transform = Compose(
        [
            torch.FloatTensor,
            Lambda(lambda x: x.permute([1, 0])),
        ]
    )

    train_ds = DreamtDataset(X_train, y_train, transform)
    model = ConvolutionnalClassifier(channel_in=7, kernel_size=7)
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum)
    criterion = nn.CrossEntropyLoss(reduction="sum")

    train_model(model, optimizer, criterion, train_ds, epochs, batch_size, lr, DEVICE)

# test_ds = DreamtDataset(X_test, y_test)

# test_dl = DataLoader()
if __name__ == "__main__":
    fire.Fire(main)