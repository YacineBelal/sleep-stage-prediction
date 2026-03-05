import fire
import torch
import torch.nn as nn

from data import Workflow, load_dreamt
from models import ConvolutionalClassifier, test_model, train_model


def main(
    nb_patients=1,
    workflow=Workflow.CENTRALIZED,
    frequency=64,
    epochs=1,
    batch_size=128,
    lr=0.001,
    momentum=0.9,
    seed=42,
):
    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    workflow = workflow if type(workflow) is Workflow else Workflow[workflow]
    X_train, X_test, y_train, y_test = load_dreamt(
        nb_patients, workflow=workflow, frequency=frequency, seed=seed
    )

    model = ConvolutionalClassifier(channel_in=7, kernel_size=7)
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum)
    criterion = nn.CrossEntropyLoss(reduction="sum")

    train_model(model, X_train, y_train, optimizer, criterion, epochs, batch_size, lr, DEVICE)

    test_model(model, X_test, y_test, criterion, workflow, device=DEVICE)

if __name__ == "__main__":
    fire.Fire(main)
