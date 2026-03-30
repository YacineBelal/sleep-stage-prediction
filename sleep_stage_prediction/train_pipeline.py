import fire
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from sleep_stage_prediction.data import DreamtDataset, Workflow, load_dreamt
from sleep_stage_prediction.models import ConvolutionalClassifier, test_model, train_model


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
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    workflow = workflow if isinstance(workflow, Workflow) else Workflow[workflow]
    X_train, X_test, y_train, y_test = load_dreamt(
        nb_patients, workflow=workflow, frequency=frequency, seed=seed
    )

    train_dl = DataLoader(
        DreamtDataset(X_train, y_train),
        batch_size=batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(seed),
    )
    test_dls = [
        DataLoader(DreamtDataset(x, y), batch_size=256)
        for x, y in zip(X_test, y_test)
    ]

    model = ConvolutionalClassifier(channel_in=7, kernel_size=7).to(DEVICE)
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum)
    criterion = nn.CrossEntropyLoss(reduction="sum")

    train_model(model, train_dl, optimizer, criterion, epochs, DEVICE)
    test_model(model, test_dls, criterion, device=DEVICE)


if __name__ == "__main__":
    fire.Fire(main)
