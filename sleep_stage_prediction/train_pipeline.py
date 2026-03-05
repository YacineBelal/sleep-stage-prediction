import fire
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
    workflow = workflow if type(workflow) is Workflow else Workflow[workflow]
    X_train, X_test, y_train, y_test = load_dreamt(
        nb_patients, workflow=workflow, frequency=frequency, seed=seed
    )

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
