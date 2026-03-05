import torch
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, Lambda
from tqdm import tqdm

from data import DreamtDataset

__all__ = ["train_model"]

def train_model(
    model, X_train, y_train, optimizer, criterion, epochs, batch_size=128, device="cpu"
):
    # TODO: this transform shouldn't be here
    transform = Compose(
        [
            torch.FloatTensor,
            Lambda(lambda x: x.permute([1, 0])),
        ]
    )
    train_ds = DreamtDataset(X_train, y_train, transform)

    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    model.to(device)
    for epoch in tqdm(range(epochs)):
        model.train()
        empirical_risk = 0.0
        for X, y in train_dl:
            X = X.to(device)
            y = y.to(device)
            optimizer.zero_grad()
            pred = model(X)
            loss = criterion(pred, y)
            loss.backward()
            optimizer.step()
            empirical_risk += loss.item()

        empirical_risk /= len(train_dl.dataset)
        print(f"Train loss: {empirical_risk}")
