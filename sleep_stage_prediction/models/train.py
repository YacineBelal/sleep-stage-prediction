import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from sleep_stage_prediction.data import DreamtDataset

__all__ = ["train_model"]


def train_model(
    model, X_train, y_train, optimizer, criterion, epochs, batch_size=128, device="cpu", seed=42
):
    # TODO refactor dataloaders creation from train and evaluate
    train_ds = DreamtDataset(X_train, y_train)
    train_dl = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(seed),
    )

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
