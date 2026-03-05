from torch.utils.data import DataLoader
from tqdm import tqdm

__all__ = ["train_model"]

def train_model(
    model, optimizer, criterion, train_ds, epochs, batch_size=128, lr=0.001, device="cpu"
):

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
