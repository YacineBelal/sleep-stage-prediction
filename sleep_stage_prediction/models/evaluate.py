import torch
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, Lambda

from data import DreamtDataset, Workflow


def test_model(model, X_test, y_test, criterion, workflow, batch_size=256, device="cpu"):
    transform = Compose(
        [
            torch.FloatTensor,
            Lambda(lambda x: x.permute([1, 0])),
        ]
    )
    if workflow == Workflow.CENTRALIZED:
        test_ds = DreamtDataset(X_test, y_test, transform)
        test_dl = DataLoader(test_ds, batch_size=batch_size, shuffle=True)
        generalization_error, accuracy = _test_model(model, test_dl, criterion, batch_size, device)

    elif workflow == Workflow.FEDERATED_CROSS_DEVICE:
        test_ds_clients = [DreamtDataset(x, y, transform) for x, y in zip(X_test, y_test)]
        test_dl_clients = [
            DataLoader(test_ds, batch_size=batch_size, shuffle=True) for test_ds in test_ds_clients
        ]
        generalization_error = []
        accuracy = []
        for test_dl in test_dl_clients:
            gen_err, acc = _test_model(model, test_dl, criterion, batch_size, device)
            generalization_error.append(gen_err)
            accuracy.append(acc)

        generalization_error = sum(generalization_error) / len(generalization_error)
        accuracy = sum(accuracy) / len(accuracy)

    print(f"Generalization Error:{generalization_error}, Accuracy {accuracy}")
    return generalization_error, accuracy


def _test_model(model, test_dl, criterion, batch_size=256, device="cpu"):
    generalization_error = 0.0
    correct = 0
    all_preds = []
    all_targets = []
    model.eval()

    with torch.no_grad():
        for X, y in test_dl:
            X = X.to(device)
            y = y.to(device)
            logits = model(X)
            loss = criterion(logits, y)
            pred = torch.argmax(logits, dim=1)
            correct += (pred == y).sum().item()
            generalization_error += loss.item()
            all_preds.append(pred.cpu())
            all_targets.append(y.cpu())

        y_pred = torch.cat(all_preds).numpy()
        y_true = torch.cat(all_targets).numpy()
        generalization_error /= len(test_dl.dataset)
        accuracy = correct / len(test_dl.dataset)

    return generalization_error, accuracy
