from sklearn.metrics import f1_score
import torch
from torch.utils.data import DataLoader

from sleep_stage_prediction.data import DreamtDataset


def test_model(model, X_test, y_test, criterion, batch_size=256, device=torch.device("cpu")):

    # TODO refactor dataloaders creation from train and evaluate
    test_ds_clients = [DreamtDataset(x, y) for x, y in zip(X_test, y_test)]
    test_dl_clients = [
        DataLoader(test_ds, batch_size=batch_size, shuffle=True) for test_ds in test_ds_clients
    ]
    generalization_error = []
    accuracy = []
    f1_scores = []
    for test_dl in test_dl_clients:
        gen_err, acc, score = _test_model(model, test_dl, criterion, batch_size, device)
        generalization_error.append(gen_err)
        accuracy.append(acc)
        f1_scores.append(score)

    generalization_error = sum(generalization_error) / len(generalization_error)
    accuracy = sum(accuracy) / len(accuracy)
    score = sum(f1_scores) / len(f1_scores)

    return {
        "Generalization Error": {generalization_error},
        "Accuracy": {accuracy},
        "F1-Score": {score},
    }


def _test_model(model, test_dl, criterion, batch_size=256, device=torch.device("cpu")):
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

        y_pred = torch.cat(all_preds)
        y_true = torch.cat(all_targets)
        generalization_error /= len(test_dl.dataset)
        accuracy = correct / len(test_dl.dataset)

    f1score = f1_score(y_true, y_pred, average="macro")

    return generalization_error, accuracy, f1score
