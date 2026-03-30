import torch
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader


def test_model(model, test_dls, criterion, batch_size=256, device=torch.device("cpu")):
    """Evaluate a model across one or more test DataLoaders.

    Works with both single-modal loaders (batches of ``(X, y)``) and
    multi-modal loaders (batches of ``(x_bvp, x_acc, x_eda_temp, x_hr, y)``).

    Args:
        test_dls: a single ``DataLoader`` or a list of ``DataLoader``s
                  (one per client for federated evaluation).
    """
    if isinstance(test_dls, DataLoader):
        test_dls = [test_dls]

    generalization_error, accuracy, f1_scores = [], [], []
    for test_dl in test_dls:
        gen_err, acc, score = _test_model(model, test_dl, criterion, device)
        generalization_error.append(gen_err)
        accuracy.append(acc)
        f1_scores.append(score)

    return {
        "Generalization Error": sum(generalization_error) / len(generalization_error),
        "Accuracy": sum(accuracy) / len(accuracy),
        "F1-Score": sum(f1_scores) / len(f1_scores),
    }


def _test_model(model, test_dl, criterion, device=torch.device("cpu")):
    generalization_error = 0.0
    correct = 0
    all_preds = []
    all_targets = []
    model.eval()

    with torch.no_grad():
        for *inputs, y in test_dl:
            inputs = [x.to(device) for x in inputs]
            y = y.to(device)
            logits = model(*inputs)
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
