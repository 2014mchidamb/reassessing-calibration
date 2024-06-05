import datasets
import numpy as np
import torch

from typing import Any, Tuple


def kernel_regression(
    x: torch.FloatTensor,
    preds: torch.FloatTensor,
    labels: torch.LongTensor,
    kernel: torch.nn.Module,
) -> Tuple[torch.FloatTensor, torch.FloatTensor]:
    """Estimates conditional expectation E[Y | Pred] using NW regression.

    Args:
        x (torch.FloatTensor): Values at which to estimate conditional expectation.
        preds (torch.FloatTensor): Predictions on dataset.
        labels (torch.LongTensor): Labels for same dataset.
        kernel (torch.nn.Module): Kernel to use.

    Raises:
        ValueError: Preds and labels don't have same shape.
        ValueError: x is not 2-dimensional.

    Returns:
        Tuple[torch.FloatTensor, torch.FloatTensor]: Kernel regression estimates, kernel density estimates.
    """
    if preds.shape != labels.shape:
        raise ValueError("Preds and labels need to have the same shape.")
    elif len(x.shape) != 2:
        raise ValueError("x needs to be 2-dimensional.")
    weights = kernel(x.unsqueeze(dim=1) - preds)
    kde = weights.mean(dim=1)
    reg = (weights * labels.unsqueeze(dim=0)).mean(dim=1) / kde
    return reg, kde


def get_preds(
    model: torch.nn.Module,
    data: torch.FloatTensor,
    batch_size: int,
    device: str = "cpu",
) -> torch.FloatTensor:
    """Computes predictiosn of model on data.

    Args:
        model (torch.nn.Module): Model to use.
        data (torch.FloatTensor): Data to consider.
        batch_size (int): Batch size if data is large.
        device (str, optional): Device. Defaults to "cpu".

    Returns:
        torch.FloatTensor: Predictions.
    """
    model.eval()
    model.to(device)
    if len(data) < batch_size:
        with torch.no_grad():
            return model(data.to(device))
    dl = torch.utils.data.DataLoader(data, batch_size=batch_size, shuffle=False)
    preds = []
    with torch.no_grad():
        for x in dl:
            preds.append(model(x.to(device)))
    return torch.cat(preds)


def get_logits_and_labels_stream(
    model: torch.nn.Module,
    dataset: datasets.IterableDataset,
    transforms: Any,
    cutoff: int = None,
    device: str = "cpu",
) -> Tuple[torch.FloatTensor, torch.LongTensor]:
    """Gets model logits on a HuggingFace streaming dataset.

    Args:
        model (torch.nn.Module): Model to use.
        dataset (datasets.IterableDataset): HuggingFace dataset.
        transforms (Any): Transforms to apply to data.
        cutoff (int, optional): Cutoff for dataset. Defaults to None.
        device (str, optional): Device. Defaults to "cpu".

    Returns:
        Tuple[torch.FloatTensor, torch.LongTensor]: Logits, labels.
    """
    model.eval()
    model.to(device)
    logits, labels = [], []
    seen = 0
    with torch.no_grad():
        for data in dataset:
            img = data["jpg"].convert("RGB")
            x, target = transforms(img).unsqueeze(0).to(device), data["cls"]
            logits.append(model(x))
            labels.append(target)
            seen += 1
            if cutoff is not None and seen == cutoff:
                break

    logits = torch.cat(logits)
    labels = torch.LongTensor(labels).to(device)
    return logits, labels


def get_preds_and_labels_stream(
    model: torch.nn.Module,
    dataset: datasets.IterableDataset,
    transforms: Any,
    cutoff: int = None,
    device: str = "cpu",
) -> Tuple[torch.FloatTensor, torch.LongTensor]:
    """Gets model predictions on a HuggingFace streaming dataset.

    Args:
        model (torch.nn.Module): Model to use.
        dataset (datasets.IterableDataset): HuggingFace dataset.
        transforms (Any): Transforms to apply to data.
        cutoff (int, optional): Cutoff for dataset. Defaults to None.
        device (str, optional): Device. Defaults to "cpu".

    Returns:
        Tuple[torch.FloatTensor, torch.LongTensor]: Predictions, labels.
    """
    logits, labels = get_logits_and_labels_stream(model, dataset, transforms, cutoff, device)
    return torch.nn.functional.softmax(logits, dim=1), labels


def get_binarized_preds_and_labels(
    preds: torch.FloatTensor,
    labels: torch.LongTensor,
) -> Tuple[torch.FloatTensor, torch.LongTensor]:
    """Converts predictions and labels to confidence calibration problem.

    Args:
        preds (torch.FloatTensor): Model predictions (probabilities).
        labels (torch.LongTensor): Labels.

    Returns:
        Tuple[torch.FloatTensor, torch.LongTensor]: Max probs, correctness labels.
    """
    max_probs = preds[torch.arange(len(preds)), preds.argmax(dim=1)].unsqueeze(dim=1)
    accs = (preds.argmax(dim=1, keepdims=True) == labels).long()
    return max_probs, accs
