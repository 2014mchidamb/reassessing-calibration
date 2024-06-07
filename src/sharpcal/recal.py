import numpy as np
import torch

from abc import ABC, abstractmethod


class BrierLoss(torch.nn.Module):
    """Squared loss on the one-hot-encoded labels."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
    
    def forward(self, x, y):
        one_hot_labels = torch.zeros_like(x)
        one_hot_labels[torch.arange(len(x)), y] = 1
        probs = torch.nn.functional.softmax(x, dim=1)
        return ((probs - one_hot_labels) ** 2).sum(dim=1).mean()


class LogitScaler(ABC):
    """Base class for temperature scaling, vector scaling, matrix scaling."""

    def __init__(self, device="cpu", log_results=False, use_mse=False) -> None:
        """
        Args:
            device (str, optional): Device. Defaults to "cpu".
            log_results (bool, optional): Whether to print resulting loss before/after scaling. Defaults to False.
            use_mse (bool, optional): Whether to use Brier score instead of log loss. Defaults to False.
        """
        self.device = device
        self.log_results = log_results  
        self.use_mse = use_mse 
        self.params = None
        self.name = "logit"

    @abstractmethod
    def scale_logits(self, logits: torch.Tensor) -> torch.Tensor:
        """Applies scaling method to logits.

        Args:
            logits (torch.Tensor): Logits tensor.

        Returns:
            torch.Tensor: Rescaled/transformed logits.
        """
        pass

    def fit(self, logits: torch.Tensor, labels: torch.Tensor) -> None:
        """Fits scaling method to provided logits and labels.

        Args:
            logits (torch.Tensor): Logits computed over dataset.
            labels (torch.Tensor): Labels for dataset.
        """
        logits, labels = logits.to(self.device), labels.to(self.device)
        if self.use_mse:
            criterion = BrierLoss()
        else:
            criterion = torch.nn.CrossEntropyLoss()

        if self.log_results:
            print(f"Loss before {self.name} scaling: {criterion(self.scale_logits(logits), labels).item()}")

        optimizer = torch.optim.LBFGS(self.params, lr=0.01, max_iter=1000)
        def eval():
            optimizer.zero_grad()
            loss = criterion(self.scale_logits(logits), labels)
            loss.backward()
            return loss
        optimizer.step(eval)

        if self.log_results:
            print(f"Loss after {self.name} scaling: {criterion(self.scale_logits(logits), labels).item()}")

    def predict_probs(self, logits: torch.Tensor) -> torch.Tensor:
        """Predicts probabilities from logits after transforming.

        Args:
            logits (torch.Tensor): Logits.

        Returns:
            torch.Tensor: Probabilities.
        """
        with torch.no_grad():
            return torch.nn.functional.softmax(self.scale_logits(logits.to(self.device)), dim=1)


class TemperatureScaler(LogitScaler):

    def __init__(self, init_T=1.5, device="cpu", log_results=False, use_mse=False) -> None:
        super().__init__(device, log_results, use_mse)
        self.T = torch.tensor([init_T], device=device, requires_grad=True)
        self.params = [self.T]
        self.name = "temperature"

    def scale_logits(self, logits):
        return logits / self.T
    

class VectorScaler(LogitScaler):

    def __init__(self, n_classes=1000, device="cpu", log_results=False, use_mse=False) -> None:
        super().__init__(device, log_results, use_mse)
        self.vec = torch.ones((1, n_classes), device=device, requires_grad=True)
        self.params = [self.vec]
        self.name = "vector"

    def scale_logits(self, logits):
        return logits * self.vec
    

class MatrixScaler(LogitScaler):

    def __init__(self, n_classes=1000, device="cpu", log_results=False, use_mse=False) -> None:
        super().__init__(device, log_results, use_mse)
        self.linear = torch.nn.Linear(n_classes, n_classes).to(device)
        self.linear.weight.data.copy_(torch.eye(n_classes).to(device))  # Identity initialization.
        self.params = self.linear.parameters()

    def scale_logits(self, logits):
        return self.linear(logits)

    def fit(self, logits, labels):
        logits, labels = logits.to(self.device), labels.to(self.device)
        criterion = torch.nn.CrossEntropyLoss()

        if self.log_results:
            print(f"Loss before matrix scaling: {criterion(self.scale_logits(logits), labels).item()}")

        optimizer = torch.optim.Adam(self.params, lr=1e-3)
        for i in range(1000):
            optimizer.zero_grad()
            loss = criterion(self.scale_logits(logits), labels)
            loss.backward()
            optimizer.step()

        if self.log_results:
            print(f"Loss after matrix scaling: {criterion(self.scale_logits(logits), labels).item()}")