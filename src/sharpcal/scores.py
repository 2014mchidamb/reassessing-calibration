import numpy as np
import torch
from abc import ABC, abstractmethod


class Score(ABC):

    def __init__(self) -> None:
        super().__init__()
        self.supports_1d = False  # Set to true if score can be applied as is in 1-D case.

    @abstractmethod
    def phi(self, x: torch.FloatTensor) -> torch.FloatTensor:
        """The convex function that induces the Bregman divergence.

        Args:
            x (torch.FloatTensor): Input tensor.

        Returns:
            torch.FloatTensor: Resulting tensor.
        """
        pass

    @abstractmethod
    def div(self, x: torch.FloatTensor, y: torch.FloatTensor) -> torch.FloatTensor:
        """The Bregman divergence associated with phi.

        Args:
            x (torch.FloatTensor): First argument of divergence.
            y (torch.FloatTensor): Second argument of divergence.

        Returns:
            torch.FloatTensor: Resulting tensor.
        """
        pass

class BrierScore(Score):

    def __init__(self) -> None:
        super().__init__()
        self.supports_1d = True

    def phi(self, x: torch.FloatTensor) -> torch.FloatTensor:
        return (x**2).sum(dim=1, keepdims=True)

    def div(self, x: torch.FloatTensor, y: torch.Tensor) -> torch.FloatTensor:
        # Just to ensure ordering doesn't matter.
        if x.shape[1] == 1 and y.shape[1] > 1:
            x, y = y, x
        if y.shape[1] == 1 and x.shape[1] > 1:
            one_hot = torch.zeros_like(x)
            one_hot[torch.arange(len(y)), y.squeeze(dim=1)] = 1
            y = one_hot
        return self.phi(x - y)
    
    def __str__(self):
        return "Brier Score"


class KL(Score):

    def __init__(self) -> None:
        super().__init__()

    def phi(self, x: torch.FloatTensor) -> torch.FloatTensor:
        x = (x * torch.log(x)).sum(dim=1, keepdims=True)
        x[torch.isnan(x)] = 0
        return x

    def div(self, x: torch.FloatTensor,
            y: torch.FloatTensor) -> torch.FloatTensor:
        x_entropy = -self.phi(x)
        y_cross = -(x * torch.log(y)).sum(dim=1, keepdims=True)
        y_cross[torch.isnan(y)] = torch.inf
        return y_cross - x_entropy
    
    def __str__(self):
        return "KL Divergence"