import numpy as np
import torch


# CAUTION: These are all 1D kernels. 
# Kernels reference: https://en.wikipedia.org/wiki/Kernel_(statistics).
class Gaussian1D(torch.nn.Module):
    """Gaussian kernel."""
    def __init__(self, bandwidth: float) -> None:
        super().__init__()
        self.bandwidth = bandwidth

    def forward(self, x: torch.FloatTensor) -> torch.FloatTensor:
        return (
            1
            / (self.bandwidth * np.sqrt(2 * np.pi))
            * torch.exp(-torch.square(x) / (2 * (self.bandwidth**2)))
        )

class Epanechnikov1D(torch.nn.Module):
    """Epanechnikov kernel."""
    def __init__(self, bandwidth: float) -> None:
        super().__init__()
        self.bandwidth = bandwidth

    def forward(self, x: torch.FloatTensor) -> torch.FloatTensor:
        return 3/(4 * self.bandwidth) * (1 - (x / self.bandwidth) ** 2) * (torch.abs(x / self.bandwidth) <= 1)
