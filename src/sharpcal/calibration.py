import matplotlib.pyplot as plt
import numpy as np
import torch

from typing import Tuple
from prediction import kernel_regression, get_binarized_preds_and_labels
from scores import Score


class SharpCal:
    """Class for generating calibration-sharpness diagrams."""
    def __init__(
        self,
        kernel: torch.nn.Module,
        score: Score,
        n_points: int = 1000,
        device: str = "cpu",
    ) -> None:
        """
        Args:
            kernel (torch.nn.Module): Kernel from kernels.py.
            score (Score): Score (Brier is the only fully supported one right now).
            n_points (int, optional): Subsample data for faster/in-memory computations. Defaults to 1000.
            device (str, optional): Device. Defaults to "cpu".
        """
        self.kernel = kernel
        self.score = score
        self.n_points = n_points
        self.device = device
        self.n_subsamples = 10  # Number of subsamples to average over if necessary.

    def check_sizes(self, preds: torch.FloatTensor, labels: torch.LongTensor):
        """Checks that sizes of provided tensors are appropriate.

        Args:
            preds (torch.FloatTensor): Predictions tensor.
            labels (torch.LongTensor): Labels tensor.

        Raises:
            ValueError: Preds and labels don't have the right shape.
            ValueError: Score does not support confidence calibration.
        """
        if len(preds.shape) != 2 or len(labels.shape) != 2 or labels.shape[1] > 1:
            raise ValueError("Preds, labels should have shapes (n, m), (n, 1).")
        if not self.score.supports_1d:
            raise ValueError(
                f"{str(self.score)} does not support 1-D; cannot construct confidence calibration curve."
            )

    def get_full_loss_and_cal(
        self, preds: torch.FloatTensor, labels: torch.LongTensor
    ) -> Tuple[float, float, float]:
        """Computes loss and calibration error, with optional subsampling.
        If self.n_points < len(preds), then data is subsampled to size self.n_points
        before computing calibration error. This is repeated self.n_subsample times, and the
        mean and standard deviation over these subsamples are returned alongside the full loss.

        Args:
            preds (torch.FloatTensor): Predictions tensor.
            labels (torch.LongTensor): Labels tensor.

        Returns:
            Tuple[float, float, float]: Full loss, calibration error averaged over subsamples, 
            and 1 standard deviation for subsampling.
        """
        self.check_sizes(preds, labels)

        preds, labels = preds.to(self.device), labels.to(self.device)
        if preds.shape[1] > 1:
            bin_preds, bin_labels = get_binarized_preds_and_labels(preds, labels)
        else:
            bin_preds, bin_labels = preds, labels

        if len(preds) > self.n_points:
            print(
                f"Subsampling for estimating calibration error because preds length of {len(preds)} exceeds {self.n_points}."
            )
            cal_errors = []
            for _ in range(self.n_subsamples):
                x = bin_preds[np.random.choice(len(bin_preds), self.n_points)]
                cond_est, _ = kernel_regression(x, bin_preds, bin_labels, self.kernel)
                cal_errors.append(self.score.div(cond_est, x).mean())
            cal_errors = torch.FloatTensor(cal_errors)
            cal_mean, cal_std = cal_errors.mean(), cal_errors.std()
        else:
            x = bin_preds
            cond_est, _ = kernel_regression(x, bin_preds, bin_labels, self.kernel)
            cal_mean = self.score.div(cond_est, x).mean()
            cal_std = 0

        total_loss = self.score.div(preds, labels).mean()
        return total_loss, cal_mean, cal_std

    def plot_cal_curve(
        self, preds: torch.FloatTensor, labels: torch.LongTensor, fname: str = None
    ) -> None:
        """Generates calibration-sharpness plot.

        Args:
            preds (torch.FloatTensor): Predictions tensor.
            labels (torch.LongTensor): Labels tensor.
            fname (str, optional): Filename; if None, simply shows plot. Defaults to None.
        """
        self.check_sizes(preds, labels)

        preds, labels = preds.to(self.device), labels.to(self.device)
        if preds.shape[1] > 1:
            bin_preds, bin_labels = self.get_binarized_preds_and_labels(preds, labels)
        else:
            bin_preds, bin_labels = preds, labels
        print(f"Model accuracy: {(bin_labels.sum() / len(bin_labels) * 100):.2f}")
        x = torch.linspace(0, 1, self.n_points, device=self.device).unsqueeze(dim=1)

        # The conditional expectation here is E[Y = c(X) | h(X)].
        cond_est, kde = kernel_regression(x, bin_preds, bin_labels, self.kernel)
        if bin_preds.std() < 1e-6:
            # Dealing with a near-constant predictor, suffices to let the cond_est be mean of valid values.
            print(f"Binarized confidences have near-zero variance for {fname}.")
            cond_est[:, :] = cond_est[~torch.isnan(cond_est)].mean()
        pw_cal_error = self.score.div(cond_est, x)

        # The difference between the pointwise loss and the calibration loss is the sharpness.
        pw_loss_gap = self.score.div(preds, labels)
        pw_loss_gap, _ = kernel_regression(x, bin_preds, pw_loss_gap, self.kernel)
        pw_loss_gap = ((pw_loss_gap - pw_cal_error) * kde).squeeze(dim=1).cpu().numpy()

        total_loss, actual_cal, cal_std = self.get_full_loss_and_cal(preds, labels)

        normalized_kde = (kde / kde.max()).squeeze(dim=1).cpu().numpy()
        x = x.squeeze(dim=1).cpu().numpy()
        cond_est = cond_est.squeeze(dim=1).cpu().numpy()

        plt.rc("axes", labelsize=16)
        fig, ax = plt.subplots()
        ax.margins(x=0)
        ax.margins(y=0)
        ax.set_xlabel(r"Confidence $(h(x))$")
        ax.set_ylabel(r"Conditional Expectation of $Y = c(X)$")
        ax.plot(x, x, linestyle="dashed", color="black")
        ax.plot(x, cond_est, color="C3")
        ax.fill_between(
            x,
            np.maximum(cond_est - pw_loss_gap / 2, 0),
            cond_est + pw_loss_gap / 2,
            color="C3",
            alpha=0.5,
        )
        ax.fill_between(x, normalized_kde, color="C0", hatch="/", alpha=0.3)
        ax.grid()

        cal_str = r"$d_{{\varphi, \mathrm{{CAL}}}} = {0:.2f} \pm {1:.2f}$".format(
            actual_cal * 100, cal_std * 100
        )
        sharp_str = r"$d_{{\phi, \mathrm{{TOT}}}} = {0:.2f}$".format(total_loss * 100)
        metrics_str = cal_str + "\n" + sharp_str
        props = dict(boxstyle="round", facecolor="gray", alpha=0.2)
        ax.text(
            0.05,
            0.95,
            metrics_str,
            transform=ax.transAxes,
            fontsize=16,
            verticalalignment="top",
            bbox=props,
        )
        ax.text(
            0.8,
            0.02,
            r"$\hat{p}(h(x))$",
            transform=ax.transAxes,
            fontsize=18,
            verticalalignment="bottom",
        )
        plt.tight_layout()

        if fname is not None:
            fig.savefig(fname)
        else:
            plt.show()
