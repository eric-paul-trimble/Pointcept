"""Reusable classification and segmentation metrics."""

from __future__ import annotations

from collections.abc import Sequence
import logging

import numpy as np
import torch


def _safe_divide(numerator: torch.Tensor, denominator: torch.Tensor) -> torch.Tensor:
    """Divide tensors while returning zero for empty classes."""
    return torch.where(
        denominator > 0, numerator / denominator, torch.zeros_like(numerator)
    )


def get_mious(
    true_positive: torch.Tensor, union: torch.Tensor, count: torch.Tensor
) -> tuple[float, float, float, np.ndarray, np.ndarray]:
    """Return mean IoU, mean accuracy, overall accuracy, and per-class values."""
    iou_per_class = _safe_divide(true_positive, union) * 100
    accuracy_per_class = _safe_divide(true_positive, count) * 100
    overall_accuracy = _safe_divide(true_positive.sum(), count.sum()) * 100

    return (
        iou_per_class.mean().item(),
        accuracy_per_class.mean().item(),
        overall_accuracy.item(),
        iou_per_class.cpu().numpy(),
        accuracy_per_class.cpu().numpy(),
    )


class ConfusionMatrix:
    """Accumulate a dense confusion matrix for integer class predictions."""

    def __init__(
        self,
        num_classes: int,
        ignore_index: int | None = None,
        device: torch.device | str | None = None,
    ) -> None:
        if num_classes <= 0:
            raise ValueError("num_classes must be positive")

        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.value = torch.zeros(
            (num_classes, num_classes), dtype=torch.int64, device=device
        )

    @torch.no_grad()
    def update(self, pred: torch.Tensor, true: torch.Tensor) -> None:
        """Add a batch of predictions and targets to the matrix."""
        pred = pred.reshape(-1).to(dtype=torch.long)
        true = true.reshape(-1).to(dtype=torch.long, device=pred.device)
        if pred.numel() != true.numel():
            raise ValueError("pred and true must contain the same number of values")
        if self.value.device != pred.device:
            self.value = self.value.to(pred.device)

        valid = (true >= 0) & (true < self.num_classes)
        valid &= (pred >= 0) & (pred < self.num_classes)
        if self.ignore_index is not None:
            valid &= true != self.ignore_index
        if not valid.any():
            return

        mapping = true[valid] * self.num_classes + pred[valid]
        bins = torch.bincount(mapping, minlength=self.num_classes**2)
        self.value += bins.reshape(self.num_classes, self.num_classes)

    def reset(self) -> None:
        """Clear all accumulated counts."""
        self.value.zero_()

    @property
    def tp(self) -> torch.Tensor:
        """Return true positives for each class."""
        return self.value.diag()

    @property
    def actual(self) -> torch.Tensor:
        """Return ground-truth counts for each class."""
        return self.value.sum(dim=1)

    @property
    def predicted(self) -> torch.Tensor:
        """Return predicted counts for each class."""
        return self.value.sum(dim=0)

    @property
    def fn(self) -> torch.Tensor:
        """Return false negatives for each class."""
        return self.actual - self.tp

    @property
    def fp(self) -> torch.Tensor:
        """Return false positives for each class."""
        return self.predicted - self.tp

    @property
    def tn(self) -> torch.Tensor:
        """Return true negatives for each class."""
        return self.total + self.tp - self.actual - self.predicted

    @property
    def count(self) -> torch.Tensor:
        """Return the number of ground-truth values for each class."""
        return self.actual

    @property
    def frequency(self) -> torch.Tensor:
        """Return the ground-truth frequency for each class."""
        return _safe_divide(self.count, self.total)

    @property
    def total(self) -> torch.Tensor:
        """Return the total number of counted values."""
        return self.value.sum()

    @property
    def overall_accuracy(self) -> torch.Tensor:
        """Return the fraction of counted predictions that are correct."""
        return _safe_divide(self.tp.sum(), self.total)

    @property
    def overall_accuray(self) -> torch.Tensor:
        """Backward-compatible alias for :attr:`overall_accuracy`."""
        return self.overall_accuracy

    @property
    def union(self) -> torch.Tensor:
        """Return the IoU union count for each class."""
        return self.actual + self.predicted - self.tp

    @property
    def precision(self) -> torch.Tensor:
        """Return per-class precision as percentages."""
        return _safe_divide(self.tp, self.predicted) * 100

    @property
    def recall(self) -> torch.Tensor:
        """Return per-class recall as percentages."""
        return _safe_divide(self.tp, self.actual) * 100

    def all_acc(self) -> tuple[float, float, np.ndarray]:
        """Return mean class accuracy, overall accuracy, and per-class accuracy."""
        accuracy = _safe_divide(self.tp, self.count) * 100
        return (
            accuracy.mean().item(),
            self.overall_accuracy.item() * 100,
            accuracy.cpu().numpy(),
        )

    def all_metrics(self) -> tuple[float, float, float, np.ndarray, np.ndarray]:
        """Return mean IoU, mean accuracy, overall accuracy, and per-class values."""
        return get_mious(self.tp, self.union, self.count)

    @staticmethod
    def cal_acc(
        true_positive: torch.Tensor, count: torch.Tensor
    ) -> tuple[float, float, np.ndarray]:
        """Calculate mean and overall accuracy from class counts."""
        accuracy = _safe_divide(true_positive, count) * 100
        overall = _safe_divide(true_positive.sum(), count.sum()) * 100
        return accuracy.mean().item(), overall.item(), accuracy.cpu().numpy()

    @staticmethod
    def print_acc(accs: Sequence[torch.Tensor]) -> None:
        """Log per-class accuracy values."""
        logging.info("\n    Class  " + "   Acc  ")
        for index, value in enumerate(accs):
            logging.info("%8d%8.2f", index, value.item())
