"""Evaluation metrics for flood segmentation.

Matches the original Sen1Floods11 paper: IoU, accuracy, precision,
recall, and full confusion matrix. All metrics ignore label=255 (nodata).
"""

import torch
import numpy as np


def compute_confusion_matrix(pred, target, ignore_index=255):
    """Compute TP, FP, TN, FN for the water class (class=1).

    Args:
        pred: (N, H, W) or (H, W) predicted class labels (0 or 1).
        target: same shape, ground truth labels.
        ignore_index: label value to ignore.

    Returns:
        dict with tp, fp, tn, fn counts.
    """
    pred = pred.flatten()
    target = target.flatten()

    # Mask out ignored pixels
    valid = target != ignore_index
    pred = pred[valid]
    target = target[valid]

    tp = int(((pred == 1) & (target == 1)).sum())
    fp = int(((pred == 1) & (target == 0)).sum())
    tn = int(((pred == 0) & (target == 0)).sum())
    fn = int(((pred == 0) & (target == 1)).sum())

    return {"tp": tp, "fp": fp, "tn": tn, "fn": fn}


def iou_from_cm(cm):
    """Compute IoU for water class from confusion matrix dict."""
    tp, fp, fn = cm["tp"], cm["fp"], cm["fn"]
    return tp / (tp + fp + fn + 1e-7)


def precision_from_cm(cm):
    tp, fp = cm["tp"], cm["fp"]
    return tp / (tp + fp + 1e-7)


def recall_from_cm(cm):
    tp, fn = cm["tp"], cm["fn"]
    return tp / (tp + fn + 1e-7)


def f1_from_cm(cm):
    p = precision_from_cm(cm)
    r = recall_from_cm(cm)
    return 2 * p * r / (p + r + 1e-7)


def accuracy_from_cm(cm):
    tp, fp, tn, fn = cm["tp"], cm["fp"], cm["tn"], cm["fn"]
    return (tp + tn) / (tp + fp + tn + fn + 1e-7)


def dice_from_cm(cm):
    """Dice score = 2*TP / (2*TP + FP + FN). Same as F1 for binary."""
    tp, fp, fn = cm["tp"], cm["fp"], cm["fn"]
    return 2 * tp / (2 * tp + fp + fn + 1e-7)


def compute_metrics(pred, target, ignore_index=255):
    """Compute all metrics for a batch.

    Args:
        pred: predicted class labels (0 or 1), any shape.
        target: ground truth labels, same shape.
        ignore_index: value to ignore.

    Returns:
        dict with iou, dice, precision, recall, f1, accuracy.
    """
    if isinstance(pred, torch.Tensor):
        pred = pred.cpu().numpy()
    if isinstance(target, torch.Tensor):
        target = target.cpu().numpy()

    cm = compute_confusion_matrix(pred, target, ignore_index)
    return {
        "iou": iou_from_cm(cm),
        "dice": dice_from_cm(cm),
        "precision": precision_from_cm(cm),
        "recall": recall_from_cm(cm),
        "f1": f1_from_cm(cm),
        "accuracy": accuracy_from_cm(cm),
        **cm,
    }


class MetricAccumulator:
    """Accumulates confusion matrix counts across batches for epoch-level metrics."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.tp = 0
        self.fp = 0
        self.tn = 0
        self.fn = 0

    def update(self, pred, target, ignore_index=255):
        if isinstance(pred, torch.Tensor):
            pred = pred.cpu().numpy()
        if isinstance(target, torch.Tensor):
            target = target.cpu().numpy()
        cm = compute_confusion_matrix(pred, target, ignore_index)
        self.tp += cm["tp"]
        self.fp += cm["fp"]
        self.tn += cm["tn"]
        self.fn += cm["fn"]

    def compute(self):
        cm = {"tp": self.tp, "fp": self.fp, "tn": self.tn, "fn": self.fn}
        return {
            "iou": iou_from_cm(cm),
            "dice": dice_from_cm(cm),
            "precision": precision_from_cm(cm),
            "recall": recall_from_cm(cm),
            "f1": f1_from_cm(cm),
            "accuracy": accuracy_from_cm(cm),
            **cm,
        }
