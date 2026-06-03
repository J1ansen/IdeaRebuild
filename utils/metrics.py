"""Metrics for node classification experiments."""

from __future__ import annotations

import torch


def accuracy(pred: torch.Tensor, labels: torch.Tensor, mask: torch.Tensor) -> float:
    if not bool(mask.any()):
        return 0.0
    return float((pred[mask] == labels[mask]).float().mean().item())


def macro_f1(pred: torch.Tensor, labels: torch.Tensor, mask: torch.Tensor, num_classes: int) -> float:
    if not bool(mask.any()):
        return 0.0
    pred_m = pred[mask]
    label_m = labels[mask]
    scores: list[float] = []
    for class_id in range(int(num_classes)):
        true_c = label_m == class_id
        pred_c = pred_m == class_id
        support = true_c.sum()
        if int(support.item()) == 0:
            continue
        tp = (true_c & pred_c).sum().float()
        precision = tp / pred_c.sum().clamp_min(1).float()
        recall = tp / support.clamp_min(1).float()
        denom = precision + recall
        f1 = torch.where(denom > 0, 2.0 * precision * recall / denom, denom)
        scores.append(float(f1.item()))
    return float(sum(scores) / len(scores)) if scores else 0.0


def split_metrics(
    logits: torch.Tensor,
    labels: torch.Tensor,
    mask: torch.Tensor,
    *,
    num_classes: int,
) -> dict[str, float]:
    pred = logits.argmax(dim=-1)
    return {
        "acc": accuracy(pred, labels, mask),
        "macro_f1": macro_f1(pred, labels, mask, num_classes),
    }

