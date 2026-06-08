"""Few-shot split helpers."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch


@dataclass(frozen=True)
class FewShotSplit:
    train_mask: torch.Tensor
    val_mask: torch.Tensor
    test_mask: torch.Tensor
    seed: int


def build_few_shot_split(
    labels: torch.Tensor,
    *,
    shots: int,
    val_per_class: int,
    seed: int,
    shot_ratio: float | None = None,
) -> FewShotSplit:
    """Build class-balanced train/validation masks and use remaining nodes for test."""

    y = labels.view(-1)
    device = y.device
    num_nodes = int(y.numel())
    train_mask = torch.zeros(num_nodes, dtype=torch.bool, device=device)
    val_mask = torch.zeros(num_nodes, dtype=torch.bool, device=device)
    test_mask = torch.zeros(num_nodes, dtype=torch.bool, device=device)

    rng = np.random.default_rng(int(seed))
    num_classes = int(y.max().item()) + 1
    for class_id in range(num_classes):
        idx = torch.where(y == class_id)[0].detach().cpu().numpy()
        rng.shuffle(idx)
        if shot_ratio is None:
            class_shots = int(shots)
        else:
            class_shots = int(np.ceil(float(shot_ratio) * int(idx.size)))
            class_shots = max(1, class_shots)
        class_shots = min(class_shots, int(idx.size))
        train_idx = idx[:class_shots]
        val_start = class_shots
        val_end = val_start + int(val_per_class)
        val_idx = idx[val_start:val_end]
        test_idx = idx[val_end:]

        if train_idx.size:
            train_mask[torch.as_tensor(train_idx, dtype=torch.long, device=device)] = True
        if val_idx.size:
            val_mask[torch.as_tensor(val_idx, dtype=torch.long, device=device)] = True
        if test_idx.size:
            test_mask[torch.as_tensor(test_idx, dtype=torch.long, device=device)] = True

    return FewShotSplit(train_mask=train_mask, val_mask=val_mask, test_mask=test_mask, seed=int(seed))
