import torch

from data.splits import build_few_shot_split


def test_build_few_shot_split_supports_fixed_shots() -> None:
    labels = torch.tensor([0] * 10 + [1] * 10)

    split = build_few_shot_split(labels, shots=3, val_per_class=2, seed=0)

    assert int(split.train_mask[:10].sum()) == 3
    assert int(split.train_mask[10:].sum()) == 3
    assert int(split.val_mask[:10].sum()) == 2
    assert int(split.val_mask[10:].sum()) == 2


def test_build_few_shot_split_supports_class_balanced_ratio() -> None:
    labels = torch.tensor([0] * 20 + [1] * 50)

    split = build_few_shot_split(labels, shots=1, shot_ratio=0.05, val_per_class=2, seed=0)

    assert int(split.train_mask[:20].sum()) == 1
    assert int(split.train_mask[20:].sum()) == 3
    assert int(split.val_mask[:20].sum()) == 2
    assert int(split.val_mask[20:].sum()) == 2
