from __future__ import annotations

from pathlib import Path

import pytest
import torch

from models.backbones import BaseGCN, infer_gcn_dimensions, load_pretrained_gcn


def test_infer_gcn_dimensions_from_state_dict() -> None:
    model = BaseGCN(in_channels=7, hidden_channels=11, num_layers=3)

    in_channels, hidden_channels, num_layers = infer_gcn_dimensions(model.state_dict())

    assert in_channels == 7
    assert hidden_channels == 11
    assert num_layers == 3


def test_load_pretrained_gcn_freezes_parameters(tmp_path: Path) -> None:
    checkpoint_path = tmp_path / "pretrained.pkl"
    source = BaseGCN(in_channels=5, hidden_channels=13, num_layers=2)
    torch.save(source.state_dict(), checkpoint_path)

    loaded = load_pretrained_gcn(checkpoint_path, freeze=True)

    assert isinstance(loaded, BaseGCN)
    assert len(loaded.convs) == 2
    assert all(not parameter.requires_grad for parameter in loaded.parameters())
    assert not loaded.training


def test_load_pretrained_gcn_missing_file_fails_fast(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="Pretrained checkpoint not found"):
        load_pretrained_gcn(tmp_path / "missing.pkl")

