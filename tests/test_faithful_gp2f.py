from __future__ import annotations

import torch
import torch.nn.functional as F

from losses.gp2f_losses import GP2FLossConfig, compute_gp2f_loss
from models.backbones import BaseGCN
from models.faithful_gp2f import FaithfulGP2F


def _toy_graph() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    x = torch.randn(8, 5)
    edge_index = torch.tensor(
        [
            [0, 1, 1, 2, 2, 3, 4, 5, 5, 6, 6, 7],
            [1, 0, 2, 1, 3, 2, 5, 4, 6, 5, 7, 6],
        ],
        dtype=torch.long,
    )
    y = torch.tensor([0, 0, 1, 1, 0, 0, 1, 1], dtype=torch.long)
    train_mask = torch.tensor([True, True, True, True, False, False, False, False])
    return x, edge_index, y, train_mask


def _model() -> FaithfulGP2F:
    backbone = BaseGCN(in_channels=5, hidden_channels=7, num_layers=2)
    return FaithfulGP2F(
        backbone,
        hidden_dim=7,
        num_classes=2,
        adapter_bottleneck_dim=3,
        adapter_beta_init=0.01,
    )


def test_backbone_is_frozen_and_adapters_are_trainable() -> None:
    model = _model()

    assert all(not p.requires_grad for p in model.backbone.parameters())
    assert all(p.requires_grad for p in model.adapters.parameters())
    assert len(model.adapters) == len(model.backbone.convs)


def test_forward_outputs_and_alpha_range() -> None:
    model = _model()
    x, edge_index, _, _ = _toy_graph()

    logits, h_pre, h_adp, h_mix, alpha = model(x, edge_index)

    assert logits.shape == (8, 2)
    assert h_pre.shape == h_adp.shape == h_mix.shape == (8, 7)
    assert 0.0 <= float(alpha.item()) <= 1.0
    assert torch.isfinite(logits).all()


def test_forward_backward_has_no_nan() -> None:
    model = _model()
    x, edge_index, y, train_mask = _toy_graph()

    logits, h_pre, h_adp, h_mix, _ = model(x, edge_index)
    loss_out = compute_gp2f_loss(
        logits=logits,
        labels=y,
        train_mask=train_mask,
        h_pre=h_pre,
        h_adp=h_adp,
        h_mix=h_mix,
        edge_index=edge_index,
        cfg=GP2FLossConfig(),
    )
    loss_out.total.backward()

    assert torch.isfinite(loss_out.total)
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)


def test_disabled_structure_losses_do_not_call_structure_paths(monkeypatch) -> None:
    import losses.gp2f_losses as loss_module

    def fail(*args, **kwargs):
        raise AssertionError("structure path should not be called")

    monkeypatch.setattr(loss_module, "original_contrastive_loss", fail)
    monkeypatch.setattr(loss_module, "original_topology_fusion_loss", fail)

    model = _model()
    x, edge_index, y, train_mask = _toy_graph()
    logits, h_pre, h_adp, h_mix, _ = model(x, edge_index)

    loss_out = loss_module.compute_gp2f_loss(
        logits=logits,
        labels=y,
        train_mask=train_mask,
        h_pre=h_pre,
        h_adp=h_adp,
        h_mix=h_mix,
        edge_index=edge_index,
        cfg=GP2FLossConfig(
            use_original_contrastive=False,
            use_original_topology_fusion=False,
            lambda_ctr=0.0,
            lambda_fus=0.0,
        ),
    )

    assert torch.isfinite(loss_out.total)


def test_toy_graph_smoke_training_three_epochs() -> None:
    torch.manual_seed(7)
    model = _model()
    x, edge_index, y, train_mask = _toy_graph()
    optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=0.01)

    for _ in range(3):
        optimizer.zero_grad()
        logits, _, _, _, _ = model(x, edge_index)
        loss = F.cross_entropy(logits[train_mask], y[train_mask])
        loss.backward()
        optimizer.step()
        assert torch.isfinite(loss)


def test_official_adapter_and_raw_alpha_forward_backward() -> None:
    backbone = BaseGCN(in_channels=5, hidden_channels=7, num_layers=2)
    model = FaithfulGP2F(
        backbone,
        hidden_dim=7,
        num_classes=2,
        adapter_bottleneck_dim=3,
        adapter_style="official_gp2f",
        adapter_alpha_init=0.1,
        fusion_alpha_style="raw",
    )
    x, edge_index, y, train_mask = _toy_graph()

    logits, _, _, _, alpha = model(x, edge_index)
    loss = F.cross_entropy(logits[train_mask], y[train_mask])
    loss.backward()

    assert torch.isfinite(alpha)
    assert any("adapter" in name and param.grad is not None for name, param in model.named_parameters())
