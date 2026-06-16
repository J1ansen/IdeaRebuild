from __future__ import annotations

import torch

from experiments.run_gp2f_prompt_graph import _config_for_variant
from models.hetero_prompt_adapter import (
    HeterophilyAwarePromptAdapter,
    prompt_adapter_gate_budget_loss,
    prompt_adapter_update_norm_loss,
)
from models.prompt_module import mean_neighbor_summary


def _toy_graph() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    z = torch.tensor(
        [
            [1.0, 0.0, 0.0],
            [0.0, 2.0, 0.0],
            [0.0, 0.0, 3.0],
            [1.0, 1.0, 1.0],
        ]
    )
    edge_index = torch.tensor(
        [
            [0, 1, 2, 0, 3],
            [1, 2, 0, 3, 2],
        ],
        dtype=torch.long,
    )
    h_adp = torch.randn(4, 5)
    return z, edge_index, h_adp


def test_high_frequency_context_matches_manual_neighbor_residual() -> None:
    z, edge_index, h_adp = _toy_graph()
    adapter = HeterophilyAwarePromptAdapter(3, 5, {"dropout": 0.0})

    out = adapter(z=z, edge_index=edge_index, h_adp=h_adp)
    low = mean_neighbor_summary(z, edge_index, num_nodes=z.size(0))

    assert torch.allclose(out["low"], low, atol=1e-6)
    assert torch.allclose(out["high"], z - low, atol=1e-6)


def test_zero_init_starts_as_no_prompt_update() -> None:
    z, edge_index, h_adp = _toy_graph()
    adapter = HeterophilyAwarePromptAdapter(
        3,
        5,
        {"dropout": 0.0, "zero_init_delta": True, "gate_init": 0.05},
    )

    out = adapter(z=z, edge_index=edge_index, h_adp=h_adp)

    assert torch.allclose(out["h_adp"], h_adp, atol=1e-7)
    assert out["prompt_update_norm"].item() == 0.0
    assert 0.04 <= out["prompt_gate_mean"].item() <= 0.06


def test_message_scale_zero_degenerates_to_no_prompt() -> None:
    z, edge_index, h_adp = _toy_graph()
    adapter = HeterophilyAwarePromptAdapter(
        3,
        5,
        {"dropout": 0.0, "zero_init_delta": False, "gate_init": 0.8},
    )

    out = adapter(z=z, edge_index=edge_index, h_adp=h_adp, message_scale=0.0)

    assert torch.allclose(out["h_adp"], h_adp, atol=1e-7)
    assert out["prompt_update_norm"].item() == 0.0


def test_bounded_update_respects_max_norm_and_update_mask() -> None:
    z, edge_index, h_adp = _toy_graph()
    update_mask = torch.tensor([True, False, True, False])
    adapter = HeterophilyAwarePromptAdapter(
        3,
        5,
        {"dropout": 0.0, "zero_init_delta": False, "gate_init": 0.95, "max_update_norm": 0.01},
    )

    out = adapter(z=z, edge_index=edge_index, h_adp=h_adp, update_mask=update_mask)
    update_norm = out["update"].norm(dim=-1)

    assert float(update_norm.max().item()) <= 0.010001
    assert torch.allclose(out["update"][~update_mask], torch.zeros_like(out["update"][~update_mask]))
    assert out["prompt_update_mask_ratio"].item() == 0.5


def test_prompt_adapter_losses_are_masked_and_finite() -> None:
    z, edge_index, h_adp = _toy_graph()
    adapter = HeterophilyAwarePromptAdapter(
        3,
        5,
        {"dropout": 0.0, "zero_init_delta": False, "gate_init": 0.8},
    )
    mask = torch.tensor([True, False, True, False])
    out = adapter(z=z, edge_index=edge_index, h_adp=h_adp, update_mask=mask)

    update_loss = prompt_adapter_update_norm_loss(out, mask)
    gate_loss = prompt_adapter_gate_budget_loss(out, max_gate=0.1, mask=mask)

    assert torch.isfinite(update_loss)
    assert torch.isfinite(gate_loss)
    assert gate_loss.item() > 0.0


def test_prompt_adapter_backward_has_gradients() -> None:
    z, edge_index, h_adp = _toy_graph()
    adapter = HeterophilyAwarePromptAdapter(
        3,
        5,
        {"dropout": 0.0, "zero_init_delta": False, "gate_init": 0.5},
    )
    out = adapter(z=z, edge_index=edge_index, h_adp=h_adp)
    loss = out["h_adp"].pow(2).mean()
    loss.backward()

    grads = [parameter.grad for parameter in adapter.parameters() if parameter.requires_grad]
    assert any(grad is not None and torch.isfinite(grad).all() and grad.abs().sum().item() > 0 for grad in grads)


def test_support_context_uses_only_support_labels() -> None:
    z, edge_index, h_adp = _toy_graph()
    labels = torch.tensor([0, 1, 2, 1])
    support_mask = torch.tensor([True, True, False, False])
    adapter = HeterophilyAwarePromptAdapter(
        3,
        5,
        {
            "dropout": 0.0,
            "num_classes": 3,
            "use_support_context": True,
            "zero_init_delta": False,
        },
    )

    out_a = adapter(z=z, edge_index=edge_index, h_adp=h_adp, support_mask=support_mask, labels=labels)
    labels_changed = labels.clone()
    labels_changed[~support_mask] = torch.tensor([0, 0])
    out_b = adapter(z=z, edge_index=edge_index, h_adp=h_adp, support_mask=support_mask, labels=labels_changed)

    assert out_a["support_context_available"].item() == 1.0
    assert torch.isclose(out_a["support_context_coverage"], torch.tensor(2 / 3), atol=1e-6)
    assert out_a["support_context_count"].item() == 2
    assert torch.allclose(out_a["h_adp"], out_b["h_adp"], atol=1e-6)


def test_p14_p15_p16_variants_disable_prompt_graph_and_enable_adapter() -> None:
    base = {"experiment": {"prompt_variant": "p14_freeze_prompt_adapter"}, "prompt_adapter": {"enabled": True}}
    p14 = _config_for_variant(base, "p14_freeze_prompt_adapter")
    p15 = _config_for_variant(base, "p15_hetero_prompt_adapter")
    p16 = _config_for_variant(base, "p16_support_prompt_adapter")
    noprompt = _config_for_variant(base, "noprompt")

    assert p14["prompt_graph"]["enabled"] is False
    assert p14["prompt_aware"]["enabled"] is False
    assert p14["prompt_adapter"]["enabled"] is True
    assert p14["training"]["freeze_base_model"] is True
    assert p15["prompt_graph"]["enabled"] is False
    assert p15["prompt_adapter"]["enabled"] is True
    assert p15["training"]["train_prompt_adapter"] is True
    assert p16["prompt_graph"]["enabled"] is False
    assert p16["prompt_adapter"]["enabled"] is True
    assert p16["prompt_adapter"]["use_support_context"] is True
    assert p16["training"]["lambda_prompt_adapter_message_help"] > 0.0
    assert noprompt["prompt_adapter"]["enabled"] is False
