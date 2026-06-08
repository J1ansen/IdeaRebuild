from __future__ import annotations

import torch
import torch.nn.functional as F

from models.backbones import BaseGCN
from models.faithful_gp2f import FaithfulGP2F
from models.prompt_module import (
    ParameterMatchedResidualControl,
    UnifiedMultiViewResidualPrompt,
    count_trainable_parameters,
    degree_summary,
    mean_neighbor_summary,
    mean_neighbor_variance,
    prompt_budget_loss,
    prompt_message_norm_loss,
)


def _config() -> dict:
    return {
        "gradient": {"h_pre_for_prompt": "detach", "z_for_prompt": "detach"},
        "semantic": {
            "enabled": True,
            "use_feature_proto": True,
            "use_hidden_proto": False,
            "beta_z": 1.0,
            "beta_h": 0.0,
            "tau_sem": 0.5,
            "use_leave_one_out_train_proto": True,
            "one_shot_train_semantic": "mask",
        },
        "structural": {
            "enabled": True,
            "base": "h_pre_detached",
            "hidden_dim": 5,
            "dropout": 0.0,
        },
        "gate": {"hidden_dim": 5, "dropout": 0.0, "null_bias_init": 1.0, "use_null": True},
        "gamma_init": 0.01,
        "gamma_max": 0.1,
        "route_budget": 0.1,
        "lambda_budget": 0.5,
        "message_norm_target": 0.05,
        "lambda_message_norm": 1.0,
    }


def _toy_inputs() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    z = torch.tensor(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
            [1.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 1.0],
        ]
    )
    h_pre = torch.tensor(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 1.0, 0.0],
            [1.0, 0.5, 0.0],
            [0.0, 0.5, 1.0],
        ]
    )
    edge_index = torch.tensor([[0, 2, 4, 5], [1, 1, 3, 3]], dtype=torch.long)
    y = torch.tensor([0, 0, 1, 1, 0, 1], dtype=torch.long)
    train_mask = torch.tensor([True, True, True, True, False, False])
    return z, h_pre, edge_index, y, train_mask


def test_zero_init_outputs_equal_input_and_gate_is_valid() -> None:
    z, h_pre, edge_index, y, train_mask = _toy_inputs()
    prompt = UnifiedMultiViewResidualPrompt(4, 3, 2, _config())

    out = prompt(z=z, h_pre=h_pre, edge_index=edge_index, train_mask=train_mask, y=y, split="train")

    assert torch.allclose(out["adapted_x"], z)
    assert out["gate"].shape == (z.size(0), 3)
    assert torch.allclose(out["gate"].sum(dim=1), torch.ones(z.size(0)))
    assert out["aux"]["connected_edge_count"] == 0


def test_zero_init_prompt_logits_match_noprompt_eval() -> None:
    z, _, edge_index, y, train_mask = _toy_inputs()
    model = FaithfulGP2F(BaseGCN(in_channels=4, hidden_channels=3, num_layers=2), hidden_dim=3, num_classes=2)
    prompt = UnifiedMultiViewResidualPrompt(4, 3, 2, _config())
    model.eval()
    prompt.eval()

    with torch.no_grad():
        base = model(z, edge_index, return_aux=True)
        h_pre = model.encode_frozen(z, edge_index)
        out = prompt(z=z, h_pre=h_pre, edge_index=edge_index, train_mask=train_mask, y=y, split="eval")
        prompted = model(z, edge_index, adapted_x=out["adapted_x"], adapted_edge_index=edge_index, return_aux=True)

    assert torch.allclose(out["adapted_x"], z)
    assert torch.allclose(prompted["logits"], base["logits"], atol=1e-6)


def test_zero_init_can_wake_up_after_optimizer_step() -> None:
    z, h_pre, edge_index, y, train_mask = _toy_inputs()
    prompt = UnifiedMultiViewResidualPrompt(4, 3, 2, _config())
    optimizer = torch.optim.Adam(prompt.parameters(), lr=0.1)

    optimizer.zero_grad()
    out = prompt(z=z, h_pre=h_pre, edge_index=edge_index, train_mask=train_mask, y=y, split="train")
    loss = out["adapted_x"].sum()
    loss.backward()
    optimizer.step()

    out_after = prompt(z=z, h_pre=h_pre, edge_index=edge_index, train_mask=train_mask, y=y, split="train")
    assert float((out_after["gamma"] * out_after["u_prompt"]).norm().item()) > 0.0


def test_prompt_evidence_is_detached_by_default() -> None:
    z, h_pre, edge_index, y, train_mask = _toy_inputs()
    z = z.clone().requires_grad_(True)
    h_pre = h_pre.clone().requires_grad_(True)
    prompt = UnifiedMultiViewResidualPrompt(4, 3, 2, _config())
    out = prompt(z=z, h_pre=h_pre, edge_index=edge_index, train_mask=train_mask, y=y, split="train")

    grad_z, grad_h = torch.autograd.grad(out["u_prompt"].sum(), (z, h_pre), allow_unused=True)

    assert grad_z is None or torch.allclose(grad_z, torch.zeros_like(z))
    assert grad_h is None or torch.allclose(grad_h, torch.zeros_like(h_pre))


def test_leave_one_out_score_excludes_self_for_train_nodes() -> None:
    z, h_pre, edge_index, y, train_mask = _toy_inputs()
    cfg = _config()
    cfg["structural"]["enabled"] = False
    prompt = UnifiedMultiViewResidualPrompt(4, 3, 2, cfg)
    out = prompt(z=z, h_pre=h_pre, edge_index=edge_index, train_mask=train_mask, y=y, split="train")
    scores = out["aux"]["semantic_scores"]

    expected = F.cosine_similarity(z[0], z[1], dim=0)
    assert torch.allclose(scores[0, 0], expected)


def test_val_test_labels_do_not_affect_prompt_outputs() -> None:
    z, h_pre, edge_index, y, train_mask = _toy_inputs()
    prompt = UnifiedMultiViewResidualPrompt(4, 3, 2, _config())
    prompt.eval()

    changed = y.clone()
    changed[~train_mask] = 1 - changed[~train_mask]
    with torch.no_grad():
        original = prompt(z=z, h_pre=h_pre, edge_index=edge_index, train_mask=train_mask, y=y, split="train")
        relabeled = prompt(
            z=z,
            h_pre=h_pre,
            edge_index=edge_index,
            train_mask=train_mask,
            y=changed,
            split="train",
        )

    assert torch.allclose(original["adapted_x"], relabeled["adapted_x"])
    assert torch.allclose(original["gate"], relabeled["gate"])
    assert torch.allclose(original["aux"]["semantic_scores"], relabeled["aux"]["semantic_scores"])


def test_one_shot_train_nodes_mask_semantic_route() -> None:
    z, h_pre, edge_index, _, _ = _toy_inputs()
    y = torch.tensor([0, 1, 0, 1, 0, 1])
    train_mask = torch.tensor([True, True, False, False, False, False])
    prompt = UnifiedMultiViewResidualPrompt(4, 3, 2, _config())

    out = prompt(z=z, h_pre=h_pre, edge_index=edge_index, train_mask=train_mask, y=y, split="train")

    assert not bool(out["aux"]["semantic_allowed"][0].item())
    assert not bool(out["aux"]["semantic_allowed"][1].item())
    assert torch.allclose(out["gate"][:2, 0], torch.zeros(2))


def test_two_step_diffusion_summary_matches_manual_values() -> None:
    features = torch.tensor([[1.0], [2.0], [3.0], [4.0]])
    edge_index = torch.tensor([[0, 2, 1, 3], [1, 1, 2, 2]])

    m1 = mean_neighbor_summary(features, edge_index)
    m2 = mean_neighbor_summary(m1, edge_index)

    assert torch.allclose(m1, torch.tensor([[0.0], [2.0], [3.0], [0.0]]))
    assert torch.allclose(m2, torch.tensor([[0.0], [1.5], [1.0], [0.0]]))


def test_enhanced_structural_summary_exposes_label_free_auxiliary_features() -> None:
    z, h_pre, edge_index, y, train_mask = _toy_inputs()
    cfg = _config()
    cfg["structural"]["include_degree_features"] = True
    cfg["structural"]["include_neighbor_variance"] = True
    cfg["structural"]["include_similarity_features"] = True
    prompt = UnifiedMultiViewResidualPrompt(4, 3, 2, cfg)

    out = prompt(z=z, h_pre=h_pre, edge_index=edge_index, train_mask=train_mask, y=y, split="train")

    assert out["aux"]["degree"].shape == (z.size(0), 2)
    assert out["aux"]["var1"].shape == h_pre.shape
    assert out["aux"]["similarity"].shape == (z.size(0), 2)
    assert out["aux"]["structural_context"].size(0) == z.size(0)
    assert torch.allclose(out["adapted_x"], z)


def test_structural_cache_matches_uncached_prompt_outputs() -> None:
    z, h_pre, edge_index, y, train_mask = _toy_inputs()
    cfg = _config()
    cfg["structural"]["include_degree_features"] = True
    cfg["structural"]["include_neighbor_variance"] = True
    cfg["structural"]["include_similarity_features"] = True
    prompt = UnifiedMultiViewResidualPrompt(4, 3, 2, cfg)
    prompt.eval()

    cache = prompt.build_structural_cache(z=z, h_pre=h_pre, edge_index=edge_index)
    with torch.no_grad():
        uncached = prompt(z=z, h_pre=h_pre, edge_index=edge_index, train_mask=train_mask, y=y, split="eval")
        cached = prompt(
            z=z,
            h_pre=h_pre,
            edge_index=edge_index,
            train_mask=train_mask,
            y=y,
            split="eval",
            structural_cache=cache,
        )

    assert torch.allclose(cached["adapted_x"], uncached["adapted_x"])
    assert torch.allclose(cached["gate"], uncached["gate"])
    assert torch.allclose(cached["aux"]["structural_context"], uncached["aux"]["structural_context"])
    assert float(cached["aux"]["structural_cache_hit"].item()) == 1.0
    assert float(uncached["aux"]["structural_cache_hit"].item()) == 0.0


def test_degree_and_neighbor_variance_are_label_free_structural_statistics() -> None:
    features = torch.tensor([[1.0], [2.0], [4.0]])
    edge_index = torch.tensor([[0, 2, 2], [1, 1, 0]])

    degree = degree_summary(edge_index, num_nodes=3, dtype=features.dtype, device=features.device)
    variance = mean_neighbor_variance(features, edge_index, num_nodes=3)

    assert degree.shape == (3, 2)
    assert torch.isfinite(degree).all()
    assert torch.allclose(variance, torch.tensor([[0.0], [2.25], [0.0]]))


def test_parameter_matched_control_is_label_and_graph_free() -> None:
    z, h_pre, _, _, _ = _toy_inputs()
    cfg = _config()
    prompt = UnifiedMultiViewResidualPrompt(4, 3, 2, cfg)
    control = ParameterMatchedResidualControl(4, 3, cfg)

    out = control(z=z, h_pre=h_pre, edge_index=None, train_mask=None, y=None)

    assert out["gate"].shape == (z.size(0), 3)
    assert count_trainable_parameters(control) > 0
    ratio = count_trainable_parameters(control) / max(1, count_trainable_parameters(prompt))
    assert 0.25 <= ratio <= 4.0


def test_budget_loss_matches_definition() -> None:
    gate = torch.tensor([[0.2, 0.3, 0.5], [0.4, 0.1, 0.5]])
    loss = prompt_budget_loss(gate, 0.2)

    expected = torch.tensor(((0.5 - 0.2) ** 2))
    assert torch.allclose(loss, expected)


def test_message_norm_loss_penalizes_only_excess_ratio() -> None:
    z = torch.tensor([[2.0, 0.0], [0.0, 4.0]])
    prompt_out = {
        "u_prompt": torch.tensor([[2.0, 0.0], [0.0, 8.0]]),
        "gamma": torch.tensor(0.1),
    }

    loss = prompt_message_norm_loss(prompt_out, z, target_ratio=0.05)

    expected = torch.tensor((((0.1 - 0.05) ** 2) + ((0.2 - 0.05) ** 2)) / 2.0)
    assert torch.allclose(loss, expected)


def test_prompt_forward_backward_has_no_nan() -> None:
    z, h_pre, edge_index, y, train_mask = _toy_inputs()
    prompt = UnifiedMultiViewResidualPrompt(4, 3, 2, _config())

    out = prompt(z=z, h_pre=h_pre, edge_index=edge_index, train_mask=train_mask, y=y, split="train")
    loss = (
        out["adapted_x"].pow(2).mean()
        + prompt_budget_loss(out["gate"], 0.1)
        + prompt_message_norm_loss(out, z, target_ratio=0.05)
    )
    loss.backward()

    assert torch.isfinite(loss)
    assert all(torch.isfinite(param.grad).all() for param in prompt.parameters() if param.grad is not None)
