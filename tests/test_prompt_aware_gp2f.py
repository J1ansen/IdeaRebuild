from __future__ import annotations

import torch
import torch.nn.functional as F

from models.backbones import BaseGCN
from models.prompt_aware_gp2f import PromptAwareGP2F
from models.prompt_graph_module import PromptGraphModuleP1


def _config(**overrides: object) -> dict:
    cfg: dict[str, object] = {
        "num_prompt_nodes": 3,
        "rho": 1.0,
        "topk_prompt_per_node": 2,
        "structural_base": "z_detached",
        "pool_strategy": "structural",
        "tau": 0.5,
        "query_dim": 4,
        "query_hidden_dim": 6,
        "query_dropout": 0.0,
        "prompt_init_std": 0.02,
        "edge_scale_init": 0.05,
        "edge_scale_max": 0.20,
    }
    cfg.update(overrides)
    return cfg


def _toy_inputs() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    x = torch.tensor(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
            [1.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 1.0],
        ]
    )
    edge_index = torch.tensor(
        [
            [0, 1, 2, 3, 4, 5, 1, 3],
            [1, 2, 3, 4, 5, 0, 0, 2],
        ],
        dtype=torch.long,
    )
    y = torch.tensor([0, 0, 1, 1, 0, 1], dtype=torch.long)
    train_mask = torch.tensor([True, False, True, False, False, False])
    return x, edge_index, y, train_mask


def _model(*, gate_init: float = 0.01, **prompt_aware_overrides: object) -> PromptAwareGP2F:
    prompt_aware_config: dict[str, object] = {"dropout": 0.0, "gate_init": gate_init}
    prompt_aware_config.update(prompt_aware_overrides)
    return PromptAwareGP2F(
        BaseGCN(in_channels=4, hidden_channels=4, num_layers=2),
        hidden_dim=4,
        num_classes=2,
        prompt_aware_config=prompt_aware_config,
    )


def test_prompt_aware_supports_directional_gate_init_and_message_scale() -> None:
    model = PromptAwareGP2F(
        BaseGCN(in_channels=4, hidden_channels=4, num_layers=2),
        hidden_dim=4,
        num_classes=2,
        prompt_aware_config={
            "dropout": 0.0,
            "gate_init": 0.01,
            "node_to_prompt_gate_init": 0.2,
            "prompt_to_node_gate_init": 0.4,
            "message_scale": 3.0,
        },
    )

    gates = model.prompt_gates
    assert torch.allclose(gates[:, 0], torch.full_like(gates[:, 0], 0.2), atol=1e-6)
    assert torch.allclose(gates[:, 1], torch.full_like(gates[:, 1], 0.4), atol=1e-6)
    assert model.prompt_message_scale == 3.0


def test_zero_edge_type_matches_no_prompt_adapted_path() -> None:
    x, edge_index, _, _ = _toy_inputs()
    model = _model()
    model.eval()
    h_pre = model.encode_frozen(x, edge_index)

    with torch.no_grad():
        no_type = model.forward_with_h_pre(x, edge_index, h_pre=h_pre, return_aux=True)
        zeros = model.forward_with_h_pre(
            x,
            edge_index,
            h_pre=h_pre,
            adapted_edge_type=torch.zeros(edge_index.size(1), dtype=torch.long),
            return_aux=True,
        )

    assert torch.allclose(no_type["logits"], zeros["logits"], atol=1e-6)
    assert zeros["prompt_aware"]["prompt_msg_norm"].item() == 0.0


def test_prompt_aware_edges_change_logits_when_gate_is_open() -> None:
    x, edge_index, _, train_mask = _toy_inputs()
    model = _model(gate_init=0.5)
    module = PromptGraphModuleP1(4, 4, _config())
    model.eval()
    module.eval()
    h_pre = model.encode_frozen(x, edge_index)

    with torch.no_grad():
        baseline = model.forward_with_h_pre(x, edge_index, h_pre=h_pre, return_aux=True)
        prompt_out = module(z=x, h_pre=h_pre.detach(), edge_index=edge_index, train_mask=train_mask)
        prompted = model.forward_with_h_pre(
            x,
            edge_index,
            h_pre=h_pre,
            adapted_x=prompt_out["adapted_x"],
            adapted_edge_index=prompt_out["adapted_edge_index"],
            adapted_edge_weight=prompt_out["adapted_edge_weight"],
            adapted_edge_type=prompt_out["adapted_edge_type"],
            return_aux=True,
        )

    assert (prompted["logits"] - baseline["logits"]).abs().max().item() > 0.0
    assert prompted["prompt_aware"]["prompt_to_original_msg_norm"].item() > 0.0
    assert prompted["prompt_aware"]["node_to_prompt_msg_norm"].item() > 0.0
    assert prompted["prompt_aware"]["prompt_to_original_update_norm"].item() > 0.0
    assert prompted["prompt_aware"]["node_to_prompt_update_norm"].item() > 0.0


def test_prompt_aware_directional_ablation_disables_requested_message() -> None:
    x, edge_index, _, train_mask = _toy_inputs()
    model = PromptAwareGP2F(
        BaseGCN(in_channels=4, hidden_channels=4, num_layers=2),
        hidden_dim=4,
        num_classes=2,
        prompt_aware_config={"dropout": 0.0, "gate_init": 0.5, "use_prompt_to_node": False},
    )
    module = PromptGraphModuleP1(4, 4, _config())
    h_pre = model.encode_frozen(x, edge_index)
    prompt_out = module(z=x, h_pre=h_pre.detach(), edge_index=edge_index, train_mask=train_mask)

    out = model.forward_with_h_pre(
        x,
        edge_index,
        h_pre=h_pre,
        adapted_x=prompt_out["adapted_x"],
        adapted_edge_index=prompt_out["adapted_edge_index"],
        adapted_edge_weight=prompt_out["adapted_edge_weight"],
        adapted_edge_type=prompt_out["adapted_edge_type"],
        return_aux=True,
    )

    assert out["prompt_aware"]["prompt_to_original_msg_norm"].item() == 0.0
    assert out["prompt_aware"]["node_to_prompt_msg_norm"].item() > 0.0


def test_prompt_aware_message_scale_increases_adapted_delta() -> None:
    x, edge_index, _, train_mask = _toy_inputs()
    module = PromptGraphModuleP1(4, 4, _config())
    small = _model(gate_init=0.2)
    large = PromptAwareGP2F(
        BaseGCN(in_channels=4, hidden_channels=4, num_layers=2),
        hidden_dim=4,
        num_classes=2,
        prompt_aware_config={"dropout": 0.0, "gate_init": 0.2, "message_scale": 5.0},
    )
    large.load_state_dict(small.state_dict(), strict=False)
    small.eval()
    large.eval()
    module.eval()

    h_pre_small = small.encode_frozen(x, edge_index)
    h_pre_large = large.encode_frozen(x, edge_index)
    prompt_out = module(z=x, h_pre=h_pre_small.detach(), edge_index=edge_index, train_mask=train_mask)

    with torch.no_grad():
        out_small = small.forward_with_h_pre(
            x,
            edge_index,
            h_pre=h_pre_small,
            adapted_x=prompt_out["adapted_x"],
            adapted_edge_index=prompt_out["adapted_edge_index"],
            adapted_edge_weight=prompt_out["adapted_edge_weight"],
            adapted_edge_type=prompt_out["adapted_edge_type"],
            return_aux=True,
        )
        out_large = large.forward_with_h_pre(
            x,
            edge_index,
            h_pre=h_pre_large,
            adapted_x=prompt_out["adapted_x"],
            adapted_edge_index=prompt_out["adapted_edge_index"],
            adapted_edge_weight=prompt_out["adapted_edge_weight"],
            adapted_edge_type=prompt_out["adapted_edge_type"],
            return_aux=True,
        )

    assert out_large["prompt_aware"]["adapted_branch_delta_norm"].item() > out_small["prompt_aware"][
        "adapted_branch_delta_norm"
    ].item()


def test_zero_message_scale_matches_no_prompt_for_original_nodes() -> None:
    x, edge_index, _, train_mask = _toy_inputs()
    model = _model(gate_init=0.5, message_scale=0.0, pool_only_prompt_update=True)
    module = PromptGraphModuleP1(4, 4, _config())
    model.eval()
    module.eval()
    h_pre = model.encode_frozen(x, edge_index)
    prompt_out = module(z=x, h_pre=h_pre.detach(), edge_index=edge_index, train_mask=train_mask)

    with torch.no_grad():
        baseline = model.forward_with_h_pre(x, edge_index, h_pre=h_pre, return_aux=True)
        prompted = model.forward_with_h_pre(
            x,
            edge_index,
            h_pre=h_pre,
            adapted_x=prompt_out["adapted_x"],
            adapted_edge_index=prompt_out["adapted_edge_index"],
            adapted_edge_weight=prompt_out["adapted_edge_weight"],
            adapted_edge_type=prompt_out["adapted_edge_type"],
            prompt_update_mask=prompt_out["pool_mask"],
            return_aux=True,
        )

    assert torch.allclose(prompted["h_adp"], baseline["h_adp"], atol=1e-6)
    assert torch.allclose(prompted["logits"], baseline["logits"], atol=1e-6)
    assert prompted["prompt_aware"]["adapted_branch_delta_norm"].item() == 0.0
    assert prompted["prompt_aware"]["prompt_message_scale"].item() == 0.0


def test_zero_init_prompt_messages_start_as_no_prompt_but_receive_gradients() -> None:
    x, edge_index, y, train_mask = _toy_inputs()
    model = _model(
        gate_init=0.5,
        message_scale=1.0,
        pool_only_prompt_update=True,
        zero_init_prompt_messages=True,
    )
    module = PromptGraphModuleP1(4, 4, _config())
    model.eval()
    module.eval()
    h_pre = model.encode_frozen(x, edge_index)
    prompt_out = module(z=x, h_pre=h_pre.detach(), edge_index=edge_index, train_mask=train_mask)

    baseline = model.forward_with_h_pre(x, edge_index, h_pre=h_pre, return_aux=True)
    prompted = model.forward_with_h_pre(
        x,
        edge_index,
        h_pre=h_pre,
        adapted_x=prompt_out["adapted_x"],
        adapted_edge_index=prompt_out["adapted_edge_index"],
        adapted_edge_weight=prompt_out["adapted_edge_weight"],
        adapted_edge_type=prompt_out["adapted_edge_type"],
        prompt_update_mask=prompt_out["pool_mask"],
        return_aux=True,
    )

    assert torch.allclose(prompted["h_adp"], baseline["h_adp"], atol=1e-6)
    assert torch.allclose(prompted["logits"], baseline["logits"], atol=1e-6)
    assert prompted["prompt_aware"]["zero_init_prompt_messages"].item() == 1.0

    loss = F.cross_entropy(prompted["logits"][train_mask], y[train_mask])
    loss.backward()

    assert any(
        parameter.grad is not None and torch.isfinite(parameter.grad).all() and parameter.grad.abs().sum() > 0
        for parameter in model.prompt_to_node_msgs.parameters()
    )


def test_v2_conditioned_receiver_zero_init_matches_no_prompt_and_gets_gradients() -> None:
    x, edge_index, y, train_mask = _toy_inputs()
    model = _model(
        gate_init=0.5,
        message_scale=1.0,
        pool_only_prompt_update=True,
        receiver_version="v2_conditioned",
        prompt_message_norm="layernorm",
        zero_init_prompt_messages=True,
    )
    module = PromptGraphModuleP1(4, 4, _config())
    model.eval()
    module.eval()
    h_pre = model.encode_frozen(x, edge_index)
    prompt_out = module(z=x, h_pre=h_pre.detach(), edge_index=edge_index, train_mask=train_mask)

    baseline = model.forward_with_h_pre(x, edge_index, h_pre=h_pre, return_aux=True)
    prompted = model.forward_with_h_pre(
        x,
        edge_index,
        h_pre=h_pre,
        adapted_x=prompt_out["adapted_x"],
        adapted_edge_index=prompt_out["adapted_edge_index"],
        adapted_edge_weight=prompt_out["adapted_edge_weight"],
        adapted_edge_type=prompt_out["adapted_edge_type"],
        prompt_update_mask=prompt_out["pool_mask"],
        return_aux=True,
    )

    assert torch.allclose(prompted["h_adp"], baseline["h_adp"], atol=1e-6)
    assert prompted["prompt_aware"]["receiver_version"] == "v2_conditioned"
    assert prompted["prompt_aware"]["prompt_receiver_gate_mean"].item() > 0.0

    loss = F.cross_entropy(prompted["logits"][train_mask], y[train_mask])
    loss.backward()

    assert any(
        parameter.grad is not None and torch.isfinite(parameter.grad).all() and parameter.grad.abs().sum() > 0
        for parameter in model.prompt_to_node_conditioned_msgs.parameters()
    )


def test_v2_conditioned_receiver_changes_logits_when_not_zero_initialized() -> None:
    x, edge_index, _, train_mask = _toy_inputs()
    model = _model(
        gate_init=0.5,
        message_scale=1.0,
        receiver_version="v2_conditioned",
        prompt_message_norm="layernorm",
        zero_init_prompt_messages=False,
    )
    module = PromptGraphModuleP1(4, 4, _config())
    model.eval()
    module.eval()
    h_pre = model.encode_frozen(x, edge_index)
    prompt_out = module(z=x, h_pre=h_pre.detach(), edge_index=edge_index, train_mask=train_mask)

    with torch.no_grad():
        baseline = model.forward_with_h_pre(x, edge_index, h_pre=h_pre, return_aux=True)
        prompted = model.forward_with_h_pre(
            x,
            edge_index,
            h_pre=h_pre,
            adapted_x=prompt_out["adapted_x"],
            adapted_edge_index=prompt_out["adapted_edge_index"],
            adapted_edge_weight=prompt_out["adapted_edge_weight"],
            adapted_edge_type=prompt_out["adapted_edge_type"],
            prompt_update_mask=prompt_out["pool_mask"],
            return_aux=True,
        )

    assert (prompted["logits"] - baseline["logits"]).abs().max().item() > 0.0
    assert prompted["prompt_aware"]["prompt_to_original_update_norm"].item() > 0.0


def test_prompt_aware_weighted_sum_keeps_edge_scale_effect() -> None:
    x, edge_index, _, train_mask = _toy_inputs()
    model = PromptAwareGP2F(
        BaseGCN(in_channels=4, hidden_channels=4, num_layers=2),
        hidden_dim=4,
        num_classes=2,
        prompt_aware_config={
            "dropout": 0.0,
            "gate_init": 0.5,
            "message_norm": "weighted_sum",
        },
    )
    module = PromptGraphModuleP1(4, 4, _config())
    h_pre = model.encode_frozen(x, edge_index)
    weak = module(z=x, h_pre=h_pre.detach(), edge_index=edge_index, train_mask=train_mask, edge_scale_multiplier=0.1)
    strong = module(z=x, h_pre=h_pre.detach(), edge_index=edge_index, train_mask=train_mask, edge_scale_multiplier=1.0)

    with torch.no_grad():
        weak_out = model.forward_with_h_pre(
            x,
            edge_index,
            h_pre=h_pre,
            adapted_x=weak["adapted_x"],
            adapted_edge_index=weak["adapted_edge_index"],
            adapted_edge_weight=weak["adapted_edge_weight"],
            adapted_edge_type=weak["adapted_edge_type"],
            return_aux=True,
        )
        strong_out = model.forward_with_h_pre(
            x,
            edge_index,
            h_pre=h_pre,
            adapted_x=strong["adapted_x"],
            adapted_edge_index=strong["adapted_edge_index"],
            adapted_edge_weight=strong["adapted_edge_weight"],
            adapted_edge_type=strong["adapted_edge_type"],
            return_aux=True,
        )

    assert strong_out["prompt_aware"]["adapted_branch_delta_norm"].item() > weak_out["prompt_aware"][
        "adapted_branch_delta_norm"
    ].item()


def test_pool_only_prompt_update_preserves_non_pool_original_nodes() -> None:
    x, edge_index, _, _ = _toy_inputs()
    model = PromptAwareGP2F(
        BaseGCN(in_channels=4, hidden_channels=4, num_layers=1),
        hidden_dim=4,
        num_classes=2,
        prompt_aware_config={
            "dropout": 0.0,
            "gate_init": 0.5,
            "message_scale": 5.0,
            "pool_only_prompt_update": True,
        },
    )
    model.eval()
    prompt_node = torch.tensor([[0.5, -0.5, 0.5, -0.5]], dtype=x.dtype)
    adapted_x = torch.cat([x, prompt_node], dim=0)
    prompt_id = x.size(0)
    prompt_edges = torch.tensor(
        [
            [prompt_id, prompt_id],
            [0, 1],
        ],
        dtype=torch.long,
    )
    adapted_edge_index = torch.cat([edge_index, prompt_edges], dim=1)
    adapted_edge_type = torch.cat(
        [
            torch.zeros(edge_index.size(1), dtype=torch.long),
            torch.full((prompt_edges.size(1),), 2, dtype=torch.long),
        ]
    )
    adapted_edge_weight = torch.ones(adapted_edge_index.size(1), dtype=x.dtype)
    prompt_update_mask = torch.tensor([True, False, False, False, False, False])
    h_pre = model.encode_frozen(x, edge_index)

    with torch.no_grad():
        baseline = model.forward_with_h_pre(x, edge_index, h_pre=h_pre, return_aux=True)
        masked = model.forward_with_h_pre(
            x,
            edge_index,
            h_pre=h_pre,
            adapted_x=adapted_x,
            adapted_edge_index=adapted_edge_index,
            adapted_edge_weight=adapted_edge_weight,
            adapted_edge_type=adapted_edge_type,
            prompt_update_mask=prompt_update_mask,
            return_aux=True,
        )
        unmasked = model.forward_with_h_pre(
            x,
            edge_index,
            h_pre=h_pre,
            adapted_x=adapted_x,
            adapted_edge_index=adapted_edge_index,
            adapted_edge_weight=adapted_edge_weight,
            adapted_edge_type=adapted_edge_type,
            return_aux=True,
        )

    assert torch.allclose(masked["h_adp"][1], baseline["h_adp"][1], atol=1e-6)
    assert not torch.allclose(unmasked["h_adp"][1], baseline["h_adp"][1], atol=1e-6)
    assert not torch.allclose(masked["h_adp"][0], baseline["h_adp"][0], atol=1e-6)
    assert masked["prompt_aware"]["pool_only_prompt_update"].item() == 1.0


def test_prompt_aware_modules_receive_gradients_and_frozen_branch_isolated() -> None:
    x, edge_index, y, train_mask = _toy_inputs()
    model = _model(gate_init=0.5)
    module = PromptGraphModuleP1(4, 4, _config())
    h_pre = model.encode_frozen(x, edge_index)
    h_pre_before = h_pre.detach().clone()
    prompt_out = module(z=x, h_pre=h_pre.detach(), edge_index=edge_index, train_mask=train_mask)

    out = model.forward_with_h_pre(
        x,
        edge_index,
        h_pre=h_pre,
        adapted_x=prompt_out["adapted_x"],
        adapted_edge_index=prompt_out["adapted_edge_index"],
        adapted_edge_weight=prompt_out["adapted_edge_weight"],
        adapted_edge_type=prompt_out["adapted_edge_type"],
        return_aux=True,
    )
    loss = F.cross_entropy(out["logits"][train_mask], y[train_mask])
    loss.backward()

    assert torch.allclose(h_pre.detach(), h_pre_before)
    assert model.prompt_gate_logit.grad is not None
    assert any(parameter.grad is not None for parameter in model.prompt_to_node_msgs.parameters())
    assert any(parameter.grad is not None for parameter in model.node_to_prompt_msgs.parameters())
    assert module.prompt_node_x.grad is not None
    assert torch.isfinite(loss)
