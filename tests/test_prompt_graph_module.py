from __future__ import annotations

from pathlib import Path

import torch
import torch.nn.functional as F

from models.backbones import BaseGCN
from models.faithful_gp2f import FaithfulGP2F
from models.prompt_graph_module import (
    PromptGraphModuleP1,
    prompt_acceptance_budget_loss,
    prompt_acceptance_loss,
    prompt_balance_loss,
    prompt_class_route_loss,
    prompt_edge_l1_loss,
    prompt_key_proto_loss,
    prompt_role_diversity_loss,
    prompt_usage_consistency_loss,
    prompt_view_entropy_loss,
    prompt_view_prior_loss,
    utility_receive_gate_budget_loss,
)
from models.discrete_feature_prompt import SelectiveDiscreteFeaturePromptGraph
from experiments.run_gp2f_prompt_graph import (
    _acceptance_supervision_loss,
    _benefit_supervision_loss,
    _class_balanced_support_query_split,
    _correction_alignment_losses,
    _edge_utility_supervision_loss,
    _prompt_message_help_loss,
    _prompt_slot_usage_stats,
    _prompt_correction_losses,
    _query_proto_alignment_loss,
    _utility_receive_gate_loss,
    _config_for_variant,
)
from utils.io import read_yaml


def _config(**overrides: object) -> dict:
    cfg: dict[str, object] = {
        "num_prompt_nodes": 4,
        "rho": 0.4,
        "topk_prompt_per_node": 2,
        "structural_base": "z_detached",
        "pool_strategy": "structural",
        "tau": 0.5,
        "query_dim": 3,
        "query_hidden_dim": 5,
        "query_dropout": 0.0,
        "prompt_init_std": 0.02,
        "edge_scale_init": 0.01,
        "edge_scale_max": 0.20,
    }
    cfg.update(overrides)
    return cfg


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
    edge_index = torch.tensor(
        [
            [0, 1, 2, 3, 4, 5, 1, 3],
            [1, 2, 3, 4, 5, 0, 0, 2],
        ],
        dtype=torch.long,
    )
    y = torch.tensor([0, 0, 1, 1, 0, 1], dtype=torch.long)
    train_mask = torch.tensor([True, False, True, False, False, False])
    return z, h_pre, edge_index, y, train_mask


def test_selective_discrete_feature_prompt_graph_adds_prompt_to_node_edges() -> None:
    z, h_pre, edge_index, _y, train_mask = _toy_inputs()
    no_prompt_logits = torch.tensor(
        [
            [2.0, 0.1],
            [0.3, 0.2],
            [0.1, 2.0],
            [0.4, 0.3],
            [0.2, 0.1],
            [0.1, 0.2],
        ]
    )
    module = SelectiveDiscreteFeaturePromptGraph(
        source_dim=z.size(1),
        hidden_dim=h_pre.size(1),
        config={
            "tokenizer": "binary_nonzero",
            "min_df": 1,
            "max_df_ratio": 1.0,
            "rho": 0.5,
            "topk_feature_prompt_per_node": 2,
            "feature_edge_weight": 0.2,
        },
    )

    out_zero = module(
        z=z,
        h_pre=h_pre,
        edge_index=edge_index,
        train_mask=train_mask,
        edge_scale_multiplier=0.0,
        no_prompt_logits=no_prompt_logits,
        h_adp_no_prompt=h_pre * 0.8,
    )
    out = module(
        z=z,
        h_pre=h_pre,
        edge_index=edge_index,
        train_mask=train_mask,
        edge_scale_multiplier=1.0,
        no_prompt_logits=no_prompt_logits,
        h_adp_no_prompt=h_pre * 0.8,
    )

    assert out["adapted_x"].size(0) > z.size(0)
    assert out["adapted_edge_index"].size(1) > edge_index.size(1)
    prompt_edges = out["adapted_edge_index"][:, edge_index.size(1) :]
    assert bool((prompt_edges[0] >= z.size(0)).all())
    assert bool((prompt_edges[1] < z.size(0)).all())
    assert out["pool_mask"].dtype == torch.bool
    assert out["aux"]["p23_feature_prompt_count"].item() == out["prompt_node_x"].size(0)
    assert out_zero["aux"]["p23_static_cache_hit"].item() == 0.0
    assert out["aux"]["p23_static_cache_hit"].item() == 1.0
    assert out_zero["aux"]["prompt_edge_weight"].sum().item() == 0.0
    assert out["aux"]["prompt_edge_weight"].sum().item() > 0.0


def test_p23_variant_config_uses_selective_discrete_feature_prompt_module() -> None:
    cfg = _config_for_variant(
        {
            "experiment": {"prompt_variant": "p23_selective_discrete_feature_prompting"},
            "prompt_graph": {},
            "prompt_adapter": {"enabled": True},
            "prompt_aware": {"enabled": True},
            "training": {},
        },
        "p23_selective_discrete_feature_prompting",
    )

    assert cfg["prompt_graph"]["enabled"] is True
    assert cfg["prompt_graph"]["module_type"] == "selective_discrete_feature_prompt"
    assert cfg["prompt_graph"]["static_graph"] is True
    assert cfg["prompt_adapter"]["enabled"] is False
    assert cfg["prompt_aware"]["enabled"] is False
    assert cfg["training"]["train_prompt_graph_module"] is False


def test_class_balanced_support_query_split_is_disjoint_and_train_only() -> None:
    y = torch.tensor([0, 0, 0, 1, 1, 1, 2, 2, 2, 2])
    train_mask = torch.tensor([True, True, True, True, True, True, True, True, False, False])
    support, query, stats = _class_balanced_support_query_split(
        y,
        train_mask,
        support_ratio=0.67,
        min_query_per_class=1,
        seed=7,
    )

    assert bool((support & query).any()) is False
    assert bool((support | query)[~train_mask].any()) is False
    assert int(query.sum().item()) == 3
    assert int(support.sum().item()) == 5
    for class_id in [0, 1, 2]:
        class_train = train_mask & (y == class_id)
        assert int((query & class_train).sum().item()) == 1
    assert stats["support_count"] == 5
    assert stats["query_count"] == 3


def test_query_proto_alignment_uses_support_query_pool_only() -> None:
    labels = torch.tensor([0, 0, 1, 1, 0, 1])
    support_mask = torch.tensor([True, False, True, False, False, False])
    query_mask = torch.tensor([False, True, False, True, False, False])
    pool_mask = torch.tensor([True, True, True, True, False, False])
    h_off = torch.tensor(
        [
            [1.0, 0.0],
            [0.7, 0.3],
            [0.0, 1.0],
            [0.3, 0.7],
            [1.0, 1.0],
            [1.0, 1.0],
        ]
    )
    h_on = torch.tensor(
        [
            [1.0, 0.0],
            [0.9, 0.1],
            [0.0, 1.0],
            [0.1, 0.9],
            [5.0, -5.0],
            [-5.0, 5.0],
        ]
    )

    loss_a, stats_a = _query_proto_alignment_loss(
        h_on=h_on,
        h_off=h_off,
        labels=labels,
        support_mask=support_mask,
        query_mask=query_mask,
        pool_mask=pool_mask,
        margin=0.0,
    )
    labels_changed = labels.clone()
    labels_changed[4:] = torch.tensor([1, 0])
    loss_b, stats_b = _query_proto_alignment_loss(
        h_on=h_on,
        h_off=h_off,
        labels=labels_changed,
        support_mask=support_mask,
        query_mask=query_mask,
        pool_mask=pool_mask,
        margin=0.0,
    )

    assert torch.isfinite(loss_a)
    assert float(loss_a.item()) <= 1e-7
    assert torch.allclose(loss_a, loss_b)
    assert stats_a["query_proto_supervised_count"] == 2.0
    assert stats_a["query_proto_class_count"] == 2.0
    assert stats_a["query_proto_mean_delta_dist"] > 0.0
    assert stats_b["query_proto_mean_delta_dist"] == stats_a["query_proto_mean_delta_dist"]


def _node_to_prompt_edges(out: dict, num_nodes: int, original_edges: int) -> torch.Tensor:
    prompt_edges = out["adapted_edge_index"][:, original_edges:]
    mask = (prompt_edges[0] < num_nodes) & (prompt_edges[1] >= num_nodes)
    return prompt_edges[:, mask]


def test_prompt_nodes_are_appended_after_original_nodes() -> None:
    z, h_pre, edge_index, _, train_mask = _toy_inputs()
    module = PromptGraphModuleP1(4, 3, _config())

    out = module(z=z, h_pre=h_pre, edge_index=edge_index, train_mask=train_mask)

    assert torch.allclose(out["adapted_x"][: z.size(0)], z)
    assert out["adapted_x"].shape == (z.size(0) + module.num_prompt_nodes, z.size(1))
    assert torch.allclose(out["adapted_x"][z.size(0) :], out["prompt_node_x"])


def test_augmented_graph_contains_original_edges_prompt_edges_and_weights() -> None:
    z, h_pre, edge_index, _, train_mask = _toy_inputs()
    module = PromptGraphModuleP1(4, 3, _config())

    out = module(z=z, h_pre=h_pre, edge_index=edge_index, train_mask=train_mask)

    assert torch.equal(out["adapted_edge_index"][:, : edge_index.size(1)], edge_index)
    assert out["adapted_edge_index"].size(1) == edge_index.size(1) + out["prompt_edge_count"]
    assert out["adapted_edge_weight"].numel() == out["adapted_edge_index"].size(1)
    assert out["adapted_edge_type"].numel() == out["adapted_edge_index"].size(1)
    assert torch.allclose(out["adapted_edge_weight"][: edge_index.size(1)], torch.ones(edge_index.size(1)))
    assert torch.equal(out["adapted_edge_type"][: edge_index.size(1)], torch.zeros(edge_index.size(1), dtype=torch.long))


def test_prompt_edges_have_directional_edge_types() -> None:
    z, h_pre, edge_index, _, train_mask = _toy_inputs()
    module = PromptGraphModuleP1(4, 3, _config())

    out = module(z=z, h_pre=h_pre, edge_index=edge_index, train_mask=train_mask)
    prompt_edge_type = out["adapted_edge_type"][edge_index.size(1) :]
    half = prompt_edge_type.numel() // 2

    assert torch.equal(prompt_edge_type[:half], torch.ones(half, dtype=torch.long))
    assert torch.equal(prompt_edge_type[half:], torch.full((half,), 2, dtype=torch.long))
    assert out["aux"]["edge_type_counts"] == [edge_index.size(1), half, half]


def test_prompt_edges_are_bidirectional() -> None:
    z, h_pre, edge_index, _, train_mask = _toy_inputs()
    module = PromptGraphModuleP1(4, 3, _config())

    out = module(z=z, h_pre=h_pre, edge_index=edge_index, train_mask=train_mask)
    prompt_edges = out["adapted_edge_index"][:, edge_index.size(1) :]
    edge_set = {(int(src), int(dst)) for src, dst in prompt_edges.t().tolist()}

    for src, dst in list(edge_set):
        assert (dst, src) in edge_set


def test_receiver_only_prompt_edges_disable_node_to_prompt_edges() -> None:
    z, h_pre, edge_index, _, train_mask = _toy_inputs()
    module = PromptGraphModuleP1(4, 3, _config(use_receiver_only_prompt=True))

    out = module(z=z, h_pre=h_pre, edge_index=edge_index, train_mask=train_mask)
    prompt_edges = out["adapted_edge_index"][:, edge_index.size(1) :]
    prompt_types = out["adapted_edge_type"][edge_index.size(1) :]

    assert int((prompt_types == 1).sum().item()) == 0
    assert int((prompt_types == 2).sum().item()) == prompt_types.numel()
    assert torch.all(prompt_edges[0] >= z.size(0))
    assert torch.all(prompt_edges[1] < z.size(0))
    assert out["aux"]["edge_type_counts"] == [edge_index.size(1), 0, prompt_types.numel()]


def test_pool_nodes_have_at_most_topk_prompt_connections_and_nonpool_has_none() -> None:
    z, h_pre, edge_index, _, train_mask = _toy_inputs()
    module = PromptGraphModuleP1(4, 3, _config(rho=0.0, topk_prompt_per_node=2))

    out = module(z=z, h_pre=h_pre, edge_index=edge_index, train_mask=train_mask)
    node_prompt_edges = _node_to_prompt_edges(out, z.size(0), edge_index.size(1))
    pool_mask = out["pool_mask"]

    for node_id in range(z.size(0)):
        count = int((node_prompt_edges[0] == node_id).sum().item())
        if bool(pool_mask[node_id].item()):
            assert count <= module.topk_prompt_per_node
        else:
            assert count == 0


def test_pool_mask_contains_train_nodes_and_rho_zero_keeps_supervised_pool() -> None:
    z, h_pre, edge_index, _, train_mask = _toy_inputs()
    module = PromptGraphModuleP1(4, 3, _config(rho=0.0))

    out = module(z=z, h_pre=h_pre, edge_index=edge_index, train_mask=train_mask)

    assert torch.all(out["pool_mask"][train_mask])
    assert int(out["pool_mask"].sum().item()) == int(train_mask.sum().item())


def test_utility_structural_pool_falls_back_to_structural_order_without_no_prompt_evidence() -> None:
    z, h_pre, edge_index, _, train_mask = _toy_inputs()
    structural = PromptGraphModuleP1(4, 3, _config(pool_strategy="structural", rho=0.5))
    utility = PromptGraphModuleP1(4, 3, _config(pool_strategy="utility_structural", rho=0.5))

    out_struct = structural(z=z, h_pre=h_pre, edge_index=edge_index, train_mask=train_mask)
    out_utility = utility(z=z, h_pre=h_pre, edge_index=edge_index, train_mask=train_mask)

    assert torch.equal(out_struct["pool_mask"], out_utility["pool_mask"])
    assert out_utility["aux"]["pool_strategy_id"] == 2
    assert torch.allclose(out_utility["aux"]["pool_uncertainty_component"], torch.zeros(z.size(0)))
    assert torch.allclose(out_utility["aux"]["pool_disagreement_component"], torch.zeros(z.size(0)))


def test_edge_utility_gate_has_per_edge_values() -> None:
    z, h_pre, edge_index, _, train_mask = _toy_inputs()
    module = PromptGraphModuleP1(
        4,
        3,
        _config(
            use_edge_utility=True,
            edge_utility_init=0.4,
            use_multiview_routing=True,
            use_attribute_view=True,
            use_enhanced_role_view=True,
            role_context_dim=8,
        ),
    )

    out = module(z=z, h_pre=h_pre, edge_index=edge_index, train_mask=train_mask)
    edge_utility = out["aux"]["edge_utility"]
    edge_logits = out["aux"]["edge_utility_logit"]

    assert edge_utility.shape == out["aux"]["assignment_prob"].shape
    assert edge_logits.shape == edge_utility.shape
    assert torch.all(edge_utility >= 0.0)
    assert torch.all(edge_utility <= 1.0)
    assert out["aux"]["use_edge_utility"] == 1


def test_edge_utility_supervision_uses_train_pool_ce_delta() -> None:
    z, h_pre, edge_index, y, train_mask = _toy_inputs()
    module = PromptGraphModuleP1(4, 3, _config(use_edge_utility=True, edge_utility_init=0.5))
    out = module(z=z, h_pre=h_pre, edge_index=edge_index, train_mask=train_mask)
    logits_off = torch.tensor(
        [
            [0.0, 1.0],
            [1.0, 0.0],
            [1.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 0.0],
        ]
    )
    logits_on = torch.tensor(
        [
            [2.0, 0.0],
            [1.0, 0.0],
            [0.0, 2.0],
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 0.0],
        ]
    )

    loss, stats = _edge_utility_supervision_loss(
        prompt_out=out,
        logits_on=logits_on,
        logits_off=logits_off,
        labels=y,
        train_mask=train_mask,
        margin=0.0,
    )

    assert torch.isfinite(loss)
    assert stats["edge_utility_supervised_count"] == float(train_mask.sum().item())
    assert stats["edge_utility_positive_count"] == float(train_mask.sum().item())
    assert stats["edge_utility_negative_count"] == 0.0


def test_correction_alignment_loss_uses_classifier_direction_for_helpful_nodes() -> None:
    z, _, _, y, train_mask = _toy_inputs()
    model = FaithfulGP2F(BaseGCN(in_channels=4, hidden_channels=3, num_layers=2), hidden_dim=3, num_classes=2)
    prompt_out = {"pool_mask": train_mask.clone(), "edge_scale": z.new_tensor(1.0)}
    h_off = torch.zeros(z.size(0), 3)
    h_on = torch.zeros(z.size(0), 3)
    h_on[0] = model.classifier.weight.detach()[0]
    h_on[2] = model.classifier.weight.detach()[1]
    logits_off = torch.zeros(z.size(0), 2)
    logits_on = torch.zeros(z.size(0), 2)
    logits_on[0, 0] = 2.0
    logits_on[2, 1] = 2.0

    align_loss, anti_harm, stats = _correction_alignment_losses(
        model=model,
        prompt_out=prompt_out,
        h_on=h_on,
        h_off=h_off,
        logits_on=logits_on,
        logits_off=logits_off,
        labels=y,
        train_mask=train_mask,
        margin=0.0,
    )

    assert torch.isfinite(align_loss)
    assert torch.isfinite(anti_harm)
    assert stats["correction_alignment_node_count"] == float(train_mask.sum().item())
    assert stats["correction_alignment_cosine_mean"] > 0.99


def test_p13_config_keeps_zero_message_scale_safety_option() -> None:
    cfg = read_yaml(Path(__file__).resolve().parents[1] / "configs" / "gp2f_prompt_p13_utility_correction.yaml")

    assert cfg["experiment"]["prompt_variant"] == "p13_utility_correction"
    assert cfg["prompt_graph"]["pool_strategy"] == "utility_structural"
    assert cfg["prompt_graph"]["use_edge_utility"] is True
    assert 0.0 in cfg["prompt_aware"]["message_scale_grid"]


def test_val_test_labels_do_not_affect_p1_outputs() -> None:
    z, h_pre, edge_index, y, train_mask = _toy_inputs()
    changed_y = y.clone()
    changed_y[~train_mask] = 1 - changed_y[~train_mask]
    module = PromptGraphModuleP1(4, 3, _config())
    module.eval()

    with torch.no_grad():
        original = module(z=z, h_pre=h_pre, edge_index=edge_index, train_mask=train_mask)
        relabeled = module(z=z, h_pre=h_pre, edge_index=edge_index, train_mask=train_mask)

    assert torch.equal(y[train_mask], changed_y[train_mask])
    assert torch.allclose(original["adapted_x"], relabeled["adapted_x"])
    assert torch.equal(original["adapted_edge_index"], relabeled["adapted_edge_index"])
    assert torch.allclose(original["adapted_edge_weight"], relabeled["adapted_edge_weight"])


def test_prompt_edge_count_matches_connected_edge_count() -> None:
    z, h_pre, edge_index, _, train_mask = _toy_inputs()
    module = PromptGraphModuleP1(4, 3, _config())

    out = module(z=z, h_pre=h_pre, edge_index=edge_index, train_mask=train_mask)

    assert out["prompt_edge_count"] == out["aux"]["connected_edge_count"]
    assert out["prompt_edge_count"] == out["adapted_edge_index"].size(1) - edge_index.size(1)


def test_losses_are_finite_and_have_expected_shape() -> None:
    z, h_pre, edge_index, _, train_mask = _toy_inputs()
    module = PromptGraphModuleP1(4, 3, _config())

    out = module(z=z, h_pre=h_pre, edge_index=edge_index, train_mask=train_mask)

    assert torch.isfinite(prompt_edge_l1_loss(out))
    assert torch.isfinite(prompt_balance_loss(out))
    assert out["aux"]["prompt_usage"].shape == (module.num_prompt_nodes,)
    assert out["aux"]["prompt_usage_full"].shape == (module.num_prompt_nodes,)
    assert torch.allclose(out["aux"]["prompt_usage_full"].sum(), torch.tensor(1.0))


def test_edge_scale_multiplier_controls_prompt_edge_weights() -> None:
    z, h_pre, edge_index, _, train_mask = _toy_inputs()
    module = PromptGraphModuleP1(4, 3, _config())

    zero = module(z=z, h_pre=h_pre, edge_index=edge_index, train_mask=train_mask, edge_scale_multiplier=0.0)
    full = module(z=z, h_pre=h_pre, edge_index=edge_index, train_mask=train_mask, edge_scale_multiplier=1.0)

    original_edges = edge_index.size(1)
    assert torch.allclose(zero["adapted_edge_weight"][original_edges:], torch.zeros_like(zero["adapted_edge_weight"][original_edges:]))
    assert float(full["adapted_edge_weight"][original_edges:].max().item()) > 0.0


def test_rejection_gate_scales_prompt_edges_and_is_reported() -> None:
    z, h_pre, edge_index, _, train_mask = _toy_inputs()
    gated = PromptGraphModuleP1(
        4,
        3,
        _config(use_rejection_gate=True, rejection_gate_bias_init=-2.0),
    )
    open_gate = PromptGraphModuleP1(
        4,
        3,
        _config(use_rejection_gate=True, rejection_gate_bias_init=2.0),
    )
    open_gate.load_state_dict(gated.state_dict(), strict=False)
    with torch.no_grad():
        open_gate.rejection_gate_mlp[-1].bias.fill_(2.0)

    gated_out = gated(z=z, h_pre=h_pre, edge_index=edge_index, train_mask=train_mask)
    open_out = open_gate(z=z, h_pre=h_pre, edge_index=edge_index, train_mask=train_mask)
    original_edges = edge_index.size(1)

    assert 0.0 < gated_out["aux"]["pool_acceptance_mean"].item() < 1.0
    assert open_out["aux"]["pool_acceptance_mean"].item() > gated_out["aux"]["pool_acceptance_mean"].item()
    assert open_out["adapted_edge_weight"][original_edges:].mean() > gated_out["adapted_edge_weight"][original_edges:].mean()
    assert torch.isfinite(prompt_acceptance_loss(gated_out))


def test_rejection_gate_can_use_role_and_routing_features() -> None:
    z, h_pre, edge_index, _, train_mask = _toy_inputs()
    module = PromptGraphModuleP1(
        4,
        3,
        _config(
            use_rejection_gate=True,
            use_multiview_routing=True,
            rejection_gate_use_role_context=True,
            rejection_gate_use_routing_features=True,
            rho=1.0,
        ),
    )

    out = module(z=z, h_pre=h_pre, edge_index=edge_index, train_mask=train_mask)
    context = out["aux"]["rejection_gate_context"]

    assert context.shape[0] == int(out["pool_mask"].sum().item())
    assert context.shape[1] == 5 * z.size(1) + 6 + 4
    assert out["aux"]["pool_acceptance_gate"].shape[0] == context.shape[0]


def test_hard_acceptance_keeps_only_top_ratio_prompt_edges() -> None:
    z, h_pre, edge_index, _, train_mask = _toy_inputs()
    module = PromptGraphModuleP1(
        4,
        3,
        _config(
            use_rejection_gate=True,
            rejection_gate_bias_init=0.0,
            use_hard_acceptance=True,
            hard_acceptance_ratio=0.5,
            hard_acceptance_straight_through=False,
            rho=1.0,
            topk_prompt_per_node=1,
        ),
    )

    out = module(z=z, h_pre=h_pre, edge_index=edge_index, train_mask=train_mask)
    mask = out["aux"]["hard_acceptance_mask"]
    prompt_weights = out["aux"]["prompt_edge_weight"]
    forward_weights = prompt_weights[: int(mask.numel())]

    assert out["aux"]["use_hard_acceptance"] == 1
    assert torch.allclose(out["aux"]["hard_acceptance_selected_ratio"], mask.mean())
    assert int(mask.sum().item()) == 3
    assert torch.all(forward_weights[mask.bool()] > 0)
    assert torch.allclose(forward_weights[~mask.bool()], torch.zeros_like(forward_weights[~mask.bool()]))


def test_acceptance_budget_penalizes_only_out_of_range_mean() -> None:
    z, h_pre, edge_index, _, train_mask = _toy_inputs()
    module = PromptGraphModuleP1(
        4,
        3,
        _config(use_rejection_gate=True, rejection_gate_bias_init=-2.0),
    )

    out = module(z=z, h_pre=h_pre, edge_index=edge_index, train_mask=train_mask)

    assert torch.isfinite(prompt_acceptance_budget_loss(out, min_acceptance=0.01, max_acceptance=0.50))
    assert prompt_acceptance_budget_loss(out, min_acceptance=0.90, max_acceptance=None) > 0
    assert prompt_acceptance_budget_loss(out, min_acceptance=None, max_acceptance=0.01) > 0


def test_train_only_acceptance_supervision_ignores_val_test_labels() -> None:
    z, h_pre, edge_index, y, train_mask = _toy_inputs()
    module = PromptGraphModuleP1(
        4,
        3,
        _config(use_rejection_gate=True, rejection_gate_bias_init=-1.0, rho=1.0),
    )
    changed_y = y.clone()
    changed_y[~train_mask] = 1 - changed_y[~train_mask]
    out = module(z=z, h_pre=h_pre, edge_index=edge_index, train_mask=train_mask)
    logits_off = torch.tensor(
        [
            [2.0, 0.0],
            [1.0, 0.0],
            [0.0, 2.0],
            [0.0, 1.0],
            [1.0, 0.0],
            [0.0, 1.0],
        ]
    )
    logits_on = logits_off.clone()
    logits_on[0] = torch.tensor([0.0, 2.0])
    logits_on[2] = torch.tensor([2.0, 0.0])

    original_loss, original_stats = _acceptance_supervision_loss(
        prompt_out=out,
        logits_on=logits_on,
        logits_off=logits_off,
        labels=y,
        train_mask=train_mask,
    )
    relabeled_loss, relabeled_stats = _acceptance_supervision_loss(
        prompt_out=out,
        logits_on=logits_on,
        logits_off=logits_off,
        labels=changed_y,
        train_mask=train_mask,
    )

    assert torch.isfinite(original_loss)
    assert torch.allclose(original_loss, relabeled_loss)
    assert original_stats == relabeled_stats


def test_acceptance_supervision_ce_delta_produces_positive_and_negative_targets() -> None:
    z, h_pre, edge_index, y, train_mask = _toy_inputs()
    module = PromptGraphModuleP1(
        4,
        3,
        _config(use_rejection_gate=True, rejection_gate_bias_init=-1.0, rho=1.0),
    )
    out = module(z=z, h_pre=h_pre, edge_index=edge_index, train_mask=train_mask)
    logits_off = torch.tensor(
        [
            [0.0, 2.0],
            [1.0, 0.0],
            [0.0, 2.0],
            [0.0, 1.0],
            [1.0, 0.0],
            [0.0, 1.0],
        ]
    )
    logits_on = logits_off.clone()
    logits_on[0] = torch.tensor([2.0, 0.0])
    logits_on[2] = torch.tensor([2.0, 0.0])

    loss, stats = _acceptance_supervision_loss(
        prompt_out=out,
        logits_on=logits_on,
        logits_off=logits_off,
        labels=y,
        train_mask=train_mask,
        signal="ce_delta",
    )

    assert torch.isfinite(loss)
    assert stats["acceptance_supervised_count"] == 2.0
    assert stats["acceptance_positive_count"] == 1.0
    assert stats["acceptance_negative_count"] == 1.0


def test_prompt_role_diversity_loss_is_finite_and_has_gradients() -> None:
    module = PromptGraphModuleP1(4, 3, _config())

    loss = prompt_role_diversity_loss(module)
    loss.backward()

    assert torch.isfinite(loss)
    assert module.prompt_keys.grad is not None


def test_multiview_role_diversity_loss_has_gradients_for_all_view_keys() -> None:
    module = PromptGraphModuleP1(4, 3, _config(use_multiview_routing=True))

    loss = prompt_role_diversity_loss(module)
    loss.backward()

    assert torch.isfinite(loss)
    assert module.semantic_prompt_keys.grad is not None
    assert module.structural_prompt_keys.grad is not None
    assert module.role_prompt_keys.grad is not None
    assert module.semantic_prompt_keys.grad.abs().sum().item() > 0.0
    assert module.structural_prompt_keys.grad.abs().sum().item() > 0.0
    assert module.role_prompt_keys.grad.abs().sum().item() > 0.0


def test_capacity_routing_spreads_tied_assignments_across_prompts() -> None:
    z, h_pre, edge_index, _, train_mask = _toy_inputs()
    module = PromptGraphModuleP1(
        4,
        3,
        _config(rho=1.0, topk_prompt_per_node=1, use_capacity_routing=True, capacity_factor=1.0),
    )
    with torch.no_grad():
        for parameter in module.query_mlp.parameters():
            parameter.zero_()
        module.prompt_keys.zero_()

    out = module(z=z, h_pre=h_pre, edge_index=edge_index, train_mask=train_mask)

    assert out["aux"]["capacity_routing_enabled"] == 1
    assert out["aux"]["capacity_overflow_count"] == 0
    assert int((out["aux"]["prompt_usage"] > 0).sum().item()) >= 3


def test_prompt_graph_receives_gradients_through_faithful_gp2f() -> None:
    z, _, edge_index, y, train_mask = _toy_inputs()
    module = PromptGraphModuleP1(4, 3, _config())
    model = FaithfulGP2F(BaseGCN(in_channels=4, hidden_channels=3, num_layers=2), hidden_dim=3, num_classes=2)

    h_pre = model.encode_frozen(z, edge_index)
    prompt_out = module(z=z, h_pre=h_pre.detach(), edge_index=edge_index, train_mask=train_mask)
    model_out = model.forward_with_h_pre(
        z,
        edge_index,
        h_pre=h_pre,
        adapted_x=prompt_out["adapted_x"],
        adapted_edge_index=prompt_out["adapted_edge_index"],
        adapted_edge_weight=prompt_out["adapted_edge_weight"],
        return_aux=True,
    )
    loss = (
        F.cross_entropy(model_out["logits"][train_mask], y[train_mask])
        + prompt_edge_l1_loss(prompt_out)
        + prompt_balance_loss(prompt_out)
    )
    loss.backward()

    assert model_out["logits"].shape == (z.size(0), 2)
    assert module.prompt_node_x.grad is not None
    assert module.prompt_keys.grad is not None
    assert module.edge_scale_logit.grad is not None
    assert any(parameter.grad is not None for parameter in module.query_mlp.parameters())
    assert torch.isfinite(loss)
    assert all(torch.isfinite(param.grad).all() for param in module.parameters() if param.grad is not None)


def test_full_softmax_balance_gives_all_prompt_keys_gradient() -> None:
    z, h_pre, edge_index, _, train_mask = _toy_inputs()
    module = PromptGraphModuleP1(4, 3, _config(num_prompt_nodes=4, topk_prompt_per_node=1))

    out = module(z=z, h_pre=h_pre, edge_index=edge_index, train_mask=train_mask)
    loss = prompt_balance_loss(out)
    loss.backward()

    assert module.prompt_keys.grad is not None
    assert torch.all(module.prompt_keys.grad.abs().sum(dim=1) > 0)


def test_forward_backward_has_no_nan() -> None:
    z, h_pre, edge_index, _, train_mask = _toy_inputs()
    module = PromptGraphModuleP1(4, 3, _config(structural_base="h_pre_detached"))

    out = module(z=z, h_pre=h_pre, edge_index=edge_index, train_mask=train_mask)
    loss = out["adapted_edge_weight"].pow(2).mean() + out["prompt_node_x"].pow(2).mean()
    loss.backward()

    assert torch.isfinite(loss)
    assert all(torch.isfinite(param.grad).all() for param in module.parameters() if param.grad is not None)


def test_multiview_routing_reports_view_gate_distribution() -> None:
    z, h_pre, edge_index, _, train_mask = _toy_inputs()
    module = PromptGraphModuleP1(4, 3, _config(use_multiview_routing=True, rho=1.0))

    out = module(z=z, h_pre=h_pre, edge_index=edge_index, train_mask=train_mask)
    view_gate = out["aux"]["view_gate"]

    assert out["aux"]["use_multiview_routing"] == 1
    assert view_gate.shape == (int(out["pool_mask"].sum().item()), 3)
    assert torch.allclose(view_gate.sum(dim=-1), torch.ones(view_gate.size(0)))
    assert out["aux"]["view_gate_mean"].shape == (3,)
    assert torch.isfinite(prompt_view_entropy_loss(out))


def test_attribute_view_expands_routing_to_four_views() -> None:
    z, h_pre, edge_index, _, train_mask = _toy_inputs()
    module = PromptGraphModuleP1(
        4,
        3,
        _config(use_multiview_routing=True, use_attribute_view=True, use_enhanced_role_view=True, rho=1.0),
    )

    out = module(z=z, h_pre=h_pre, edge_index=edge_index, train_mask=train_mask)
    aux = out["aux"]
    view_gate = aux["view_gate"]

    assert aux["use_attribute_view"] == 1
    assert aux["use_enhanced_role_view"] == 1
    assert view_gate.shape == (int(out["pool_mask"].sum().item()), 4)
    assert torch.allclose(view_gate.sum(dim=-1), torch.ones(view_gate.size(0)))
    assert aux["view_gate_mean"].shape == (4,)
    assert torch.isfinite(aux["attribute_route_margin"])
    assert torch.isfinite(prompt_view_prior_loss(out, [0.15, 0.25, 0.30, 0.30]))


def test_attribute_view_is_label_free_under_val_test_relabeling() -> None:
    z, h_pre, edge_index, y, train_mask = _toy_inputs()
    changed_y = y.clone()
    changed_y[~train_mask] = 1 - changed_y[~train_mask]
    module = PromptGraphModuleP1(
        4,
        3,
        _config(
            use_multiview_routing=True,
            use_attribute_view=True,
            use_enhanced_role_view=True,
            use_pattern_prompt_bank=True,
            rho=1.0,
        ),
    )

    out = module(z=z, h_pre=h_pre, edge_index=edge_index, train_mask=train_mask)
    original_loss = prompt_usage_consistency_loss(out, y, train_mask)
    relabeled_loss = prompt_usage_consistency_loss(out, changed_y, train_mask)

    assert torch.allclose(original_loss, relabeled_loss)
    assert out["aux"]["attribute_query"].shape[0] == int(out["pool_mask"].sum().item())


def test_train_only_usage_consistency_ignores_val_test_labels() -> None:
    z, h_pre, edge_index, y, train_mask = _toy_inputs()
    module = PromptGraphModuleP1(4, 3, _config(use_multiview_routing=True, rho=1.0))
    changed_y = y.clone()
    changed_y[~train_mask] = 1 - changed_y[~train_mask]

    out = module(z=z, h_pre=h_pre, edge_index=edge_index, train_mask=train_mask)
    original_loss = prompt_usage_consistency_loss(out, y, train_mask)
    relabeled_loss = prompt_usage_consistency_loss(out, changed_y, train_mask)

    assert torch.isfinite(original_loss)
    assert torch.allclose(original_loss, relabeled_loss)


def test_prompt_slot_usage_stats_reports_dominant_and_active_count() -> None:
    usage = torch.tensor([0.70, 0.10, 0.04, 0.05, 0.11])

    stats = _prompt_slot_usage_stats(usage)
    empty_stats = _prompt_slot_usage_stats(torch.empty(0))

    assert stats["dominant_prompt_slot_ratio"] == torch.tensor(0.70).item()
    assert stats["active_prompt_slot_count@0.05"] == 4.0
    assert empty_stats["dominant_prompt_slot_ratio"] == 0.0
    assert empty_stats["active_prompt_slot_count@0.05"] == 0.0


def test_utility_receive_gate_softly_scales_prompt_edges() -> None:
    z, h_pre, edge_index, _, train_mask = _toy_inputs()
    module = PromptGraphModuleP1(
        4,
        3,
        _config(
            use_utility_receive_gate=True,
            utility_receive_gate_min=0.10,
            utility_receive_gate_init=0.50,
            rho=1.0,
        ),
    )

    out = module(z=z, h_pre=h_pre, edge_index=edge_index, train_mask=train_mask)
    aux = out["aux"]

    assert aux["use_utility_receive_gate"] == 1
    assert aux["utility_receive_gate"].numel() == aux["pool_idx"].numel()
    assert aux["utility_receive_gate"].min().item() >= 0.10
    assert aux["utility_receive_gate"].max().item() <= 1.0
    assert torch.isfinite(aux["utility_receive_gate_logit"]).all()


def test_utility_receive_gate_loss_is_train_only() -> None:
    z, h_pre, edge_index, y, _ = _toy_inputs()
    train_mask = torch.tensor([True, True, True, True, False, False])
    module = PromptGraphModuleP1(
        4,
        3,
        _config(use_utility_receive_gate=True, utility_receive_gate_min=0.10, rho=1.0),
    )
    out = module(z=z, h_pre=h_pre, edge_index=edge_index, train_mask=train_mask)
    logits_off = torch.tensor(
        [
            [2.0, 0.0],
            [1.8, 0.0],
            [0.0, 2.0],
            [0.0, 1.8],
            [9.0, -9.0],
            [-9.0, 9.0],
        ]
    )
    logits_on = logits_off.clone()
    logits_on[0, y[0]] += 0.5
    logits_on[1, y[1]] -= 0.5
    logits_on[2, y[2]] += 0.5
    logits_on[3, y[3]] -= 0.5
    relabeled = y.clone()
    relabeled[4:] = 1 - relabeled[4:]

    loss, stats = _utility_receive_gate_loss(
        prompt_out=out,
        logits_on=logits_on,
        logits_off=logits_off,
        labels=y,
        train_mask=train_mask,
        quantile=0.5,
        eps=0.0,
        class_balanced=True,
    )
    relabeled_loss, relabeled_stats = _utility_receive_gate_loss(
        prompt_out=out,
        logits_on=logits_on,
        logits_off=logits_off,
        labels=relabeled,
        train_mask=train_mask,
        quantile=0.5,
        eps=0.0,
        class_balanced=True,
    )

    assert torch.isfinite(loss)
    assert stats["utility_gate_supervised_count"] == 4.0
    assert stats["utility_gate_positive_count"] == 2.0
    assert stats["utility_gate_negative_count"] == 2.0
    assert torch.allclose(loss, relabeled_loss)
    assert stats["utility_receive_gate_loss"] == relabeled_stats["utility_receive_gate_loss"]


def test_margin_utility_receive_gate_loss_uses_positive_and_negative_deltas() -> None:
    z, h_pre, edge_index, y, _ = _toy_inputs()
    train_mask = torch.tensor([True, True, True, True, False, False])
    module = PromptGraphModuleP1(
        4,
        3,
        _config(use_utility_receive_gate=True, utility_receive_gate_min=0.0, rho=1.0),
    )
    out = module(z=z, h_pre=h_pre, edge_index=edge_index, train_mask=train_mask)
    logits_off = torch.tensor(
        [
            [2.0, 0.0],
            [2.0, 0.0],
            [0.0, 2.0],
            [0.0, 2.0],
            [9.0, -9.0],
            [-9.0, 9.0],
        ]
    )
    logits_on = logits_off.clone()
    logits_on[0, y[0]] += 0.7
    logits_on[1, y[1]] -= 0.7
    logits_on[2, y[2]] += 0.7
    logits_on[3, y[3]] -= 0.7

    loss, stats = _utility_receive_gate_loss(
        prompt_out=out,
        logits_on=logits_on,
        logits_off=logits_off,
        labels=y,
        train_mask=train_mask,
        eps=0.002,
        class_balanced=True,
        label_strategy="margin",
    )
    budget = utility_receive_gate_budget_loss(out, min_receive=0.20, max_receive=0.60)

    assert torch.isfinite(loss)
    assert torch.isfinite(budget)
    assert stats["utility_gate_label_strategy"] == "margin"
    assert stats["utility_gate_positive_count"] == 2.0
    assert stats["utility_gate_negative_count"] == 2.0


def test_class_aware_routing_expands_prompt_slots() -> None:
    module = PromptGraphModuleP1(
        4,
        3,
        _config(num_prompt_nodes=2, use_class_aware_routing=True, num_classes=3, residual_prompt_count=2),
    )

    assert module.num_prompt_nodes == 5
    assert module.num_class_prompt_slots == 3
    assert module.prompt_node_x.shape[0] == 5
    assert module.prompt_keys.shape[0] == 5


def test_train_only_class_key_prototype_initialization_ignores_val_test_labels() -> None:
    z, h_pre, edge_index, y, train_mask = _toy_inputs()
    changed_y = y.clone()
    changed_y[~train_mask] = 1 - changed_y[~train_mask]
    module = PromptGraphModuleP1(
        4,
        3,
        _config(
            use_multiview_routing=True,
            use_class_aware_routing=True,
            num_classes=2,
            residual_prompt_count=1,
            query_dim=3,
            rho=1.0,
        ),
    )

    stats = module.initialize_class_keys_from_train_prototypes(
        z=z,
        h_pre=h_pre,
        edge_index=edge_index,
        train_mask=train_mask,
        labels=y,
    )
    initialized_keys = module.structural_prompt_keys.detach().clone()
    module.initialize_class_keys_from_train_prototypes(
        z=z,
        h_pre=h_pre,
        edge_index=edge_index,
        train_mask=train_mask,
        labels=changed_y,
    )

    assert stats["class_key_proto_init_coverage"] == 1.0
    assert stats["class_key_proto_init_missing_classes"] == []
    assert torch.allclose(initialized_keys[:2], module.structural_prompt_keys.detach()[:2])
    assert module.class_key_init_coverage == 1.0


def test_class_key_prototype_initialization_reports_missing_classes() -> None:
    z, h_pre, edge_index, y, train_mask = _toy_inputs()
    only_class_zero = train_mask & (y == 0)
    module = PromptGraphModuleP1(
        4,
        3,
        _config(use_class_aware_routing=True, num_classes=2, residual_prompt_count=1, query_dim=3, rho=1.0),
    )

    stats = module.initialize_class_keys_from_train_prototypes(
        z=z,
        h_pre=h_pre,
        edge_index=edge_index,
        train_mask=only_class_zero,
        labels=y,
    )

    assert stats["class_key_proto_init_coverage"] == 0.5
    assert stats["class_key_proto_init_missing_classes"] == [1]


def test_class_route_loss_is_train_only_and_ignores_val_test_labels() -> None:
    z, h_pre, edge_index, y, train_mask = _toy_inputs()
    changed_y = y.clone()
    changed_y[~train_mask] = 1 - changed_y[~train_mask]
    module = PromptGraphModuleP1(
        4,
        3,
        _config(use_class_aware_routing=True, num_classes=2, residual_prompt_count=1, rho=1.0),
    )

    out = module(z=z, h_pre=h_pre, edge_index=edge_index, train_mask=train_mask)
    original_loss = prompt_class_route_loss(out, y, train_mask)
    relabeled_loss = prompt_class_route_loss(out, changed_y, train_mask)

    assert torch.isfinite(original_loss)
    assert torch.allclose(original_loss, relabeled_loss)


def test_class_route_loss_is_zero_when_class_aware_routing_is_disabled() -> None:
    z, h_pre, edge_index, y, train_mask = _toy_inputs()
    module = PromptGraphModuleP1(
        4,
        3,
        _config(use_class_aware_routing=False, use_multiview_routing=True, rho=1.0),
    )

    out = module(z=z, h_pre=h_pre, edge_index=edge_index, train_mask=train_mask)
    loss = prompt_class_route_loss(out, y, train_mask)

    assert out["aux"]["num_class_prompt_slots"] == 0
    assert torch.allclose(loss, torch.zeros_like(loss))


def test_key_proto_loss_uses_class_slots_and_has_gradients() -> None:
    z, h_pre, edge_index, y, train_mask = _toy_inputs()
    module = PromptGraphModuleP1(
        4,
        3,
        _config(
            use_class_aware_routing=True,
            use_multiview_routing=True,
            num_classes=2,
            residual_prompt_count=1,
            rho=1.0,
        ),
    )

    out = module(z=z, h_pre=h_pre, edge_index=edge_index, train_mask=train_mask)
    loss = prompt_key_proto_loss(module, out, y, train_mask) + prompt_class_route_loss(out, y, train_mask)
    loss.backward()

    assert torch.isfinite(loss)
    assert module.structural_prompt_keys.grad is not None
    assert module.structural_prompt_keys.grad[:2].abs().sum() > 0
    assert any(parameter.grad is not None for parameter in module.structural_query_mlp.parameters())


def test_residual_prompt_slots_are_not_class_targets() -> None:
    z, h_pre, edge_index, y, train_mask = _toy_inputs()
    module = PromptGraphModuleP1(
        4,
        3,
        _config(use_class_aware_routing=True, num_classes=2, residual_prompt_count=2, rho=1.0),
    )

    out = module(z=z, h_pre=h_pre, edge_index=edge_index, train_mask=train_mask)
    logits = out["aux"]["routing_logits"]
    loss = prompt_class_route_loss(out, y, train_mask)
    manual = F.cross_entropy(logits[train_mask[out["aux"]["pool_idx"]]][:, :2], y[train_mask])

    assert torch.allclose(loss, manual)


def test_pattern_medoid_initialization_is_label_free_and_reports_coverage() -> None:
    z, h_pre, edge_index, _, train_mask = _toy_inputs()
    module = PromptGraphModuleP1(
        4,
        3,
        _config(use_multiview_routing=True, use_pattern_prompt_bank=True, rho=1.0),
    )
    before = module.structural_prompt_keys.detach().clone()

    stats = module.initialize_pattern_keys_from_pool_medoids(
        z=z,
        h_pre=h_pre,
        edge_index=edge_index,
        train_mask=train_mask,
    )

    assert stats["pattern_key_init_coverage"] == 1.0
    assert len(stats["pattern_key_init_selected_nodes"]) == module.num_prompt_nodes
    assert not torch.allclose(before, module.structural_prompt_keys.detach())
    assert module.pattern_key_init_coverage == 1.0
    selected = torch.tensor(stats["pattern_key_init_selected_nodes"], dtype=torch.long)
    assert torch.allclose(module.prompt_node_x.detach(), z[selected])


def test_benefit_gate_scales_edges_and_supervision_is_train_only() -> None:
    z, h_pre, edge_index, y, train_mask = _toy_inputs()
    module = PromptGraphModuleP1(
        4,
        3,
        _config(use_benefit_gate=True, benefit_gate_bias_init=-2.0, rho=1.0),
    )

    out = module(z=z, h_pre=h_pre, edge_index=edge_index, train_mask=train_mask)
    benefit = out["aux"]["benefit_gate"]
    logits_off = torch.tensor(
        [
            [2.0, 0.0],
            [1.5, 0.2],
            [0.0, 2.0],
            [0.1, 1.5],
            [2.0, 0.0],
            [0.0, 2.0],
        ]
    )
    logits_on = logits_off.clone()
    logits_on[train_mask, y[train_mask]] += torch.tensor([0.5, -0.5])
    loss, stats = _benefit_supervision_loss(
        prompt_out=out,
        logits_on=logits_on,
        logits_off=logits_off,
        labels=y,
        train_mask=train_mask,
        margin=0.01,
        balance_targets=True,
    )

    assert benefit.shape == out["aux"]["assignment_prob"].shape
    assert 0.0 < float(benefit.mean().item()) < 1.0
    assert torch.isfinite(loss)
    assert stats["benefit_supervised_count"] == 2.0
    assert stats["benefit_positive_count"] == 1.0
    assert stats["benefit_negative_count"] == 1.0


def test_benefit_supervision_quantile_keeps_zero_delta_samples() -> None:
    z, h_pre, edge_index, y, train_mask = _toy_inputs()
    module = PromptGraphModuleP1(
        4,
        3,
        _config(use_benefit_gate=True, benefit_gate_bias_init=-2.0, rho=1.0),
    )

    out = module(z=z, h_pre=h_pre, edge_index=edge_index, train_mask=train_mask)
    logits_off = torch.tensor(
        [
            [2.0, 0.0],
            [1.5, 0.2],
            [0.0, 2.0],
            [0.1, 1.5],
            [2.0, 0.0],
            [0.0, 2.0],
        ]
    )
    logits_on = logits_off.clone()
    loss, stats = _benefit_supervision_loss(
        prompt_out=out,
        logits_on=logits_on,
        logits_off=logits_off,
        labels=y,
        train_mask=train_mask,
        margin=0.01,
        balance_targets=True,
        label_strategy="quantile",
        quantile=0.5,
    )

    assert torch.isfinite(loss)
    assert stats["benefit_label_strategy_quantile"] == 1.0
    assert stats["benefit_supervised_count"] == 2.0
    assert stats["benefit_positive_count"] == 1.0
    assert stats["benefit_negative_count"] == 1.0


def test_benefit_supervision_hybrid_ignores_tiny_delta_samples() -> None:
    z, h_pre, edge_index, y, train_mask = _toy_inputs()
    module = PromptGraphModuleP1(
        4,
        3,
        _config(use_benefit_gate=True, benefit_gate_bias_init=-2.0, rho=1.0),
    )

    out = module(z=z, h_pre=h_pre, edge_index=edge_index, train_mask=train_mask)
    logits_off = torch.tensor(
        [
            [2.0, 0.0],
            [1.5, 0.2],
            [0.0, 2.0],
            [0.1, 1.5],
            [2.0, 0.0],
            [0.0, 2.0],
        ]
    )
    logits_on = logits_off.clone()
    loss, stats = _benefit_supervision_loss(
        prompt_out=out,
        logits_on=logits_on,
        logits_off=logits_off,
        labels=y,
        train_mask=train_mask,
        margin=0.0,
        balance_targets=True,
        label_strategy="hybrid_quantile_margin",
        quantile=0.5,
        eps=1e-4,
    )

    assert torch.isfinite(loss)
    assert stats["benefit_supervised_count"] == 0.0
    assert stats["benefit_ignored_count"] == 2.0


def test_prompt_correction_loss_supervises_only_positive_train_pool_nodes() -> None:
    z, h_pre, edge_index, y, train_mask = _toy_inputs()
    module = PromptGraphModuleP1(
        4,
        3,
        _config(use_benefit_gate=True, use_hard_receive_gate=True, hard_receive_ratio=1.0, rho=1.0),
    )
    out = module(z=z, h_pre=h_pre, edge_index=edge_index, train_mask=train_mask)
    logits_off = torch.tensor(
        [
            [2.0, 0.0],
            [1.5, 0.2],
            [0.0, 2.0],
            [0.1, 1.5],
            [2.0, 0.0],
            [0.0, 2.0],
        ]
    )
    logits_on = logits_off.clone().requires_grad_(True)
    train_idx = torch.where(train_mask)[0]
    logits_on = logits_on.clone()
    logits_on[train_idx[0], y[train_idx[0]]] += 0.6
    logits_on[train_idx[1], y[train_idx[1]]] -= 0.6

    correction, anti_harm, stats = _prompt_correction_losses(
        prompt_out=out,
        logits_on=logits_on,
        logits_off=logits_off,
        labels=y,
        train_mask=train_mask,
        eps=1e-4,
    )

    assert torch.isfinite(correction)
    assert torch.isfinite(anti_harm)
    assert stats["prompt_correction_supervised_count"] == 1.0
    assert stats["prompt_correction_harmful_count"] == 1.0
    assert stats["prompt_correction_target_norm"] > 0.0
    assert stats["prompt_delta_logit_norm"] > 0.0


def test_prompt_correction_ignores_tiny_delta_samples() -> None:
    z, h_pre, edge_index, y, train_mask = _toy_inputs()
    module = PromptGraphModuleP1(4, 3, _config(use_benefit_gate=True, rho=1.0))
    out = module(z=z, h_pre=h_pre, edge_index=edge_index, train_mask=train_mask)
    logits_off = torch.tensor(
        [
            [2.0, 0.0],
            [1.5, 0.2],
            [0.0, 2.0],
            [0.1, 1.5],
            [2.0, 0.0],
            [0.0, 2.0],
        ]
    )
    logits_on = logits_off.clone().requires_grad_(True)

    correction, anti_harm, stats = _prompt_correction_losses(
        prompt_out=out,
        logits_on=logits_on,
        logits_off=logits_off,
        labels=y,
        train_mask=train_mask,
        eps=1e-4,
    )

    assert correction.item() == 0.0
    assert anti_harm.item() == 0.0
    assert stats["prompt_correction_supervised_count"] == 0.0
    assert stats["prompt_correction_harmful_count"] == 0.0


def test_prompt_message_help_loss_is_train_pool_only_and_class_balanced() -> None:
    z, h_pre, edge_index, y, train_mask = _toy_inputs()
    module = PromptGraphModuleP1(4, 3, _config(rho=1.0))
    out = module(z=z, h_pre=h_pre, edge_index=edge_index, train_mask=train_mask)
    logits_off = torch.tensor(
        [
            [2.0, 0.0],
            [10.0, -10.0],
            [0.0, 2.0],
            [-10.0, 10.0],
            [10.0, -10.0],
            [-10.0, 10.0],
        ]
    )
    logits_on = logits_off.clone()
    train_idx = torch.where(train_mask)[0]
    logits_on[train_idx[0], y[train_idx[0]]] += 0.2
    logits_on[train_idx[1], y[train_idx[1]]] -= 0.2
    logits_on.requires_grad_()

    loss, anti_harm, stats = _prompt_message_help_loss(
        prompt_out=out,
        logits_on=logits_on,
        logits_off=logits_off,
        labels=y,
        train_mask=train_mask,
        margin=0.01,
        class_balanced=True,
        anti_harm_floor=0.0,
    )

    assert torch.isfinite(loss)
    assert torch.isfinite(anti_harm)
    assert stats["message_help_supervised_count"] == 2.0
    assert stats["message_help_class_count"] == 2.0
    assert stats["prompt_class_anti_harm_loss"] >= 0.0
    assert set(stats["delta_ce_by_class_train_pool"].keys()) == {"0", "1"}
    loss.backward()
    assert logits_on.grad is not None
    assert logits_on.grad[~train_mask].abs().sum().item() == 0.0


def test_hard_receive_gate_limits_prompt_receivers() -> None:
    z, h_pre, edge_index, _, train_mask = _toy_inputs()
    module = PromptGraphModuleP1(
        4,
        3,
        _config(
            rho=1.0,
            use_benefit_gate=True,
            use_hard_receive_gate=True,
            hard_receive_ratio=0.5,
            use_hard_acceptance=False,
            use_receiver_only_prompt=True,
        ),
    )

    out = module(z=z, h_pre=h_pre, edge_index=edge_index, train_mask=train_mask)
    aux = out["aux"]

    assert aux["use_hard_receive_gate"] == 1
    assert aux["hard_receive_selected_ratio"].item() == 0.5
    assert torch.count_nonzero(aux["hard_receive_mask"]).item() == 3
    assert out["adapted_edge_type"].eq(1).sum().item() == 0
