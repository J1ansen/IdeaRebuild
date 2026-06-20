from __future__ import annotations

import torch

from experiments.run_gp2f_prompt_graph import _config_for_variant
from models.class_conditioned_pattern_prompt_router import (
    ClassConditionedPatternPromptRouter,
    prompt_router_pattern_balance_loss,
)
from models.hetero_prompt_adapter import prompt_adapter_message_help_loss
from models.p21_adaptive_filter import P21LiteAdaptiveFilter
from models.utility_supervised_pattern_prompt_router import UtilitySupervisedPatternPromptRouter


def _toy_graph() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    z = torch.tensor(
        [
            [1.0, 0.0, 0.0],
            [0.0, 2.0, 0.0],
            [0.0, 0.0, 3.0],
            [1.0, 1.0, 1.0],
            [2.0, 0.0, 1.0],
        ]
    )
    edge_index = torch.tensor(
        [
            [0, 1, 2, 0, 3, 4],
            [1, 2, 0, 3, 2, 0],
        ],
        dtype=torch.long,
    )
    h_adp = torch.randn(5, 6)
    return z, edge_index, h_adp


def _router(**overrides) -> ClassConditionedPatternPromptRouter:
    config = {"num_classes": 3, "num_patterns": 6, "dropout": 0.0}
    config.update(overrides)
    return ClassConditionedPatternPromptRouter(3, 6, config)


def _utility_router(**overrides) -> UtilitySupervisedPatternPromptRouter:
    config = {"num_classes": 3, "num_patterns": 6, "dropout": 0.0}
    config.update(overrides)
    return UtilitySupervisedPatternPromptRouter(3, 6, config)


def _p21_filter(**overrides) -> P21LiteAdaptiveFilter:
    config = {"dropout": 0.0}
    config.update(overrides)
    return P21LiteAdaptiveFilter(3, 6, config)


def test_consumes_base_logits_flag() -> None:
    assert ClassConditionedPatternPromptRouter.consumes_base_logits is True
    assert UtilitySupervisedPatternPromptRouter.consumes_base_logits is True
    assert P21LiteAdaptiveFilter.consumes_base_logits is True


def test_forward_shapes_and_routing_normalisation() -> None:
    z, edge_index, h_adp = _router_inputs()
    router = _router()
    base_logits = torch.randn(5, 3)
    out = router(z=z, edge_index=edge_index, h_adp=h_adp, base_logits=base_logits)

    assert out["h_adp"].shape == h_adp.shape
    assert out["delta"].shape == h_adp.shape
    assert out["pattern_weights"].shape == (5, 6)
    assert out["easy_prob"].shape == (5,)
    assert out["gate"].shape == (5,)
    assert out["soft_class_prob"].shape == (5, 3)
    # PatternRouter outputs a simplex per node.
    assert torch.allclose(out["pattern_weights"].sum(dim=-1), torch.ones(5), atol=1e-5)
    assert torch.all(out["easy_prob"] >= 0.0) and torch.all(out["easy_prob"] <= 1.0)
    assert torch.all(out["gate"] >= 0.0) and torch.all(out["gate"] <= 1.0)


def _router_inputs() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return _toy_graph()


def test_zero_init_message_starts_as_no_prompt_update() -> None:
    z, edge_index, h_adp = _toy_graph()
    router = _router(zero_init_message=True, gate_init=0.10)
    out = router(z=z, edge_index=edge_index, h_adp=h_adp, base_logits=torch.randn(5, 3))

    assert torch.allclose(out["h_adp"], h_adp, atol=1e-6)
    assert out["prompt_update_norm"].item() == 0.0
    assert 0.09 <= out["prompt_gate_mean"].item() <= 0.11


def test_bounded_message_respects_max_norm_and_update_mask() -> None:
    z, edge_index, h_adp = _toy_graph()
    update_mask = torch.tensor([True, False, True, False, True])
    router = _router(zero_init_message=False, gate_init=0.95, max_update_norm=0.01)
    out = router(
        z=z,
        edge_index=edge_index,
        h_adp=h_adp,
        base_logits=torch.randn(5, 3),
        update_mask=update_mask,
    )
    # delta (constrained message) norm is clipped to max_update_norm.
    assert float(out["delta"].norm(dim=-1).max().item()) <= 0.010001
    masked = out["update"][~update_mask]
    assert torch.allclose(masked, torch.zeros_like(masked))
    assert abs(out["prompt_update_mask_ratio"].item() - 0.6) < 1e-5


def test_message_scale_zero_degenerates_to_no_prompt() -> None:
    z, edge_index, h_adp = _toy_graph()
    router = _router(zero_init_message=False, gate_init=0.8)
    out = router(z=z, edge_index=edge_index, h_adp=h_adp, base_logits=torch.randn(5, 3), message_scale=0.0)

    assert torch.allclose(out["h_adp"], h_adp, atol=1e-6)
    assert out["prompt_update_norm"].item() == 0.0


def test_support_nodes_force_true_class_routing() -> None:
    z, edge_index, h_adp = _toy_graph()
    router = _router()
    labels = torch.tensor([0, 1, 2, 1, 0])
    support_mask = torch.tensor([True, True, False, False, False])
    out = router(
        z=z,
        edge_index=edge_index,
        h_adp=h_adp,
        base_logits=torch.randn(5, 3),
        support_mask=support_mask,
        labels=labels,
    )
    p = out["soft_class_prob"]
    # Support nodes use one-hot of their true class.
    assert torch.allclose(p[0], torch.tensor([1.0, 0.0, 0.0]), atol=1e-6)
    assert torch.allclose(p[1], torch.tensor([0.0, 1.0, 0.0]), atol=1e-6)
    # Non-support rows remain a soft distribution (sum to one, not one-hot in general).
    assert torch.allclose(p[2:].sum(dim=-1), torch.ones(3), atol=1e-5)


def test_diagnostic_keys_present() -> None:
    z, edge_index, h_adp = _toy_graph()
    router = _router()
    out = router(z=z, edge_index=edge_index, h_adp=h_adp, base_logits=torch.randn(5, 3))
    for key in (
        "easy_prob",
        "pattern_weights",
        "soft_class_prob",
        "class_prompt_usage",
        "hetero_prompt_usage",
        "pattern_usage_mean",
        "pattern_usage_entropy",
        "prompt_gate_mean",
        "prompt_delta_norm",
        "raw_gate",
    ):
        assert key in out


def test_backward_has_gradients() -> None:
    z, edge_index, h_adp = _toy_graph()
    router = _router(zero_init_message=False, gate_init=0.5)
    out = router(z=z, edge_index=edge_index, h_adp=h_adp, base_logits=torch.randn(5, 3))
    loss = out["h_adp"].pow(2).mean() + prompt_router_pattern_balance_loss(out)
    loss.backward()
    grads = [p.grad for p in router.parameters() if p.requires_grad]
    assert any(g is not None and torch.isfinite(g).all() and g.abs().sum().item() > 0 for g in grads)


def test_pattern_balance_loss_is_finite_and_nonnegative() -> None:
    z, edge_index, h_adp = _toy_graph()
    router = _router()
    out = router(z=z, edge_index=edge_index, h_adp=h_adp, base_logits=torch.randn(5, 3))
    loss = prompt_router_pattern_balance_loss(out)
    assert torch.isfinite(loss)
    assert loss.item() >= 0.0


def test_pattern_balance_floor_penalises_collapse_not_uniform() -> None:
    # Diverse (near-uniform) usage -> no penalty (does NOT reward uniform).
    diverse = {"pattern_weights": torch.full((8, 6), 1.0 / 6.0)}
    assert prompt_router_pattern_balance_loss(diverse, entropy_floor=0.5).item() == 0.0
    # Collapsed usage (all mass on one pattern) -> positive penalty.
    collapsed = torch.zeros(8, 6)
    collapsed[:, 2] = 1.0
    loss = prompt_router_pattern_balance_loss({"pattern_weights": collapsed}, entropy_floor=0.5)
    assert loss.item() > 0.0
    # entropy_floor=0 disables the floor entirely.
    assert prompt_router_pattern_balance_loss({"pattern_weights": collapsed}, entropy_floor=0.0).item() == 0.0


def test_nonzero_init_message_breaks_routing_symmetry() -> None:
    z, edge_index, h_adp = _toy_graph()
    router = _router(zero_init_message=False, message_init_scale=0.1, gate_init=0.05)
    out = router(z=z, edge_index=edge_index, h_adp=h_adp, base_logits=torch.randn(5, 3))
    # Message is non-zero at init, so the prompt perturbs h_adp.
    assert out["prompt_delta_norm"].item() > 0.0
    # PatternRouter receives gradient (so it can specialise).
    loss = out["h_adp"].pow(2).mean()
    loss.backward()
    pattern_grads = [p.grad for p in router.pattern_router.parameters() if p.grad is not None]
    assert any(g.abs().sum().item() > 0 for g in pattern_grads)


def test_pattern_router_can_start_with_low_reject_prior() -> None:
    z, edge_index, h_adp = _toy_graph()
    router = _router(pattern_reject_init_prob=0.05)
    out = router(z=z, edge_index=edge_index, h_adp=h_adp, base_logits=torch.randn(5, 3))

    pattern_mean = out["pattern_weights"].mean(dim=0)
    assert abs(float(pattern_mean[0].item()) - 0.05) < 1e-5
    assert torch.allclose(
        pattern_mean[1:],
        torch.full_like(pattern_mean[1:], 0.95 / 5.0),
        atol=1e-5,
    )


def test_pattern_supervision_trains_router_toward_helpful_pattern() -> None:
    import torch.nn as nn

    from experiments.run_gp2f_prompt_graph import prompt_router_pattern_supervision_loss

    torch.manual_seed(0)
    z, edge_index, h_adp = _toy_graph()
    router = _router(num_patterns=4, message_init_scale=0.2)
    labels = torch.tensor([0, 1, 2, 0, 1])
    mask = torch.ones(5, dtype=torch.bool)
    base_logits = torch.randn(5, 3)
    out = router(z=z, edge_index=edge_index, h_adp=h_adp, base_logits=base_logits, labels=labels, support_mask=mask)

    class _Model:
        def __init__(self) -> None:
            self.alpha = torch.tensor(0.5)
            self.classifier = nn.Linear(6, 3)

    model = _Model()
    h_pre = torch.randn(5, 6)
    no_prompt_logits = model.classifier(0.5 * h_pre + 0.5 * h_adp)
    loss, stats = prompt_router_pattern_supervision_loss(
        adapter_out=out,
        model=model,
        h_pre=h_pre,
        h_adp_base=h_adp,
        no_prompt_logits=no_prompt_logits,
        labels=labels,
        mask=mask,
        temperature=0.05,
        probe_norm=0.08,
    )
    assert torch.isfinite(loss)
    assert 0.0 <= stats["prompt_router_pattern_routing_agreement"] <= 1.0
    assert stats["prompt_router_pattern_supervision_count"] == 5.0
    # The supervision loss must flow gradient into the PatternRouter.
    loss.backward()
    grads = [p.grad for p in router.pattern_router.parameters() if p.grad is not None]
    assert any(g.abs().sum().item() > 0 for g in grads)


def test_pattern_utility_trains_pattern_experts() -> None:
    import torch.nn as nn

    from experiments.run_gp2f_prompt_graph import prompt_router_pattern_utility_loss

    torch.manual_seed(1)
    z, edge_index, h_adp = _toy_graph()
    router = _router(num_patterns=4, message_init_scale=0.2)
    labels = torch.tensor([0, 1, 2, 0, 1])
    mask = torch.ones(5, dtype=torch.bool)
    base_logits = torch.randn(5, 3)
    out = router(z=z, edge_index=edge_index, h_adp=h_adp, base_logits=base_logits, labels=labels, support_mask=mask)

    class _Model:
        def __init__(self) -> None:
            self.alpha = torch.tensor(0.5)
            self.classifier = nn.Linear(6, 3)

    model = _Model()
    h_pre = torch.randn(5, 6)
    no_prompt_logits = model.classifier(0.5 * h_pre + 0.5 * h_adp)
    loss, stats = prompt_router_pattern_utility_loss(
        adapter_out=out,
        model=model,
        h_pre=h_pre,
        h_adp_base=h_adp,
        no_prompt_logits=no_prompt_logits,
        labels=labels,
        mask=mask,
        temperature=0.5,
        probe_norm=0.08,
        margin=0.001,
        anti_harm_weight=0.5,
        min_teacher_delta=1e-5,
        helpful_fraction=0.3,
        unhelpful_node_weight=0.1,
    )

    assert torch.isfinite(loss)
    assert stats["prompt_router_pattern_utility_count"] == 15.0
    assert 0.0 <= stats["prompt_router_pattern_utility_positive_ratio"] <= 1.0
    assert 0.0 <= stats["prompt_router_pattern_utility_helpful_node_ratio"] <= 1.0
    loss.backward()
    expert_grads = [p.grad for p in router.pattern_experts.parameters() if p.grad is not None]
    assert any(g.abs().sum().item() > 0 for g in expert_grads)


def test_class_pattern_reliability_trains_router_by_class() -> None:
    import torch.nn as nn

    from experiments.run_gp2f_prompt_graph import (
        prompt_router_class_pattern_reliability_loss,
        prompt_router_expert_utility_supervision_loss,
    )

    torch.manual_seed(2)
    z, edge_index, h_adp = _toy_graph()
    router = _router(num_patterns=4, message_init_scale=0.2)
    labels = torch.tensor([0, 1, 2, 0, 1])
    mask = torch.ones(5, dtype=torch.bool)
    base_logits = torch.randn(5, 3)
    out = router(z=z, edge_index=edge_index, h_adp=h_adp, base_logits=base_logits, labels=labels, support_mask=mask)

    class _Model:
        def __init__(self) -> None:
            self.alpha = torch.tensor(0.5)
            self.classifier = nn.Linear(6, 3)

    model = _Model()
    h_pre = torch.randn(5, 6)
    no_prompt_logits = model.classifier(0.5 * h_pre + 0.5 * h_adp)
    loss, stats = prompt_router_class_pattern_reliability_loss(
        adapter_out=out,
        model=model,
        h_pre=h_pre,
        h_adp_base=h_adp,
        no_prompt_logits=no_prompt_logits,
        labels=labels,
        mask=mask,
        temperature=0.5,
        probe_norm=0.08,
        min_class_count=2,
    )

    assert torch.isfinite(loss)
    assert stats["prompt_router_class_pattern_reliability_count"] == 4.0
    assert 0.0 <= stats["prompt_router_class_pattern_reliable_pair_ratio"] <= 1.0
    assert 0.0 <= stats["prompt_router_class_pattern_nonreject_target_mass"] <= 1.0
    loss.backward()
    router_grads = [p.grad for p in router.pattern_router.parameters() if p.grad is not None]
    assert any(g.abs().sum().item() > 0 for g in router_grads)

    pool_loss, pool_stats = prompt_router_expert_utility_supervision_loss(
        adapter_out=out,
        model=model,
        h_pre=h_pre,
        h_adp_base=h_adp,
        no_prompt_logits=no_prompt_logits,
        labels=labels,
        mask=torch.tensor([True, False, True, False, True]),
        num_classes=3,
        prefix="prompt_router_expert_test_pool",
        temperature=0.5,
        margin=1e-5,
        target_mode="margin_softmax",
        gain_temperature=0.02,
        gate_weight=0.25,
        gate_target_mode="margin_sigmoid",
        gate_target_temperature=0.02,
    )
    assert torch.isfinite(pool_loss)
    assert pool_stats["prompt_router_expert_test_pool_count"] == 3.0
    assert "prompt_router_expert_test_pool_learned_weighted_delta_ce" in pool_stats
    assert "prompt_router_expert_test_pool_gate_target_mean" in pool_stats
    assert "prompt_router_expert_test_pool_oracle_best_expert_acc_lift_vs_no_prompt" in pool_stats


def test_expert_utility_supervision_reports_oracle_and_trains_router() -> None:
    import torch.nn as nn

    from experiments.run_gp2f_prompt_graph import prompt_router_expert_utility_supervision_loss

    torch.manual_seed(3)
    z, edge_index, h_adp = _toy_graph()
    router = _utility_router(num_patterns=4, message_init_scale=0.2)
    labels = torch.tensor([0, 1, 2, 0, 1])
    mask = torch.ones(5, dtype=torch.bool)
    base_logits = torch.randn(5, 3)
    out = router(z=z, edge_index=edge_index, h_adp=h_adp, base_logits=base_logits, labels=labels, support_mask=mask)

    class _Model:
        def __init__(self) -> None:
            self.alpha = torch.tensor(0.5)
            self.classifier = nn.Linear(6, 3)

    model = _Model()
    h_pre = torch.randn(5, 6)
    no_prompt_logits = model.classifier(0.5 * h_pre + 0.5 * h_adp)
    loss, stats = prompt_router_expert_utility_supervision_loss(
        adapter_out=out,
        model=model,
        h_pre=h_pre,
        h_adp_base=h_adp,
        no_prompt_logits=no_prompt_logits,
        labels=labels,
        mask=mask,
        num_classes=3,
        temperature=0.5,
        margin=1e-5,
        target_mode="soft",
        gate_weight=0.25,
    )

    assert torch.isfinite(loss)
    assert stats["prompt_router_expert_count"] == 5.0
    assert "prompt_router_expert_oracle_best_expert_gain" in stats
    assert "prompt_router_expert_oracle_best_expert_macro_f1_lift_vs_no_prompt" in stats
    assert "prompt_router_expert_gate_supervision_loss" in stats
    assert 0.0 <= stats["prompt_router_expert_gate_target_mean"] <= 1.0
    assert 0.0 <= stats["prompt_router_expert_gate_target_std"] <= 0.5
    assert 0.0 <= stats["prompt_router_expert_gate_mean"] <= 1.0
    assert 0.0 <= stats["prompt_router_expert_no_correction_ratio"] <= 1.0
    assert 0.0 <= stats["prompt_router_expert_router_accuracy_to_best_expert"] <= 1.0
    loss.backward()
    router_grads = [p.grad for p in router.pattern_router.parameters() if p.grad is not None]
    gate_grads = [p.grad for p in router.receive_gate.parameters() if p.grad is not None]
    assert any(g.abs().sum().item() > 0 for g in router_grads)
    assert any(g.abs().sum().item() > 0 for g in gate_grads)


def test_prompt_adapter_message_help_can_penalise_pool_harm() -> None:
    logits_no_prompt = torch.tensor([[0.0, 2.0], [2.0, 0.0], [0.0, 2.0]])
    logits_prompt = torch.tensor([[2.0, 0.0], [2.0, 0.0], [0.0, 2.0]], requires_grad=True)
    labels = torch.tensor([1, 0, 1])
    mask = torch.tensor([True, True, False])

    loss, stats = prompt_adapter_message_help_loss(
        logits_prompt=logits_prompt,
        logits_no_prompt=logits_no_prompt,
        labels=labels,
        mask=mask,
        margin=0.0,
        anti_harm_weight=1.0,
        anti_harm_margin=0.0,
        class_balanced=False,
    )

    assert torch.isfinite(loss)
    assert stats["prompt_adapter_message_help_count"] == 2.0
    assert stats["prompt_adapter_message_help_anti_harm_loss"] > 0.0
    loss.backward()
    assert logits_prompt.grad is not None
    assert logits_prompt.grad[0].abs().sum().item() > 0.0
    assert logits_prompt.grad[2].abs().sum().item() == 0.0


def test_p21_lite_adaptive_filter_shapes_prior_and_mask() -> None:
    z, edge_index, h_adp = _toy_graph()
    filt = _p21_filter(beta_init=0.05, beta_max=0.20, channel_prior=[0.45, 0.15, 0.25, 0.15])
    update_mask = torch.tensor([True, False, True, False, True])
    out = filt(
        z=z,
        edge_index=edge_index,
        h_adp=h_adp,
        update_mask=update_mask,
        base_logits=torch.randn(5, 3),
        h_pre=torch.randn(5, 6),
    )

    assert out["h_adp"].shape == h_adp.shape
    assert out["alpha"].shape == (5, 4)
    assert torch.allclose(out["alpha"].sum(dim=-1), torch.ones(5), atol=1e-5)
    assert torch.allclose(
        out["alpha_global"],
        torch.tensor([0.45, 0.15, 0.25, 0.15], dtype=out["alpha_global"].dtype),
        atol=1e-5,
    )
    assert 0.049 <= float(out["beta"].item()) <= 0.051
    assert out["gate"].shape == (5,)
    assert out["channel_deltas"].shape == (5, 4, 6)
    assert torch.allclose(out["update"][~update_mask], torch.zeros_like(out["update"][~update_mask]))
    assert float(out["delta"].norm(dim=-1).max().item()) <= 0.050001
    for key in (
        "p21_alpha_ego_mean",
        "p21_alpha_low_mean",
        "p21_alpha_two_mean",
        "p21_alpha_high_mean",
        "p21_channel_high_norm",
        "p21_gate_mean",
        "p21_channel_delta_low_norm",
        "p21_ego_low_discrepancy",
    ):
        assert key in out


def test_p21_lite_adaptive_filter_backpropagates_to_router_and_beta() -> None:
    z, edge_index, h_adp = _toy_graph()
    filt = _p21_filter(residual_scale=0.1)
    out = filt(z=z, edge_index=edge_index, h_adp=h_adp, base_logits=torch.randn(5, 3))
    loss = out["h_adp"].pow(2).mean() + out["alpha_entropy"]
    loss.backward()

    router_grads = [p.grad for p in filt.router.parameters() if p.grad is not None]
    gate_grads = [p.grad for p in filt.gate_router.parameters() if p.grad is not None]
    project_grads = [p.grad for p in filt.project.parameters() if p.grad is not None]
    assert any(g.abs().sum().item() > 0 for g in router_grads)
    assert any(g.abs().sum().item() > 0 for g in gate_grads)
    assert any(g.abs().sum().item() > 0 for g in project_grads)


def test_p21_channel_utility_supervision_trains_router() -> None:
    import torch.nn as nn

    from experiments.run_gp2f_prompt_graph import p21_channel_utility_supervision_loss

    torch.manual_seed(0)
    z, edge_index, h_adp = _toy_graph()
    filt = _p21_filter(residual_scale=0.3, beta_init=0.1, beta_max=0.3, gate_max=0.3)
    h_pre = torch.randn(5, 6)
    base_logits = torch.randn(5, 3)
    out = filt(z=z, edge_index=edge_index, h_adp=h_adp, base_logits=base_logits, h_pre=h_pre)

    class _Model:
        def __init__(self) -> None:
            self.alpha = torch.tensor(0.5)
            self.classifier = nn.Linear(6, 3)

    model = _Model()
    labels = torch.tensor([0, 1, 2, 0, 1])
    mask = torch.ones(5, dtype=torch.bool)
    no_prompt_logits = model.classifier(0.5 * h_pre + 0.5 * h_adp)
    loss, stats = p21_channel_utility_supervision_loss(
        adapter_out=out,
        model=model,
        h_pre=h_pre,
        h_adp_base=h_adp,
        logits_no_prompt=no_prompt_logits,
        labels=labels,
        mask=mask,
        temperature=0.05,
    )

    assert torch.isfinite(loss)
    assert stats["p21_channel_utility_count"] == 5.0
    assert "p21_channel_utility_mean_oracle_delta_ce" in stats
    loss.backward()
    router_grads = [p.grad for p in filt.router.parameters() if p.grad is not None]
    assert any(g.abs().sum().item() > 0 for g in router_grads)


def test_deployment_utility_loss_trains_actual_prompt_logits() -> None:
    from experiments.run_gp2f_prompt_graph import prompt_router_deployment_utility_loss

    logits_no_prompt = torch.tensor([[0.0, 2.0], [2.0, 0.0], [0.0, 2.0]])
    logits_prompt = torch.tensor([[2.0, 0.0], [2.0, 0.0], [0.0, 2.0]], requires_grad=True)
    labels = torch.tensor([1, 0, 1])
    mask = torch.tensor([True, True, False])

    loss, stats = prompt_router_deployment_utility_loss(
        logits_prompt=logits_prompt,
        logits_no_prompt=logits_no_prompt,
        labels=labels,
        mask=mask,
        margin=0.001,
        anti_harm_weight=1.0,
        anti_harm_margin=0.0,
        class_balanced=False,
    )

    assert torch.isfinite(loss)
    assert stats["prompt_router_deployment_count"] == 2.0
    assert stats["prompt_router_deployment_harmful_delta_ratio"] == 0.5
    assert stats["prompt_router_deployment_positive_delta_ratio"] == 0.0
    assert stats["prompt_router_deployment_anti_harm_loss"] > 0.0
    loss.backward()
    assert logits_prompt.grad is not None
    assert logits_prompt.grad[0].abs().sum().item() > 0.0
    assert logits_prompt.grad[2].abs().sum().item() == 0.0


def test_p20_variant_config_enables_router_and_disables_graph() -> None:
    base = {
        "experiment": {"prompt_variant": "p20_class_conditioned_pattern_prompt_router"},
        "prompt_adapter": {"enabled": True},
    }
    cfg = _config_for_variant(base, "p20_class_conditioned_pattern_prompt_router")
    assert cfg["prompt_graph"]["enabled"] is False
    assert cfg["prompt_aware"]["enabled"] is False
    assert cfg["prompt_adapter"]["enabled"] is True
    assert cfg["prompt_adapter"]["module_type"] == "class_conditioned_pattern_router"
    assert cfg["prompt_adapter"]["num_patterns"] == 6
    assert cfg["prompt_adapter"]["zero_init_message"] is False
    assert cfg["training"]["train_prompt_adapter"] is True
    # Light anti-collapse floor is on by default; it only fires when usage collapses.
    assert cfg["training"]["lambda_prompt_router_pattern_balance"] == 0.02
    assert cfg["training"]["prompt_router_pattern_balance_entropy_floor"] == 0.5
    assert cfg["training"]["prompt_adapter_message_help_anti_harm_weight"] == 0.5
    assert cfg["training"]["prompt_adapter_utility_gate_margin"] == 1e-5
    # Direct pattern-routing supervision is on by default (drives specialisation).
    assert cfg["training"]["lambda_prompt_router_pattern_supervision"] > 0.0
    assert cfg["training"]["lambda_prompt_router_pattern_utility"] > 0.0
    assert cfg["training"]["lambda_prompt_router_class_pattern_reliability"] > 0.0
    assert cfg["training"]["prompt_adapter_episode_count_per_epoch"] == 3


def test_p20_utility_variant_config_enables_expert_supervision() -> None:
    base = {
        "experiment": {"prompt_variant": "p20_utility_supervised_pattern_prompt_router"},
        "prompt_adapter": {"enabled": True},
    }
    cfg = _config_for_variant(base, "p20_utility_supervised_pattern_prompt_router")
    assert cfg["prompt_graph"]["enabled"] is False
    assert cfg["prompt_aware"]["enabled"] is False
    assert cfg["prompt_adapter"]["enabled"] is True
    assert cfg["prompt_adapter"]["module_type"] == "utility_supervised_pattern_router"
    assert cfg["training"]["freeze_base_model"] is True
    assert cfg["training"]["early_stop_metric"] == "train_loss"
    assert cfg["training"]["lambda_prompt_router_deployment_utility"] == 0.10
    assert cfg["training"]["prompt_router_deployment_utility_margin"] == 0.0005
    assert cfg["training"]["prompt_router_deployment_utility_anti_harm_weight"] == 1.0
    assert cfg["training"]["prompt_router_deployment_utility_gain_reward_weight"] == 0.25
    assert cfg["training"]["prompt_router_deployment_utility_gain_reward_cap"] == 0.02
    assert cfg["training"]["lambda_prompt_router_expert_utility_supervision"] > 0.0
    assert cfg["training"]["prompt_router_expert_utility_target"] == "margin_softmax"
    assert cfg["training"]["prompt_router_expert_utility_gain_temperature"] == 0.02
    assert cfg["training"]["lambda_prompt_router_pattern_balance"] == 0.0
    assert cfg["training"]["lambda_prompt_router_pattern_supervision"] == 0.0
    assert cfg["training"]["lambda_prompt_router_pattern_utility"] == 0.0
    assert cfg["training"]["lambda_prompt_router_class_pattern_reliability"] == 0.0
    assert cfg["prompt_adapter"]["prompt_router_expert_utility_temperature"] == 0.5
    assert cfg["prompt_adapter"]["prompt_router_expert_utility_margin"] == 0.0005
    assert cfg["prompt_adapter"]["prompt_router_expert_utility_target"] == "margin_softmax"
    assert cfg["prompt_adapter"]["prompt_router_expert_utility_gain_temperature"] == 0.02
    assert cfg["training"]["prompt_router_expert_utility_gate_weight"] == 0.25
    assert cfg["prompt_adapter"]["prompt_router_expert_utility_gate_weight"] == 0.25
    assert cfg["training"]["prompt_router_expert_utility_gate_target"] == "margin_sigmoid"
    assert cfg["prompt_adapter"]["prompt_router_expert_utility_gate_target"] == "margin_sigmoid"
    assert cfg["training"]["prompt_router_expert_utility_gate_temperature"] == 0.02
    assert cfg["prompt_adapter"]["prompt_router_expert_utility_gate_temperature"] == 0.02


def test_p21_variant_config_enables_lite_adaptive_filter() -> None:
    base = {
        "experiment": {"prompt_variant": "p21_lite_adaptive_filter"},
        "prompt_adapter": {"enabled": True},
    }
    cfg = _config_for_variant(base, "p21_lite_adaptive_filter")
    assert cfg["prompt_graph"]["enabled"] is False
    assert cfg["prompt_aware"]["enabled"] is False
    assert cfg["prompt_adapter"]["enabled"] is True
    assert cfg["prompt_adapter"]["module_type"] == "p21_lite_adaptive_filter"
    assert cfg["training"]["freeze_base_model"] is True
    assert cfg["training"]["early_stop_metric"] == "train_loss"
    assert cfg["prompt_adapter"]["channel_prior"] == [0.45, 0.15, 0.25, 0.15]
    assert cfg["prompt_adapter"]["beta_init"] == 0.10
    assert cfg["prompt_adapter"]["use_node_wise_gate"] is True
    assert cfg["prompt_adapter"]["use_candidate_pool"] is False
    assert cfg["training"]["prompt_adapter_update_mask"] == "all"
    assert cfg["training"]["lambda_p21_channel_utility"] == 0.20
    assert cfg["prompt_adapter"]["beta_max"] == 0.30
    assert cfg["prompt_adapter"]["gate_max"] == 0.30
    assert cfg["prompt_adapter"]["max_update_norm"] == 0.08
    assert cfg["training"]["lambda_prompt_router_pattern_supervision"] == 0.0
    assert cfg["training"]["lambda_prompt_router_pattern_utility"] == 0.0
    assert cfg["training"]["lambda_prompt_router_deployment_utility"] == 0.10
    assert cfg["training"]["lambda_prompt_adapter_gate_budget"] == 0.0
    assert cfg["training"]["lambda_prompt_adapter_utility_gate"] == 0.0
    assert cfg["training"]["prompt_adapter_episode_count_per_epoch"] == 1


def test_p20_yaml_config_uses_conservative_gate_and_nonzero_message() -> None:
    from pathlib import Path

    from utils.io import read_yaml

    config_path = Path(__file__).resolve().parents[1] / "configs" / "gp2f_prompt_p20_class_conditioned_pattern_prompt_router.yaml"
    raw = read_yaml(config_path)
    cfg = _config_for_variant(raw, "p20_class_conditioned_pattern_prompt_router")
    assert cfg["prompt_adapter"]["module_type"] == "class_conditioned_pattern_router"
    assert cfg["prompt_adapter"]["zero_init_message"] is False
    assert cfg["prompt_adapter"]["message_init_scale"] == 0.1
    assert cfg["prompt_adapter"]["easy_init"] == 0.30
    assert cfg["prompt_adapter"]["pattern_reject_init_prob"] == 0.05
    assert cfg["prompt_adapter"]["gate_init"] == 0.05
    assert cfg["prompt_adapter"]["max_update_norm"] == 0.04
    assert cfg["prompt_adapter"]["gate_budget"] == 0.12
    assert cfg["prompt_adapter"]["use_candidate_pool"] is True
    assert cfg["prompt_adapter"]["candidate_pool_strategy"] == "utility_structural"
    assert cfg["prompt_adapter"]["candidate_pool_ratio"] == 0.30
    assert cfg["training"]["log_every"] == 5
    assert cfg["training"]["prompt_adapter_update_mask"] == "candidate_pool"
    assert cfg["training"]["lambda_prompt_adapter_update_norm"] == 0.02
    assert cfg["training"]["lambda_prompt_adapter_gate_budget"] == 0.25
    assert cfg["training"]["lambda_prompt_adapter_utility_gate"] == 0.10
    assert cfg["training"]["prompt_adapter_message_help_anti_harm_weight"] == 0.5
    assert cfg["training"]["prompt_adapter_utility_gate_margin"] == 0.00001
    assert cfg["training"]["lambda_prompt_router_pattern_utility"] == 0.05
    assert cfg["training"]["prompt_router_pattern_utility_margin"] == 0.001
    assert cfg["training"]["prompt_router_pattern_utility_min_teacher_delta"] == 0.00001
    assert cfg["training"]["prompt_router_pattern_utility_helpful_fraction"] == 0.30
    assert cfg["training"]["prompt_router_pattern_utility_unhelpful_node_weight"] == 0.1
    assert cfg["training"]["lambda_prompt_router_class_pattern_reliability"] == 0.03
    assert cfg["training"]["prompt_router_class_pattern_reliability_positive_margin"] == 0.00001


def test_p20_utility_yaml_config_uses_prompt_only_expert_supervision() -> None:
    from pathlib import Path

    from utils.io import read_yaml

    config_path = (
        Path(__file__).resolve().parents[1]
        / "configs"
        / "gp2f_prompt_p20_utility_supervised_pattern_prompt_router.yaml"
    )
    raw = read_yaml(config_path)
    cfg = _config_for_variant(raw, "p20_utility_supervised_pattern_prompt_router")
    assert cfg["prompt_adapter"]["module_type"] == "utility_supervised_pattern_router"
    assert cfg["prompt_adapter"]["num_patterns"] == 6
    assert cfg["training"]["freeze_base_model"] is True
    assert cfg["training"]["early_stop_metric"] == "train_loss"
    assert cfg["training"]["prompt_adapter_update_mask"] == "candidate_pool"
    assert cfg["training"]["lambda_prompt_router_deployment_utility"] == 0.10
    assert cfg["training"]["prompt_router_deployment_utility_margin"] == 0.0005
    assert cfg["training"]["prompt_router_deployment_utility_anti_harm_weight"] == 1.0
    assert cfg["training"]["prompt_router_deployment_utility_gain_reward_weight"] == 0.25
    assert cfg["training"]["prompt_router_deployment_utility_gain_reward_cap"] == 0.02
    assert cfg["training"]["lambda_prompt_router_expert_utility_supervision"] == 0.20
    assert cfg["training"]["lambda_prompt_router_pattern_supervision"] == 0.0
    assert cfg["training"]["lambda_prompt_router_pattern_utility"] == 0.0
    assert cfg["training"]["prompt_router_expert_utility_margin"] == 0.0005
    assert cfg["prompt_adapter"]["prompt_router_expert_utility_margin"] == 0.0005
    assert cfg["training"]["prompt_router_expert_utility_target"] == "margin_softmax"
    assert cfg["prompt_adapter"]["prompt_router_expert_utility_target"] == "margin_softmax"
    assert cfg["training"]["prompt_router_expert_utility_gain_temperature"] == 0.02
    assert cfg["prompt_adapter"]["prompt_router_expert_utility_gain_temperature"] == 0.02
    assert cfg["training"]["prompt_router_expert_utility_gate_weight"] == 0.25
    assert cfg["prompt_adapter"]["prompt_router_expert_utility_gate_weight"] == 0.25
    assert cfg["training"]["prompt_router_expert_utility_gate_target"] == "margin_sigmoid"
    assert cfg["prompt_adapter"]["prompt_router_expert_utility_gate_target"] == "margin_sigmoid"
