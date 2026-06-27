import torch

from experiments.run_gp2f_prompt_graph import _config_for_variant, _p23_node_gate_utility_loss, _set_module_trainable
from models.discrete_feature_prompt import GraphiteStylePromptGraphAdapter
from models.p23_prompt_receiver import P23HubAwarePromptReceiver, P23V01PromptModule
from models.p23_static_prompt_graph import P23StaticPromptGraphBuilder


def _toy_inputs():
    x_raw = torch.tensor(
        [
            [1.0, 0.0, 0.0, 1.0],
            [1.0, 0.0, 1.0, 0.0],
            [0.0, 1.0, 1.0, 0.0],
            [0.0, 1.0, 0.0, 1.0],
            [1.0, 1.0, 0.0, 0.0],
        ]
    )
    z = torch.tensor(
        [
            [0.1, 0.2, 0.3],
            [0.2, 0.1, 0.4],
            [0.8, 0.7, 0.1],
            [0.7, 0.8, 0.2],
            [0.5, 0.5, 0.5],
        ]
    )
    edge_index = torch.tensor(
        [
            [0, 1, 2, 3, 4, 1, 2, 3],
            [1, 2, 3, 4, 0, 0, 1, 2],
        ],
        dtype=torch.long,
    )
    train_mask = torch.tensor([True, False, True, False, False])
    logits = torch.tensor(
        [
            [2.0, 0.1],
            [0.7, 0.6],
            [0.2, 1.5],
            [0.8, 0.7],
            [0.5, 0.4],
        ]
    )
    h_pre = torch.randn(5, 6)
    h_adp = h_pre + 0.05 * torch.randn(5, 6)
    return x_raw, z, edge_index, train_mask, logits, h_pre, h_adp


def _config():
    return {
        "feature_tokenizer": {"mode": "binary_nonzero", "binary_threshold": 0.0},
        "pool": {"adaptive_ratio": False, "rho": 0.4, "force_train_nodes": True},
        "feature_filter": {"min_df_pool": 1, "max_df_pool_ratio": 1.0, "max_df_global_ratio": 1.0},
        "prompt_graph": {"init_scope": "global_same_feature", "edge_type_prompt_to_node": 2},
        "receiver": {"max_update_norm": 0.03, "prompt_dropout": 0.0, "init_gate_bias": -1.0},
    }


def test_p23_defaults_keep_aligner_trainable_and_gate_utility_off():
    cfg = _config_for_variant({"experiment": {"prompt_variant": "p23_v0_1"}}, "p23_v0_1")
    assert cfg["training"]["freeze_input_aligner_for_p23"] is False
    assert cfg["prompt_graph"]["lambda_p23_node_gate_utility"] == 0.0


def test_p23_graphite_variant_uses_graphite_adapter_module_defaults():
    cfg = _config_for_variant({"experiment": {"prompt_variant": "p23_graphite_adapter"}}, "p23_graphite_adapter")
    assert cfg["prompt_graph"]["module_type"] == "graphite_style_prompt_graph"
    assert cfg["prompt_graph"]["feature_edge_weight"] == 1.0
    assert cfg["prompt_graph"]["learn_feature_edge_weight"] is True
    assert cfg["prompt_aware"]["enabled"] is False
    assert cfg["prompt_adapter"]["enabled"] is False


def test_train_nodes_are_forced_into_pool():
    x_raw, z, edge_index, train_mask, logits, h_pre, h_adp = _toy_inputs()
    state = P23StaticPromptGraphBuilder(_config()).build(
        x_raw=x_raw,
        z_snapshot=z,
        edge_index=edge_index,
        train_mask=train_mask,
        no_prompt_logits=logits,
        h_pre_snapshot=h_pre,
        h_adp0_snapshot=h_adp,
    )
    assert torch.all(state.pool_mask[train_mask])


def test_static_graph_reuses_same_topology_for_same_run_inputs():
    x_raw, z, edge_index, train_mask, logits, h_pre, h_adp = _toy_inputs()
    builder = P23StaticPromptGraphBuilder(_config())
    first = builder.build(
        x_raw=x_raw,
        z_snapshot=z,
        edge_index=edge_index,
        train_mask=train_mask,
        no_prompt_logits=logits,
        h_pre_snapshot=h_pre,
        h_adp0_snapshot=h_adp,
    )
    second = builder.build(
        x_raw=x_raw,
        z_snapshot=z,
        edge_index=edge_index,
        train_mask=train_mask,
        no_prompt_logits=logits,
        h_pre_snapshot=h_pre,
        h_adp0_snapshot=h_adp,
    )
    assert torch.equal(first.pool_mask, second.pool_mask)
    assert torch.equal(first.feature_ids, second.feature_ids)
    assert torch.equal(first.prompt_edge_index, second.prompt_edge_index)


def test_discrete_features_are_extracted_from_raw_x_not_z():
    x_raw, z, edge_index, train_mask, logits, h_pre, h_adp = _toy_inputs()
    z_without_raw_feature_signal = torch.zeros_like(z)
    state = P23StaticPromptGraphBuilder(_config()).build(
        x_raw=x_raw,
        z_snapshot=z_without_raw_feature_signal,
        edge_index=edge_index,
        train_mask=train_mask,
        no_prompt_logits=logits,
        h_pre_snapshot=h_pre,
        h_adp0_snapshot=h_adp,
    )
    assert state.feature_ids.numel() > 0
    assert int(state.feature_ids[0].item()) in {0, 1, 2, 3}


def test_prompt_edges_are_prompt_to_node_type_two():
    x_raw, z, edge_index, train_mask, logits, h_pre, h_adp = _toy_inputs()
    state = P23StaticPromptGraphBuilder(_config()).build(
        x_raw=x_raw,
        z_snapshot=z,
        edge_index=edge_index,
        train_mask=train_mask,
        no_prompt_logits=logits,
        h_pre_snapshot=h_pre,
        h_adp0_snapshot=h_adp,
    )
    assert state.prompt_edge_index.size(0) == 2
    assert torch.all(state.prompt_edge_type == 2)


def test_receiver_output_shape_and_bounded_update():
    x_raw, z, edge_index, train_mask, logits, h_pre, h_adp = _toy_inputs()
    state = P23StaticPromptGraphBuilder(_config()).build(
        x_raw=x_raw,
        z_snapshot=z,
        edge_index=edge_index,
        train_mask=train_mask,
        no_prompt_logits=logits,
        h_pre_snapshot=h_pre,
        h_adp0_snapshot=h_adp,
    )
    receiver = P23HubAwarePromptReceiver(source_dim=3, hidden_dim=6, config=_config())
    out = receiver(h_base=h_pre, state=state)
    assert out["h_adp"].shape == h_pre.shape
    assert out["prompt_update"].norm(dim=-1).max() <= 0.030001


def test_receiver_edge_scale_zero_degenerates_to_no_prompt_update():
    x_raw, z, edge_index, train_mask, logits, h_pre, h_adp = _toy_inputs()
    state = P23StaticPromptGraphBuilder(_config()).build(
        x_raw=x_raw,
        z_snapshot=z,
        edge_index=edge_index,
        train_mask=train_mask,
        no_prompt_logits=logits,
        h_pre_snapshot=h_pre,
        h_adp0_snapshot=h_adp,
    )
    receiver = P23HubAwarePromptReceiver(source_dim=3, hidden_dim=6, config=_config())
    out = receiver(h_base=h_pre, state=state, edge_scale=0.0)
    assert torch.allclose(out["h_adp"], h_pre)
    assert torch.allclose(out["prompt_update"], torch.zeros_like(h_pre))


def test_p23_learnable_prompt_delta_gets_gradients():
    x_raw, z, edge_index, train_mask, logits, h_pre, h_adp = _toy_inputs()
    cfg = _config()
    cfg["prompt_token"] = {"learn_delta": True}
    module = P23V01PromptModule(source_dim=3, hidden_dim=6, config=cfg)
    module.build_state(
        x_raw=x_raw,
        z_snapshot=z,
        edge_index=edge_index,
        train_mask=train_mask,
        no_prompt_logits=logits,
        h_pre_snapshot=h_pre,
        h_adp0_snapshot=h_adp,
    )
    assert module.prompt_delta is not None
    out = module.apply_receiver(h_pre)
    loss = out["h_adp"].sum()
    loss.backward()
    assert module.prompt_delta.grad is not None
    assert torch.isfinite(module.prompt_delta.grad).all()


def test_p23_node_to_prompt_channel_reports_context():
    x_raw, z, edge_index, train_mask, logits, h_pre, h_adp = _toy_inputs()
    cfg = _config()
    cfg["node_to_prompt"] = {"enabled": True, "scale": 0.5}
    module = P23V01PromptModule(source_dim=3, hidden_dim=6, config=cfg)
    module.build_state(
        x_raw=x_raw,
        z_snapshot=z,
        edge_index=edge_index,
        train_mask=train_mask,
        no_prompt_logits=logits,
        h_pre_snapshot=h_pre,
        h_adp0_snapshot=h_adp,
    )
    out = module.apply_receiver(h_pre)
    aux = module.receiver_aux(out)
    assert aux["p23_node_to_prompt_enabled"].item() == 1.0
    assert aux["p23_prompt_context_norm_mean"].item() >= 0.0
    assert "node_to_prompt_attention" in out


def test_hub_score_contributes_to_budget_loss():
    x_raw, z, edge_index, train_mask, logits, h_pre, h_adp = _toy_inputs()
    state = P23StaticPromptGraphBuilder(_config()).build(
        x_raw=x_raw,
        z_snapshot=z,
        edge_index=edge_index,
        train_mask=train_mask,
        no_prompt_logits=logits,
        h_pre_snapshot=h_pre,
        h_adp0_snapshot=h_adp,
    )
    receiver = P23HubAwarePromptReceiver(source_dim=3, hidden_dim=6, config=_config())
    out = receiver(h_base=h_pre, state=state)
    assert out["hub_budget_loss"].item() >= 0.0
    assert "hub_score" in state.feature_static_stats


def test_p23_module_uses_original_graph_and_receiver_only_prompt():
    x_raw, z, edge_index, train_mask, logits, h_pre, h_adp = _toy_inputs()
    module = P23V01PromptModule(source_dim=3, hidden_dim=6, config=_config())
    module.build_state(
        x_raw=x_raw,
        z_snapshot=z,
        edge_index=edge_index,
        train_mask=train_mask,
        no_prompt_logits=logits,
        h_pre_snapshot=h_pre,
        h_adp0_snapshot=h_adp,
    )
    prompt_out = module(z=z, h_pre=h_pre, edge_index=edge_index, train_mask=train_mask)
    assert prompt_out["adapted_x"].shape == z.shape
    assert torch.equal(prompt_out["adapted_edge_index"], edge_index)
    assert prompt_out["aux"]["edge_type_counts"][2] == prompt_out["prompt_edge_count"]


def test_prompt_graph_keeps_topk_feature_prompts_per_node():
    x_raw, z, edge_index, train_mask, logits, h_pre, h_adp = _toy_inputs()
    cfg = _config()
    cfg["prompt_graph"] = {
        "init_scope": "global_same_feature",
        "edge_type_prompt_to_node": 2,
        "topk_feature_prompt_per_node": 1,
    }
    state = P23StaticPromptGraphBuilder(cfg).build(
        x_raw=x_raw,
        z_snapshot=z,
        edge_index=edge_index,
        train_mask=train_mask,
        no_prompt_logits=logits,
        h_pre_snapshot=h_pre,
        h_adp0_snapshot=h_adp,
    )
    if state.prompt_edge_index.numel() > 0:
        dst = state.prompt_edge_index[1]
        assert int(torch.bincount(dst, minlength=x_raw.size(0)).max().item()) <= 1


def test_pool_rel_uses_pool_concentration_not_pool_frequency():
    x_raw, z, edge_index, train_mask, logits, h_pre, h_adp = _toy_inputs()
    state = P23StaticPromptGraphBuilder(_config()).build(
        x_raw=x_raw,
        z_snapshot=z,
        edge_index=edge_index,
        train_mask=train_mask,
        no_prompt_logits=logits,
        h_pre_snapshot=h_pre,
        h_adp0_snapshot=h_adp,
    )
    df_pool = state.feature_static_stats["df_pool"]
    df_global = state.feature_static_stats["df_global"]
    expected = df_pool / (df_global + 1.0)
    assert torch.allclose(state.feature_static_stats["pool_rel"], expected)
    assert "pool_freq" in state.feature_static_stats


def test_p23_node_gate_utility_loss_uses_train_mask_only():
    labels = torch.tensor([0, 0, 1, 1])
    train_mask = torch.tensor([True, True, False, False])
    node_gate = torch.tensor([0.2, 0.8, 0.2, 0.8], requires_grad=True)
    base_logits = torch.tensor([[2.0, 0.0], [2.0, 0.0], [2.0, 0.0], [0.0, 2.0]])
    prompt_logits = torch.tensor([[2.5, 0.0], [0.0, 2.0], [0.0, 2.5], [1.0, 0.0]], requires_grad=True)
    model_out = {
        "logits": prompt_logits,
        "p23_no_receiver_logits": base_logits,
        "p23_receiver": {"node_receive_gate": node_gate},
    }
    prompt_out = {"aux": {}, "pool_mask": torch.ones(4, dtype=torch.bool)}
    loss, stats = _p23_node_gate_utility_loss(
        model_out=model_out,
        prompt_out=prompt_out,
        labels=labels,
        train_mask=train_mask,
        positive_margin=0.001,
        negative_margin=-0.001,
        class_balanced=False,
    )
    assert loss.requires_grad
    assert stats["p23_node_gate_utility_count"] == 2.0
    assert stats["p23_node_gate_positive_count"] == 1.0
    assert stats["p23_node_gate_negative_count"] == 1.0


def test_p23_node_gate_utility_ignores_neutral_delta_ce():
    labels = torch.tensor([0, 1])
    train_mask = torch.tensor([True, True])
    node_gate = torch.tensor([0.5, 0.5], requires_grad=True)
    logits = torch.tensor([[2.0, 0.0], [0.0, 2.0]], requires_grad=True)
    model_out = {
        "logits": logits,
        "p23_no_receiver_logits": logits.detach().clone(),
        "p23_receiver": {"node_receive_gate": node_gate},
    }
    prompt_out = {"aux": {}, "pool_mask": torch.ones(2, dtype=torch.bool)}
    loss, stats = _p23_node_gate_utility_loss(
        model_out=model_out,
        prompt_out=prompt_out,
        labels=labels,
        train_mask=train_mask,
        positive_margin=0.001,
        negative_margin=-0.001,
        class_balanced=True,
    )
    assert float(loss.item()) == 0.0
    assert stats["p23_node_gate_utility_count"] == 0.0


def test_freezing_input_aligner_does_not_freeze_p23_receiver():
    module = P23V01PromptModule(source_dim=3, hidden_dim=6, config=_config())
    aligner = torch.nn.Linear(3, 3)
    _set_module_trainable(aligner, False)
    _set_module_trainable(module, True)
    assert not any(parameter.requires_grad for parameter in aligner.parameters())
    assert any(parameter.requires_grad for parameter in module.receiver.parameters())


def test_graphite_style_adapter_expands_adapted_graph_with_raw_feature_nodes():
    x_raw, z, edge_index, train_mask, logits, h_pre, h_adp = _toy_inputs()
    cfg = {
        "feature_tokenizer": {"mode": "binary_nonzero", "binary_threshold": 0.0, "binary_topk": 0},
        "feature_filter": {"min_df_global": 1, "max_df_global_ratio": 1.0},
        "feature_edge_weight": 1.0,
        "learn_feature_edge_weight": True,
    }
    module = GraphiteStylePromptGraphAdapter(source_dim=3, hidden_dim=6, config=cfg)
    out = module(
        z=z,
        x_raw=x_raw,
        h_pre=h_pre,
        edge_index=edge_index,
        train_mask=train_mask,
        no_prompt_logits=logits,
        h_adp_no_prompt=h_adp,
    )
    assert out["adapted_x"].size(0) > z.size(0)
    assert out["adapted_edge_index"].size(1) > edge_index.size(1)
    assert torch.all(out["pool_mask"])
    assert out["aux"]["p23_graphite_enabled"].item() == 1.0
    assert out["aux"]["edge_type_counts"][1] == out["prompt_edge_count"]
    feature_edge_weight = out["adapted_edge_weight"][edge_index.size(1) :]
    assert feature_edge_weight.requires_grad


def test_graphite_style_adapter_static_cache_reuses_topology():
    x_raw, z, edge_index, train_mask, logits, h_pre, h_adp = _toy_inputs()
    cfg = {
        "feature_tokenizer": {"mode": "binary_nonzero", "binary_threshold": 0.0},
        "feature_filter": {"min_df_global": 1, "max_df_global_ratio": 1.0},
        "static_graph": True,
    }
    module = GraphiteStylePromptGraphAdapter(source_dim=3, hidden_dim=6, config=cfg)
    first = module(
        z=z,
        x_raw=x_raw,
        h_pre=h_pre,
        edge_index=edge_index,
        train_mask=train_mask,
        no_prompt_logits=logits,
        h_adp_no_prompt=h_adp,
    )
    second = module(
        z=z + 1.0,
        x_raw=x_raw,
        h_pre=h_pre,
        edge_index=edge_index,
        train_mask=train_mask,
        no_prompt_logits=logits,
        h_adp_no_prompt=h_adp,
    )
    assert torch.equal(first["adapted_edge_index"], second["adapted_edge_index"])
    assert second["aux"]["p23_graphite_static_cache_hit"].item() == 1.0
