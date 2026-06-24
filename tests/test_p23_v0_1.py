import torch

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
