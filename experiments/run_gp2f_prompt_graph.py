"""Run GP2F P1 prompt-graph adaptation experiments."""

from __future__ import annotations

import argparse
import copy
import csv
import math
from datetime import datetime
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from tqdm.auto import tqdm

from data import build_few_shot_split, load_node_dataset
from experiments.run_gp2f_baseline import (
    InputAligner,
    _adapter_stats,
    _deep_update,
    _environment_info,
    _format_mean_std,
    _model_variant,
    _resolve_path,
    _resolve_run_seeds,
    _split_counts,
    set_seed,
)
from models import FaithfulGP2F, PromptAwareGP2F, PromptGraphModuleP1, load_pretrained_gcn
from models.prompt_graph_module import (
    prompt_acceptance_budget_loss,
    prompt_acceptance_loss,
    prompt_balance_loss,
    prompt_class_route_loss,
    prompt_edge_l1_loss,
    prompt_key_proto_loss,
    prompt_role_diversity_loss,
    prompt_usage_consistency_loss,
    prompt_view_entropy_loss,
)
from models.prompt_module import count_trainable_parameters
from utils.io import read_yaml, write_json, write_yaml
from utils.metrics import split_metrics


PROMPT_GRAPH_VARIANTS = {
    "noprompt",
    "p1_graph",
    "p1_random_pool",
    "p2_prompt_aware",
    "p2_strength",
    "p2_multiview",
    "p2_pattern_benefit",
    "p5_multi_expert",
    "p5_lite_receiver",
    "p2_strength_random_pool",
    "p2_no_node_to_prompt",
    "p2_no_prompt_to_node",
}


def _prompt_variant(config: dict[str, Any]) -> str:
    experiment_cfg = config.get("experiment", {})
    prompt_graph_cfg = config.get("prompt_graph", {})
    return str(experiment_cfg.get("prompt_variant", prompt_graph_cfg.get("variant", "p1_graph")))


def _config_for_variant(config: dict[str, Any], variant: str) -> dict[str, Any]:
    if variant not in PROMPT_GRAPH_VARIANTS:
        raise ValueError(f"Unsupported prompt_variant={variant!r}")
    out = copy.deepcopy(config)
    out.setdefault("experiment", {})["prompt_variant"] = variant
    prompt_graph = out.setdefault("prompt_graph", {})
    if variant == "noprompt":
        prompt_graph["enabled"] = False
    else:
        prompt_graph["enabled"] = True
    prompt_aware = out.setdefault("prompt_aware", {})
    prompt_aware["enabled"] = variant.startswith("p2_") or variant.startswith("p5_")
    if variant == "p2_no_node_to_prompt":
        prompt_aware["use_node_to_prompt"] = False
        prompt_aware.setdefault("use_prompt_to_node", True)
    elif variant == "p2_no_prompt_to_node":
        prompt_aware["use_prompt_to_node"] = False
        prompt_aware.setdefault("use_node_to_prompt", True)
    elif variant in {
        "p2_prompt_aware",
        "p2_strength",
        "p2_strength_random_pool",
        "p2_multiview",
        "p2_pattern_benefit",
        "p5_multi_expert",
        "p5_lite_receiver",
    }:
        prompt_aware.setdefault("use_node_to_prompt", True)
        prompt_aware.setdefault("use_prompt_to_node", True)
    if variant == "p2_pattern_benefit":
        prompt_aware["use_node_to_prompt"] = False
        prompt_aware["use_prompt_to_node"] = True
    if variant in {
        "p2_strength",
        "p2_strength_random_pool",
        "p2_multiview",
        "p2_pattern_benefit",
        "p5_multi_expert",
        "p5_lite_receiver",
    }:
        prompt_aware.setdefault("gate_init", 0.05)
        prompt_aware.setdefault("message_scale", 5.0)
        prompt_aware.setdefault("prompt_message_norm", "weighted_mean")
        prompt_graph.setdefault("edge_scale_init", 0.05)
        prompt_graph.setdefault("edge_scale_max", 0.75)
        prompt_graph.setdefault("lambda_prompt_balance", 0.02)
    if variant in {"p2_multiview", "p5_multi_expert", "p5_lite_receiver"}:
        prompt_graph["use_multiview_routing"] = True
        prompt_graph.setdefault("use_class_aware_routing", True)
        prompt_graph.setdefault("residual_prompt_count", 2)
        prompt_graph.setdefault("lambda_class_route", 0.05)
        prompt_graph.setdefault("lambda_key_proto", 0.001)
        prompt_graph.setdefault("lambda_prompt_usage_consistency", 0.02)
    if variant == "p5_multi_expert":
        prompt_aware["receiver_version"] = "v4_multi_expert_residual"
        prompt_aware.setdefault("prompt_fusion", "multi_expert_residual_correction")
        prompt_aware.setdefault("zero_init_prompt_messages", True)
        prompt_aware.setdefault("prompt_message_norm", "layernorm")
        prompt_aware.setdefault("use_bounded_prompt_update", True)
        prompt_aware.setdefault("max_prompt_update_norm", 0.05)
        prompt_graph.setdefault("residual_prompt_count", 8)
        prompt_graph.setdefault("num_prompt_nodes", 16)
        prompt_graph.setdefault("init_class_keys_from_train_proto", True)
        prompt_graph.setdefault("init_pattern_keys_from_pool_medoids", True)
        prompt_graph.setdefault("use_benefit_gate", True)
        prompt_graph.setdefault("use_hard_receive_gate", True)
        prompt_graph.setdefault("hard_receive_ratio", 0.20)
        prompt_graph.setdefault("lambda_prompt_acceptance_supervision", 0.0)
        prompt_graph.setdefault("lambda_prompt_acceptance_budget", 0.0)
        prompt_graph.setdefault("lambda_prompt_message_help", 0.10)
        prompt_graph.setdefault("prompt_message_help_margin", 0.01)
        prompt_graph.setdefault("prompt_message_help_warmup_epochs", 10)
        prompt_graph.setdefault("prompt_message_help_class_balanced", True)
        prompt_graph.setdefault("lambda_prompt_class_anti_harm", 0.10)
        prompt_graph.setdefault("prompt_class_anti_harm_floor", 0.0)
        prompt_graph.setdefault("lambda_prompt_benefit_supervision", 0.05)
        prompt_graph.setdefault("benefit_supervision_warmup_epochs", 10)
        prompt_graph.setdefault("benefit_supervision_label_strategy", "hybrid_quantile_margin")
        prompt_graph.setdefault("benefit_supervision_quantile", 0.20)
        prompt_graph.setdefault("benefit_delta_eps", 1e-4)
    if variant == "p5_lite_receiver":
        prompt_aware["receiver_version"] = "v4_multi_expert_residual"
        prompt_aware.setdefault("prompt_fusion", "multi_expert_residual_correction")
        prompt_aware.setdefault("zero_init_prompt_messages", True)
        prompt_aware.setdefault("prompt_message_norm", "weighted_mean")
        prompt_aware.setdefault("use_bounded_prompt_update", True)
        prompt_aware.setdefault("max_prompt_update_norm", 0.20)
        prompt_aware.setdefault("prompt_to_node_gate_init", 0.10)
        prompt_graph.setdefault("residual_prompt_count", 8)
        prompt_graph.setdefault("num_prompt_nodes", 16)
        prompt_graph.setdefault("init_class_keys_from_train_proto", False)
        prompt_graph.setdefault("init_pattern_keys_from_pool_medoids", True)
        prompt_graph.setdefault("use_benefit_gate", False)
        prompt_graph.setdefault("use_hard_receive_gate", False)
        prompt_graph.setdefault("use_hard_acceptance", False)
        prompt_graph.setdefault("lambda_prompt_balance", 0.0)
        prompt_graph.setdefault("lambda_prompt_role_diversity", 0.0)
        prompt_graph.setdefault("lambda_prompt_acceptance", 0.0)
        prompt_graph.setdefault("lambda_prompt_acceptance_supervision", 0.0)
        prompt_graph.setdefault("lambda_prompt_acceptance_budget", 0.0)
        prompt_graph.setdefault("lambda_prompt_usage_consistency", 0.0)
        prompt_graph.setdefault("lambda_prompt_view_entropy", 0.0)
        prompt_graph.setdefault("lambda_class_route", 0.0)
        prompt_graph.setdefault("lambda_key_proto", 0.0)
        prompt_graph.setdefault("lambda_prompt_benefit_supervision", 0.0)
        prompt_graph.setdefault("lambda_prompt_correction", 0.0)
        prompt_graph.setdefault("lambda_prompt_anti_harm", 0.0)
        prompt_graph.setdefault("lambda_prompt_class_anti_harm", 0.0)
        prompt_graph.setdefault("lambda_prompt_message_help", 0.10)
        prompt_graph.setdefault("prompt_message_help_margin", 0.005)
        prompt_graph.setdefault("prompt_message_help_warmup_epochs", 5)
        prompt_graph.setdefault("prompt_message_help_class_balanced", True)
    if variant == "p2_pattern_benefit":
        prompt_graph["use_multiview_routing"] = True
        prompt_graph["use_pattern_prompt_bank"] = True
        prompt_graph["use_receiver_only_prompt"] = True
        prompt_graph["use_benefit_gate"] = True
        prompt_aware.setdefault("zero_init_prompt_messages", False)
        prompt_aware.setdefault("prompt_to_node_gate_init", 0.20)
        prompt_graph.setdefault("init_pattern_keys_from_pool_medoids", True)
        prompt_graph.setdefault("benefit_gate_bias_init", -1.0)
        prompt_graph.setdefault("lambda_prompt_benefit_supervision", 0.10)
        prompt_graph.setdefault("benefit_supervision_warmup_epochs", 20)
        prompt_graph.setdefault("benefit_delta_margin", 0.0)
        prompt_graph.setdefault("benefit_supervision_label_strategy", "hybrid_quantile_margin")
        prompt_graph.setdefault("benefit_supervision_quantile", 0.20)
        prompt_graph.setdefault("benefit_delta_eps", 1e-4)
        prompt_graph.setdefault("benefit_supervision_quantile_warmup_epochs", 20)
        prompt_graph.setdefault("benefit_supervision_probe_scale", 2.0)
        prompt_graph.setdefault("benefit_supervision_balance_targets", True)
        prompt_graph.setdefault("lambda_prompt_correction", 0.10)
        prompt_graph.setdefault("lambda_prompt_anti_harm", 0.02)
        prompt_graph.setdefault("prompt_correction_warmup_epochs", 20)
        prompt_graph.setdefault("prompt_correction_eps", 0.0001)
        prompt_graph.setdefault("prompt_correction_target", "delta_prob_to_label")
        prompt_graph.setdefault("use_hard_receive_gate", True)
        prompt_graph.setdefault("hard_receive_ratio", 0.15)
        prompt_graph.setdefault("hard_receive_straight_through", True)
    if variant in {"p1_random_pool", "p2_strength_random_pool"}:
        prompt_graph["pool_strategy"] = "random"
    elif variant in {
        "p1_graph",
        "p2_prompt_aware",
        "p2_strength",
        "p2_multiview",
        "p2_pattern_benefit",
        "p5_multi_expert",
        "p5_lite_receiver",
        "p2_no_node_to_prompt",
        "p2_no_prompt_to_node",
    }:
        prompt_graph.setdefault("pool_strategy", "structural")
    return out


def _class_balanced_support_query_split(
    labels: torch.Tensor,
    train_mask: torch.Tensor,
    *,
    support_ratio: float = 0.70,
    min_query_per_class: int = 1,
    seed: int = 0,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
    """Split training nodes into support/query masks without using val/test labels."""

    device = labels.device
    support = torch.zeros_like(train_mask, dtype=torch.bool, device=device)
    query = torch.zeros_like(train_mask, dtype=torch.bool, device=device)
    support_ratio = min(max(float(support_ratio), 0.0), 1.0)
    min_query_per_class = max(0, int(min_query_per_class))
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))

    by_class: dict[str, dict[str, int]] = {}
    train_idx_all = torch.where(train_mask.bool())[0]
    if train_idx_all.numel() == 0:
        return support, query, {"enabled": True, "support_count": 0, "query_count": 0, "by_class": by_class}

    for class_id in torch.unique(labels[train_idx_all].detach().cpu()).tolist():
        class_id = int(class_id)
        idx = torch.where(train_mask.bool() & (labels == class_id))[0]
        count = int(idx.numel())
        if count == 0:
            continue
        if count <= min_query_per_class:
            support_idx = idx
            query_idx = idx.new_empty(0)
        else:
            perm = torch.randperm(count, generator=generator, device="cpu").to(device=idx.device)
            query_count = min(max(min_query_per_class, int(round(count * (1.0 - support_ratio)))), count - 1)
            query_idx = idx[perm[:query_count]]
            support_idx = idx[perm[query_count:]]
        support[support_idx] = True
        query[query_idx] = True
        by_class[str(class_id)] = {
            "total": count,
            "support": int(support_idx.numel()),
            "query": int(query_idx.numel()),
        }

    # Tiny-shot fallback: if every class had too few nodes, keep the old train-only behavior.
    if int(query.sum().item()) == 0:
        support = train_mask.bool().clone()

    return support, query, {
        "enabled": True,
        "support_count": int(support.sum().item()),
        "query_count": int(query.sum().item()),
        "support_ratio": float(support.float().mean().item()),
        "train_support_ratio": float(
            support.float().sum().item() / max(1, int(train_mask.bool().sum().item()))
        ),
        "by_class": by_class,
    }


def _support_query_masks_for_epoch(
    labels: torch.Tensor,
    train_mask: torch.Tensor,
    prompt_graph_cfg: dict[str, Any],
    *,
    seed: int,
    epoch: int = 0,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
    split_cfg = prompt_graph_cfg.get("support_query_split", {})
    enabled = bool(split_cfg.get("enabled", prompt_graph_cfg.get("support_query_split_enabled", False)))
    if not enabled:
        train = train_mask.bool().clone()
        empty = torch.zeros_like(train, dtype=torch.bool)
        return train, empty, {
            "enabled": False,
            "support_count": int(train.sum().item()),
            "query_count": 0,
            "train_support_ratio": 1.0,
            "by_class": {},
        }
    support_ratio = float(split_cfg.get("support_ratio", prompt_graph_cfg.get("support_query_support_ratio", 0.70)))
    min_query = int(split_cfg.get("min_query_per_class", prompt_graph_cfg.get("support_query_min_query_per_class", 1)))
    resample = bool(split_cfg.get("resample_each_epoch", prompt_graph_cfg.get("support_query_resample_each_epoch", True)))
    split_seed = int(seed if resample else int(split_cfg.get("seed", seed)))
    if resample:
        split_seed = split_seed * 1000003 + int(epoch)
    return _class_balanced_support_query_split(
        labels,
        train_mask,
        support_ratio=support_ratio,
        min_query_per_class=min_query,
        seed=split_seed,
    )


def _build_prompt_graph_module(
    *,
    variant: str,
    source_dim: int,
    hidden_dim: int,
    num_classes: int,
    prompt_graph_cfg: dict[str, Any],
    device: torch.device,
) -> PromptGraphModuleP1 | None:
    if variant == "noprompt" or not bool(prompt_graph_cfg.get("enabled", True)):
        return None
    resolved_cfg = dict(prompt_graph_cfg)
    resolved_cfg.setdefault("num_classes", int(num_classes))
    return PromptGraphModuleP1(source_dim, hidden_dim, resolved_cfg).to(device)


@torch.no_grad()
def _maybe_initialize_class_keys(
    *,
    prompt_graph_module: PromptGraphModuleP1 | None,
    input_aligner: InputAligner,
    model: FaithfulGP2F,
    x: torch.Tensor,
    edge_index: torch.Tensor,
    labels: torch.Tensor,
    train_mask: torch.Tensor,
    prompt_graph_cfg: dict[str, Any],
) -> dict[str, Any]:
    if prompt_graph_module is None or not bool(prompt_graph_cfg.get("init_class_keys_from_train_proto", False)):
        return {"class_key_proto_init_coverage": 0.0, "class_key_proto_init_missing_classes": []}
    was_training = prompt_graph_module.training
    input_was_training = input_aligner.training
    model_was_training = model.training
    input_aligner.eval()
    model.eval()
    prompt_graph_module.eval()
    z = input_aligner(x)
    h_pre = model.encode_frozen(z, edge_index)
    stats = prompt_graph_module.initialize_class_keys_from_train_prototypes(
        z=z,
        h_pre=h_pre.detach(),
        edge_index=edge_index,
        train_mask=train_mask,
        labels=labels,
        normalize=bool(prompt_graph_cfg.get("class_key_init_normalize", True)),
        source=str(prompt_graph_cfg.get("class_key_init_source", "structural_query")),
    )
    input_aligner.train(input_was_training)
    model.train(model_was_training)
    prompt_graph_module.train(was_training)
    return stats


@torch.no_grad()
def _maybe_initialize_pattern_keys(
    *,
    prompt_graph_module: PromptGraphModuleP1 | None,
    input_aligner: InputAligner,
    model: FaithfulGP2F,
    x: torch.Tensor,
    edge_index: torch.Tensor,
    train_mask: torch.Tensor,
    prompt_graph_cfg: dict[str, Any],
) -> dict[str, Any]:
    if prompt_graph_module is None or not bool(prompt_graph_cfg.get("init_pattern_keys_from_pool_medoids", False)):
        return {"pattern_key_init_coverage": 0.0, "pattern_key_init_selected_nodes": []}
    was_training = prompt_graph_module.training
    input_was_training = input_aligner.training
    model_was_training = model.training
    input_aligner.eval()
    model.eval()
    prompt_graph_module.eval()
    z = input_aligner(x)
    h_pre = model.encode_frozen(z, edge_index)
    stats = prompt_graph_module.initialize_pattern_keys_from_pool_medoids(
        z=z,
        h_pre=h_pre.detach(),
        edge_index=edge_index,
        train_mask=train_mask,
        normalize=bool(prompt_graph_cfg.get("pattern_key_init_normalize", True)),
    )
    input_aligner.train(input_was_training)
    model.train(model_was_training)
    prompt_graph_module.train(was_training)
    return stats


def _default_prompt_graph_out(z: torch.Tensor, edge_index: torch.Tensor) -> dict[str, Any]:
    edge_weight = torch.ones(edge_index.size(1), dtype=z.dtype, device=z.device)
    return {
        "adapted_x": z,
        "adapted_edge_index": edge_index,
        "adapted_edge_weight": edge_weight,
        "adapted_edge_type": torch.zeros(edge_index.size(1), dtype=torch.long, device=edge_index.device),
        "prompt_node_x": z.new_zeros((0, z.size(-1))),
        "pool_mask": torch.zeros(z.size(0), dtype=torch.bool, device=z.device),
        "prompt_edge_count": 0,
        "edge_scale": z.new_tensor(0.0),
        "aux": {
            "prompt_edge_weight": z.new_zeros(0),
            "prompt_usage": z.new_zeros(0),
            "prompt_usage_entropy": z.new_tensor(0.0),
            "connected_edge_count": 0,
            "edge_type_counts": [int(edge_index.size(1)), 0, 0],
        },
    }


def _set_module_trainable(module: torch.nn.Module | None, trainable: bool) -> None:
    if module is None:
        return
    for parameter in module.parameters():
        parameter.requires_grad = bool(trainable)


def _trainable_parameters(module: torch.nn.Module | None) -> list[torch.nn.Parameter]:
    if module is None:
        return []
    return [parameter for parameter in module.parameters() if parameter.requires_grad]


def _prompt_aware_parameter_names(model: FaithfulGP2F) -> set[str]:
    if not isinstance(model, PromptAwareGP2F):
        return set()
    prefixes = (
        "prompt_gate_logit",
        "node_to_prompt_msgs.",
        "prompt_to_node_msgs.",
        "node_to_prompt_conditioned_msgs.",
        "prompt_to_node_conditioned_msgs.",
        "prompt_node_residual_corrections.",
        "prompt_slot_residual_corrections.",
        "prototype_direction_projections.",
        "prototype_direction_strength",
        "prompt_update_norms.",
    )
    return {
        name
        for name, parameter in model.named_parameters()
        if parameter.requires_grad and any(name == prefix.rstrip(".") or name.startswith(prefix) for prefix in prefixes)
    }


def _set_prompt_aware_trainable(model: FaithfulGP2F, trainable: bool) -> None:
    if not isinstance(model, PromptAwareGP2F):
        return
    names = _prompt_aware_parameter_names(model)
    if names:
        for name, parameter in model.named_parameters():
            if name in names:
                parameter.requires_grad = bool(trainable)
        return
    prefixes = (
        "prompt_gate_logit",
        "node_to_prompt_msgs.",
        "prompt_to_node_msgs.",
        "node_to_prompt_conditioned_msgs.",
        "prompt_to_node_conditioned_msgs.",
        "prompt_node_residual_corrections.",
        "prompt_slot_residual_corrections.",
        "prototype_direction_projections.",
        "prototype_direction_strength",
        "prompt_update_norms.",
    )
    for name, parameter in model.named_parameters():
        if any(name == prefix.rstrip(".") or name.startswith(prefix) for prefix in prefixes):
            parameter.requires_grad = bool(trainable)


def _trainable_parameter_summary(
    *,
    input_aligner: InputAligner,
    model: FaithfulGP2F,
    prompt_graph_module: torch.nn.Module | None,
) -> dict[str, Any]:
    groups: dict[str, list[str]] = {}
    counts: dict[str, int] = {}
    modules: list[tuple[str, torch.nn.Module | None]] = [
        ("input_aligner", input_aligner),
        ("model", model),
        ("prompt_graph_module", prompt_graph_module),
    ]
    for group_name, module in modules:
        names: list[str] = []
        total = 0
        if module is not None:
            for name, parameter in module.named_parameters():
                if parameter.requires_grad:
                    names.append(f"{group_name}.{name}")
                    total += int(parameter.numel())
        groups[group_name] = names
        counts[f"{group_name}_trainable_parameter_count"] = total
    all_names = [name for names in groups.values() for name in names]
    return {
        "trainable_parameter_names": all_names,
        "trainable_parameter_groups": groups,
        "actual_trainable_parameter_count": sum(counts.values()),
        **counts,
    }


def _optimizer_groups(
    *,
    input_aligner: InputAligner,
    model: FaithfulGP2F,
    prompt_graph_module: torch.nn.Module | None,
    training_cfg: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[torch.nn.Parameter], dict[str, float]]:
    base_lr = float(training_cfg.get("lr", 0.001))
    base_weight_decay = float(training_cfg.get("weight_decay", 5e-4))
    prompt_lr_cfg = training_cfg.get("prompt_lr")
    if prompt_lr_cfg is None:
        prompt_lr = base_lr * float(training_cfg.get("prompt_lr_multiplier", 1.0))
    else:
        prompt_lr = float(prompt_lr_cfg)
    prompt_weight_decay = float(training_cfg.get("prompt_weight_decay", base_weight_decay))

    prompt_aware_names = _prompt_aware_parameter_names(model)
    prompt_aware_params = [
        parameter
        for name, parameter in model.named_parameters()
        if parameter.requires_grad and name in prompt_aware_names
    ]
    prompt_aware_param_ids = {id(parameter) for parameter in prompt_aware_params}
    model_base_params = [
        parameter
        for parameter in model.parameters()
        if parameter.requires_grad and id(parameter) not in prompt_aware_param_ids
    ]
    base_params = _trainable_parameters(input_aligner) + model_base_params
    prompt_params = prompt_aware_params + _trainable_parameters(prompt_graph_module)
    groups: list[dict[str, Any]] = []
    if base_params:
        groups.append({"params": base_params, "lr": base_lr, "weight_decay": base_weight_decay, "name": "base"})
    if prompt_params:
        groups.append({"params": prompt_params, "lr": prompt_lr, "weight_decay": prompt_weight_decay, "name": "prompt_graph"})
    trainable_params = [parameter for group in groups for parameter in group["params"]]
    return groups, trainable_params, {
        "base_lr": base_lr if base_params else 0.0,
        "prompt_lr": prompt_lr if prompt_params else 0.0,
        "base_weight_decay": base_weight_decay if base_params else 0.0,
        "prompt_weight_decay": prompt_weight_decay if prompt_params else 0.0,
        "prompt_aware_parameter_count": sum(int(parameter.numel()) for parameter in prompt_aware_params),
    }


def _resolve_base_checkpoint(
    training_cfg: dict[str, Any],
    *,
    repo_root: Path,
    seed: int,
) -> Path | None:
    checkpoint_path = training_cfg.get("base_checkpoint_path")
    if checkpoint_path:
        formatted = str(checkpoint_path).format(seed=int(seed))
        return _resolve_path(formatted, base_dir=repo_root)
    checkpoint_root = training_cfg.get("base_checkpoint_root")
    if not checkpoint_root:
        return None
    root = _resolve_path(checkpoint_root, base_dir=repo_root)
    candidates = sorted((root / f"seed_{int(seed)}").glob("**/best_model.pt"))
    if not candidates:
        raise FileNotFoundError(f"No base checkpoint found under {root / f'seed_{int(seed)}'}")
    return candidates[0]


def _load_base_checkpoint(
    checkpoint_path: Path,
    *,
    model: FaithfulGP2F,
    input_aligner: InputAligner,
    prompt_graph_module: torch.nn.Module | None,
    device: torch.device,
    load_prompt: bool = False,
) -> None:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model"])
    input_aligner.load_state_dict(checkpoint["input_aligner"])
    if load_prompt and prompt_graph_module is not None and checkpoint.get("prompt_graph_module") is not None:
        prompt_graph_module.load_state_dict(checkpoint["prompt_graph_module"])


def _save_checkpoint(
    path: Path,
    *,
    model: FaithfulGP2F,
    input_aligner: InputAligner,
    prompt_graph_module: torch.nn.Module | None,
    epoch: int,
    metrics: dict[str, float],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "input_aligner": input_aligner.state_dict(),
            "prompt_graph_module": prompt_graph_module.state_dict() if prompt_graph_module is not None else None,
            "epoch": int(epoch),
            "metrics": metrics,
        },
        path,
    )


def _monitor_improved(monitor: str, value: float, best_value: float, has_best: bool) -> bool:
    if not has_best:
        return True
    return value < best_value if monitor in {"train_loss", "total", "cls", "edge_l1", "prompt_balance"} else value > best_value


def _pool_label_distribution(pool_mask: torch.Tensor, labels: torch.Tensor, num_classes: int) -> list[float]:
    pool_labels = labels[pool_mask]
    if pool_labels.numel() == 0:
        return [0.0 for _ in range(int(num_classes))]
    counts = torch.stack([(pool_labels == class_id).sum() for class_id in range(int(num_classes))]).float()
    return [float(value) for value in (counts / counts.sum().clamp_min(1.0)).tolist()]


def _prompt_usage_by_true_class(
    prompt_out: dict[str, Any],
    labels: torch.Tensor,
    num_classes: int,
) -> list[list[float]]:
    aux = prompt_out.get("aux", {})
    top_prompt_ids = aux.get("top_prompt_ids")
    assignment_prob = aux.get("assignment_prob")
    pool_mask = prompt_out.get("pool_mask")
    prompt_node_x = prompt_out.get("prompt_node_x")
    if not (
        isinstance(top_prompt_ids, torch.Tensor)
        and isinstance(assignment_prob, torch.Tensor)
        and isinstance(pool_mask, torch.Tensor)
        and isinstance(prompt_node_x, torch.Tensor)
    ):
        return []
    num_prompt_nodes = int(prompt_node_x.size(0))
    usage = torch.zeros((int(num_classes), num_prompt_nodes), dtype=assignment_prob.dtype, device=assignment_prob.device)
    pool_idx = torch.where(pool_mask)[0]
    if pool_idx.numel() == 0:
        return [[0.0 for _ in range(num_prompt_nodes)] for _ in range(int(num_classes))]
    pool_labels = labels[pool_idx]
    for row in range(top_prompt_ids.size(0)):
        class_id = int(pool_labels[row].item())
        usage[class_id].index_add_(0, top_prompt_ids[row], assignment_prob[row])
    usage = usage / usage.sum(dim=1, keepdim=True).clamp_min(1e-12)
    return [[float(value) for value in row] for row in usage.detach().cpu().tolist()]


def _prompt_mediated_same_label_reachability(prompt_out: dict[str, Any], labels: torch.Tensor) -> float:
    aux = prompt_out.get("aux", {})
    top_prompt_ids = aux.get("top_prompt_ids")
    assignment_prob = aux.get("assignment_prob")
    pool_mask = prompt_out.get("pool_mask")
    prompt_node_x = prompt_out.get("prompt_node_x")
    if not (
        isinstance(top_prompt_ids, torch.Tensor)
        and isinstance(assignment_prob, torch.Tensor)
        and isinstance(pool_mask, torch.Tensor)
        and isinstance(prompt_node_x, torch.Tensor)
    ):
        return 0.0
    pool_idx = torch.where(pool_mask)[0]
    if pool_idx.numel() < 2:
        return 0.0
    top1 = top_prompt_ids[torch.arange(top_prompt_ids.size(0), device=top_prompt_ids.device), assignment_prob.argmax(dim=1)]
    pool_labels = labels[pool_idx]
    total_pairs = 0
    same_pairs = 0
    for prompt_id in range(int(prompt_node_x.size(0))):
        group_labels = pool_labels[top1 == prompt_id]
        group_size = int(group_labels.numel())
        if group_size < 2:
            continue
        for left in range(group_size):
            for right in range(left + 1, group_size):
                total_pairs += 1
                same_pairs += int(group_labels[left].item() == group_labels[right].item())
    if total_pairs == 0:
        return 0.0
    return float(same_pairs / total_pairs)


def _class_router_hit_rate(
    prompt_out: dict[str, Any],
    labels: torch.Tensor,
    mask: torch.Tensor,
) -> float:
    aux = prompt_out.get("aux", {})
    full_prob = aux.get("routing_full_prob")
    pool_mask = prompt_out.get("pool_mask")
    class_slots = int(aux.get("num_class_prompt_slots", 0))
    if not (isinstance(full_prob, torch.Tensor) and isinstance(pool_mask, torch.Tensor) and class_slots > 0):
        return 0.0
    pool_idx = aux.get("pool_idx")
    if not isinstance(pool_idx, torch.Tensor):
        pool_idx = torch.where(pool_mask)[0]
    selected = mask.to(device=pool_mask.device, dtype=torch.bool)[pool_idx]
    if int(selected.sum().item()) == 0:
        return 0.0
    y = labels.to(device=pool_mask.device)[pool_idx][selected]
    valid = (y >= 0) & (y < class_slots)
    if int(valid.sum().item()) == 0:
        return 0.0
    top1 = full_prob[selected].argmax(dim=-1)[valid]
    return float((top1 == y[valid]).float().mean().detach().item())


def _prompt_label_purity(prompt_out: dict[str, Any], labels: torch.Tensor, mask: torch.Tensor) -> float:
    aux = prompt_out.get("aux", {})
    full_prob = aux.get("routing_full_prob")
    pool_mask = prompt_out.get("pool_mask")
    if not (isinstance(full_prob, torch.Tensor) and isinstance(pool_mask, torch.Tensor)):
        return 0.0
    pool_idx = aux.get("pool_idx")
    if not isinstance(pool_idx, torch.Tensor):
        pool_idx = torch.where(pool_mask)[0]
    selected = mask.to(device=pool_mask.device, dtype=torch.bool)[pool_idx]
    if int(selected.sum().item()) < 2:
        return 0.0
    top1 = full_prob[selected].argmax(dim=-1)
    y = labels.to(device=pool_mask.device)[pool_idx][selected]
    total_pairs = 0
    same_pairs = 0
    for prompt_id in top1.unique().tolist():
        group_labels = y[top1 == int(prompt_id)]
        group_size = int(group_labels.numel())
        if group_size < 2:
            continue
        same = group_labels[:, None] == group_labels[None, :]
        pair_count = group_size * (group_size - 1) // 2
        total_pairs += pair_count
        same_pairs += int(torch.triu(same, diagonal=1).sum().item())
    if total_pairs == 0:
        return 0.0
    return float(same_pairs / total_pairs)


def _route_compactness_stats(prompt_out: dict[str, Any], labels: torch.Tensor, train_mask: torch.Tensor) -> dict[str, float]:
    aux = prompt_out.get("aux", {})
    full_prob = aux.get("routing_full_prob")
    pool_mask = prompt_out.get("pool_mask")
    if not (isinstance(full_prob, torch.Tensor) and isinstance(pool_mask, torch.Tensor)):
        return {"same_class_route_compactness": 0.0, "different_class_route_separation": 0.0}
    pool_idx = aux.get("pool_idx")
    if not isinstance(pool_idx, torch.Tensor):
        pool_idx = torch.where(pool_mask)[0]
    train_pool = train_mask.to(device=pool_mask.device, dtype=torch.bool)[pool_idx]
    if int(train_pool.sum().item()) < 2:
        return {"same_class_route_compactness": 0.0, "different_class_route_separation": 0.0}
    probs = full_prob[train_pool]
    y = labels.to(device=pool_mask.device)[pool_idx][train_pool]
    dist = torch.cdist(probs, probs, p=2)
    same = y[:, None] == y[None, :]
    eye = torch.eye(same.size(0), dtype=torch.bool, device=same.device)
    same = same & ~eye
    diff = (~same) & ~eye
    return {
        "same_class_route_compactness": float(dist[same].mean().detach().item()) if bool(same.any()) else 0.0,
        "different_class_route_separation": float(dist[diff].mean().detach().item()) if bool(diff.any()) else 0.0,
    }


def _prompt_slot_usage_stats(
    usage: torch.Tensor | None,
    *,
    active_threshold: float = 0.05,
) -> dict[str, float]:
    if not isinstance(usage, torch.Tensor) or usage.numel() == 0:
        return {
            "dominant_prompt_slot_ratio": 0.0,
            "active_prompt_slot_count@0.05": 0.0,
        }
    detached = usage.detach()
    return {
        "dominant_prompt_slot_ratio": float(detached.max().item()),
        "active_prompt_slot_count@0.05": float((detached >= float(active_threshold)).sum().item()),
    }


def _prompt_graph_diagnostics(
    prompt_out: dict[str, Any],
    *,
    z: torch.Tensor,
    train_mask: torch.Tensor,
    val_mask: torch.Tensor | None = None,
    test_mask: torch.Tensor | None = None,
    labels: torch.Tensor | None = None,
    num_classes: int | None = None,
) -> dict[str, Any]:
    pool_mask = prompt_out["pool_mask"].detach()
    aux = prompt_out.get("aux", {})
    prompt_edge_weight = aux.get("prompt_edge_weight")
    prompt_usage = aux.get("prompt_usage")
    prompt_usage_full = aux.get("prompt_usage_full")
    prompt_node_x = prompt_out.get("prompt_node_x")
    prompt_edge_count = int(prompt_out.get("prompt_edge_count", 0))
    prompt_node_count = int(prompt_node_x.size(0)) if isinstance(prompt_node_x, torch.Tensor) else 0
    edge_weight_mean = (
        float(prompt_edge_weight.detach().mean().item())
        if isinstance(prompt_edge_weight, torch.Tensor) and prompt_edge_weight.numel() > 0
        else 0.0
    )
    diagnostics: dict[str, Any] = {
        "prompt_node_count": prompt_node_count,
        "prompt_edge_count": prompt_edge_count,
        "pool_ratio": float(pool_mask.float().mean().item()),
        "train_pool_ratio": float((pool_mask & train_mask.bool()).float().mean().item()),
        "edge_scale": float(prompt_out["edge_scale"].detach().item()),
        "raw_edge_scale": (
            float(aux["raw_edge_scale"].detach().item())
            if isinstance(aux.get("raw_edge_scale"), torch.Tensor)
            else float(prompt_out["edge_scale"].detach().item())
        ),
        "edge_scale_multiplier": (
            float(aux["edge_scale_multiplier"].detach().item())
            if isinstance(aux.get("edge_scale_multiplier"), torch.Tensor)
            else 1.0
        ),
        "mean_prompt_edge_weight": edge_weight_mean,
        "pool_acceptance_mean": (
            float(aux["pool_acceptance_mean"].detach().item())
            if isinstance(aux.get("pool_acceptance_mean"), torch.Tensor)
            else 1.0
        ),
        "pool_acceptance_min": (
            float(aux["pool_acceptance_min"].detach().item())
            if isinstance(aux.get("pool_acceptance_min"), torch.Tensor)
            else 1.0
        ),
        "pool_acceptance_max": (
            float(aux["pool_acceptance_max"].detach().item())
            if isinstance(aux.get("pool_acceptance_max"), torch.Tensor)
            else 1.0
        ),
        "use_hard_acceptance": float(aux.get("use_hard_acceptance", 0)),
        "hard_acceptance_ratio": (
            float(aux["hard_acceptance_ratio"].detach().item())
            if isinstance(aux.get("hard_acceptance_ratio"), torch.Tensor)
            else 0.0
        ),
        "hard_acceptance_selected_ratio": (
            float(aux["hard_acceptance_selected_ratio"].detach().item())
            if isinstance(aux.get("hard_acceptance_selected_ratio"), torch.Tensor)
            else 0.0
        ),
        "prompt_usage_entropy": (
            float(aux["prompt_usage_entropy"].detach().item())
            if isinstance(aux.get("prompt_usage_entropy"), torch.Tensor)
            else 0.0
        ),
        "prompt_usage_full_entropy": (
            float(aux["prompt_usage_full_entropy"].detach().item())
            if isinstance(aux.get("prompt_usage_full_entropy"), torch.Tensor)
            else 0.0
        ),
        "capacity_routing_enabled": float(aux.get("capacity_routing_enabled", 0)),
        "prompt_capacity": float(aux.get("prompt_capacity", 0)),
        "capacity_overflow_count": float(aux.get("capacity_overflow_count", 0)),
        "adapted_node_count": int(prompt_out["adapted_x"].size(0)),
        "connected_edge_count": int(aux.get("connected_edge_count", prompt_edge_count)),
        "use_multiview_routing": float(aux.get("use_multiview_routing", 0)),
        "use_class_aware_routing": float(aux.get("use_class_aware_routing", 0)),
        "use_pattern_prompt_bank": float(aux.get("use_pattern_prompt_bank", 0)),
        "use_receiver_only_prompt": float(aux.get("use_receiver_only_prompt", 0)),
        "use_benefit_gate": float(aux.get("use_benefit_gate", 0)),
        "num_class_prompt_slots": float(aux.get("num_class_prompt_slots", 0)),
        "residual_prompt_count": float(aux.get("residual_prompt_count", 0)),
        "class_key_proto_init_coverage": float(aux.get("class_key_proto_init_coverage", 0.0)),
        "pattern_key_init_coverage": float(aux.get("pattern_key_init_coverage", 0.0)),
        "benefit_gate_mean": (
            float(aux["benefit_gate_mean"].detach().item())
            if isinstance(aux.get("benefit_gate_mean"), torch.Tensor)
            else 1.0
        ),
        "benefit_gate_min": (
            float(aux["benefit_gate_min"].detach().item())
            if isinstance(aux.get("benefit_gate_min"), torch.Tensor)
            else 1.0
        ),
        "benefit_gate_max": (
            float(aux["benefit_gate_max"].detach().item())
            if isinstance(aux.get("benefit_gate_max"), torch.Tensor)
            else 1.0
        ),
        "use_utility_receive_gate": float(aux.get("use_utility_receive_gate", 0)),
        "utility_receive_gate_mean": (
            float(aux["utility_receive_gate_mean"].detach().item())
            if isinstance(aux.get("utility_receive_gate_mean"), torch.Tensor)
            else 1.0
        ),
        "utility_receive_gate_min": (
            float(aux["utility_receive_gate_min"].detach().item())
            if isinstance(aux.get("utility_receive_gate_min"), torch.Tensor)
            else 1.0
        ),
        "utility_receive_gate_max": (
            float(aux["utility_receive_gate_max"].detach().item())
            if isinstance(aux.get("utility_receive_gate_max"), torch.Tensor)
            else 1.0
        ),
        "utility_receive_gate_floor": (
            float(aux["utility_receive_gate_floor"].detach().item())
            if isinstance(aux.get("utility_receive_gate_floor"), torch.Tensor)
            else 0.0
        ),
        "use_hard_receive_gate": float(aux.get("use_hard_receive_gate", 0)),
        "hard_receive_ratio": (
            float(aux["hard_receive_ratio"].detach().item())
            if isinstance(aux.get("hard_receive_ratio"), torch.Tensor)
            else 0.0
        ),
        "hard_receive_selected_ratio": (
            float(aux["hard_receive_selected_ratio"].detach().item())
            if isinstance(aux.get("hard_receive_selected_ratio"), torch.Tensor)
            else 0.0
        ),
        "receive_gate_mean": (
            float(aux["receive_gate_mean"].detach().item())
            if isinstance(aux.get("receive_gate_mean"), torch.Tensor)
            else 1.0
        ),
        "receive_gate_min": (
            float(aux["receive_gate_min"].detach().item())
            if isinstance(aux.get("receive_gate_min"), torch.Tensor)
            else 1.0
        ),
        "receive_gate_max": (
            float(aux["receive_gate_max"].detach().item())
            if isinstance(aux.get("receive_gate_max"), torch.Tensor)
            else 1.0
        ),
        "view_gate_entropy": (
            float(aux["view_gate_entropy"].detach().item())
            if isinstance(aux.get("view_gate_entropy"), torch.Tensor)
            else 0.0
        ),
    }
    view_gate_mean = aux.get("view_gate_mean")
    if isinstance(view_gate_mean, torch.Tensor) and view_gate_mean.numel() == 3:
        view_values = [float(value) for value in view_gate_mean.detach().cpu().tolist()]
    else:
        view_values = [0.0, 1.0, 0.0]
    diagnostics["semantic_view_weight"] = view_values[0]
    diagnostics["structural_view_weight"] = view_values[1]
    diagnostics["role_view_weight"] = view_values[2]
    if isinstance(prompt_usage, torch.Tensor):
        diagnostics["prompt_usage_distribution"] = [float(value) for value in prompt_usage.detach().cpu().tolist()]
    else:
        diagnostics["prompt_usage_distribution"] = []
    if isinstance(prompt_usage_full, torch.Tensor):
        diagnostics["prompt_usage_full_distribution"] = [float(value) for value in prompt_usage_full.detach().cpu().tolist()]
    else:
        diagnostics["prompt_usage_full_distribution"] = []
    slot_usage = prompt_usage_full if isinstance(prompt_usage_full, torch.Tensor) else prompt_usage
    diagnostics.update(_prompt_slot_usage_stats(slot_usage))
    if labels is not None and num_classes is not None:
        diagnostics["pool_label_distribution"] = _pool_label_distribution(pool_mask, labels, int(num_classes))
        diagnostics["prompt_usage_by_true_class"] = _prompt_usage_by_true_class(prompt_out, labels, int(num_classes))
        diagnostics["prompt_mediated_same_label_reachability"] = _prompt_mediated_same_label_reachability(prompt_out, labels)
        diagnostics["class_router_hit_rate_train"] = _class_router_hit_rate(prompt_out, labels, train_mask)
        diagnostics["prompt_label_purity_train"] = _prompt_label_purity(prompt_out, labels, train_mask)
        diagnostics["residual_prompt_usage_ratio"] = sum(
            diagnostics.get("prompt_usage_full_distribution", [])[int(aux.get("num_class_prompt_slots", 0)) :]
        )
        diagnostics.update(_route_compactness_stats(prompt_out, labels, train_mask))
        if val_mask is not None:
            diagnostics["class_router_hit_rate_val"] = _class_router_hit_rate(prompt_out, labels, val_mask)
        if test_mask is not None:
            diagnostics["class_router_hit_rate_test"] = _class_router_hit_rate(prompt_out, labels, test_mask)
    edge_type_counts = aux.get("edge_type_counts")
    if isinstance(edge_type_counts, list):
        diagnostics["edge_type_counts"] = [int(value) for value in edge_type_counts]
    else:
        edge_type = prompt_out.get("adapted_edge_type")
        if isinstance(edge_type, torch.Tensor):
            diagnostics["edge_type_counts"] = [
                int((edge_type == type_id).sum().item()) for type_id in range(3)
            ]
        else:
            diagnostics["edge_type_counts"] = [int(prompt_out["adapted_edge_index"].size(1)), 0, 0]
    return diagnostics


def _prompt_aware_diagnostics(model_out: dict[str, Any]) -> dict[str, Any]:
    aux = model_out.get("prompt_aware", {})
    if not isinstance(aux, dict):
        aux = {}

    def scalar(name: str) -> float:
        value = aux.get(name)
        if isinstance(value, torch.Tensor):
            return float(value.detach().mean().item())
        return 0.0

    gates = aux.get("prompt_gate_by_layer")
    if isinstance(gates, torch.Tensor):
        gate_by_layer = [
            [float(item) for item in row]
            for row in gates.detach().cpu().tolist()
        ]
    else:
        gate_by_layer = []
    return {
        "prompt_msg_norm": scalar("prompt_msg_norm"),
        "prompt_to_original_msg_norm": scalar("prompt_to_original_msg_norm"),
        "node_to_prompt_msg_norm": scalar("node_to_prompt_msg_norm"),
        "prompt_to_original_update_norm": scalar("prompt_to_original_update_norm"),
        "correction_norm": scalar("prompt_to_original_update_norm"),
        "node_to_prompt_update_norm": scalar("node_to_prompt_update_norm"),
        "raw_prompt_update_norm": scalar("raw_prompt_update_norm"),
        "unbounded_prompt_update_norm": scalar("unbounded_prompt_update_norm"),
        "prompt_update_clip_ratio": scalar("prompt_update_clip_ratio"),
        "bounded_correction_clip_ratio": scalar("prompt_update_clip_ratio"),
        "prompt_gate_mean": scalar("prompt_gate_mean"),
        "prompt_gate_node_to_prompt_mean": scalar("prompt_gate_node_to_prompt_mean"),
        "prompt_gate_prompt_to_node_mean": scalar("prompt_gate_prompt_to_node_mean"),
        "prompt_gate_by_layer": gate_by_layer,
        "adapted_branch_delta_norm": scalar("adapted_branch_delta_norm"),
        "prompt_message_scale": scalar("prompt_message_scale"),
        "prompt_message_norm": str(aux.get("prompt_message_norm", "")),
        "bounded_prompt_update": scalar("bounded_prompt_update"),
        "max_prompt_update_norm": scalar("max_prompt_update_norm"),
        "prompt_update_bound_mode": str(aux.get("prompt_update_bound_mode", "")),
        "receiver_version": str(aux.get("receiver_version", "")),
        "prompt_fusion": str(aux.get("prompt_fusion", "")),
        "prompt_slot_head_count": scalar("prompt_slot_head_count"),
        "prompt_receiver_gate_mean": scalar("prompt_receiver_gate_mean"),
        "prototype_direction_strength_mean": scalar("prototype_direction_strength_mean"),
        "prototype_direction_normalize": scalar("prototype_direction_normalize"),
        "pool_only_prompt_update": scalar("pool_only_prompt_update"),
        "zero_init_prompt_messages": scalar("zero_init_prompt_messages"),
    }


def _forward_prompt_graph(
    *,
    model: FaithfulGP2F,
    prompt_graph_module: PromptGraphModuleP1 | None,
    z: torch.Tensor,
    edge_index: torch.Tensor,
    train_mask: torch.Tensor,
    edge_scale_multiplier: float | torch.Tensor = 1.0,
) -> tuple[dict[str, Any], dict[str, Any]]:
    h_pre = model.encode_frozen(z, edge_index)
    if prompt_graph_module is None:
        prompt_out = _default_prompt_graph_out(z, edge_index)
    else:
        prompt_out = prompt_graph_module(
            z=z,
            h_pre=h_pre.detach(),
            edge_index=edge_index,
            train_mask=train_mask,
            edge_scale_multiplier=edge_scale_multiplier,
        )
    forward_kwargs: dict[str, Any] = {
        "h_pre": h_pre,
        "adapted_x": prompt_out["adapted_x"],
        "adapted_edge_index": prompt_out["adapted_edge_index"],
        "adapted_edge_weight": prompt_out["adapted_edge_weight"],
        "return_aux": True,
    }
    if isinstance(model, PromptAwareGP2F):
        forward_kwargs["adapted_edge_type"] = prompt_out.get("adapted_edge_type")
        forward_kwargs["prompt_update_mask"] = prompt_out.get("pool_mask")
    model_out = model.forward_with_h_pre(z, edge_index, **forward_kwargs)
    model_out["h_pre_shared"] = h_pre
    return model_out, prompt_out


def _forward_no_prompt_with_h_pre(
    *,
    model: FaithfulGP2F,
    z: torch.Tensor,
    edge_index: torch.Tensor,
    h_pre: torch.Tensor,
) -> dict[str, Any]:
    prompt_out = _default_prompt_graph_out(z, edge_index)
    forward_kwargs: dict[str, Any] = {
        "h_pre": h_pre,
        "adapted_x": prompt_out["adapted_x"],
        "adapted_edge_index": prompt_out["adapted_edge_index"],
        "adapted_edge_weight": prompt_out["adapted_edge_weight"],
        "return_aux": True,
    }
    if isinstance(model, PromptAwareGP2F):
        forward_kwargs["adapted_edge_type"] = prompt_out.get("adapted_edge_type")
        forward_kwargs["prompt_update_mask"] = prompt_out.get("pool_mask")
    return model.forward_with_h_pre(z, edge_index, **forward_kwargs)


def _forward_prompt_graph_with_message_scale(
    *,
    model: FaithfulGP2F,
    z: torch.Tensor,
    edge_index: torch.Tensor,
    h_pre: torch.Tensor,
    prompt_out: dict[str, Any],
    message_scale: float,
) -> dict[str, Any]:
    old_scale: float | None = None
    if isinstance(model, PromptAwareGP2F):
        old_scale = float(model.prompt_message_scale)
        model.prompt_message_scale = float(message_scale)
    try:
        forward_kwargs: dict[str, Any] = {
            "h_pre": h_pre,
            "adapted_x": prompt_out["adapted_x"],
            "adapted_edge_index": prompt_out["adapted_edge_index"],
            "adapted_edge_weight": prompt_out["adapted_edge_weight"],
            "return_aux": True,
        }
        if isinstance(model, PromptAwareGP2F):
            forward_kwargs["adapted_edge_type"] = prompt_out.get("adapted_edge_type")
            forward_kwargs["prompt_update_mask"] = prompt_out.get("pool_mask")
        return model.forward_with_h_pre(z, edge_index, **forward_kwargs)
    finally:
        if old_scale is not None and isinstance(model, PromptAwareGP2F):
            model.prompt_message_scale = old_scale


def _acceptance_supervision_loss(
    *,
    prompt_out: dict[str, Any],
    logits_on: torch.Tensor,
    logits_off: torch.Tensor,
    labels: torch.Tensor,
    train_mask: torch.Tensor,
    positive_margin: float = 0.0,
    negative_margin: float = 0.0,
    balance_targets: bool = True,
    signal: str = "ce_delta",
) -> tuple[torch.Tensor, dict[str, float]]:
    """Train-only supervision for whether a pool node should receive prompt.

    Targets are built only from train-pool nodes by comparing prompt-on and
    prompt-off predictions. Validation/test labels are never inspected here.
    """

    aux = prompt_out.get("aux", {})
    pool_mask = prompt_out.get("pool_mask")
    gate_logits = aux.get("pool_acceptance_logit")
    edge_scale = prompt_out.get("edge_scale")
    fallback = edge_scale.new_tensor(0.0) if isinstance(edge_scale, torch.Tensor) else torch.tensor(0.0)
    empty_stats = {
        "prompt_acceptance_supervision": 0.0,
        "acceptance_supervised_count": 0.0,
        "acceptance_positive_count": 0.0,
        "acceptance_negative_count": 0.0,
        "acceptance_ignored_count": 0.0,
        "acceptance_target_mean": 0.0,
        "acceptance_score_delta_mean": 0.0,
        "ce_delta_positive_ratio_train": 0.0,
        "ce_delta_negative_ratio_train": 0.0,
        "prompt_helpful_acceptance_precision_train": 0.0,
        "prompt_harmful_rejection_precision_train": 0.0,
    }
    if not (isinstance(pool_mask, torch.Tensor) and isinstance(gate_logits, torch.Tensor) and gate_logits.numel() > 0):
        return fallback, empty_stats
    pool_idx = torch.where(pool_mask.bool())[0]
    if pool_idx.numel() != gate_logits.numel():
        raise ValueError("pool_acceptance_logit rows must match pool_mask true count")
    train_pool = train_mask.to(device=pool_mask.device, dtype=torch.bool)[pool_idx]
    if int(train_pool.sum().item()) == 0:
        return fallback, empty_stats

    pool_idx_train = pool_idx[train_pool]
    y = labels.to(device=pool_mask.device)[pool_idx_train]
    on = logits_on.detach()[pool_idx_train]
    off = logits_off.detach()[pool_idx_train]
    on_pred = on.argmax(dim=-1)
    off_pred = off.argmax(dim=-1)
    on_correct = on_pred == y
    off_correct = off_pred == y

    positive_margin = max(float(positive_margin), 0.0)
    negative_margin = max(float(negative_margin), 0.0)
    if signal == "true_logit_delta":
        score_delta = on.gather(1, y.view(-1, 1)).squeeze(1) - off.gather(1, y.view(-1, 1)).squeeze(1)
    elif signal == "ce_delta":
        off_ce = F.cross_entropy(off, y, reduction="none")
        on_ce = F.cross_entropy(on, y, reduction="none")
        score_delta = off_ce - on_ce
    else:
        raise ValueError(
            f"Unsupported acceptance supervision signal={signal!r}; expected 'ce_delta' or 'true_logit_delta'"
        )
    positive = (on_correct & ~off_correct) | (score_delta > positive_margin)
    negative = (off_correct & ~on_correct) | (score_delta < -negative_margin)
    valid = positive ^ negative
    ignored = ~valid
    train_pool_count = max(1, int(train_pool.sum().item()))
    if int(valid.sum().item()) == 0:
        stats = dict(empty_stats)
        stats["acceptance_ignored_count"] = float(ignored.sum().item())
        stats["ce_delta_positive_ratio_train"] = float(positive.sum().item() / train_pool_count)
        stats["ce_delta_negative_ratio_train"] = float(negative.sum().item() / train_pool_count)
        return fallback, stats

    targets = positive[valid].to(dtype=gate_logits.dtype)
    selected_logits = gate_logits[train_pool][valid]
    if balance_targets and bool((targets > 0.5).any()) and bool((targets < 0.5).any()):
        pos_count = targets.sum().clamp_min(1.0)
        neg_count = (1.0 - targets).sum().clamp_min(1.0)
        pos_weight = neg_count / pos_count
        loss = F.binary_cross_entropy_with_logits(selected_logits, targets, pos_weight=pos_weight)
    else:
        loss = F.binary_cross_entropy_with_logits(selected_logits, targets)
    selected_gate = torch.sigmoid(selected_logits)
    accepted = selected_gate >= 0.5
    rejected = ~accepted
    helpful = targets > 0.5
    harmful = targets < 0.5
    helpful_acceptance_precision = (
        float((helpful & accepted).float().sum().item() / max(1, int(accepted.sum().item())))
        if int(accepted.sum().item()) > 0
        else 0.0
    )
    harmful_rejection_precision = (
        float((harmful & rejected).float().sum().item() / max(1, int(rejected.sum().item())))
        if int(rejected.sum().item()) > 0
        else 0.0
    )
    stats = {
        "prompt_acceptance_supervision": float(loss.detach().item()),
        "acceptance_supervised_count": float(valid.sum().item()),
        "acceptance_positive_count": float((positive & valid).sum().item()),
        "acceptance_negative_count": float((negative & valid).sum().item()),
        "acceptance_ignored_count": float(ignored.sum().item()),
        "acceptance_target_mean": float(targets.mean().detach().item()),
        "acceptance_score_delta_mean": float(score_delta[valid].mean().detach().item()),
        "ce_delta_positive_ratio_train": float(positive.sum().item() / train_pool_count),
        "ce_delta_negative_ratio_train": float(negative.sum().item() / train_pool_count),
        "prompt_helpful_acceptance_precision_train": helpful_acceptance_precision,
        "prompt_harmful_rejection_precision_train": harmful_rejection_precision,
    }
    return loss, stats


def _benefit_supervision_loss(
    *,
    prompt_out: dict[str, Any],
    logits_on: torch.Tensor,
    logits_off: torch.Tensor,
    labels: torch.Tensor,
    train_mask: torch.Tensor,
    margin: float,
    balance_targets: bool,
    label_strategy: str = "margin",
    quantile: float = 0.20,
    eps: float = 0.0,
) -> tuple[torch.Tensor, dict[str, float]]:
    aux = prompt_out.get("aux", {})
    pool_mask = prompt_out.get("pool_mask")
    benefit_logits = aux.get("benefit_gate_logit")
    edge_scale = prompt_out.get("edge_scale")
    fallback = edge_scale.new_tensor(0.0) if isinstance(edge_scale, torch.Tensor) else torch.tensor(0.0)
    empty_stats = {
        "prompt_benefit_supervision": 0.0,
        "benefit_supervised_count": 0.0,
        "benefit_positive_count": 0.0,
        "benefit_negative_count": 0.0,
        "benefit_ignored_count": 0.0,
        "mean_delta_ce_train_pool": 0.0,
        "positive_delta_ratio_train_pool": 0.0,
        "benefit_weight_delta_corr_train": 0.0,
        "benefit_label_strategy_quantile": 0.0,
        "benefit_quantile_fraction": 0.0,
        "benefit_positive_threshold": 0.0,
        "benefit_negative_threshold": 0.0,
        "benefit_delta_eps": 0.0,
    }
    if not (
        isinstance(pool_mask, torch.Tensor)
        and isinstance(benefit_logits, torch.Tensor)
        and benefit_logits.numel() > 0
    ):
        return fallback, empty_stats
    pool_idx = aux.get("pool_idx")
    if not isinstance(pool_idx, torch.Tensor):
        pool_idx = torch.where(pool_mask.bool())[0]
    if pool_idx.numel() != benefit_logits.size(0):
        raise ValueError("benefit_gate_logit rows must match pool index count")
    train_pool = train_mask.to(device=pool_mask.device, dtype=torch.bool)[pool_idx]
    if int(train_pool.sum().item()) == 0:
        return fallback, empty_stats

    pool_idx_train = pool_idx[train_pool]
    y = labels.to(device=pool_mask.device)[pool_idx_train]
    on = logits_on.detach()[pool_idx_train]
    off = logits_off.detach()[pool_idx_train]
    off_ce = F.cross_entropy(off, y, reduction="none")
    on_ce = F.cross_entropy(on, y, reduction="none")
    delta_ce = off_ce - on_ce
    margin = max(float(margin), 0.0)
    eps = max(float(eps), 0.0)
    strategy = str(label_strategy).lower()
    stats = dict(empty_stats)
    stats["benefit_label_strategy_quantile"] = float(strategy in {"quantile", "hybrid_quantile_margin"})
    stats["benefit_quantile_fraction"] = float(quantile)
    stats["benefit_delta_eps"] = float(eps)
    if strategy == "margin":
        positive = delta_ce > margin
        negative = delta_ce < -margin
        positive_threshold = delta_ce.new_tensor(margin)
        negative_threshold = delta_ce.new_tensor(-margin)
    elif strategy in {"quantile", "hybrid_quantile_margin"}:
        fraction = min(max(float(quantile), 0.0), 0.5)
        if delta_ce.numel() < 2 or fraction <= 0.0:
            positive = delta_ce > margin
            negative = delta_ce < -margin
            positive_threshold = delta_ce.new_tensor(margin)
            negative_threshold = delta_ce.new_tensor(-margin)
        else:
            k = min(max(1, int(math.ceil(float(delta_ce.numel()) * fraction))), delta_ce.numel() // 2)
            sorted_delta, sorted_idx = torch.sort(delta_ce)
            negative_threshold = sorted_delta[k - 1]
            positive_threshold = sorted_delta[-k]
            positive = torch.zeros_like(delta_ce, dtype=torch.bool)
            negative = torch.zeros_like(delta_ce, dtype=torch.bool)
            negative[sorted_idx[:k]] = True
            positive[sorted_idx[-k:]] = True
        if strategy == "hybrid_quantile_margin":
            positive = positive & (delta_ce > eps)
            negative = negative & (delta_ce < -eps)
    else:
        raise ValueError(
            f"Unsupported benefit supervision label_strategy={label_strategy!r}; "
            "expected 'margin', 'quantile', or 'hybrid_quantile_margin'"
        )
    valid = positive ^ negative
    ignored = ~valid
    train_count = max(1, int(train_pool.sum().item()))
    stats["mean_delta_ce_train_pool"] = float(delta_ce.mean().detach().item())
    stats["positive_delta_ratio_train_pool"] = float(positive.float().mean().detach().item())
    stats["benefit_ignored_count"] = float(ignored.sum().item())
    stats["benefit_positive_threshold"] = float(positive_threshold.detach().item())
    stats["benefit_negative_threshold"] = float(negative_threshold.detach().item())
    if int(valid.sum().item()) == 0:
        return fallback, stats

    selected_logits = benefit_logits[train_pool][valid].reshape(-1)
    targets = positive[valid].to(dtype=selected_logits.dtype).unsqueeze(-1).expand(-1, benefit_logits.size(1)).reshape(-1)
    if balance_targets and bool((targets > 0.5).any()) and bool((targets < 0.5).any()):
        pos_count = targets.sum().clamp_min(1.0)
        neg_count = (1.0 - targets).sum().clamp_min(1.0)
        pos_weight = neg_count / pos_count
        loss = F.binary_cross_entropy_with_logits(selected_logits, targets, pos_weight=pos_weight)
    else:
        loss = F.binary_cross_entropy_with_logits(selected_logits, targets)

    benefit_weight = torch.sigmoid(benefit_logits[train_pool]).mean(dim=-1)
    valid_weight = benefit_weight[valid]
    valid_delta = delta_ce[valid]
    if valid_weight.numel() > 1 and float(valid_weight.std(unbiased=False).item()) > 1e-12:
        centered_weight = valid_weight - valid_weight.mean()
        centered_delta = valid_delta - valid_delta.mean()
        corr = (centered_weight * centered_delta).mean() / (
            centered_weight.pow(2).mean().sqrt() * centered_delta.pow(2).mean().sqrt()
        ).clamp_min(1e-12)
        stats["benefit_weight_delta_corr_train"] = float(corr.detach().item())
    stats.update(
        {
            "prompt_benefit_supervision": float(loss.detach().item()),
            "benefit_supervised_count": float(valid.sum().item()),
            "benefit_positive_count": float(positive[valid].sum().item()),
            "benefit_negative_count": float(negative[valid].sum().item()),
            "benefit_ignored_count": float(ignored.sum().item()),
            "positive_delta_ratio_train_pool": float(positive.sum().item() / train_count),
        }
    )
    return loss, stats


def _prompt_correction_losses(
    *,
    prompt_out: dict[str, Any],
    logits_on: torch.Tensor,
    logits_off: torch.Tensor,
    labels: torch.Tensor,
    train_mask: torch.Tensor,
    eps: float,
    target: str = "delta_prob_to_label",
) -> tuple[torch.Tensor, torch.Tensor, dict[str, float]]:
    aux = prompt_out.get("aux", {})
    pool_mask = prompt_out.get("pool_mask")
    edge_scale = prompt_out.get("edge_scale")
    fallback = edge_scale.new_tensor(0.0) if isinstance(edge_scale, torch.Tensor) else logits_on.new_tensor(0.0)
    empty_stats = {
        "prompt_correction_loss": 0.0,
        "prompt_anti_harm_loss": 0.0,
        "prompt_correction_supervised_count": 0.0,
        "prompt_correction_harmful_count": 0.0,
        "prompt_correction_target_norm": 0.0,
        "prompt_delta_logit_norm": 0.0,
        "prompt_delta_logit_label_alignment": 0.0,
        "harmful_prompt_delta_ratio": 0.0,
    }
    if not isinstance(pool_mask, torch.Tensor):
        return fallback, fallback, empty_stats
    pool_idx = aux.get("pool_idx")
    if not isinstance(pool_idx, torch.Tensor):
        pool_idx = torch.where(pool_mask.bool())[0]
    if pool_idx.numel() == 0:
        return fallback, fallback, empty_stats
    train_pool = train_mask.to(device=pool_mask.device, dtype=torch.bool)[pool_idx]
    if int(train_pool.sum().item()) == 0:
        return fallback, fallback, empty_stats

    pool_idx_train = pool_idx[train_pool]
    y = labels.to(device=logits_on.device)[pool_idx_train]
    on = logits_on[pool_idx_train]
    off = logits_off.detach()[pool_idx_train]
    off_ce = F.cross_entropy(off, y, reduction="none")
    on_ce = F.cross_entropy(on.detach(), y, reduction="none")
    delta_ce = off_ce - on_ce
    eps = max(float(eps), 0.0)
    positive = delta_ce > eps
    harmful = delta_ce < -eps

    delta_logits = on - off
    off_prob = F.softmax(off, dim=-1)
    if target == "delta_prob_to_label":
        target_delta = F.one_hot(y, num_classes=logits_on.size(-1)).to(dtype=logits_on.dtype) - off_prob
    else:
        raise ValueError(
            f"Unsupported prompt_correction_target={target!r}; expected 'delta_prob_to_label'"
        )

    weight = torch.ones_like(delta_ce, dtype=logits_on.dtype)
    benefit_gate = aux.get("benefit_gate")
    if isinstance(benefit_gate, torch.Tensor) and benefit_gate.numel() > 0:
        benefit_gate = benefit_gate.to(device=logits_on.device, dtype=logits_on.dtype)
        if benefit_gate.size(0) == pool_idx.numel():
            weight = weight * benefit_gate[train_pool].mean(dim=-1).detach()
    receive_gate = aux.get("effective_receive_gate")
    if isinstance(receive_gate, torch.Tensor) and receive_gate.numel() > 0:
        receive_gate = receive_gate.to(device=logits_on.device, dtype=logits_on.dtype)
        if receive_gate.size(0) == pool_idx.numel():
            weight = weight * receive_gate[train_pool].detach()

    if int(positive.sum().item()) > 0:
        pos_weight = weight[positive].clamp_min(1e-6)
        pos_loss = (delta_logits[positive] - target_delta[positive]).pow(2).mean(dim=-1)
        correction_loss = (pos_loss * pos_weight).sum() / pos_weight.sum().clamp_min(1e-6)
        target_norm = target_delta[positive].norm(dim=-1).mean()
        delta_norm = delta_logits[positive].norm(dim=-1).mean()
        label_alignment = delta_logits[positive, y[positive]].mean()
    else:
        correction_loss = fallback
        target_norm = fallback
        delta_norm = fallback
        label_alignment = fallback

    if int(harmful.sum().item()) > 0:
        harmful_delta = delta_logits[harmful]
        harmful_y = y[harmful]
        true_class_increase = harmful_delta.gather(1, harmful_y.view(-1, 1)).squeeze(-1)
        wrong_mask = torch.ones_like(harmful_delta, dtype=torch.bool)
        wrong_mask.scatter_(1, harmful_y.view(-1, 1), False)
        wrong_class_increase = harmful_delta.masked_fill(~wrong_mask, float("-inf")).max(dim=-1).values
        harmful_score = F.relu(wrong_class_increase - true_class_increase)
        harm_weight = weight[harmful].clamp_min(1e-6)
        anti_harm_loss = (harmful_score * harm_weight).sum() / harm_weight.sum().clamp_min(1e-6)
        harmful_prompt_delta_ratio = float((harmful_score > 0.0).float().mean().detach().item())
    else:
        anti_harm_loss = fallback
        harmful_prompt_delta_ratio = 0.0

    stats = {
        "prompt_correction_loss": float(correction_loss.detach().item()),
        "prompt_anti_harm_loss": float(anti_harm_loss.detach().item()),
        "prompt_correction_supervised_count": float(positive.sum().item()),
        "prompt_correction_harmful_count": float(harmful.sum().item()),
        "prompt_correction_target_norm": float(target_norm.detach().item()),
        "prompt_delta_logit_norm": float(delta_norm.detach().item()),
        "prompt_delta_logit_label_alignment": float(label_alignment.detach().item()),
        "harmful_prompt_delta_ratio": harmful_prompt_delta_ratio,
    }
    return correction_loss, anti_harm_loss, stats


def _prompt_message_help_loss(
    *,
    prompt_out: dict[str, Any],
    logits_on: torch.Tensor,
    logits_off: torch.Tensor,
    labels: torch.Tensor,
    train_mask: torch.Tensor,
    margin: float,
    class_balanced: bool = True,
    anti_harm_floor: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
    pool_mask = prompt_out.get("pool_mask")
    edge_scale = prompt_out.get("edge_scale")
    fallback = edge_scale.new_tensor(0.0) if isinstance(edge_scale, torch.Tensor) else logits_on.new_tensor(0.0)
    empty_stats: dict[str, Any] = {
        "prompt_message_help_loss": 0.0,
        "class_balanced_mean_delta_ce": 0.0,
        "class_balanced_positive_delta_ratio": 0.0,
        "message_help_supervised_count": 0.0,
        "message_help_class_count": 0.0,
        "message_help_margin": float(margin),
        "message_help_class_balanced": float(class_balanced),
        "prompt_class_anti_harm_loss": 0.0,
        "prompt_class_anti_harm_floor": float(anti_harm_floor),
        "delta_ce_by_class_train_pool": {},
    }
    if not isinstance(pool_mask, torch.Tensor):
        return fallback, fallback, empty_stats

    train_pool_mask = train_mask.to(device=pool_mask.device, dtype=torch.bool) & pool_mask.bool()
    train_pool_idx = torch.where(train_pool_mask)[0]
    if train_pool_idx.numel() == 0:
        return fallback, fallback, empty_stats

    y = labels.to(device=logits_on.device)[train_pool_idx]
    ce_off = F.cross_entropy(logits_off.detach()[train_pool_idx], y, reduction="none")
    ce_on = F.cross_entropy(logits_on[train_pool_idx], y, reduction="none")
    delta_ce = ce_off - ce_on
    per_node_loss = F.relu(float(margin) - delta_ce)
    positive = delta_ce > 0.0

    class_losses: list[torch.Tensor] = []
    class_mean_delta: list[torch.Tensor] = []
    class_positive_ratio: list[torch.Tensor] = []
    class_anti_harm_losses: list[torch.Tensor] = []
    by_class: dict[str, dict[str, float]] = {}
    for class_id in torch.unique(y.detach()).tolist():
        class_mask = y == int(class_id)
        if not bool(class_mask.any()):
            continue
        class_losses.append(per_node_loss[class_mask].mean())
        mean_delta = delta_ce[class_mask].mean()
        class_mean_delta.append(mean_delta.detach())
        class_anti_harm_losses.append(F.relu(float(anti_harm_floor) - mean_delta))
        class_positive_ratio.append(positive[class_mask].to(dtype=logits_on.dtype).detach().mean())
        by_class[str(int(class_id))] = {
            "count": float(class_mask.sum().item()),
            "mean_delta_ce": float(delta_ce[class_mask].detach().mean().item()),
            "positive_delta_ratio": float(positive[class_mask].float().detach().mean().item()),
            "mean_ce_no_prompt": float(ce_off[class_mask].detach().mean().item()),
            "mean_ce_prompt": float(ce_on[class_mask].detach().mean().item()),
        }

    if class_balanced and class_losses:
        loss = torch.stack(class_losses).mean()
        balanced_delta = torch.stack(class_mean_delta).mean()
        balanced_positive = torch.stack(class_positive_ratio).mean()
        class_count = float(len(class_losses))
        anti_harm_loss = torch.stack(class_anti_harm_losses).mean()
    else:
        loss = per_node_loss.mean()
        balanced_delta = delta_ce.detach().mean()
        balanced_positive = positive.to(dtype=logits_on.dtype).detach().mean()
        class_count = float(len(class_losses))
        anti_harm_loss = F.relu(float(anti_harm_floor) - delta_ce.mean())

    stats: dict[str, Any] = {
        "prompt_message_help_loss": float(loss.detach().item()),
        "class_balanced_mean_delta_ce": float(balanced_delta.item()),
        "class_balanced_positive_delta_ratio": float(balanced_positive.item()),
        "message_help_supervised_count": float(train_pool_idx.numel()),
        "message_help_class_count": class_count,
        "message_help_margin": float(margin),
        "message_help_class_balanced": float(class_balanced),
        "prompt_class_anti_harm_loss": float(anti_harm_loss.detach().item()),
        "prompt_class_anti_harm_floor": float(anti_harm_floor),
        "delta_ce_by_class_train_pool": by_class,
    }
    return loss, anti_harm_loss, stats


def _query_proto_alignment_loss(
    *,
    h_on: torch.Tensor,
    h_off: torch.Tensor,
    labels: torch.Tensor,
    support_mask: torch.Tensor,
    query_mask: torch.Tensor,
    pool_mask: torch.Tensor,
    margin: float = 0.005,
    normalize: bool = True,
    class_balanced: bool = True,
) -> tuple[torch.Tensor, dict[str, Any]]:
    fallback = h_on.new_tensor(0.0)
    support_mask = support_mask.to(device=h_on.device, dtype=torch.bool)
    query_mask = query_mask.to(device=h_on.device, dtype=torch.bool)
    pool_mask = pool_mask.to(device=h_on.device, dtype=torch.bool)
    query_pool = query_mask & pool_mask
    empty_stats: dict[str, Any] = {
        "query_proto_alignment_loss": 0.0,
        "query_proto_supervised_count": float(query_pool.sum().item()),
        "query_proto_class_count": 0.0,
        "query_proto_mean_delta_dist": 0.0,
        "query_proto_positive_ratio": 0.0,
        "query_proto_margin": float(margin),
        "query_proto_class_balanced": float(class_balanced),
        "query_proto_by_class": {},
    }
    if int(query_pool.sum().item()) == 0 or int(support_mask.sum().item()) == 0:
        return fallback, empty_stats

    labels = labels.to(device=h_on.device)
    feat_on = F.normalize(h_on, dim=-1, eps=1e-12) if normalize else h_on
    feat_off = F.normalize(h_off.detach(), dim=-1, eps=1e-12) if normalize else h_off.detach()
    support_feat = F.normalize(h_off.detach(), dim=-1, eps=1e-12) if normalize else h_off.detach()

    class_losses: list[torch.Tensor] = []
    class_delta: list[torch.Tensor] = []
    class_positive: list[torch.Tensor] = []
    by_class: dict[str, dict[str, float]] = {}
    for class_id in torch.unique(labels[support_mask | query_pool].detach()).tolist():
        class_id = int(class_id)
        support_class = support_mask & (labels == class_id)
        query_class = query_pool & (labels == class_id)
        if not bool(support_class.any()) or not bool(query_class.any()):
            continue
        proto = support_feat[support_class].mean(dim=0)
        proto = F.normalize(proto, dim=0, eps=1e-12) if normalize else proto
        dist_off = 1.0 - F.cosine_similarity(feat_off[query_class], proto.unsqueeze(0), dim=-1, eps=1e-12)
        dist_on = 1.0 - F.cosine_similarity(feat_on[query_class], proto.unsqueeze(0), dim=-1, eps=1e-12)
        delta_dist = dist_off - dist_on
        loss = F.relu(float(margin) - delta_dist)
        class_losses.append(loss.mean())
        class_delta.append(delta_dist.detach().mean())
        class_positive.append((delta_dist > 0.0).to(dtype=h_on.dtype).detach().mean())
        by_class[str(class_id)] = {
            "count": float(query_class.sum().item()),
            "mean_delta_dist": float(delta_dist.detach().mean().item()),
            "positive_delta_ratio": float((delta_dist > 0.0).float().detach().mean().item()),
        }

    if not class_losses:
        return fallback, empty_stats
    if class_balanced:
        loss = torch.stack(class_losses).mean()
        mean_delta = torch.stack(class_delta).mean()
        positive_ratio = torch.stack(class_positive).mean()
    else:
        loss = torch.stack(class_losses).mean()
        mean_delta = torch.stack(class_delta).mean()
        positive_ratio = torch.stack(class_positive).mean()
    stats = {
        "query_proto_alignment_loss": float(loss.detach().item()),
        "query_proto_supervised_count": float(query_pool.sum().item()),
        "query_proto_class_count": float(len(class_losses)),
        "query_proto_mean_delta_dist": float(mean_delta.detach().item()),
        "query_proto_positive_ratio": float(positive_ratio.detach().item()),
        "query_proto_margin": float(margin),
        "query_proto_class_balanced": float(class_balanced),
        "query_proto_by_class": by_class,
    }
    return loss, stats


def _utility_receive_gate_loss(
    *,
    prompt_out: dict[str, Any],
    logits_on: torch.Tensor,
    logits_off: torch.Tensor,
    labels: torch.Tensor,
    train_mask: torch.Tensor,
    quantile: float = 0.20,
    eps: float = 1e-4,
    class_balanced: bool = True,
) -> tuple[torch.Tensor, dict[str, Any]]:
    aux = prompt_out.get("aux", {})
    pool_mask = prompt_out.get("pool_mask")
    gate = aux.get("utility_receive_gate")
    edge_scale = prompt_out.get("edge_scale")
    fallback = edge_scale.new_tensor(0.0) if isinstance(edge_scale, torch.Tensor) else logits_on.new_tensor(0.0)
    empty_stats: dict[str, Any] = {
        "utility_receive_gate_loss": 0.0,
        "utility_gate_supervised_count": 0.0,
        "utility_gate_positive_count": 0.0,
        "utility_gate_negative_count": 0.0,
        "utility_gate_ignored_count": 0.0,
        "utility_gate_target_mean": 0.0,
        "utility_gate_delta_corr_train": 0.0,
        "utility_gate_positive_threshold_mean": 0.0,
        "utility_gate_negative_threshold_mean": 0.0,
        "utility_gate_quantile": float(quantile),
        "utility_gate_eps": float(eps),
        "utility_gate_class_balanced": float(class_balanced),
    }
    if not (
        isinstance(pool_mask, torch.Tensor)
        and isinstance(gate, torch.Tensor)
        and gate.numel() > 0
    ):
        return fallback, empty_stats

    pool_idx = aux.get("pool_idx")
    if not isinstance(pool_idx, torch.Tensor):
        pool_idx = torch.where(pool_mask.bool())[0]
    if pool_idx.numel() != gate.numel():
        raise ValueError("utility_receive_gate rows must match pool index count")
    train_pool = train_mask.to(device=pool_mask.device, dtype=torch.bool)[pool_idx]
    if int(train_pool.sum().item()) < 2:
        return fallback, empty_stats

    y_all = labels.to(device=logits_on.device)[pool_idx]
    train_pool_idx = pool_idx[train_pool]
    y = y_all[train_pool]
    ce_off = F.cross_entropy(logits_off.detach()[train_pool_idx], y, reduction="none")
    ce_on = F.cross_entropy(logits_on.detach()[train_pool_idx], y, reduction="none")
    delta_ce = ce_off - ce_on
    gate_train = gate[train_pool].clamp(min=1e-6, max=1.0 - 1e-6)
    fraction = min(max(float(quantile), 0.0), 0.5)
    eps = max(float(eps), 0.0)

    selected_masks: list[torch.Tensor] = []
    target_values: list[torch.Tensor] = []
    positive_thresholds: list[torch.Tensor] = []
    negative_thresholds: list[torch.Tensor] = []
    classes = torch.unique(y.detach()).tolist() if class_balanced else [-1]
    for class_id in classes:
        class_mask = torch.ones_like(y, dtype=torch.bool) if int(class_id) == -1 else y == int(class_id)
        rows = torch.where(class_mask)[0]
        if rows.numel() < 2 or fraction <= 0.0:
            continue
        k = min(max(1, int(math.ceil(float(rows.numel()) * fraction))), int(rows.numel()) // 2)
        if k <= 0:
            continue
        class_delta = delta_ce[rows]
        sorted_delta, order = torch.sort(class_delta)
        neg_rows = rows[order[:k]]
        pos_rows = rows[order[-k:]]
        pos_keep = delta_ce[pos_rows] > eps
        neg_keep = delta_ce[neg_rows] < -eps
        selected = torch.zeros_like(y, dtype=torch.bool)
        target = torch.zeros_like(delta_ce)
        if bool(pos_keep.any()):
            selected[pos_rows[pos_keep]] = True
            target[pos_rows[pos_keep]] = 1.0
        if bool(neg_keep.any()):
            selected[neg_rows[neg_keep]] = True
            target[neg_rows[neg_keep]] = 0.0
        if bool(selected.any()):
            selected_masks.append(selected)
            target_values.append(target)
            negative_thresholds.append(sorted_delta[k - 1].detach())
            positive_thresholds.append(sorted_delta[-k].detach())

    if not selected_masks:
        return fallback, empty_stats
    supervised = torch.stack(selected_masks, dim=0).any(dim=0)
    target = torch.stack(target_values, dim=0).sum(dim=0).clamp(max=1.0)
    if int(supervised.sum().item()) == 0:
        return fallback, empty_stats

    loss = F.binary_cross_entropy(gate_train[supervised], target[supervised])
    positive = (target[supervised] > 0.5)
    negative = ~positive
    stats = {
        "utility_receive_gate_loss": float(loss.detach().item()),
        "utility_gate_supervised_count": float(supervised.sum().item()),
        "utility_gate_positive_count": float(positive.sum().item()),
        "utility_gate_negative_count": float(negative.sum().item()),
        "utility_gate_ignored_count": float((~supervised).sum().item()),
        "utility_gate_target_mean": float(target[supervised].detach().mean().item()),
        "utility_gate_delta_corr_train": _pearson_corr(gate_train.detach(), delta_ce.detach()),
        "utility_gate_positive_threshold_mean": (
            float(torch.stack(positive_thresholds).mean().item()) if positive_thresholds else 0.0
        ),
        "utility_gate_negative_threshold_mean": (
            float(torch.stack(negative_thresholds).mean().item()) if negative_thresholds else 0.0
        ),
        "utility_gate_quantile": float(quantile),
        "utility_gate_eps": float(eps),
        "utility_gate_class_balanced": float(class_balanced),
    }
    return loss, stats


def _edge_scale_multiplier(epoch: int, prompt_graph_cfg: dict[str, Any]) -> float:
    warmup_epochs = int(prompt_graph_cfg.get("edge_scale_warmup_epochs", 0))
    start = float(prompt_graph_cfg.get("edge_scale_warmup_start", 1.0 if warmup_epochs <= 0 else 0.0))
    start = min(max(start, 0.0), 1.0)
    if warmup_epochs <= 0:
        return 1.0
    progress = min(1.0, max(0.0, float(epoch) / float(warmup_epochs)))
    return start + (1.0 - start) * progress


def _message_scale_grid(config: dict[str, Any]) -> list[float]:
    raw = config.get("prompt_aware", {}).get("message_scale_grid")
    if raw is None:
        return []
    if isinstance(raw, str):
        values = [float(piece.strip()) for piece in raw.split(",") if piece.strip()]
    else:
        values = [float(value) for value in raw]
    unique: list[float] = []
    seen: set[float] = set()
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        unique.append(value)
    return unique


def _scale_label(value: float) -> str:
    return str(float(value)).replace("-", "m").replace(".", "p")


def _candidate_monitor_value(result: dict[str, Any], metric: str) -> float:
    if metric in result.get("best", {}):
        return float(result["best"].get(metric, 0.0))
    if metric in result.get("final", {}):
        return float(result["final"].get(metric, 0.0))
    return float(result["best"].get("val_acc", 0.0))


def _select_message_scale_candidate(
    candidates: list[dict[str, Any]],
    *,
    metric: str,
) -> dict[str, Any]:
    if not candidates:
        raise RuntimeError("message_scale_grid selection requires at least one candidate")
    minimize = metric in {"train_loss", "total", "cls", "edge_l1", "prompt_balance"}
    ordered = sorted(
        candidates,
        key=lambda item: (
            _candidate_monitor_value(item, metric),
            -float(item.get("candidate_message_scale", 0.0)),
        ),
        reverse=not minimize,
    )
    return ordered[0]


def _compact_scale_candidate(result: dict[str, Any], metric: str) -> dict[str, Any]:
    return {
        "message_scale": float(result.get("candidate_message_scale", result["best"].get("prompt_message_scale", 0.0))),
        "selection_metric": metric,
        "selection_value": _candidate_monitor_value(result, metric),
        "best_epoch": result["best"].get("best_epoch", 0.0),
        "best_val_acc": result["best"].get("val_acc", 0.0),
        "best_val_macro_f1": result["best"].get("val_macro_f1", 0.0),
        "best_test_acc": result["best"].get("test_acc", 0.0),
        "best_test_macro_f1": result["best"].get("test_macro_f1", 0.0),
        "final_test_acc": result["final"].get("test_acc", 0.0),
        "final_test_macro_f1": result["final"].get("test_macro_f1", 0.0),
        "run_dir": result.get("run_dir", ""),
    }


@torch.no_grad()
def _init_equivalence(
    *,
    model: FaithfulGP2F,
    prompt_graph_module: PromptGraphModuleP1 | None,
    input_aligner: InputAligner,
    x: torch.Tensor,
    edge_index: torch.Tensor,
    train_mask: torch.Tensor,
) -> dict[str, float]:
    model.eval()
    input_aligner.eval()
    if prompt_graph_module is not None:
        prompt_graph_module.eval()
    z = input_aligner(x)
    baseline = model(z, edge_index, return_aux=True)
    prompted_zero, prompt_out_zero = _forward_prompt_graph(
        model=model,
        prompt_graph_module=prompt_graph_module,
        z=z,
        edge_index=edge_index,
        train_mask=train_mask,
        edge_scale_multiplier=0.0,
    )
    prompted_full, _ = _forward_prompt_graph(
        model=model,
        prompt_graph_module=prompt_graph_module,
        z=z,
        edge_index=edge_index,
        train_mask=train_mask,
        edge_scale_multiplier=1.0,
    )
    return {
        "init_original_x_delta": float((prompt_out_zero["adapted_x"][: z.size(0)] - z).abs().max().item()),
        "init_logit_delta": float((prompted_zero["logits"] - baseline["logits"]).abs().max().item()),
        "init_logit_delta_full_edge_scale": float((prompted_full["logits"] - baseline["logits"]).abs().max().item()),
    }


@torch.no_grad()
def evaluate_prompt_graph(
    *,
    model: FaithfulGP2F,
    prompt_graph_module: PromptGraphModuleP1 | None,
    input_aligner: InputAligner,
    x: torch.Tensor,
    edge_index: torch.Tensor,
    labels: torch.Tensor,
    train_mask: torch.Tensor,
    val_mask: torch.Tensor,
    test_mask: torch.Tensor,
    num_classes: int,
    edge_scale_multiplier: float = 1.0,
) -> dict[str, Any]:
    model.eval()
    input_aligner.eval()
    if prompt_graph_module is not None:
        prompt_graph_module.eval()
    z = input_aligner(x)
    model_out, prompt_out = _forward_prompt_graph(
        model=model,
        prompt_graph_module=prompt_graph_module,
        z=z,
        edge_index=edge_index,
        train_mask=train_mask,
        edge_scale_multiplier=edge_scale_multiplier,
    )
    logits = model_out["logits"]
    branch_cosine = torch.nn.functional.cosine_similarity(model_out["h_pre"], model_out["h_adp"], dim=-1).mean()
    train = split_metrics(logits, labels, train_mask, num_classes=num_classes)
    val = split_metrics(logits, labels, val_mask, num_classes=num_classes)
    test = split_metrics(logits, labels, test_mask, num_classes=num_classes)
    return {
        "train_acc": train["acc"],
        "train_macro_f1": train["macro_f1"],
        "val_acc": val["acc"],
        "val_macro_f1": val["macro_f1"],
        "test_acc": test["acc"],
        "test_macro_f1": test["macro_f1"],
        "alpha": float(model_out["alpha"].detach().item()),
        "branch_cosine": float(branch_cosine.detach().item()),
        **_prompt_aware_diagnostics(model_out),
        **_prompt_graph_diagnostics(
            prompt_out,
            z=z,
            train_mask=train_mask,
            val_mask=val_mask,
            test_mask=test_mask,
            labels=labels,
            num_classes=num_classes,
        ),
    }


def _empty_prompt_message_utility(scale: float, reason: str) -> dict[str, Any]:
    return {
        "diagnostic": "prompt_message_utility",
        "message_scale": float(scale),
        "model_state": "final_after_training",
        "reason": reason,
        "train_pool_count": 0,
        "val_pool_count": 0,
        "test_pool_count": 0,
        "mean_delta_ce": 0.0,
        "median_delta_ce": 0.0,
        "positive_delta_ratio": 0.0,
        "mean_ce_no_prompt": 0.0,
        "mean_ce_prompt": 0.0,
        "mean_delta_ce_train_pool": 0.0,
        "positive_delta_ratio_train_pool": 0.0,
        "mean_delta_ce_val_pool": 0.0,
        "positive_delta_ratio_val_pool": 0.0,
        "mean_delta_ce_test_pool": 0.0,
        "positive_delta_ratio_test_pool": 0.0,
        "delta_ce_by_class": {},
        "delta_ce_by_prompt_slot": {},
        "delta_ce_by_acceptance_gate": {},
        "delta_ce_by_structural_score": {},
        "acceptance_delta_correlation": 0.0,
        "structural_score_delta_correlation": 0.0,
        "node_records": [],
    }


def _group_delta_stats(
    *,
    keys: torch.Tensor,
    delta_ce: torch.Tensor,
    positive: torch.Tensor,
) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    if keys.numel() == 0:
        return out
    for key in torch.unique(keys.detach()).tolist():
        key_int = int(key)
        mask = keys == key_int
        if not bool(mask.any()):
            continue
        out[str(key_int)] = {
            "count": int(mask.sum().item()),
            "mean_delta_ce": float(delta_ce[mask].mean().item()),
            "positive_delta_ratio": float(positive[mask].float().mean().item()),
        }
    return out


def _bin_delta_stats(
    *,
    values: torch.Tensor,
    delta_ce: torch.Tensor,
    positive: torch.Tensor,
    prefix: str,
) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    if values.numel() == 0:
        return out
    if values.numel() < 3 or float((values.max() - values.min()).abs().item()) < 1e-12:
        mask = torch.ones_like(values, dtype=torch.bool)
        out[f"{prefix}_all"] = {
            "count": int(mask.sum().item()),
            "value_min": float(values.min().item()),
            "value_max": float(values.max().item()),
            "value_mean": float(values.mean().item()),
            "mean_delta_ce": float(delta_ce[mask].mean().item()),
            "positive_delta_ratio": float(positive[mask].float().mean().item()),
        }
        return out
    q1 = torch.quantile(values.float(), 1.0 / 3.0)
    q2 = torch.quantile(values.float(), 2.0 / 3.0)
    bins = {
        f"{prefix}_low": values <= q1,
        f"{prefix}_mid": (values > q1) & (values <= q2),
        f"{prefix}_high": values > q2,
    }
    for name, mask in bins.items():
        if not bool(mask.any()):
            continue
        out[name] = {
            "count": int(mask.sum().item()),
            "value_min": float(values[mask].min().item()),
            "value_max": float(values[mask].max().item()),
            "value_mean": float(values[mask].mean().item()),
            "mean_delta_ce": float(delta_ce[mask].mean().item()),
            "positive_delta_ratio": float(positive[mask].float().mean().item()),
        }
    return out


def _pearson_corr(x: torch.Tensor, y: torch.Tensor) -> float:
    if x.numel() < 2 or y.numel() < 2:
        return 0.0
    x_centered = x.float() - x.float().mean()
    y_centered = y.float() - y.float().mean()
    denom = x_centered.norm() * y_centered.norm()
    if float(denom.item()) <= 1e-12:
        return 0.0
    return float((x_centered * y_centered).sum().div(denom).item())


def _pool_split_delta_stats(
    *,
    split_name: str,
    split_mask: torch.Tensor | None,
    pool_mask: torch.Tensor,
    logits_no_prompt: torch.Tensor,
    logits_prompt: torch.Tensor,
    labels: torch.Tensor,
) -> dict[str, Any]:
    prefix = f"{split_name}_pool"
    if split_mask is None:
        return {
            f"{prefix}_count": 0,
            f"mean_delta_ce_{prefix}": 0.0,
            f"median_delta_ce_{prefix}": 0.0,
            f"positive_delta_ratio_{prefix}": 0.0,
            f"mean_ce_no_prompt_{prefix}": 0.0,
            f"mean_ce_prompt_{prefix}": 0.0,
            f"delta_ce_by_class_{prefix}": {},
        }
    mask = split_mask.to(device=pool_mask.device, dtype=torch.bool) & pool_mask.bool()
    idx = torch.where(mask)[0]
    if idx.numel() == 0:
        return {
            f"{prefix}_count": 0,
            f"mean_delta_ce_{prefix}": 0.0,
            f"median_delta_ce_{prefix}": 0.0,
            f"positive_delta_ratio_{prefix}": 0.0,
            f"mean_ce_no_prompt_{prefix}": 0.0,
            f"mean_ce_prompt_{prefix}": 0.0,
            f"delta_ce_by_class_{prefix}": {},
        }
    y = labels.to(device=idx.device)[idx]
    ce_no_prompt = F.cross_entropy(logits_no_prompt[idx], y, reduction="none")
    ce_prompt = F.cross_entropy(logits_prompt[idx], y, reduction="none")
    delta_ce = ce_no_prompt - ce_prompt
    positive = delta_ce > 0.0
    return {
        f"{prefix}_count": int(idx.numel()),
        f"mean_delta_ce_{prefix}": float(delta_ce.mean().item()),
        f"median_delta_ce_{prefix}": float(delta_ce.median().item()),
        f"positive_delta_ratio_{prefix}": float(positive.float().mean().item()),
        f"mean_ce_no_prompt_{prefix}": float(ce_no_prompt.mean().item()),
        f"mean_ce_prompt_{prefix}": float(ce_prompt.mean().item()),
        f"delta_ce_by_class_{prefix}": _group_delta_stats(keys=y, delta_ce=delta_ce, positive=positive),
    }


@torch.no_grad()
def diagnose_prompt_message_utility(
    *,
    model: FaithfulGP2F,
    prompt_graph_module: PromptGraphModuleP1 | None,
    input_aligner: InputAligner,
    x: torch.Tensor,
    edge_index: torch.Tensor,
    labels: torch.Tensor,
    train_mask: torch.Tensor,
    val_mask: torch.Tensor | None = None,
    test_mask: torch.Tensor | None = None,
    message_scale: float,
) -> dict[str, Any]:
    """Measure whether prompt-on logits reduce CE on train-pool nodes.

    This diagnostic intentionally uses a fixed positive message scale. It does
    not let validation select scale=0 before measuring prompt utility.
    """

    if prompt_graph_module is None:
        return _empty_prompt_message_utility(message_scale, "prompt_graph_module_disabled")
    model.eval()
    input_aligner.eval()
    prompt_graph_module.eval()

    old_scale: float | None = None
    if isinstance(model, PromptAwareGP2F):
        old_scale = float(model.prompt_message_scale)
        model.prompt_message_scale = float(message_scale)

    try:
        z = input_aligner(x)
        model_prompt, prompt_out = _forward_prompt_graph(
            model=model,
            prompt_graph_module=prompt_graph_module,
            z=z,
            edge_index=edge_index,
            train_mask=train_mask,
            edge_scale_multiplier=1.0,
        )
        model_no_prompt = _forward_no_prompt_with_h_pre(
            model=model,
            z=z,
            edge_index=edge_index,
            h_pre=model_prompt["h_pre_shared"],
        )
    finally:
        if old_scale is not None:
            model.prompt_message_scale = old_scale

    pool_mask = prompt_out.get("pool_mask")
    if not isinstance(pool_mask, torch.Tensor):
        return _empty_prompt_message_utility(message_scale, "missing_pool_mask")
    train_pool_mask = train_mask.to(device=pool_mask.device, dtype=torch.bool) & pool_mask.bool()
    train_pool_idx = torch.where(train_pool_mask)[0]
    if train_pool_idx.numel() == 0:
        return _empty_prompt_message_utility(message_scale, "empty_train_pool")

    y = labels.to(device=train_pool_idx.device)[train_pool_idx]
    ce_no_prompt = F.cross_entropy(model_no_prompt["logits"][train_pool_idx], y, reduction="none")
    ce_prompt = F.cross_entropy(model_prompt["logits"][train_pool_idx], y, reduction="none")
    delta_ce = ce_no_prompt - ce_prompt
    positive = delta_ce > 0.0

    aux = prompt_out.get("aux", {})
    pool_idx = aux.get("pool_idx")
    pool_acceptance = aux.get("pool_acceptance_gate")
    structural_score = aux.get("structural_score")
    routing_full_prob = aux.get("routing_full_prob")
    top_prompt_ids = aux.get("top_prompt_ids")

    pool_position = torch.full((pool_mask.numel(),), -1, dtype=torch.long, device=pool_mask.device)
    if isinstance(pool_idx, torch.Tensor) and pool_idx.numel() > 0:
        pool_position[pool_idx] = torch.arange(pool_idx.numel(), dtype=torch.long, device=pool_idx.device)
    train_pool_pos = pool_position[train_pool_idx]
    valid_pool_pos = train_pool_pos >= 0

    if isinstance(pool_acceptance, torch.Tensor) and pool_acceptance.numel() > 0 and bool(valid_pool_pos.all()):
        acceptance = pool_acceptance[train_pool_pos]
    else:
        acceptance = delta_ce.new_ones(delta_ce.numel())
    if isinstance(structural_score, torch.Tensor) and structural_score.numel() >= pool_mask.numel():
        train_structural_score = structural_score[train_pool_idx]
    else:
        train_structural_score = delta_ce.new_zeros(delta_ce.numel())
    if isinstance(top_prompt_ids, torch.Tensor) and top_prompt_ids.numel() > 0 and bool(valid_pool_pos.all()):
        primary_prompt_slot = top_prompt_ids[train_pool_pos, 0].to(dtype=torch.long)
    elif isinstance(routing_full_prob, torch.Tensor) and routing_full_prob.numel() > 0 and bool(valid_pool_pos.all()):
        primary_prompt_slot = routing_full_prob[train_pool_pos].argmax(dim=-1).to(dtype=torch.long)
    else:
        primary_prompt_slot = torch.full_like(train_pool_idx, -1)

    result = {
        "diagnostic": "prompt_message_utility",
        "message_scale": float(message_scale),
        "model_state": "final_after_training",
        "reason": "",
        "train_pool_count": int(train_pool_idx.numel()),
        "val_pool_count": int(((val_mask.to(device=pool_mask.device, dtype=torch.bool) & pool_mask.bool()).sum().item())) if isinstance(val_mask, torch.Tensor) else 0,
        "test_pool_count": int(((test_mask.to(device=pool_mask.device, dtype=torch.bool) & pool_mask.bool()).sum().item())) if isinstance(test_mask, torch.Tensor) else 0,
        "mean_delta_ce": float(delta_ce.mean().item()),
        "median_delta_ce": float(delta_ce.median().item()),
        "positive_delta_ratio": float(positive.float().mean().item()),
        "mean_ce_no_prompt": float(ce_no_prompt.mean().item()),
        "mean_ce_prompt": float(ce_prompt.mean().item()),
        "delta_ce_by_class": _group_delta_stats(keys=y, delta_ce=delta_ce, positive=positive),
        "delta_ce_by_prompt_slot": _group_delta_stats(
            keys=primary_prompt_slot,
            delta_ce=delta_ce,
            positive=positive,
        ),
        "delta_ce_by_acceptance_gate": _bin_delta_stats(
            values=acceptance,
            delta_ce=delta_ce,
            positive=positive,
            prefix="acceptance",
        ),
        "delta_ce_by_structural_score": _bin_delta_stats(
            values=train_structural_score,
            delta_ce=delta_ce,
            positive=positive,
            prefix="structural_score",
        ),
        "acceptance_delta_correlation": _pearson_corr(acceptance, delta_ce),
        "structural_score_delta_correlation": _pearson_corr(train_structural_score, delta_ce),
        "node_records": [],
    }
    result.update(
        _pool_split_delta_stats(
            split_name="train",
            split_mask=train_mask,
            pool_mask=pool_mask,
            logits_no_prompt=model_no_prompt["logits"],
            logits_prompt=model_prompt["logits"],
            labels=labels,
        )
    )
    result.update(
        _pool_split_delta_stats(
            split_name="val",
            split_mask=val_mask,
            pool_mask=pool_mask,
            logits_no_prompt=model_no_prompt["logits"],
            logits_prompt=model_prompt["logits"],
            labels=labels,
        )
    )
    result.update(
        _pool_split_delta_stats(
            split_name="test",
            split_mask=test_mask,
            pool_mask=pool_mask,
            logits_no_prompt=model_no_prompt["logits"],
            logits_prompt=model_prompt["logits"],
            labels=labels,
        )
    )
    for row, node_id in enumerate(train_pool_idx.detach().cpu().tolist()):
        result["node_records"].append(
            {
                "node_id": int(node_id),
                "label": int(y[row].detach().cpu().item()),
                "ce_no_prompt": float(ce_no_prompt[row].detach().cpu().item()),
                "ce_prompt": float(ce_prompt[row].detach().cpu().item()),
                "delta_ce": float(delta_ce[row].detach().cpu().item()),
                "benefited": bool(positive[row].detach().cpu().item()),
                "primary_prompt_slot": int(primary_prompt_slot[row].detach().cpu().item()),
                "acceptance_gate": float(acceptance[row].detach().cpu().item()),
                "structural_score": float(train_structural_score[row].detach().cpu().item()),
            }
        )
    return result


def _write_prompt_message_utility_csv(path: Path, utility: dict[str, Any]) -> None:
    records = utility.get("node_records", [])
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "node_id",
        "label",
        "ce_no_prompt",
        "ce_prompt",
        "delta_ce",
        "benefited",
        "primary_prompt_slot",
        "acceptance_gate",
        "structural_score",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            writer.writerow({key: record.get(key, "") for key in fieldnames})


def run_single(
    config: dict[str, Any],
    *,
    repo_root: Path,
    run_index: int = 0,
    total_runs: int = 1,
    run_group_dir: Path | None = None,
) -> dict[str, Any]:
    experiment_cfg = config.get("experiment", {})
    data_cfg = config.get("data", {})
    pretrained_cfg = config.get("pretrained", {})
    model_cfg = config.get("model", {})
    prompt_graph_cfg = config.get("prompt_graph", {})
    prompt_aware_cfg = config.get("prompt_aware", {})
    training_cfg = config.get("training", {})

    seed = int(experiment_cfg.get("seed", 0))
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    target_dataset = str(experiment_cfg.get("target_dataset", "Cora"))
    variant = _prompt_variant(config)

    data_root = _resolve_path(data_cfg.get("root", "/Users/jackson/MyIdea/data"), base_dir=repo_root)
    loaded = load_node_dataset(
        target_dataset,
        data_root,
        download_if_missing=bool(data_cfg.get("download_if_missing", False)),
    )
    graph = loaded.data.to(device)
    split = build_few_shot_split(
        graph.y,
        shots=int(data_cfg.get("shots", experiment_cfg.get("shots", 5))),
        shot_ratio=data_cfg.get("shot_ratio", experiment_cfg.get("shot_ratio")),
        val_per_class=int(data_cfg.get("val_per_class", experiment_cfg.get("val_per_class", 30))),
        seed=seed,
    )

    checkpoint_path = _resolve_path(
        pretrained_cfg.get("checkpoint_path", "pretrained_gnns/lr_0.0005_weightdecay_0.0005_hid_dim_128.pkl"),
        base_dir=repo_root,
    )
    backbone = load_pretrained_gcn(
        checkpoint_path,
        device=device,
        freeze=bool(pretrained_cfg.get("freeze_backbone", True)),
    )
    source_dim = int(backbone.convs[0].lin.weight.shape[1])
    hidden_dim = int(backbone.convs[0].lin.weight.shape[0])
    input_aligner = InputAligner(
        loaded.num_features,
        source_dim,
        style=str(model_cfg.get("input_aligner", "linear")),
    ).to(device)
    model_kwargs = {
        "hidden_dim": hidden_dim,
        "num_classes": loaded.num_classes,
        "adapter_bottleneck_dim": int(model_cfg.get("adapter_bottleneck_dim", 16)),
        "adapter_beta_init": float(model_cfg.get("adapter_beta_init", 0.01)),
        "adapter_alpha_init": float(model_cfg.get("adapter_alpha_init", 0.1)),
        "adapter_style": str(model_cfg.get("adapter_style", "stable_zero_init")),
        "alpha_init": float(model_cfg.get("alpha_init", 0.5)),
        "fusion_alpha_style": str(model_cfg.get("fusion_alpha_style", "sigmoid")),
    }
    if bool(prompt_aware_cfg.get("enabled", False)):
        prompt_aware_cfg = dict(prompt_aware_cfg)
        prompt_aware_cfg.setdefault("num_prompt_nodes", int(prompt_graph_cfg.get("num_prompt_nodes", 16)))
        prompt_aware_cfg.setdefault("prompt_slot_head_count", int(prompt_graph_cfg.get("num_prompt_nodes", 16)))
        model = PromptAwareGP2F(backbone, prompt_aware_config=prompt_aware_cfg, **model_kwargs).to(device)
    else:
        model = FaithfulGP2F(backbone, **model_kwargs).to(device)
    prompt_graph_module = _build_prompt_graph_module(
        variant=variant,
        source_dim=source_dim,
        hidden_dim=hidden_dim,
        num_classes=loaded.num_classes,
        prompt_graph_cfg=prompt_graph_cfg,
        device=device,
    )

    base_checkpoint_path = _resolve_base_checkpoint(training_cfg, repo_root=repo_root, seed=seed)
    if base_checkpoint_path is not None:
        _load_base_checkpoint(
            base_checkpoint_path,
            model=model,
            input_aligner=input_aligner,
            prompt_graph_module=prompt_graph_module,
            device=device,
            load_prompt=bool(training_cfg.get("load_prompt_from_base_checkpoint", False)),
        )
    class_key_init_stats = _maybe_initialize_class_keys(
        prompt_graph_module=prompt_graph_module,
        input_aligner=input_aligner,
        model=model,
        x=graph.x,
        edge_index=graph.edge_index,
        labels=graph.y,
        train_mask=split.train_mask,
        prompt_graph_cfg=prompt_graph_cfg,
    )
    pattern_key_init_stats = _maybe_initialize_pattern_keys(
        prompt_graph_module=prompt_graph_module,
        input_aligner=input_aligner,
        model=model,
        x=graph.x,
        edge_index=graph.edge_index,
        train_mask=split.train_mask,
        prompt_graph_cfg=prompt_graph_cfg,
    )
    freeze_base_model = bool(training_cfg.get("freeze_base_model", False))
    if freeze_base_model:
        _set_module_trainable(input_aligner, False)
        _set_module_trainable(model, False)
    if isinstance(model, PromptAwareGP2F):
        _set_prompt_aware_trainable(
            model,
            bool(training_cfg.get("train_prompt_aware_module", True)),
        )
    if prompt_graph_module is not None:
        _set_module_trainable(prompt_graph_module, bool(training_cfg.get("train_prompt_graph_module", True)))

    trainable_summary = _trainable_parameter_summary(
        input_aligner=input_aligner,
        model=model,
        prompt_graph_module=prompt_graph_module,
    )
    optimizer_groups, trainable_params, optimizer_summary = _optimizer_groups(
        input_aligner=input_aligner,
        model=model,
        prompt_graph_module=prompt_graph_module,
        training_cfg=training_cfg,
    )
    if not trainable_params:
        raise RuntimeError("No trainable parameters are enabled for this run")
    optimizer = torch.optim.Adam(optimizer_groups)

    output_root = _resolve_path(training_cfg.get("output_dir", "outputs/gp2f_prompt_p1"), base_dir=repo_root)
    if run_group_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_group_dir = output_root / loaded.name / timestamp
    run_dir = run_group_dir / f"seed_{seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    write_yaml(run_dir / "config.yaml", config)

    init_eq = _init_equivalence(
        model=model,
        prompt_graph_module=prompt_graph_module,
        input_aligner=input_aligner,
        x=graph.x,
        edge_index=graph.edge_index,
        train_mask=split.train_mask,
    )

    epochs = int(training_cfg.get("epochs", 200))
    log_every = int(training_cfg.get("log_every", 20))
    eval_every = max(1, int(training_cfg.get("eval_every", 1)))
    monitor = str(training_cfg.get("early_stop_metric", "val_acc"))
    patience = int(training_cfg.get("early_stop_patience", 0))
    min_epochs = int(training_cfg.get("early_stop_min_epochs", 0))
    keep_checkpoint = bool(training_cfg.get("keep_best_checkpoint", True))
    lambda_edge_l1 = float(prompt_graph_cfg.get("lambda_edge_l1", 0.0))
    lambda_prompt_balance = float(prompt_graph_cfg.get("lambda_prompt_balance", 0.0))
    lambda_prompt_role_diversity = float(prompt_graph_cfg.get("lambda_prompt_role_diversity", 0.0))
    lambda_prompt_acceptance = float(prompt_graph_cfg.get("lambda_prompt_acceptance", 0.0))
    lambda_prompt_acceptance_budget = float(prompt_graph_cfg.get("lambda_prompt_acceptance_budget", 0.0))
    acceptance_budget_min_raw = prompt_graph_cfg.get("acceptance_budget_min")
    acceptance_budget_max_raw = prompt_graph_cfg.get("acceptance_budget_max")
    acceptance_budget_min = None if acceptance_budget_min_raw is None else float(acceptance_budget_min_raw)
    acceptance_budget_max = None if acceptance_budget_max_raw is None else float(acceptance_budget_max_raw)
    lambda_prompt_acceptance_supervision = float(prompt_graph_cfg.get("lambda_prompt_acceptance_supervision", 0.0))
    acceptance_supervision_positive_margin = float(prompt_graph_cfg.get("acceptance_supervision_positive_margin", 0.0))
    acceptance_supervision_negative_margin = float(prompt_graph_cfg.get("acceptance_supervision_negative_margin", 0.0))
    acceptance_supervision_balance_targets = bool(prompt_graph_cfg.get("acceptance_supervision_balance_targets", True))
    acceptance_supervision_signal = str(prompt_graph_cfg.get("acceptance_supervision_signal", "ce_delta"))
    lambda_prompt_usage_consistency = float(prompt_graph_cfg.get("lambda_prompt_usage_consistency", 0.0))
    usage_consistency_margin = float(prompt_graph_cfg.get("usage_consistency_margin", 0.25))
    usage_consistency_negative_weight = float(prompt_graph_cfg.get("usage_consistency_negative_weight", 0.05))
    lambda_prompt_view_entropy = float(prompt_graph_cfg.get("lambda_prompt_view_entropy", 0.0))
    lambda_class_route = float(prompt_graph_cfg.get("lambda_class_route", 0.0))
    lambda_key_proto = float(prompt_graph_cfg.get("lambda_key_proto", 0.0))
    lambda_prompt_benefit_supervision = float(prompt_graph_cfg.get("lambda_prompt_benefit_supervision", 0.0))
    benefit_supervision_warmup_epochs = int(prompt_graph_cfg.get("benefit_supervision_warmup_epochs", 0))
    benefit_delta_margin = float(prompt_graph_cfg.get("benefit_delta_margin", 0.0))
    benefit_supervision_balance_targets = bool(prompt_graph_cfg.get("benefit_supervision_balance_targets", True))
    benefit_supervision_label_strategy = str(prompt_graph_cfg.get("benefit_supervision_label_strategy", "margin"))
    benefit_supervision_quantile = float(prompt_graph_cfg.get("benefit_supervision_quantile", 0.20))
    benefit_delta_eps = float(prompt_graph_cfg.get("benefit_delta_eps", 0.0))
    benefit_supervision_quantile_warmup_epochs = int(
        prompt_graph_cfg.get("benefit_supervision_quantile_warmup_epochs", 0)
    )
    benefit_supervision_probe_scale_raw = prompt_graph_cfg.get("benefit_supervision_probe_scale")
    benefit_supervision_probe_scale = (
        None if benefit_supervision_probe_scale_raw is None else float(benefit_supervision_probe_scale_raw)
    )
    lambda_prompt_correction = float(prompt_graph_cfg.get("lambda_prompt_correction", 0.0))
    lambda_prompt_anti_harm = float(prompt_graph_cfg.get("lambda_prompt_anti_harm", 0.0))
    prompt_correction_warmup_epochs = int(prompt_graph_cfg.get("prompt_correction_warmup_epochs", 0))
    prompt_correction_eps = float(prompt_graph_cfg.get("prompt_correction_eps", 0.0))
    prompt_correction_target = str(prompt_graph_cfg.get("prompt_correction_target", "delta_prob_to_label"))
    lambda_prompt_message_help = float(prompt_graph_cfg.get("lambda_prompt_message_help", 0.0))
    prompt_message_help_margin = float(prompt_graph_cfg.get("prompt_message_help_margin", 0.0))
    prompt_message_help_warmup_epochs = int(prompt_graph_cfg.get("prompt_message_help_warmup_epochs", 0))
    prompt_message_help_class_balanced = bool(prompt_graph_cfg.get("prompt_message_help_class_balanced", True))
    lambda_prompt_message_help_query = float(prompt_graph_cfg.get("lambda_prompt_message_help_query", 0.0))
    prompt_message_help_query_warmup_epochs = int(
        prompt_graph_cfg.get("prompt_message_help_query_warmup_epochs", prompt_message_help_warmup_epochs)
    )
    lambda_prompt_class_anti_harm = float(prompt_graph_cfg.get("lambda_prompt_class_anti_harm", 0.0))
    prompt_class_anti_harm_floor = float(prompt_graph_cfg.get("prompt_class_anti_harm_floor", 0.0))
    lambda_utility_receive_gate = float(prompt_graph_cfg.get("lambda_utility_receive_gate", 0.0))
    lambda_utility_receive_gate_query = float(prompt_graph_cfg.get("lambda_utility_receive_gate_query", 0.0))
    utility_receive_gate_warmup_epochs = int(prompt_graph_cfg.get("utility_receive_gate_warmup_epochs", 0))
    utility_receive_gate_query_warmup_epochs = int(
        prompt_graph_cfg.get("utility_receive_gate_query_warmup_epochs", utility_receive_gate_warmup_epochs)
    )
    utility_receive_gate_quantile = float(prompt_graph_cfg.get("utility_receive_gate_quantile", 0.20))
    utility_receive_gate_eps = float(prompt_graph_cfg.get("utility_receive_gate_eps", 1e-4))
    utility_receive_gate_class_balanced = bool(prompt_graph_cfg.get("utility_receive_gate_class_balanced", True))
    lambda_query_proto_alignment = float(prompt_graph_cfg.get("lambda_query_proto_alignment", 0.0))
    query_proto_alignment_warmup_epochs = int(prompt_graph_cfg.get("query_proto_alignment_warmup_epochs", 0))
    query_proto_margin = float(prompt_graph_cfg.get("query_proto_margin", 0.005))
    query_proto_class_balanced = bool(prompt_graph_cfg.get("query_proto_class_balanced", True))
    support_query_enabled = bool(
        prompt_graph_cfg.get("support_query_split", {}).get(
            "enabled", prompt_graph_cfg.get("support_query_split_enabled", False)
        )
    )
    edge_scale_warmup_epochs = int(prompt_graph_cfg.get("edge_scale_warmup_epochs", 0))
    edge_scale_warmup_start = float(prompt_graph_cfg.get("edge_scale_warmup_start", 1.0 if edge_scale_warmup_epochs <= 0 else 0.0))
    best_checkpoint_path = run_dir / "best_model.pt"
    best_val = -1.0
    best_metrics: dict[str, Any] = {}
    loss_curve: list[dict[str, Any]] = []
    prompt_curve: list[dict[str, Any]] = []
    epochs_no_improve = 0
    stopped_epoch = epochs
    early_stopped = False

    run_label = f"{loaded.name} seed={seed} {variant}"
    if total_runs > 1:
        run_label = f"{run_label} ({run_index + 1}/{total_runs})"
    progress = tqdm(range(1, epochs + 1), desc=run_label, unit="epoch", dynamic_ncols=True)
    for epoch in progress:
        model.train()
        input_aligner.train()
        if prompt_graph_module is not None:
            prompt_graph_module.train()
        optimizer.zero_grad()

        z = input_aligner(graph.x)
        current_edge_scale_multiplier = _edge_scale_multiplier(epoch, prompt_graph_cfg)
        model_out, prompt_out = _forward_prompt_graph(
            model=model,
            prompt_graph_module=prompt_graph_module,
            z=z,
            edge_index=graph.edge_index,
            train_mask=split.train_mask,
            edge_scale_multiplier=current_edge_scale_multiplier,
        )
        support_mask, query_mask, support_query_stats = _support_query_masks_for_epoch(
            graph.y,
            split.train_mask,
            prompt_graph_cfg,
            seed=seed,
            epoch=epoch,
        )
        prompt_supervision_mask = support_mask if support_query_enabled else split.train_mask
        prompt_query_mask = query_mask if support_query_enabled else split.train_mask
        cls_loss = F.cross_entropy(model_out["logits"][split.train_mask], graph.y[split.train_mask])
        edge_l1 = prompt_edge_l1_loss(prompt_out) if prompt_graph_module is not None else z.new_tensor(0.0)
        prompt_balance = prompt_balance_loss(prompt_out) if prompt_graph_module is not None else z.new_tensor(0.0)
        prompt_role_diversity = (
            prompt_role_diversity_loss(prompt_graph_module) if prompt_graph_module is not None else z.new_tensor(0.0)
        )
        prompt_acceptance = prompt_acceptance_loss(prompt_out) if prompt_graph_module is not None else z.new_tensor(0.0)
        prompt_acceptance_budget = (
            prompt_acceptance_budget_loss(
                prompt_out,
                min_acceptance=acceptance_budget_min,
                max_acceptance=acceptance_budget_max,
            )
            if prompt_graph_module is not None
            else z.new_tensor(0.0)
        )
        needs_no_prompt_delta = (
            prompt_graph_module is not None
            and (
                lambda_prompt_acceptance_supervision > 0.0
                or (
                    lambda_prompt_benefit_supervision > 0.0
                    and epoch > benefit_supervision_warmup_epochs
                )
                or (
                    (lambda_prompt_correction > 0.0 or lambda_prompt_anti_harm > 0.0)
                    and epoch > prompt_correction_warmup_epochs
                )
                or (
                    lambda_prompt_message_help > 0.0
                    and epoch > prompt_message_help_warmup_epochs
                )
                or (
                    lambda_prompt_message_help_query > 0.0
                    and epoch > prompt_message_help_query_warmup_epochs
                )
                or (
                    lambda_utility_receive_gate > 0.0
                    and epoch > utility_receive_gate_warmup_epochs
                )
                or (
                    lambda_utility_receive_gate_query > 0.0
                    and epoch > utility_receive_gate_query_warmup_epochs
                )
                or (
                    lambda_query_proto_alignment > 0.0
                    and epoch > query_proto_alignment_warmup_epochs
                )
            )
        )
        no_prompt_out = None
        if needs_no_prompt_delta:
            with torch.no_grad():
                no_prompt_out = _forward_no_prompt_with_h_pre(
                    model=model,
                    z=z,
                    edge_index=graph.edge_index,
                    h_pre=model_out["h_pre_shared"],
                )
        if prompt_graph_module is not None and lambda_prompt_acceptance_supervision > 0.0 and no_prompt_out is not None:
            prompt_acceptance_supervision, acceptance_supervision_stats = _acceptance_supervision_loss(
                prompt_out=prompt_out,
                logits_on=model_out["logits"],
                logits_off=no_prompt_out["logits"],
                labels=graph.y,
                train_mask=split.train_mask,
                positive_margin=acceptance_supervision_positive_margin,
                negative_margin=acceptance_supervision_negative_margin,
                balance_targets=acceptance_supervision_balance_targets,
                signal=acceptance_supervision_signal,
            )
        else:
            prompt_acceptance_supervision = z.new_tensor(0.0)
            acceptance_supervision_stats = {
                "prompt_acceptance_supervision": 0.0,
                "acceptance_supervised_count": 0.0,
                "acceptance_positive_count": 0.0,
                "acceptance_negative_count": 0.0,
                "acceptance_ignored_count": 0.0,
                "acceptance_target_mean": 0.0,
                "acceptance_score_delta_mean": 0.0,
            }
        if (
            prompt_graph_module is not None
            and lambda_prompt_benefit_supervision > 0.0
            and epoch > benefit_supervision_warmup_epochs
            and no_prompt_out is not None
        ):
            benefit_logits_on = model_out["logits"]
            if (
                benefit_supervision_probe_scale is not None
                and benefit_supervision_probe_scale > 0.0
                and isinstance(model, PromptAwareGP2F)
            ):
                with torch.no_grad():
                    probe_out = _forward_prompt_graph_with_message_scale(
                        model=model,
                        z=z,
                        edge_index=graph.edge_index,
                        h_pre=model_out["h_pre_shared"],
                        prompt_out=prompt_out,
                        message_scale=benefit_supervision_probe_scale,
                    )
                benefit_logits_on = probe_out["logits"]
            benefit_label_strategy_epoch = benefit_supervision_label_strategy
            if (
                benefit_supervision_label_strategy == "hybrid_quantile_margin"
                and epoch <= benefit_supervision_warmup_epochs + benefit_supervision_quantile_warmup_epochs
            ):
                benefit_label_strategy_epoch = "quantile"
            prompt_benefit_supervision, benefit_supervision_stats = _benefit_supervision_loss(
                prompt_out=prompt_out,
                logits_on=benefit_logits_on,
                logits_off=no_prompt_out["logits"],
                labels=graph.y,
                train_mask=split.train_mask,
                margin=benefit_delta_margin,
                balance_targets=benefit_supervision_balance_targets,
                label_strategy=benefit_label_strategy_epoch,
                quantile=benefit_supervision_quantile,
                eps=benefit_delta_eps,
            )
        else:
            prompt_benefit_supervision = z.new_tensor(0.0)
            benefit_supervision_stats = {
                "prompt_benefit_supervision": 0.0,
                "benefit_supervised_count": 0.0,
                "benefit_positive_count": 0.0,
                "benefit_negative_count": 0.0,
                "benefit_ignored_count": 0.0,
                "mean_delta_ce_train_pool": 0.0,
                "positive_delta_ratio_train_pool": 0.0,
                "benefit_weight_delta_corr_train": 0.0,
                "benefit_label_strategy_quantile": 0.0,
                "benefit_quantile_fraction": 0.0,
                "benefit_positive_threshold": 0.0,
                "benefit_negative_threshold": 0.0,
                "benefit_delta_eps": 0.0,
            }
        if (
            prompt_graph_module is not None
            and (lambda_prompt_correction > 0.0 or lambda_prompt_anti_harm > 0.0)
            and epoch > prompt_correction_warmup_epochs
            and no_prompt_out is not None
        ):
            prompt_correction_loss, prompt_anti_harm_loss, prompt_correction_stats = _prompt_correction_losses(
                prompt_out=prompt_out,
                logits_on=model_out["logits"],
                logits_off=no_prompt_out["logits"],
                labels=graph.y,
                train_mask=split.train_mask,
                eps=prompt_correction_eps,
                target=prompt_correction_target,
            )
        else:
            prompt_correction_loss = z.new_tensor(0.0)
            prompt_anti_harm_loss = z.new_tensor(0.0)
            prompt_correction_stats = {
                "prompt_correction_loss": 0.0,
                "prompt_anti_harm_loss": 0.0,
                "prompt_correction_supervised_count": 0.0,
                "prompt_correction_harmful_count": 0.0,
                "prompt_correction_target_norm": 0.0,
                "prompt_delta_logit_norm": 0.0,
                "prompt_delta_logit_label_alignment": 0.0,
                "harmful_prompt_delta_ratio": 0.0,
            }
        if (
            prompt_graph_module is not None
            and lambda_prompt_message_help > 0.0
            and epoch > prompt_message_help_warmup_epochs
            and no_prompt_out is not None
        ):
            prompt_message_help, prompt_class_anti_harm, prompt_message_help_stats = _prompt_message_help_loss(
                prompt_out=prompt_out,
                logits_on=model_out["logits"],
                logits_off=no_prompt_out["logits"],
                labels=graph.y,
                train_mask=split.train_mask,
                margin=prompt_message_help_margin,
                class_balanced=prompt_message_help_class_balanced,
                anti_harm_floor=prompt_class_anti_harm_floor,
            )
        else:
            prompt_message_help = z.new_tensor(0.0)
            prompt_class_anti_harm = z.new_tensor(0.0)
            prompt_message_help_stats = {
                "prompt_message_help_loss": 0.0,
                "class_balanced_mean_delta_ce": 0.0,
                "class_balanced_positive_delta_ratio": 0.0,
                "message_help_supervised_count": 0.0,
                "message_help_class_count": 0.0,
                "message_help_margin": prompt_message_help_margin,
                "message_help_class_balanced": float(prompt_message_help_class_balanced),
                "prompt_class_anti_harm_loss": 0.0,
                "prompt_class_anti_harm_floor": prompt_class_anti_harm_floor,
                "delta_ce_by_class_train_pool": {},
            }
        if (
            prompt_graph_module is not None
            and lambda_prompt_message_help_query > 0.0
            and epoch > prompt_message_help_query_warmup_epochs
            and no_prompt_out is not None
        ):
            (
                prompt_message_help_query,
                _prompt_message_help_query_anti_harm,
                prompt_message_help_query_stats,
            ) = _prompt_message_help_loss(
                prompt_out=prompt_out,
                logits_on=model_out["logits"],
                logits_off=no_prompt_out["logits"],
                labels=graph.y,
                train_mask=prompt_query_mask,
                margin=prompt_message_help_margin,
                class_balanced=prompt_message_help_class_balanced,
                anti_harm_floor=prompt_class_anti_harm_floor,
            )
            prompt_message_help_query_stats = {
                "prompt_message_help_query_loss": prompt_message_help_query_stats["prompt_message_help_loss"],
                "query_class_balanced_mean_delta_ce": prompt_message_help_query_stats[
                    "class_balanced_mean_delta_ce"
                ],
                "query_class_balanced_positive_delta_ratio": prompt_message_help_query_stats[
                    "class_balanced_positive_delta_ratio"
                ],
                "message_help_query_supervised_count": prompt_message_help_query_stats[
                    "message_help_supervised_count"
                ],
                "message_help_query_class_count": prompt_message_help_query_stats["message_help_class_count"],
                "delta_ce_by_class_query_pool": prompt_message_help_query_stats["delta_ce_by_class_train_pool"],
            }
        else:
            prompt_message_help_query = z.new_tensor(0.0)
            prompt_message_help_query_stats = {
                "prompt_message_help_query_loss": 0.0,
                "query_class_balanced_mean_delta_ce": 0.0,
                "query_class_balanced_positive_delta_ratio": 0.0,
                "message_help_query_supervised_count": 0.0,
                "message_help_query_class_count": 0.0,
                "delta_ce_by_class_query_pool": {},
            }
        if (
            prompt_graph_module is not None
            and lambda_utility_receive_gate > 0.0
            and epoch > utility_receive_gate_warmup_epochs
            and no_prompt_out is not None
        ):
            utility_receive_gate_loss, utility_receive_gate_stats = _utility_receive_gate_loss(
                prompt_out=prompt_out,
                logits_on=model_out["logits"],
                logits_off=no_prompt_out["logits"],
                labels=graph.y,
                train_mask=split.train_mask,
                quantile=utility_receive_gate_quantile,
                eps=utility_receive_gate_eps,
                class_balanced=utility_receive_gate_class_balanced,
            )
        else:
            utility_receive_gate_loss = z.new_tensor(0.0)
            utility_receive_gate_stats = {
                "utility_receive_gate_loss": 0.0,
                "utility_gate_supervised_count": 0.0,
                "utility_gate_positive_count": 0.0,
                "utility_gate_negative_count": 0.0,
                "utility_gate_ignored_count": 0.0,
                "utility_gate_target_mean": 0.0,
                "utility_gate_delta_corr_train": 0.0,
                "utility_gate_positive_threshold_mean": 0.0,
                "utility_gate_negative_threshold_mean": 0.0,
                "utility_gate_quantile": utility_receive_gate_quantile,
                "utility_gate_eps": utility_receive_gate_eps,
                "utility_gate_class_balanced": float(utility_receive_gate_class_balanced),
            }
        if (
            prompt_graph_module is not None
            and lambda_utility_receive_gate_query > 0.0
            and epoch > utility_receive_gate_query_warmup_epochs
            and no_prompt_out is not None
        ):
            utility_receive_gate_query_loss, utility_receive_gate_query_stats = _utility_receive_gate_loss(
                prompt_out=prompt_out,
                logits_on=model_out["logits"],
                logits_off=no_prompt_out["logits"],
                labels=graph.y,
                train_mask=prompt_query_mask,
                quantile=utility_receive_gate_quantile,
                eps=utility_receive_gate_eps,
                class_balanced=utility_receive_gate_class_balanced,
            )
            utility_receive_gate_query_stats = {
                "utility_receive_gate_query_loss": utility_receive_gate_query_stats["utility_receive_gate_loss"],
                "utility_gate_query_supervised_count": utility_receive_gate_query_stats[
                    "utility_gate_supervised_count"
                ],
                "utility_gate_query_positive_count": utility_receive_gate_query_stats[
                    "utility_gate_positive_count"
                ],
                "utility_gate_query_negative_count": utility_receive_gate_query_stats[
                    "utility_gate_negative_count"
                ],
                "utility_gate_query_ignored_count": utility_receive_gate_query_stats["utility_gate_ignored_count"],
                "utility_gate_query_target_mean": utility_receive_gate_query_stats["utility_gate_target_mean"],
                "utility_gate_query_delta_corr": utility_receive_gate_query_stats["utility_gate_delta_corr_train"],
            }
        else:
            utility_receive_gate_query_loss = z.new_tensor(0.0)
            utility_receive_gate_query_stats = {
                "utility_receive_gate_query_loss": 0.0,
                "utility_gate_query_supervised_count": 0.0,
                "utility_gate_query_positive_count": 0.0,
                "utility_gate_query_negative_count": 0.0,
                "utility_gate_query_ignored_count": 0.0,
                "utility_gate_query_target_mean": 0.0,
                "utility_gate_query_delta_corr": 0.0,
            }
        if (
            prompt_graph_module is not None
            and lambda_query_proto_alignment > 0.0
            and epoch > query_proto_alignment_warmup_epochs
            and no_prompt_out is not None
        ):
            query_proto_alignment, query_proto_alignment_stats = _query_proto_alignment_loss(
                h_on=model_out["h_mix"],
                h_off=no_prompt_out["h_mix"],
                labels=graph.y,
                support_mask=support_mask,
                query_mask=query_mask,
                pool_mask=prompt_out.get("pool_mask", torch.zeros_like(split.train_mask)),
                margin=query_proto_margin,
                class_balanced=query_proto_class_balanced,
            )
        else:
            query_proto_alignment = z.new_tensor(0.0)
            query_proto_alignment_stats = {
                "query_proto_alignment_loss": 0.0,
                "query_proto_supervised_count": 0.0,
                "query_proto_class_count": 0.0,
                "query_proto_mean_delta_dist": 0.0,
                "query_proto_positive_ratio": 0.0,
                "query_proto_margin": query_proto_margin,
                "query_proto_class_balanced": float(query_proto_class_balanced),
                "query_proto_by_class": {},
            }
        prompt_usage_consistency = (
            prompt_usage_consistency_loss(
                prompt_out,
                graph.y,
                prompt_supervision_mask,
                margin=usage_consistency_margin,
                negative_weight=usage_consistency_negative_weight,
            )
            if prompt_graph_module is not None
            else z.new_tensor(0.0)
        )
        prompt_view_entropy = prompt_view_entropy_loss(prompt_out) if prompt_graph_module is not None else z.new_tensor(0.0)
        class_route = (
            prompt_class_route_loss(prompt_out, graph.y, prompt_supervision_mask)
            if prompt_graph_module is not None
            else z.new_tensor(0.0)
        )
        key_proto = (
            prompt_key_proto_loss(prompt_graph_module, prompt_out, graph.y, prompt_supervision_mask)
            if prompt_graph_module is not None
            else z.new_tensor(0.0)
        )
        loss = (
            cls_loss
            + lambda_edge_l1 * edge_l1
            + lambda_prompt_balance * prompt_balance
            + lambda_prompt_role_diversity * prompt_role_diversity
            + lambda_prompt_acceptance * prompt_acceptance
            + lambda_prompt_acceptance_budget * prompt_acceptance_budget
            + lambda_prompt_acceptance_supervision * prompt_acceptance_supervision
            + lambda_prompt_usage_consistency * prompt_usage_consistency
            + lambda_prompt_view_entropy * prompt_view_entropy
            + lambda_class_route * class_route
            + lambda_key_proto * key_proto
            + lambda_prompt_benefit_supervision * prompt_benefit_supervision
            + lambda_prompt_correction * prompt_correction_loss
            + lambda_prompt_anti_harm * prompt_anti_harm_loss
            + lambda_prompt_message_help * prompt_message_help
            + lambda_prompt_message_help_query * prompt_message_help_query
            + lambda_prompt_class_anti_harm * prompt_class_anti_harm
            + lambda_utility_receive_gate * utility_receive_gate_loss
            + lambda_utility_receive_gate_query * utility_receive_gate_query_loss
            + lambda_query_proto_alignment * query_proto_alignment
        )
        if not torch.isfinite(loss):
            raise RuntimeError(f"Non-finite loss at epoch {epoch}: {loss.item()}")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable_params, float(training_cfg.get("grad_clip", 1.0)))
        optimizer.step()

        prompt_log = _prompt_graph_diagnostics(
            prompt_out,
            z=z,
            train_mask=split.train_mask,
        )
        prompt_aware_log = _prompt_aware_diagnostics(model_out)
        log_item = {
            "epoch": float(epoch),
            "total": float(loss.detach().item()),
            "cls": float(cls_loss.detach().item()),
            "edge_l1": float(edge_l1.detach().item()),
            "prompt_balance": float(prompt_balance.detach().item()),
            "prompt_role_diversity": float(prompt_role_diversity.detach().item()),
            "prompt_acceptance": float(prompt_acceptance.detach().item()),
            "prompt_acceptance_budget": float(prompt_acceptance_budget.detach().item()),
            "prompt_acceptance_supervision": float(prompt_acceptance_supervision.detach().item()),
            "prompt_usage_consistency": float(prompt_usage_consistency.detach().item()),
            "prompt_view_entropy": float(prompt_view_entropy.detach().item()),
            "class_route": float(class_route.detach().item()),
            "key_proto": float(key_proto.detach().item()),
            "prompt_benefit_supervision": float(prompt_benefit_supervision.detach().item()),
            "prompt_correction_loss": float(prompt_correction_loss.detach().item()),
            "prompt_anti_harm_loss": float(prompt_anti_harm_loss.detach().item()),
            "prompt_message_help_loss": float(prompt_message_help.detach().item()),
            "prompt_message_help_query_loss": float(prompt_message_help_query.detach().item()),
            "prompt_class_anti_harm_loss": float(prompt_class_anti_harm.detach().item()),
            "utility_receive_gate_loss": float(utility_receive_gate_loss.detach().item()),
            "utility_receive_gate_query_loss": float(utility_receive_gate_query_loss.detach().item()),
            "query_proto_alignment_loss": float(query_proto_alignment.detach().item()),
            "lambda_edge_l1": lambda_edge_l1,
            "lambda_prompt_balance": lambda_prompt_balance,
            "lambda_prompt_role_diversity": lambda_prompt_role_diversity,
            "lambda_prompt_acceptance": lambda_prompt_acceptance,
            "lambda_prompt_acceptance_budget": lambda_prompt_acceptance_budget,
            "lambda_prompt_acceptance_supervision": lambda_prompt_acceptance_supervision,
            "lambda_prompt_usage_consistency": lambda_prompt_usage_consistency,
            "lambda_prompt_view_entropy": lambda_prompt_view_entropy,
            "lambda_class_route": lambda_class_route,
            "lambda_key_proto": lambda_key_proto,
            "lambda_prompt_benefit_supervision": lambda_prompt_benefit_supervision,
            "lambda_prompt_correction": lambda_prompt_correction,
            "lambda_prompt_anti_harm": lambda_prompt_anti_harm,
            "lambda_prompt_message_help": lambda_prompt_message_help,
            "lambda_prompt_message_help_query": lambda_prompt_message_help_query,
            "lambda_prompt_class_anti_harm": lambda_prompt_class_anti_harm,
            "lambda_utility_receive_gate": lambda_utility_receive_gate,
            "lambda_utility_receive_gate_query": lambda_utility_receive_gate_query,
            "lambda_query_proto_alignment": lambda_query_proto_alignment,
            "utility_receive_gate_warmup_epochs": utility_receive_gate_warmup_epochs,
            "utility_receive_gate_query_warmup_epochs": utility_receive_gate_query_warmup_epochs,
            "utility_receive_gate_quantile": utility_receive_gate_quantile,
            "utility_receive_gate_eps": utility_receive_gate_eps,
            "utility_receive_gate_class_balanced": float(utility_receive_gate_class_balanced),
            "prompt_correction_warmup_epochs": prompt_correction_warmup_epochs,
            "prompt_correction_eps": prompt_correction_eps,
            "prompt_correction_target": prompt_correction_target,
            "prompt_message_help_warmup_epochs": prompt_message_help_warmup_epochs,
            "prompt_message_help_query_warmup_epochs": prompt_message_help_query_warmup_epochs,
            "prompt_message_help_margin": prompt_message_help_margin,
            "prompt_message_help_class_balanced": float(prompt_message_help_class_balanced),
            "prompt_class_anti_harm_floor": prompt_class_anti_harm_floor,
            "support_query_enabled": float(support_query_stats.get("enabled", False)),
            "support_count": float(support_query_stats.get("support_count", 0)),
            "query_count": float(support_query_stats.get("query_count", 0)),
            "train_support_ratio": float(support_query_stats.get("train_support_ratio", 1.0)),
            "query_proto_alignment_warmup_epochs": query_proto_alignment_warmup_epochs,
            "query_proto_margin": query_proto_margin,
            "query_proto_class_balanced": float(query_proto_class_balanced),
            "benefit_supervision_label_strategy_quantile": float(benefit_supervision_label_strategy == "quantile"),
            "benefit_supervision_quantile_config": benefit_supervision_quantile,
            "benefit_delta_eps_config": benefit_delta_eps,
            "benefit_supervision_quantile_warmup_epochs": benefit_supervision_quantile_warmup_epochs,
            "benefit_supervision_probe_scale": (
                0.0 if benefit_supervision_probe_scale is None else benefit_supervision_probe_scale
            ),
            "edge_scale_multiplier": current_edge_scale_multiplier,
            "alpha": float(model_out["alpha"].detach().item()),
            **prompt_aware_log,
            **prompt_log,
            **acceptance_supervision_stats,
            **benefit_supervision_stats,
            **prompt_correction_stats,
            **prompt_message_help_stats,
            **prompt_message_help_query_stats,
            **utility_receive_gate_stats,
            **utility_receive_gate_query_stats,
            **query_proto_alignment_stats,
        }
        loss_curve.append(log_item)
        prompt_curve.append({"epoch": float(epoch), **prompt_aware_log, **prompt_log})

        should_eval = epoch == 1 or epoch % eval_every == 0 or epoch == epochs
        metrics: dict[str, Any] | None = None
        if should_eval:
            metrics = evaluate_prompt_graph(
                model=model,
                prompt_graph_module=prompt_graph_module,
                input_aligner=input_aligner,
                x=graph.x,
                edge_index=graph.edge_index,
                labels=graph.y,
                train_mask=split.train_mask,
                val_mask=split.val_mask,
                test_mask=split.test_mask,
                num_classes=loaded.num_classes,
                edge_scale_multiplier=current_edge_scale_multiplier,
            )
            monitor_lookup = {
                **metrics,
                "train_loss": float(loss.detach().item()),
                "total": float(loss.detach().item()),
                "cls": float(cls_loss.detach().item()),
                "edge_l1": float(edge_l1.detach().item()),
                "prompt_balance": float(prompt_balance.detach().item()),
            }
            monitor_value = float(monitor_lookup.get(monitor, metrics["val_acc"]))
            if _monitor_improved(monitor, monitor_value, best_val, bool(best_metrics)):
                best_val = monitor_value
                best_metrics = {
                    "best_epoch": float(epoch),
                    "monitor_value": monitor_value,
                    **metrics,
                    "total": float(loss.detach().item()),
                    "cls": float(cls_loss.detach().item()),
                    "edge_l1": float(edge_l1.detach().item()),
                    "prompt_balance": float(prompt_balance.detach().item()),
                    "prompt_role_diversity": float(prompt_role_diversity.detach().item()),
                    "prompt_acceptance": float(prompt_acceptance.detach().item()),
                    "prompt_acceptance_budget": float(prompt_acceptance_budget.detach().item()),
                    "prompt_acceptance_supervision": float(prompt_acceptance_supervision.detach().item()),
                    "prompt_benefit_supervision": float(prompt_benefit_supervision.detach().item()),
                    "prompt_usage_consistency": float(prompt_usage_consistency.detach().item()),
                    "prompt_view_entropy": float(prompt_view_entropy.detach().item()),
                    "class_route": float(class_route.detach().item()),
                    "key_proto": float(key_proto.detach().item()),
                    "prompt_correction_loss": float(prompt_correction_loss.detach().item()),
                    "prompt_anti_harm_loss": float(prompt_anti_harm_loss.detach().item()),
                    "prompt_message_help_loss": float(prompt_message_help.detach().item()),
                    "prompt_message_help_query_loss": float(prompt_message_help_query.detach().item()),
                    "prompt_class_anti_harm_loss": float(prompt_class_anti_harm.detach().item()),
                    "utility_receive_gate_loss": float(utility_receive_gate_loss.detach().item()),
                    "utility_receive_gate_query_loss": float(utility_receive_gate_query_loss.detach().item()),
                    "query_proto_alignment_loss": float(query_proto_alignment.detach().item()),
                    "support_query_enabled": float(support_query_stats.get("enabled", False)),
                    "support_count": float(support_query_stats.get("support_count", 0)),
                    "query_count": float(support_query_stats.get("query_count", 0)),
                    "train_support_ratio": float(support_query_stats.get("train_support_ratio", 1.0)),
                    **acceptance_supervision_stats,
                    **benefit_supervision_stats,
                    **prompt_correction_stats,
                    **prompt_message_help_stats,
                    **prompt_message_help_query_stats,
                    **utility_receive_gate_stats,
                    **utility_receive_gate_query_stats,
                    **query_proto_alignment_stats,
                    "edge_scale_multiplier": current_edge_scale_multiplier,
                    **_adapter_stats(model),
                }
                _save_checkpoint(
                    best_checkpoint_path,
                    model=model,
                    input_aligner=input_aligner,
                    prompt_graph_module=prompt_graph_module,
                    epoch=epoch,
                    metrics=best_metrics,
                )
                epochs_no_improve = 0
            else:
                epochs_no_improve += eval_every

        if epoch == 1 or epoch % log_every == 0 or epoch == epochs:
            if metrics is None:
                metrics = evaluate_prompt_graph(
                    model=model,
                    prompt_graph_module=prompt_graph_module,
                    input_aligner=input_aligner,
                    x=graph.x,
                    edge_index=graph.edge_index,
                    labels=graph.y,
                    train_mask=split.train_mask,
                    val_mask=split.val_mask,
                    test_mask=split.test_mask,
                    num_classes=loaded.num_classes,
                    edge_scale_multiplier=current_edge_scale_multiplier,
                )
            progress.set_postfix(
                {
                    "loss": f"{loss.item():.4f}",
                    "val": f"{metrics['val_acc']:.4f}",
                    "test": f"{metrics['test_acc']:.4f}",
                    "pool": f"{metrics['pool_ratio']:.2f}",
                    "edge": f"{metrics['mean_prompt_edge_weight']:.3f}",
                    "best": f"{best_val:.4f}",
                }
            )
        if patience > 0 and epoch >= min_epochs and epochs_no_improve >= patience:
            stopped_epoch = epoch
            early_stopped = True
            if metrics is None:
                metrics = evaluate_prompt_graph(
                    model=model,
                    prompt_graph_module=prompt_graph_module,
                    input_aligner=input_aligner,
                    x=graph.x,
                    edge_index=graph.edge_index,
                    labels=graph.y,
                    train_mask=split.train_mask,
                    val_mask=split.val_mask,
                    test_mask=split.test_mask,
                    num_classes=loaded.num_classes,
                    edge_scale_multiplier=current_edge_scale_multiplier,
                )
            progress.set_postfix(
                {
                    "loss": f"{loss.item():.4f}",
                    "val": f"{metrics['val_acc']:.4f}",
                    "test": f"{metrics['test_acc']:.4f}",
                    "pool": f"{metrics['pool_ratio']:.2f}",
                    "edge": f"{metrics['mean_prompt_edge_weight']:.3f}",
                    "stop": "early",
                }
            )
            break

    final_metrics = evaluate_prompt_graph(
        model=model,
        prompt_graph_module=prompt_graph_module,
        input_aligner=input_aligner,
        x=graph.x,
        edge_index=graph.edge_index,
        labels=graph.y,
        train_mask=split.train_mask,
        val_mask=split.val_mask,
        test_mask=split.test_mask,
        num_classes=loaded.num_classes,
        edge_scale_multiplier=1.0,
    )
    prompt_message_utility: dict[str, Any] | None = None
    if bool(training_cfg.get("diagnose_prompt_message_utility", False)):
        diagnostic_scale = float(prompt_aware_cfg.get("message_scale", 0.0))
        prompt_message_utility = diagnose_prompt_message_utility(
            model=model,
            prompt_graph_module=prompt_graph_module,
            input_aligner=input_aligner,
            x=graph.x,
            edge_index=graph.edge_index,
            labels=graph.y,
            train_mask=split.train_mask,
            val_mask=split.val_mask,
            test_mask=split.test_mask,
            message_scale=diagnostic_scale,
        )
        write_json(run_dir / "prompt_message_utility.json", prompt_message_utility)
        _write_prompt_message_utility_csv(run_dir / "prompt_message_utility_nodes.csv", prompt_message_utility)
    result = {
        "dataset": loaded.name,
        "seed": seed,
        "split_seed": split.seed,
        "prompt_variant": variant,
        "rho": float(prompt_graph_cfg.get("rho", 0.0)),
        "topk_prompt_per_node": int(prompt_graph_cfg.get("topk_prompt_per_node", 0)),
        "num_nodes": int(graph.num_nodes),
        "num_features": loaded.num_features,
        "num_classes": loaded.num_classes,
        "checkpoint_path": str(checkpoint_path),
        "source_dim": source_dim,
        "hidden_dim": hidden_dim,
        "aligner_style": str(model_cfg.get("input_aligner", "linear")),
        "adapter_style": str(model_cfg.get("adapter_style", "stable_zero_init")),
        "fusion_alpha_style": str(model_cfg.get("fusion_alpha_style", "sigmoid")),
        "model_variant": _model_variant(model_cfg),
        "final": {**final_metrics, **init_eq},
        "best": {**best_metrics, **init_eq},
        "prompt_graph_parameter_count": count_trainable_parameters(prompt_graph_module),
        "class_key_initialization": class_key_init_stats,
        "pattern_key_initialization": pattern_key_init_stats,
        "trainable_parameters": trainable_summary,
        "optimizer": optimizer_summary,
        "regularization": {
            "lambda_edge_l1": lambda_edge_l1,
            "lambda_prompt_balance": lambda_prompt_balance,
            "lambda_prompt_role_diversity": lambda_prompt_role_diversity,
            "lambda_prompt_acceptance": lambda_prompt_acceptance,
            "lambda_prompt_acceptance_budget": lambda_prompt_acceptance_budget,
            "acceptance_budget_min": acceptance_budget_min,
            "acceptance_budget_max": acceptance_budget_max,
            "lambda_prompt_acceptance_supervision": lambda_prompt_acceptance_supervision,
            "acceptance_supervision_positive_margin": acceptance_supervision_positive_margin,
            "acceptance_supervision_negative_margin": acceptance_supervision_negative_margin,
            "acceptance_supervision_balance_targets": acceptance_supervision_balance_targets,
            "acceptance_supervision_signal": acceptance_supervision_signal,
            "lambda_prompt_usage_consistency": lambda_prompt_usage_consistency,
            "usage_consistency_margin": usage_consistency_margin,
            "usage_consistency_negative_weight": usage_consistency_negative_weight,
            "lambda_prompt_view_entropy": lambda_prompt_view_entropy,
            "lambda_class_route": lambda_class_route,
            "lambda_key_proto": lambda_key_proto,
            "lambda_prompt_benefit_supervision": lambda_prompt_benefit_supervision,
            "benefit_supervision_warmup_epochs": benefit_supervision_warmup_epochs,
            "benefit_delta_margin": benefit_delta_margin,
            "benefit_supervision_label_strategy": benefit_supervision_label_strategy,
            "benefit_supervision_quantile": benefit_supervision_quantile,
            "benefit_delta_eps": benefit_delta_eps,
            "benefit_supervision_quantile_warmup_epochs": benefit_supervision_quantile_warmup_epochs,
            "benefit_supervision_probe_scale": benefit_supervision_probe_scale,
            "benefit_supervision_balance_targets": benefit_supervision_balance_targets,
            "lambda_prompt_correction": lambda_prompt_correction,
            "lambda_prompt_anti_harm": lambda_prompt_anti_harm,
            "lambda_prompt_message_help": lambda_prompt_message_help,
            "lambda_prompt_class_anti_harm": lambda_prompt_class_anti_harm,
            "prompt_correction_warmup_epochs": prompt_correction_warmup_epochs,
            "prompt_correction_eps": prompt_correction_eps,
            "prompt_correction_target": prompt_correction_target,
            "prompt_message_help_warmup_epochs": prompt_message_help_warmup_epochs,
            "prompt_message_help_margin": prompt_message_help_margin,
            "prompt_message_help_class_balanced": prompt_message_help_class_balanced,
            "prompt_class_anti_harm_floor": prompt_class_anti_harm_floor,
            "edge_scale_warmup_epochs": edge_scale_warmup_epochs,
            "edge_scale_warmup_start": edge_scale_warmup_start,
        },
        "base_checkpoint_path": str(base_checkpoint_path) if base_checkpoint_path is not None else "",
        "freeze_base_model": freeze_base_model,
        "train_prompt_graph_module": bool(training_cfg.get("train_prompt_graph_module", True)),
        "early_stopped": early_stopped,
        "stopped_epoch": stopped_epoch,
        "early_stop_metric": monitor,
        "best_checkpoint_path": str(best_checkpoint_path) if best_checkpoint_path.exists() and keep_checkpoint else "",
        "split_counts": _split_counts(
            graph.y,
            {"train": split.train_mask, "val": split.val_mask, "test": split.test_mask},
            loaded.num_classes,
        ),
        "environment": _environment_info(),
        "run_dir": str(run_dir),
    }
    if prompt_message_utility is not None:
        result["prompt_message_utility"] = {
            key: value for key, value in prompt_message_utility.items() if key != "node_records"
        }
    write_json(run_dir / "metrics.json", result)
    write_json(run_dir / "loss_curve.json", {"loss_curve": loss_curve})
    write_json(run_dir / "prompt_curve.json", {"prompt_curve": prompt_curve})
    if best_checkpoint_path.exists() and not keep_checkpoint:
        best_checkpoint_path.unlink()
    print(f"Saved results to {run_dir}")
    return result


def run(config: dict[str, Any], *, repo_root: Path) -> dict[str, Any]:
    variant = _prompt_variant(config)
    config = _config_for_variant(config, variant)
    seeds = _resolve_run_seeds(config)
    output_root = _resolve_path(config.get("training", {}).get("output_dir", "outputs/gp2f_prompt_p1"), base_dir=repo_root)
    target_dataset = str(config.get("experiment", {}).get("target_dataset", "Cora"))
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    summary_dir = output_root / target_dataset / timestamp
    summary_dir.mkdir(parents=True, exist_ok=True)
    scale_grid = _message_scale_grid(config)
    diagnostic_mode = bool(config.get("training", {}).get("diagnose_prompt_message_utility", False))
    if diagnostic_mode:
        if not scale_grid:
            scale_grid = [0.1, 0.25, 0.5, 1.0]
        scale_grid = [float(scale) for scale in scale_grid if float(scale) > 0.0]
        if not scale_grid:
            raise ValueError("diagnose_prompt_message_utility requires at least one positive message scale")
    use_scale_grid = variant.startswith("p2_") and len(scale_grid) > 1
    if diagnostic_mode:
        use_scale_grid = False
    scale_selection_metric = str(config.get("training", {}).get("message_scale_selection_metric", "val_acc"))

    results: list[dict[str, Any]] = []
    for idx, seed in enumerate(seeds):
        if diagnostic_mode:
            for scale in scale_grid:
                seed_config = _deep_update(
                    config,
                    {
                        "experiment": {"seed": seed},
                        "prompt_aware": {"message_scale": float(scale), "message_scale_grid": [float(scale)]},
                        "training": {"diagnose_prompt_message_utility": True},
                    },
                )
                result = run_single(
                    seed_config,
                    repo_root=repo_root,
                    run_index=idx,
                    total_runs=len(seeds),
                    run_group_dir=summary_dir / f"scale_{_scale_label(scale)}",
                )
                result["diagnostic_message_scale"] = float(scale)
                result["selected_message_scale"] = float(scale)
                result["message_scale_selection_metric"] = "diagnostic_fixed_scale"
                results.append(result)
        elif use_scale_grid:
            candidates: list[dict[str, Any]] = []
            for scale in scale_grid:
                seed_config = _deep_update(
                    config,
                    {
                        "experiment": {"seed": seed},
                        "prompt_aware": {"message_scale": float(scale)},
                    },
                )
                result = run_single(
                    seed_config,
                    repo_root=repo_root,
                    run_index=idx,
                    total_runs=len(seeds),
                    run_group_dir=summary_dir / f"scale_{_scale_label(scale)}",
                )
                result["candidate_message_scale"] = float(scale)
                candidates.append(result)
            selected = _select_message_scale_candidate(candidates, metric=scale_selection_metric)
            selected = copy.deepcopy(selected)
            selected["selected_message_scale"] = float(selected.get("candidate_message_scale", 0.0))
            selected["message_scale_selection_metric"] = scale_selection_metric
            selected["message_scale_candidates"] = [
                _compact_scale_candidate(candidate, scale_selection_metric) for candidate in candidates
            ]
            results.append(selected)
        else:
            seed_config = _deep_update(config, {"experiment": {"seed": seed}})
            result = run_single(
                seed_config,
                repo_root=repo_root,
                run_index=idx,
                total_runs=len(seeds),
                run_group_dir=summary_dir,
            )
            if scale_grid:
                result["selected_message_scale"] = float(scale_grid[0])
                result["message_scale_selection_metric"] = scale_selection_metric
            results.append(result)

    best_test = [float(result["best"].get("test_acc", 0.0)) for result in results]
    best_macro_f1 = [float(result["best"].get("test_macro_f1", 0.0)) for result in results]
    final_test = [float(result["final"].get("test_acc", 0.0)) for result in results]
    final_macro_f1 = [float(result["final"].get("test_macro_f1", 0.0)) for result in results]
    summary = {
        "dataset": target_dataset,
        "prompt_variant": variant,
        "seeds": seeds,
        "num_runs": len(seeds),
        "best_test_acc_mean_std": _format_mean_std(best_test),
        "best_test_macro_f1_mean_std": _format_mean_std(best_macro_f1),
        "final_test_acc_mean_std": _format_mean_std(final_test),
        "final_test_macro_f1_mean_std": _format_mean_std(final_macro_f1),
        "message_scale_grid": scale_grid,
        "message_scale_selection_metric": scale_selection_metric if scale_grid else "",
        "message_scale_selection_enabled": use_scale_grid,
        "diagnose_prompt_message_utility": diagnostic_mode,
        "runs": results,
        "environment": _environment_info(),
    }
    if diagnostic_mode:
        utility_by_scale: dict[str, dict[str, float]] = {}
        for scale in scale_grid:
            scale_results = [
                result.get("prompt_message_utility", {})
                for result in results
                if float(result.get("diagnostic_message_scale", -1.0)) == float(scale)
            ]
            if not scale_results:
                continue
            utility_by_scale[str(float(scale))] = {
                "runs": float(len(scale_results)),
                "mean_delta_ce": float(
                    sum(float(item.get("mean_delta_ce", 0.0)) for item in scale_results) / len(scale_results)
                ),
                "positive_delta_ratio": float(
                    sum(float(item.get("positive_delta_ratio", 0.0)) for item in scale_results)
                    / len(scale_results)
                ),
                "mean_ce_no_prompt": float(
                    sum(float(item.get("mean_ce_no_prompt", 0.0)) for item in scale_results) / len(scale_results)
                ),
                "mean_ce_prompt": float(
                    sum(float(item.get("mean_ce_prompt", 0.0)) for item in scale_results) / len(scale_results)
                ),
                "mean_delta_ce_train_pool": float(
                    sum(float(item.get("mean_delta_ce_train_pool", 0.0)) for item in scale_results)
                    / len(scale_results)
                ),
                "positive_delta_ratio_train_pool": float(
                    sum(float(item.get("positive_delta_ratio_train_pool", 0.0)) for item in scale_results)
                    / len(scale_results)
                ),
                "mean_delta_ce_val_pool": float(
                    sum(float(item.get("mean_delta_ce_val_pool", 0.0)) for item in scale_results)
                    / len(scale_results)
                ),
                "positive_delta_ratio_val_pool": float(
                    sum(float(item.get("positive_delta_ratio_val_pool", 0.0)) for item in scale_results)
                    / len(scale_results)
                ),
                "mean_delta_ce_test_pool": float(
                    sum(float(item.get("mean_delta_ce_test_pool", 0.0)) for item in scale_results)
                    / len(scale_results)
                ),
                "positive_delta_ratio_test_pool": float(
                    sum(float(item.get("positive_delta_ratio_test_pool", 0.0)) for item in scale_results)
                    / len(scale_results)
                ),
                "acceptance_delta_correlation": float(
                    sum(float(item.get("acceptance_delta_correlation", 0.0)) for item in scale_results)
                    / len(scale_results)
                ),
                "structural_score_delta_correlation": float(
                    sum(float(item.get("structural_score_delta_correlation", 0.0)) for item in scale_results)
                    / len(scale_results)
                ),
            }
        summary["prompt_message_utility_by_scale"] = utility_by_scale
    write_json(summary_dir / "summary.json", summary)
    with (summary_dir / "summary.csv").open("w", encoding="utf-8", newline="") as handle:
        fieldnames = [
            "seed",
            "best_epoch",
            "best_test_acc",
            "best_test_macro_f1",
            "final_test_acc",
            "final_test_macro_f1",
            "alpha",
            "branch_cosine",
            "rho",
            "pool_ratio",
            "train_pool_ratio",
            "prompt_node_count",
            "prompt_edge_count",
            "edge_scale",
            "raw_edge_scale",
            "edge_scale_multiplier",
            "use_multiview_routing",
            "use_class_aware_routing",
            "use_pattern_prompt_bank",
            "use_receiver_only_prompt",
            "use_benefit_gate",
            "num_class_prompt_slots",
            "residual_prompt_count",
            "class_key_proto_init_coverage",
            "pattern_key_init_coverage",
            "semantic_view_weight",
            "structural_view_weight",
            "role_view_weight",
            "view_gate_entropy",
            "mean_prompt_edge_weight",
            "pool_acceptance_mean",
            "pool_acceptance_min",
            "pool_acceptance_max",
            "use_hard_acceptance",
            "hard_acceptance_ratio",
            "hard_acceptance_selected_ratio",
            "prompt_acceptance_budget",
            "prompt_acceptance_supervision",
            "acceptance_supervised_count",
            "acceptance_positive_count",
            "acceptance_negative_count",
            "acceptance_ignored_count",
            "acceptance_target_mean",
            "acceptance_score_delta_mean",
            "ce_delta_positive_ratio_train",
            "ce_delta_negative_ratio_train",
            "prompt_helpful_acceptance_precision_train",
            "prompt_harmful_rejection_precision_train",
            "prompt_benefit_supervision",
            "benefit_supervised_count",
            "benefit_positive_count",
            "benefit_negative_count",
            "benefit_ignored_count",
            "mean_delta_ce_train_pool",
            "positive_delta_ratio_train_pool",
            "benefit_weight_delta_corr_train",
            "benefit_label_strategy_quantile",
            "benefit_quantile_fraction",
            "benefit_positive_threshold",
            "benefit_negative_threshold",
            "benefit_delta_eps",
            "prompt_correction_loss",
            "prompt_anti_harm_loss",
            "prompt_correction_supervised_count",
            "prompt_correction_harmful_count",
            "prompt_correction_target_norm",
            "prompt_delta_logit_norm",
            "prompt_delta_logit_label_alignment",
            "harmful_prompt_delta_ratio",
            "prompt_message_help_loss",
            "prompt_message_help_query_loss",
            "prompt_class_anti_harm_loss",
            "class_balanced_mean_delta_ce",
            "class_balanced_positive_delta_ratio",
            "query_class_balanced_mean_delta_ce",
            "query_class_balanced_positive_delta_ratio",
            "message_help_supervised_count",
            "message_help_query_supervised_count",
            "message_help_class_count",
            "message_help_query_class_count",
            "message_help_margin",
            "message_help_class_balanced",
            "prompt_class_anti_harm_floor",
            "utility_receive_gate_loss",
            "utility_receive_gate_query_loss",
            "utility_gate_supervised_count",
            "utility_gate_query_supervised_count",
            "utility_gate_positive_count",
            "utility_gate_query_positive_count",
            "utility_gate_negative_count",
            "utility_gate_query_negative_count",
            "utility_gate_ignored_count",
            "utility_gate_query_ignored_count",
            "utility_gate_target_mean",
            "utility_gate_query_target_mean",
            "utility_gate_delta_corr_train",
            "utility_gate_query_delta_corr",
            "query_proto_alignment_loss",
            "query_proto_supervised_count",
            "query_proto_class_count",
            "query_proto_mean_delta_dist",
            "query_proto_positive_ratio",
            "support_query_enabled",
            "support_count",
            "query_count",
            "train_support_ratio",
            "use_utility_receive_gate",
            "utility_receive_gate_mean",
            "utility_receive_gate_min",
            "utility_receive_gate_max",
            "utility_receive_gate_floor",
            "benefit_gate_mean",
            "benefit_gate_min",
            "benefit_gate_max",
            "use_hard_receive_gate",
            "hard_receive_ratio",
            "hard_receive_selected_ratio",
            "receive_gate_mean",
            "receive_gate_min",
            "receive_gate_max",
            "class_route",
            "key_proto",
            "class_router_hit_rate_train",
            "class_router_hit_rate_val",
            "class_router_hit_rate_test",
            "prompt_label_purity_train",
            "residual_prompt_usage_ratio",
            "same_class_route_compactness",
            "different_class_route_separation",
            "prompt_usage_entropy",
            "prompt_usage_full_entropy",
            "dominant_prompt_slot_ratio",
            "active_prompt_slot_count@0.05",
            "prompt_msg_norm",
            "prompt_to_original_msg_norm",
            "node_to_prompt_msg_norm",
            "prompt_to_original_update_norm",
            "correction_norm",
            "node_to_prompt_update_norm",
            "raw_prompt_update_norm",
            "unbounded_prompt_update_norm",
            "prompt_update_clip_ratio",
            "bounded_correction_clip_ratio",
            "prompt_gate_mean",
            "prompt_gate_node_to_prompt_mean",
            "prompt_gate_prompt_to_node_mean",
            "prompt_receiver_gate_mean",
            "prompt_message_scale",
            "selected_message_scale",
            "prompt_message_norm",
            "bounded_prompt_update",
            "max_prompt_update_norm",
            "prompt_update_bound_mode",
            "receiver_version",
            "prompt_fusion",
            "prompt_slot_head_count",
            "prototype_direction_strength_mean",
            "prototype_direction_normalize",
            "pool_only_prompt_update",
            "zero_init_prompt_messages",
            "adapted_branch_delta_norm",
            "capacity_routing_enabled",
            "prompt_capacity",
            "capacity_overflow_count",
            "edge_type_counts",
            "init_original_x_delta",
            "init_logit_delta",
            "init_logit_delta_full_edge_scale",
            "early_stopped",
            "stopped_epoch",
            "diagnostic_message_scale",
            "utility_mean_delta_ce",
            "utility_positive_delta_ratio",
            "utility_mean_ce_no_prompt",
            "utility_mean_ce_prompt",
            "utility_mean_delta_ce_train_pool",
            "utility_positive_delta_ratio_train_pool",
            "utility_mean_delta_ce_val_pool",
            "utility_positive_delta_ratio_val_pool",
            "utility_mean_delta_ce_test_pool",
            "utility_positive_delta_ratio_test_pool",
            "utility_acceptance_delta_correlation",
            "utility_structural_score_delta_correlation",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for result in results:
            writer.writerow(
                {
                    "seed": result["seed"],
                    "best_epoch": result["best"].get("best_epoch", 0.0),
                    "best_test_acc": result["best"].get("test_acc", 0.0),
                    "best_test_macro_f1": result["best"].get("test_macro_f1", 0.0),
                    "final_test_acc": result["final"].get("test_acc", 0.0),
                    "final_test_macro_f1": result["final"].get("test_macro_f1", 0.0),
                    "alpha": result["best"].get("alpha", 0.0),
                    "branch_cosine": result["best"].get("branch_cosine", 0.0),
                    "rho": result.get("rho", 0.0),
                    "pool_ratio": result["best"].get("pool_ratio", 0.0),
                    "train_pool_ratio": result["best"].get("train_pool_ratio", 0.0),
                    "prompt_node_count": result["best"].get("prompt_node_count", 0),
                    "prompt_edge_count": result["best"].get("prompt_edge_count", 0),
                    "edge_scale": result["best"].get("edge_scale", 0.0),
                    "raw_edge_scale": result["best"].get("raw_edge_scale", 0.0),
                    "edge_scale_multiplier": result["best"].get("edge_scale_multiplier", 0.0),
                    "use_multiview_routing": result["best"].get("use_multiview_routing", 0.0),
                    "use_class_aware_routing": result["best"].get("use_class_aware_routing", 0.0),
                    "use_pattern_prompt_bank": result["best"].get("use_pattern_prompt_bank", 0.0),
                    "use_receiver_only_prompt": result["best"].get("use_receiver_only_prompt", 0.0),
                    "use_benefit_gate": result["best"].get("use_benefit_gate", 0.0),
                    "num_class_prompt_slots": result["best"].get("num_class_prompt_slots", 0.0),
                    "residual_prompt_count": result["best"].get("residual_prompt_count", 0.0),
                    "class_key_proto_init_coverage": result["best"].get("class_key_proto_init_coverage", 0.0),
                    "pattern_key_init_coverage": result["best"].get("pattern_key_init_coverage", 0.0),
                    "semantic_view_weight": result["best"].get("semantic_view_weight", 0.0),
                    "structural_view_weight": result["best"].get("structural_view_weight", 0.0),
                    "role_view_weight": result["best"].get("role_view_weight", 0.0),
                    "view_gate_entropy": result["best"].get("view_gate_entropy", 0.0),
                    "mean_prompt_edge_weight": result["best"].get("mean_prompt_edge_weight", 0.0),
                    "pool_acceptance_mean": result["best"].get("pool_acceptance_mean", 1.0),
                    "pool_acceptance_min": result["best"].get("pool_acceptance_min", 1.0),
                    "pool_acceptance_max": result["best"].get("pool_acceptance_max", 1.0),
                    "use_hard_acceptance": result["best"].get("use_hard_acceptance", 0.0),
                    "hard_acceptance_ratio": result["best"].get("hard_acceptance_ratio", 0.0),
                    "hard_acceptance_selected_ratio": result["best"].get("hard_acceptance_selected_ratio", 0.0),
                    "prompt_acceptance_budget": result["best"].get("prompt_acceptance_budget", 0.0),
                    "prompt_acceptance_supervision": result["best"].get("prompt_acceptance_supervision", 0.0),
                    "acceptance_supervised_count": result["best"].get("acceptance_supervised_count", 0.0),
                    "acceptance_positive_count": result["best"].get("acceptance_positive_count", 0.0),
                    "acceptance_negative_count": result["best"].get("acceptance_negative_count", 0.0),
                    "acceptance_ignored_count": result["best"].get("acceptance_ignored_count", 0.0),
                    "acceptance_target_mean": result["best"].get("acceptance_target_mean", 0.0),
                    "acceptance_score_delta_mean": result["best"].get("acceptance_score_delta_mean", 0.0),
                    "ce_delta_positive_ratio_train": result["best"].get("ce_delta_positive_ratio_train", 0.0),
                    "ce_delta_negative_ratio_train": result["best"].get("ce_delta_negative_ratio_train", 0.0),
                    "prompt_helpful_acceptance_precision_train": result["best"].get(
                        "prompt_helpful_acceptance_precision_train", 0.0
                    ),
                    "prompt_harmful_rejection_precision_train": result["best"].get(
                        "prompt_harmful_rejection_precision_train", 0.0
                    ),
                    "prompt_benefit_supervision": result["best"].get("prompt_benefit_supervision", 0.0),
                    "benefit_supervised_count": result["best"].get("benefit_supervised_count", 0.0),
                    "benefit_positive_count": result["best"].get("benefit_positive_count", 0.0),
                    "benefit_negative_count": result["best"].get("benefit_negative_count", 0.0),
                    "benefit_ignored_count": result["best"].get("benefit_ignored_count", 0.0),
                    "mean_delta_ce_train_pool": result["best"].get("mean_delta_ce_train_pool", 0.0),
                    "positive_delta_ratio_train_pool": result["best"].get("positive_delta_ratio_train_pool", 0.0),
                    "benefit_weight_delta_corr_train": result["best"].get("benefit_weight_delta_corr_train", 0.0),
                    "benefit_label_strategy_quantile": result["best"].get("benefit_label_strategy_quantile", 0.0),
                    "benefit_quantile_fraction": result["best"].get("benefit_quantile_fraction", 0.0),
                    "benefit_positive_threshold": result["best"].get("benefit_positive_threshold", 0.0),
                    "benefit_negative_threshold": result["best"].get("benefit_negative_threshold", 0.0),
                    "benefit_delta_eps": result["best"].get("benefit_delta_eps", 0.0),
                    "prompt_correction_loss": result["best"].get("prompt_correction_loss", 0.0),
                    "prompt_anti_harm_loss": result["best"].get("prompt_anti_harm_loss", 0.0),
                    "prompt_correction_supervised_count": result["best"].get("prompt_correction_supervised_count", 0.0),
                    "prompt_correction_harmful_count": result["best"].get("prompt_correction_harmful_count", 0.0),
                    "prompt_correction_target_norm": result["best"].get("prompt_correction_target_norm", 0.0),
                    "prompt_delta_logit_norm": result["best"].get("prompt_delta_logit_norm", 0.0),
                    "prompt_delta_logit_label_alignment": result["best"].get("prompt_delta_logit_label_alignment", 0.0),
                    "harmful_prompt_delta_ratio": result["best"].get("harmful_prompt_delta_ratio", 0.0),
                    "prompt_message_help_loss": result["best"].get("prompt_message_help_loss", 0.0),
                    "prompt_message_help_query_loss": result["best"].get("prompt_message_help_query_loss", 0.0),
                    "prompt_class_anti_harm_loss": result["best"].get("prompt_class_anti_harm_loss", 0.0),
                    "class_balanced_mean_delta_ce": result["best"].get("class_balanced_mean_delta_ce", 0.0),
                    "class_balanced_positive_delta_ratio": result["best"].get(
                        "class_balanced_positive_delta_ratio", 0.0
                    ),
                    "query_class_balanced_mean_delta_ce": result["best"].get(
                        "query_class_balanced_mean_delta_ce", 0.0
                    ),
                    "query_class_balanced_positive_delta_ratio": result["best"].get(
                        "query_class_balanced_positive_delta_ratio", 0.0
                    ),
                    "message_help_supervised_count": result["best"].get("message_help_supervised_count", 0.0),
                    "message_help_query_supervised_count": result["best"].get(
                        "message_help_query_supervised_count", 0.0
                    ),
                    "message_help_class_count": result["best"].get("message_help_class_count", 0.0),
                    "message_help_query_class_count": result["best"].get("message_help_query_class_count", 0.0),
                    "message_help_margin": result["best"].get("message_help_margin", 0.0),
                    "message_help_class_balanced": result["best"].get("message_help_class_balanced", 0.0),
                    "prompt_class_anti_harm_floor": result["best"].get("prompt_class_anti_harm_floor", 0.0),
                    "utility_receive_gate_loss": result["best"].get("utility_receive_gate_loss", 0.0),
                    "utility_receive_gate_query_loss": result["best"].get("utility_receive_gate_query_loss", 0.0),
                    "utility_gate_supervised_count": result["best"].get("utility_gate_supervised_count", 0.0),
                    "utility_gate_query_supervised_count": result["best"].get(
                        "utility_gate_query_supervised_count", 0.0
                    ),
                    "utility_gate_positive_count": result["best"].get("utility_gate_positive_count", 0.0),
                    "utility_gate_query_positive_count": result["best"].get(
                        "utility_gate_query_positive_count", 0.0
                    ),
                    "utility_gate_negative_count": result["best"].get("utility_gate_negative_count", 0.0),
                    "utility_gate_query_negative_count": result["best"].get(
                        "utility_gate_query_negative_count", 0.0
                    ),
                    "utility_gate_ignored_count": result["best"].get("utility_gate_ignored_count", 0.0),
                    "utility_gate_query_ignored_count": result["best"].get(
                        "utility_gate_query_ignored_count", 0.0
                    ),
                    "utility_gate_target_mean": result["best"].get("utility_gate_target_mean", 0.0),
                    "utility_gate_query_target_mean": result["best"].get("utility_gate_query_target_mean", 0.0),
                    "utility_gate_delta_corr_train": result["best"].get("utility_gate_delta_corr_train", 0.0),
                    "utility_gate_query_delta_corr": result["best"].get("utility_gate_query_delta_corr", 0.0),
                    "query_proto_alignment_loss": result["best"].get("query_proto_alignment_loss", 0.0),
                    "query_proto_supervised_count": result["best"].get("query_proto_supervised_count", 0.0),
                    "query_proto_class_count": result["best"].get("query_proto_class_count", 0.0),
                    "query_proto_mean_delta_dist": result["best"].get("query_proto_mean_delta_dist", 0.0),
                    "query_proto_positive_ratio": result["best"].get("query_proto_positive_ratio", 0.0),
                    "support_query_enabled": result["best"].get("support_query_enabled", 0.0),
                    "support_count": result["best"].get("support_count", 0.0),
                    "query_count": result["best"].get("query_count", 0.0),
                    "train_support_ratio": result["best"].get("train_support_ratio", 1.0),
                    "use_utility_receive_gate": result["best"].get("use_utility_receive_gate", 0.0),
                    "utility_receive_gate_mean": result["best"].get("utility_receive_gate_mean", 1.0),
                    "utility_receive_gate_min": result["best"].get("utility_receive_gate_min", 1.0),
                    "utility_receive_gate_max": result["best"].get("utility_receive_gate_max", 1.0),
                    "utility_receive_gate_floor": result["best"].get("utility_receive_gate_floor", 0.0),
                    "benefit_gate_mean": result["best"].get("benefit_gate_mean", 1.0),
                    "benefit_gate_min": result["best"].get("benefit_gate_min", 1.0),
                    "benefit_gate_max": result["best"].get("benefit_gate_max", 1.0),
                    "use_hard_receive_gate": result["best"].get("use_hard_receive_gate", 0.0),
                    "hard_receive_ratio": result["best"].get("hard_receive_ratio", 0.0),
                    "hard_receive_selected_ratio": result["best"].get("hard_receive_selected_ratio", 0.0),
                    "receive_gate_mean": result["best"].get("receive_gate_mean", 1.0),
                    "receive_gate_min": result["best"].get("receive_gate_min", 1.0),
                    "receive_gate_max": result["best"].get("receive_gate_max", 1.0),
                    "class_route": result["best"].get("class_route", 0.0),
                    "key_proto": result["best"].get("key_proto", 0.0),
                    "class_router_hit_rate_train": result["best"].get("class_router_hit_rate_train", 0.0),
                    "class_router_hit_rate_val": result["best"].get("class_router_hit_rate_val", 0.0),
                    "class_router_hit_rate_test": result["best"].get("class_router_hit_rate_test", 0.0),
                    "prompt_label_purity_train": result["best"].get("prompt_label_purity_train", 0.0),
                    "residual_prompt_usage_ratio": result["best"].get("residual_prompt_usage_ratio", 0.0),
                    "same_class_route_compactness": result["best"].get("same_class_route_compactness", 0.0),
                    "different_class_route_separation": result["best"].get("different_class_route_separation", 0.0),
                    "prompt_usage_entropy": result["best"].get("prompt_usage_entropy", 0.0),
                    "prompt_usage_full_entropy": result["best"].get("prompt_usage_full_entropy", 0.0),
                    "dominant_prompt_slot_ratio": result["best"].get("dominant_prompt_slot_ratio", 0.0),
                    "active_prompt_slot_count@0.05": result["best"].get("active_prompt_slot_count@0.05", 0.0),
                    "prompt_msg_norm": result["best"].get("prompt_msg_norm", 0.0),
                    "prompt_to_original_msg_norm": result["best"].get("prompt_to_original_msg_norm", 0.0),
                    "node_to_prompt_msg_norm": result["best"].get("node_to_prompt_msg_norm", 0.0),
                    "prompt_to_original_update_norm": result["best"].get("prompt_to_original_update_norm", 0.0),
                    "correction_norm": result["best"].get("correction_norm", 0.0),
                    "node_to_prompt_update_norm": result["best"].get("node_to_prompt_update_norm", 0.0),
                    "raw_prompt_update_norm": result["best"].get("raw_prompt_update_norm", 0.0),
                    "unbounded_prompt_update_norm": result["best"].get("unbounded_prompt_update_norm", 0.0),
                    "prompt_update_clip_ratio": result["best"].get("prompt_update_clip_ratio", 0.0),
                    "bounded_correction_clip_ratio": result["best"].get("bounded_correction_clip_ratio", 0.0),
                    "prompt_gate_mean": result["best"].get("prompt_gate_mean", 0.0),
                    "prompt_gate_node_to_prompt_mean": result["best"].get("prompt_gate_node_to_prompt_mean", 0.0),
                    "prompt_gate_prompt_to_node_mean": result["best"].get("prompt_gate_prompt_to_node_mean", 0.0),
                    "prompt_receiver_gate_mean": result["best"].get("prompt_receiver_gate_mean", 0.0),
                    "prompt_message_scale": result["best"].get("prompt_message_scale", 0.0),
                    "selected_message_scale": result.get("selected_message_scale", result["best"].get("prompt_message_scale", 0.0)),
                    "prompt_message_norm": result["best"].get("prompt_message_norm", ""),
                    "bounded_prompt_update": result["best"].get("bounded_prompt_update", 0.0),
                    "max_prompt_update_norm": result["best"].get("max_prompt_update_norm", 0.0),
                    "prompt_update_bound_mode": result["best"].get("prompt_update_bound_mode", ""),
                    "receiver_version": result["best"].get("receiver_version", ""),
                    "prompt_fusion": result["best"].get("prompt_fusion", ""),
                    "prompt_slot_head_count": result["best"].get("prompt_slot_head_count", 0.0),
                    "prototype_direction_strength_mean": result["best"].get(
                        "prototype_direction_strength_mean", 0.0
                    ),
                    "prototype_direction_normalize": result["best"].get("prototype_direction_normalize", 0.0),
                    "pool_only_prompt_update": result["best"].get("pool_only_prompt_update", 0.0),
                    "zero_init_prompt_messages": result["best"].get("zero_init_prompt_messages", 0.0),
                    "adapted_branch_delta_norm": result["best"].get("adapted_branch_delta_norm", 0.0),
                    "capacity_routing_enabled": result["best"].get("capacity_routing_enabled", 0.0),
                    "prompt_capacity": result["best"].get("prompt_capacity", 0.0),
                    "capacity_overflow_count": result["best"].get("capacity_overflow_count", 0.0),
                    "edge_type_counts": result["best"].get("edge_type_counts", [0, 0, 0]),
                    "init_original_x_delta": result["best"].get("init_original_x_delta", 0.0),
                    "init_logit_delta": result["best"].get("init_logit_delta", 0.0),
                    "init_logit_delta_full_edge_scale": result["best"].get("init_logit_delta_full_edge_scale", 0.0),
                    "early_stopped": result.get("early_stopped", False),
                    "stopped_epoch": result.get("stopped_epoch", 0),
                    "diagnostic_message_scale": result.get("diagnostic_message_scale", ""),
                    "utility_mean_delta_ce": result.get("prompt_message_utility", {}).get("mean_delta_ce", ""),
                    "utility_positive_delta_ratio": result.get("prompt_message_utility", {}).get(
                        "positive_delta_ratio", ""
                    ),
                    "utility_mean_ce_no_prompt": result.get("prompt_message_utility", {}).get("mean_ce_no_prompt", ""),
                    "utility_mean_ce_prompt": result.get("prompt_message_utility", {}).get("mean_ce_prompt", ""),
                    "utility_mean_delta_ce_train_pool": result.get("prompt_message_utility", {}).get(
                        "mean_delta_ce_train_pool", ""
                    ),
                    "utility_positive_delta_ratio_train_pool": result.get("prompt_message_utility", {}).get(
                        "positive_delta_ratio_train_pool", ""
                    ),
                    "utility_mean_delta_ce_val_pool": result.get("prompt_message_utility", {}).get(
                        "mean_delta_ce_val_pool", ""
                    ),
                    "utility_positive_delta_ratio_val_pool": result.get("prompt_message_utility", {}).get(
                        "positive_delta_ratio_val_pool", ""
                    ),
                    "utility_mean_delta_ce_test_pool": result.get("prompt_message_utility", {}).get(
                        "mean_delta_ce_test_pool", ""
                    ),
                    "utility_positive_delta_ratio_test_pool": result.get("prompt_message_utility", {}).get(
                        "positive_delta_ratio_test_pool", ""
                    ),
                    "utility_acceptance_delta_correlation": result.get("prompt_message_utility", {}).get(
                        "acceptance_delta_correlation", ""
                    ),
                    "utility_structural_score_delta_correlation": result.get("prompt_message_utility", {}).get(
                        "structural_score_delta_correlation", ""
                    ),
                }
            )
    print("=" * 72)
    print(f"Summary | dataset={target_dataset} | variant={variant} | runs={len(seeds)}")
    print(f"Best Test Acc:      {summary['best_test_acc_mean_std']}")
    print(f"Best Test Macro-F1: {summary['best_test_macro_f1_mean_std']}")
    print(f"Final Test Acc:     {summary['final_test_acc_mean_std']}")
    print(f"Final Test Macro-F1: {summary['final_test_macro_f1_mean_std']}")
    print(f"Saved summary to {summary_dir / 'summary.json'}")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="GP2F prompt graph P1 runner")
    parser.add_argument("--config", type=str, default="configs/gp2f_prompt_p1.yaml")
    parser.add_argument("--target_dataset", type=str, default=None)
    parser.add_argument("--shots", type=int, default=None)
    parser.add_argument("--shot_ratio", type=float, default=None)
    parser.add_argument("--shot_mode", type=str, choices=["shots", "percent"], default=None)
    parser.add_argument("--shot_value", type=float, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--seeds", type=str, default=None)
    parser.add_argument("--runs", type=int, default=None)
    parser.add_argument("--prompt_variant", type=str, choices=sorted(PROMPT_GRAPH_VARIANTS), default=None)
    parser.add_argument("--rho", type=float, default=None)
    parser.add_argument("--num_prompt_nodes", type=int, default=None)
    parser.add_argument("--topk_prompt_per_node", type=int, default=None)
    parser.add_argument("--capacity_factor", type=float, default=None)
    parser.add_argument("--enable_capacity_routing", action="store_true")
    parser.add_argument("--disable_capacity_routing", action="store_true")
    parser.add_argument("--edge_scale_init", type=float, default=None)
    parser.add_argument("--edge_scale_max", type=float, default=None)
    parser.add_argument("--edge_scale_warmup_epochs", type=int, default=None)
    parser.add_argument("--edge_scale_warmup_start", type=float, default=None)
    parser.add_argument("--tau", type=float, default=None)
    parser.add_argument("--prompt_gate_init", type=float, default=None)
    parser.add_argument("--node_to_prompt_gate_init", type=float, default=None)
    parser.add_argument("--prompt_to_node_gate_init", type=float, default=None)
    parser.add_argument(
        "--prompt_message_norm",
        type=str,
        choices=["weighted_mean", "weighted_sum", "degree_mean", "layernorm"],
        default=None,
    )
    parser.add_argument(
        "--receiver_version",
        type=str,
        choices=[
            "v1_linear",
            "v2_conditioned",
            "v3_node_residual",
            "v4_multi_expert_residual",
            "v5_prototype_directional",
            "v6_classifier_directional",
        ],
        default=None,
    )
    parser.add_argument("--prompt_message_scale", type=float, default=None)
    parser.add_argument("--prompt_message_scale_grid", type=str, default=None)
    parser.add_argument("--diagnose_prompt_message_utility", action="store_true")
    parser.add_argument("--diagnostic_message_scales", type=str, default=None)
    parser.add_argument("--message_scale_selection_metric", type=str, default=None)
    parser.add_argument("--zero_init_prompt_messages", action="store_true")
    parser.add_argument("--disable_zero_init_prompt_messages", action="store_true")
    parser.add_argument("--pool_only_prompt_update", action="store_true")
    parser.add_argument("--disable_pool_only_prompt_update", action="store_true")
    parser.add_argument("--enable_bounded_prompt_update", action="store_true")
    parser.add_argument("--disable_bounded_prompt_update", action="store_true")
    parser.add_argument("--max_prompt_update_norm", type=float, default=None)
    parser.add_argument("--prompt_update_bound_mode", type=str, choices=["norm_clip", "tanh"], default=None)
    parser.add_argument("--lambda_edge_l1", type=float, default=None)
    parser.add_argument("--lambda_prompt_balance", type=float, default=None)
    parser.add_argument("--lambda_prompt_usage_consistency", type=float, default=None)
    parser.add_argument("--lambda_prompt_view_entropy", type=float, default=None)
    parser.add_argument("--lambda_class_route", type=float, default=None)
    parser.add_argument("--lambda_key_proto", type=float, default=None)
    parser.add_argument("--lambda_prompt_benefit_supervision", type=float, default=None)
    parser.add_argument("--benefit_supervision_warmup_epochs", type=int, default=None)
    parser.add_argument("--benefit_delta_margin", type=float, default=None)
    parser.add_argument(
        "--benefit_supervision_label_strategy",
        type=str,
        choices=["margin", "quantile", "hybrid_quantile_margin"],
        default=None,
    )
    parser.add_argument("--benefit_supervision_quantile", type=float, default=None)
    parser.add_argument("--benefit_delta_eps", type=float, default=None)
    parser.add_argument("--benefit_supervision_quantile_warmup_epochs", type=int, default=None)
    parser.add_argument("--benefit_supervision_probe_scale", type=float, default=None)
    parser.add_argument("--lambda_prompt_correction", type=float, default=None)
    parser.add_argument("--lambda_prompt_anti_harm", type=float, default=None)
    parser.add_argument("--prompt_correction_warmup_epochs", type=int, default=None)
    parser.add_argument("--prompt_correction_eps", type=float, default=None)
    parser.add_argument("--prompt_correction_target", type=str, choices=["delta_prob_to_label"], default=None)
    parser.add_argument("--lambda_prompt_message_help", type=float, default=None)
    parser.add_argument("--prompt_message_help_margin", type=float, default=None)
    parser.add_argument("--prompt_message_help_warmup_epochs", type=int, default=None)
    parser.add_argument("--lambda_prompt_class_anti_harm", type=float, default=None)
    parser.add_argument("--prompt_class_anti_harm_floor", type=float, default=None)
    parser.add_argument("--lambda_utility_receive_gate", type=float, default=None)
    parser.add_argument("--utility_receive_gate_warmup_epochs", type=int, default=None)
    parser.add_argument("--utility_receive_gate_quantile", type=float, default=None)
    parser.add_argument("--utility_receive_gate_eps", type=float, default=None)
    parser.add_argument("--enable_utility_receive_gate", action="store_true")
    parser.add_argument("--disable_utility_receive_gate", action="store_true")
    parser.add_argument("--utility_receive_gate_min", type=float, default=None)
    parser.add_argument("--utility_receive_gate_init", type=float, default=None)
    parser.add_argument("--enable_prompt_message_help_class_balanced", action="store_true")
    parser.add_argument("--disable_prompt_message_help_class_balanced", action="store_true")
    parser.add_argument("--enable_hard_receive_gate", action="store_true")
    parser.add_argument("--disable_hard_receive_gate", action="store_true")
    parser.add_argument("--hard_receive_ratio", type=float, default=None)
    parser.add_argument("--residual_prompt_count", type=int, default=None)
    parser.add_argument("--enable_class_aware_routing", action="store_true")
    parser.add_argument("--disable_class_aware_routing", action="store_true")
    parser.add_argument("--enable_class_key_proto_init", action="store_true")
    parser.add_argument("--disable_class_key_proto_init", action="store_true")
    parser.add_argument("--lambda_prompt_acceptance_budget", type=float, default=None)
    parser.add_argument("--acceptance_budget_min", type=float, default=None)
    parser.add_argument("--acceptance_budget_max", type=float, default=None)
    parser.add_argument("--enable_hard_acceptance", action="store_true")
    parser.add_argument("--disable_hard_acceptance", action="store_true")
    parser.add_argument("--hard_acceptance_ratio", type=float, default=None)
    parser.add_argument("--lambda_prompt_acceptance_supervision", type=float, default=None)
    parser.add_argument("--acceptance_supervision_signal", type=str, default=None)
    parser.add_argument("--acceptance_supervision_positive_margin", type=float, default=None)
    parser.add_argument("--acceptance_supervision_negative_margin", type=float, default=None)
    parser.add_argument("--enable_multiview_routing", action="store_true")
    parser.add_argument("--disable_multiview_routing", action="store_true")
    parser.add_argument("--base_checkpoint_path", type=str, default=None)
    parser.add_argument("--base_checkpoint_root", type=str, default=None)
    parser.add_argument("--freeze_base", action="store_true")
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--prompt_lr", type=float, default=None)
    parser.add_argument("--weight_decay", type=float, default=None)
    parser.add_argument("--prompt_weight_decay", type=float, default=None)
    parser.add_argument("--patience", type=int, default=None)
    parser.add_argument("--eval_every", type=int, default=None)
    parser.add_argument("--output_dir", type=str, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    config = read_yaml(_resolve_path(args.config, base_dir=repo_root))
    overrides: dict[str, Any] = {}
    if args.target_dataset is not None:
        overrides.setdefault("experiment", {})["target_dataset"] = args.target_dataset
    if args.prompt_variant is not None:
        overrides.setdefault("experiment", {})["prompt_variant"] = args.prompt_variant
    if args.shots is not None:
        overrides.setdefault("data", {})["shots"] = int(args.shots)
        overrides.setdefault("data", {})["shot_ratio"] = None
    if args.shot_ratio is not None:
        overrides.setdefault("data", {})["shot_ratio"] = float(args.shot_ratio)
    if args.shot_value is not None:
        if args.shot_mode == "percent":
            overrides.setdefault("data", {})["shot_ratio"] = float(args.shot_value)
        elif args.shot_mode == "shots":
            overrides.setdefault("data", {})["shots"] = int(args.shot_value)
            overrides.setdefault("data", {})["shot_ratio"] = None
        else:
            raise ValueError("--shot_value requires --shot_mode to be either 'percent' or 'shots'")
    if args.seed is not None:
        overrides.setdefault("experiment", {})["seed"] = args.seed
    if args.seeds is not None:
        overrides.setdefault("experiment", {})["seeds"] = [int(piece.strip()) for piece in args.seeds.split(",") if piece.strip()]
    if args.runs is not None:
        overrides.setdefault("experiment", {})["runs"] = args.runs
        overrides.setdefault("experiment", {})["seeds"] = None
    if args.epochs is not None:
        overrides.setdefault("training", {})["epochs"] = args.epochs
    if args.lr is not None:
        overrides.setdefault("training", {})["lr"] = args.lr
    if args.prompt_lr is not None:
        overrides.setdefault("training", {})["prompt_lr"] = args.prompt_lr
    if args.weight_decay is not None:
        overrides.setdefault("training", {})["weight_decay"] = args.weight_decay
    if args.prompt_weight_decay is not None:
        overrides.setdefault("training", {})["prompt_weight_decay"] = args.prompt_weight_decay
    if args.patience is not None:
        overrides.setdefault("training", {})["early_stop_patience"] = args.patience
    if args.eval_every is not None:
        overrides.setdefault("training", {})["eval_every"] = args.eval_every
    if args.output_dir is not None:
        overrides.setdefault("training", {})["output_dir"] = args.output_dir
    if args.base_checkpoint_path is not None:
        overrides.setdefault("training", {})["base_checkpoint_path"] = args.base_checkpoint_path
    if args.base_checkpoint_root is not None:
        overrides.setdefault("training", {})["base_checkpoint_root"] = args.base_checkpoint_root
    if args.freeze_base:
        overrides.setdefault("training", {})["freeze_base_model"] = True
    if args.rho is not None:
        overrides.setdefault("prompt_graph", {})["rho"] = float(args.rho)
    if args.num_prompt_nodes is not None:
        overrides.setdefault("prompt_graph", {})["num_prompt_nodes"] = int(args.num_prompt_nodes)
    if args.topk_prompt_per_node is not None:
        overrides.setdefault("prompt_graph", {})["topk_prompt_per_node"] = int(args.topk_prompt_per_node)
    if args.capacity_factor is not None:
        overrides.setdefault("prompt_graph", {})["capacity_factor"] = float(args.capacity_factor)
    if args.enable_capacity_routing:
        overrides.setdefault("prompt_graph", {})["use_capacity_routing"] = True
    if args.disable_capacity_routing:
        overrides.setdefault("prompt_graph", {})["use_capacity_routing"] = False
    if args.edge_scale_init is not None:
        overrides.setdefault("prompt_graph", {})["edge_scale_init"] = float(args.edge_scale_init)
    if args.edge_scale_max is not None:
        overrides.setdefault("prompt_graph", {})["edge_scale_max"] = float(args.edge_scale_max)
    if args.edge_scale_warmup_epochs is not None:
        overrides.setdefault("prompt_graph", {})["edge_scale_warmup_epochs"] = int(args.edge_scale_warmup_epochs)
    if args.edge_scale_warmup_start is not None:
        overrides.setdefault("prompt_graph", {})["edge_scale_warmup_start"] = float(args.edge_scale_warmup_start)
    if args.tau is not None:
        overrides.setdefault("prompt_graph", {})["tau"] = float(args.tau)
    if args.prompt_gate_init is not None:
        overrides.setdefault("prompt_aware", {})["gate_init"] = float(args.prompt_gate_init)
    if args.node_to_prompt_gate_init is not None:
        overrides.setdefault("prompt_aware", {})["node_to_prompt_gate_init"] = float(args.node_to_prompt_gate_init)
    if args.prompt_to_node_gate_init is not None:
        overrides.setdefault("prompt_aware", {})["prompt_to_node_gate_init"] = float(args.prompt_to_node_gate_init)
    if args.prompt_message_norm is not None:
        overrides.setdefault("prompt_aware", {})["prompt_message_norm"] = args.prompt_message_norm
    if args.receiver_version is not None:
        overrides.setdefault("prompt_aware", {})["receiver_version"] = args.receiver_version
    if args.prompt_message_scale is not None:
        overrides.setdefault("prompt_aware", {})["message_scale"] = float(args.prompt_message_scale)
        overrides.setdefault("prompt_aware", {})["message_scale_grid"] = [float(args.prompt_message_scale)]
    if args.prompt_message_scale_grid is not None:
        overrides.setdefault("prompt_aware", {})["message_scale_grid"] = [
            float(piece.strip()) for piece in args.prompt_message_scale_grid.split(",") if piece.strip()
        ]
    if args.diagnose_prompt_message_utility:
        overrides.setdefault("training", {})["diagnose_prompt_message_utility"] = True
        overrides.setdefault("prompt_aware", {})["message_scale_grid"] = [0.1, 0.25, 0.5, 1.0]
    if args.diagnostic_message_scales is not None:
        overrides.setdefault("training", {})["diagnose_prompt_message_utility"] = True
        overrides.setdefault("prompt_aware", {})["message_scale_grid"] = [
            float(piece.strip()) for piece in args.diagnostic_message_scales.split(",") if piece.strip()
        ]
    if args.message_scale_selection_metric is not None:
        overrides.setdefault("training", {})["message_scale_selection_metric"] = args.message_scale_selection_metric
    if args.zero_init_prompt_messages:
        overrides.setdefault("prompt_aware", {})["zero_init_prompt_messages"] = True
    if args.disable_zero_init_prompt_messages:
        overrides.setdefault("prompt_aware", {})["zero_init_prompt_messages"] = False
    if args.pool_only_prompt_update:
        overrides.setdefault("prompt_aware", {})["pool_only_prompt_update"] = True
    if args.disable_pool_only_prompt_update:
        overrides.setdefault("prompt_aware", {})["pool_only_prompt_update"] = False
    if args.enable_bounded_prompt_update:
        overrides.setdefault("prompt_aware", {})["use_bounded_prompt_update"] = True
    if args.disable_bounded_prompt_update:
        overrides.setdefault("prompt_aware", {})["use_bounded_prompt_update"] = False
    if args.max_prompt_update_norm is not None:
        overrides.setdefault("prompt_aware", {})["max_prompt_update_norm"] = float(args.max_prompt_update_norm)
    if args.prompt_update_bound_mode is not None:
        overrides.setdefault("prompt_aware", {})["prompt_update_bound_mode"] = args.prompt_update_bound_mode
    if args.lambda_edge_l1 is not None:
        overrides.setdefault("prompt_graph", {})["lambda_edge_l1"] = float(args.lambda_edge_l1)
    if args.lambda_prompt_balance is not None:
        overrides.setdefault("prompt_graph", {})["lambda_prompt_balance"] = float(args.lambda_prompt_balance)
    if args.lambda_prompt_usage_consistency is not None:
        overrides.setdefault("prompt_graph", {})["lambda_prompt_usage_consistency"] = float(args.lambda_prompt_usage_consistency)
    if args.lambda_prompt_view_entropy is not None:
        overrides.setdefault("prompt_graph", {})["lambda_prompt_view_entropy"] = float(args.lambda_prompt_view_entropy)
    if args.lambda_class_route is not None:
        overrides.setdefault("prompt_graph", {})["lambda_class_route"] = float(args.lambda_class_route)
    if args.lambda_key_proto is not None:
        overrides.setdefault("prompt_graph", {})["lambda_key_proto"] = float(args.lambda_key_proto)
    if args.lambda_prompt_benefit_supervision is not None:
        overrides.setdefault("prompt_graph", {})["lambda_prompt_benefit_supervision"] = float(
            args.lambda_prompt_benefit_supervision
        )
    if args.benefit_supervision_warmup_epochs is not None:
        overrides.setdefault("prompt_graph", {})["benefit_supervision_warmup_epochs"] = int(
            args.benefit_supervision_warmup_epochs
        )
    if args.benefit_delta_margin is not None:
        overrides.setdefault("prompt_graph", {})["benefit_delta_margin"] = float(args.benefit_delta_margin)
    if args.benefit_supervision_label_strategy is not None:
        overrides.setdefault("prompt_graph", {})["benefit_supervision_label_strategy"] = (
            args.benefit_supervision_label_strategy
        )
    if args.benefit_supervision_quantile is not None:
        overrides.setdefault("prompt_graph", {})["benefit_supervision_quantile"] = float(
            args.benefit_supervision_quantile
        )
    if args.benefit_delta_eps is not None:
        overrides.setdefault("prompt_graph", {})["benefit_delta_eps"] = float(args.benefit_delta_eps)
    if args.benefit_supervision_quantile_warmup_epochs is not None:
        overrides.setdefault("prompt_graph", {})["benefit_supervision_quantile_warmup_epochs"] = int(
            args.benefit_supervision_quantile_warmup_epochs
        )
    if args.benefit_supervision_probe_scale is not None:
        overrides.setdefault("prompt_graph", {})["benefit_supervision_probe_scale"] = float(
            args.benefit_supervision_probe_scale
        )
    if args.lambda_prompt_correction is not None:
        overrides.setdefault("prompt_graph", {})["lambda_prompt_correction"] = float(args.lambda_prompt_correction)
    if args.lambda_prompt_anti_harm is not None:
        overrides.setdefault("prompt_graph", {})["lambda_prompt_anti_harm"] = float(args.lambda_prompt_anti_harm)
    if args.prompt_correction_warmup_epochs is not None:
        overrides.setdefault("prompt_graph", {})["prompt_correction_warmup_epochs"] = int(
            args.prompt_correction_warmup_epochs
        )
    if args.prompt_correction_eps is not None:
        overrides.setdefault("prompt_graph", {})["prompt_correction_eps"] = float(args.prompt_correction_eps)
    if args.prompt_correction_target is not None:
        overrides.setdefault("prompt_graph", {})["prompt_correction_target"] = args.prompt_correction_target
    if args.lambda_prompt_message_help is not None:
        overrides.setdefault("prompt_graph", {})["lambda_prompt_message_help"] = float(
            args.lambda_prompt_message_help
        )
    if args.prompt_message_help_margin is not None:
        overrides.setdefault("prompt_graph", {})["prompt_message_help_margin"] = float(
            args.prompt_message_help_margin
        )
    if args.prompt_message_help_warmup_epochs is not None:
        overrides.setdefault("prompt_graph", {})["prompt_message_help_warmup_epochs"] = int(
            args.prompt_message_help_warmup_epochs
        )
    if args.lambda_prompt_class_anti_harm is not None:
        overrides.setdefault("prompt_graph", {})["lambda_prompt_class_anti_harm"] = float(
            args.lambda_prompt_class_anti_harm
        )
    if args.prompt_class_anti_harm_floor is not None:
        overrides.setdefault("prompt_graph", {})["prompt_class_anti_harm_floor"] = float(
            args.prompt_class_anti_harm_floor
        )
    if args.lambda_utility_receive_gate is not None:
        overrides.setdefault("prompt_graph", {})["lambda_utility_receive_gate"] = float(
            args.lambda_utility_receive_gate
        )
    if args.utility_receive_gate_warmup_epochs is not None:
        overrides.setdefault("prompt_graph", {})["utility_receive_gate_warmup_epochs"] = int(
            args.utility_receive_gate_warmup_epochs
        )
    if args.utility_receive_gate_quantile is not None:
        overrides.setdefault("prompt_graph", {})["utility_receive_gate_quantile"] = float(
            args.utility_receive_gate_quantile
        )
    if args.utility_receive_gate_eps is not None:
        overrides.setdefault("prompt_graph", {})["utility_receive_gate_eps"] = float(args.utility_receive_gate_eps)
    if args.enable_utility_receive_gate:
        overrides.setdefault("prompt_graph", {})["use_utility_receive_gate"] = True
    if args.disable_utility_receive_gate:
        overrides.setdefault("prompt_graph", {})["use_utility_receive_gate"] = False
    if args.utility_receive_gate_min is not None:
        overrides.setdefault("prompt_graph", {})["utility_receive_gate_min"] = float(args.utility_receive_gate_min)
    if args.utility_receive_gate_init is not None:
        overrides.setdefault("prompt_graph", {})["utility_receive_gate_init"] = float(args.utility_receive_gate_init)
    if args.enable_prompt_message_help_class_balanced:
        overrides.setdefault("prompt_graph", {})["prompt_message_help_class_balanced"] = True
    if args.disable_prompt_message_help_class_balanced:
        overrides.setdefault("prompt_graph", {})["prompt_message_help_class_balanced"] = False
    if args.enable_hard_receive_gate:
        overrides.setdefault("prompt_graph", {})["use_hard_receive_gate"] = True
    if args.disable_hard_receive_gate:
        overrides.setdefault("prompt_graph", {})["use_hard_receive_gate"] = False
    if args.hard_receive_ratio is not None:
        overrides.setdefault("prompt_graph", {})["hard_receive_ratio"] = float(args.hard_receive_ratio)
    if args.residual_prompt_count is not None:
        overrides.setdefault("prompt_graph", {})["residual_prompt_count"] = int(args.residual_prompt_count)
    if args.enable_class_aware_routing:
        overrides.setdefault("prompt_graph", {})["use_class_aware_routing"] = True
    if args.disable_class_aware_routing:
        overrides.setdefault("prompt_graph", {})["use_class_aware_routing"] = False
    if args.enable_class_key_proto_init:
        overrides.setdefault("prompt_graph", {})["init_class_keys_from_train_proto"] = True
    if args.disable_class_key_proto_init:
        overrides.setdefault("prompt_graph", {})["init_class_keys_from_train_proto"] = False
    if args.lambda_prompt_acceptance_budget is not None:
        overrides.setdefault("prompt_graph", {})["lambda_prompt_acceptance_budget"] = float(args.lambda_prompt_acceptance_budget)
    if args.acceptance_budget_min is not None:
        overrides.setdefault("prompt_graph", {})["acceptance_budget_min"] = float(args.acceptance_budget_min)
    if args.acceptance_budget_max is not None:
        overrides.setdefault("prompt_graph", {})["acceptance_budget_max"] = float(args.acceptance_budget_max)
    if args.enable_hard_acceptance:
        overrides.setdefault("prompt_graph", {})["use_hard_acceptance"] = True
    if args.disable_hard_acceptance:
        overrides.setdefault("prompt_graph", {})["use_hard_acceptance"] = False
    if args.hard_acceptance_ratio is not None:
        overrides.setdefault("prompt_graph", {})["hard_acceptance_ratio"] = float(args.hard_acceptance_ratio)
    if args.lambda_prompt_acceptance_supervision is not None:
        overrides.setdefault("prompt_graph", {})["lambda_prompt_acceptance_supervision"] = float(
            args.lambda_prompt_acceptance_supervision
        )
    if args.acceptance_supervision_signal is not None:
        overrides.setdefault("prompt_graph", {})["acceptance_supervision_signal"] = args.acceptance_supervision_signal
    if args.acceptance_supervision_positive_margin is not None:
        overrides.setdefault("prompt_graph", {})["acceptance_supervision_positive_margin"] = float(
            args.acceptance_supervision_positive_margin
        )
    if args.acceptance_supervision_negative_margin is not None:
        overrides.setdefault("prompt_graph", {})["acceptance_supervision_negative_margin"] = float(
            args.acceptance_supervision_negative_margin
        )
    if args.enable_multiview_routing:
        overrides.setdefault("prompt_graph", {})["use_multiview_routing"] = True
    if args.disable_multiview_routing:
        overrides.setdefault("prompt_graph", {})["use_multiview_routing"] = False
    run(_deep_update(config, overrides), repo_root=repo_root)


if __name__ == "__main__":
    main()
