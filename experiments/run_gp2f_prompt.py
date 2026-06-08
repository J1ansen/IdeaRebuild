"""Run the GP2F residual prompt P0 experiments."""

from __future__ import annotations

import argparse
import copy
import csv
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
from models import (
    FaithfulGP2F,
    ParameterMatchedResidualControl,
    UnifiedMultiViewResidualPrompt,
    load_pretrained_gcn,
)
from models.prompt_module import count_trainable_parameters, prompt_budget_loss, prompt_message_norm_loss
from utils.io import read_yaml, write_json, write_yaml
from utils.metrics import split_metrics


PROMPT_VARIANTS = {
    "full_p0",
    "noprompt",
    "semantic_only",
    "structural_only",
    "semantic_structural_no_null",
    "param_control",
}


def _route_budget_label(route_budget: float) -> str:
    return f"{float(route_budget):.4f}".rstrip("0").rstrip(".").replace(".", "p")


def _prompt_variant(config: dict[str, Any]) -> str:
    experiment_cfg = config.get("experiment", {})
    prompt_cfg = config.get("prompt", {})
    return str(experiment_cfg.get("prompt_variant", prompt_cfg.get("variant", "full_p0")))


def _config_for_variant(config: dict[str, Any], variant: str) -> dict[str, Any]:
    if variant not in PROMPT_VARIANTS:
        raise ValueError(f"Unsupported prompt_variant={variant!r}")
    out = copy.deepcopy(config)
    out.setdefault("experiment", {})["prompt_variant"] = variant
    prompt = out.setdefault("prompt", {})
    prompt.setdefault("semantic", {})
    prompt.setdefault("structural", {})
    prompt.setdefault("gate", {})

    if variant == "noprompt":
        prompt["enabled"] = False
    else:
        prompt["enabled"] = True

    if variant == "semantic_only":
        prompt["semantic"]["enabled"] = True
        prompt["structural"]["enabled"] = False
        prompt["gate"]["use_null"] = True
    elif variant == "structural_only":
        prompt["semantic"]["enabled"] = False
        prompt["structural"]["enabled"] = True
        prompt["gate"]["use_null"] = True
    elif variant == "semantic_structural_no_null":
        prompt["semantic"]["enabled"] = True
        prompt["structural"]["enabled"] = True
        prompt["gate"]["use_null"] = False
    elif variant in {"full_p0", "param_control"}:
        prompt["semantic"]["enabled"] = True
        prompt["structural"]["enabled"] = True
        prompt["gate"]["use_null"] = True
    return out


def _build_prompt_module(
    *,
    variant: str,
    source_dim: int,
    hidden_dim: int,
    num_classes: int,
    prompt_cfg: dict[str, Any],
    device: torch.device,
) -> torch.nn.Module | None:
    if variant == "noprompt":
        return None
    if variant == "param_control":
        return ParameterMatchedResidualControl(source_dim, hidden_dim, prompt_cfg).to(device)
    return UnifiedMultiViewResidualPrompt(source_dim, hidden_dim, num_classes, prompt_cfg).to(device)


def _default_prompt_out(z: torch.Tensor) -> dict[str, Any]:
    zeros = torch.zeros_like(z)
    gate = z.new_zeros((z.size(0), 3))
    gate[:, 2] = 1.0
    return {
        "adapted_x": z,
        "u_sem": zeros,
        "u_struct": zeros,
        "u_prompt": zeros,
        "gate": gate,
        "gamma": z.new_tensor(0.0),
        "aux": {
            "semantic_margin": z.new_zeros(z.size(0)),
            "semantic_allowed": torch.zeros(z.size(0), dtype=torch.bool, device=z.device),
            "connected_edge_count": 0,
        },
    }


def _norm_ratio(numerator: torch.Tensor, denominator: torch.Tensor) -> float:
    denom = denominator.norm().detach().clamp_min(1e-12)
    return float((numerator.norm().detach() / denom).item())


def _prompt_diagnostics(prompt_out: dict[str, Any], z: torch.Tensor) -> dict[str, float]:
    gate = prompt_out["gate"].detach()
    u_sem = prompt_out["u_sem"].detach()
    u_struct = prompt_out["u_struct"].detach()
    u_prompt = prompt_out["u_prompt"].detach()
    gamma = prompt_out["gamma"].detach()
    aux = prompt_out.get("aux", {})
    semantic_margin = aux.get("semantic_margin")
    semantic_allowed = aux.get("semantic_allowed")
    diagnostics = {
        "route_semantic": float(gate[:, 0].mean().item()),
        "route_structural": float(gate[:, 1].mean().item()),
        "route_null": float(gate[:, 2].mean().item()),
        "non_null_ratio": float((gate[:, 0] + gate[:, 1]).mean().item()),
        "gamma": float(gamma.item()),
        "prompt_message_norm": _norm_ratio(gamma * u_prompt, z.detach()),
        "u_sem_norm": _norm_ratio(u_sem, z.detach()),
        "u_struct_norm": _norm_ratio(u_struct, z.detach()),
        "u_prompt_norm": _norm_ratio(u_prompt, z.detach()),
        "semantic_margin_mean": float(semantic_margin.detach().mean().item()) if isinstance(semantic_margin, torch.Tensor) else 0.0,
        "semantic_allowed_ratio": float(semantic_allowed.float().mean().item()) if isinstance(semantic_allowed, torch.Tensor) else 0.0,
        "connected_edge_count": float(aux.get("connected_edge_count", 0)),
    }
    if isinstance(aux.get("var1"), torch.Tensor):
        diagnostics["structural_neighbor_var_norm"] = _norm_ratio(aux["var1"].detach(), z.detach())
    if isinstance(aux.get("degree"), torch.Tensor):
        diagnostics["structural_degree_mean"] = float(aux["degree"].detach().mean().item())
    if isinstance(aux.get("similarity"), torch.Tensor):
        diagnostics["structural_similarity_mean"] = float(aux["similarity"].detach().mean().item())
    if isinstance(aux.get("structural_cache_hit"), torch.Tensor):
        diagnostics["structural_cache_hit"] = float(aux["structural_cache_hit"].detach().item())
    return diagnostics


def _forward_prompt(
    *,
    model: FaithfulGP2F,
    prompt_module: torch.nn.Module | None,
    z: torch.Tensor,
    edge_index: torch.Tensor,
    train_mask: torch.Tensor,
    labels: torch.Tensor,
    split: str,
    structural_cache: dict[str, torch.Tensor] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    if prompt_module is None:
        prompt_out = _default_prompt_out(z)
        model_out = model(
            z,
            edge_index,
            adapted_x=prompt_out["adapted_x"],
            adapted_edge_index=edge_index,
            return_aux=True,
        )
    else:
        h_pre_prompt = model.encode_frozen(z, edge_index)
        prompt_out = prompt_module(
            z=z,
            h_pre=h_pre_prompt,
            edge_index=edge_index,
            train_mask=train_mask,
            y=labels,
            split=split,
            structural_cache=structural_cache,
        )
        model_out = model.forward_with_h_pre(
            z,
            edge_index,
            h_pre=h_pre_prompt,
            adapted_x=prompt_out["adapted_x"],
            adapted_edge_index=edge_index,
            return_aux=True,
        )
    return model_out, prompt_out


@torch.no_grad()
def _init_equivalence(
    *,
    model: FaithfulGP2F,
    prompt_module: torch.nn.Module | None,
    input_aligner: InputAligner,
    x: torch.Tensor,
    edge_index: torch.Tensor,
    train_mask: torch.Tensor,
    labels: torch.Tensor,
    structural_cache: dict[str, torch.Tensor] | None = None,
) -> dict[str, float]:
    model.eval()
    input_aligner.eval()
    if prompt_module is not None:
        prompt_module.eval()
    z = input_aligner(x)
    baseline = model(z, edge_index, return_aux=True)
    prompted, prompt_out = _forward_prompt(
        model=model,
        prompt_module=prompt_module,
        z=z,
        edge_index=edge_index,
        train_mask=train_mask,
        labels=labels,
        split="eval",
        structural_cache=structural_cache,
    )
    return {
        "init_adapted_x_delta": float((prompt_out["adapted_x"] - z).abs().max().item()),
        "init_logit_delta": float((prompted["logits"] - baseline["logits"]).abs().max().item()),
    }


@torch.no_grad()
def evaluate_prompt(
    *,
    model: FaithfulGP2F,
    prompt_module: torch.nn.Module | None,
    input_aligner: InputAligner,
    x: torch.Tensor,
    edge_index: torch.Tensor,
    labels: torch.Tensor,
    train_mask: torch.Tensor,
    val_mask: torch.Tensor,
    test_mask: torch.Tensor,
    num_classes: int,
    structural_cache: dict[str, torch.Tensor] | None = None,
) -> dict[str, float]:
    model.eval()
    input_aligner.eval()
    if prompt_module is not None:
        prompt_module.eval()
    z = input_aligner(x)
    model_out, prompt_out = _forward_prompt(
        model=model,
        prompt_module=prompt_module,
        z=z,
        edge_index=edge_index,
        train_mask=train_mask,
        labels=labels,
        split="eval",
        structural_cache=structural_cache,
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
        **_prompt_diagnostics(prompt_out, z),
    }


def _save_checkpoint(
    path: Path,
    *,
    model: FaithfulGP2F,
    input_aligner: InputAligner,
    prompt_module: torch.nn.Module | None,
    epoch: int,
    metrics: dict[str, float],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "input_aligner": input_aligner.state_dict(),
            "prompt_module": prompt_module.state_dict() if prompt_module is not None else None,
            "epoch": int(epoch),
            "metrics": metrics,
        },
        path,
    )


def _parameter_counts(
    *,
    variant: str,
    prompt_module: torch.nn.Module | None,
    source_dim: int,
    hidden_dim: int,
    prompt_cfg: dict[str, Any],
) -> dict[str, int]:
    prompt_count = count_trainable_parameters(prompt_module) if variant != "param_control" else 0
    if variant == "param_control":
        control_count = count_trainable_parameters(prompt_module)
    elif variant == "noprompt":
        control_count = 0
    else:
        control = ParameterMatchedResidualControl(source_dim, hidden_dim, prompt_cfg)
        control_count = count_trainable_parameters(control)
    return {
        "prompt_parameter_count": prompt_count,
        "control_parameter_count": control_count,
    }


def _trainable_parameter_summary(
    *,
    input_aligner: InputAligner,
    model: FaithfulGP2F,
    prompt_module: torch.nn.Module | None,
) -> dict[str, Any]:
    groups: dict[str, list[str]] = {}
    counts: dict[str, int] = {}
    modules: list[tuple[str, torch.nn.Module | None]] = [
        ("input_aligner", input_aligner),
        ("model", model),
        ("prompt_module", prompt_module),
    ]
    for group_name, module in modules:
        names: list[str] = []
        total = 0
        if module is not None:
            for name, param in module.named_parameters():
                if param.requires_grad:
                    names.append(f"{group_name}.{name}")
                    total += int(param.numel())
        groups[group_name] = names
        counts[f"{group_name}_trainable_parameter_count"] = total
    all_names = [name for names in groups.values() for name in names]
    return {
        "trainable_parameter_names": all_names,
        "trainable_parameter_groups": groups,
        "actual_trainable_parameter_count": sum(counts.values()),
        **counts,
    }


def _monitor_improved(monitor: str, value: float, best_value: float, has_best: bool) -> bool:
    if not has_best:
        return True
    return value < best_value if monitor in {"train_loss", "total", "cls", "budget", "message_norm"} else value > best_value


def _warmup_factor(epoch: int, warmup_epochs: int, *, delay_epochs: int = 0) -> float:
    if epoch <= int(delay_epochs):
        return 0.0
    if int(warmup_epochs) <= 0:
        return 1.0
    return min(1.0, max(0.0, (float(epoch) - float(delay_epochs)) / float(warmup_epochs)))


def _set_module_trainable(module: torch.nn.Module | None, trainable: bool) -> None:
    if module is None:
        return
    for parameter in module.parameters():
        parameter.requires_grad = bool(trainable)


def _trainable_parameters(module: torch.nn.Module | None) -> list[torch.nn.Parameter]:
    if module is None:
        return []
    return [parameter for parameter in module.parameters() if parameter.requires_grad]


def _optimizer_groups(
    *,
    input_aligner: InputAligner,
    model: FaithfulGP2F,
    prompt_module: torch.nn.Module | None,
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

    base_params = _trainable_parameters(input_aligner) + _trainable_parameters(model)
    prompt_params = _trainable_parameters(prompt_module)
    groups: list[dict[str, Any]] = []
    if base_params:
        groups.append({"params": base_params, "lr": base_lr, "weight_decay": base_weight_decay, "name": "base"})
    if prompt_params:
        groups.append({"params": prompt_params, "lr": prompt_lr, "weight_decay": prompt_weight_decay, "name": "prompt"})
    trainable_params = [param for group in groups for param in group["params"]]
    return groups, trainable_params, {
        "base_lr": base_lr if base_params else 0.0,
        "prompt_lr": prompt_lr if prompt_params else 0.0,
        "base_weight_decay": base_weight_decay if base_params else 0.0,
        "prompt_weight_decay": prompt_weight_decay if prompt_params else 0.0,
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
    prompt_module: torch.nn.Module | None,
    device: torch.device,
    load_prompt: bool = False,
) -> None:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model"])
    input_aligner.load_state_dict(checkpoint["input_aligner"])
    if load_prompt and prompt_module is not None and checkpoint.get("prompt_module") is not None:
        prompt_module.load_state_dict(checkpoint["prompt_module"])


@torch.no_grad()
def _maybe_build_structural_cache(
    *,
    prompt_module: torch.nn.Module | None,
    prompt_cfg: dict[str, Any],
    input_aligner: InputAligner,
    model: FaithfulGP2F,
    graph: Any,
    freeze_base_model: bool,
) -> tuple[dict[str, torch.Tensor] | None, dict[str, Any]]:
    structural_cfg = prompt_cfg.get("structural", {})
    cache_cfg = structural_cfg.get("cache", {})
    if not bool(cache_cfg.get("enabled", False)):
        return None, {
            "structural_cache_enabled": False,
            "structural_cache_used": False,
            "structural_cache_reason": "disabled",
        }
    if prompt_module is None or not hasattr(prompt_module, "build_structural_cache"):
        return None, {
            "structural_cache_enabled": True,
            "structural_cache_used": False,
            "structural_cache_reason": "unsupported_prompt_variant",
        }
    if bool(cache_cfg.get("require_freeze_base", True)) and not freeze_base_model:
        return None, {
            "structural_cache_enabled": True,
            "structural_cache_used": False,
            "structural_cache_reason": "base_not_frozen",
        }

    input_aligner_was_training = input_aligner.training
    model_was_training = model.training
    prompt_was_training = prompt_module.training
    input_aligner.eval()
    model.eval()
    prompt_module.eval()
    z = input_aligner(graph.x)
    h_pre = model.encode_frozen(z, graph.edge_index)
    cache = prompt_module.build_structural_cache(z=z, h_pre=h_pre, edge_index=graph.edge_index)
    if input_aligner_was_training:
        input_aligner.train()
    if model_was_training:
        model.train()
    if prompt_was_training:
        prompt_module.train()

    cache_numel = int(sum(value.numel() for value in cache.values() if isinstance(value, torch.Tensor)))
    cache_bytes = int(sum(value.numel() * value.element_size() for value in cache.values() if isinstance(value, torch.Tensor)))
    return cache, {
        "structural_cache_enabled": True,
        "structural_cache_used": True,
        "structural_cache_reason": "built",
        "structural_cache_numel": cache_numel,
        "structural_cache_bytes": cache_bytes,
        "structural_cache_keys": sorted(cache.keys()),
    }


def run_single_budget(
    config: dict[str, Any],
    *,
    repo_root: Path,
    route_budget: float,
    run_dir: Path,
    run_label: str,
) -> dict[str, Any]:
    experiment_cfg = config.get("experiment", {})
    data_cfg = config.get("data", {})
    pretrained_cfg = config.get("pretrained", {})
    model_cfg = config.get("model", {})
    prompt_cfg = config.get("prompt", {})
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
    model = FaithfulGP2F(
        backbone,
        hidden_dim=hidden_dim,
        num_classes=loaded.num_classes,
        adapter_bottleneck_dim=int(model_cfg.get("adapter_bottleneck_dim", 16)),
        adapter_beta_init=float(model_cfg.get("adapter_beta_init", 0.01)),
        adapter_alpha_init=float(model_cfg.get("adapter_alpha_init", 0.1)),
        adapter_style=str(model_cfg.get("adapter_style", "stable_zero_init")),
        alpha_init=float(model_cfg.get("alpha_init", 0.5)),
        fusion_alpha_style=str(model_cfg.get("fusion_alpha_style", "sigmoid")),
    ).to(device)
    prompt_module = _build_prompt_module(
        variant=variant,
        source_dim=source_dim,
        hidden_dim=hidden_dim,
        num_classes=loaded.num_classes,
        prompt_cfg=prompt_cfg,
        device=device,
    )

    base_checkpoint_path = _resolve_base_checkpoint(training_cfg, repo_root=repo_root, seed=seed)
    if base_checkpoint_path is not None:
        _load_base_checkpoint(
            base_checkpoint_path,
            model=model,
            input_aligner=input_aligner,
            prompt_module=prompt_module,
            device=device,
            load_prompt=bool(training_cfg.get("load_prompt_from_base_checkpoint", False)),
        )
    freeze_base_model = bool(training_cfg.get("freeze_base_model", False))
    if freeze_base_model:
        _set_module_trainable(input_aligner, False)
        _set_module_trainable(model, False)
    if prompt_module is not None:
        _set_module_trainable(prompt_module, bool(training_cfg.get("train_prompt_module", True)))

    structural_cache, structural_cache_info = _maybe_build_structural_cache(
        prompt_module=prompt_module,
        prompt_cfg=prompt_cfg,
        input_aligner=input_aligner,
        model=model,
        graph=graph,
        freeze_base_model=freeze_base_model,
    )

    param_counts = _parameter_counts(
        variant=variant,
        prompt_module=prompt_module,
        source_dim=source_dim,
        hidden_dim=hidden_dim,
        prompt_cfg=prompt_cfg,
    )
    trainable_summary = _trainable_parameter_summary(
        input_aligner=input_aligner,
        model=model,
        prompt_module=prompt_module,
    )

    optimizer_groups, trainable_params, optimizer_summary = _optimizer_groups(
        input_aligner=input_aligner,
        model=model,
        prompt_module=prompt_module,
        training_cfg=training_cfg,
    )
    if not trainable_params:
        raise RuntimeError("No trainable parameters are enabled for this run")
    optimizer = torch.optim.Adam(optimizer_groups)

    run_dir.mkdir(parents=True, exist_ok=True)
    budget_config = _deep_update(config, {"prompt": {"route_budget": float(route_budget)}})
    write_yaml(run_dir / "config.yaml", budget_config)

    init_eq = _init_equivalence(
        model=model,
        prompt_module=prompt_module,
        input_aligner=input_aligner,
        x=graph.x,
        edge_index=graph.edge_index,
        train_mask=split.train_mask,
        labels=graph.y,
        structural_cache=structural_cache,
    )

    epochs = int(training_cfg.get("epochs", 200))
    log_every = int(training_cfg.get("log_every", 20))
    eval_every = max(1, int(training_cfg.get("eval_every", 1)))
    monitor = str(training_cfg.get("early_stop_metric", "val_acc"))
    patience = int(training_cfg.get("early_stop_patience", 0))
    min_epochs = int(training_cfg.get("early_stop_min_epochs", 0))
    keep_checkpoint = bool(training_cfg.get("keep_best_checkpoint", True))
    lambda_budget = float(prompt_cfg.get("lambda_budget", 0.0))
    lambda_message_norm = float(prompt_cfg.get("lambda_message_norm", 0.0))
    message_norm_target = float(prompt_cfg.get("message_norm_target", 0.0))
    budget_warmup_epochs = int(prompt_cfg.get("budget_warmup_epochs", 0))
    budget_warmup_delay = int(prompt_cfg.get("budget_warmup_delay_epochs", 0))
    message_norm_warmup_epochs = int(prompt_cfg.get("message_norm_warmup_epochs", 0))
    message_norm_warmup_delay = int(prompt_cfg.get("message_norm_warmup_delay_epochs", 0))
    best_checkpoint_path = run_dir / "best_model.pt"
    best_val = -1.0
    best_metrics: dict[str, float] = {}
    loss_curve: list[dict[str, float]] = []
    prompt_curve: list[dict[str, float]] = []
    epochs_no_improve = 0
    stopped_epoch = epochs
    early_stopped = False

    progress = tqdm(range(1, epochs + 1), desc=run_label, unit="epoch", dynamic_ncols=True)
    for epoch in progress:
        model.train()
        input_aligner.train()
        if prompt_module is not None:
            prompt_module.train()
        optimizer.zero_grad()

        z = input_aligner(graph.x)
        model_out, prompt_out = _forward_prompt(
            model=model,
            prompt_module=prompt_module,
            z=z,
            edge_index=graph.edge_index,
            train_mask=split.train_mask,
            labels=graph.y,
            split="train",
            structural_cache=structural_cache,
        )
        cls_loss = F.cross_entropy(model_out["logits"][split.train_mask], graph.y[split.train_mask])
        budget = prompt_budget_loss(prompt_out["gate"], route_budget) if prompt_module is not None else z.new_tensor(0.0)
        message_norm = (
            prompt_message_norm_loss(prompt_out, z, target_ratio=message_norm_target)
            if prompt_module is not None
            else z.new_tensor(0.0)
        )
        lambda_budget_eff = lambda_budget * _warmup_factor(
            epoch,
            budget_warmup_epochs,
            delay_epochs=budget_warmup_delay,
        )
        lambda_message_norm_eff = lambda_message_norm * _warmup_factor(
            epoch,
            message_norm_warmup_epochs,
            delay_epochs=message_norm_warmup_delay,
        )
        loss = cls_loss + lambda_budget_eff * budget + lambda_message_norm_eff * message_norm
        if not torch.isfinite(loss):
            raise RuntimeError(f"Non-finite loss at epoch {epoch}: {loss.item()}")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable_params, float(training_cfg.get("grad_clip", 1.0)))
        optimizer.step()

        prompt_log = _prompt_diagnostics(prompt_out, z)
        log_item = {
            "epoch": float(epoch),
            "total": float(loss.detach().item()),
            "cls": float(cls_loss.detach().item()),
            "budget": float(budget.detach().item()),
            "message_norm": float(message_norm.detach().item()),
            "lambda_budget_effective": float(lambda_budget_eff),
            "lambda_message_norm_effective": float(lambda_message_norm_eff),
            "route_budget": float(route_budget),
            "alpha": float(model_out["alpha"].detach().item()),
            **prompt_log,
        }
        loss_curve.append(log_item)
        prompt_curve.append({"epoch": float(epoch), **prompt_log})

        should_eval = epoch == 1 or epoch % eval_every == 0 or epoch == epochs
        metrics: dict[str, float] | None = None
        if should_eval:
            metrics = evaluate_prompt(
                model=model,
                prompt_module=prompt_module,
                input_aligner=input_aligner,
                x=graph.x,
                edge_index=graph.edge_index,
                labels=graph.y,
                train_mask=split.train_mask,
                val_mask=split.val_mask,
                test_mask=split.test_mask,
                num_classes=loaded.num_classes,
                structural_cache=structural_cache,
            )
            monitor_lookup = {
                **metrics,
                "train_loss": float(loss.detach().item()),
                "total": float(loss.detach().item()),
                "cls": float(cls_loss.detach().item()),
                "budget": float(budget.detach().item()),
                "message_norm": float(message_norm.detach().item()),
            }
            monitor_value = float(monitor_lookup.get(monitor, metrics["val_acc"]))
            if _monitor_improved(monitor, monitor_value, best_val, bool(best_metrics)):
                best_val = monitor_value
                best_metrics = {
                    "best_epoch": float(epoch),
                    "monitor_value": monitor_value,
                    "route_budget": float(route_budget),
                    **metrics,
                    "total": float(loss.detach().item()),
                    "cls": float(cls_loss.detach().item()),
                    "budget": float(budget.detach().item()),
                    "message_norm": float(message_norm.detach().item()),
                    "lambda_budget_effective": float(lambda_budget_eff),
                    "lambda_message_norm_effective": float(lambda_message_norm_eff),
                    **_adapter_stats(model),
                }
                _save_checkpoint(
                    best_checkpoint_path,
                    model=model,
                    input_aligner=input_aligner,
                    prompt_module=prompt_module,
                    epoch=epoch,
                    metrics=best_metrics,
                )
                epochs_no_improve = 0
            else:
                epochs_no_improve += eval_every

        if epoch == 1 or epoch % log_every == 0 or epoch == epochs:
            if metrics is None:
                metrics = evaluate_prompt(
                    model=model,
                    prompt_module=prompt_module,
                    input_aligner=input_aligner,
                    x=graph.x,
                    edge_index=graph.edge_index,
                    labels=graph.y,
                    train_mask=split.train_mask,
                    val_mask=split.val_mask,
                    test_mask=split.test_mask,
                    num_classes=loaded.num_classes,
                    structural_cache=structural_cache,
                )
            progress.set_postfix(
                {
                    "loss": f"{loss.item():.4f}",
                    "val": f"{metrics['val_acc']:.4f}",
                    "test": f"{metrics['test_acc']:.4f}",
                    "null": f"{metrics['route_null']:.2f}",
                    "best": f"{best_val:.4f}",
                }
            )
        if patience > 0 and epoch >= min_epochs and epochs_no_improve >= patience:
            stopped_epoch = epoch
            early_stopped = True
            if metrics is None:
                metrics = evaluate_prompt(
                    model=model,
                    prompt_module=prompt_module,
                    input_aligner=input_aligner,
                    x=graph.x,
                    edge_index=graph.edge_index,
                    labels=graph.y,
                    train_mask=split.train_mask,
                    val_mask=split.val_mask,
                    test_mask=split.test_mask,
                    num_classes=loaded.num_classes,
                    structural_cache=structural_cache,
                )
            progress.set_postfix(
                {
                    "loss": f"{loss.item():.4f}",
                    "val": f"{metrics['val_acc']:.4f}",
                    "test": f"{metrics['test_acc']:.4f}",
                    "null": f"{metrics['route_null']:.2f}",
                    "stop": "early",
                }
            )
            break

    final_metrics = evaluate_prompt(
        model=model,
        prompt_module=prompt_module,
        input_aligner=input_aligner,
        x=graph.x,
        edge_index=graph.edge_index,
        labels=graph.y,
        train_mask=split.train_mask,
        val_mask=split.val_mask,
        test_mask=split.test_mask,
        num_classes=loaded.num_classes,
        structural_cache=structural_cache,
    )
    result = {
        "dataset": loaded.name,
        "seed": seed,
        "split_seed": split.seed,
        "prompt_variant": variant,
        "route_budget": float(route_budget),
        "selected_route_budget": float(route_budget),
        "selection_metric": str(prompt_cfg.get("selection_metric", "val_acc")),
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
        "final": {**final_metrics, **param_counts, **init_eq},
        "best": {**best_metrics, **param_counts, **init_eq},
        "trainable_parameters": trainable_summary,
        "optimizer": optimizer_summary,
        "structural_cache": structural_cache_info,
        "regularization_schedule": {
            "lambda_budget": lambda_budget,
            "lambda_message_norm": lambda_message_norm,
            "budget_warmup_epochs": budget_warmup_epochs,
            "budget_warmup_delay_epochs": budget_warmup_delay,
            "message_norm_warmup_epochs": message_norm_warmup_epochs,
            "message_norm_warmup_delay_epochs": message_norm_warmup_delay,
            "message_norm_target": message_norm_target,
        },
        "base_checkpoint_path": str(base_checkpoint_path) if base_checkpoint_path is not None else "",
        "freeze_base_model": freeze_base_model,
        "train_prompt_module": bool(training_cfg.get("train_prompt_module", True)),
        "early_stopped": early_stopped,
        "stopped_epoch": stopped_epoch,
        "early_stop_metric": monitor,
        "best_checkpoint_path": str(best_checkpoint_path) if best_checkpoint_path.exists() and keep_checkpoint else "",
        "split_counts": _split_counts(graph.y, {"train": split.train_mask, "val": split.val_mask, "test": split.test_mask}, loaded.num_classes),
        "environment": _environment_info(),
        "run_dir": str(run_dir),
    }
    write_json(run_dir / "metrics.json", result)
    write_json(run_dir / "loss_curve.json", {"loss_curve": loss_curve})
    write_json(run_dir / "prompt_curve.json", {"prompt_curve": prompt_curve})
    if best_checkpoint_path.exists() and not keep_checkpoint:
        best_checkpoint_path.unlink()
    print(f"Saved results to {run_dir}")
    return result


def _budget_values(config: dict[str, Any], variant: str) -> list[float]:
    prompt_cfg = config.get("prompt", {})
    if variant == "noprompt":
        return [0.0]
    if str(prompt_cfg.get("route_budget_selection", "fixed")) == "validation_only":
        values = prompt_cfg.get("route_budget_grid", [prompt_cfg.get("route_budget", 0.1)])
        return [float(value) for value in values]
    return [float(prompt_cfg.get("route_budget", 0.1))]


def _select_budget_result(results: list[dict[str, Any]], selection_metric: str) -> dict[str, Any]:
    if not results:
        raise ValueError("No budget results to select from")
    reverse = selection_metric not in {"train_loss", "total", "cls", "budget"}
    return sorted(results, key=lambda item: float(item["best"].get(selection_metric, item["best"].get("val_acc", 0.0))), reverse=reverse)[0]


def run(config: dict[str, Any], *, repo_root: Path) -> dict[str, Any]:
    variant = _prompt_variant(config)
    config = _config_for_variant(config, variant)
    seeds = _resolve_run_seeds(config)
    output_root = _resolve_path(config.get("training", {}).get("output_dir", "outputs/gp2f_prompt_p0"), base_dir=repo_root)
    target_dataset = str(config.get("experiment", {}).get("target_dataset", "Cora"))
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    summary_dir = output_root / target_dataset / timestamp
    summary_dir.mkdir(parents=True, exist_ok=True)

    selected_results: list[dict[str, Any]] = []
    all_budget_results: list[list[dict[str, Any]]] = []
    selection_metric = str(config.get("prompt", {}).get("selection_metric", "val_acc"))
    for seed_index, seed in enumerate(seeds):
        seed_config = _deep_update(config, {"experiment": {"seed": seed}})
        budgets = _budget_values(seed_config, variant)
        seed_budget_results: list[dict[str, Any]] = []
        for budget_index, budget in enumerate(budgets):
            budget_config = _deep_update(seed_config, {"prompt": {"route_budget": float(budget)}})
            budget_label = _route_budget_label(float(budget))
            budget_run_dir = summary_dir / f"seed_{seed}" / f"budget_{budget_label}"
            run_label = f"{target_dataset} seed={seed} budget={budget:g}"
            if len(seeds) > 1 or len(budgets) > 1:
                run_label = f"{run_label} ({seed_index + 1}/{len(seeds)}, {budget_index + 1}/{len(budgets)})"
            seed_budget_results.append(
                run_single_budget(
                    budget_config,
                    repo_root=repo_root,
                    route_budget=float(budget),
                    run_dir=budget_run_dir,
                    run_label=run_label,
                )
            )
        selected = _select_budget_result(seed_budget_results, selection_metric)
        selected["selected_route_budget"] = float(selected["route_budget"])
        selected["selection_metric"] = selection_metric
        selected["selected_epoch"] = float(selected["best"].get("best_epoch", 0.0))
        selected_results.append(selected)
        all_budget_results.append(seed_budget_results)

    best_test = [float(result["best"].get("test_acc", 0.0)) for result in selected_results]
    best_macro_f1 = [float(result["best"].get("test_macro_f1", 0.0)) for result in selected_results]
    final_test = [float(result["final"].get("test_acc", 0.0)) for result in selected_results]
    final_macro_f1 = [float(result["final"].get("test_macro_f1", 0.0)) for result in selected_results]
    summary = {
        "dataset": target_dataset,
        "prompt_variant": variant,
        "selection_metric": selection_metric,
        "seeds": seeds,
        "num_runs": len(seeds),
        "best_test_acc_mean_std": _format_mean_std(best_test),
        "best_test_macro_f1_mean_std": _format_mean_std(best_macro_f1),
        "final_test_acc_mean_std": _format_mean_std(final_test),
        "final_test_macro_f1_mean_std": _format_mean_std(final_macro_f1),
        "runs": selected_results,
        "budget_runs": all_budget_results,
        "environment": _environment_info(),
    }
    write_json(summary_dir / "summary.json", summary)
    with (summary_dir / "summary.csv").open("w", encoding="utf-8", newline="") as handle:
        fieldnames = [
            "seed",
            "selected_route_budget",
            "best_epoch",
            "best_test_acc",
            "best_test_macro_f1",
            "final_test_acc",
            "final_test_macro_f1",
            "route_semantic",
            "route_structural",
            "route_null",
            "prompt_message_norm",
            "structural_neighbor_var_norm",
            "structural_degree_mean",
            "structural_similarity_mean",
            "structural_cache_hit",
            "gamma",
            "init_adapted_x_delta",
            "init_logit_delta",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for result in selected_results:
            writer.writerow(
                {
                    "seed": result["seed"],
                    "selected_route_budget": result.get("selected_route_budget", 0.0),
                    "best_epoch": result["best"].get("best_epoch", 0.0),
                    "best_test_acc": result["best"].get("test_acc", 0.0),
                    "best_test_macro_f1": result["best"].get("test_macro_f1", 0.0),
                    "final_test_acc": result["final"].get("test_acc", 0.0),
                    "final_test_macro_f1": result["final"].get("test_macro_f1", 0.0),
                    "route_semantic": result["best"].get("route_semantic", 0.0),
                    "route_structural": result["best"].get("route_structural", 0.0),
                    "route_null": result["best"].get("route_null", 0.0),
                    "prompt_message_norm": result["best"].get("prompt_message_norm", 0.0),
                    "structural_neighbor_var_norm": result["best"].get("structural_neighbor_var_norm", 0.0),
                    "structural_degree_mean": result["best"].get("structural_degree_mean", 0.0),
                    "structural_similarity_mean": result["best"].get("structural_similarity_mean", 0.0),
                    "structural_cache_hit": result["best"].get("structural_cache_hit", 0.0),
                    "gamma": result["best"].get("gamma", 0.0),
                    "init_adapted_x_delta": result["best"].get("init_adapted_x_delta", 0.0),
                    "init_logit_delta": result["best"].get("init_logit_delta", 0.0),
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
    parser = argparse.ArgumentParser(description="GP2F residual prompt P0 runner")
    parser.add_argument("--config", type=str, default="configs/gp2f_prompt_p0.yaml")
    parser.add_argument("--target_dataset", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--seeds", type=str, default=None)
    parser.add_argument("--runs", type=int, default=None)
    parser.add_argument("--route_budget", type=float, default=None)
    parser.add_argument("--route_budget_grid", type=str, default=None)
    parser.add_argument("--prompt_variant", type=str, choices=sorted(PROMPT_VARIANTS), default=None)
    parser.add_argument("--gamma_init", type=float, default=None)
    parser.add_argument("--gamma_max", type=float, default=None)
    parser.add_argument("--lambda_budget", type=float, default=None)
    parser.add_argument("--lambda_message_norm", type=float, default=None)
    parser.add_argument("--message_norm_target", type=float, default=None)
    parser.add_argument("--base_checkpoint_path", type=str, default=None)
    parser.add_argument("--base_checkpoint_root", type=str, default=None)
    parser.add_argument("--freeze_base", action="store_true")
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--prompt_lr", type=float, default=None)
    parser.add_argument("--prompt_lr_multiplier", type=float, default=None)
    parser.add_argument("--weight_decay", type=float, default=None)
    parser.add_argument("--prompt_weight_decay", type=float, default=None)
    parser.add_argument("--budget_warmup_epochs", type=int, default=None)
    parser.add_argument("--message_norm_warmup_epochs", type=int, default=None)
    parser.add_argument("--null_bias_init", type=float, default=None)
    parser.add_argument("--structural_base", type=str, choices=["z_detached", "z", "h_pre_detached", "h_pre"], default=None)
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
    if args.prompt_lr_multiplier is not None:
        overrides.setdefault("training", {})["prompt_lr_multiplier"] = args.prompt_lr_multiplier
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
    if args.route_budget is not None:
        overrides.setdefault("prompt", {})["route_budget"] = float(args.route_budget)
        overrides.setdefault("prompt", {})["route_budget_selection"] = "fixed"
    if args.route_budget_grid is not None:
        overrides.setdefault("prompt", {})["route_budget_grid"] = [
            float(piece.strip()) for piece in args.route_budget_grid.split(",") if piece.strip()
        ]
        overrides.setdefault("prompt", {})["route_budget_selection"] = "validation_only"
    if args.gamma_init is not None:
        overrides.setdefault("prompt", {})["gamma_init"] = float(args.gamma_init)
    if args.gamma_max is not None:
        overrides.setdefault("prompt", {})["gamma_max"] = float(args.gamma_max)
    if args.lambda_budget is not None:
        overrides.setdefault("prompt", {})["lambda_budget"] = float(args.lambda_budget)
    if args.lambda_message_norm is not None:
        overrides.setdefault("prompt", {})["lambda_message_norm"] = float(args.lambda_message_norm)
    if args.message_norm_target is not None:
        overrides.setdefault("prompt", {})["message_norm_target"] = float(args.message_norm_target)
    if args.budget_warmup_epochs is not None:
        overrides.setdefault("prompt", {})["budget_warmup_epochs"] = int(args.budget_warmup_epochs)
    if args.message_norm_warmup_epochs is not None:
        overrides.setdefault("prompt", {})["message_norm_warmup_epochs"] = int(args.message_norm_warmup_epochs)
    if args.null_bias_init is not None:
        overrides.setdefault("prompt", {}).setdefault("gate", {})["null_bias_init"] = float(args.null_bias_init)
    if args.structural_base is not None:
        overrides.setdefault("prompt", {}).setdefault("structural", {})["base"] = args.structural_base
    if args.base_checkpoint_path is not None:
        overrides.setdefault("training", {})["base_checkpoint_path"] = args.base_checkpoint_path
    if args.base_checkpoint_root is not None:
        overrides.setdefault("training", {})["base_checkpoint_root"] = args.base_checkpoint_root
    if args.freeze_base:
        overrides.setdefault("training", {})["freeze_base_model"] = True
    run(_deep_update(config, overrides), repo_root=repo_root)


if __name__ == "__main__":
    main()
