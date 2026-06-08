"""Post-hoc diagnostics for P1/P2 prompt graph experiments.

This script does not train or select hyperparameters. It loads saved best
checkpoints and measures whether the prompt graph pool/message actually helps
the nodes it touches.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from data import build_few_shot_split, load_node_dataset
from experiments.run_gp2f_baseline import InputAligner, _resolve_path, set_seed
from experiments.run_gp2f_prompt_graph import (
    _build_prompt_graph_module,
    _config_for_variant,
    _forward_prompt_graph,
    _prompt_variant,
)
from models import FaithfulGP2F, PromptAwareGP2F, load_pretrained_gcn
from utils.io import read_yaml, write_json


def _load_json(path: Path) -> dict[str, Any]:
    import json

    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return data


def _mean_bool(mask: torch.Tensor) -> float:
    if mask.numel() == 0:
        return 0.0
    return float(mask.float().mean().item())


def _masked_mean(values: torch.Tensor, mask: torch.Tensor) -> float:
    mask = mask.bool()
    if not bool(mask.any()):
        return 0.0
    return float(values[mask].float().mean().item())


def _masked_sum(mask: torch.Tensor) -> int:
    return int(mask.bool().sum().item())


def _confidence_and_margin(logits: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    probs = torch.softmax(logits, dim=-1)
    sorted_probs = probs.sort(dim=-1, descending=True).values
    confidence = sorted_probs[:, 0]
    if sorted_probs.size(1) == 1:
        margin = confidence
    else:
        margin = sorted_probs[:, 0] - sorted_probs[:, 1]
    return confidence, margin


def _bottom_fraction_mask(values: torch.Tensor, fraction: float, eligible_mask: torch.Tensor) -> torch.Tensor:
    eligible_idx = torch.where(eligible_mask.bool())[0]
    out = torch.zeros_like(eligible_mask, dtype=torch.bool)
    if eligible_idx.numel() == 0:
        return out
    k = max(1, int(round(float(fraction) * int(eligible_idx.numel()))))
    k = min(k, int(eligible_idx.numel()))
    selected = eligible_idx[torch.topk(values[eligible_idx], k=k, largest=False).indices]
    out[selected] = True
    return out


def _top_fraction_mask(values: torch.Tensor, fraction: float, eligible_mask: torch.Tensor) -> torch.Tensor:
    eligible_idx = torch.where(eligible_mask.bool())[0]
    out = torch.zeros_like(eligible_mask, dtype=torch.bool)
    if eligible_idx.numel() == 0:
        return out
    k = max(1, int(round(float(fraction) * int(eligible_idx.numel()))))
    k = min(k, int(eligible_idx.numel()))
    selected = eligible_idx[torch.topk(values[eligible_idx], k=k, largest=True).indices]
    out[selected] = True
    return out


def _overlap_ratio(left: torch.Tensor, right: torch.Tensor) -> float:
    denom = int(left.bool().sum().item())
    if denom == 0:
        return 0.0
    return float((left.bool() & right.bool()).float().sum().item() / float(denom))


def _accuracy(pred: torch.Tensor, labels: torch.Tensor, mask: torch.Tensor) -> float:
    mask = mask.bool()
    if not bool(mask.any()):
        return 0.0
    return float((pred[mask] == labels[mask]).float().mean().item())


def _split_name(idx: int, train_mask: torch.Tensor, val_mask: torch.Tensor, test_mask: torch.Tensor) -> str:
    if bool(train_mask[idx].item()):
        return "train"
    if bool(val_mask[idx].item()):
        return "val"
    if bool(test_mask[idx].item()):
        return "test"
    return "unused"


def _instantiate_from_config(
    config: dict[str, Any],
    *,
    repo_root: Path,
    device: torch.device,
) -> tuple[Any, Any, Any, InputAligner, FaithfulGP2F, Any, Any]:
    config = _config_for_variant(config, _prompt_variant(config))
    experiment_cfg = config.get("experiment", {})
    data_cfg = config.get("data", {})
    pretrained_cfg = config.get("pretrained", {})
    model_cfg = config.get("model", {})
    prompt_graph_cfg = config.get("prompt_graph", {})
    prompt_aware_cfg = config.get("prompt_aware", {})

    seed = int(experiment_cfg.get("seed", 0))
    set_seed(seed)
    target_dataset = str(experiment_cfg.get("target_dataset", "Cora"))
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
        model = PromptAwareGP2F(backbone, prompt_aware_config=prompt_aware_cfg, **model_kwargs).to(device)
    else:
        model = FaithfulGP2F(backbone, **model_kwargs).to(device)
    prompt_graph_module = _build_prompt_graph_module(
        variant=_prompt_variant(config),
        source_dim=source_dim,
        hidden_dim=hidden_dim,
        prompt_graph_cfg=prompt_graph_cfg,
        device=device,
    )
    return loaded, graph, split, input_aligner, model, prompt_graph_module, config


def _load_checkpoint(
    checkpoint_path: Path,
    *,
    model: FaithfulGP2F,
    input_aligner: InputAligner,
    prompt_graph_module: torch.nn.Module | None,
    device: torch.device,
) -> None:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model"])
    input_aligner.load_state_dict(checkpoint["input_aligner"])
    if prompt_graph_module is not None and checkpoint.get("prompt_graph_module") is not None:
        prompt_graph_module.load_state_dict(checkpoint["prompt_graph_module"])


@torch.no_grad()
def _forward_with_scale(
    *,
    model: FaithfulGP2F,
    prompt_graph_module: Any,
    input_aligner: InputAligner,
    x: torch.Tensor,
    edge_index: torch.Tensor,
    train_mask: torch.Tensor,
    edge_scale_multiplier: float,
    message_scale: float | None,
) -> tuple[dict[str, Any], dict[str, Any], torch.Tensor]:
    old_scale = None
    if isinstance(model, PromptAwareGP2F) and message_scale is not None:
        old_scale = float(model.prompt_message_scale)
        model.prompt_message_scale = float(message_scale)
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
    if old_scale is not None:
        model.prompt_message_scale = old_scale
    return model_out, prompt_out, z


def _pool_assignment_top1(prompt_out: dict[str, Any], num_nodes: int) -> torch.Tensor:
    out = torch.full((num_nodes,), -1, dtype=torch.long, device=prompt_out["pool_mask"].device)
    aux = prompt_out.get("aux", {})
    top_prompt_ids = aux.get("top_prompt_ids")
    assignment_prob = aux.get("assignment_prob")
    if not (isinstance(top_prompt_ids, torch.Tensor) and isinstance(assignment_prob, torch.Tensor)):
        return out
    pool_idx = torch.where(prompt_out["pool_mask"])[0]
    if pool_idx.numel() == 0:
        return out
    top1 = top_prompt_ids[
        torch.arange(top_prompt_ids.size(0), device=top_prompt_ids.device),
        assignment_prob.argmax(dim=1),
    ]
    out[pool_idx] = top1
    return out


def diagnose_run(
    result: dict[str, Any],
    *,
    repo_root: Path,
    output_dir: Path,
    device: torch.device,
) -> dict[str, Any]:
    run_dir = Path(str(result["run_dir"]))
    config_path = run_dir / "config.yaml"
    checkpoint_path = Path(str(result.get("best_checkpoint_path") or run_dir / "best_model.pt"))
    config = read_yaml(config_path)
    loaded, graph, split, input_aligner, model, prompt_graph_module, config = _instantiate_from_config(
        config,
        repo_root=repo_root,
        device=device,
    )
    _load_checkpoint(
        checkpoint_path,
        model=model,
        input_aligner=input_aligner,
        prompt_graph_module=prompt_graph_module,
        device=device,
    )
    edge_scale_multiplier = float(result.get("best", {}).get("edge_scale_multiplier", 1.0))
    selected_scale = float(result.get("selected_message_scale", result.get("best", {}).get("prompt_message_scale", 0.0)))
    on_out, prompt_out, _ = _forward_with_scale(
        model=model,
        prompt_graph_module=prompt_graph_module,
        input_aligner=input_aligner,
        x=graph.x,
        edge_index=graph.edge_index,
        train_mask=split.train_mask,
        edge_scale_multiplier=edge_scale_multiplier,
        message_scale=selected_scale,
    )
    off_out, _, _ = _forward_with_scale(
        model=model,
        prompt_graph_module=prompt_graph_module,
        input_aligner=input_aligner,
        x=graph.x,
        edge_index=graph.edge_index,
        train_mask=split.train_mask,
        edge_scale_multiplier=edge_scale_multiplier,
        message_scale=0.0,
    )

    labels = graph.y
    logits_on = on_out["logits"]
    logits_off = off_out["logits"]
    pred_on = logits_on.argmax(dim=-1)
    pred_off = logits_off.argmax(dim=-1)
    correct_on = pred_on == labels
    correct_off = pred_off == labels
    conf_off, margin_off = _confidence_and_margin(logits_off)
    conf_on, margin_on = _confidence_and_margin(logits_on)
    pool_mask = prompt_out["pool_mask"].detach().bool()
    test_mask = split.test_mask.bool()
    val_mask = split.val_mask.bool()
    train_mask = split.train_mask.bool()
    non_pool_mask = ~pool_mask
    test_pool = test_mask & pool_mask
    test_non_pool = test_mask & non_pool_mask
    test_wrong_off = test_mask & (~correct_off)
    test_fixed = test_mask & (~correct_off) & correct_on
    test_broken = test_mask & correct_off & (~correct_on)
    h_delta = (on_out["h_adp"] - off_out["h_adp"]).norm(dim=-1)
    logit_delta = (logits_on - logits_off).norm(dim=-1)
    aux = prompt_out.get("aux", {})
    structural_score = aux.get("structural_score")
    if not isinstance(structural_score, torch.Tensor):
        structural_score = torch.zeros_like(labels, dtype=logits_on.dtype)
    prompt_top1 = _pool_assignment_top1(prompt_out, int(labels.numel()))

    pool_fraction = float(pool_mask.float().mean().item())
    low_conf_mask = _bottom_fraction_mask(conf_off, pool_fraction, test_mask)
    low_margin_mask = _bottom_fraction_mask(margin_off, pool_fraction, test_mask)
    high_structural_mask = _top_fraction_mask(structural_score, pool_fraction, test_mask)

    summary = {
        "dataset": loaded.name,
        "seed": int(result["seed"]),
        "prompt_variant": str(result.get("prompt_variant", "")),
        "run_dir": str(run_dir),
        "checkpoint_path": str(checkpoint_path),
        "selected_message_scale": selected_scale,
        "edge_scale_multiplier": edge_scale_multiplier,
        "num_nodes": int(labels.numel()),
        "test_nodes": _masked_sum(test_mask),
        "pool_nodes": _masked_sum(pool_mask),
        "test_pool_nodes": _masked_sum(test_pool),
        "pool_ratio": float(pool_mask.float().mean().item()),
        "test_pool_ratio": _mean_bool(pool_mask[test_mask]),
        "test_acc_prompt_off": _accuracy(pred_off, labels, test_mask),
        "test_acc_prompt_on": _accuracy(pred_on, labels, test_mask),
        "test_pool_acc_prompt_off": _accuracy(pred_off, labels, test_pool),
        "test_pool_acc_prompt_on": _accuracy(pred_on, labels, test_pool),
        "test_non_pool_acc_prompt_off": _accuracy(pred_off, labels, test_non_pool),
        "test_non_pool_acc_prompt_on": _accuracy(pred_on, labels, test_non_pool),
        "test_fixed_by_prompt": _masked_sum(test_fixed),
        "test_broken_by_prompt": _masked_sum(test_broken),
        "test_pool_fixed_by_prompt": _masked_sum(test_fixed & pool_mask),
        "test_pool_broken_by_prompt": _masked_sum(test_broken & pool_mask),
        "test_non_pool_fixed_by_prompt": _masked_sum(test_fixed & non_pool_mask),
        "test_non_pool_broken_by_prompt": _masked_sum(test_broken & non_pool_mask),
        "test_prediction_change_ratio": _mean_bool((pred_on != pred_off)[test_mask]),
        "test_pool_prediction_change_ratio": _mean_bool((pred_on != pred_off)[test_pool]),
        "test_non_pool_prediction_change_ratio": _mean_bool((pred_on != pred_off)[test_non_pool]),
        "pool_error_precision_test": _masked_mean((~correct_off).float(), test_pool),
        "pool_error_recall_test": _overlap_ratio(test_wrong_off, test_pool),
        "pool_low_conf_overlap_test": _overlap_ratio(test_pool, low_conf_mask),
        "pool_low_margin_overlap_test": _overlap_ratio(test_pool, low_margin_mask),
        "pool_high_structural_overlap_test": _overlap_ratio(test_pool, high_structural_mask),
        "structural_score_pool_mean": _masked_mean(structural_score, pool_mask),
        "structural_score_non_pool_mean": _masked_mean(structural_score, non_pool_mask),
        "structural_score_test_wrong_mean": _masked_mean(structural_score, test_wrong_off),
        "structural_score_test_correct_mean": _masked_mean(structural_score, test_mask & correct_off),
        "confidence_off_pool_mean": _masked_mean(conf_off, pool_mask),
        "confidence_off_non_pool_mean": _masked_mean(conf_off, non_pool_mask),
        "margin_off_pool_mean": _masked_mean(margin_off, pool_mask),
        "margin_off_non_pool_mean": _masked_mean(margin_off, non_pool_mask),
        "h_delta_norm_pool_mean": _masked_mean(h_delta, pool_mask),
        "h_delta_norm_non_pool_mean": _masked_mean(h_delta, non_pool_mask),
        "logit_delta_norm_pool_mean": _masked_mean(logit_delta, pool_mask),
        "logit_delta_norm_non_pool_mean": _masked_mean(logit_delta, non_pool_mask),
        "prompt_usage_distribution": [
            float(value) for value in aux.get("prompt_usage", torch.zeros(0)).detach().cpu().tolist()
        ]
        if isinstance(aux.get("prompt_usage"), torch.Tensor)
        else [],
        "prompt_usage_full_distribution": [
            float(value) for value in aux.get("prompt_usage_full", torch.zeros(0)).detach().cpu().tolist()
        ]
        if isinstance(aux.get("prompt_usage_full"), torch.Tensor)
        else [],
        "prompt_usage_entropy": float(aux.get("prompt_usage_entropy", torch.tensor(0.0)).detach().item())
        if isinstance(aux.get("prompt_usage_entropy"), torch.Tensor)
        else 0.0,
        "prompt_usage_full_entropy": float(aux.get("prompt_usage_full_entropy", torch.tensor(0.0)).detach().item())
        if isinstance(aux.get("prompt_usage_full_entropy"), torch.Tensor)
        else 0.0,
    }

    run_output_dir = output_dir / loaded.name / f"seed_{int(result['seed'])}"
    run_output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    for idx in range(int(labels.numel())):
        rows.append(
            {
                "node_id": idx,
                "split": _split_name(idx, train_mask, val_mask, test_mask),
                "label": int(labels[idx].item()),
                "pool": int(pool_mask[idx].item()),
                "prompt_top1": int(prompt_top1[idx].item()),
                "structural_score": float(structural_score[idx].detach().item()),
                "pred_off": int(pred_off[idx].item()),
                "pred_on": int(pred_on[idx].item()),
                "correct_off": int(correct_off[idx].item()),
                "correct_on": int(correct_on[idx].item()),
                "fixed_by_prompt": int((not bool(correct_off[idx].item())) and bool(correct_on[idx].item())),
                "broken_by_prompt": int(bool(correct_off[idx].item()) and (not bool(correct_on[idx].item()))),
                "confidence_off": float(conf_off[idx].detach().item()),
                "confidence_on": float(conf_on[idx].detach().item()),
                "margin_off": float(margin_off[idx].detach().item()),
                "margin_on": float(margin_on[idx].detach().item()),
                "h_delta_norm": float(h_delta[idx].detach().item()),
                "logit_delta_norm": float(logit_delta[idx].detach().item()),
            }
        )
    with (run_output_dir / "node_diagnostics.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()) if rows else ["node_id"])
        writer.writeheader()
        writer.writerows(rows)
    write_json(run_output_dir / "diagnostics.json", summary)
    return summary


def diagnose_summary(summary_path: Path, *, repo_root: Path, output_dir: Path) -> dict[str, Any]:
    payload = _load_json(summary_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    runs = payload.get("runs", [])
    diagnostics = [
        diagnose_run(run, repo_root=repo_root, output_dir=output_dir, device=device)
        for run in runs
    ]
    if not diagnostics:
        raise RuntimeError(f"No runs found in {summary_path}")

    def mean(key: str) -> float:
        return float(sum(float(item.get(key, 0.0)) for item in diagnostics) / float(len(diagnostics)))

    aggregate = {
        "source_summary": str(summary_path),
        "dataset": diagnostics[0]["dataset"],
        "num_runs": len(diagnostics),
        "mean_test_acc_prompt_off": mean("test_acc_prompt_off"),
        "mean_test_acc_prompt_on": mean("test_acc_prompt_on"),
        "mean_test_pool_acc_prompt_off": mean("test_pool_acc_prompt_off"),
        "mean_test_pool_acc_prompt_on": mean("test_pool_acc_prompt_on"),
        "mean_test_non_pool_acc_prompt_off": mean("test_non_pool_acc_prompt_off"),
        "mean_test_non_pool_acc_prompt_on": mean("test_non_pool_acc_prompt_on"),
        "mean_pool_error_precision_test": mean("pool_error_precision_test"),
        "mean_pool_error_recall_test": mean("pool_error_recall_test"),
        "mean_pool_low_conf_overlap_test": mean("pool_low_conf_overlap_test"),
        "mean_pool_low_margin_overlap_test": mean("pool_low_margin_overlap_test"),
        "mean_test_prediction_change_ratio": mean("test_prediction_change_ratio"),
        "mean_test_pool_prediction_change_ratio": mean("test_pool_prediction_change_ratio"),
        "mean_test_non_pool_prediction_change_ratio": mean("test_non_pool_prediction_change_ratio"),
        "mean_h_delta_norm_pool": mean("h_delta_norm_pool_mean"),
        "mean_h_delta_norm_non_pool": mean("h_delta_norm_non_pool_mean"),
        "mean_prompt_usage_entropy": mean("prompt_usage_entropy"),
        "runs": diagnostics,
    }
    out_root = output_dir / diagnostics[0]["dataset"]
    out_root.mkdir(parents=True, exist_ok=True)
    write_json(out_root / "summary_diagnostics.json", aggregate)
    with (out_root / "summary_diagnostics.csv").open("w", encoding="utf-8", newline="") as handle:
        fieldnames = list(diagnostics[0].keys())
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(diagnostics)
    return aggregate


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Diagnose saved prompt graph runs")
    parser.add_argument("--summary_path", action="append", required=True)
    parser.add_argument("--output_dir", type=str, default="outputs/prompt_diagnostics")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    output_dir = _resolve_path(args.output_dir, base_dir=repo_root)
    summaries = []
    for item in args.summary_path:
        aggregate = diagnose_summary(Path(item), repo_root=repo_root, output_dir=output_dir)
        summaries.append(aggregate)
        print(
            f"{aggregate['dataset']}: off={aggregate['mean_test_acc_prompt_off']:.4f} "
            f"on={aggregate['mean_test_acc_prompt_on']:.4f} "
            f"pool_error_precision={aggregate['mean_pool_error_precision_test']:.4f} "
            f"pool_error_recall={aggregate['mean_pool_error_recall_test']:.4f}"
        )
    write_json(output_dir / "all_summary_diagnostics.json", {"summaries": summaries})
    print(f"Saved diagnostics to {output_dir}")


if __name__ == "__main__":
    main()
