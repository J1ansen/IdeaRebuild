"""Post-hoc confidence/oracle rejection diagnostics for P2 prompt graphs.

The script does not train or change saved checkpoints. It compares prompt-off
logits with prompt-on logits, then applies split-derived acceptance masks:

    mixed_logits_i = prompt_on_i if accept_i else prompt_off_i

Train-threshold strategies tune the acceptance threshold on train-pool labels
only and report validation/test metrics. Oracle metrics are diagnostic upper
bounds and must not be used for model selection.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any

import torch

from experiments.diagnose_prompt_graph import (
    _accuracy,
    _confidence_and_margin,
    _instantiate_from_config,
    _load_checkpoint,
    _load_json,
    _masked_sum,
    _forward_with_scale,
)
from experiments.run_gp2f_baseline import _resolve_path
from utils.io import read_yaml, write_json


def _candidate_thresholds(values: torch.Tensor, mask: torch.Tensor, *, max_points: int = 31) -> list[float]:
    idx = torch.where(mask.bool())[0]
    if idx.numel() == 0:
        return []
    selected = values[idx].detach().float().cpu()
    if selected.numel() <= max_points:
        thresholds = selected.unique(sorted=True)
    else:
        qs = torch.linspace(0.0, 1.0, steps=max_points)
        thresholds = torch.quantile(selected, qs).unique(sorted=True)
    return [float(item) for item in thresholds.tolist()]


def _mix_logits(logits_off: torch.Tensor, logits_on: torch.Tensor, accept_mask: torch.Tensor) -> torch.Tensor:
    accept = accept_mask.bool().view(-1, 1)
    return torch.where(accept, logits_on, logits_off)


def _eval_acceptance(
    *,
    logits_off: torch.Tensor,
    logits_on: torch.Tensor,
    labels: torch.Tensor,
    accept_mask: torch.Tensor,
    split_mask: torch.Tensor,
    pool_mask: torch.Tensor,
) -> dict[str, float | int]:
    mixed = _mix_logits(logits_off, logits_on, accept_mask)
    pred = mixed.argmax(dim=-1)
    pred_off = logits_off.argmax(dim=-1)
    pred_on = logits_on.argmax(dim=-1)
    split_mask = split_mask.bool()
    split_pool = split_mask & pool_mask.bool()
    accepted = accept_mask.bool() & split_mask
    accepted_pool = accept_mask.bool() & split_pool
    fixed = split_mask & (~(pred_off == labels)) & (pred == labels)
    broken = split_mask & (pred_off == labels) & (~(pred == labels))
    return {
        "acc": _accuracy(pred, labels, split_mask),
        "pool_acc": _accuracy(pred, labels, split_pool),
        "accepted_ratio": float(accepted.float().mean().item()) if bool(split_mask.any()) else 0.0,
        "pool_accepted_ratio": float(accepted_pool.float().mean().item()) if bool(split_pool.any()) else 0.0,
        "fixed": _masked_sum(fixed),
        "broken": _masked_sum(broken),
        "pool_fixed": _masked_sum(fixed & pool_mask.bool()),
        "pool_broken": _masked_sum(broken & pool_mask.bool()),
        "prediction_change_ratio": float((pred[split_mask] != pred_off[split_mask]).float().mean().item())
        if bool(split_mask.any())
        else 0.0,
    }


def _oracle_accept_mask(
    *,
    logits_off: torch.Tensor,
    logits_on: torch.Tensor,
    labels: torch.Tensor,
    eligible_mask: torch.Tensor,
) -> torch.Tensor:
    pred_off = logits_off.argmax(dim=-1)
    pred_on = logits_on.argmax(dim=-1)
    off_correct = pred_off == labels
    on_correct = pred_on == labels
    accept = torch.zeros_like(eligible_mask, dtype=torch.bool)
    accept[eligible_mask.bool()] = on_correct[eligible_mask.bool()] & (~off_correct[eligible_mask.bool()])
    return accept


def _best_threshold_strategy(
    *,
    name: str,
    values: torch.Tensor,
    direction: str,
    logits_off: torch.Tensor,
    logits_on: torch.Tensor,
    labels: torch.Tensor,
    train_mask: torch.Tensor,
    pool_mask: torch.Tensor,
) -> dict[str, Any] | None:
    train_pool = train_mask.bool() & pool_mask.bool()
    thresholds = _candidate_thresholds(values, train_pool)
    if not thresholds:
        return None
    best: dict[str, Any] | None = None
    for threshold in thresholds:
        if direction == "low":
            accept = pool_mask.bool() & (values <= threshold)
        elif direction == "high":
            accept = pool_mask.bool() & (values >= threshold)
        else:
            raise ValueError(f"Unsupported direction={direction!r}")
        train_eval = _eval_acceptance(
            logits_off=logits_off,
            logits_on=logits_on,
            labels=labels,
            accept_mask=accept,
            split_mask=train_mask,
            pool_mask=pool_mask,
        )
        candidate = {
            "strategy": name,
            "direction": direction,
            "threshold": float(threshold),
            "accept_mask": accept,
            "train_acc": float(train_eval["acc"]),
            "train_pool_acc": float(train_eval["pool_acc"]),
            "train_pool_accepted_ratio": float(train_eval["pool_accepted_ratio"]),
            "train_pool_fixed": int(train_eval["pool_fixed"]),
            "train_pool_broken": int(train_eval["pool_broken"]),
        }
        if best is None:
            best = candidate
            continue
        key = (
            candidate["train_acc"],
            candidate["train_pool_acc"],
            -abs(candidate["train_pool_accepted_ratio"] - 0.5),
        )
        best_key = (
            best["train_acc"],
            best["train_pool_acc"],
            -abs(best["train_pool_accepted_ratio"] - 0.5),
        )
        if key > best_key:
            best = candidate
    return best


def _row_for_strategy(
    *,
    dataset: str,
    seed: int,
    selected_message_scale: float,
    strategy: dict[str, Any],
    logits_off: torch.Tensor,
    logits_on: torch.Tensor,
    labels: torch.Tensor,
    train_mask: torch.Tensor,
    val_mask: torch.Tensor,
    test_mask: torch.Tensor,
    pool_mask: torch.Tensor,
) -> dict[str, Any]:
    accept = strategy["accept_mask"].bool()
    train_eval = _eval_acceptance(
        logits_off=logits_off,
        logits_on=logits_on,
        labels=labels,
        accept_mask=accept,
        split_mask=train_mask,
        pool_mask=pool_mask,
    )
    val_eval = _eval_acceptance(
        logits_off=logits_off,
        logits_on=logits_on,
        labels=labels,
        accept_mask=accept,
        split_mask=val_mask,
        pool_mask=pool_mask,
    )
    test_eval = _eval_acceptance(
        logits_off=logits_off,
        logits_on=logits_on,
        labels=labels,
        accept_mask=accept,
        split_mask=test_mask,
        pool_mask=pool_mask,
    )
    return {
        "dataset": dataset,
        "seed": seed,
        "selected_message_scale": selected_message_scale,
        "strategy": strategy["strategy"],
        "direction": strategy.get("direction", ""),
        "threshold": strategy.get("threshold", ""),
        "train_acc": train_eval["acc"],
        "val_acc": val_eval["acc"],
        "test_acc": test_eval["acc"],
        "train_pool_acc": train_eval["pool_acc"],
        "val_pool_acc": val_eval["pool_acc"],
        "test_pool_acc": test_eval["pool_acc"],
        "train_pool_accepted_ratio": train_eval["pool_accepted_ratio"],
        "val_pool_accepted_ratio": val_eval["pool_accepted_ratio"],
        "test_pool_accepted_ratio": test_eval["pool_accepted_ratio"],
        "test_pool_fixed": test_eval["pool_fixed"],
        "test_pool_broken": test_eval["pool_broken"],
        "test_prediction_change_ratio": test_eval["prediction_change_ratio"],
    }


def diagnose_run_rejection(
    result: dict[str, Any],
    *,
    repo_root: Path,
    device: torch.device,
) -> list[dict[str, Any]]:
    run_dir = Path(str(result["run_dir"]))
    checkpoint_path = Path(str(result.get("best_checkpoint_path") or run_dir / "best_model.pt"))
    config = read_yaml(run_dir / "config.yaml")
    loaded, graph, split, input_aligner, model, prompt_graph_module, _ = _instantiate_from_config(
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
    pool_mask = prompt_out["pool_mask"].detach().bool()
    train_mask = split.train_mask.bool()
    val_mask = split.val_mask.bool()
    test_mask = split.test_mask.bool()
    conf_off, margin_off = _confidence_and_margin(logits_off)
    conf_on, margin_on = _confidence_and_margin(logits_on)
    logit_delta = (logits_on - logits_off).norm(dim=-1)
    aux = prompt_out.get("aux", {})
    structural_score = aux.get("structural_score")
    if not isinstance(structural_score, torch.Tensor):
        structural_score = torch.zeros_like(labels, dtype=logits_off.dtype)
    pool_acceptance = aux.get("pool_acceptance_gate")
    acceptance_full = torch.zeros_like(labels, dtype=logits_off.dtype)
    if isinstance(pool_acceptance, torch.Tensor):
        pool_idx = torch.where(pool_mask)[0]
        if pool_acceptance.numel() == pool_idx.numel():
            acceptance_full[pool_idx] = pool_acceptance.detach().to(dtype=acceptance_full.dtype)

    strategies: list[dict[str, Any]] = [
        {"strategy": "prompt_off", "accept_mask": torch.zeros_like(pool_mask)},
        {"strategy": "prompt_on_all_pool", "accept_mask": pool_mask.clone()},
    ]
    oracle_test = _oracle_accept_mask(
        logits_off=logits_off,
        logits_on=logits_on,
        labels=labels,
        eligible_mask=pool_mask & test_mask,
    )
    strategies.append({"strategy": "oracle_test_upper_bound", "accept_mask": oracle_test})
    oracle_all = _oracle_accept_mask(
        logits_off=logits_off,
        logits_on=logits_on,
        labels=labels,
        eligible_mask=pool_mask,
    )
    strategies.append({"strategy": "oracle_all_splits_upper_bound", "accept_mask": oracle_all})

    threshold_specs = [
        ("train_low_conf_off", conf_off, "low"),
        ("train_low_margin_off", margin_off, "low"),
        ("train_high_structural_score", structural_score, "high"),
        ("train_high_logit_delta", logit_delta, "high"),
        ("train_high_learned_acceptance", acceptance_full, "high"),
        ("train_low_conf_on", conf_on, "low"),
        ("train_low_margin_on", margin_on, "low"),
    ]
    for name, values, direction in threshold_specs:
        best = _best_threshold_strategy(
            name=name,
            values=values,
            direction=direction,
            logits_off=logits_off,
            logits_on=logits_on,
            labels=labels,
            train_mask=train_mask,
            pool_mask=pool_mask,
        )
        if best is not None:
            strategies.append(best)

    rows = [
        _row_for_strategy(
            dataset=loaded.name,
            seed=int(result["seed"]),
            selected_message_scale=selected_scale,
            strategy=strategy,
            logits_off=logits_off,
            logits_on=logits_on,
            labels=labels,
            train_mask=train_mask,
            val_mask=val_mask,
            test_mask=test_mask,
            pool_mask=pool_mask,
        )
        for strategy in strategies
    ]
    return rows


def diagnose_summary(summary_path: Path, *, repo_root: Path, output_dir: Path) -> dict[str, Any]:
    payload = _load_json(summary_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rows: list[dict[str, Any]] = []
    for result in payload.get("runs", []):
        rows.extend(diagnose_run_rejection(result, repo_root=repo_root, device=device))
    if not rows:
        raise RuntimeError(f"No runs found in {summary_path}")

    dataset = str(rows[0]["dataset"])
    out_root = output_dir / dataset
    out_root.mkdir(parents=True, exist_ok=True)
    with (out_root / "rejection_rows.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    by_strategy: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_strategy.setdefault(str(row["strategy"]), []).append(row)

    def mean(items: list[dict[str, Any]], key: str) -> float:
        return float(sum(float(item[key]) for item in items) / float(len(items)))

    strategy_rows = []
    for strategy, items in sorted(by_strategy.items()):
        strategy_rows.append(
            {
                "dataset": dataset,
                "strategy": strategy,
                "runs": len(items),
                "mean_train_acc": mean(items, "train_acc"),
                "mean_val_acc": mean(items, "val_acc"),
                "mean_test_acc": mean(items, "test_acc"),
                "mean_train_pool_acc": mean(items, "train_pool_acc"),
                "mean_val_pool_acc": mean(items, "val_pool_acc"),
                "mean_test_pool_acc": mean(items, "test_pool_acc"),
                "mean_test_pool_accepted_ratio": mean(items, "test_pool_accepted_ratio"),
                "mean_test_pool_fixed": mean(items, "test_pool_fixed"),
                "mean_test_pool_broken": mean(items, "test_pool_broken"),
            }
        )
    with (out_root / "rejection_summary.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(strategy_rows[0].keys()))
        writer.writeheader()
        writer.writerows(strategy_rows)

    aggregate = {
        "source_summary": str(summary_path),
        "dataset": dataset,
        "rows": rows,
        "strategies": strategy_rows,
    }
    write_json(out_root / "rejection_summary.json", aggregate)
    return aggregate


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Post-hoc P2 rejection diagnostics")
    parser.add_argument("--summary_path", action="append", required=True)
    parser.add_argument("--output_dir", type=str, default="outputs/prompt_diagnostics/p2_rejection_oracle")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    output_dir = _resolve_path(args.output_dir, base_dir=repo_root)
    summaries = []
    for item in args.summary_path:
        aggregate = diagnose_summary(Path(item), repo_root=repo_root, output_dir=output_dir)
        summaries.append(aggregate)
        strategy_by_name = {row["strategy"]: row for row in aggregate["strategies"]}
        off = strategy_by_name.get("prompt_off", {})
        on = strategy_by_name.get("prompt_on_all_pool", {})
        oracle = strategy_by_name.get("oracle_test_upper_bound", {})
        print(
            f"{aggregate['dataset']}: off={float(off.get('mean_test_acc', 0.0)):.4f} "
            f"on={float(on.get('mean_test_acc', 0.0)):.4f} "
            f"oracle_test={float(oracle.get('mean_test_acc', 0.0)):.4f}"
        )
    write_json(output_dir / "all_rejection_summary.json", {"summaries": summaries})
    print(f"Saved rejection diagnostics to {output_dir}")


if __name__ == "__main__":
    main()
