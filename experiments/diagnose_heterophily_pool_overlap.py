"""Diagnose overlap between candidate pools and heterophilic nodes.

This script is intentionally model-free. It checks whether a static pool
construction can cover nodes that are incident to heterophilic edges
(`y_u != y_v`) before wiring the pool into P23 training.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch

from data import build_few_shot_split, load_node_dataset
from experiments.run_gp2f_baseline import set_seed


def _to_undirected_unique(edge_index: torch.Tensor) -> torch.Tensor:
    src, dst = edge_index.to(dtype=torch.long)
    keep = src != dst
    src = src[keep]
    dst = dst[keep]
    lo = torch.minimum(src, dst)
    hi = torch.maximum(src, dst)
    edges = torch.stack([lo, hi], dim=0)
    if edges.numel() == 0:
        return edges
    num_nodes = int(edge_index.max().item()) + 1 if edge_index.numel() else 0
    key = edges[0] * max(1, num_nodes) + edges[1]
    order = torch.argsort(key)
    edges = edges[:, order]
    key = key[order]
    unique = torch.ones(key.numel(), dtype=torch.bool, device=key.device)
    unique[1:] = key[1:] != key[:-1]
    return edges[:, unique]


def heterophily_stats(edge_index: torch.Tensor, labels: torch.Tensor, num_nodes: int) -> dict[str, torch.Tensor]:
    undirected = _to_undirected_unique(edge_index.cpu())
    y = labels.cpu().view(-1).to(dtype=torch.long)
    hetero_edge = y[undirected[0]] != y[undirected[1]]
    incident_total = torch.zeros(num_nodes, dtype=torch.float32)
    incident_hetero = torch.zeros(num_nodes, dtype=torch.float32)
    if undirected.numel() > 0:
        ones = torch.ones(undirected.size(1), dtype=torch.float32)
        incident_total.index_add_(0, undirected[0], ones)
        incident_total.index_add_(0, undirected[1], ones)
        hetero_ones = hetero_edge.to(dtype=torch.float32)
        incident_hetero.index_add_(0, undirected[0], hetero_ones)
        incident_hetero.index_add_(0, undirected[1], hetero_ones)
    hetero_ratio = incident_hetero / incident_total.clamp_min(1.0)
    hetero_node = incident_hetero > 0
    return {
        "edge_index_undirected": undirected,
        "hetero_edge": hetero_edge,
        "incident_total": incident_total,
        "incident_hetero": incident_hetero,
        "hetero_ratio": hetero_ratio,
        "hetero_node": hetero_node,
    }


def _top_ratio_pool(score: torch.Tensor, ratio: float) -> torch.Tensor:
    n = int(score.numel())
    k = max(1, min(n, int(round(float(ratio) * n))))
    pool = torch.zeros(n, dtype=torch.bool)
    if k <= 0:
        return pool
    idx = torch.topk(score, k=k, largest=True).indices
    pool[idx] = True
    return pool


def _random_pool(num_nodes: int, ratio: float, seed: int) -> torch.Tensor:
    k = max(1, min(num_nodes, int(round(float(ratio) * num_nodes))))
    gen = torch.Generator()
    gen.manual_seed(int(seed))
    perm = torch.randperm(num_nodes, generator=gen)
    pool = torch.zeros(num_nodes, dtype=torch.bool)
    pool[perm[:k]] = True
    return pool


def _degree_score(num_nodes: int, undirected: torch.Tensor) -> torch.Tensor:
    degree = torch.zeros(num_nodes, dtype=torch.float32)
    if undirected.numel() == 0:
        return degree
    ones = torch.ones(undirected.size(1), dtype=torch.float32)
    degree.index_add_(0, undirected[0], ones)
    degree.index_add_(0, undirected[1], ones)
    return degree


def _support_neighbor_entropy_score(
    *,
    num_nodes: int,
    undirected: torch.Tensor,
    labels: torch.Tensor,
    train_mask: torch.Tensor,
    num_classes: int,
) -> torch.Tensor:
    counts = torch.zeros((num_nodes, num_classes), dtype=torch.float32)
    y = labels.cpu().view(-1).to(dtype=torch.long)
    train = train_mask.cpu().to(dtype=torch.bool)
    if undirected.numel() == 0:
        return counts.new_zeros(num_nodes)
    src, dst = undirected
    for center, neigh in ((src, dst), (dst, src)):
        labeled = train[neigh]
        if bool(labeled.any()):
            cls = y[neigh[labeled]]
            counts.index_add_(0, center[labeled], torch.nn.functional.one_hot(cls, num_classes=num_classes).to(torch.float32))
    total = counts.sum(dim=-1)
    prob = counts / total.clamp_min(1.0).unsqueeze(-1)
    entropy = -(prob * prob.clamp_min(1e-12).log()).sum(dim=-1)
    if num_classes > 1:
        entropy = entropy / torch.log(entropy.new_tensor(float(num_classes))).clamp_min(1e-12)
    coverage = (total > 0).to(torch.float32)
    confidence = 1.0 - prob.max(dim=-1).values
    return coverage * (0.5 * entropy + 0.5 * confidence)


def _feature_label_ambiguity_score(
    *,
    x: torch.Tensor,
    labels: torch.Tensor,
    train_mask: torch.Tensor,
    num_classes: int,
    topk: int,
) -> torch.Tensor:
    x_cpu = x.cpu()
    y = labels.cpu().view(-1).to(dtype=torch.long)
    train = train_mask.cpu().to(dtype=torch.bool)
    if x_cpu.numel() == 0 or not bool(train.any()):
        return torch.zeros(x_cpu.size(0), dtype=torch.float32)
    k = min(max(1, int(topk)), int(x_cpu.size(1)))
    membership = torch.zeros_like(x_cpu, dtype=torch.bool)
    membership.scatter_(1, torch.topk(x_cpu.abs(), k=k, dim=-1).indices, True)
    train_membership = membership[train].to(torch.float32)
    one_hot = torch.nn.functional.one_hot(y[train], num_classes=num_classes).to(torch.float32)
    label_counts = train_membership.t() @ one_hot
    totals = label_counts.sum(dim=-1)
    prob = label_counts / totals.clamp_min(1.0).unsqueeze(-1)
    entropy = -(prob * prob.clamp_min(1e-12).log()).sum(dim=-1)
    if num_classes > 1:
        entropy = entropy / torch.log(entropy.new_tensor(float(num_classes))).clamp_min(1e-12)
    ambiguity = torch.where(totals > 0, entropy, torch.zeros_like(entropy))
    return (membership.to(torch.float32) @ ambiguity) / float(k)


def pool_scores(
    *,
    strategy: str,
    num_nodes: int,
    x: torch.Tensor,
    labels: torch.Tensor,
    train_mask: torch.Tensor,
    num_classes: int,
    stats: dict[str, torch.Tensor],
    feature_topk: int,
) -> torch.Tensor:
    if strategy == "oracle_hetero_ratio":
        return stats["hetero_ratio"]
    if strategy == "degree":
        return _degree_score(num_nodes, stats["edge_index_undirected"])
    if strategy == "support_neighbor_entropy":
        return _support_neighbor_entropy_score(
            num_nodes=num_nodes,
            undirected=stats["edge_index_undirected"],
            labels=labels,
            train_mask=train_mask,
            num_classes=num_classes,
        )
    if strategy == "feature_label_ambiguity":
        return _feature_label_ambiguity_score(
            x=x,
            labels=labels,
            train_mask=train_mask,
            num_classes=num_classes,
            topk=feature_topk,
        )
    if strategy == "hybrid_support_feature":
        a = pool_scores(
            strategy="support_neighbor_entropy",
            num_nodes=num_nodes,
            x=x,
            labels=labels,
            train_mask=train_mask,
            num_classes=num_classes,
            stats=stats,
            feature_topk=feature_topk,
        )
        b = pool_scores(
            strategy="feature_label_ambiguity",
            num_nodes=num_nodes,
            x=x,
            labels=labels,
            train_mask=train_mask,
            num_classes=num_classes,
            stats=stats,
            feature_topk=feature_topk,
        )
        return 0.5 * _minmax(a) + 0.5 * _minmax(b)
    raise ValueError(f"Unsupported pool strategy: {strategy}")


def _minmax(score: torch.Tensor) -> torch.Tensor:
    lo = score.min()
    hi = score.max()
    return (score - lo) / (hi - lo).clamp_min(1e-12)


def overlap_metrics(pool: torch.Tensor, target: torch.Tensor) -> dict[str, float]:
    pool = pool.to(dtype=torch.bool).cpu()
    target = target.to(dtype=torch.bool).cpu()
    tp = float((pool & target).sum().item())
    fp = float((pool & ~target).sum().item())
    fn = float((~pool & target).sum().item())
    pool_count = float(pool.sum().item())
    target_count = float(target.sum().item())
    precision = tp / max(1.0, pool_count)
    recall = tp / max(1.0, target_count)
    f1 = 2.0 * precision * recall / max(1e-12, precision + recall)
    iou = tp / max(1.0, tp + fp + fn)
    return {
        "pool_count": pool_count,
        "hetero_node_count": target_count,
        "overlap_count": tp,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "iou": iou,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Diagnose candidate-pool overlap with heterophilic nodes.")
    parser.add_argument("--dataset", default="Chameleon")
    parser.add_argument("--data_root", default="/Users/jackson/MyIdea/data")
    parser.add_argument("--download_if_missing", action="store_true")
    parser.add_argument("--shot_ratio", type=float, default=0.10)
    parser.add_argument("--shots", type=int, default=5)
    parser.add_argument("--val_per_class", type=int, default=30)
    parser.add_argument("--seeds", type=str, default="0")
    parser.add_argument("--pool_ratios", type=str, default="0.05,0.10,0.15,0.20,0.30")
    parser.add_argument(
        "--hetero_ratio_thresholds",
        type=str,
        default="0.0,0.25,0.50,0.75",
        help="Node is target-positive if its incident heterophilic-edge ratio is at least this value.",
    )
    parser.add_argument(
        "--strategies",
        type=str,
        default="random,degree,support_neighbor_entropy,feature_label_ambiguity,hybrid_support_feature,oracle_hetero_ratio",
    )
    parser.add_argument("--feature_topk", type=int, default=16)
    parser.add_argument("--output_csv", default="")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    seeds = [int(item.strip()) for item in args.seeds.split(",") if item.strip()]
    ratios = [float(item.strip()) for item in args.pool_ratios.split(",") if item.strip()]
    target_thresholds = [float(item.strip()) for item in args.hetero_ratio_thresholds.split(",") if item.strip()]
    strategies = [item.strip() for item in args.strategies.split(",") if item.strip()]
    loaded = load_node_dataset(args.dataset, args.data_root, download_if_missing=bool(args.download_if_missing))
    data = loaded.data
    num_nodes = int(data.y.numel())
    stats = heterophily_stats(data.edge_index, data.y, num_nodes)
    hetero_edge_ratio = float(stats["hetero_edge"].to(torch.float32).mean().item()) if stats["hetero_edge"].numel() else 0.0
    hetero_node = stats["hetero_node"]
    hetero_node_ratio = float(hetero_node.to(torch.float32).mean().item())

    rows: list[dict[str, Any]] = []
    print(
        f"Dataset={loaded.name} | nodes={num_nodes} | edges_undirected={stats['edge_index_undirected'].size(1)} | "
        f"hetero_edge_ratio={hetero_edge_ratio:.4f} | hetero_node_ratio={hetero_node_ratio:.4f}"
    )
    header = (
        "seed target strategy ratio pool% overlap% precision recall f1 iou "
        "hetero_nodes pool_nodes score_mean_in score_mean_out"
    )
    print(header)
    for seed in seeds:
        set_seed(seed)
        split = build_few_shot_split(
            data.y,
            shots=int(args.shots),
            val_per_class=int(args.val_per_class),
            seed=seed,
            shot_ratio=float(args.shot_ratio),
        )
        for strategy in strategies:
            score = None
            if strategy != "random":
                score = pool_scores(
                    strategy=strategy,
                    num_nodes=num_nodes,
                    x=data.x,
                    labels=data.y,
                    train_mask=split.train_mask,
                    num_classes=int(loaded.num_classes),
                    stats=stats,
                    feature_topk=int(args.feature_topk),
                ).cpu()
            for ratio in ratios:
                if strategy == "random":
                    pool = _random_pool(num_nodes, ratio, seed=seed)
                    score_for_stats = torch.zeros(num_nodes)
                else:
                    pool = _top_ratio_pool(score, ratio)
                    score_for_stats = score
                for target_threshold in target_thresholds:
                    if target_threshold <= 0.0:
                        target = hetero_node
                    else:
                        target = stats["hetero_ratio"] >= float(target_threshold)
                    metrics = overlap_metrics(pool, target)
                    score_in = (
                        float(score_for_stats[pool & target].mean().item())
                        if bool((pool & target).any())
                        else 0.0
                    )
                    score_out = (
                        float(score_for_stats[~pool & target].mean().item())
                        if bool((~pool & target).any())
                        else 0.0
                    )
                    row = {
                        "dataset": loaded.name,
                        "seed": seed,
                        "target_hetero_ratio_threshold": target_threshold,
                        "strategy": strategy,
                        "pool_ratio": ratio,
                        "pool_node_ratio": metrics["pool_count"] / max(1.0, float(num_nodes)),
                        "hetero_edge_ratio": hetero_edge_ratio,
                        "hetero_node_ratio_any": hetero_node_ratio,
                        "target_node_ratio": metrics["hetero_node_count"] / max(1.0, float(num_nodes)),
                        **metrics,
                        "score_mean_overlap": score_in,
                        "score_mean_missed_hetero": score_out,
                    }
                    rows.append(row)
                    print(
                        f"{seed} >= {target_threshold:.2f} {strategy} {ratio:.2f} {row['pool_node_ratio']:.4f} "
                        f"{metrics['overlap_count'] / max(1.0, float(num_nodes)):.4f} "
                        f"{metrics['precision']:.4f} {metrics['recall']:.4f} "
                        f"{metrics['f1']:.4f} {metrics['iou']:.4f} "
                        f"{int(metrics['hetero_node_count'])} {int(metrics['pool_count'])} "
                        f"{score_in:.4f} {score_out:.4f}"
                    )

    if args.output_csv:
        path = Path(args.output_csv)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else [])
            writer.writeheader()
            writer.writerows(rows)
        print(f"Saved CSV to {path}")


if __name__ == "__main__":
    main()
