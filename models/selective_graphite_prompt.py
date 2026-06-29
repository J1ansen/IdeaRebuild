"""Class-aware selective GRAPHITE prompt graph adapter.

This module keeps the original GRAPHITE-style idea of adding feature nodes, but
adds lightweight reliability controls so high-degree mixed-label hubs do not
dominate imbalanced datasets.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import nn
import torch.nn.functional as F


def _safe_entropy(prob: torch.Tensor) -> torch.Tensor:
    if prob.numel() == 0:
        return prob.new_zeros(prob.shape[:-1])
    denom = torch.log(prob.new_tensor(float(max(2, prob.size(-1)))))
    entropy = -(prob * prob.clamp_min(1e-12).log()).sum(dim=-1)
    return entropy / denom.clamp_min(1e-12)


class ClassAwareSelectiveGraphitePromptGraphAdapter(nn.Module):
    """Selective GRAPHITE adapter with label-aware feature reliability.

    Feature prompts are selected by labelled-neighbor purity and anti-hub
    statistics. The adapted GP2F branch receives the expanded graph, while the
    frozen branch keeps the original graph.
    """

    def __init__(self, source_dim: int, hidden_dim: int, config: dict[str, Any] | None = None) -> None:
        super().__init__()
        self.source_dim = int(source_dim)
        self.hidden_dim = int(hidden_dim)
        self.config = dict(config or {})
        self.num_classes = int(self.config.get("num_classes", 0))

        tokenizer_cfg = dict(self.config.get("feature_tokenizer", {}))
        self.tokenizer = str(tokenizer_cfg.get("mode", self.config.get("tokenizer", "binary_nonzero")))
        if self.tokenizer not in {"binary_nonzero", "topk_activation"}:
            raise ValueError("Selective GRAPHITE supports binary_nonzero and topk_activation tokenizers")
        self.binary_threshold = float(tokenizer_cfg.get("binary_threshold", self.config.get("binary_threshold", 0.0)))
        self.binary_topk = int(tokenizer_cfg.get("binary_topk", tokenizer_cfg.get("max_nonzero_per_node", 0)))
        self.topk = int(tokenizer_cfg.get("topk", self.config.get("topk_feature_dims", 16)))

        filter_cfg = dict(self.config.get("feature_filter", {}))
        self.min_df_global = float(filter_cfg.get("min_df_global", filter_cfg.get("min_df_pool", 2)))
        self.max_df_global_ratio = float(filter_cfg.get("max_df_global_ratio", filter_cfg.get("max_df_ratio", 1.0)))
        self.min_label_count = float(filter_cfg.get("min_label_count", 1))
        self.min_purity = float(filter_cfg.get("min_purity", 0.0))
        self.max_entropy = float(filter_cfg.get("max_entropy", 1.0))
        self.max_feature_nodes = int(filter_cfg.get("max_feature_nodes", 0))

        self.feature_edge_weight = float(self.config.get("feature_edge_weight", 1.0))
        self.original_edge_weight = float(self.config.get("original_edge_weight", 1.0))
        self.static_graph = bool(self.config.get("static_graph", True))
        self.detach_feature_prompt = bool(self.config.get("detach_feature_prompt", True))
        self.anti_hub_power = float(self.config.get("anti_hub_power", 0.5))
        self.reliability_power = float(self.config.get("reliability_power", 1.0))
        self.label_node_weight = float(self.config.get("label_node_weight", 4.0))
        self.unlabeled_node_weight = float(self.config.get("unlabeled_node_weight", 0.25))
        self.offclass_label_weight = float(self.config.get("offclass_label_weight", 0.05))
        self.edge_dropout = float(self.config.get("edge_dropout", 0.0))
        self.learn_feature_edge_weight = bool(self.config.get("learn_feature_edge_weight", True))
        self.use_feature_gate = bool(self.config.get("use_feature_gate", True))
        self.feature_edge_min_scale = float(self.config.get("feature_edge_min_scale", 0.0))
        self.feature_gate_floor = float(self.config.get("feature_gate_floor", 0.0))
        if not 0.0 <= self.feature_gate_floor < 1.0:
            raise ValueError("feature_gate_floor must be in [0, 1)")
        if self.feature_edge_min_scale < 0.0:
            raise ValueError("feature_edge_min_scale must be non-negative")

        if self.learn_feature_edge_weight:
            init_dynamic = max(self.feature_edge_weight - self.feature_edge_min_scale, 1e-6)
            self.feature_edge_log_scale = nn.Parameter(torch.log(torch.tensor(init_dynamic)))
        else:
            self.register_parameter("feature_edge_log_scale", None)

        self.feature_gate = nn.Sequential(
            nn.LayerNorm(6),
            nn.Linear(6, 16),
            nn.GELU(),
            nn.Linear(16, 1),
        )
        nn.init.zeros_(self.feature_gate[-1].weight)
        nn.init.zeros_(self.feature_gate[-1].bias)

        self._cached_graph: dict[str, Any] | None = None
        self._cache_signature: tuple[int, int, int, int, str] | None = None

    def clear_cache(self) -> None:
        self._cached_graph = None
        self._cache_signature = None

    def _token_membership(self, x_raw: torch.Tensor) -> torch.Tensor:
        if self.tokenizer == "binary_nonzero":
            membership = x_raw.abs() > self.binary_threshold
            if self.binary_topk > 0 and self.binary_topk < int(x_raw.size(1)):
                capped = torch.zeros_like(membership)
                scores = x_raw.abs().masked_fill(~membership, float("-inf"))
                topk = torch.topk(scores, k=self.binary_topk, dim=-1).indices
                capped.scatter_(1, topk, True)
                membership = capped & membership
            return membership
        k = min(max(1, self.topk), int(x_raw.size(1)))
        topk = torch.topk(x_raw.abs(), k=k, dim=-1).indices
        membership = torch.zeros(x_raw.shape, dtype=torch.bool, device=x_raw.device)
        membership.scatter_(1, topk, True)
        return membership

    def _signature(self, z: torch.Tensor, x_raw: torch.Tensor, edge_index: torch.Tensor) -> tuple[int, int, int, int, str]:
        return (int(z.size(0)), int(z.size(1)), int(x_raw.size(1)), int(edge_index.size(1)), str(z.device))

    def _feature_scale(self, ref: torch.Tensor, edge_scale_multiplier: float | torch.Tensor) -> torch.Tensor:
        scale = torch.as_tensor(edge_scale_multiplier, dtype=ref.dtype, device=ref.device)
        if self.feature_edge_log_scale is None:
            base = ref.new_tensor(float(self.feature_edge_weight))
        else:
            dynamic = self.feature_edge_log_scale.to(device=ref.device, dtype=ref.dtype).exp()
            base = ref.new_tensor(float(self.feature_edge_min_scale)) + dynamic
        return base * scale

    def _build_graph(
        self,
        *,
        z: torch.Tensor,
        x_raw: torch.Tensor,
        edge_index: torch.Tensor,
        train_mask: torch.Tensor,
        labels: torch.Tensor | None,
    ) -> dict[str, Any]:
        device = z.device
        dtype = z.dtype
        membership = self._token_membership(x_raw.to(device=device))
        num_nodes = int(z.size(0))
        num_features = int(membership.size(1))
        df = membership.to(dtype=dtype).sum(dim=0)
        valid = (df >= self.min_df_global) & (df <= max(1.0, self.max_df_global_ratio * float(num_nodes)))

        train = train_mask.to(device=device, dtype=torch.bool)
        y = labels.to(device=device, dtype=torch.long) if labels is not None else None
        class_count = max(1, self.num_classes)
        label_counts = z.new_zeros((num_features, class_count))
        if y is not None and bool(train.any()):
            train_membership = membership[train].to(dtype=dtype)
            train_y = y[train].clamp_min(0)
            one_hot = F.one_hot(train_y, num_classes=class_count).to(dtype=dtype)
            label_counts = train_membership.t() @ one_hot
        total_label = label_counts.sum(dim=-1)
        prob = label_counts / total_label.clamp_min(1.0).unsqueeze(-1)
        purity, dominant_class = prob.max(dim=-1)
        entropy = _safe_entropy(prob)
        sorted_prob, _ = torch.sort(prob, dim=-1, descending=True)
        margin = sorted_prob[:, 0] - (sorted_prob[:, 1] if sorted_prob.size(1) > 1 else 0.0)
        evidence = (total_label / max(1.0, float(train.sum().item()))).clamp(0.0, 1.0).sqrt()
        idf = torch.log((z.new_tensor(float(num_nodes)) + 1.0) / (df + 1.0))
        idf_norm = idf / idf.max().clamp_min(1e-12)
        reliability = (purity * (1.0 - entropy).clamp_min(0.0) * margin.clamp_min(0.0) * evidence.clamp_min(1e-6)).pow(
            self.reliability_power
        )
        reliability = reliability * (0.5 + 0.5 * idf_norm)
        valid = valid & (total_label >= self.min_label_count) & (purity >= self.min_purity) & (entropy <= self.max_entropy)
        candidate = torch.where(valid)[0]
        if self.max_feature_nodes > 0 and candidate.numel() > self.max_feature_nodes:
            top = torch.topk(reliability[candidate], k=self.max_feature_nodes, largest=True).indices
            candidate = candidate[top]
        feature_ids = candidate

        if feature_ids.numel() == 0:
            prompt_x = z.new_zeros((0, z.size(1)))
            prompt_edges = torch.empty((2, 0), dtype=torch.long, device=edge_index.device)
            edge_feature_local = torch.empty(0, dtype=torch.long, device=device)
            edge_node_index = torch.empty(0, dtype=torch.long, device=device)
            feature_stats = z.new_zeros((0, 6))
            base_edge_weight = z.new_zeros(0)
            usage = z.new_zeros(0)
            feature_dominant_class = torch.empty(0, dtype=torch.long, device=device)
        else:
            selected_membership = membership[:, feature_ids]
            dominant = dominant_class[feature_ids]
            node_idx, feature_local = torch.nonzero(selected_membership, as_tuple=True)
            weights = torch.ones((num_nodes, feature_ids.numel()), dtype=dtype, device=device) * self.unlabeled_node_weight
            if y is not None and bool(train.any()):
                train_idx = torch.where(train)[0]
                train_labels = y[train_idx]
                connected_train = selected_membership[train_idx]
                label_match = train_labels.unsqueeze(-1) == dominant.unsqueeze(0)
                weights[train_idx] = torch.where(
                    connected_train & label_match,
                    weights.new_tensor(self.label_node_weight),
                    torch.where(connected_train, weights.new_tensor(self.offclass_label_weight), weights[train_idx]),
                )
            weighted_membership = selected_membership.to(dtype=dtype) * weights
            sums = weighted_membership.t() @ z
            denom = weighted_membership.sum(dim=0).clamp_min(1.0)
            prompt_x = sums / denom.unsqueeze(-1)
            if self.detach_feature_prompt:
                prompt_x = prompt_x.detach()

            prompt_node = feature_local + num_nodes
            node_to_prompt = torch.stack([node_idx, prompt_node], dim=0)
            prompt_to_node = torch.stack([prompt_node, node_idx], dim=0)
            prompt_edges = torch.cat([node_to_prompt, prompt_to_node], dim=1).to(device=edge_index.device)
            edge_feature_local = torch.cat([feature_local, feature_local], dim=0)
            edge_node_index = torch.cat([node_idx, node_idx], dim=0)
            usage = torch.zeros(feature_ids.numel(), dtype=dtype, device=device)
            usage.index_add_(0, feature_local, torch.ones_like(feature_local, dtype=dtype))
            rel = reliability[feature_ids].clamp_min(0.0)
            anti_hub = usage.clamp_min(1.0).pow(-self.anti_hub_power)
            base_feature_weight = rel * anti_hub
            if base_feature_weight.numel() > 0:
                base_feature_weight = base_feature_weight / base_feature_weight.mean().clamp_min(1e-6)
            base_edge_weight = base_feature_weight[edge_feature_local]
            feature_stats = torch.stack(
                [
                    (df[feature_ids] / max(1.0, float(num_nodes))).clamp(0.0, 1.0),
                    purity[feature_ids],
                    entropy[feature_ids],
                    margin[feature_ids],
                    evidence[feature_ids],
                    idf_norm[feature_ids],
                ],
                dim=-1,
            )
            feature_dominant_class = dominant.detach()

        usage_prob = usage / usage.sum().clamp_min(1.0) if usage.numel() > 0 else usage
        usage_entropy = -(usage_prob * usage_prob.clamp_min(1e-12).log()).sum()
        if usage_prob.numel() > 1:
            usage_entropy = usage_entropy / torch.log(usage_prob.new_tensor(float(usage_prob.numel()))).clamp_min(1e-12)

        return {
            "prompt_x": prompt_x,
            "prompt_edges": prompt_edges,
            "edge_feature_local": edge_feature_local,
            "edge_node_index": edge_node_index,
            "base_edge_weight": base_edge_weight,
            "pool_mask": torch.ones(num_nodes, dtype=torch.bool, device=device),
            "feature_stats": feature_stats.detach(),
            "feature_ids": feature_ids.detach(),
            "feature_dominant_class": feature_dominant_class.detach(),
            "usage": usage.detach(),
            "aux": {
                "prompt_usage": usage.detach(),
                "prompt_usage_entropy": usage_entropy.detach(),
                "connected_edge_count": int(prompt_edges.size(1)),
                "edge_type_counts": [int(edge_index.size(1)), int(prompt_edges.size(1)), 0],
                "p23_selective_graphite_enabled": z.new_tensor(1.0),
                "p23_pool_ratio": z.new_tensor(1.0),
                "p23_feature_prompt_count": z.new_tensor(float(feature_ids.numel())),
                "p23_prompt_edge_count": z.new_tensor(float(prompt_edges.size(1))),
                "p23_selective_feature_prompt_count": z.new_tensor(float(feature_ids.numel())),
                "p23_selective_feature_edge_count": z.new_tensor(float(prompt_edges.size(1))),
                "p23_selective_valid_feature_ratio": valid.to(dtype=dtype).mean() if valid.numel() > 0 else z.new_tensor(0.0),
                "p23_selective_feature_purity_mean": purity[feature_ids].mean() if feature_ids.numel() > 0 else z.new_tensor(0.0),
                "p23_selective_feature_entropy_mean": entropy[feature_ids].mean() if feature_ids.numel() > 0 else z.new_tensor(0.0),
                "p23_selective_feature_margin_mean": margin[feature_ids].mean() if feature_ids.numel() > 0 else z.new_tensor(0.0),
                "p23_selective_feature_label_count_mean": total_label[feature_ids].mean() if feature_ids.numel() > 0 else z.new_tensor(0.0),
                "p23_selective_feature_degree_max": usage.max() if usage.numel() > 0 else z.new_tensor(0.0),
            },
        }

    def _edge_weight(self, graph: dict[str, Any], ref: torch.Tensor, edge_scale_multiplier: float | torch.Tensor) -> torch.Tensor:
        base = graph["base_edge_weight"].to(device=ref.device, dtype=ref.dtype)
        if base.numel() == 0:
            return base
        if self.use_feature_gate:
            gate = self._feature_gate_values(graph, ref)
            edge_feature_local = graph["edge_feature_local"].to(device=ref.device)
            base = base * gate[edge_feature_local]
        weight = base * self._feature_scale(ref, edge_scale_multiplier)
        if self.training and self.edge_dropout > 0.0:
            keep = torch.rand_like(weight) >= self.edge_dropout
            weight = weight * keep.to(dtype=weight.dtype) / max(1e-6, 1.0 - self.edge_dropout)
        return weight

    def _feature_gate_values(self, graph: dict[str, Any], ref: torch.Tensor) -> torch.Tensor:
        stats = graph["feature_stats"].to(device=ref.device, dtype=ref.dtype)
        if stats.numel() == 0:
            return stats.new_zeros(0)
        if not self.use_feature_gate:
            return stats.new_ones(stats.size(0))
        gate = torch.sigmoid(self.feature_gate(stats).squeeze(-1))
        if self.feature_gate_floor > 0.0:
            floor = ref.new_tensor(float(self.feature_gate_floor))
            gate = floor + (1.0 - floor) * gate
        return gate

    def forward(
        self,
        *,
        z: torch.Tensor,
        h_pre: torch.Tensor,
        edge_index: torch.Tensor,
        train_mask: torch.Tensor,
        edge_scale_multiplier: float | torch.Tensor = 1.0,
        no_prompt_logits: torch.Tensor | None = None,
        h_adp_no_prompt: torch.Tensor | None = None,
        x_raw: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
    ) -> dict[str, Any]:
        del h_pre, no_prompt_logits, h_adp_no_prompt
        x_raw = z if x_raw is None else x_raw.to(device=z.device)
        signature = self._signature(z, x_raw, edge_index)
        if self.static_graph and self._cached_graph is not None and self._cache_signature == signature:
            graph = self._cached_graph
            cache_hit = True
        else:
            graph = self._build_graph(z=z, x_raw=x_raw, edge_index=edge_index, train_mask=train_mask, labels=labels)
            self._cached_graph = {
                key: value.detach() if isinstance(value, torch.Tensor) else value
                for key, value in graph.items()
            }
            # Aux contains ints and tensors; detach tensors while preserving scalars.
            self._cached_graph["aux"] = {
                key: value.detach() if isinstance(value, torch.Tensor) else value
                for key, value in graph["aux"].items()
            }
            self._cache_signature = signature if self.static_graph else None
            cache_hit = False

        prompt_x = graph["prompt_x"].to(device=z.device, dtype=z.dtype)
        prompt_edges = graph["prompt_edges"].to(device=edge_index.device)
        prompt_edge_weight = self._edge_weight(graph, z, edge_scale_multiplier)
        feature_gate = self._feature_gate_values(graph, z)
        adapted_x = torch.cat([z, prompt_x], dim=0)
        base_weight = torch.ones(edge_index.size(1), dtype=z.dtype, device=z.device) * float(self.original_edge_weight)
        adapted_edge_index = torch.cat([edge_index, prompt_edges], dim=1)
        adapted_edge_weight = torch.cat([base_weight, prompt_edge_weight], dim=0)
        adapted_edge_type = torch.cat(
            [
                torch.zeros(edge_index.size(1), dtype=torch.long, device=edge_index.device),
                torch.ones(prompt_edges.size(1), dtype=torch.long, device=edge_index.device),
            ],
            dim=0,
        )
        aux = {
            key: (value.to(device=z.device, dtype=z.dtype) if isinstance(value, torch.Tensor) and value.is_floating_point() else value)
            for key, value in graph["aux"].items()
        }
        aux["prompt_edge_weight"] = prompt_edge_weight
        aux["p23_selective_edge_feature_local"] = graph["edge_feature_local"].to(device=z.device)
        aux["p23_selective_edge_node_index"] = graph["edge_node_index"].to(device=z.device)
        aux["p23_selective_feature_gate"] = feature_gate
        aux["p23_selective_feature_dominant_class"] = graph["feature_dominant_class"].to(device=z.device)
        aux["p23_selective_static_cache_hit"] = z.new_tensor(float(cache_hit))
        aux["p23_selective_feature_edge_weight_mean"] = prompt_edge_weight.mean() if prompt_edge_weight.numel() > 0 else z.new_tensor(0.0)
        return {
            "adapted_x": adapted_x,
            "adapted_edge_index": adapted_edge_index,
            "adapted_edge_weight": adapted_edge_weight,
            "adapted_edge_type": adapted_edge_type,
            "prompt_node_x": prompt_x,
            "pool_mask": graph["pool_mask"].to(device=z.device),
            "prompt_edge_count": int(prompt_edges.size(1)),
            "edge_scale": torch.as_tensor(edge_scale_multiplier, dtype=z.dtype, device=z.device),
            "aux": aux,
        }
