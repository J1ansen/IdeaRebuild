"""Prompt-aware GP2F variant for P2 adapted-branch message passing."""

from __future__ import annotations

from typing import Any

import torch
from torch import nn
import torch.nn.functional as F

from models.faithful_gp2f import FaithfulGP2F


def _gate_logit(value: float) -> torch.Tensor:
    value = min(max(float(value), 1e-6), 1.0 - 1e-6)
    return torch.logit(torch.tensor(value, dtype=torch.float32))


def _safe_mean_norm(x: torch.Tensor) -> torch.Tensor:
    if x.numel() == 0:
        return x.new_tensor(0.0)
    return x.norm(dim=-1).mean()


def _prompt_gate_init_tensor(
    *,
    num_layers: int,
    default_value: float,
    node_to_prompt_value: float | None,
    prompt_to_node_value: float | None,
) -> torch.Tensor:
    node_value = default_value if node_to_prompt_value is None else node_to_prompt_value
    prompt_value = default_value if prompt_to_node_value is None else prompt_to_node_value
    values = torch.tensor([float(node_value), float(prompt_value)], dtype=torch.float32)
    return torch.stack([_gate_logit(float(item)) for item in values]).repeat(num_layers, 1)


class PromptAwareGP2F(FaithfulGP2F):
    """GP2F with direction-aware prompt-edge messages in the adapted branch.

    Original graph edges still use the frozen backbone convolutions plus
    residual adapters. Prompt edges are consumed by separate per-layer message
    transforms and small learnable gates.
    """

    def __init__(
        self,
        backbone: nn.Module,
        *,
        hidden_dim: int,
        num_classes: int,
        adapter_bottleneck_dim: int = 16,
        adapter_beta_init: float = 0.01,
        adapter_alpha_init: float = 0.1,
        adapter_style: str = "stable_zero_init",
        alpha_init: float = 0.5,
        fusion_alpha_style: str = "sigmoid",
        prompt_aware_config: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(
            backbone,
            hidden_dim=hidden_dim,
            num_classes=num_classes,
            adapter_bottleneck_dim=adapter_bottleneck_dim,
            adapter_beta_init=adapter_beta_init,
            adapter_alpha_init=adapter_alpha_init,
            adapter_style=adapter_style,
            alpha_init=alpha_init,
            fusion_alpha_style=fusion_alpha_style,
        )
        cfg = dict(prompt_aware_config or {})
        self.prompt_aware_dropout = float(cfg.get("dropout", 0.2))
        self.use_node_to_prompt = bool(cfg.get("use_node_to_prompt", True))
        self.use_prompt_to_node = bool(cfg.get("use_prompt_to_node", True))
        gate_init = float(cfg.get("gate_init", 0.01))
        node_to_prompt_gate_init = cfg.get("node_to_prompt_gate_init")
        prompt_to_node_gate_init = cfg.get("prompt_to_node_gate_init")
        self.prompt_message_scale = float(cfg.get("message_scale", 1.0))
        self.prompt_message_norm = str(cfg.get("message_norm", "weighted_mean"))
        self.pool_only_prompt_update = bool(cfg.get("pool_only_prompt_update", False))
        valid_norms = {"weighted_mean", "weighted_sum", "degree_mean"}
        if self.prompt_message_norm not in valid_norms:
            raise ValueError(
                f"Unsupported prompt_aware.message_norm={self.prompt_message_norm!r}; "
                f"expected one of {sorted(valid_norms)}"
            )

        in_dim = int(self.backbone.convs[0].lin.weight.shape[1])
        layer_input_dims = [in_dim] + [self.hidden_dim for _ in range(len(self.backbone.convs) - 1)]
        self.node_to_prompt_msgs = nn.ModuleList(
            [nn.Linear(dim, self.hidden_dim) for dim in layer_input_dims]
        )
        self.prompt_to_node_msgs = nn.ModuleList(
            [nn.Linear(dim, self.hidden_dim) for dim in layer_input_dims]
        )
        self.prompt_gate_logit = nn.Parameter(
            _prompt_gate_init_tensor(
                num_layers=len(self.backbone.convs),
                default_value=gate_init,
                node_to_prompt_value=(
                    None if node_to_prompt_gate_init is None else float(node_to_prompt_gate_init)
                ),
                prompt_to_node_value=(
                    None if prompt_to_node_gate_init is None else float(prompt_to_node_gate_init)
                ),
            )
        )
        self._last_prompt_aware_aux: dict[str, Any] = {}

    @property
    def prompt_gates(self) -> torch.Tensor:
        return torch.sigmoid(self.prompt_gate_logit)

    def _aggregate_prompt_message(
        self,
        *,
        h: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: torch.Tensor | None,
        edge_type: torch.Tensor,
        type_id: int,
        transform: nn.Linear,
    ) -> torch.Tensor:
        mask = edge_type == int(type_id)
        out = h.new_zeros((h.size(0), self.hidden_dim))
        if not bool(mask.any()):
            return out
        src = edge_index[0, mask]
        dst = edge_index[1, mask]
        msg = transform(h[src])
        msg = F.dropout(msg, p=self.prompt_aware_dropout, training=self.training)
        if edge_weight is None:
            weight = h.new_ones(src.numel())
        else:
            weight = edge_weight[mask].to(dtype=h.dtype, device=h.device)
        weighted_msg = msg * weight.unsqueeze(-1)
        out.index_add_(0, dst, weighted_msg)
        if self.prompt_message_norm == "weighted_sum":
            return out
        denom = h.new_zeros(h.size(0))
        if self.prompt_message_norm == "degree_mean":
            denom.index_add_(0, dst, torch.ones_like(weight))
        else:
            denom.index_add_(0, dst, weight)
        return out / denom.clamp_min(1e-12).unsqueeze(-1)

    def _encode_adapted_prompt_aware(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: torch.Tensor | None,
        edge_type: torch.Tensor | None,
        *,
        original_node_count: int,
        prompt_update_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if edge_type is None:
            self._last_prompt_aware_aux = {
                "prompt_msg_norm": x.new_tensor(0.0),
                "prompt_to_original_msg_norm": x.new_tensor(0.0),
                "node_to_prompt_msg_norm": x.new_tensor(0.0),
                "prompt_to_original_update_norm": x.new_tensor(0.0),
                "node_to_prompt_update_norm": x.new_tensor(0.0),
                "raw_prompt_update_norm": x.new_tensor(0.0),
                "prompt_gate_mean": self.prompt_gates.mean(),
                "prompt_gate_node_to_prompt_mean": self.prompt_gates[:, 0].mean(),
                "prompt_gate_prompt_to_node_mean": self.prompt_gates[:, 1].mean(),
                "prompt_gate_by_layer": self.prompt_gates,
                "adapted_branch_delta_norm": x.new_tensor(0.0),
                "prompt_message_scale": x.new_tensor(float(self.prompt_message_scale)),
                "prompt_message_norm": self.prompt_message_norm,
                "pool_only_prompt_update": x.new_tensor(float(self.pool_only_prompt_update)),
            }
            return self._encode_adapted(x, edge_index, edge_weight)

        if edge_type.numel() != edge_index.size(1):
            raise ValueError("edge_type length must match adapted_edge_index edge count")
        edge_type = edge_type.to(device=edge_index.device, dtype=torch.long)
        original_mask = edge_type == 0
        original_edge_index = edge_index[:, original_mask]
        original_edge_weight = edge_weight[original_mask] if edge_weight is not None else None

        h = x
        clean_h = x
        update_mask: torch.Tensor | None = None
        if self.pool_only_prompt_update and prompt_update_mask is not None:
            original_mask = prompt_update_mask.to(device=x.device, dtype=torch.bool)
            if original_mask.numel() != int(original_node_count):
                raise ValueError("prompt_update_mask length must match original node count")
            update_mask = torch.zeros(x.size(0), dtype=torch.bool, device=x.device)
            update_mask[:original_node_count] = original_mask
            update_mask[original_node_count:] = True
        gates = self.prompt_gates.to(dtype=x.dtype, device=x.device)
        prompt_norms: list[torch.Tensor] = []
        prompt_to_original_norms: list[torch.Tensor] = []
        node_to_prompt_norms: list[torch.Tensor] = []
        prompt_to_original_update_norms: list[torch.Tensor] = []
        node_to_prompt_update_norms: list[torch.Tensor] = []
        raw_prompt_update_norms: list[torch.Tensor] = []
        branch_delta_norms: list[torch.Tensor] = []
        num_layers = len(self.backbone.convs)
        for layer_idx, (conv, adapter) in enumerate(zip(self.backbone.convs, self.adapters)):
            if original_edge_weight is None:
                h_conv = conv(h, original_edge_index)
            else:
                h_conv = conv(h, original_edge_index, original_edge_weight)
            h_base = adapter(h_conv)
            if update_mask is not None:
                if original_edge_weight is None:
                    clean_h_conv = conv(clean_h, original_edge_index)
                else:
                    clean_h_conv = conv(clean_h, original_edge_index, original_edge_weight)
                clean_h_base = adapter(clean_h_conv)
            else:
                clean_h_base = h_base

            node_to_prompt = self._aggregate_prompt_message(
                h=h,
                edge_index=edge_index,
                edge_weight=edge_weight,
                edge_type=edge_type,
                type_id=1,
                transform=self.node_to_prompt_msgs[layer_idx],
            )
            prompt_to_node = self._aggregate_prompt_message(
                h=h,
                edge_index=edge_index,
                edge_weight=edge_weight,
                edge_type=edge_type,
                type_id=2,
                transform=self.prompt_to_node_msgs[layer_idx],
            )
            if not self.use_node_to_prompt:
                node_to_prompt = torch.zeros_like(node_to_prompt)
            if not self.use_prompt_to_node:
                prompt_to_node = torch.zeros_like(prompt_to_node)
            gated_node_to_prompt = gates[layer_idx, 0] * node_to_prompt
            gated_prompt_to_node = gates[layer_idx, 1] * prompt_to_node
            raw_prompt_update = gated_node_to_prompt + gated_prompt_to_node
            prompt_update = float(self.prompt_message_scale) * raw_prompt_update
            prompted_h = h_base + prompt_update
            if update_mask is not None:
                h = torch.where(update_mask.unsqueeze(-1), prompted_h, clean_h_base)
            else:
                h = prompted_h

            prompt_norms.append(_safe_mean_norm(prompt_update))
            prompt_to_original_norms.append(_safe_mean_norm(prompt_to_node[:original_node_count]))
            node_to_prompt_norms.append(_safe_mean_norm(node_to_prompt[original_node_count:]))
            prompt_to_original_update_norms.append(
                _safe_mean_norm((float(self.prompt_message_scale) * gated_prompt_to_node)[:original_node_count])
            )
            node_to_prompt_update_norms.append(
                _safe_mean_norm((float(self.prompt_message_scale) * gated_node_to_prompt)[original_node_count:])
            )
            raw_prompt_update_norms.append(_safe_mean_norm(raw_prompt_update))
            branch_delta_norms.append(_safe_mean_norm(prompt_update[:original_node_count]))

            if layer_idx < num_layers - 1:
                act = getattr(self.backbone, "act", None)
                h = act(h) if act is not None else F.relu(h)
                if update_mask is not None:
                    clean_h = act(clean_h_base) if act is not None else F.relu(clean_h_base)
                dropout = float(getattr(self.backbone, "dropout", 0.0))
                h = F.dropout(h, p=dropout, training=self.training)
                if update_mask is not None:
                    clean_h = F.dropout(clean_h, p=dropout, training=self.training)

        self._last_prompt_aware_aux = {
            "prompt_msg_norm": torch.stack(prompt_norms).mean() if prompt_norms else x.new_tensor(0.0),
            "prompt_to_original_msg_norm": (
                torch.stack(prompt_to_original_norms).mean() if prompt_to_original_norms else x.new_tensor(0.0)
            ),
            "node_to_prompt_msg_norm": (
                torch.stack(node_to_prompt_norms).mean() if node_to_prompt_norms else x.new_tensor(0.0)
            ),
            "prompt_to_original_update_norm": (
                torch.stack(prompt_to_original_update_norms).mean()
                if prompt_to_original_update_norms
                else x.new_tensor(0.0)
            ),
            "node_to_prompt_update_norm": (
                torch.stack(node_to_prompt_update_norms).mean()
                if node_to_prompt_update_norms
                else x.new_tensor(0.0)
            ),
            "raw_prompt_update_norm": (
                torch.stack(raw_prompt_update_norms).mean()
                if raw_prompt_update_norms
                else x.new_tensor(0.0)
            ),
            "prompt_gate_mean": gates.mean(),
            "prompt_gate_node_to_prompt_mean": gates[:, 0].mean(),
            "prompt_gate_prompt_to_node_mean": gates[:, 1].mean(),
            "prompt_gate_by_layer": gates,
            "adapted_branch_delta_norm": (
                torch.stack(branch_delta_norms).mean() if branch_delta_norms else x.new_tensor(0.0)
            ),
            "prompt_message_scale": x.new_tensor(float(self.prompt_message_scale)),
            "prompt_message_norm": self.prompt_message_norm,
            "pool_only_prompt_update": x.new_tensor(float(self.pool_only_prompt_update)),
        }
        return h

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        *,
        adapted_x: torch.Tensor | None = None,
        adapted_edge_index: torch.Tensor | None = None,
        edge_weight: torch.Tensor | None = None,
        adapted_edge_weight: torch.Tensor | None = None,
        adapted_edge_type: torch.Tensor | None = None,
        prompt_update_mask: torch.Tensor | None = None,
        return_aux: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor] | dict[str, Any]:
        if adapted_x is None:
            adapted_x = x
        if adapted_edge_index is None:
            adapted_edge_index = edge_index
        if adapted_edge_weight is None:
            adapted_edge_weight = edge_weight

        h_pre = self._encode_backbone(x, edge_index, edge_weight)
        return self.forward_with_h_pre(
            x,
            edge_index,
            h_pre=h_pre,
            adapted_x=adapted_x,
            adapted_edge_index=adapted_edge_index,
            adapted_edge_weight=adapted_edge_weight,
            adapted_edge_type=adapted_edge_type,
            prompt_update_mask=prompt_update_mask,
            return_aux=return_aux,
        )

    def forward_with_h_pre(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        *,
        h_pre: torch.Tensor,
        adapted_x: torch.Tensor | None = None,
        adapted_edge_index: torch.Tensor | None = None,
        adapted_edge_weight: torch.Tensor | None = None,
        adapted_edge_type: torch.Tensor | None = None,
        prompt_update_mask: torch.Tensor | None = None,
        return_aux: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor] | dict[str, Any]:
        if adapted_x is None:
            adapted_x = x
        if adapted_edge_index is None:
            adapted_edge_index = edge_index

        h_adp_full = self._encode_adapted_prompt_aware(
            adapted_x,
            adapted_edge_index,
            adapted_edge_weight,
            adapted_edge_type,
            original_node_count=h_pre.size(0),
            prompt_update_mask=prompt_update_mask,
        )
        h_adp = h_adp_full[: h_pre.size(0)]

        alpha = self.alpha
        h_mix = alpha * h_pre + (1.0 - alpha) * h_adp
        logits = self.classifier(h_mix)

        if return_aux:
            return {
                "logits": logits,
                "h_pre": h_pre,
                "h_adp": h_adp,
                "h_mix": h_mix,
                "alpha": alpha,
                "h_adp_full": h_adp_full,
                "prompt_aware": self._last_prompt_aware_aux,
            }
        return logits, h_pre, h_adp, h_mix, alpha
