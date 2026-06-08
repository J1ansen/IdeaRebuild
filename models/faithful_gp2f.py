"""Faithful GP2F baseline model without prompt hubs or routing."""

from __future__ import annotations

from typing import Any

import torch
from torch import nn
import torch.nn.functional as F

from models.adapters import build_adapter


class FaithfulGP2F(nn.Module):
    """Frozen pretrained encoder plus layer-wise residual adapters."""

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
    ) -> None:
        super().__init__()
        if not hasattr(backbone, "convs"):
            raise TypeError("backbone must expose a `convs` ModuleList.")

        self.backbone = backbone
        for parameter in self.backbone.parameters():
            parameter.requires_grad = False
        self.backbone.eval()

        self.hidden_dim = int(hidden_dim)
        self.adapter_style = adapter_style
        self.fusion_alpha_style = fusion_alpha_style
        self.adapters = nn.ModuleList(
            [
                build_adapter(
                    style=adapter_style,
                    hidden_dim=self.hidden_dim,
                    bottleneck_dim=int(adapter_bottleneck_dim),
                    beta_init=float(adapter_beta_init),
                    alpha_init=float(adapter_alpha_init),
                )
                for _ in self.backbone.convs
            ]
        )
        if fusion_alpha_style == "sigmoid":
            alpha = min(max(float(alpha_init), 1e-6), 1.0 - 1e-6)
            self.alpha_logit = nn.Parameter(torch.logit(torch.tensor(alpha)))
            self.raw_alpha = None
        elif fusion_alpha_style == "raw":
            self.alpha_logit = None
            self.raw_alpha = nn.Parameter(torch.tensor(float(alpha_init)))
        else:
            raise ValueError(f"Unsupported fusion alpha style: {fusion_alpha_style}")
        self.classifier = nn.Linear(self.hidden_dim, int(num_classes))

    def train(self, mode: bool = True) -> "FaithfulGP2F":
        super().train(mode)
        self.backbone.eval()
        return self

    @property
    def alpha(self) -> torch.Tensor:
        if self.fusion_alpha_style == "sigmoid":
            if self.alpha_logit is None:
                raise RuntimeError("alpha_logit is missing for sigmoid fusion")
            return torch.sigmoid(self.alpha_logit)
        if self.raw_alpha is None:
            raise RuntimeError("raw_alpha is missing for raw fusion")
        return self.raw_alpha

    def _encode_backbone(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.backbone(x, edge_index, edge_weight)

    def encode_frozen(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Encode with the frozen pretrained branch without changing forward semantics."""

        return self._encode_backbone(x, edge_index, edge_weight)

    def _encode_adapted(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: torch.Tensor | None = None,
    ) -> torch.Tensor:
        h = x
        num_layers = len(self.backbone.convs)
        for layer_idx, (conv, adapter) in enumerate(zip(self.backbone.convs, self.adapters)):
            if edge_weight is None:
                h_base = conv(h, edge_index)
            else:
                h_base = conv(h, edge_index, edge_weight)
            h = adapter(h_base)
            if layer_idx < num_layers - 1:
                act = getattr(self.backbone, "act", None)
                h = act(h) if act is not None else F.relu(h)
                dropout = float(getattr(self.backbone, "dropout", 0.0))
                h = F.dropout(h, p=dropout, training=self.training)
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
        return_aux: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor] | dict[str, Any]:
        """Forward using a precomputed frozen-branch representation.

        This keeps the default ``forward`` behavior unchanged while allowing
        prompt runners to share exactly the same ``h_pre`` tensor between
        prompt evidence and final GP2F fusion.
        """

        if adapted_x is None:
            adapted_x = x
        if adapted_edge_index is None:
            adapted_edge_index = edge_index

        h_adp_full = self._encode_adapted(adapted_x, adapted_edge_index, adapted_edge_weight)
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
            }
        return logits, h_pre, h_adp, h_mix, alpha
