"""P22 class-conditional heterophily pattern enrichment bank."""

from __future__ import annotations

import math
from typing import Any

import torch
from torch import nn
import torch.nn.functional as F

from models.prompt_module import mean_neighbor_summary


def _logit_from_fraction(value: float) -> float:
    value = min(max(float(value), 1e-6), 1.0 - 1e-6)
    return float(torch.logit(torch.tensor(value)).item())


def _minmax(values: torch.Tensor) -> torch.Tensor:
    if values.numel() == 0:
        return values
    lo = values.min()
    hi = values.max()
    return (values - lo) / (hi - lo).clamp_min(1e-12)


@torch.no_grad()
def estimate_class_transition_from_support(
    *,
    labels: torch.Tensor,
    support_mask: torch.Tensor,
    edge_index: torch.Tensor,
    num_classes: int,
    smoothing: float = 0.5,
) -> torch.Tensor:
    """Estimate T[c, d] = P(neighbor class=d | center class=c) from support edges."""

    device = labels.device
    transition = torch.full(
        (int(num_classes), int(num_classes)),
        float(smoothing),
        device=device,
        dtype=torch.float32,
    )
    if edge_index.numel() > 0:
        src, dst = edge_index.to(device)
        support = support_mask.to(device=device, dtype=torch.bool)
        edge_mask = support[src] & support[dst]
        src_y = labels[src[edge_mask]].long()
        dst_y = labels[dst[edge_mask]].long()
        valid = (src_y >= 0) & (src_y < int(num_classes)) & (dst_y >= 0) & (dst_y < int(num_classes))
        src_y = src_y[valid]
        dst_y = dst_y[valid]
        if src_y.numel() > 0:
            flat_idx = src_y * int(num_classes) + dst_y
            counts = torch.bincount(flat_idx, minlength=int(num_classes) * int(num_classes)).float()
            transition = transition + counts.view(int(num_classes), int(num_classes))
    return transition / transition.sum(dim=-1, keepdim=True).clamp_min(1e-12)


class P22ClassPatternEnrichmentBank(nn.Module):
    """Basis-supervised logit-evidence adapter for P22.

    The module leaves ``edge_index`` and hidden states unchanged. It builds a
    detached structural-predictive signature from NoPrompt logits and hidden
    relations, learns shared pattern tokens, then lets patterns choose explicit
    heterophily evidence bases before emitting a bounded class-level logit bias.
    """

    consumes_base_logits = True
    emits_logits = True

    def __init__(self, source_dim: int, hidden_dim: int, config: dict[str, Any] | None = None) -> None:
        super().__init__()
        self.source_dim = int(source_dim)
        self.hidden_dim = int(hidden_dim)
        self.config = dict(config or {})
        self.num_classes = int(self.config.get("num_classes", 0))
        if self.num_classes <= 0:
            raise ValueError("P22 requires prompt_adapter.num_classes to be positive")
        self.num_patterns = int(self.config.get("num_patterns", 6))
        if self.num_patterns <= 0:
            raise ValueError("P22 num_patterns must be positive")
        self.pattern_dim = int(self.config.get("pattern_dim", 32))
        self.pattern_init = str(self.config.get("pattern_init", "kmeans_all_signature"))
        self.pattern_init_use_labels = bool(self.config.get("pattern_init_use_labels", False))
        if self.pattern_init_use_labels:
            raise ValueError("P22 pattern_init_use_labels must remain false")
        self.class_pattern_smoothing = float(self.config.get("class_pattern_smoothing", 0.5))
        self.detach_class_pattern = bool(self.config.get("detach_class_pattern", True))
        self.eps = float(self.config.get("eps", 1e-8))
        self.use_basis_evidence = bool(self.config.get("use_basis_evidence", True))
        self.basis_types = list(
            self.config.get(
                "basis_types",
                [
                    "ego_logprob",
                    "onehop_logprob",
                    "twohop_logprob",
                    "highpass_ego_onehop",
                    "highpass_onehop_twohop",
                    "class_transition",
                ],
            )
        )
        self.num_bases = int(self.config.get("num_bases", len(self.basis_types)))
        if self.num_bases != len(self.basis_types):
            raise ValueError("P22 num_bases must match len(basis_types)")
        self.enrichment_weight = float(self.config.get("enrichment_weight", 0.5))
        self.basis_weight_scale = float(self.config.get("basis_weight_scale", 1.0))
        self.use_class_transition = bool(self.config.get("use_class_transition", True))
        self.class_transition_smoothing = float(self.config.get("class_transition_smoothing", 0.5))
        self.basis_usage_entropy_floor = float(self.config.get("basis_usage_entropy_floor", 0.60))
        self.pattern_scale_max = float(self.config.get("pattern_scale_max", 1.0))
        pattern_scale_init = float(self.config.get("pattern_scale_init", 0.10))
        self.raw_pattern_scale = nn.Parameter(
            torch.tensor(_logit_from_fraction(pattern_scale_init / max(self.pattern_scale_max, 1e-12)))
        )
        self.pattern_scale_warmup_epochs = int(self.config.get("pattern_scale_warmup_epochs", 20))
        self.register_buffer("current_epoch", torch.tensor(0.0), persistent=False)
        self.temperature_init = float(self.config.get("pattern_temperature_init", 0.70))
        self.temperature_final = float(self.config.get("pattern_temperature_final", 0.25))
        self.temperature_warmdown_epochs = int(self.config.get("pattern_temperature_warmdown_epochs", 50))
        self.normalize_evidence = bool(self.config.get("normalize_pattern_evidence", True))
        self.evidence_std_floor = float(self.config.get("pattern_evidence_std_floor", 0.5))
        self.use_pattern_gate = bool(self.config.get("use_pattern_gate", False))
        if self.use_pattern_gate:
            raise ValueError("P22 keeps use_pattern_gate disabled in the basis-supervised version")

        signature_dim = 2 * self.num_classes + 8
        dropout = float(self.config.get("dropout", 0.0))
        encoder_hidden = int(self.config.get("pattern_encoder_hidden_dim", max(32, self.pattern_dim)))
        self.signature_norm = nn.LayerNorm(signature_dim)
        self.pattern_encoder = nn.Sequential(
            nn.Linear(signature_dim, encoder_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(encoder_hidden, self.pattern_dim),
        )
        self.pattern_tokens = nn.Parameter(torch.randn(self.num_patterns, self.pattern_dim) * 0.02)
        self.pattern_basis_logits = nn.Parameter(torch.zeros(self.num_patterns, self.num_bases))
        self.register_buffer("pattern_tokens_initialized", torch.tensor(False), persistent=True)

    def set_epoch(self, epoch: int | float) -> None:
        self.current_epoch.fill_(float(epoch))

    def _temperature(self) -> torch.Tensor:
        if self.temperature_warmdown_epochs <= 0:
            value = self.temperature_final
        else:
            t = min(1.0, max(0.0, float(self.current_epoch.item()) / float(self.temperature_warmdown_epochs)))
            value = self.temperature_init + t * (self.temperature_final - self.temperature_init)
        return self.pattern_tokens.new_tensor(max(value, 1e-4))

    def _scale(self) -> torch.Tensor:
        if self.pattern_scale_warmup_epochs <= 0:
            warmup = 1.0
        else:
            warmup = min(1.0, max(0.0, (float(self.current_epoch.item()) + 1.0) / float(self.pattern_scale_warmup_epochs)))
        return self.pattern_scale_max * torch.sigmoid(self.raw_pattern_scale) * self.raw_pattern_scale.new_tensor(warmup)

    def _degree_norm(self, edge_index: torch.Tensor, num_nodes: int, ref: torch.Tensor) -> torch.Tensor:
        degree = ref.new_zeros(num_nodes)
        if edge_index.numel() > 0:
            degree.index_add_(0, edge_index[1].to(ref.device), ref.new_ones(edge_index.size(1)))
        return _minmax(torch.log1p(degree))

    def _signature(
        self,
        *,
        h_adp: torch.Tensor,
        edge_index: torch.Tensor,
        base_logits: torch.Tensor,
    ) -> torch.Tensor:
        h = h_adp.detach()
        logits = base_logits.detach().to(device=h.device, dtype=h.dtype)
        prob = F.softmax(logits, dim=-1)
        neigh_prob = mean_neighbor_summary(prob, edge_index, num_nodes=int(prob.size(0)))
        two_prob = mean_neighbor_summary(neigh_prob, edge_index, num_nodes=int(prob.size(0)))
        low = mean_neighbor_summary(h, edge_index, num_nodes=int(h.size(0)))
        two = mean_neighbor_summary(low, edge_index, num_nodes=int(h.size(0)))

        prob_diff = (prob - neigh_prob).abs()
        neigh_two_diff = (neigh_prob - two_prob).abs()
        entropy = -(prob * prob.clamp_min(1e-12).log()).sum(dim=-1)
        neigh_entropy = -(neigh_prob * neigh_prob.clamp_min(1e-12).log()).sum(dim=-1)
        if prob.size(-1) > 1:
            norm = math.log(float(prob.size(-1)))
            entropy = entropy / norm
            neigh_entropy = neigh_entropy / norm
        top2 = torch.topk(prob, k=min(2, prob.size(-1)), dim=-1).values
        margin = top2[:, 0] if top2.size(-1) == 1 else top2[:, 0] - top2[:, 1]
        degree_norm = self._degree_norm(edge_index, int(h.size(0)), h)
        ego_low_norm = _minmax((h - low).norm(dim=-1))
        low_two_norm = _minmax((low - two).norm(dim=-1))
        cos_ego_low = F.cosine_similarity(h, low, dim=-1, eps=1e-12)
        cos_ego_two = F.cosine_similarity(h, two, dim=-1, eps=1e-12)
        signature = torch.cat(
            [
                prob_diff,
                neigh_two_diff,
                entropy.unsqueeze(-1),
                margin.unsqueeze(-1),
                neigh_entropy.unsqueeze(-1),
                degree_norm.unsqueeze(-1),
                ego_low_norm.unsqueeze(-1),
                low_two_norm.unsqueeze(-1),
                cos_ego_low.unsqueeze(-1),
                cos_ego_two.unsqueeze(-1),
            ],
            dim=-1,
        )
        return self.signature_norm(signature)

    @torch.no_grad()
    def _kmeans_centers(self, x: torch.Tensor) -> torch.Tensor:
        n = int(x.size(0))
        k = self.num_patterns
        if n == 0:
            return x.new_zeros(k, x.size(-1))
        if n < k:
            reps = (k + n - 1) // n
            return x.repeat(reps, 1)[:k]
        # Deterministic farthest-ish seed spread over signature norm order.
        order = torch.argsort(x.norm(dim=-1))
        seed_pos = torch.linspace(0, n - 1, steps=k, device=x.device).round().long()
        centers = x[order[seed_pos]].clone()
        iters = int(self.config.get("kmeans_iters", 15))
        for _ in range(max(1, iters)):
            dist = torch.cdist(x, centers)
            assign = dist.argmin(dim=-1)
            new_centers = centers.clone()
            for idx in range(k):
                mask = assign == idx
                if bool(mask.any()):
                    new_centers[idx] = x[mask].mean(dim=0)
            if torch.allclose(new_centers, centers, atol=1e-5, rtol=1e-4):
                centers = new_centers
                break
            centers = new_centers
        return centers

    @torch.no_grad()
    def _maybe_initialize_tokens(self, signature: torch.Tensor) -> None:
        if bool(self.pattern_tokens_initialized.item()):
            return
        if self.pattern_init == "random":
            self.pattern_tokens_initialized.fill_(True)
            return
        if self.pattern_init != "kmeans_all_signature":
            raise ValueError(f"Unsupported P22 pattern_init={self.pattern_init!r}")
        was_training = self.pattern_encoder.training
        self.pattern_encoder.eval()
        encoded_all = F.normalize(self.pattern_encoder(signature.detach()), dim=-1, eps=1e-12)
        centers = self._kmeans_centers(encoded_all)
        self.pattern_tokens.copy_(F.normalize(centers, dim=-1, eps=1e-12))
        if was_training:
            self.pattern_encoder.train()
        self.pattern_tokens_initialized.fill_(True)

    def _class_pattern_enrichment(
        self,
        *,
        pattern_weights: torch.Tensor,
        support_mask: torch.Tensor | None,
        labels: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        n, k = pattern_weights.shape
        if support_mask is None:
            support_mask = torch.ones(n, dtype=torch.bool, device=pattern_weights.device)
        support_mask = support_mask.to(device=pattern_weights.device, dtype=torch.bool)
        if labels is None:
            raise ValueError("P22 requires labels to build class-pattern enrichment")
        labels = labels.to(device=pattern_weights.device, dtype=torch.long)
        if not bool(support_mask.any()):
            support_mask = torch.ones(n, dtype=torch.bool, device=pattern_weights.device)
        global_pattern = pattern_weights[support_mask].mean(dim=0).detach().clamp_min(self.eps)
        global_pattern = global_pattern / global_pattern.sum().clamp_min(self.eps)
        class_pattern = pattern_weights.new_zeros(self.num_classes, k)
        class_count = pattern_weights.new_zeros(self.num_classes, 1)
        alpha = pattern_weights.new_tensor(max(0.0, self.class_pattern_smoothing))
        for class_id in range(self.num_classes):
            class_mask = support_mask & (labels == class_id)
            count = class_mask.to(dtype=pattern_weights.dtype).sum()
            class_count[class_id, 0] = count
            numerator = pattern_weights[class_mask].sum(dim=0) if bool(class_mask.any()) else pattern_weights.new_zeros(k)
            class_pattern[class_id] = (numerator + alpha * global_pattern) / (count + alpha).clamp_min(self.eps)
        enrichment = (class_pattern.clamp_min(self.eps).log() - global_pattern.clamp_min(self.eps).log().unsqueeze(0))
        return enrichment, class_pattern, global_pattern

    def _basis_evidence(
        self,
        *,
        base_logits: torch.Tensor,
        edge_index: torch.Tensor,
        support_mask: torch.Tensor | None,
        labels: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        logits = base_logits.detach()
        num_nodes = int(logits.size(0))
        prob = F.softmax(logits, dim=-1)
        neigh_prob = mean_neighbor_summary(prob, edge_index, num_nodes=num_nodes)
        two_prob = mean_neighbor_summary(neigh_prob, edge_index, num_nodes=num_nodes)
        if support_mask is None:
            support = torch.ones(num_nodes, dtype=torch.bool, device=logits.device)
        else:
            support = support_mask.to(device=logits.device, dtype=torch.bool)
        if labels is None:
            labels_t = torch.zeros(num_nodes, dtype=torch.long, device=logits.device)
        else:
            labels_t = labels.to(device=logits.device, dtype=torch.long)
        transition = estimate_class_transition_from_support(
            labels=labels_t,
            support_mask=support,
            edge_index=edge_index,
            num_classes=self.num_classes,
            smoothing=self.class_transition_smoothing,
        ).to(device=logits.device, dtype=logits.dtype)

        values: list[torch.Tensor] = []
        for basis_type in self.basis_types:
            if basis_type == "ego_logprob":
                values.append(prob.clamp_min(self.eps).log())
            elif basis_type == "onehop_logprob":
                values.append(neigh_prob.clamp_min(self.eps).log())
            elif basis_type == "twohop_logprob":
                values.append(two_prob.clamp_min(self.eps).log())
            elif basis_type == "highpass_ego_onehop":
                values.append(prob - neigh_prob)
            elif basis_type == "highpass_onehop_twohop":
                values.append(neigh_prob - two_prob)
            elif basis_type == "class_transition":
                if self.use_class_transition:
                    transition_evidence = neigh_prob @ transition.t()
                    values.append(transition_evidence.clamp_min(self.eps).log())
                else:
                    values.append(logits.new_zeros(num_nodes, self.num_classes))
            else:
                raise ValueError(f"Unsupported P22 basis type: {basis_type!r}")
        return torch.stack(values, dim=1), transition

    def forward(
        self,
        *,
        z: torch.Tensor | None = None,
        edge_index: torch.Tensor,
        h_adp: torch.Tensor,
        update_mask: torch.Tensor | None = None,
        base_logits: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
        support_mask: torch.Tensor | None = None,
        **_: Any,
    ) -> dict[str, torch.Tensor]:
        if base_logits is None:
            raise ValueError("P22 requires detached NoPrompt base_logits")
        signature = self._signature(h_adp=h_adp, edge_index=edge_index, base_logits=base_logits)
        self._maybe_initialize_tokens(signature)
        encoded = F.normalize(self.pattern_encoder(signature), dim=-1)
        tokens = F.normalize(self.pattern_tokens, dim=-1)
        pattern_logits = encoded @ tokens.t() / self._temperature()
        pattern_weights = F.softmax(pattern_logits, dim=-1)
        pattern_weights_for_stats = pattern_weights.detach() if self.detach_class_pattern else pattern_weights
        enrichment, class_pattern, global_pattern = self._class_pattern_enrichment(
            pattern_weights=pattern_weights_for_stats,
            support_mask=support_mask,
            labels=labels,
        )
        enrichment_evidence = pattern_weights @ enrichment.t()
        if self.use_basis_evidence:
            basis_evidence, transition_matrix = self._basis_evidence(
                base_logits=base_logits,
                edge_index=edge_index,
                support_mask=support_mask,
                labels=labels,
            )
            pattern_basis_weight = F.softmax(self.pattern_basis_logits, dim=-1)
            basis_pattern_evidence = torch.einsum("nk,kb,nbc->nc", pattern_weights, pattern_basis_weight, basis_evidence)
            student_basis = pattern_weights @ pattern_basis_weight
            basis_usage = pattern_weights.mean(dim=0) @ pattern_basis_weight
            basis_usage = basis_usage / basis_usage.sum().clamp_min(self.eps)
            basis_usage_entropy = -(basis_usage * basis_usage.clamp_min(1e-12).log()).sum()
            if self.num_bases > 1:
                basis_usage_entropy = basis_usage_entropy / math.log(float(self.num_bases))
            basis_usage_loss = F.relu(self.basis_usage_entropy_floor - basis_usage_entropy).pow(2)
        else:
            basis_evidence = h_adp.new_zeros(h_adp.size(0), self.num_bases, self.num_classes)
            transition_matrix = h_adp.new_zeros(self.num_classes, self.num_classes)
            pattern_basis_weight = F.softmax(self.pattern_basis_logits, dim=-1)
            basis_pattern_evidence = h_adp.new_zeros(h_adp.size(0), self.num_classes)
            student_basis = pattern_weights @ pattern_basis_weight
            basis_usage = student_basis.mean(dim=0)
            basis_usage = basis_usage / basis_usage.sum().clamp_min(self.eps)
            basis_usage_entropy = h_adp.new_tensor(0.0)
            basis_usage_loss = h_adp.new_tensor(0.0)

        pattern_evidence = self.enrichment_weight * enrichment_evidence + self.basis_weight_scale * basis_pattern_evidence
        pattern_evidence = pattern_evidence - pattern_evidence.mean(dim=-1, keepdim=True)
        if self.normalize_evidence:
            std = pattern_evidence.std(dim=-1, keepdim=True).detach().clamp_min(self.evidence_std_floor)
            pattern_evidence = pattern_evidence / std
        pattern_scale = self._scale()
        logit_bias = pattern_scale * pattern_evidence
        if update_mask is not None:
            # The mask only gates deployment of evidence; support statistics stay
            # controlled by support_mask.
            mask = update_mask.to(device=h_adp.device, dtype=h_adp.dtype).unsqueeze(-1)
            logit_bias = logit_bias * mask
        logits = base_logits.detach().to(device=h_adp.device, dtype=h_adp.dtype) + logit_bias

        zero = h_adp.new_zeros(h_adp.shape)
        pattern_usage_mean = pattern_weights.mean(dim=0)
        pattern_usage_entropy = -(pattern_usage_mean * pattern_usage_mean.clamp_min(1e-12).log()).sum()
        if self.num_patterns > 1:
            pattern_usage_entropy = pattern_usage_entropy / math.log(float(self.num_patterns))
        update_mask_eff = (
            torch.ones(h_adp.size(0), dtype=torch.bool, device=h_adp.device)
            if update_mask is None
            else update_mask.to(device=h_adp.device, dtype=torch.bool)
        )
        pattern_reg = F.relu(pattern_usage_mean.max() - float(self.config.get("pattern_usage_max", 0.80)))
        pattern_reg = pattern_reg + F.relu(float(self.config.get("pattern_usage_entropy_floor", 0.35)) - pattern_usage_entropy)
        class_pattern_kl = (
            class_pattern.clamp_min(self.eps)
            * (class_pattern.clamp_min(self.eps).log() - global_pattern.clamp_min(self.eps).log().unsqueeze(0))
        ).sum(dim=-1).mean()
        topk = min(3, self.num_patterns)
        class_pattern_topk_value, class_pattern_topk_index = torch.topk(class_pattern.detach(), k=topk, dim=-1)
        return {
            "h_adp": h_adp,
            "logits": logits,
            "pattern_evidence": pattern_evidence,
            "enrichment_evidence": enrichment_evidence,
            "basis_pattern_evidence": basis_pattern_evidence,
            "basis_evidence": basis_evidence,
            "logit_bias": logit_bias,
            "pattern_scale": pattern_scale,
            "pattern_weights": pattern_weights,
            "pattern_logits": pattern_logits,
            "pattern_basis_weight": pattern_basis_weight,
            "student_basis": student_basis,
            "basis_usage": basis_usage,
            "basis_usage_entropy": basis_usage_entropy,
            "basis_usage_loss": basis_usage_loss,
            "transition_matrix": transition_matrix,
            "class_pattern_enrichment": enrichment,
            "class_pattern": class_pattern,
            "class_pattern_kl_to_global": class_pattern_kl,
            "class_pattern_topk_value": class_pattern_topk_value,
            "class_pattern_topk_index": class_pattern_topk_index,
            "global_pattern": global_pattern,
            "pattern_usage_mean": pattern_usage_mean,
            "pattern_usage_entropy": pattern_usage_entropy,
            "pattern_max_prob_mean": pattern_weights.max(dim=-1).values.mean(),
            "pattern_reg": pattern_reg,
            "signature": signature.detach(),
            "delta": zero,
            "raw_delta": zero,
            "update": zero,
            "gate": update_mask_eff.to(dtype=h_adp.dtype),
            "raw_gate": update_mask_eff.to(dtype=h_adp.dtype),
            "prompt_update_norm": h_adp.new_tensor(0.0),
            "prompt_update_max_norm": h_adp.new_tensor(0.0),
            "prompt_raw_delta_norm": h_adp.new_tensor(0.0),
            "prompt_delta_norm": h_adp.new_tensor(0.0),
            "prompt_gate_mean": update_mask_eff.to(dtype=h_adp.dtype).mean(),
            "prompt_gate_min": update_mask_eff.to(dtype=h_adp.dtype).min(),
            "prompt_gate_max": update_mask_eff.to(dtype=h_adp.dtype).max(),
            "prompt_raw_gate_mean": update_mask_eff.to(dtype=h_adp.dtype).mean(),
            "prompt_raw_gate_min": update_mask_eff.to(dtype=h_adp.dtype).min(),
            "prompt_raw_gate_max": update_mask_eff.to(dtype=h_adp.dtype).max(),
            "prompt_update_mask_ratio": update_mask_eff.to(dtype=h_adp.dtype).mean(),
            "prompt_update_clip_ratio": h_adp.new_tensor(0.0),
            "high_frequency_norm": h_adp.new_tensor(0.0),
            "low_frequency_norm": h_adp.new_tensor(0.0),
        }
