# Rebuild Plan

## Objective

Build a clean, auditable framework for cross-domain few-shot node classification, starting from a faithful GP2F baseline and later extending only the adapted branch with heterophily-aware virtual prompt hubs.

## Phase 0: Audit First

Phase 0 produces `docs/rebuild_audit.md`.

The audit should compare:

- the current MyIdea implementation
- the official GP2F implementation
- the minimal components required for a faithful baseline

No core model implementation should be added during Phase 0.

## Phase 1: Faithful GP2F Baseline

Target module layout:

- `models/backbones.py`
- `models/faithful_gp2f.py`
- `models/adapters.py`
- `losses/gp2f_losses.py`
- `experiments/run_gp2f_baseline.py`
- `configs/gp2f_baseline.yaml`
- `tests/test_faithful_gp2f.py`

Model requirements:

- pretrained checkpoints are reused through `models.backbones.load_pretrained_gcn`
- single target-dataset runs must not trigger pretraining
- frozen branch uses a pretrained backbone with all backbone parameters frozen
- adapted branch reuses the same frozen encoder layers
- each encoder layer is followed by a trainable residual bottleneck adapter
- fusion uses a learnable scalar `alpha = sigmoid(alpha_logit)`
- classifier consumes the fused representation
- forward output includes `logits`, `h_pre`, `h_adp`, `h_mix`, and `alpha`

Default loss policy:

- classification loss enabled
- original adjacency-driven contrastive loss disabled on heterophilic targets by default
- original topology fusion loss disabled on heterophilic targets by default

## Phase 2: Prompt-Hub Interface Only

Future `forward` signature should reserve:

```python
def forward(
    self,
    x,
    edge_index,
    adapted_x=None,
    adapted_edge_index=None,
    return_aux=False,
):
    ...
```

Default behavior:

```python
adapted_x = x
adapted_edge_index = edge_index
```

The frozen branch should keep using the original graph, while the adapted branch may later receive an augmented graph.
