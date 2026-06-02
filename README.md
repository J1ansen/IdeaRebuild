# IdeaRebuild

Clean research codebase for cross-domain few-shot node classification on heterophilic target graphs.

## Current Scope

This repository is being rebuilt around a faithful GP2F baseline before adding new prompt-hub mechanisms.

Implemented phases:

- Phase 0: architecture audit and rebuild planning.

Planned phases:

- Phase 1: faithful GP2F node-classification baseline.
- Phase 2: adapted-branch prompt graph interface.
- Phase 3: virtual prompt hubs for heterophilic target graphs.

## Design Boundary

The first baseline must use only the original target graph `(x, edge_index)`.

Do not add these mechanisms until a faithful GP2F baseline is verified:

- virtual prompt hubs
- Gumbel routing
- null routes
- prompt-induced topology losses

## Reusing Pretrained Models

Single target-dataset experiments should load an existing pretrained checkpoint
instead of launching pretraining again.

The default baseline config points to:

```text
/Users/jackson/MyIdea/pretrained_gnns/lr_0.0005_weightdecay_0.0005_hid_dim_128.pkl
```

The loader in `models/backbones.py` fails fast if the checkpoint is missing and
freezes the loaded GCN by default.
- feature-risk candidate pools
