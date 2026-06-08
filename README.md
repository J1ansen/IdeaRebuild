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

## Reusing Existing Datasets

The default config points to the existing PyG cache:

```text
/Users/jackson/MyIdea/data
```

Dataset loaders should use this as their root and should fail if a required
processed dataset is missing, instead of silently downloading during baseline
experiments.

## Minimal Baseline Run

```bash
source .venv/bin/activate
python -m experiments.run_gp2f_baseline \
  --config configs/gp2f_baseline.yaml \
  --target_dataset Cora \
  --epochs 3
```

For repeated 5-shot splits, use `--runs`. Seeds are generated as
`seed, seed + 1, ...` unless explicit seeds are set in the config.

```bash
python -m experiments.run_gp2f_baseline \
  --config configs/gp2f_baseline.yaml \
  --target_dataset PubMed \
  --runs 5
```

The command writes metrics, loss curves, and a config snapshot under:

```text
outputs/gp2f_baseline/<dataset>/<timestamp>/
```

## Formal Experiment Runs

Run Cora-pretrained GRACE checkpoint on PubMed with 5 seeds:

```bash
python -m experiments.run_gp2f_baseline \
  --config configs/gp2f_baseline.yaml \
  --target_dataset PubMed \
  --preset B0 \
  --seeds 0,1,2,3,4
```

The built-in presets isolate each GP2F loss component:

| Preset | Losses |
|---|---|
| `B0` | `CE` |
| `B1` | `CE + L_ctr` |
| `B2` | `CE + L_fus` |
| `B3` | `CE + L_ctr + L_fus` |

Compare the full original GP2F structure-loss setting:

```bash
python -m experiments.run_gp2f_baseline \
  --config configs/gp2f_baseline.yaml \
  --target_dataset PubMed \
  --preset B3 \
  --seeds 0,1,2,3,4
```

Useful overrides:

```bash
--epochs 200
--lr 0.001
--weight_decay 0.0005
--patience 40
--output_dir outputs/gp2f_baseline
```

Each run group writes per-seed metrics/checkpoints plus `summary.json` and
`summary.csv`, including `mean+-std` fields for test accuracy and macro-F1.
The summaries also record whether topology loss used dense faithful consistency
or the sampled approximation path.

## Official-Style GP2F Compatibility

The stable default is `StableDualBranch`: bounded fusion and conservative
zero-init adapters. It should not be reported as an exact official GP2F
reproduction. To run closer to the official GP2F implementation, merge the
compatibility style config:

```bash
python -m experiments.run_gp2f_baseline \
  --config configs/gp2f_baseline.yaml \
  --style_config configs/gp2f_official_style.yaml \
  --target_dataset Cora \
  --preset B1 \
  --seeds 0,1,2,3,4
```

This switches to:

- `Linear + PReLU` feature projector
- official GP2F adapter initialization
- raw learnable fusion alpha
- train-loss early stopping

The default config remains the stable variant for heterophily-oriented
experiments.

The runner uses a live `tqdm` progress bar with ETA and prints final
`mean+-std` summaries for test accuracy and macro-F1.
- feature-risk candidate pools
