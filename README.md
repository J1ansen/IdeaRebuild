# IdeaRebuild

Research codebase for cross-domain few-shot node classification with a
Cora-pretrained GP2F-style backbone and heterophily-oriented prompt adaptation.

This repository is a staged rebuild. The current code contains a faithful
dual-branch GP2F baseline, residual prompt experiments, explicit prompt graph
construction, and a prompt-aware adapted branch. The current prompt module is
still diagnostic: it improves train-node routing consistency, but has not yet
shown stable test-set gains on heterophilic targets.

## Abstract

GP2F performs well in cross-domain few-shot node classification on homophilic
graphs, but its adjacency-driven structure losses and graph propagation can be
fragile on heterophilic target graphs. This project keeps the stable part of
GP2F, namely the frozen pretrained branch and the adapted downstream branch,
and studies whether prompt nodes and prompt-specific message passing can provide
controlled auxiliary information for the adapted branch.

The implemented method does not modify the original graph, does not use
pseudo-label graph construction, and does not claim to improve original graph
homophily. Prompt graph edges are used only by the adapted branch. Validation is
allowed to select `message_scale=0.0`, so the model can fall back to a no-prompt
behavior when prompt messages are harmful.

## Method

### Dual-branch GP2F Scaffold

The pretrained Cora GRACE checkpoint is loaded once and reused. The frozen
branch always uses the original target graph:

```text
h_pre = frozen_backbone(z, original_edge_index)
```

The adapted branch uses either the original graph or a prompt-augmented graph:

```text
h_adp = adapted_branch(adapted_x, adapted_edge_index, adapted_edge_weight)
```

The final representation is fused by a learnable bounded fusion weight:

```text
h_mix = alpha * h_pre + (1 - alpha) * h_adp
logits = classifier(h_mix)
```

For heterophily-oriented experiments, GP2F's original adjacency-driven
contrastive and topology-fusion losses are disabled by default. The primary
comparison is CE-only NoPrompt vs prompt variants.

### P0: Residual Feature Prompt

P0 is implemented as a residual feature prompt:

```text
adapted_x = z + gamma * (g_sem * u_sem + g_struct * u_struct)
```

It does not create prompt nodes or prompt edges. It is best interpreted as a
structure-context-conditioned residual feature prompt, not as topology
reconstruction.

### P1/P2/P3: Prompt Graph Adaptation

The current explicit prompt graph path appends learnable prompt nodes after all
original nodes:

```text
adapted_x = [original node features; prompt node features]
```

Original nodes remain the first `N` rows, and final classification only uses
the first `N` outputs.

Prompt graph construction follows these rules:

- frozen branch uses only `original_edge_index`;
- prompt edges enter only the adapted branch;
- original graph edges are never deleted, rewired, or relabeled;
- pool nodes are `train_mask` plus top-`rho` structurally unreliable nodes;
- no validation/test labels and no pseudo-labels are used for pool selection;
- every selected pool node connects to at most `topk_prompt_per_node` prompt
  nodes;
- prompt edges are bidirectional: node-to-prompt and prompt-to-node.

The structural pool score is label-free and uses one-step and two-step diffusion
summaries:

```text
m1_i = mean_neighbor_summary(base_i)
m2_i = mean_neighbor_summary(m1_i)
score_i = mean(1 - cos(base_i, m1_i),
               1 - cos(m1_i, m2_i),
               normalized_neighbor_variance_i)
```

### Multi-view Router

The P2/P3 router computes prompt assignment from three views:

- semantic view: node base feature evidence;
- structural view: `[base, m1, m2, base - m1, m1 - m2]`;
- role view: degree and structural-difference statistics.

The view gate combines these logits before top-k prompt selection. This is a
router over prompt slots, not a class prediction module.

### Train-only Class-aware Slots

The first `num_classes` prompt slots are treated as class-aware routing anchors.
Additional residual slots absorb patterns not clearly aligned with class-aware
slots. For a train pool node with label `c`, `L_class_route` encourages routing
probability to the `c`-th class slot.

This supervision is applied only on:

```text
train_mask & pool_mask
```

Validation and test labels are only used for reporting.

P3 also initializes class prompt keys from train-only structural-query
prototypes:

```text
mu_c = mean(query_i), where i in train_mask & pool_mask and y_i = c
```

If a class has no train node in the pool, its prompt key remains randomly
initialized and the missing class is recorded.

### Prompt-aware Adapted Branch

P3 uses `PromptAwareGP2F` to separate original graph messages and prompt
messages. Original edges are processed by the adapted GNN path. Prompt edges are
processed by a prompt-specific receiver.

For `receiver_version=v2_conditioned`, prompt messages use node-conditioned
features:

```text
msg_{p->i} = MLP([h_i, h_p, h_i - h_p, h_i * h_p])
```

The update is residual and gated:

```text
h = h_graph + message_scale * beta * msg_prompt
```

The current default uses:

- `zero_init_prompt_messages=true`;
- `prompt_message_norm=layernorm`;
- `pool_only_prompt_update=true`;
- validation-selected `message_scale` from `[0.0, 0.1, 0.25, 0.5, 1.0]`.

Zero initialization keeps the initial behavior close to NoPrompt.

## Training Objective

The main task loss is cross entropy on train nodes:

```text
L_CE = CE(logits[train_mask], y[train_mask])
```

The current P3 config may add these train-only auxiliary losses:

```text
L = L_CE
  + lambda_class_route * L_class_route
  + lambda_key_proto * L_key_proto
  + lambda_prompt_role_diversity * L_role_diversity
  + lambda_prompt_usage_consistency * L_usage_consistency
  + lambda_prompt_acceptance_supervision * L_acceptance_supervision
```

Default P3 values:

```text
lambda_class_route = 0.05
lambda_key_proto = 0.001
lambda_prompt_role_diversity = 0.01
lambda_prompt_usage_consistency = 0.02
lambda_prompt_acceptance_supervision = 0.20
lambda_prompt_balance = 0.005
lambda_edge_l1 = 0.0
```

The original GP2F adjacency-driven contrastive/topology losses are disabled in
the prompt configs by default.

## Current Experimental Findings

Recent 5% shot heterophilic quick checks used seeds `0,1,2` and 80 epochs.

| Dataset | Variant | Best Test Acc | Best Test Macro-F1 | Observation |
|---|---|---:|---:|---|
| Actor | P2.5 | 26.25+-0.33 | 17.18+-7.30 | class-aware routing baseline |
| Actor | P3 | 26.25+-0.33 | 17.18+-7.30 | no improvement |
| chameleon | P2.5 | 33.95+-3.01 | 31.29+-1.89 | class-aware routing baseline |
| chameleon | P3 | 33.95+-3.01 | 31.29+-1.89 | no improvement |

Important diagnostics:

- train-node class routing can be high, especially on chameleon;
- class-key prototype initialization coverage is usually complete in these
  5% shot runs;
- validation often selects `message_scale=0.0`;
- when `message_scale=0.0`, prompt message norm and prompt update norm are zero;
- positive prompt message scales do not yet produce stable validation/test gains.

Therefore, the current prompt graph and prompt-aware receiver should not be
claimed as an effective final method. The current evidence supports a narrower
claim: train-only class-aware routing is implementable and interpretable, but
the prompt message itself is not yet reliably useful.

## Current Problems

| Problem | Expected Behavior | Current Behavior | Planned Fix |
|---|---|---|---|
| Prompt message usefulness | Prompt messages reduce CE and improve validation/test metrics | Validation often chooses `message_scale=0.0` | Directly diagnose and optimize `CE_no_prompt - CE_prompt` on train pool nodes |
| Prompt-aware receiver | Separating prompt/original messages improves adapted branch | P3 matches P2.5 but does not improve it | Redesign message generation as residual correction rather than only edge message passing |
| Router generalization | Train-only routing anchors generalize to val/test nodes | Train hit rate is high, val/test hit rate is much lower | Add label-free consistency or prototype smoothing; do not use pseudo-label supervision yet |
| Rejection gate | Learn fine-grained accept/reject decisions | Model often globally closes prompt through `message_scale=0.0` | Reframe rejection as a prompt-benefit predictor supervised by train-only CE delta |
| Pool/scale tuning | More prompt access improves hard nodes | Larger or positive scale can amplify noise | Do not expand pool first; prove prompt messages are useful inside the current pool |

## Reproducible Commands

Create and activate the environment, then run from the repository root.

### GP2F Baseline

```bash
python -m experiments.run_gp2f_baseline \
  --config configs/gp2f_baseline.yaml \
  --target_dataset PubMed \
  --preset B0 \
  --seeds 0,1,2,3,4
```

The GP2F baseline presets are:

| Preset | Losses |
|---|---|
| `B0` | `CE` |
| `B1` | `CE + L_ctr` |
| `B2` | `CE + L_fus` |
| `B3` | `CE + L_ctr + L_fus` |

### P3 Prompt Graph Quick Check

```bash
.venv/bin/python -m experiments.run_gp2f_prompt_graph \
  --config configs/gp2f_prompt_p3.yaml \
  --target_dataset Actor \
  --prompt_variant p2_multiview \
  --shot_mode percent \
  --shot_value 0.05 \
  --epochs 80 \
  --seeds 0,1,2
```

```bash
.venv/bin/python -m experiments.run_gp2f_prompt_graph \
  --config configs/gp2f_prompt_p3.yaml \
  --target_dataset chameleon \
  --prompt_variant p2_multiview \
  --shot_mode percent \
  --shot_value 0.05 \
  --epochs 80 \
  --seeds 0,1,2
```

Outputs are written under:

```text
outputs/<experiment_name>/<dataset>/<timestamp>/
```

Each run stores per-seed metrics, curves, config snapshots, checkpoints,
`summary.json`, and `summary.csv`.

## Data and Checkpoints

The default dataset cache is:

```text
/Users/jackson/MyIdea/data
```

The default pretrained checkpoint is:

```text
pretrained_gnns/lr_0.0005_weightdecay_0.0005_hid_dim_128.pkl
```

The code is intended to reuse existing processed datasets and pretrained GNNs.
It should fail fast if required data/checkpoints are missing rather than
silently downloading or pretraining during a single experiment run.

## Design Boundaries

The current project does not claim or implement:

- original graph homophily improvement;
- original graph topology reconstruction;
- pseudo-label-based prompt graph construction;
- validation/test label usage for pool, routing, prototype, or loss;
- DFS/random-walk prompt routing;
- full class compatibility matrix modeling;
- a final proven prompt method.

The current safe description is:

```text
GP2F dual-branch scaffold
+ train-only class-aware prompt routing
+ prompt-augmented adapted graph
+ prompt-aware receiver
```

The current open bottleneck is prompt message usefulness, not pool size.

## Next Optimization Direction

The next implementation stage should prioritize:

1. fixed positive-scale diagnostics to measure whether prompt messages reduce
   train-node CE;
2. a prompt-benefit predictor trained from train-only CE delta;
3. message generation as node-level residual correction;
4. receiver ablations for `v1_linear`, `v2_conditioned`, normalization, and
   gate initialization;
5. only after prompt messages are useful, revisit pool expansion and stronger
   prompt edge weights.

