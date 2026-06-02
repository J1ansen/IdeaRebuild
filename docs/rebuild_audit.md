# Rebuild Audit

Date: 2026-06-02

Scope: Phase 0 only. This audit compares the local official GP2F code at `/Users/jackson/GP2F-main` with the previous idea code at `/Users/jackson/MyIdea`. No core model code has been modified in either repository.

## 1. Checked Files

### MyIdea

| Area | Files checked |
|---|---|
| model path | `/Users/jackson/MyIdea/models/dual_branch.py`, `/Users/jackson/MyIdea/models/gp2f_adapter.py`, `/Users/jackson/MyIdea/models/gp2f_prompt.py`, `/Users/jackson/MyIdea/models/prompt_conv.py`, `/Users/jackson/MyIdea/models/base_gnn.py`, `/Users/jackson/MyIdea/models/graph_utils.py` |
| prompt path | `/Users/jackson/MyIdea/prompts/gumbel_route.py`, `/Users/jackson/MyIdea/prompts/structure_hub.py`, `/Users/jackson/MyIdea/prompts/hub_prompt.py`, `/Users/jackson/MyIdea/prompts/cluster_generator.py`, `/Users/jackson/MyIdea/prompts/target_ssl.py`, `/Users/jackson/MyIdea/prompts/structure_descriptor.py` |
| losses and training | `/Users/jackson/MyIdea/loss.py`, `/Users/jackson/MyIdea/train.py` |
| data and tests | `/Users/jackson/MyIdea/load_data.py`, `/Users/jackson/MyIdea/test/test_pure_dual_branch.py`, `/Users/jackson/MyIdea/test/test_hub_prompt.py`, `/Users/jackson/MyIdea/test/test_structure_hub_audit.py`, `/Users/jackson/MyIdea/test/test_prompt_topology_audit.py` |

### Official GP2F

| Area | Files checked |
|---|---|
| main flow | `/Users/jackson/GP2F-main/main.py` |
| GP2F modules | `/Users/jackson/GP2F-main/model/prompt/GP2F.py` |
| backbone/head/projector | `/Users/jackson/GP2F-main/model/backbones/GCN.py`, `/Users/jackson/GP2F-main/model/classifier.py`, `/Users/jackson/GP2F-main/model/projector.py` |
| data/utils | `/Users/jackson/GP2F-main/data.py`, `/Users/jackson/GP2F-main/util.py` |

## 2. Current MyIdea Architecture Audit

| Item | Current MyIdea behavior |
|---|---|
| frozen branch input | Uses aligned target features `x_aligned = input_aligner(data.x)` and the original target `data.edge_index`. `DualBranchGNN.set_frozen_graph()` precomputes a normalized original graph with heterophily weighting disabled. |
| frozen branch model source | Loaded by `load_pretrained_backbone()` into `BaseGCN`, using checkpoint dimensions inferred from `pretrained_gnns`. |
| frozen branch frozen params | `backbone.parameters()` and `DualBranchGNN.frozen_branch.parameters()` are set `requires_grad=False`; forward is also wrapped in `torch.no_grad()`. |
| adapted branch input | If prompt edges are enabled, adapted branch receives `prompt_nodes`, `prompt_edge_index`, `edge_type`, optional `edge_weight_scale`, and local heterophily scores. If `--no_prompt_edges`, it uses only `edge_index`. |
| adapted branch model source | Two modes exist. `adapt_branch="gp2f"` uses `PromptModule` around the frozen backbone layers. `adapt_branch="legacy"` uses a newly trained `PromptBranchEncoder` built from `PromptAwareGNNConv`, followed by `GP2FAdapter`. |
| adapted branch trainable params | In `gp2f` mode, `PromptModule.adapters` and per-layer `alphas` are trainable; frozen backbone layers remain frozen. In `legacy` mode, `PromptBranchEncoder`, projection layers, and final adapter are trainable. |
| adapter insertion position | In `gp2f` mode, adapter is inserted after every frozen GNN layer via `PromptModule.forward()`: `x_gnn = conv(...)`, then `x = x_gnn + alpha * adapter(x_gnn)`. In `legacy` mode, `GP2FAdapter` is applied once after the new mini-GNN branch output. |
| fusion implementation | If `trainer_style="gp2f"`, `fusion_mode="gp2f"` and `DualBranchFramework` computes `alpha * h_frozen + (1-alpha) * h_adapted`. `alpha` is a raw learnable scalar, not sigmoid-constrained. If `trainer_style="legacy"`, epoch scheduled gate is used. The model still carries `GateSchedule` even in GP2F mode. |
| prompt node / prompt edge main path | Enabled by default because `use_prompt_edges = not args.no_prompt_edges`. Default `prompt_init="dual_freq"`, with options `target_ssl`, `structure_hub`, and `hub_only`. Thus default training is not original-graph-only. |
| Gumbel routing main path | For `dual_freq` and `target_ssl`, `PromptRouter` builds `EX_homo/EX_hete` with Gumbel-Softmax and prompt edges. For `hub_only`, `HubPromptBuilder` uses Gumbel-Softmax route probabilities. For `structure_hub`, routing is fixed clustering/role assignment rather than Gumbel. |
| current losses | `trainer_style="gp2f"` uses `compute_gp2f_loss`: CE + original/other topology consistency + cross-view contrastive + optional hub routing losses. `trainer_style="legacy"` uses `compute_total_loss`: CE + sparsity + branch consistency KL + prompt route contrastive + optional orthogonal/balance losses. |
| losses depending on original adjacency | `gp2f_contrastive_loss()` uses `edge_index` to build positive masks. `gp2f_topology_consistency_loss()` uses `edge_index` to build dense target adjacency. Frozen branch also uses original adjacency. |
| variable mixing risk | High. The default path mixes input alignment, prompt initialization, prompt edges, heterophily edge weighting, adapter training, GP2F contrastive/topology losses, optional hub losses, and learnable fusion. This makes it difficult to attribute gains or drops to faithful GP2F itself. |

## 3. Official GP2F Architecture Audit

| Item | Official GP2F behavior |
|---|---|
| frozen branch | In `prompt_tuning()`, a pretrained GCN is loaded, all parameters are frozen, and the model is run in eval mode. The frozen output is produced by the same pretrained encoder on projected target features and the original target graph. |
| adapted branch sharing | The adapted branch does not instantiate a separate trainable GNN. `PromptModule` receives the frozen `gnn_model` and iterates through `gnn_model.convs`. |
| layer-wise adapter | `PromptModule` has one bottleneck MLP adapter and one scalar `alpha` per GNN layer. Each layer computes `x_gnn = conv(x, edge_index)` followed by `x = x_gnn + alpha * adapter(x_gnn)`, then the original activation/dropout except after the last layer. |
| fusion parameter | `DualBranchFramework(mode="adaptive")` defines `self.alpha = nn.Parameter(torch.tensor(alpha_init))`. Fusion is `alpha * frozen_output + (1-alpha) * prompt_output`. The official code does not apply sigmoid or clamp to alpha. |
| classification head | `LogReg(hidden_dim, num_class)` is a single linear classifier trained jointly with projector, prompt adapters, and fusion alpha. |
| projector | `Projector(down_dim, input_dim)` maps downstream feature dimensions to the pretraining feature dimension before both branches. |
| contrastive loss | `CrossViewContrastiveLoss` builds positive masks from original `edge_index`. Intra-view positives exclude self-loops from adjacency; cross-view positives include original adjacency plus diagonal. Loss is symmetric frozen-to-tuned and tuned-to-frozen. |
| topology-consistent fusion loss | `consistent_topology_loss_with_fused_sim()` compares fused representation similarity against the original dense adjacency. It thresholds fused similarity by percentile and applies BCE only on pairs where topology and feature-similarity constraint agree. |
| node classification training | Minimal components to migrate: pretrained frozen GCN, feature projector/input aligner, layer-wise `PromptModule`, `DualBranchFramework`, classifier, CE loss, optional original contrastive loss, optional original topology fusion loss, few-shot train/val/test split, metric logging. |

## 4. Difference Matrix

| Component | Official GP2F | Current MyIdea | Must rebuild? | Rebuild plan |
|---|---|---|---|---|
| Frozen branch | Frozen pretrained GCN on original target graph. | Frozen pretrained `BaseGCN` on original graph; wrapped in `torch.no_grad()` and domain alignment. | Partially | Keep concept, but isolate in a new baseline module and make frozen behavior testable. |
| Adapted branch | Same frozen encoder layers plus per-layer adapters. | `gp2f` mode matches the layer-wise pattern; `legacy` mode uses a separate trainable mini-GNN. Default train script allows both. | Yes | New `FaithfulGP2F` should only use shared frozen layers plus per-layer adapters. No legacy branch in baseline path. |
| Layer-wise adapter | Bottleneck MLP after every encoder layer. | Implemented in `models/gp2f_prompt.py`; separate `GP2FAdapter` is single-output adapter used by legacy path. | Yes | Move/implement clean `ResidualBottleneckAdapter` and enforce `len(adapters) == len(backbone.convs)`. |
| Fusion alpha | Raw learnable scalar in official code. | Raw learnable scalar in `DualBranchFramework`; scheduled gate exists for legacy and no-prompt paths. | Yes | Baseline should use a single learnable `alpha_logit` with `sigmoid` if following new design requirement, and should not include epoch gate. |
| Classification head | Linear head on fused output. | Linear head on fused output. | No, but isolate | Keep simple linear classifier in `FaithfulGP2F`. |
| Contrastive loss | Original adjacency-driven cross-view contrastive. | Implemented as `gp2f_contrastive_loss()` and always computed in `compute_gp2f_loss`, multiplied by lambda. | Yes | Move into `losses/gp2f_losses.py` and gate execution by `use_original_contrastive`, so disabled mode does not access the path. |
| Topology fusion loss | Original adjacency-driven topology consistency on fused similarity. | Implemented, with extra `orig/prompt/mixed/off` modes. | Yes | Implement only original topology loss for Phase 1, config-gated by `use_original_topology_fusion`. No prompt-induced topology loss. |
| Prompt graph interface | No prompt nodes/edges in official GP2F node classification baseline. | Default MyIdea training enables prompt edges and multiple prompt builders. | Yes | Phase 1 baseline input must be original graph only, with only future `adapted_x/adapted_edge_index` signature reserved. |
| Gumbel routing | Not part of official GP2F baseline. | Present in `PromptRouter` and `HubPromptBuilder`; default prompt path uses it unless disabled or using structure_hub. | Yes | Exclude from baseline entry point and tests. |
| Heterophily edge weighting | Not part of official GP2F. | `PromptAwareGNNConv` and graph utilities can scale original/prompt edges by heterophily. | Yes | Exclude from faithful baseline. Revisit only in later prompt-hub phases. |
| Experiment attribution | GP2F losses and architecture are isolated in official flow. | Default path combines prompt graph, routing, hub losses, topology losses, and heterophily weighting. | Yes | New experiment runner should have explicit B0-B3 loss switches and log alpha/similarity separately. |

## 5. Dataset Mapping Notes

| Required target | MyIdea canonical name |
|---|---|
| Actor | `Actor` |
| Chameleon | `chameleon`; aliases include `Chameleon` |
| Squirrel | `squirrel`; aliases include `Squirrel` |
| Cora | `Cora` |
| CiteSeer | `CiteSeer` |
| PubMed | `PubMed` |

MyIdea uses PyG `WikipediaNetwork(name="chameleon"|"squirrel")` and documents squirrel as the PyG SQUIRREL-F variant. Official GP2F uses capitalized `Chameleon` and `Squirrel` in `dataset4node()`.

## 6. Phase 1 File Plan

New repository target: `/Users/jackson/IdeaRebuild`.

Planned new files:

| File | Purpose |
|---|---|
| `models/adapters.py` | `ResidualBottleneckAdapter` with trainable beta and type hints. |
| `models/backbones.py` | Pretrained GCN checkpoint loader; reuses existing `.pkl/.pth` files and freezes the loaded backbone. |
| `models/faithful_gp2f.py` | `FaithfulGP2F` model with frozen branch, adapted branch, sigmoid alpha fusion, classifier, and future adapted graph interface. |
| `losses/gp2f_losses.py` | CE, optional original contrastive loss, optional original topology fusion loss, config dataclass, and loss outputs. |
| `experiments/run_gp2f_baseline.py` | Isolated B0-B3 runner, no prompt graph path. |
| `configs/gp2f_baseline.yaml` | Baseline config with heterophilic defaults disabling original structure losses. |
| `tests/test_faithful_gp2f.py` | Freezing, adapter count, alpha range, no-NaN forward/backward, disabled loss-path, toy smoke training. |
| `data/` or `datasets/` loader module | Minimal dataset loading and few-shot split wrapper, adapted from MyIdea only after selecting a clean interface. |

Expected modifications to existing new-repo files:

| File | Planned change |
|---|---|
| `README.md` | Add usage once baseline runner exists. |
| `docs/rebuild_plan.md` | Update with Phase 1 implementation decisions after confirmation. |
| `configs/gp2f_baseline.yaml` | Expand with data paths, pretrained paths, logging, and B0-B3 presets. |

## 7. Blocking Interface Issues For Faithful Reproduction

1. Git initialization is blocked in `/Users/jackson/IdeaRebuild` because the environment rejects creating `.git`. Code files can be written, but commits/branches cannot currently be created by Codex.
2. The faithful baseline needs a clean pretrained-backbone interface. MyIdea's `load_pretrained_backbone()` assumes GCN checkpoints and infers dimensions from checkpoint keys; this is reusable but should be copied into the new repo as a small isolated loader.
3. Official GP2F uses a raw unconstrained fusion alpha, while the requested Phase 1 design requires `alpha = sigmoid(alpha_logit)`. This is intentionally safer and bounded, but it is a deliberate deviation from official source code.
4. Official GP2F always computes the contrastive path during GP2F training. Phase 1 requires disabled structure losses to avoid accessing those paths when switches are off, so tests must verify no hidden dense adjacency construction occurs.
5. MyIdea's current default dataset names differ in case for `chameleon` and `squirrel`; the new loader should normalize aliases explicitly.
6. MyIdea's current no-prompt ablation uses `adapt_branch="legacy"` and scheduled gate, so it cannot serve as the faithful GP2F baseline.

## 8. Phase 0 Code Modification Status

No core model code was modified.

Files changed in the new repository during the initial Phase 0 setup:

- `/Users/jackson/IdeaRebuild/README.md`
- `/Users/jackson/IdeaRebuild/docs/rebuild_plan.md`
- `/Users/jackson/IdeaRebuild/docs/rebuild_audit.md`
- `/Users/jackson/IdeaRebuild/configs/gp2f_baseline.yaml`
- `/Users/jackson/IdeaRebuild/.gitignore`
- package marker files under `models/`, `losses/`, `experiments/`, and `tests/`

Post-audit infrastructure added after the pretrained-checkpoint reuse request:

- `/Users/jackson/IdeaRebuild/models/backbones.py`
- `/Users/jackson/IdeaRebuild/tests/test_pretrained_backbone.py`
- updates to `/Users/jackson/IdeaRebuild/configs/gp2f_baseline.yaml`, `/Users/jackson/IdeaRebuild/README.md`, and `/Users/jackson/IdeaRebuild/docs/rebuild_plan.md`

No files were changed under:

- `/Users/jackson/MyIdea`
- `/Users/jackson/GP2F-main`
