# Experiment Records

This file records runnable experiment evidence separately from design notes. All results below were produced from the local repository state at the time of the run.

## 2026-06-03 20:45 CST - P0 Fixed Budget Sanity, Cora

Purpose: check whether the implemented `Unified Minimal Multi-view Residual Prompt` improves over a CE-only dual-branch NoPrompt baseline under the same split protocol.

Command template:

```bash
for variant in noprompt param_control full_p0; do
  .venv/bin/python -m experiments.run_gp2f_prompt \
    --config configs/gp2f_prompt_p0.yaml \
    --target_dataset Cora \
    --prompt_variant "${variant}" \
    --route_budget 0.10 \
    --seeds 0,1,2,3,4 \
    --eval_every 5 \
    --output_dir outputs/prompt_eval_zdetached
done
```

Shared setting:

- Dataset: `Cora`
- Seeds: `0,1,2,3,4`
- Loss: CE-only
- Structural prompt base: `z_detached`
- Prompt budget: fixed `route_budget=0.10`
- Evaluation interval: `eval_every=5`
- Output root: `outputs/prompt_eval_zdetached/Cora`

Results:

| Variant | Output Dir | Best Test Acc | Best Test Macro-F1 | Final Test Acc | Final Test Macro-F1 |
| --- | --- | ---: | ---: | ---: | ---: |
| `noprompt` | `20260603_204151` | 69.05+-3.84 | 66.43+-4.01 | 54.02+-14.59 | 52.40+-14.86 |
| `param_control` | `20260603_204208` | 65.38+-5.89 | 63.69+-5.45 | 43.79+-7.38 | 43.03+-9.63 |
| `full_p0` | `20260603_204247` | 65.84+-6.11 | 64.00+-5.84 | 33.66+-5.79 | 29.38+-10.37 |

Prompt diagnostics at best epoch:

| Variant | mean(g_sem) | mean(g_struct) | mean(g_null) | Non-null Ratio | Prompt Norm | Gamma |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `noprompt` | 0.0000 | 0.0000 | 1.0000 | 0.0000 | 0.0000 | 0.0000 |
| `param_control` | 0.1552 | 0.1550 | 0.6899 | 0.3101 | 0.0689 | 0.0564 |
| `full_p0` | 0.1479 | 0.1480 | 0.7041 | 0.2959 | 0.0626 | 0.0557 |

Prompt diagnostics at final epoch:

| Variant | mean(g_sem) | mean(g_struct) | mean(g_null) | Non-null Ratio | Prompt Norm | Gamma |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `noprompt` | 0.0000 | 0.0000 | 1.0000 | 0.0000 | 0.0000 | 0.0000 |
| `param_control` | 0.1983 | 0.1985 | 0.6032 | 0.3968 | 0.2549 | 0.0600 |
| `full_p0` | 0.1751 | 0.2173 | 0.6077 | 0.3923 | 0.3579 | 0.0594 |

Sanity checks:

- `init_adapted_x_delta=0.0` and `init_logit_delta=0.0` for all variants, so zero-init equivalence is working.
- `connected_edge_count=0` by design for P0; this run is residual feature prompting, not topology prompting.

Interpretation:

- Under `route_budget=0.10`, `full_p0` does not improve over `noprompt` on Cora. It is slightly above `param_control` on best accuracy/F1, but the margin is small and both are below NoPrompt.
- The final-epoch collapse is severe for `full_p0`, while the best-epoch metrics are usable. This suggests the prompt path is learnable but currently over-intervenes or destabilizes training.
- The next useful experiment is not a new prompt module yet. First test smaller intervention settings, especially `route_budget=0.00/0.05`, lower `gamma_max` or stronger budget penalty, and rely on validation-selected checkpoints.

## 2026-06-03 20:49 CST - P0 Route Budget Sweep, Cora

Purpose: test whether the P0 degradation under `route_budget=0.10` comes from too much prompt intervention.

Command template:

```bash
for budget in 0.00 0.05; do
  for variant in param_control full_p0; do
    .venv/bin/python -m experiments.run_gp2f_prompt \
      --config configs/gp2f_prompt_p0.yaml \
      --target_dataset Cora \
      --prompt_variant "${variant}" \
      --route_budget "${budget}" \
      --seeds 0,1,2,3,4 \
      --eval_every 5 \
      --output_dir outputs/prompt_eval_zdetached_budget_sweep
  done
done
```

Shared setting:

- Dataset: `Cora`
- Seeds: `0,1,2,3,4`
- Loss: CE-only
- Structural prompt base: `z_detached`
- Evaluation interval: `eval_every=5`
- Output root: `outputs/prompt_eval_zdetached_budget_sweep/Cora`

Results:

| Variant | Route Budget | Output Dir | Best Test Acc | Best Test Macro-F1 | Final Test Acc | Final Test Macro-F1 |
| --- | ---: | --- | ---: | ---: | ---: | ---: |
| `param_control` | 0.00 | `20260603_204620` | 67.18+-6.16 | 65.00+-5.67 | 47.97+-8.99 | 46.09+-11.89 |
| `full_p0` | 0.00 | `20260603_204658` | 66.09+-6.15 | 64.23+-6.09 | 35.35+-9.49 | 32.55+-12.22 |
| `param_control` | 0.05 | `20260603_204820` | 65.75+-5.65 | 64.03+-5.27 | 43.60+-8.35 | 42.07+-11.59 |
| `full_p0` | 0.05 | `20260603_204909` | 66.03+-6.09 | 64.19+-5.90 | 34.52+-11.36 | 29.11+-15.50 |

Prompt diagnostics at best epoch:

| Variant | Route Budget | Non-null Ratio | Prompt Norm | Gamma |
| --- | ---: | ---: | ---: | ---: |
| `param_control` | 0.00 | 0.2515 | 0.0479 | 0.0559 |
| `full_p0` | 0.00 | 0.2487 | 0.0543 | 0.0558 |
| `param_control` | 0.05 | 0.2803 | 0.0591 | 0.0563 |
| `full_p0` | 0.05 | 0.2708 | 0.0589 | 0.0558 |

Prompt diagnostics at final epoch:

| Variant | Route Budget | Non-null Ratio | Prompt Norm |
| --- | ---: | ---: | ---: |
| `param_control` | 0.00 | 0.3004 | 0.1341 |
| `full_p0` | 0.00 | 0.3093 | 0.3474 |
| `param_control` | 0.05 | 0.3632 | 0.2041 |
| `full_p0` | 0.05 | 0.3614 | 0.3784 |

Interpretation:

- Reducing `route_budget` from `0.10` to `0.00/0.05` lowers best-epoch prompt norm, so the budget loss is active.
- Lower budget does not make `full_p0` outperform `noprompt` on Cora. Best `full_p0` is 66.09+-6.15 versus NoPrompt 69.05+-3.84.
- `param_control` remains close to or above `full_p0`, which means current P0 gains cannot yet be attributed to semantic/structural prompt evidence.
- `full_p0` final collapse persists across all tested budgets. This points to residual prompt drift or base-parameter co-adaptation, not only route budget size.
- Next fix should target training stability and attribution before running expensive heterophilic datasets: lower `gamma_max`, stronger budget regularization, prompt-only fine-tuning from a fixed NoPrompt checkpoint, or early checkpoint selection only.

## 2026-06-03 21:08 CST - P0 Stability Fix, Prompt-only Cora

Purpose: verify whether the stability fixes make P0 safe to train when the base dual-branch model is fixed. This isolates the residual prompt contribution from continued adapter/classifier training.

Stability changes:

- `gamma_init=0.01`
- `gamma_max=0.1`
- `lambda_budget=0.5`
- `lambda_message_norm=1.0`
- `message_norm_target=0.05`
- Support loading a seed-matched NoPrompt checkpoint through `--base_checkpoint_root`
- Support `--freeze_base`, so only the prompt/control module is trainable

Command template:

```bash
for variant in param_control full_p0; do
  .venv/bin/python -m experiments.run_gp2f_prompt \
    --config configs/gp2f_prompt_p0.yaml \
    --target_dataset Cora \
    --prompt_variant "${variant}" \
    --route_budget 0.05 \
    --seeds 0,1,2,3,4 \
    --eval_every 5 \
    --base_checkpoint_root outputs/prompt_eval_zdetached/Cora/20260603_204151 \
    --freeze_base \
    --output_dir outputs/prompt_stability_prompt_only
done
```

Shared setting:

- Dataset: `Cora`
- Seeds: `0,1,2,3,4`
- Loss: CE-only plus budget/message-norm regularization
- Base initialization: seed-matched NoPrompt best checkpoint from `outputs/prompt_eval_zdetached/Cora/20260603_204151`
- Base model: frozen
- Prompt budget: fixed `route_budget=0.05`
- Evaluation interval: `eval_every=5`

Results:

| Variant | Output Dir | Best Test Acc | Best Test Macro-F1 | Final Test Acc | Final Test Macro-F1 |
| --- | --- | ---: | ---: | ---: | ---: |
| `param_control` | `outputs/prompt_stability_prompt_only/Cora/20260603_210809` | 68.96+-3.79 | 66.36+-3.92 | 68.88+-3.94 | 66.32+-4.05 |
| `full_p0` | `outputs/prompt_stability_prompt_only/Cora/20260603_210821` | 68.94+-3.78 | 66.35+-3.90 | 68.98+-3.89 | 66.41+-4.02 |

Prompt diagnostics at best epoch:

| Variant | Non-null Ratio | Prompt Norm | Gamma | Message Norm Loss |
| --- | ---: | ---: | ---: | ---: |
| `param_control` | 0.4178 | 0.0016 | 0.0100 | 0.0000 |
| `full_p0` | 0.4051 | 0.0023 | 0.0101 | 0.0000 |

Interpretation:

- The prompt-only protocol is stable. Final metrics no longer collapse because the base adapter/classifier/fusion parameters are not drifting.
- `full_p0` is essentially tied with `param_control` and the NoPrompt checkpoint on Cora. This is good for stability, but not yet evidence that semantic/structural views are useful.
- Message norm is extremely small, so the current regularization is conservative. The module is behaving like a safe perturbation, not an effective adaptation signal yet.

## 2026-06-03 21:09 CST - P0 Stability Fix, Joint Cora

Purpose: check whether the same stability fixes prevent collapse when the base dual-branch parameters and prompt module are trained jointly.

Command template:

```bash
for variant in param_control full_p0; do
  .venv/bin/python -m experiments.run_gp2f_prompt \
    --config configs/gp2f_prompt_p0.yaml \
    --target_dataset Cora \
    --prompt_variant "${variant}" \
    --route_budget 0.05 \
    --seeds 0,1,2,3,4 \
    --eval_every 5 \
    --output_dir outputs/prompt_stability_joint
done
```

Shared setting:

- Dataset: `Cora`
- Seeds: `0,1,2,3,4`
- Loss: CE-only plus budget/message-norm regularization
- Base initialization: pretrained GNN checkpoint plus fresh adapter/classifier
- Base model: trainable
- Prompt budget: fixed `route_budget=0.05`
- Evaluation interval: `eval_every=5`

Results:

| Variant | Output Dir | Best Test Acc | Best Test Macro-F1 | Final Test Acc | Final Test Macro-F1 |
| --- | --- | ---: | ---: | ---: | ---: |
| `param_control` | `outputs/prompt_stability_joint/Cora/20260603_210902` | 69.07+-3.82 | 66.37+-4.12 | 55.25+-13.58 | 53.58+-13.83 |
| `full_p0` | `outputs/prompt_stability_joint/Cora/20260603_210943` | 69.05+-3.78 | 66.35+-4.08 | 55.38+-13.58 | 53.69+-13.83 |

Prompt diagnostics at best epoch:

| Variant | Non-null Ratio | Prompt Norm | Gamma | Message Norm Loss |
| --- | ---: | ---: | ---: | ---: |
| `param_control` | 0.1386 | 0.0010 | 0.0111 | 0.0000 |
| `full_p0` | 0.1319 | 0.0010 | 0.0111 | 0.0000 |

Interpretation:

- The best checkpoint is now back in the NoPrompt range, so the new regularization prevents prompt magnitude from becoming the main failure mode.
- The final checkpoint still collapses in joint training. Diagnostics show the prompt message remains tiny, so the remaining collapse is mostly base dual-branch overfitting/drift rather than prompt explosion.
- `full_p0` and `param_control` remain almost identical. Current P0 is stable but not yet effective as a view-specific prompt module on Cora.
- For method validation, the next useful step is not an expensive dataset sweep. We need either a less conservative prompt schedule/grid or a stronger prompt view, then rerun `NoPrompt` / `ParamControl` / `Full-P0` under the same CE-only protocol.

## 2026-06-03 21:15 CST - P0 Strength Probe, Prompt-only Cora

Purpose: test whether the previous prompt-only results were neutral because the message norm was too conservative.

Command template:

```bash
for variant in param_control full_p0; do
  .venv/bin/python -m experiments.run_gp2f_prompt \
    --config configs/gp2f_prompt_p0.yaml \
    --target_dataset Cora \
    --prompt_variant "${variant}" \
    --route_budget 0.10 \
    --gamma_max 0.2 \
    --lambda_message_norm 0.1 \
    --message_norm_target 0.10 \
    --seeds 0,1,2,3,4 \
    --eval_every 5 \
    --base_checkpoint_root outputs/prompt_eval_zdetached/Cora/20260603_204151 \
    --freeze_base \
    --output_dir outputs/prompt_strength_probe
done
```

Shared setting:

- Dataset: `Cora`
- Seeds: `0,1,2,3,4`
- Loss: CE-only plus budget/message-norm regularization
- Base initialization: seed-matched NoPrompt best checkpoint from `outputs/prompt_eval_zdetached/Cora/20260603_204151`
- Base model: frozen
- Prompt budget: fixed `route_budget=0.10`
- Prompt strength: `gamma_max=0.2`, `message_norm_target=0.10`, `lambda_message_norm=0.1`

Results:

| Variant | Output Dir | Best Test Acc | Best Test Macro-F1 | Final Test Acc | Final Test Macro-F1 |
| --- | --- | ---: | ---: | ---: | ---: |
| `param_control` | `outputs/prompt_strength_probe/Cora/20260603_211453` | 68.96+-3.79 | 66.36+-3.92 | 68.88+-3.92 | 66.31+-4.03 |
| `full_p0` | `outputs/prompt_strength_probe/Cora/20260603_211506` | 68.93+-3.79 | 66.34+-3.91 | 68.96+-3.93 | 66.39+-4.05 |

Prompt diagnostics at best epoch:

| Variant | Non-null Ratio | Prompt Norm | Gamma | Message Norm Loss |
| --- | ---: | ---: | ---: | ---: |
| `param_control` | 0.4178 | 0.0016 | 0.0100 | 0.0000 |
| `full_p0` | 0.4056 | 0.0023 | 0.0101 | 0.0000 |

Interpretation:

- Relaxing the prompt strength settings does not make `full_p0` outperform `param_control`.
- The learned prompt message is still tiny, and `gamma` stays near its initialization. The model is choosing not to rely on the current residual prompt views.
- This strengthens the conclusion that the present P0 implementation is stable but underpowered. The next module work should improve the prompt signal itself, not just tune budget/gamma.

## 2026-06-03 21:42 CST - Enhanced Structural P0 Probe, Prompt-only Cora

Purpose: test the first strengthened P0 implementation while keeping the P0 boundary unchanged: no prompt nodes, no prompt edges, no candidate connection pool, no pseudo-label graph construction.

Implementation changes in this probe:

- Structural View now supports label-free enhanced context:
  - normalized log in/out degree
  - one-hop neighbor variance
  - local cosine summaries between `base`, `m1`, and `m2`
- Gate input includes the enhanced structural context when enabled.
- Prompt/control parameters use a separate optimizer group.
- Config default uses `prompt_lr=0.003`, `null_bias_init=0.5`, `budget_warmup_epochs=50`, and `message_norm_warmup_epochs=20`.

Command template:

```bash
for variant in param_control full_p0; do
  .venv/bin/python -m experiments.run_gp2f_prompt \
    --config configs/gp2f_prompt_p0.yaml \
    --target_dataset Cora \
    --prompt_variant "${variant}" \
    --route_budget 0.10 \
    --seeds 0,1,2,3,4 \
    --eval_every 5 \
    --base_checkpoint_root outputs/prompt_eval_zdetached/Cora/20260603_204151 \
    --freeze_base \
    --output_dir outputs/prompt_enhanced_probe
done
```

Shared setting:

- Dataset: `Cora`
- Seeds: `0,1,2,3,4`
- Loss: CE-only plus scheduled budget/message-norm regularization
- Base initialization: seed-matched NoPrompt best checkpoint from `outputs/prompt_eval_zdetached/Cora/20260603_204151`
- Base model: frozen
- Prompt budget: fixed `route_budget=0.10`

Results:

| Variant | Output Dir | Best Test Acc | Best Test Macro-F1 | Final Test Acc | Final Test Macro-F1 |
| --- | --- | ---: | ---: | ---: | ---: |
| `param_control` | `outputs/prompt_enhanced_probe/Cora/20260603_214150` | 69.09+-3.88 | 66.51+-4.05 | 69.00+-3.77 | 66.40+-3.97 |
| `full_p0` | `outputs/prompt_enhanced_probe/Cora/20260603_214204` | 68.97+-3.87 | 66.39+-4.02 | 69.01+-3.81 | 66.40+-3.99 |

Prompt diagnostics at best epoch:

| Variant | Non-null Ratio | Prompt Norm | Gamma | Structural Var Norm | Degree Mean | Similarity Mean |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `param_control` | 0.4805 | 0.0049 | 0.0102 | 0.0000 | 0.0000 | 0.0000 |
| `full_p0` | 0.4224 | 0.0032 | 0.0103 | 0.0573 | 0.2752 | 0.3847 |

Interpretation:

- Enhanced structural statistics are computed and logged for `full_p0`, so the new view is active.
- `full_p0` still does not outperform `param_control` on Cora. The result remains effectively tied.
- The model increases Null route by the end of training and keeps prompt message norm small. This suggests that, under frozen-base Cora, the current residual prompt is still treated as a weak optional correction rather than a useful adaptation mechanism.
- This does not invalidate P0 stability. It does mean that Cora->Cora is now mostly a safety sanity check, not evidence of method advantage. The next meaningful test should be a cross-domain or heterophilic target, and the next method step should consider either a stronger structural route objective or a P1 edge/pool prompt as a separate module.

## 2026-06-04 10:13 CST - Enhanced Structural P0 Probe, Cross-domain / Heterophilic Targets

Purpose: test whether the stabilized enhanced P0 becomes useful when the target is no longer Cora->Cora. This run uses a frozen-base attribution protocol: first train a seed-matched CE-only NoPrompt base, then load each seed's best base checkpoint, freeze the GP2F dual-branch model, and train only the prompt/control module.

Command templates:

```bash
.venv/bin/python -m experiments.run_gp2f_prompt \
  --config configs/gp2f_prompt_p0.yaml \
  --target_dataset PubMed \
  --prompt_variant noprompt \
  --seeds 0,1,2,3,4 \
  --eval_every 5 \
  --output_dir outputs/prompt_crossdomain_noprompt

for variant in param_control full_p0; do
  .venv/bin/python -m experiments.run_gp2f_prompt \
    --config configs/gp2f_prompt_p0.yaml \
    --target_dataset PubMed \
    --prompt_variant "${variant}" \
    --route_budget 0.10 \
    --seeds 0,1,2,3,4 \
    --eval_every 5 \
    --base_checkpoint_root outputs/prompt_crossdomain_noprompt/PubMed/20260604_095409 \
    --freeze_base \
    --output_dir outputs/prompt_crossdomain_probe
done
```

The same protocol was repeated for `Actor`, using `outputs/prompt_crossdomain_noprompt/Actor/20260604_100848` as the seed-matched NoPrompt base checkpoint root.

Shared setting:

- Source checkpoint: Cora GRACE pretrained GNN, hidden dim 128
- Target datasets: `PubMed`, `Actor`
- Seeds: `0,1,2,3,4`
- Loss: CE-only for base; CE-only plus scheduled budget/message-norm regularization for prompt/control
- Prompt/control protocol: frozen GP2F base, fixed `route_budget=0.10`
- P0 boundary: no prompt nodes, no prompt edges, no pseudo-label graph construction

Results:

| Dataset | Variant | Output Dir | Best Test Acc | Best Test Macro-F1 | Final Test Acc | Final Test Macro-F1 |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| PubMed | `noprompt` | `outputs/prompt_crossdomain_noprompt/PubMed/20260604_095409` | 59.15+-8.05 | 56.83+-9.39 | 58.77+-6.18 | 58.68+-5.82 |
| PubMed | `param_control` | `outputs/prompt_crossdomain_probe/PubMed/20260604_095640` | 59.18+-8.01 | 56.86+-9.35 | 59.15+-8.04 | 56.83+-9.38 |
| PubMed | `full_p0` | `outputs/prompt_crossdomain_probe/PubMed/20260604_095758` | 59.19+-8.01 | 56.86+-9.36 | 59.15+-8.05 | 56.82+-9.39 |
| Actor | `noprompt` | `outputs/prompt_crossdomain_noprompt/Actor/20260604_100848` | 19.12+-4.93 | 11.33+-3.45 | 20.41+-2.17 | 18.56+-1.13 |
| Actor | `param_control` | `outputs/prompt_crossdomain_probe/Actor/20260604_101002` | 19.14+-4.94 | 11.35+-3.45 | 19.13+-4.94 | 11.34+-3.45 |
| Actor | `full_p0` | `outputs/prompt_crossdomain_probe/Actor/20260604_101031` | 19.13+-4.94 | 11.35+-3.45 | 19.12+-4.94 | 11.34+-3.45 |

Best-checkpoint prompt diagnostics:

| Dataset | Variant | Best Epoch | Route sem/struct/null | Non-null Ratio | Prompt Norm | Gamma | u_prompt Norm | Semantic Margin | Structural Var Norm | Structural Sim |
| --- | --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| PubMed | `param_control` | 1.0 | 0.272/0.271/0.457 | 0.543 | 0.0032 | 0.0100 | 0.3203 | 0.0000 | 0.0000 | 0.0000 |
| PubMed | `full_p0` | 1.0 | 0.264/0.269/0.466 | 0.534 | 0.0029 | 0.0100 | 0.2900 | 0.5305 | 0.0023 | 0.7506 |
| Actor | `param_control` | 1.0 | 0.272/0.272/0.456 | 0.544 | 0.0017 | 0.0100 | 0.1729 | 0.0000 | 0.0000 | 0.0000 |
| Actor | `full_p0` | 1.0 | 0.270/0.271/0.459 | 0.541 | 0.0016 | 0.0100 | 0.1573 | 0.1780 | 0.0086 | 0.2718 |

Interpretation:

- `full_p0` remains effectively tied with `param_control` and NoPrompt on both PubMed and Actor. There is no evidence yet that the current residual multi-view prompt contributes target adaptation beyond the frozen CE-only base.
- The best epoch is consistently epoch 1 for prompt/control runs. This means validation selection prefers the near-zero initialized prompt state rather than a learned prompt intervention.
- P0 is behaving safely: `connected_edge_count=0`, zero-init equivalence is preserved, `gamma` stays near initialization, and prompt message norms are tiny.
- The current problem is not instability; it is weak activation/usefulness. The prompt views are computed, but the classifier does not find them helpful under the frozen-base protocol.
- For PubMed, the enhanced structural probe is expensive because structural summaries are recomputed each epoch on a larger graph. Caching label-free structural summaries is needed before running larger grids.
- The next method step should not be another broad sweep of the same P0 settings. More useful directions are: validation-grid over less conservative prompt activation, cached structural summaries, or a separate P1 pool/edge prompt module with strict isolation from the P0 baseline.

## 2026-06-05 21:06 CST - P0 Structural Cache and Strong Activation Probe

Purpose: reduce P0 experiment overhead and test whether the previous neutral P0 results were mainly caused by overly conservative route/gamma settings.

Implementation update:

- Added an in-memory structural summary cache for `UnifiedMultiViewResidualPrompt`.
- Cache is used only when `prompt.structural.cache.enabled=true` and the base model is frozen.
- The cache stores only raw label-free summaries: `base`, `m1`, `m2`, optional `var1`, `degree`, and `similarity`.
- `structural_context` and `gate_context` are materialized per forward pass, so trainable prompt projection/gate behavior is unchanged.
- Cache diagnostics are written to `metrics.json` and `summary.csv` via `structural_cache_hit`.

Smoke command:

```bash
.venv/bin/python -m experiments.run_gp2f_prompt \
  --config configs/gp2f_prompt_p0.yaml \
  --target_dataset PubMed \
  --prompt_variant full_p0 \
  --route_budget 0.10 \
  --seeds 0 \
  --epochs 3 \
  --eval_every 1 \
  --base_checkpoint_root outputs/prompt_crossdomain_noprompt/PubMed/20260604_095409 \
  --freeze_base \
  --output_dir outputs/prompt_cache_smoke
```

Cache smoke result:

| Dataset | Output Dir | Cache Used | Cache Bytes | Cache Hit |
| --- | --- | ---: | ---: | ---: |
| PubMed | `outputs/prompt_cache_smoke/PubMed/20260605_210312` | 1 | 452386848 | 1.0 |

Strong activation probe command:

```bash
for variant in param_control full_p0; do
  .venv/bin/python -m experiments.run_gp2f_prompt \
    --config configs/gp2f_prompt_p0.yaml \
    --target_dataset Actor \
    --prompt_variant "${variant}" \
    --route_budget 0.35 \
    --gamma_init 0.05 \
    --gamma_max 0.30 \
    --lambda_budget 0.10 \
    --lambda_message_norm 0.05 \
    --message_norm_target 0.20 \
    --null_bias_init 0.0 \
    --prompt_lr 0.005 \
    --seeds 0,1,2,3,4 \
    --eval_every 5 \
    --base_checkpoint_root outputs/prompt_crossdomain_noprompt/Actor/20260604_100848 \
    --freeze_base \
    --output_dir outputs/prompt_activation_probe
done
```

Shared setting:

- Dataset: `Actor`
- Seeds: `0,1,2,3,4`
- Base initialization: seed-matched NoPrompt best checkpoint from `outputs/prompt_crossdomain_noprompt/Actor/20260604_100848`
- Base model: frozen
- Prompt budget: fixed `route_budget=0.35`
- Stronger activation: `gamma_init=0.05`, `gamma_max=0.30`, `null_bias_init=0.0`, weaker budget/message regularization

Results:

| Variant | Output Dir | Best Test Acc | Best Test Macro-F1 | Final Test Acc | Final Test Macro-F1 |
| --- | --- | ---: | ---: | ---: | ---: |
| `param_control` | `outputs/prompt_activation_probe/Actor/20260605_210430` | 19.13+-4.98 | 11.46+-3.37 | 19.27+-4.82 | 11.60+-3.19 |
| `full_p0` | `outputs/prompt_activation_probe/Actor/20260605_210502` | 19.17+-4.96 | 11.58+-3.27 | 19.17+-4.69 | 12.32+-2.98 |

Best-checkpoint diagnostics:

| Variant | Best Epoch | Route sem/struct/null | Prompt Norm | Gamma | Cache Hit | Semantic Margin | Structural Sim |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: |
| `param_control` | 4.6 | 0.321/0.322/0.357 | 0.0450 | 0.0510 | 0.0 | 0.0000 | 0.0000 |
| `full_p0` | 12.4 | 0.267/0.269/0.464 | 0.0542 | 0.0528 | 1.0 | 0.1780 | 0.2718 |

Interpretation:

- The stronger activation setting works mechanically: prompt norm and gamma are no longer stuck near zero, and best epoch is no longer always epoch 1.
- The improvement over `param_control` is still tiny: about `+0.04` best accuracy and `+0.12` best macro-F1 on Actor.
- `full_p0` uses semantic/structural evidence and the cache is active, but the evidence does not translate into a reliable advantage over a parameter-matched residual control.
- This weakens the hypothesis that P0 only failed because it was too conservative. The current residual feature prompt is likely under-expressive for the target adaptation problem.
- Next method step should move toward a separate P1 module: candidate pool / edge or message-channel prompt, with strict baseline isolation from P0 and GP2F.

## 2026-06-05 23:57 CST - P1 Heterophilous Quick Matrix

Purpose: run a first heterophilous sanity matrix for explicit prompt graph adaptation. This checks whether P1 prompt nodes and prompt edges improve over the CE-only dual-branch NoPrompt baseline on heterophilous datasets.

Command templates:

```bash
for dataset in Actor chameleon squirrel; do
  .venv/bin/python -m experiments.run_gp2f_prompt_graph \
    --config configs/gp2f_prompt_p1.yaml \
    --target_dataset "$dataset" \
    --prompt_variant noprompt \
    --epochs 100 \
    --seeds 0,1,2 \
    --eval_every 5 \
    --output_dir outputs/hetero_prompt_graph_eval/noprompt
done

for dataset in Actor chameleon squirrel; do
  .venv/bin/python -m experiments.run_gp2f_prompt_graph \
    --config configs/gp2f_prompt_p1.yaml \
    --target_dataset "$dataset" \
    --prompt_variant p1_graph \
    --epochs 100 \
    --seeds 0,1,2 \
    --eval_every 5 \
    --output_dir outputs/hetero_prompt_graph_eval/p1_graph
done

for dataset in Actor chameleon squirrel; do
  .venv/bin/python -m experiments.run_gp2f_prompt_graph \
    --config configs/gp2f_prompt_p1.yaml \
    --target_dataset "$dataset" \
    --prompt_variant p1_graph \
    --enable_capacity_routing \
    --epochs 100 \
    --seeds 0,1,2 \
    --eval_every 5 \
    --output_dir outputs/hetero_prompt_graph_eval/p1_capacity
done
```

Shared setting:

- Datasets: `Actor`, `chameleon`, `squirrel`
- Seeds: `0,1,2`
- Loss: CE-only plus prompt graph regularizers
- P1 pool: structural unreliability pool with `rho=0.20`
- Prompt graph: `8` prompt nodes, `topk_prompt_per_node=2`
- Edge warmup: `edge_scale_init=0.03`, `edge_scale_max=0.50`, warmup over `20` epochs
- Evaluation interval: `eval_every=5`

Results:

| Dataset | Variant | Output Dir | Best Test Acc | Best Test Macro-F1 | Final Test Acc | Final Test Macro-F1 |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| Actor | `noprompt` | `outputs/hetero_prompt_graph_eval/noprompt/Actor/20260605_234010` | 18.78+-6.74 | 12.17+-4.44 | 20.12+-2.62 | 18.43+-1.57 |
| Actor | `p1_graph` | `outputs/hetero_prompt_graph_eval/p1_graph/Actor/20260605_234205` | 18.83+-6.75 | 12.19+-4.45 | 20.13+-2.65 | 18.44+-1.60 |
| Actor | `p1_capacity` | `outputs/hetero_prompt_graph_eval/p1_capacity/Actor/20260605_234908` | 18.81+-6.74 | 12.18+-4.46 | 20.13+-2.64 | 18.43+-1.59 |
| chameleon | `noprompt` | `outputs/hetero_prompt_graph_eval/noprompt/chameleon/20260605_234022` | 28.61+-3.86 | 24.27+-9.33 | 28.05+-5.42 | 26.71+-5.44 |
| chameleon | `p1_graph` | `outputs/hetero_prompt_graph_eval/p1_graph/chameleon/20260605_234316` | 28.72+-3.95 | 24.39+-9.44 | 28.12+-5.48 | 26.81+-5.51 |
| chameleon | `p1_capacity` | `outputs/hetero_prompt_graph_eval/p1_capacity/chameleon/20260605_235009` | 28.70+-3.96 | 24.37+-9.42 | 28.18+-5.62 | 26.90+-5.66 |
| squirrel | `noprompt` | `outputs/hetero_prompt_graph_eval/noprompt/squirrel/20260605_234038` | 21.57+-1.10 | 13.60+-3.81 | 21.29+-1.12 | 20.22+-1.45 |
| squirrel | `p1_graph` | `outputs/hetero_prompt_graph_eval/p1_graph/squirrel/20260605_234422` | 21.55+-1.08 | 13.58+-3.78 | 21.36+-1.15 | 20.27+-1.48 |
| squirrel | `p1_capacity` | `outputs/hetero_prompt_graph_eval/p1_capacity/squirrel/20260605_235122` | 21.57+-1.10 | 13.60+-3.81 | 21.32+-1.15 | 20.24+-1.50 |

Final-epoch P1 diagnostics:

| Dataset | Variant | Prompt Edges | Pool Ratio | Mean Prompt Edge Weight | Edge Scale | Usage Entropy | Full Usage Entropy | Branch Cosine | Init Logit Delta |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Actor | `p1_graph` | 6158.7 | 0.2026 | 0.0142 | 0.0283 | 0.4064 | 0.9998 | 0.9993 | 0.0012 |
| Actor | `p1_capacity` | 6156.0 | 0.2025 | 0.0142 | 0.0283 | 0.9232 | 0.9997 | 0.9993 | 0.0011 |
| chameleon | `p1_graph` | 1901.3 | 0.2088 | 0.0133 | 0.0266 | 0.5634 | 0.9990 | 0.9564 | 0.0010 |
| chameleon | `p1_capacity` | 1902.7 | 0.2089 | 0.0132 | 0.0263 | 0.9372 | 0.9989 | 0.9571 | 0.0009 |
| squirrel | `p1_graph` | 4246.7 | 0.2041 | 0.0138 | 0.0276 | 0.4081 | 0.9995 | 0.9953 | 0.0009 |
| squirrel | `p1_capacity` | 4246.7 | 0.2041 | 0.0138 | 0.0276 | 0.9347 | 0.9989 | 0.9951 | 0.0009 |

Interpretation:

- P1 is mechanically active: the pool ratio is near `rho=0.20`, prompt edges are generated, prompt edge weights are nonzero, and init logit deltas stay small.
- P1 does not yet show meaningful improvement over NoPrompt on the tested heterophilous datasets. The best-accuracy gains are about `+0.05` on Actor, `+0.11` on chameleon, and `-0.02` on squirrel, all well inside seed variance.
- Capacity routing increases prompt usage entropy substantially, but it does not improve accuracy. This suggests that prompt-node load balancing is not the current bottleneck.
- Prompt edge weights remain small and branch cosine remains very high on Actor/squirrel. The adapted branch is only weakly changed by prompt edges, so P1 is stable but under-expressive.
- The current P1 result supports moving toward a stronger prompt-aware adapted branch or edge-type-specific message passing. More sweeping over `rho` or capacity is unlikely to be the highest-value next step.

## 2026-06-06 15:38 CST - P2 Prompt-Aware Quick Matrix

Purpose: compare NoPrompt, P1 explicit prompt graph, and P2 prompt-aware adapted branch under the same quick heterophilous setting. P2 adds edge-type-aware prompt message channels while keeping the frozen branch on the original graph.

Command templates:

```bash
for variant in noprompt p1_graph p2_prompt_aware; do
  for dataset in Actor chameleon squirrel; do
    .venv/bin/python -m experiments.run_gp2f_prompt_p2 \
      --config configs/gp2f_prompt_p2.yaml \
      --target_dataset "$dataset" \
      --prompt_variant "$variant" \
      --epochs 100 \
      --seeds 0,1,2 \
      --eval_every 5 \
      --output_dir outputs/prompt_p2_quick
  done
done
```

Shared setting:

- Datasets: `Actor`, `chameleon`, `squirrel`
- Seeds: `0,1,2`
- Loss: CE-only plus prompt graph regularizers
- P1/P2 pool: structural unreliability pool with `rho=0.20`
- Prompt graph: `8` prompt nodes, `topk_prompt_per_node=2`
- Edge warmup: `edge_scale_init=0.03`, `edge_scale_max=0.50`, warmup over `20` epochs
- P2 prompt-aware gate init: `0.01`
- Evaluation interval: `eval_every=5`

Results:

| Dataset | Variant | Output Dir | Best Test Acc | Best Test Macro-F1 | Final Test Acc | Final Test Macro-F1 |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| Actor | `noprompt` | `outputs/prompt_p2_quick/Actor/20260606_153807` | 18.78+-6.74 | 12.17+-4.44 | 20.12+-2.62 | 18.43+-1.57 |
| Actor | `p1_graph` | `outputs/prompt_p2_quick/Actor/20260606_153955` | 18.83+-6.75 | 12.19+-4.45 | 20.12+-2.65 | 18.43+-1.59 |
| Actor | `p2_prompt_aware` | `outputs/prompt_p2_quick/Actor/20260606_154900` | 18.79+-6.75 | 12.18+-4.45 | 20.11+-2.61 | 18.43+-1.54 |
| chameleon | `noprompt` | `outputs/prompt_p2_quick/chameleon/20260606_153820` | 28.61+-3.86 | 24.27+-9.33 | 28.05+-5.42 | 26.71+-5.44 |
| chameleon | `p1_graph` | `outputs/prompt_p2_quick/chameleon/20260606_154113` | 28.72+-3.95 | 24.39+-9.44 | 28.13+-5.50 | 26.82+-5.52 |
| chameleon | `p2_prompt_aware` | `outputs/prompt_p2_quick/chameleon/20260606_155101` | 28.61+-3.81 | 24.23+-9.28 | 28.10+-5.47 | 26.76+-5.48 |
| squirrel | `noprompt` | `outputs/prompt_p2_quick/squirrel/20260606_153837` | 21.57+-1.10 | 13.60+-3.81 | 21.29+-1.12 | 20.22+-1.45 |
| squirrel | `p1_graph` | `outputs/prompt_p2_quick/squirrel/20260606_154251` | 21.55+-1.08 | 13.58+-3.78 | 21.36+-1.15 | 20.27+-1.48 |
| squirrel | `p2_prompt_aware` | `outputs/prompt_p2_quick/squirrel/20260606_155300` | 21.57+-1.12 | 13.61+-3.82 | 21.29+-1.11 | 20.21+-1.43 |

P2 best-checkpoint diagnostics:

| Dataset | Pool Ratio | Prompt Edges | Mean Prompt Edge Weight | Prompt Msg Norm | Adapted Delta Norm | Branch Cosine | Init Logit Delta |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Actor | 20.29 | 6166.7 | 0.0064 | 0.00068 | 0.00068 | 0.99997 | 0.00040 |
| chameleon | 20.85 | 1898.7 | 0.0120 | 0.00092 | 0.00092 | 0.99689 | 0.00035 |
| squirrel | 20.41 | 4246.7 | 0.0082 | 0.00071 | 0.00071 | 0.99996 | 0.00050 |

Interpretation:

- P2 is mechanically connected: pool size, prompt edge counts, edge weights, and prompt-aware metrics are populated.
- P2 does not improve over P1 or NoPrompt in this quick matrix. The best-accuracy differences are effectively zero and far below seed variance.
- The prompt-aware gate and edge weights are still conservative. `prompt_msg_norm` is around `7e-4` to `9e-4`, and branch cosine remains close to `1.0`, so the adapted branch is barely changed.
- Since P1 and P2 both remain near NoPrompt, the immediate bottleneck is unlikely to be only "missing edge-type-specific message passing." More likely bottlenecks are prompt signal strength, pool/connection quality, and lack of an auxiliary objective that makes prompt nodes learn useful roles.
- Next step should be a focused activation/connection ablation before adding more architecture: increase prompt-aware gate or edge scale, compare `rho` values, test random vs structural pool again under P2, and add diagnostics for prompt message contribution by pool/non-pool nodes.

## 2026-06-06 16:55 CST - P2 Strength Optimization Probe

Purpose: verify whether the previous P2 failure was caused by prompt-aware messages being too weak. This probe strengthens prompt-aware message injection while keeping the same CE-only protocol and structural unreliability pool.

Implementation changes:

- Added configurable prompt-aware message normalization: `weighted_mean`, `weighted_sum`, and `degree_mean`.
- Added `prompt_aware.message_scale` so prompt-aware updates can be amplified without changing graph construction.
- Added direction-specific gate initialization for node-to-prompt and prompt-to-node channels.
- Trained P2 prompt-aware parameters in the prompt optimizer group instead of the base optimizer group.
- Added diagnostics for raw prompt update norm, node-to-prompt update norm, prompt-to-original update norm, direction-specific gate means, message scale, and message norm.
- Added `p2_strength` variant and `configs/gp2f_prompt_p2_strength.yaml`.

Validation:

```bash
.venv/bin/python -m pytest tests/test_prompt_aware_gp2f.py tests/test_prompt_graph_module.py -q
.venv/bin/python -m pytest -q
```

Results:

- `tests/test_prompt_aware_gp2f.py tests/test_prompt_graph_module.py`: `21 passed`
- Full test suite: `56 passed`

Smoke command:

```bash
.venv/bin/python -m experiments.run_gp2f_prompt_p2 \
  --config configs/gp2f_prompt_p2_strength.yaml \
  --target_dataset Cora \
  --prompt_variant p2_strength \
  --epochs 3 \
  --seeds 0 \
  --eval_every 1 \
  --output_dir outputs/p2_optimized_smoke
```

Smoke output:

- Output dir: `outputs/p2_optimized_smoke/Cora/20260606_161923`
- `prompt_msg_norm`: `0.0226`
- `prompt_to_original_update_norm`: `0.0226`
- `node_to_prompt_update_norm`: `0.0439`
- `raw_prompt_update_norm`: `0.0045`
- `prompt_gate_mean`: `0.0501`
- `prompt_message_scale`: `5.0`
- `branch_cosine`: `0.9466`
- `init_logit_delta_full_edge_scale`: `0.0061`

This confirms the optimized P2 no longer has the previous "almost invisible prompt message" problem.

Probe commands:

```bash
.venv/bin/python -m experiments.run_gp2f_prompt_p2 \
  --config configs/gp2f_prompt_p2_strength.yaml \
  --target_dataset Actor \
  --prompt_variant p2_strength \
  --epochs 100 \
  --seeds 0,1,2 \
  --eval_every 5 \
  --output_dir outputs/p2_optimized_probe

.venv/bin/python -m experiments.run_gp2f_prompt_p2 \
  --config configs/gp2f_prompt_p2_strength.yaml \
  --target_dataset chameleon \
  --prompt_variant p2_strength \
  --epochs 100 \
  --seeds 0,1,2 \
  --eval_every 5 \
  --output_dir outputs/p2_optimized_probe
```

Results:

| Dataset | Variant | Output Dir | Best Test Acc | Best Test Macro-F1 | Final Test Acc | Final Test Macro-F1 |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| Actor | `p2_strength` | `outputs/p2_optimized_probe/Actor/20260606_162022` | 21.63+-4.64 | 14.66+-4.44 | 21.36+-1.46 | 19.20+-1.65 |
| chameleon | `p2_strength` | `outputs/p2_optimized_probe/chameleon/20260606_164552` | 26.56+-4.40 | 23.67+-1.15 | 24.39+-2.38 | 22.59+-1.91 |

Reference from the previous quick matrix:

| Dataset | Reference Variant | Best Test Acc | Best Test Macro-F1 | Final Test Acc | Final Test Macro-F1 |
| --- | --- | ---: | ---: | ---: | ---: |
| Actor | `noprompt` | 18.78+-6.74 | 12.17+-4.44 | 20.12+-2.62 | 18.43+-1.57 |
| Actor | old `p2_prompt_aware` | 18.79+-6.75 | 12.18+-4.45 | 20.11+-2.61 | 18.43+-1.54 |
| chameleon | `noprompt` | 28.61+-3.86 | 24.27+-9.33 | 28.05+-5.42 | 26.71+-5.44 |
| chameleon | old `p2_prompt_aware` | 28.61+-3.81 | 24.23+-9.28 | 28.10+-5.47 | 26.76+-5.48 |

Best-checkpoint diagnostics:

| Dataset | Seed | Best Epoch | Test Acc | Val Acc | Mean Edge Weight | Prompt Msg Norm | Prompt-to-Original Update | Node-to-Prompt Update | Raw Update | Gate Mean | Message Scale | Branch Cosine |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Actor | 0 | 45 | 0.2199 | 0.2333 | 0.0283 | 1.1445 | 1.1455 | 0.2359 | 0.2289 | 0.0571 | 5.0 | 0.8059 |
| Actor | 1 | 5 | 0.1682 | 0.2267 | 0.0063 | 0.0373 | 0.0373 | 0.0492 | 0.0074 | 0.0507 | 5.0 | 0.9335 |
| Actor | 2 | 10 | 0.2609 | 0.2667 | 0.0129 | 0.0884 | 0.0885 | 0.0548 | 0.0177 | 0.0514 | 5.0 | 0.8893 |
| chameleon | 0 | 85 | 0.2550 | 0.3133 | 0.0315 | 1.2169 | 1.2207 | 0.1207 | 0.2434 | 0.0630 | 5.0 | 0.7469 |
| chameleon | 1 | 15 | 0.3140 | 0.2533 | 0.0196 | 0.2359 | 0.2365 | 0.0885 | 0.0472 | 0.0522 | 5.0 | 0.8304 |
| chameleon | 2 | 70 | 0.2279 | 0.2467 | 0.0303 | 1.1414 | 1.1449 | 0.1481 | 0.2283 | 0.0604 | 5.0 | 0.7125 |

Interpretation:

- The optimization succeeds mechanically. P2-strength strongly changes the adapted branch, with `prompt_msg_norm` increasing from about `7e-4` in old P2 to as high as `1.2`, and branch cosine dropping from nearly `1.0` to roughly `0.71-0.93`.
- Actor shows a small but visible quick-probe improvement over NoPrompt and old P2. However, the seed variance is still high, so this is not yet a reliable result.
- chameleon degrades relative to NoPrompt. This means the bottleneck is no longer only prompt strength. Stronger prompt-aware messages can become harmful when the pool or prompt-node assignment is not sufficiently reliable.
- The next useful optimization should focus on connection quality and supervision for prompt node roles, not only message strength. Candidate next steps are: validation-only strength selection, random-pool versus structural-pool under P2-strength, adaptive pool thresholds, prompt-node diversity/role regularization, and a stricter comparison where only pool nodes receive prompt-aware updates.

## 2026-06-06 17:15 CST - P2 Pool and Strength Validation

Purpose: validate two hypotheses from the P2-strength probe:

1. Whether the structural unreliability pool is better than a random pool under the same P2-strength setting.
2. Whether fixed strong prompt injection (`message_scale=5.0`) is too aggressive on chameleon.

Implementation change:

- Added `p2_strength_random_pool` as a runner variant. It keeps the same P2-strength settings but changes only `prompt_graph.pool_strategy` from `structural` to `random`.

Validation:

```bash
.venv/bin/python -m pytest tests/test_prompt_aware_gp2f.py tests/test_prompt_graph_module.py -q
```

Result: `21 passed`.

Pool validation commands:

```bash
.venv/bin/python -m experiments.run_gp2f_prompt_p2 \
  --config configs/gp2f_prompt_p2_strength.yaml \
  --target_dataset chameleon \
  --prompt_variant p2_strength_random_pool \
  --epochs 100 \
  --seeds 0,1,2 \
  --eval_every 5 \
  --output_dir outputs/p2_validation_pool

.venv/bin/python -m experiments.run_gp2f_prompt_p2 \
  --config configs/gp2f_prompt_p2_strength.yaml \
  --target_dataset Actor \
  --prompt_variant p2_strength_random_pool \
  --epochs 100 \
  --seeds 0,1,2 \
  --eval_every 5 \
  --output_dir outputs/p2_validation_pool
```

Pool validation results:

| Dataset | Variant | Output Dir | Best Test Acc | Best Test Macro-F1 | Final Test Acc | Final Test Macro-F1 |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| Actor | `p2_strength` structural pool | `outputs/p2_optimized_probe/Actor/20260606_162022` | 21.63+-4.64 | 14.66+-4.44 | 21.36+-1.46 | 19.20+-1.65 |
| Actor | `p2_strength_random_pool` | `outputs/p2_validation_pool/Actor/20260606_170539` | 19.42+-6.69 | 15.19+-3.42 | 21.49+-1.89 | 18.85+-2.21 |
| chameleon | `p2_strength` structural pool | `outputs/p2_optimized_probe/chameleon/20260606_164552` | 26.56+-4.40 | 23.67+-1.15 | 24.39+-2.38 | 22.59+-1.91 |
| chameleon | `p2_strength_random_pool` | `outputs/p2_validation_pool/chameleon/20260606_170442` | 26.01+-3.42 | 19.28+-6.00 | 24.28+-2.57 | 22.59+-1.56 |

Strength validation commands:

```bash
.venv/bin/python -m experiments.run_gp2f_prompt_p2 \
  --config configs/gp2f_prompt_p2_strength.yaml \
  --target_dataset chameleon \
  --prompt_variant p2_strength \
  --prompt_message_scale 1.0 \
  --epochs 100 \
  --seeds 0,1,2 \
  --eval_every 5 \
  --output_dir outputs/p2_validation_strength

.venv/bin/python -m experiments.run_gp2f_prompt_p2 \
  --config configs/gp2f_prompt_p2_strength.yaml \
  --target_dataset chameleon \
  --prompt_variant p2_strength \
  --prompt_message_scale 2.0 \
  --epochs 100 \
  --seeds 0,1,2 \
  --eval_every 5 \
  --output_dir outputs/p2_validation_strength
```

Strength validation results on chameleon:

| Message Scale | Output Dir | Best Test Acc | Best Test Macro-F1 | Final Test Acc | Final Test Macro-F1 |
| ---: | --- | ---: | ---: | ---: | ---: |
| 1.0 | `outputs/p2_validation_strength/chameleon/20260606_170714` | 27.97+-3.29 | 22.21+-7.50 | 26.69+-4.61 | 24.50+-4.35 |
| 2.0 | `outputs/p2_validation_strength/chameleon/20260606_170848` | 26.53+-2.63 | 20.49+-6.13 | 24.50+-3.50 | 22.73+-2.71 |
| 5.0 | `outputs/p2_optimized_probe/chameleon/20260606_164552` | 26.56+-4.40 | 23.67+-1.15 | 24.39+-2.38 | 22.59+-1.91 |

Reference NoPrompt on chameleon:

| Variant | Best Test Acc | Best Test Macro-F1 | Final Test Acc | Final Test Macro-F1 |
| --- | ---: | ---: | ---: | ---: |
| `noprompt` | 28.61+-3.86 | 24.27+-9.33 | 28.05+-5.42 | 26.71+-5.44 |

Key diagnostics:

| Setting | Seed | Best Epoch | Test Acc | Val Acc | Prompt Msg Norm | Branch Cosine |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| chameleon structural, scale 1.0 | 0 | 50 | 0.2892 | 0.3000 | 0.2920 | 0.8880 |
| chameleon structural, scale 1.0 | 1 | 30 | 0.3069 | 0.3000 | 0.1071 | 0.9233 |
| chameleon structural, scale 1.0 | 2 | 1 | 0.2431 | 0.2133 | 0.0046 | 0.9977 |
| Actor random pool, scale 5.0 | 0 | 10 | 0.1351 | 0.2333 | 0.0740 | 0.9372 |
| Actor random pool, scale 5.0 | 1 | 20 | 0.1806 | 0.2267 | 0.2125 | 0.9004 |
| Actor random pool, scale 5.0 | 2 | 10 | 0.2668 | 0.2733 | 0.0601 | 0.9558 |

Interpretation:

- On Actor, the structural pool is meaningfully better than the random pool in best-test accuracy. This supports keeping the structural unreliability pool as a real design component, at least on Actor.
- On chameleon, structural and random pools are both below NoPrompt. The pool rule alone does not solve the instability there.
- On chameleon, reducing `message_scale` from `5.0` to `1.0` improves both best and final metrics. This confirms fixed strong prompt injection is too aggressive for some heterophilous datasets.
- Even the best chameleon P2 setting in this probe (`message_scale=1.0`) still trails NoPrompt, so the next step should not be another global strength increase. P2 needs validation-only strength selection plus a better rule for when prompt messages are allowed to affect original nodes.

Actionable next steps:

- Implement a validation-only grid for `prompt_message_scale`, e.g. `[0.0, 0.5, 1.0, 2.0, 5.0]`, where `0.0` is effectively a P2 graph/control with no prompt-aware message.
- Add a `pool_only_prompt_update` option so prompt-to-original updates are applied only to pool nodes. This should reduce damage to reliable nodes.
- Add a prompt-role regularizer or auxiliary objective, because current prompt nodes can strongly change representations but are not guaranteed to learn useful roles.
- Keep `p2_strength_random_pool` as a sanity baseline in future P2 tables.

## 2026-06-06 17:50 CST - P2 Pool-Only and Message-Scale Grid Optimization

Goal:

- Reduce P2 damage on chameleon by preventing prompt messages from directly updating non-pool original nodes.
- Add validation-only `prompt_message_scale` selection so P2 can choose a conservative prompt strength per split.
- Include `message_scale=0.0` as a safe fallback candidate. This keeps the prompt graph scaffold but injects zero prompt-edge message into original-node representations.

Code changes:

- Added `pool_only_prompt_update` to `PromptAwareGP2F`.
- Passed the P1/P2 `pool_mask` into the prompt-aware adapted branch.
- Added validation-only `message_scale_grid` execution and per-seed candidate selection.
- Updated P2 configs to default to:
  - `message_scale_grid: [0.0, 0.5, 1.0, 2.0, 5.0]`
  - `pool_only_prompt_update: true`
- Added a unit test that verifies `message_scale=0.0` matches the NoPrompt adapted path for original nodes.

Validation commands:

```bash
.venv/bin/python -m experiments.run_gp2f_prompt_p2 \
  --config configs/gp2f_prompt_p2_strength.yaml \
  --target_dataset chameleon \
  --prompt_variant p2_strength \
  --prompt_message_scale_grid 0.5,1.0,2.0,5.0 \
  --pool_only_prompt_update \
  --epochs 100 \
  --seeds 0,1,2 \
  --eval_every 5 \
  --output_dir outputs/p2_pool_only_grid
```

Result:

| Dataset | Variant | Output Dir | Best Test Acc | Best Test Macro-F1 | Final Test Acc | Final Test Macro-F1 |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| chameleon | P2 strength + pool-only + scale grid `[0.5,1,2,5]` | `outputs/p2_pool_only_grid/chameleon/20260606_173833` | 27.37+-3.17 | 22.80+-8.30 | 26.78+-3.86 | 24.20+-4.65 |

Selected scale by seed:

| Seed | Selected Scale | Best Epoch | Val Acc | Test Acc | Test Macro-F1 | Prompt Msg Norm | Branch Cosine |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 0.5 | 70 | 0.3467 | 0.3040 | 0.2908 | 0.1550 | 0.9369 |
| 1 | 2.0 | 55 | 0.3000 | 0.2764 | 0.2593 | 0.4786 | 0.8837 |
| 2 | 5.0 | 1 | 0.2267 | 0.2407 | 0.1339 | 0.0287 | 0.9543 |

Candidate details:

| Seed | Scale | Val Acc | Test Acc | Test Macro-F1 |
| ---: | ---: | ---: | ---: | ---: |
| 0 | 0.5 | 0.3467 | 0.3040 | 0.2908 |
| 0 | 1.0 | 0.3000 | 0.2926 | 0.2694 |
| 0 | 2.0 | 0.3133 | 0.2636 | 0.2345 |
| 0 | 5.0 | 0.3133 | 0.2688 | 0.2597 |
| 1 | 0.5 | 0.2867 | 0.3121 | 0.2513 |
| 1 | 1.0 | 0.2933 | 0.3078 | 0.2630 |
| 1 | 2.0 | 0.3000 | 0.2764 | 0.2593 |
| 1 | 5.0 | 0.2800 | 0.2645 | 0.2323 |
| 2 | 0.5 | 0.2133 | 0.2431 | 0.1357 |
| 2 | 1.0 | 0.2133 | 0.2431 | 0.1355 |
| 2 | 2.0 | 0.2200 | 0.2412 | 0.1344 |
| 2 | 5.0 | 0.2267 | 0.2407 | 0.1339 |

Reference:

| Dataset | Variant | Best Test Acc | Best Test Macro-F1 | Final Test Acc | Final Test Macro-F1 |
| --- | --- | ---: | ---: | ---: | ---: |
| chameleon | NoPrompt | 28.61+-3.86 | 24.27+-9.33 | 28.05+-5.42 | 26.71+-5.44 |
| chameleon | P2 strength, scale 1.0, no pool-only | 27.97+-3.29 | 22.21+-7.50 | 26.69+-4.61 | 24.50+-4.35 |

Smoke test for default grid:

```bash
.venv/bin/python -m experiments.run_gp2f_prompt_p2 \
  --config configs/gp2f_prompt_p2_strength.yaml \
  --target_dataset Cora \
  --prompt_variant p2_strength \
  --epochs 2 \
  --seeds 0 \
  --eval_every 1 \
  --output_dir outputs/p2_optimization_smoke_default
```

Smoke result:

- Output: `outputs/p2_optimization_smoke_default/Cora/20260606_174718`
- Grid: `[0.0, 0.5, 1.0, 2.0, 5.0]`
- Selected scale: `0.0`
- `message_scale_selection_enabled: true`
- At `scale=0.0`, `prompt_msg_norm=0.0`, `adapted_branch_delta_norm=0.0`, `branch_cosine=1.0`.

Tests:

```bash
.venv/bin/python -m pytest tests/test_prompt_aware_gp2f.py tests/test_prompt_graph_module.py -q
.venv/bin/python -m pytest -q
```

Result:

- P2 tests: `23 passed`
- Full suite: `58 passed`

Interpretation:

- Pool-only update is technically correct and now test-protected, but it did not by itself make chameleon beat NoPrompt.
- The chameleon result remains below NoPrompt, so the current P2 prompt messages are still not reliably useful on this dataset.
- Adding `0.0` to the grid is necessary for stable reporting. Without it, validation-only selection is forced to choose a nonzero prompt even when all prompt candidates are harmful.
- Seed 2 remains the main failure case: it selects a late scale by validation, but the best epoch is epoch 1 and macro-F1 is very low. This suggests the prompt module is not learning stable roles for that split.

Next optimization direction:

- Keep the conservative default grid with `0.0`.
- Run the same default-grid protocol on Actor, where structural pool previously beat random pool, to check whether the new safety fallback preserves gains.
- For chameleon, the next real module change should be role/usage regularization or a better pool scoring objective, not stronger prompt injection.

## 2026-06-06 18:10 CST - P2 Default-Grid Validation on Actor

Goal:

- Check whether the safer P2 defaults still preserve the positive signal previously seen on Actor.
- Validate `message_scale_grid: [0.0, 0.5, 1.0, 2.0, 5.0]` with `pool_only_prompt_update: true`.

Command:

```bash
.venv/bin/python -m experiments.run_gp2f_prompt_p2 \
  --config configs/gp2f_prompt_p2_strength.yaml \
  --target_dataset Actor \
  --prompt_variant p2_strength \
  --epochs 100 \
  --seeds 0,1,2 \
  --eval_every 5 \
  --output_dir outputs/p2_default_grid_validation
```

Result:

| Dataset | Variant | Output Dir | Best Test Acc | Best Test Macro-F1 | Final Test Acc | Final Test Macro-F1 |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| Actor | P2 strength + pool-only + default scale grid | `outputs/p2_default_grid_validation/Actor/20260606_174928` | 21.92+-3.59 | 17.00+-0.17 | 22.09+-0.94 | 17.47+-3.08 |

Selected scale by seed:

| Seed | Selected Scale | Best Epoch | Val Acc | Test Acc | Test Macro-F1 | Prompt Msg Norm | Branch Cosine |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 5.0 | 20 | 0.2267 | 0.1921 | 0.1693 | 0.4597 | 0.9034 |
| 1 | 2.0 | 65 | 0.2533 | 0.2057 | 0.1686 | 0.6605 | 0.8899 |
| 2 | 5.0 | 10 | 0.2867 | 0.2599 | 0.1719 | 0.0884 | 0.9283 |

Candidate details:

| Seed | Scale | Val Acc | Test Acc | Test Macro-F1 |
| ---: | ---: | ---: | ---: | ---: |
| 0 | 0.0 | 0.2200 | 0.1320 | 0.1091 |
| 0 | 0.5 | 0.2200 | 0.1325 | 0.1093 |
| 0 | 1.0 | 0.2200 | 0.1324 | 0.1090 |
| 0 | 2.0 | 0.2200 | 0.1329 | 0.1087 |
| 0 | 5.0 | 0.2267 | 0.1921 | 0.1693 |
| 1 | 0.0 | 0.2133 | 0.1688 | 0.0849 |
| 1 | 0.5 | 0.2133 | 0.1688 | 0.0848 |
| 1 | 1.0 | 0.2067 | 0.1690 | 0.0854 |
| 1 | 2.0 | 0.2533 | 0.2057 | 0.1686 |
| 1 | 5.0 | 0.2200 | 0.1685 | 0.0933 |
| 2 | 0.0 | 0.2600 | 0.2626 | 0.1710 |
| 2 | 0.5 | 0.2600 | 0.2638 | 0.1716 |
| 2 | 1.0 | 0.2600 | 0.2633 | 0.1714 |
| 2 | 2.0 | 0.2733 | 0.2626 | 0.1700 |
| 2 | 5.0 | 0.2867 | 0.2599 | 0.1719 |

Reference:

| Dataset | Variant | Best Test Acc | Best Test Macro-F1 | Final Test Acc | Final Test Macro-F1 |
| --- | --- | ---: | ---: | ---: | ---: |
| Actor | P2 strength structural pool, fixed scale 5.0 | 21.63+-4.64 | 14.66+-4.44 | 21.36+-1.46 | 19.20+-1.65 |
| Actor | P2 strength random pool, fixed scale 5.0 | 19.42+-6.69 | 15.19+-3.42 | 21.49+-1.89 | 18.85+-2.21 |

Interpretation:

- Actor behaves differently from chameleon: validation selection preferred nonzero prompt messages for all three seeds.
- Compared with the earlier fixed-scale structural pool result, best-test accuracy improves slightly (`21.63` -> `21.92`) and Macro-F1 becomes more stable (`14.66+-4.44` -> `17.00+-0.17`).
- The improvement remains modest. P2 has a real signal on Actor, but the current prompt nodes still do not learn sufficiently discriminative roles.
- This supports keeping P2 as a meaningful graph-prompt branch, but the next optimization should add a role/usage objective rather than just changing global strength.

## 2026-06-06 18:08 CST - P2 Default-Grid Validation on squirrel

Goal:

- Check whether the safer P2 defaults improve squirrel.
- Use the same validation-only message scale grid as Actor/chameleon.

Command:

```bash
.venv/bin/python -m experiments.run_gp2f_prompt_p2 \
  --config configs/gp2f_prompt_p2_strength.yaml \
  --target_dataset squirrel \
  --prompt_variant p2_strength \
  --epochs 100 \
  --seeds 0,1,2 \
  --eval_every 5 \
  --output_dir outputs/p2_default_grid_validation
```

Result:

| Dataset | Variant | Output Dir | Best Test Acc | Best Test Macro-F1 | Final Test Acc | Final Test Macro-F1 |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| squirrel | P2 strength + pool-only + default scale grid | `outputs/p2_default_grid_validation/squirrel/20260606_180843` | 22.07+-1.78 | 14.61+-5.42 | 20.98+-0.60 | 19.43+-0.43 |

Selected scale by seed:

| Seed | Selected Scale | Best Epoch | Val Acc | Test Acc | Test Macro-F1 | Prompt Msg Norm | Branch Cosine |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 0.0 | 1 | 0.2067 | 0.2031 | 0.0992 | 0.0000 | 1.0000 |
| 1 | 0.0 | 10 | 0.2667 | 0.2203 | 0.1337 | 0.0000 | 1.0000 |
| 2 | 5.0 | 30 | 0.2733 | 0.2388 | 0.2055 | 0.7423 | 0.9073 |

Candidate details:

| Seed | Scale | Val Acc | Test Acc | Test Macro-F1 |
| ---: | ---: | ---: | ---: | ---: |
| 0 | 0.0 | 0.2067 | 0.2031 | 0.0992 |
| 0 | 0.5 | 0.2067 | 0.2031 | 0.0992 |
| 0 | 1.0 | 0.2067 | 0.2029 | 0.0991 |
| 0 | 2.0 | 0.2067 | 0.2027 | 0.0990 |
| 0 | 5.0 | 0.2067 | 0.2018 | 0.0994 |
| 1 | 0.0 | 0.2667 | 0.2203 | 0.1337 |
| 1 | 0.5 | 0.2667 | 0.2203 | 0.1337 |
| 1 | 1.0 | 0.2667 | 0.2207 | 0.1339 |
| 1 | 2.0 | 0.2667 | 0.2207 | 0.1345 |
| 1 | 5.0 | 0.2667 | 0.2113 | 0.1814 |
| 2 | 0.0 | 0.2533 | 0.2236 | 0.1752 |
| 2 | 0.5 | 0.2533 | 0.2240 | 0.1759 |
| 2 | 1.0 | 0.2600 | 0.2392 | 0.2111 |
| 2 | 2.0 | 0.2600 | 0.2348 | 0.1955 |
| 2 | 5.0 | 0.2733 | 0.2388 | 0.2055 |

Reference:

| Dataset | Variant | Best Test Acc | Best Test Macro-F1 | Final Test Acc | Final Test Macro-F1 |
| --- | --- | ---: | ---: | ---: | ---: |
| squirrel | NoPrompt quick | 21.57+-1.10 | 13.60+-3.81 | 21.29+-1.12 | 20.22+-1.45 |
| squirrel | P1 graph quick | 21.55+-1.08 | 13.58+-3.78 | 21.25+-1.07 | 20.13+-1.41 |
| squirrel | old P2 quick | 21.57+-1.12 | 13.61+-3.82 | 21.29+-1.11 | 20.21+-1.43 |

Interpretation:

- squirrel shows only a weak and unstable P2 signal.
- Seeds 0 and 1 select `message_scale=0.0`, so validation-only selection effectively falls back to NoPrompt for two thirds of the runs.
- The mean best-test improvement mainly comes from seed 2 selecting `message_scale=5.0`.
- Final accuracy and final Macro-F1 are not better than the NoPrompt quick reference, which means stronger prompt messages do not produce durable training improvement on squirrel.
- Current P2 should not be claimed as effective on squirrel yet. The next useful change should target better prompt role learning or pool scoring, not simply stronger message injection.

## 2026-06-06 18:45 CST - P2 Pool/Message Post-hoc Diagnostics

Goal:

- Diagnose why P2 prompt messages do not consistently improve performance.
- Load saved best checkpoints and compare the same checkpoint with prompt message enabled versus `message_scale=0.0`.
- Measure whether the pool covers NoPrompt errors, whether prompt messages fix or break pool nodes, and whether non-pool nodes are protected.

Implementation:

- Added `experiments/diagnose_prompt_graph.py`.
- The script writes per-seed `diagnostics.json` and `node_diagnostics.csv`.
- It does not train, select hyperparameters, or use test labels for model selection. Test labels are used only for reporting diagnostics.

Command:

```bash
.venv/bin/python -m experiments.diagnose_prompt_graph \
  --summary_path outputs/p2_default_grid_validation/Actor/20260606_174928/summary.json \
  --summary_path outputs/p2_pool_only_grid/chameleon/20260606_173833/summary.json \
  --summary_path outputs/p2_default_grid_validation/squirrel/20260606_180843/summary.json \
  --output_dir outputs/prompt_diagnostics/p2_pool_message
```

Aggregate result:

| Dataset | Prompt Off Test Acc | Prompt On Test Acc | Pool Off Acc | Pool On Acc | Non-pool Off Acc | Non-pool On Acc |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Actor | 0.2239 | 0.2192 | 0.2192 | 0.1955 | 0.2251 | 0.2251 |
| chameleon | 0.2732 | 0.2737 | 0.2576 | 0.2599 | 0.2773 | 0.2773 |
| squirrel | 0.2195 | 0.2207 | 0.2232 | 0.2295 | 0.2185 | 0.2185 |

Pool/message diagnostics:

| Dataset | Avg Pool Fixed | Avg Pool Broken | Pool Prediction Change | Pool Error Precision | Pool Error Recall | Pool Delta Norm |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Actor | 102.3 | 137.3 | 0.4120 | 0.7808 | 0.2009 | 3.5874 |
| chameleon | 28.3 | 27.3 | 0.3076 | 0.7424 | 0.2051 | 2.0038 |
| squirrel | 34.0 | 27.7 | 0.1929 | 0.7768 | 0.2000 | 2.4214 |

Per-seed highlights:

| Dataset | Seed | Selected Scale | Pool Acc Off | Pool Acc On | Fixed | Broken | Pool Change Ratio |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Actor | 0 | 5.0 | 0.1834 | 0.1373 | 109 | 177 | 0.5792 |
| Actor | 1 | 2.0 | 0.1986 | 0.1858 | 98 | 117 | 0.3500 |
| Actor | 2 | 5.0 | 0.2755 | 0.2634 | 100 | 118 | 0.3070 |
| chameleon | 0 | 0.5 | 0.3413 | 0.2983 | 11 | 29 | 0.1551 |
| chameleon | 1 | 2.0 | 0.2536 | 0.3033 | 58 | 37 | 0.5687 |
| chameleon | 2 | 5.0 | 0.1780 | 0.1780 | 16 | 16 | 0.1991 |
| squirrel | 0 | 0.0 | 0.2120 | 0.2120 | 0 | 0 | 0.0000 |
| squirrel | 1 | 0.0 | 0.2664 | 0.2664 | 0 | 0 | 0.0000 |
| squirrel | 2 | 5.0 | 0.1913 | 0.2101 | 102 | 83 | 0.5788 |

Interpretation:

- The structural pool is not useless. Pool error precision is high (`0.74` to `0.78`), meaning most selected test-pool nodes are indeed wrong under prompt-off prediction.
- However, pool error recall is only about `0.20`, which is expected for a `rho≈0.20` pool but means the prompt can touch only a small part of all wrong nodes.
- `pool_only_prompt_update` works as intended: non-pool accuracy is unchanged across prompt off/on for all three datasets.
- The main failure is inside the pool. Prompt messages change many pool predictions, but they fix and break nodes at similar rates. On Actor they break more than they fix.
- This means the bottleneck is no longer message strength or non-pool contamination. It is prompt assignment/role quality inside the selected pool.
- Next method work should focus on prompt role learning, per-node rejection/gating inside the pool, and better pool scoring/partitioning. Simply increasing message scale is not justified.

## 2026-06-06 20:26 CST - P2 Rejection Gate and Prompt Role Regularization

Goal:

- Improve the P2 bottleneck found in the previous diagnostics: prompt messages were correctly restricted to the selected pool, but inside the pool they fixed and broke nodes at similar rates.
- Add a learnable per-pool-node soft acceptance gate so selected pool nodes are not forced to receive the same prompt strength.
- Add prompt-role diversity regularization so prompt nodes do not collapse into redundant roles.

Implementation changes:

- Added `use_rejection_gate` to `PromptGraphModuleP1`.
- The rejection gate is a small MLP over `[base_i, m1_i, m2_i, base_i - m1_i, m1_i - m2_i]` and scales prompt-edge weights per pool node.
- Added diagnostics: `pool_acceptance_mean`, `pool_acceptance_min`, `pool_acceptance_max`.
- Added `prompt_role_diversity_loss` over prompt keys.
- Added `prompt_acceptance_loss` over the soft acceptance gate.
- Updated `configs/gp2f_prompt_p2_strength.yaml` with:
  - `use_rejection_gate: true`
  - `rejection_gate_hidden_dim: 128`
  - `rejection_gate_bias_init: -1.0`
  - `lambda_prompt_role_diversity: 0.01`
  - `lambda_prompt_acceptance: 0.001`

Verification:

```bash
.venv/bin/python -m pytest tests/test_prompt_graph_module.py tests/test_prompt_aware_gp2f.py -q
```

Result: `25 passed`.

Experiment command:

```bash
for dataset in Actor chameleon squirrel; do
  .venv/bin/python -m experiments.run_gp2f_prompt_p2 \
    --config configs/gp2f_prompt_p2_strength.yaml \
    --target_dataset "$dataset" \
    --prompt_variant p2_strength \
    --epochs 100 \
    --seeds 0,1,2 \
    --eval_every 5 \
    --output_dir outputs/p2_rejection_gate_validation
done
```

Training summary:

| Dataset | Output Dir | Best Test Acc | Best Test Macro-F1 | Final Test Acc | Final Test Macro-F1 |
| --- | --- | ---: | ---: | ---: | ---: |
| Actor | `outputs/p2_rejection_gate_validation/Actor/20260606_202626` | 23.04+-2.73 | 17.38+-0.62 | 21.92+-1.90 | 18.50+-0.51 |
| chameleon | `outputs/p2_rejection_gate_validation/chameleon/20260606_203700` | 27.74+-4.45 | 26.35+-5.34 | 26.10+-2.90 | 24.93+-3.10 |
| squirrel | `outputs/p2_rejection_gate_validation/squirrel/20260606_204713` | 22.19+-1.97 | 14.87+-5.85 | 20.69+-0.23 | 19.27+-0.49 |

Post-hoc diagnostic command:

```bash
.venv/bin/python -m experiments.diagnose_prompt_graph \
  --summary_path outputs/p2_rejection_gate_validation/Actor/20260606_202626/summary.json \
  --summary_path outputs/p2_rejection_gate_validation/chameleon/20260606_203700/summary.json \
  --summary_path outputs/p2_rejection_gate_validation/squirrel/20260606_204713/summary.json \
  --output_dir outputs/prompt_diagnostics/p2_rejection_gate
```

Aggregate prompt-off vs prompt-on diagnostics:

| Dataset | Prompt Off Test Acc | Prompt On Test Acc | Pool Off Acc | Pool On Acc | Non-pool Off Acc | Non-pool On Acc |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Actor | 0.2320 | 0.2304 | 0.2268 | 0.2192 | 0.2332 | 0.2332 |
| chameleon | 0.2718 | 0.2774 | 0.2839 | 0.3109 | 0.2688 | 0.2688 |
| squirrel | 0.2198 | 0.2219 | 0.2229 | 0.2335 | 0.2190 | 0.2190 |

Pool/message diagnostics:

| Dataset | Pool Error Precision | Pool Error Recall | Pool Prediction Change | Pool Delta Norm | Prompt Usage Entropy |
| --- | ---: | ---: | ---: | ---: | ---: |
| Actor | 0.7732 | 0.2013 | 0.4548 | 7.9033 | 0.4902 |
| chameleon | 0.7161 | 0.1991 | 0.4919 | 8.6483 | 0.5643 |
| squirrel | 0.7771 | 0.2002 | 0.1606 | 0.7227 | 0.3577 |

Per-seed pool effects:

| Dataset | Seed | Selected Scale | Prompt Off Acc | Prompt On Acc | Pool Off Acc | Pool On Acc | Fixed | Broken | Pool Change |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Actor | 0 | 5.0 | 0.2319 | 0.2233 | 0.2072 | 0.1640 | 137 | 201 | 0.5958 |
| Actor | 1 | 5.0 | 0.2019 | 0.2074 | 0.1973 | 0.2250 | 187 | 146 | 0.4919 |
| Actor | 2 | 5.0 | 0.2621 | 0.2606 | 0.2760 | 0.2686 | 89 | 100 | 0.2766 |
| chameleon | 0 | 0.5 | 0.3102 | 0.3045 | 0.3357 | 0.3071 | 12 | 24 | 0.1548 |
| chameleon | 1 | 5.0 | 0.2959 | 0.3016 | 0.2974 | 0.3255 | 88 | 76 | 0.6487 |
| chameleon | 2 | 5.0 | 0.2093 | 0.2260 | 0.2186 | 0.3000 | 89 | 54 | 0.6721 |
| squirrel | 0 | 0.0 | 0.2031 | 0.2031 | 0.2120 | 0.2120 | 0 | 0 | 0.0000 |
| squirrel | 1 | 0.0 | 0.2203 | 0.2203 | 0.2664 | 0.2664 | 0 | 0 | 0.0000 |
| squirrel | 2 | 2.0 | 0.2360 | 0.2423 | 0.1903 | 0.2220 | 102 | 70 | 0.4817 |

Interpretation:

- The rejection gate improves the picture on chameleon and squirrel: prompt-on accuracy is higher than prompt-off, and pool accuracy improves.
- Actor remains problematic. Seed 1 improves, but seeds 0 and 2 still break more nodes than they fix. This means the current gate is not enough to identify harmful prompt messages on Actor.
- Non-pool accuracy is unchanged, so the pool-only protection is still working.
- The pool itself remains useful: pool error precision is still high (`0.716` to `0.777`), while recall stays near the expected `rho=0.20` level.
- The main remaining bottleneck is not pool selection or global message strength. It is per-node/per-role prompt assignment quality inside the pool.
- Next diagnostic should test whether a better pool oracle or confidence-based rejection can separate useful prompt recipients from harmful ones. If oracle/confidence rejection helps, implement learned rejection with a supervised auxiliary signal from training nodes only. If it does not help, P2 needs a stronger prompt-aware message mechanism rather than more gating.

## 2026-06-08 CST - P2 Confidence and Oracle-style Rejection Diagnostics

Goal:

- Test whether P2 would improve if the model could identify which pool nodes should actually receive prompt messages.
- Do this without retraining: compare prompt-off logits and prompt-on logits from the same saved checkpoints, then apply different post-hoc acceptance masks.
- Use train-pool labels only to fit confidence/margin/structural thresholds. Test labels are used only for final reporting.
- Also report oracle upper bounds as diagnostics. Oracle rows use labels to choose helpful prompted nodes and must not be treated as a deployable method.

Implementation:

- Added `experiments/diagnose_prompt_rejection.py`.
- For each node:
  - if accepted, use `prompt_on` logits;
  - otherwise, use `prompt_off` logits.
- Evaluated:
  - `prompt_off`
  - `prompt_on_all_pool`
  - train-threshold rejection rules:
    - low prompt-off confidence
    - low prompt-off margin
    - high structural score
    - high logit delta
    - high learned acceptance gate
  - oracle diagnostic upper bounds.

Command:

```bash
.venv/bin/python -m experiments.diagnose_prompt_rejection \
  --summary_path outputs/p2_rejection_gate_validation/Actor/20260606_202626/summary.json \
  --summary_path outputs/p2_rejection_gate_validation/chameleon/20260606_203700/summary.json \
  --summary_path outputs/p2_rejection_gate_validation/squirrel/20260606_204713/summary.json \
  --output_dir outputs/prompt_diagnostics/p2_rejection_oracle
```

Syntax check:

```bash
.venv/bin/python -m py_compile experiments/diagnose_prompt_rejection.py
```

Aggregate result:

| Dataset | Strategy | Val Acc | Test Acc | Test Pool Acc | Test Pool Accept Ratio | Pool Fixed | Pool Broken |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Actor | `prompt_off` | 0.2444 | 0.2320 | 0.2268 | 0.0000 | 0.0 | 0.0 |
| Actor | `prompt_on_all_pool` | 0.2578 | 0.2304 | 0.2192 | 0.1954 | 137.7 | 149.0 |
| Actor | `train_low_conf_off` | 0.2622 | 0.2310 | 0.2219 | 0.1305 | 108.7 | 116.0 |
| Actor | `train_low_margin_off` | 0.2578 | 0.2313 | 0.2237 | 0.1362 | 115.7 | 120.3 |
| Actor | `train_high_learned_acceptance` | 0.2622 | 0.2320 | 0.2270 | 0.1616 | 126.7 | 126.3 |
| Actor | `oracle_test_upper_bound` | 0.2444 | 0.2505 | 0.3196 | 0.0181 | 137.7 | 0.0 |
| chameleon | `prompt_off` | 0.2844 | 0.2718 | 0.2839 | 0.0000 | 0.0 | 0.0 |
| chameleon | `prompt_on_all_pool` | 0.3044 | 0.2774 | 0.3109 | 0.1869 | 63.0 | 51.3 |
| chameleon | `train_low_conf_off` | 0.3044 | 0.2774 | 0.3109 | 0.1866 | 63.0 | 51.3 |
| chameleon | `train_low_margin_off` | 0.3044 | 0.2774 | 0.3109 | 0.1869 | 63.0 | 51.3 |
| chameleon | `train_high_learned_acceptance` | 0.3044 | 0.2774 | 0.3109 | 0.1868 | 63.0 | 51.3 |
| chameleon | `oracle_test_upper_bound` | 0.2844 | 0.3018 | 0.4311 | 0.0277 | 63.0 | 0.0 |
| squirrel | `prompt_off` | 0.2444 | 0.2198 | 0.2229 | 0.0000 | 0.0 | 0.0 |
| squirrel | `prompt_on_all_pool` | 0.2511 | 0.2219 | 0.2335 | 0.1941 | 34.0 | 23.3 |
| squirrel | `train_low_conf_off` | 0.2511 | 0.2219 | 0.2335 | 0.1941 | 34.0 | 23.3 |
| squirrel | `train_low_margin_off` | 0.2511 | 0.2219 | 0.2335 | 0.1941 | 34.0 | 23.3 |
| squirrel | `train_high_learned_acceptance` | 0.2511 | 0.2219 | 0.2335 | 0.1847 | 34.0 | 23.3 |
| squirrel | `oracle_test_upper_bound` | 0.2444 | 0.2266 | 0.2566 | 0.0065 | 34.0 | 0.0 |

Interpretation:

- Oracle rejection confirms the hypothesis: if the model could perfectly identify helpful prompt recipients, test accuracy would improve.
  - Actor: `0.2320 -> 0.2505`
  - chameleon: `0.2718 -> 0.3018`
  - squirrel: `0.2198 -> 0.2266`
- The oracle accepts only a very small fraction of test-pool nodes:
  - Actor: `0.0181`
  - chameleon: `0.0277`
  - squirrel: `0.0065`
- This means most prompt messages are unnecessary or harmful. The useful signal is sparse, not broad.
- Train-threshold confidence/margin/structural rejection does not approach the oracle upper bound.
  - On Actor it almost falls back to prompt-off.
  - On chameleon and squirrel it is effectively the same as all-pool prompting.
- Because the setting is 5-shot, train-pool labels are too few to learn a reliable rejection rule. The train threshold overfits or degenerates into accepting almost the same nodes as the original gate.
- Conclusion: the next improvement should not be another simple confidence threshold. P2 needs either:
  - a better learned rejection signal trained with stronger supervision or self-supervision;
  - a validation-calibrated conservative budget that accepts far fewer pool nodes;
  - or a prompt-aware message mechanism that reduces harmful class changes instead of merely gating edge strength.

## 2026-06-08 CST - P2 Shot Relaxation Diagnostics

Goal:

- Relax the few-shot restriction and test whether more labeled nodes make P2 prompt usage/rejection more reliable.
- Compare `10-shot`, `50-shot`, and `5% shot` on Actor and chameleon.
- Reuse the P2 strength runner and the same rejection/oracle diagnostic as above.

Implementation changes:

- Added ratio-based few-shot splitting through `shot_ratio`.
- Added `--shots` and `--shot_ratio` CLI overrides to the P2 graph runner.
- Updated diagnostics to use the same ratio split when loading saved configs.
- Added split tests covering fixed-shot and ratio-shot masks.

Validation:

```bash
.venv/bin/python -m pytest tests/test_splits.py tests/test_prompt_graph_module.py tests/test_prompt_aware_gp2f.py -q
```

Result: `27 passed`.

Training command:

```bash
.venv/bin/python -m experiments.run_gp2f_prompt_p2 \
  --config configs/gp2f_prompt_p2_strength.yaml \
  --target_dataset Actor \
  --prompt_variant p2_strength \
  --shots 10 \
  --epochs 100 \
  --seeds 0,1,2 \
  --eval_every 5 \
  --output_dir outputs/p2_shot_relaxation/shots10

.venv/bin/python -m experiments.run_gp2f_prompt_p2 \
  --config configs/gp2f_prompt_p2_strength.yaml \
  --target_dataset Actor \
  --prompt_variant p2_strength \
  --shots 50 \
  --epochs 100 \
  --seeds 0,1,2 \
  --eval_every 5 \
  --output_dir outputs/p2_shot_relaxation/shots50

.venv/bin/python -m experiments.run_gp2f_prompt_p2 \
  --config configs/gp2f_prompt_p2_strength.yaml \
  --target_dataset Actor \
  --prompt_variant p2_strength \
  --shot_ratio 0.05 \
  --epochs 100 \
  --seeds 0,1,2 \
  --eval_every 5 \
  --output_dir outputs/p2_shot_relaxation/ratio5

.venv/bin/python -m experiments.run_gp2f_prompt_p2 \
  --config configs/gp2f_prompt_p2_strength.yaml \
  --target_dataset chameleon \
  --prompt_variant p2_strength \
  --shots 10 \
  --epochs 100 \
  --seeds 0,1,2 \
  --eval_every 5 \
  --output_dir outputs/p2_shot_relaxation/shots10

.venv/bin/python -m experiments.run_gp2f_prompt_p2 \
  --config configs/gp2f_prompt_p2_strength.yaml \
  --target_dataset chameleon \
  --prompt_variant p2_strength \
  --shots 50 \
  --epochs 100 \
  --seeds 0,1,2 \
  --eval_every 5 \
  --output_dir outputs/p2_shot_relaxation/shots50

.venv/bin/python -m experiments.run_gp2f_prompt_p2 \
  --config configs/gp2f_prompt_p2_strength.yaml \
  --target_dataset chameleon \
  --prompt_variant p2_strength \
  --shot_ratio 0.05 \
  --epochs 100 \
  --seeds 0,1,2 \
  --eval_every 5 \
  --output_dir outputs/p2_shot_relaxation/ratio5
```

Training summary:

| Dataset | Setting | Best Test Acc | Best Test Macro-F1 | Final Test Acc | Final Test Macro-F1 |
| --- | --- | ---: | ---: | ---: | ---: |
| Actor | 10-shot | 22.32+-1.15 | 20.08+-1.81 | 21.41+-0.40 | 19.97+-0.97 |
| Actor | 50-shot | 20.89+-4.35 | 17.48+-4.72 | 21.31+-0.59 | 18.91+-1.62 |
| Actor | 5% shot | 25.34+-0.87 | 18.38+-4.65 | 25.02+-0.95 | 19.78+-0.28 |
| chameleon | 10-shot | 30.09+-4.98 | 27.96+-3.15 | 27.51+-5.49 | 26.21+-3.81 |
| chameleon | 50-shot | 34.10+-2.98 | 33.43+-2.64 | 32.91+-1.75 | 32.62+-1.70 |
| chameleon | 5% shot | 33.15+-2.03 | 31.46+-2.04 | 31.54+-1.82 | 29.10+-1.48 |

Rejection diagnostic command:

```bash
.venv/bin/python -m experiments.diagnose_prompt_rejection \
  --summary_path outputs/p2_shot_relaxation/shots10/Actor/20260608_115125/summary.json \
  --summary_path outputs/p2_shot_relaxation/shots50/Actor/20260608_120105/summary.json \
  --summary_path outputs/p2_shot_relaxation/ratio5/Actor/20260608_121256/summary.json \
  --summary_path outputs/p2_shot_relaxation/shots10/chameleon/20260608_122456/summary.json \
  --summary_path outputs/p2_shot_relaxation/shots50/chameleon/20260608_123129/summary.json \
  --summary_path outputs/p2_shot_relaxation/ratio5/chameleon/20260608_123843/summary.json \
  --output_dir outputs/prompt_diagnostics/p2_shot_relaxation_rejection
```

Rejection diagnostic summary:

| Dataset | Setting | Prompt Off | Prompt On All Pool | Best Train-Based Rejection | Oracle Test Upper Bound |
| --- | --- | ---: | ---: | ---: | ---: |
| Actor | 10-shot | 0.2212 | 0.2232 | 0.2232 | 0.2327 |
| Actor | 50-shot | 0.2088 | 0.2089 | 0.2090 | 0.2215 |
| Actor | 5% shot | 0.2478 | 0.2534 | 0.2530 | 0.2739 |
| chameleon | 10-shot | 0.2924 | 0.3009 | 0.3009 | 0.3178 |
| chameleon | 50-shot | 0.3426 | 0.3410 | 0.3436 | 0.3545 |
| chameleon | 5% shot | 0.3299 | 0.3315 | 0.3318 | 0.3429 |

Interpretation:

- Relaxing supervision helps the base task more than it helps prompt rejection.
- Actor benefits most from `5% shot`, but `50-shot` does not improve over `10-shot`; fixed 50 labels per class may still be unstable for Actor under this split and message-scale selection.
- chameleon improves clearly with more labels, and `50-shot` is the best training setting.
- Prompt-on usually gives only a small gain over prompt-off, and sometimes hurts.
- The oracle gap remains under every setting:
  - Actor 5%: `0.2534 -> 0.2739`
  - chameleon 50-shot: `0.3436 -> 0.3545`
- Train-based rejection does not reliably approach oracle. It either matches all-pool prompting or gives only a tiny improvement.
- Conclusion: simply increasing shots is not enough to solve the prompt recipient selection problem. The next P2 improvement should focus on conservative prompt acceptance or a better learned rejection objective, not only more labels.

## 2026-06-08 CST - P2.1 Multi-view Routing Implementation

Goal:

- Optimize P2 without changing the GP2F dual-branch scaffold.
- Add multi-view prompt connection scoring before prompt-edge generation.
- Add train-only prompt usage consistency so labeled train-pool nodes from the same class learn similar prompt usage patterns.
- Keep heterophilic settings free of pseudo-label construction and keep prompt rejection available.

Implementation:

- Updated `PromptGraphModuleP1` with optional multi-view routing:
  - semantic view: self/base representation query;
  - structural view: existing one-hop/two-step diffusion context query;
  - role view: degree, structural discrepancy, and neighbor variance query.
- Added a node-wise view gate:

```text
score_ij =
  w_sem_i    * score_sem_ij
+ w_struct_i * score_struct_ij
+ w_role_i   * score_role_ij
```

- Rejection gate remains separate from view gate:
  - view gate decides which evidence view guides prompt assignment;
  - rejection gate controls whether pool nodes receive prompt intervention.
- Added `prompt_usage_consistency_loss`, restricted to `train_mask & pool_mask`.
  - Same-class train-pool nodes are encouraged to use similar prompt distributions.
  - Different-class separation is weak and margin-based.
  - Validation/test labels are not used.
- Added optional `prompt_view_entropy_loss` for future experiments.
- Added `configs/gp2f_prompt_p2_multiview.yaml`.
- Added CLI overrides:
  - `--enable_multiview_routing`
  - `--disable_multiview_routing`
  - `--lambda_prompt_usage_consistency`
  - `--lambda_prompt_view_entropy`
- Added CSV/metrics fields:
  - `use_multiview_routing`
  - `semantic_view_weight`
  - `structural_view_weight`
  - `role_view_weight`
  - `view_gate_entropy`

Validation:

```bash
.venv/bin/python -m pytest tests/test_prompt_graph_module.py tests/test_prompt_aware_gp2f.py tests/test_splits.py -q
.venv/bin/python -m py_compile models/prompt_graph_module.py experiments/run_gp2f_prompt_graph.py experiments/run_gp2f_prompt_p2.py
```

Result: `29 passed`.

Smoke command:

```bash
.venv/bin/python -m experiments.run_gp2f_prompt_p2 \
  --config configs/gp2f_prompt_p2_multiview.yaml \
  --target_dataset Actor \
  --prompt_variant p2_multiview \
  --shots 10 \
  --epochs 3 \
  --seeds 0 \
  --eval_every 1 \
  --prompt_message_scale 1.0 \
  --output_dir outputs/smoke_p2_multiview
```

Smoke output:

- Completed successfully.
- Summary path: `outputs/smoke_p2_multiview/Actor/20260608_131518/summary.json`.
- Multi-view routing was active:
  - `semantic_view_weight=0.3621`
  - `structural_view_weight=0.3193`
  - `role_view_weight=0.3186`
  - `view_gate_entropy=0.9983`

Notes:

- The smoke result is not an effectiveness result because it only ran 3 epochs and used edge warmup.
- Next effectiveness check should use:
  - Actor with `5% shot`;
  - chameleon with `50-shot`;
  - Cora/CiteSeer still constrained to `5-shot`.

## 2026-06-08 CST - P2.1 Multi-view Effectiveness Check

Purpose:

- Validate whether multi-view routing improves P2 under relaxed heterophilic supervision.
- Actor uses `shot_ratio=0.05`.
- chameleon uses `shots=50`.
- Both use validation-only prompt message scale selection from the configured scale grid.

Commands:

```bash
.venv/bin/python -m experiments.run_gp2f_prompt_p2 \
  --config configs/gp2f_prompt_p2_multiview.yaml \
  --target_dataset Actor \
  --prompt_variant p2_multiview \
  --shot_ratio 0.05 \
  --seeds 0,1,2 \
  --epochs 100 \
  --eval_every 5 \
  --output_dir outputs/p2_multiview_validation

.venv/bin/python -m experiments.run_gp2f_prompt_p2 \
  --config configs/gp2f_prompt_p2_multiview.yaml \
  --target_dataset chameleon \
  --prompt_variant p2_multiview \
  --shots 50 \
  --seeds 0,1,2 \
  --epochs 100 \
  --eval_every 5 \
  --output_dir outputs/p2_multiview_validation
```

Main results:

| Dataset | Setting | Best Acc | Best Macro-F1 | Final Acc | Final Macro-F1 | Summary |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| Actor | 5% shot | 24.94+-0.63 | 21.49+-1.24 | 24.23+-0.50 | 20.26+-0.54 | `outputs/p2_multiview_validation/Actor/20260608_131825/summary.json` |
| chameleon | 50-shot | 34.90+-1.98 | 34.19+-1.75 | 31.81+-2.79 | 31.64+-2.54 | `outputs/p2_multiview_validation/chameleon/20260608_133016/summary.json` |

Comparison against previous P2 strength runs:

| Dataset | Previous P2 Best Acc | P2.1 Best Acc | Change | Previous P2 Best Macro-F1 | P2.1 Best Macro-F1 | Change |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Actor | 25.34+-0.87 | 24.94+-0.63 | -0.40 | 18.38+-4.65 | 21.49+-1.24 | +3.11 |
| chameleon | 34.10+-2.98 | 34.90+-1.98 | +0.80 | 33.43+-2.64 | 34.19+-1.75 | +0.76 |

Selected message scale and view gate:

| Dataset | Seed | Scale | Semantic | Structural | Role | View Entropy | Pool Acceptance |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Actor | 0 | 1.0 | 0.2903 | 0.4537 | 0.2560 | 0.9700 | 0.2954 |
| Actor | 1 | 1.0 | 0.1389 | 0.8124 | 0.0488 | 0.5102 | 0.3091 |
| Actor | 2 | 2.0 | 0.2879 | 0.4453 | 0.2668 | 0.9727 | 0.2896 |
| chameleon | 0 | 0.0 | 0.3330 | 0.3335 | 0.3335 | 1.0000 | 0.3037 |
| chameleon | 1 | 1.0 | 0.3368 | 0.3393 | 0.3240 | 0.9998 | 0.2896 |
| chameleon | 2 | 2.0 | 0.3291 | 0.3464 | 0.3245 | 0.9996 | 0.2867 |

Rejection diagnostics:

```bash
.venv/bin/python -m experiments.diagnose_prompt_rejection \
  --summary_path outputs/p2_multiview_validation/Actor/20260608_131825/summary.json \
  --summary_path outputs/p2_multiview_validation/chameleon/20260608_133016/summary.json \
  --output_dir outputs/prompt_diagnostics/p2_multiview_rejection
```

| Dataset | Prompt Off | Prompt On All Pool | Best Train-only Rejection | Oracle Test Upper Bound |
| --- | ---: | ---: | ---: | ---: |
| Actor | 24.11 | 24.94 | 24.76 | 26.75 |
| chameleon | 35.04 | 34.90 | 35.09 | 35.87 |

Interpretation:

- Actor: prompt intervention is mildly useful over prompt-off, but multiview does not improve best accuracy over previous P2. It does improve Macro-F1, suggesting more balanced class behavior. Oracle rejection still has a large gap, so pool acceptance remains the main bottleneck.
- chameleon: multiview slightly improves best accuracy and Macro-F1 over previous P2, but prompt-off is still stronger than prompt-on-all-pool in the rejection diagnostic. The view gate remains almost uniform, so the model has not learned a meaningful semantic/structural/role specialization.
- Current conclusion: P2.1 multi-view routing is directionally reasonable but not yet a reliable effectiveness gain. The next useful optimization should focus on train-only prompt acceptance/rejection and stronger but still label-isolated view supervision, not simply adding more prompt nodes or increasing edge scale.

## 2026-06-08 CST - P2.2 Reliable Acceptance Gate

Purpose:

- Make the model more reliable at deciding which pool nodes should receive prompt messages.
- Keep the decision label-isolated: only train-pool labels can supervise the acceptance gate.
- Add null/rejection safety when prompt intervention is noisy.

Implementation:

- `PromptGraphModuleP1` now records `pool_acceptance_logit` in addition to `pool_acceptance_gate`.
  - This allows stable BCE-style supervision over the acceptance gate.
- Added `prompt_acceptance_budget_loss`.
  - It constrains the mean acceptance gate to a configurable soft range.
  - Default P2.1 multiview config now uses:

```yaml
lambda_prompt_acceptance: 0.0
lambda_prompt_acceptance_budget: 0.05
acceptance_budget_min: 0.02
acceptance_budget_max: 0.20
lambda_prompt_acceptance_supervision: 0.10
acceptance_supervision_positive_margin: 0.02
acceptance_supervision_negative_margin: 0.02
acceptance_supervision_balance_targets: true
```

- Added train-only acceptance supervision in `run_gp2f_prompt_graph.py`.
  - For train-pool nodes only, compare prompt-on logits and prompt-off logits.
  - If prompt-on improves the true-class signal or fixes correctness, target acceptance is high.
  - If prompt-on hurts the true-class signal or breaks correctness, target acceptance is low.
  - Validation/test labels are never used.
- Enhanced rejection gate input.
  - Previous gate used only structural query context.
  - New P2.2 gate can also use:
    - role context;
    - prompt assignment top probability;
    - routing entropy;
    - top-1/top-2 logit margin;
    - structural unreliability score.
  - Config:

```yaml
rejection_gate_use_role_context: true
rejection_gate_use_routing_features: true
```

When to increase rejection / null safety:

- Increase safety if `prompt_on_all_pool <= prompt_off` in rejection diagnostics.
- Increase safety if oracle rejection is much better than prompt-on, because the issue is recipient selection rather than prompt message capacity.
- Increase safety if `broken > fixed` among pool nodes.
- Increase safety if `view_gate_entropy` is high and view weights remain uniform, because the model has not learned a reliable routing view.
- Increase safety if `pool_acceptance_mean` is high but test/pool accuracy decreases.
- Relax safety only when prompt-on is consistently better than prompt-off and train-only rejection does not improve much over all-pool prompting.

Validation:

```bash
.venv/bin/python -m pytest tests/test_prompt_graph_module.py tests/test_prompt_aware_gp2f.py tests/test_splits.py -q
.venv/bin/python -m py_compile models/prompt_graph_module.py experiments/run_gp2f_prompt_graph.py experiments/run_gp2f_prompt_p2.py
```

Result: `32 passed`.

Smoke command:

```bash
.venv/bin/python -m experiments.run_gp2f_prompt_p2 \
  --config configs/gp2f_prompt_p2_multiview.yaml \
  --target_dataset Actor \
  --prompt_variant p2_multiview \
  --shot_ratio 0.05 \
  --seeds 0 \
  --epochs 3 \
  --eval_every 1 \
  --prompt_message_scale 1.0 \
  --output_dir outputs/smoke_p2_reliable_acceptance
```

Smoke output:

- Summary path: `outputs/smoke_p2_reliable_acceptance/Actor/20260608_141026/summary.json`.
- The new diagnostics were written:
  - `pool_acceptance_mean=0.2677`
  - `prompt_acceptance_budget=0.0045`
  - `prompt_acceptance_supervision=0.5436`
  - `acceptance_supervised_count=3`
  - `acceptance_positive_count=2`
  - `acceptance_negative_count=1`
  - `acceptance_ignored_count=379`

Notes:

- The 3-epoch smoke is only an implementation check.
- Early supervised acceptance targets are sparse because prompt edge warmup keeps prompt-on/off behavior close at the start.
- Full validation should rerun Actor 5% and chameleon 50-shot with the same seeds used in the P2.1 check, then repeat rejection diagnostics.

## 2026-06-09 CST - P2.3 Hard Acceptance Budget

Purpose:

- Make prompt intervention genuinely sparse instead of only relying on a soft acceptance gate.
- Check whether a stricter top-budget rejection mechanism can improve the model's ability to decide which pool nodes should receive prompt messages.
- Keep the setting label-isolated: no validation/test label is used for routing, acceptance, budget, or scale selection.

Implementation:

- Added hard acceptance to `PromptGraphModuleP1`.
  - `use_hard_acceptance=true`
  - `hard_acceptance_ratio=0.10`
  - `hard_acceptance_straight_through=true`
- The module now keeps only the top-ratio pool nodes by learned acceptance score.
- Prompt edge weights are multiplied by the effective hard acceptance mask.
- Added diagnostics:
  - `use_hard_acceptance`
  - `hard_acceptance_ratio`
  - `hard_acceptance_selected_ratio`
- Added CLI overrides:
  - `--enable_hard_acceptance`
  - `--disable_hard_acceptance`
  - `--hard_acceptance_ratio`
- Added unit test for hard acceptance edge masking.

Validation:

```bash
.venv/bin/python -m pytest tests/test_prompt_graph_module.py tests/test_prompt_aware_gp2f.py tests/test_splits.py -q
.venv/bin/python -m py_compile models/prompt_graph_module.py experiments/run_gp2f_prompt_graph.py experiments/run_gp2f_prompt_p2.py
```

Result: `33 passed`.

Experiment command:

```bash
.venv/bin/python -m experiments.run_gp2f_prompt_p2 \
  --config configs/gp2f_prompt_p2_multiview.yaml \
  --target_dataset Actor \
  --prompt_variant p2_multiview \
  --shot_ratio 0.05 \
  --seeds 0,1,2 \
  --epochs 100 \
  --eval_every 5 \
  --output_dir outputs/p2_hard_acceptance_5pct

.venv/bin/python -m experiments.run_gp2f_prompt_p2 \
  --config configs/gp2f_prompt_p2_multiview.yaml \
  --target_dataset chameleon \
  --prompt_variant p2_multiview \
  --shot_ratio 0.05 \
  --seeds 0,1,2 \
  --epochs 100 \
  --eval_every 5 \
  --output_dir outputs/p2_hard_acceptance_5pct

.venv/bin/python -m experiments.run_gp2f_prompt_p2 \
  --config configs/gp2f_prompt_p2_multiview.yaml \
  --target_dataset squirrel \
  --prompt_variant p2_multiview \
  --shot_ratio 0.05 \
  --seeds 0,1,2 \
  --epochs 100 \
  --eval_every 5 \
  --output_dir outputs/p2_hard_acceptance_5pct
```

Summary:

| Dataset | Best Acc | Best Macro-F1 | Final Acc | Final Macro-F1 | Selected scale |
| --- | ---: | ---: | ---: | ---: | --- |
| Actor | 25.74+-1.01 | 16.83+-7.51 | 26.07+-1.34 | 21.99+-2.51 | all seeds 0.0 |
| chameleon | 34.49+-3.16 | 31.97+-2.59 | 33.25+-2.80 | 31.84+-3.13 | all seeds 0.0 |
| squirrel | 23.70+-0.53 | 22.38+-2.59 | 23.47+-0.53 | 23.30+-0.56 | all seeds 0.0 |

Key diagnostics:

| Dataset | Hard selected ratio | Mean prompt edge weight | Prompt message norm | View gate entropy |
| --- | ---: | ---: | ---: | ---: |
| Actor | ~0.100 | 0.0001-0.0033 | 0.0 | ~1.000 |
| chameleon | ~0.100 | 0.0028-0.0032 | 0.0 | ~1.000 |
| squirrel | ~0.100 | 0.0026-0.0029 | 0.0 | ~1.000 |

Rejection diagnostics:

```bash
.venv/bin/python -m experiments.diagnose_prompt_rejection \
  --summary_path outputs/p2_hard_acceptance_5pct/Actor/20260609_113253/summary.json \
  --summary_path outputs/p2_hard_acceptance_5pct/chameleon/20260609_114218/summary.json \
  --summary_path outputs/p2_hard_acceptance_5pct/squirrel/20260609_114714/summary.json \
  --output_dir outputs/prompt_diagnostics/p2_hard_acceptance_5pct
```

| Dataset | Prompt Off | Prompt On All Pool | Oracle Test Upper Bound |
| --- | ---: | ---: | ---: |
| Actor | 25.74 | 25.74 | 25.74 |
| chameleon | 34.49 | 34.49 | 34.49 |
| squirrel | 23.70 | 23.70 | 23.70 |

Interpretation:

- Hard acceptance works mechanically: selected ratio is close to 10% on every dataset.
- However, validation consistently selected `message_scale=0.0`, so the final reported models effectively disabled prompt messages.
- This means P2.3 is stable but too conservative under the current scale-selection protocol.
- The identical prompt-off / prompt-on / oracle diagnostic is not evidence that rejection is solved. It only means the selected checkpoint has no active prompt message.
- The view gate still has near-maximum entropy, so semantic / structural / role views are not specializing.
- Current bottleneck is not only pool capacity. The stronger issue is that the prompt message path is not producing validation-visible gains, so the runner rationally chooses to turn it off.

Next diagnosis:

- Run fixed positive message-scale experiments instead of allowing scale 0 during diagnosis.
- Compare `message_scale=0.5` and `1.0` with hard ratios `0.10`, `0.20`, and soft gate.
- Temporarily disable validation selection of scale 0 when testing whether prompt messages can help.
- Inspect train-pool supervised acceptance: current full summaries show `acceptance_supervised_count=0`, so the acceptance supervision is not firing at selected checkpoints.
- Improve the train-only acceptance target generation or compute the acceptance supervision after prompt warmup.

## 2026-06-09 CST - P2.4 Zero-init Prompt Message Safety

Purpose:

- The positive-scale diagnostic showed that forcing prompt messages to be active hurts Actor, chameleon, and squirrel.
- The earlier hard-acceptance experiment was stable only because validation selected `message_scale=0.0`.
- This update makes the prompt-aware message path start exactly as NoPrompt and lets training wake it up only if useful.

Implementation:

- Added `prompt_aware.zero_init_prompt_messages`.
  - When enabled, node-to-prompt and prompt-to-node message projections are initialized to zero.
  - Initial full-edge prompt logits match NoPrompt: `init_logit_delta_full_edge_scale=0.0`.
  - The projection still receives gradients, so the prompt message path is not frozen.
- Restored safe validation-only message-scale selection.
  - `message_scale_grid: [0.0, 0.1, 0.25, 0.5, 1.0]`
  - `0.0` remains the no-message fallback.
- Added logging for `zero_init_prompt_messages`.
- Fixed acceptance target ambiguity by using strict margins (`>` / `<`) instead of allowing zero delta to be both positive and negative.

Validation:

```bash
.venv/bin/python -m pytest tests/test_prompt_graph_module.py tests/test_prompt_aware_gp2f.py tests/test_splits.py -q
.venv/bin/python -m py_compile models/prompt_aware_gp2f.py experiments/run_gp2f_prompt_graph.py experiments/run_gp2f_prompt_p2.py
```

Result: `34 passed`.

Positive-scale diagnostic before this fix:

| Dataset | Best Acc | Best Macro-F1 | Final Acc | Final Macro-F1 |
| --- | ---: | ---: | ---: | ---: |
| Actor | 17.99+-7.51 | 7.16+-2.65 | 21.51+-7.75 | 7.81+-0.78 |
| chameleon | 24.27+-0.91 | 11.50+-2.89 | 23.19+-0.64 | 9.46+-0.82 |
| squirrel | 20.22+-0.08 | 8.37+-1.31 | 20.16+-0.38 | 8.51+-0.60 |

Zero-init safe-scale experiment:

```bash
.venv/bin/python -m experiments.run_gp2f_prompt_p2 \
  --config configs/gp2f_prompt_p2_multiview.yaml \
  --target_dataset Actor \
  --prompt_variant p2_multiview \
  --shot_ratio 0.05 \
  --seeds 0,1,2 \
  --epochs 80 \
  --eval_every 5 \
  --output_dir outputs/p2_zero_init_safe_5pct

.venv/bin/python -m experiments.run_gp2f_prompt_p2 \
  --config configs/gp2f_prompt_p2_multiview.yaml \
  --target_dataset chameleon \
  --prompt_variant p2_multiview \
  --shot_ratio 0.05 \
  --seeds 0,1,2 \
  --epochs 80 \
  --eval_every 5 \
  --output_dir outputs/p2_zero_init_safe_5pct
```

Summary:

| Dataset | Best Acc | Best Macro-F1 | Final Acc | Final Macro-F1 | Selected scale |
| --- | ---: | ---: | ---: | ---: | --- |
| Actor | 26.25+-0.33 | 17.18+-7.30 | 26.44+-0.76 | 18.65+-8.96 | 0.0, 0.0, 0.1 |
| chameleon | 33.95+-3.01 | 31.29+-1.89 | 32.90+-2.08 | 31.37+-2.44 | all seeds 0.0 |

Key diagnostics:

- `zero_init_prompt_messages=1.0` for every selected run.
- `init_logit_delta_full_edge_scale=0.0`, confirming strict NoPrompt equivalence at initialization.
- `acceptance_supervised_count=0` in selected checkpoints, so the train-only acceptance target still does not provide useful supervision.
- The validation selector still mostly chooses `message_scale=0.0`, meaning the current prompt message path is stable but not yet useful.

Interpretation:

- P2.4 fixes the initialization/stability issue and prevents forced positive prompt messages from damaging performance.
- It does not solve the core effectiveness problem: the model still cannot reliably identify when prompt messages should be enabled.
- The next optimization should not increase pool size or message scale first. The priority is to create a validation-visible training signal for prompt activation, such as a delayed acceptance-supervision phase after warmup or a train-only auxiliary objective that compares prompt-on/off logits after the prompt message path has nonzero capacity.

Follow-up CE-delta acceptance smoke:

```bash
.venv/bin/python -m experiments.run_gp2f_prompt_p2 \
  --config configs/gp2f_prompt_p2_multiview.yaml \
  --target_dataset Actor \
  --prompt_variant p2_multiview \
  --shot_ratio 0.05 \
  --seeds 0 \
  --epochs 40 \
  --eval_every 5 \
  --prompt_message_scale 0.1 \
  --output_dir outputs/smoke_p2_ce_delta_acceptance
```

Result:

- Best Test Acc: `19.23`
- Final Test Acc: `17.12`
- `acceptance_supervised_count=42`
- `acceptance_positive_count=12`
- `acceptance_negative_count=30`
- `acceptance_target_mean=0.286`
- `prompt_msg_norm=1.69e-06`

Interpretation:

- CE-delta supervision does create nonzero train-only acceptance targets.
- The fixed positive prompt message still hurts, so the remaining issue is not only missing gate supervision.
- The learned signal is mostly negative, which supports the current validation behavior of selecting `message_scale=0.0`.
- Next improvement should focus on prompt message quality and representation alignment, not simply accepting more pool nodes.
