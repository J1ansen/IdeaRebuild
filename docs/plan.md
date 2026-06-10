# Prompt Module Implementation History and Current Plan

本文档按实现历史记录当前项目中 prompt 模块的演进。重点区分：

- 已经实现的内容；
- 当前实验观察；
- 尚未实现或不应作为当前方法主张的内容；
- 下一步优化方向。

当前最重要的结论是：prompt routing 已经逐步变得更可控，但 prompt
message 尚未产生稳定正收益。因此，下一步优化重点不是扩大连接池或继续增强
edge scale，而是验证并改进 prompt message 的有效性。

## 0. Shared Design Boundary

所有 prompt 阶段都遵守以下边界：

- frozen branch 始终只使用 original graph；
- original graph 的节点间边不删除、不重连、不改写；
- prompt 只影响 adapted branch；
- 不使用伪标签构图；
- 不使用 validation/test label 参与 pool、routing、prototype 或 loss；
- 异配图主实验默认关闭 GP2F adjacency-driven contrastive/topology losses；
- 当前方法不声称提升原图同配性，也不声称完成拓扑重构。

## 1. Faithful GP2F Baseline

### 目标

先复现一个可复用预训练模型的 GP2F-style baseline，避免每次单数据集实验都重新预训练。

### 已实现内容

- 复用 Cora + GRACE 预训练 GNN checkpoint：
  `pretrained_gnns/lr_0.0005_weightdecay_0.0005_hid_dim_128.pkl`
- 复用 `/Users/jackson/MyIdea/data` 数据缓存。
- 实现 `FaithfulGP2F`：
  - frozen branch；
  - adapted branch；
  - residual adapters；
  - learnable fusion alpha；
  - classifier。
- 支持 `InputAligner` 处理 target feature dim 与 source dim 不一致的情况。
- 支持 B0-B3 loss presets：
  - B0: CE；
  - B1: CE + contrastive；
  - B2: CE + topology fusion；
  - B3: CE + contrastive + topology fusion。
- 修复 topology fusion loss，使其使用：
  `S_mix = alpha * S_pre + (1 - alpha) * S_adp`。
- runner 支持：
  - 多 seed；
  - tqdm progress bar；
  - ETA；
  - mean+-std；
  - summary.json / summary.csv；
  - per-seed metrics and curves。

### 当前定位

这是后续 prompt 模块的基础对照。稳定版本不应被写成官方 GP2F 完全复现；
official-style config 只用于兼容性对照。

## 2. P0: Unified Multi-view Residual Prompt

### 目标

先不引入 prompt nodes / prompt edges，而是在 adapted branch 输入特征上添加可控 residual prompt：

```text
adapted_x = z + gamma * (g_sem * u_sem + g_struct * u_struct)
```

### 已实现内容

- `models/prompt_module.py`
- `UnifiedMultiViewResidualPrompt`
- `ParameterMatchedResidualControl`
- semantic view：
  - 使用 train labels 构造类别原型；
  - 不使用伪标签；
  - 支持 leave-one-out prototype；
  - 1-shot 情况下可 mask semantic route。
- structural view：
  - 基于无标签 one-step / two-step diffusion summary；
  - 默认可使用 `z_detached` 或 `h_pre_detached`。
- null route：
  - soft no-prompt preference；
  - 不等于 hard rejection。
- zero-init：
  - 初始 `adapted_x` 接近 `z`；
  - 初始行为接近 NoPrompt。
- route budget：
  - soft intervention budget；
  - 可通过 validation-only grid 选择。
- parameter-matched residual control：
  - 用于排除“额外参数量”带来的提升。

### 当前定位

P0 是 residual feature prompt，不是拓扑提示、不构造提示边、不提升原图同配性。
它主要用于验证多视角 residual prompt 是否能作为轻量适应模块。

## 3. P1: Prompt Graph Adaptation

### 目标

引入显式 prompt nodes 和 prompt edges，让 adapted branch 接收 prompt-augmented graph。

### 已实现内容

- `models/prompt_graph_module.py`
- `PromptGraphModuleP1`
- prompt nodes：
  - prompt 节点追加在原图节点之后；
  - 原图节点始终保持前 `N` 个位置；
  - final logits 只使用前 `N` 个原图节点。
- pool selection：
  - `pool_mask = train_mask OR top_rho_structural_unreliable_nodes`
  - 默认 `rho=0.20`；
  - `rho=0` 时仍保留 train nodes；
  - 支持 structural pool 和 random pool。
- structural unreliability score：
  - 基于 `base`、`m1`、`m2`、neighbor variance；
  - 不使用标签。
- top-k routing：
  - 每个 pool node 最多连接 `topk_prompt_per_node` 个 prompt nodes；
  - 默认 top-k = 2。
- bidirectional prompt edges：
  - node-to-prompt: `edge_type=1`；
  - prompt-to-node: `edge_type=2`；
  - original edge: `edge_type=0`。
- prompt edge weight：
  - 使用 assignment probability、edge scale、acceptance gate 共同控制。
- runner：
  - `experiments/run_gp2f_prompt_graph.py`
  - 支持 `noprompt`、`p1_graph`、`p1_random_pool` 等变体。

### 当前定位

P1 完成了显式 prompt graph 的基础工程接口。它证明 adapted branch 可以接收
prompt-augmented graph，但不保证 prompt message 有效。

## 4. P2: Prompt-aware Adapted Branch

### 目标

让 adapted branch 能识别并专门处理 prompt edges，而不是把 prompt edges 当成普通图边。

### 已实现内容

- `models/prompt_aware_gp2f.py`
- `PromptAwareGP2F`
- adapted branch 中区分三类边：
  - original edge；
  - node-to-prompt edge；
  - prompt-to-node edge。
- original graph messages：
  - 继续走 adapted GNN + adapter。
- prompt messages：
  - 通过单独 message transform；
  - 使用 prompt gates 控制 node-to-prompt / prompt-to-node 两个方向。
- `pool_only_prompt_update`：
  - prompt-to-original update 默认只作用于 pool 内原图节点。
- zero-init prompt messages：
  - 初始行为接近 NoPrompt。
- message scale grid：
  - 支持通过 validation 选择 `message_scale`；
  - grid 包含 `0.0`，允许关闭 prompt。

### 当前实验观察

P2 之后，模型可以工程上区分 prompt edge 与 original edge，但正 scale
并未稳定带来收益。验证集经常选择 `message_scale=0.0`。

### 当前定位

P2 的工程目标达成，但性能目标未达成。

## 5. P2.5: Train-only Class-aware Structural Router

### 目标

解决 P2 中“连到哪个 prompt slot 不稳定”的问题。引入 train-only
类别一致性约束，使同类训练节点倾向连接到相同或相近 prompt slot，同时保留
residual slots 和 rejection gate。

### 已实现内容

- class-aware prompt slots：
  - 前 `num_classes` 个 prompt slots 作为 class-aware routing anchors；
  - 额外保留 `residual_prompt_count` 个 residual slots。
- 自动扩展 prompt node 数量：
  - 若 `num_prompt_nodes < num_classes + residual_prompt_count`，自动提升。
- `L_class_route`：
  - 只作用于 `train_mask & pool_mask`；
  - 训练节点标签为 `c` 时，鼓励 routing 到第 `c` 个 class slot。
- `L_key_proto`：
  - 使用 train pool 节点 query prototype 约束 class prompt keys。
- 降低 prompt balance 权重：
  - 避免 uniform usage 与 class-aware routing 冲突。
- diagnostics：
  - `class_router_hit_rate_train`
  - `class_router_hit_rate_val/test`
  - `prompt_usage_by_class_train`
  - `same_class_route_compactness`
  - `different_class_route_separation`

### 当前实验观察

- train-node class routing 明显改善；
- 但 val/test routing 泛化不足；
- P2.5 在 Actor/chameleon 上没有稳定超过 NoPrompt/P2；
- validation 仍经常选择 `message_scale=0.0`。

### 当前定位

P2.5 改善了 router 可解释性，但没有解决 prompt message 是否有用的问题。

## 6. P3: Prototype-initialized Router + Conditioned Receiver

### 目标

进一步稳定 class-aware router，并改造 prompt receiver，使 prompt message
根据节点上下文动态生成。

### 已实现内容

#### 6.1 Train-only Prototype Initialization

- 使用 `train_mask & pool_mask` 节点构造 class structural-query prototype：

```text
mu_c = mean(query_i), y_i = c
```

- 将前 `num_classes` 个 class prompt keys 初始化为对应 `mu_c`。
- 不使用 validation/test label。
- 若某类无训练样本，保持随机初始化并记录缺失类别。
- diagnostics：
  - `class_key_proto_init_coverage`
  - `class_key_proto_init_missing_classes`

#### 6.2 Conditioned Prompt Receiver

- `receiver_version = v2_conditioned`
- prompt message 使用：

```text
msg_{p->i} = MLP([h_i, h_p, h_i - h_p, h_i * h_p])
```

- prompt message residual update：

```text
h = h_graph + message_scale * beta * msg_prompt
```

- 默认：
  - `zero_init_prompt_messages=true`
  - `prompt_message_norm=layernorm`
  - `pool_only_prompt_update=true`

#### 6.3 Acceptance Supervision

- 已实现 train-only CE-delta 风格 acceptance supervision。
- 目标是鼓励模型接收有帮助的 prompt、拒绝有害 prompt。
- 当前配置默认：
  - `lambda_prompt_acceptance_supervision=0.20`
  - `acceptance_supervision_signal=ce_delta`

### 当前实验结果

5% shot，seeds 0,1,2，80 epochs：

| Dataset | P2.5 Best Acc | P3 Best Acc | P2.5 Best Macro-F1 | P3 Best Macro-F1 |
|---|---:|---:|---:|---:|
| Actor | 26.25+-0.33 | 26.25+-0.33 | 17.18+-7.30 | 17.18+-7.30 |
| chameleon | 33.95+-3.01 | 33.95+-3.01 | 31.29+-1.89 | 31.29+-1.89 |

关键诊断：

- P3 的 class-key prototype init 生效；
- class router 在训练节点上可以很高；
- chameleon 三个 seed 均选择 `message_scale=0.0`；
- Actor 只有一个 seed 选择正 scale，但没有性能提升；
- prompt message norm / update norm 多数为 0 或接近 0。

### 当前定位

P3 没有带来性能提升。当前瓶颈已经不是 routing 是否可控，而是 prompt
message 接入 adapted branch 后没有稳定正收益。

## 7. Modules That Are Not Yet Effective

| 模块 | 预期效果 | 实际结果 | 当前判断 |
|---|---|---|---|
| prompt message | 降低 CE，提升 val/test | validation 经常选择 `message_scale=0.0` | 未起作用 |
| prompt-aware receiver | 分离 prompt/original message 后提升 adapted branch | P3 与 P2.5 持平 | 工程实现有效，性能无收益 |
| rejection gate | 细粒度拒绝有害 prompt | 多数情况全局关闭 prompt scale | 未学到稳定细粒度拒绝 |
| class-aware router | 同类训练节点连到同类 prompt slot | train hit rate 高 | 训练节点有效 |
| router generalization | val/test 节点也路由合理 | val/test hit rate 低 | 泛化不足 |
| pool expansion | 覆盖更多需要 prompt 的节点 | 正 scale 仍无收益 | 暂非主要瓶颈 |

## 8. Not Implemented / Not Current Claims

以下内容不能写成当前方法贡献：

- 提升原图同配性；
- 原图拓扑重构；
- 伪标签构图；
- validation/test label 参与构图或训练；
- DFS/random-walk routing；
- 完整类别相容性矩阵；
- route supervised contrastive loss。

其中 route supervised contrastive 曾作为设想写入早期 plan，但当前代码主路径没有将其作为完整 P3 方法贡献。

## 9. Next Optimization Plan

下一步应围绕 prompt message usefulness，而不是继续堆叠 routing 复杂度。

### 9.1 Fixed Positive-scale Diagnostics

固定 `message_scale` 为正值，例如：

```text
0.1, 0.25, 0.5
```

观察 train pool 节点上：

```text
CE_no_prompt - CE_prompt
```

若训练节点上该值仍不稳定为正，说明 prompt message 本身无效。

### 9.2 Prompt-benefit Predictor

将 rejection gate 改为 benefit predictor：

```text
benefit_i = CE_no_prompt_i - CE_prompt_i
```

训练目标：

- benefit > 0：鼓励接收 prompt；
- benefit < 0：鼓励拒绝 prompt。

该监督只使用 train nodes。

### 9.3 Message as Residual Correction

当前 prompt message 更像 edge message passing。下一步应让 prompt 直接学习
node-level residual correction：

```text
h_prompt_i = MLP([h_i, routed_prompt_summary_i, h_i - routed_prompt_summary_i])
h_i' = h_i + beta_i * h_prompt_i
```

重点是让 prompt message 对分类 logits 有明确帮助，而不是仅通过图传播间接影响。

### 9.4 Receiver Ablations

需要系统比较：

- `v1_linear` vs `v2_conditioned`
- `layernorm` vs `weighted_mean`
- zero-init vs small-init
- pool-only update vs all-node update
- train base model vs freeze base model

### 9.5 Pool Expansion Is Later

只有当固定正 scale 能在 train/val 上证明 prompt message 有用时，再考虑：

- 增大 `rho`；
- 增大 `topk_prompt_per_node`；
- 增大 `edge_scale_max`；
- 加强 prompt edge density。

否则扩大 pool 只会放大噪声。

