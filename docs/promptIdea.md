# Unified Minimal Multi-view Residual Prompt Design

## 1. Final Position

第一版提示模块命名为：

```text
Unified Minimal Multi-view Residual Prompt
```

它不是：

```text
topology prompt
prompt edge learning
graph rewiring
homophily boosting
```

它的核心定位是：

```text
为 GP2F adapted branch 注入可控的 residual auxiliary representation。
```

也就是说，P0 不显式构造 prompt node 与原图节点之间的边，不学习新的图拓扑，也不声称提升原图同配性。它只修改 adapted branch 的输入表示：

```text
z_prompted = z + gamma * u_prompt
```

其中：

```text
u_prompt =
  g_sem    * u_sem
+ g_struct * u_struct
+ g_null   * 0
```

三路视角为：

```text
Semantic View
+ Structural Context View
+ Null / Soft No-prompt Route
```

主张：

```text
当同配先验在异配图上失效时，模型不应强行重连或强制同类聚合；
它应在语义辅助、结构上下文辅助和不接收提示之间自适应选择。
```

## 2. Why Not Claim Homophily Boosting

P0 不改变 `edge_index`，因此原图 edge homophily 不会变化：

```text
H_edge = # same-label original edges / # original edges
```

由于原图边没有改变，不能写：

```text
P0 提升异配图同配性。
P0 重构更同配的目标图。
P0 学习更可靠的提示连边。
```

更准确的表述是：

```text
P0 improves task-oriented residual conditioning for the adapted branch.
P0 provides semantic and structural-context auxiliary representations.
P0 preserves the original graph while allowing soft prompt suppression.
```

如果后续实现 edge-based prompt ablation，可以额外报告：

```text
prompt-mediated homophily
prompt-mediated same-class reachability
```

但这属于边提示消融，不属于 P0 主方法。

## 3. Design Principles

P0 必须满足以下原则：

```text
1. frozen branch 始终只使用 original graph。
2. prompt 只影响 adapted branch。
3. 不使用伪标签构图。
4. 不用高置信预测决定 prompt 连边。
5. 不强制节点选择真实类别对应的 class prompt。
6. 不默认新增 prompt node 或 prompt edge。
7. 不按完整标签同配率手工设置数据集特定 rho。
8. 主实验使用统一 CE-only paired comparison。
9. Null route 是 soft suppression，不是 hard rejection。
10. prompt 初始应严格或近似等价于 NoPrompt。
11. prompt evidence views 默认使用 detached `h_pre` / `z`，避免提示模块反向塑造其选择证据。
12. Structural View 默认基于 `h_pre.detach()`，`z.detach()` 作为消融。
13. 主实验必须包含参数量匹配的 non-prompt residual control。
14. route budget 只能通过 validation 选择，test labels 不参与任何选择。
15. zero-init 后必须通过唤醒测试，证明初始等价 NoPrompt 且训练后 prompt 可被激活。
```

## 4. Architecture Overview

当前已实现的 GP2F baseline 保持不变。

```text
Frozen Branch:
  input: z, original_edge_index
  output: h_pre
  role: preserve pretrained/source-domain transfer ability

Adapted Branch:
  input: z_prompted, original_edge_index
  output: h_adp
  role: receive residual prompt-conditioned representation
```

整体流程：

```text
z = input_aligner(x)

h_pre = FrozenBranch(z, original_edge_index)

prompt_out = PromptModule(
  z=z,
  h_pre=h_pre,
  edge_index=original_edge_index,
  train_mask=train_mask,
  y=y,
)

z_prompted = prompt_out["adapted_x"]

h_adp = AdaptedBranch(z_prompted, original_edge_index)

h_mix = alpha * h_pre + (1 - alpha) * h_adp
```

第一版实现中：

```text
adapted_edge_index = original_edge_index
```

不新增：

```text
prompt_edge_index
prompt_node_x
```

### 4.1 Gradient Boundary and Trainable Parameters

当前 baseline 中：

```text
Frozen pretrained backbone:
  frozen, eval mode, parameters require_grad = False

Trainable baseline parameters:
  input_aligner
  adapted branch adapters
  classifier
  fusion alpha
```

P0 新增：

```text
Trainable prompt parameters:
  semantic output projection
  structural MLP / projection
  view gate
  gamma_logit
```

`h_pre` 有两个角色，必须区分：

```text
1. h_pre used in GP2F fusion:
   follows the baseline forward path.

2. h_pre used by PromptModule as evidence:
   default uses h_pre_for_prompt = h_pre.detach().
```

`z` 也有两个角色：

```text
1. z used as adapted branch base input:
   remains trainable through the original baseline path.

2. z used by PromptModule for prototypes / scores:
   default uses z_for_prompt = z.detach().
```

默认设计理由：

```text
PromptModule 可以学习如何使用证据；
但不应通过 prompt routing/statistics 分支反向改变证据本身。
```

因此，P0 默认梯度边界为：

```text
CE loss -> input_aligner/adapters/classifier/alpha through normal model path
CE loss -> prompt module parameters through u_prompt path
No gradient from PromptModule evidence statistics -> frozen backbone
No gradient from PromptModule evidence statistics -> input_aligner
```

可以作为消融打开：

```text
allow_prompt_to_aligner_grad: true
```

但不作为 P0 主方法默认设置。

## 5. Semantic View

### 5.1 Purpose

Semantic View 保留语义或表示相似信息，但它只是弱证据，不是连边规则，也不是伪标签生成器。

它适合：

```text
同配图中少数需要语义辅助的疑难节点。
```

但在异配图上：

```text
semantic similarity 不能被解释为 same-label evidence。
```

因此 Semantic View 只产生 residual message，并可被 gate 降权或关闭。

### 5.2 Class Prototypes

使用训练节点构造类别语义原型：

```text
z_proto_c = mean(z_for_prompt_i), where train_mask_i = True and y_i = c
h_proto_c = mean(h_pre_for_prompt_i), where train_mask_i = True and y_i = c
```

Semantic score：

```text
s_sem(i, c) =
  beta_z * cosine(z_i, z_proto_c)
+ beta_h * cosine(h_pre_i, h_proto_c)
```

Semantic attention：

```text
a_sem(i, :) = softmax(s_sem(i, :) / tau_sem)
```

Semantic message：

```text
u_sem_i = sum_c a_sem(i, c) * z_proto_c
```

注意：

```text
1. 不使用 argmax(q_i)。
2. 不使用 hard pseudo-label。
3. 不把 unlabeled node 直接连接到 class prompt。
4. 不声称 prototype similarity 等于同类关系。
```

### 5.3 Leave-one-out Prototype for Train Nodes

训练阶段，如果某个训练节点 `i` 参与分类损失，则它不应该从包含自身的类别原型中获得 semantic support。

因此对 train node 使用 leave-one-out prototype：

```text
h_proto / z_proto 使用 detached prompt evidence 计算。

```text
z_proto_{y_i}^{-i}
  = mean(z_for_prompt_k), where train_mask_k = True, y_k = y_i, k != i

h_proto_{y_i}^{-i}
  = mean(h_pre_for_prompt_k), where train_mask_k = True, y_k = y_i, k != i
```

验证和测试阶段使用完整训练集原型：

```text
z_proto_c = mean(train nodes of class c)
h_proto_c = mean(train nodes of class c)
```

这样可以避免训练节点获得 self-support shortcut。

### 5.4 1-shot Handling

如果 `shots = 1`，leave-one-out 后该类没有剩余 anchor。

P0 采用保守处理：

```text
For train nodes in 1-shot:
  Semantic View is masked or set to zero.

For val/test nodes:
  Semantic View can use the single train anchor as prototype.
```

这避免训练节点通过自身 prototype 作弊。

## 6. Structural Context View

### 6.1 Purpose

Structural Context View 是 P0 面向异配图的关键视角。

它不假设：

```text
一跳邻居与目标节点同类。
二跳节点与目标节点同类。
低相似度邻居一定有价值。
```

它只表达：

```text
节点所处的局部结构上下文模式。
```

在异配图中，邻接关系本身可能是判别性结构模式的一部分，因此结构上下文不应被简单当作噪声或同类支持。

### 6.2 One-step and Two-step Diffusion Summaries

默认基础表示：

```text
h_i = h_pre_for_prompt_i = detach(h_pre_i)
```

选择 `h_pre.detach()` 作为默认结构视角基础，原因是：

```text
1. hidden_dim 通常较小，参数量更可控。
2. 它保留了源域预训练得到的稳定表示。
3. detach 后 Structural View 不会反向塑造 frozen evidence。
```

但这并不表示 `h_pre` 在异配图上一定可靠。因此需要保留消融：

```text
structural_base = z_detached
structural_base = h_pre_detached
```

主方法默认：

```text
structural_base = h_pre_detached
```

一跳扩散摘要：

```text
m1_i = mean(h_j), where j in N(i)
```

两步结构扩散摘要：

```text
m2_i = mean(m1_j), where j in N(i)
```

重要命名：

```text
m2_i is a two-step structural diffusion summary.
```

它不是严格的：

```text
exact 2-hop node-set summary
```

因为它可能包含：

```text
1. 回到自身的信息。
2. 重复覆盖一跳邻居的信息。
3. degree-normalized diffusion effect。
```

严格二跳集合摘要、DFS context、random-walk context 都放到后续消融。

### 6.3 Context Feature

结构上下文输入：

```text
context_i = [
  h_i,
  m1_i,
  m2_i,
  h_i - m1_i,
  m1_i - m2_i
]
```

结构消息：

```text
u_struct_i = MLP_struct(context_i)
```

输出维度：

```text
u_struct_i has the same dimension as z_i
```

实现时使用 bottleneck projection：

```text
context_i -> hidden_dim -> source_dim
```

原因：

```text
跨域设置下 source feature space 可能较高维；
few-shot 下直接使用大 MLP 容易过拟合。
```

## 7. Null / Soft No-prompt Route

Null route 是 P0 主方法核心。

它的输出为：

```text
u_null_i = 0
```

但由于 P0 使用 softmax gate：

```text
[g_sem_i, g_struct_i, g_null_i] = softmax(...)
```

因此：

```text
g_null_i high
```

只表示：

```text
soft no-prompt preference
soft prompt suppression
```

不表示：

```text
hard rejection
strict sparse intervention
node receives exactly zero prompt
```

如果未来要主张真正的稀疏拒绝，需要额外加入：

```text
hard mask
top-budget selection
straight-through gate
Gumbel / concrete gate
thresholded inference
```

P0 暂时不使用这些机制，以保持训练稳定。

## 8. View Gate

每个节点学习一个三路 gate：

```text
[g_sem_i, g_struct_i, g_null_i] =
  softmax(Gate(context_gate_i))
```

推荐 gate 输入：

```text
context_gate_i = [
  h_pre_i,
  m1_i,
  m2_i,
  h_pre_i - m1_i,
  semantic_margin_i
]
```

其中：

```text
semantic_margin_i =
  top1(s_sem(i, :)) - top2(s_sem(i, :))
```

P0 默认不把以下信息作为 gate 输入：

```text
hard prediction
pseudo-label
test-label-derived homophily
```

最终 prompt message：

```text
u_prompt_i =
  g_sem_i    * u_sem_i
+ g_struct_i * u_struct_i
+ g_null_i   * 0
```

注入 adapted branch：

```text
z_prompted_i = z_i + gamma * u_prompt_i
```

## 9. Prompt Strength and Zero Initialization

仅使用较小的 `gamma_init` 不足以保证初始扰动小，因为：

```text
||gamma * u_prompt|| may still be large if ||u_prompt|| is large.
```

P0 要求最终输出投影层零初始化：

```text
Semantic output projection:
  zero init final layer

Structural output projection:
  zero init final layer
```

训练初始应满足：

```text
u_sem = 0
u_struct = 0
u_prompt = 0
z_prompted = z
```

因此 P0 初始化时严格或近似等价于 NoPrompt baseline。

Prompt strength：

```text
gamma = gamma_max * sigmoid(gamma_logit)
```

推荐：

```yaml
prompt:
  gamma_init: 0.05
  gamma_max: 0.5
  learnable_gamma: true
```

必须记录：

```text
prompt_message_norm = ||gamma * u_prompt|| / ||z||
```

初始化测试：

```text
At init:
  max_abs(z_prompted - z) ~= 0
  logits_p0 ~= logits_noprompt
```

## 10. Intervention Budget

因为 P0 使用 soft gate 而不是 hard prompt edge，干预程度通过 soft non-null ratio 控制。

定义：

```text
non_null_i = g_sem_i + g_struct_i
```

整体干预比例：

```text
non_null_ratio = mean_i non_null_i
```

使用统一 route budget 搜索空间：

```yaml
prompt:
  route_budget_grid: [0.0, 0.05, 0.10, 0.20, 0.35, 0.45]
```

由 validation performance 选择：

```text
rho_budget is selected by validation, not by full-label graph homophily.
```

可选预算正则：

```text
L_budget = max(0, mean(non_null_i) - rho_budget)^2
```

默认：

```yaml
prompt:
  lambda_budget: 0.01
```

注意：

```text
rho_budget 是 soft intervention budget；
不是连接池比例；
不是原图同配率；
不是 prompt edge capacity。
```

### 10.1 Validation-only Budget Selection

route budget 的选择必须只使用 validation split。

固定搜索空间：

```text
route_budget_grid = [0.0, 0.05, 0.10, 0.20, 0.35, 0.45]
```

每个 dataset / split seed 的流程：

```text
1. 固定 train / val / test split。
2. 对每个 route_budget 训练一个 P0 run。
3. 只根据 validation metric 选择 best budget 和 best epoch。
4. 使用该 run 对应 checkpoint 汇报 test accuracy / macro-F1。
5. test labels 不参与 budget、epoch、threshold 或 config 选择。
```

选择指标必须在实验前固定：

```text
selection_metric = val_acc or val_macro_f1
```

推荐：

```text
selection_metric = val_acc
```

原因：

```text
当前 baseline runner 已主要使用 validation accuracy early stopping；
第一版保持协议连续，macro-F1 作为报告指标。
```

必须记录：

```text
selected_route_budget
selected_epoch
selection_metric
validation_score
```

## 11. Loss and Experiment Protocol

### 11.1 Main Objective

主任务：

```text
L_cls = CE(logits[train_mask], y[train_mask])
```

P0 主损失：

```text
L = L_cls + lambda_budget * L_budget
```

P0 不使用：

```text
1. pseudo-label graph construction loss
2. prompt assignment supervision
3. forced class prompt supervision
4. prompt edge topology loss
5. original GP2F adjacency-driven contrastive loss
6. original GP2F adjacency-driven topology fusion loss
```

### 11.2 CE-only Paired Comparison

主实验必须使用统一协议：

```text
NoPrompt + CE-only
vs
P0 + CE-only
```

所有数据集一致，包括：

```text
Cora
CiteSeer
PubMed
Actor
chameleon
squirrel
```

这样才能清楚回答：

```text
prompt module 本身是否有效？
```

不能在主实验中按图类型切换：

```text
同配图开 GP2F structure losses
异配图关 GP2F structure losses
```

否则提升会混入 loss protocol 变化。

### 11.3 Structure-loss Baselines

修复后的 GP2F structure-loss 版本可以作为独立竞争基线或补充实验：

```text
NoPrompt + GP2F-CTR
NoPrompt + GP2F-FUS
NoPrompt + GP2F-CTR+FUS
P0 + GP2F-CTR/FUS variants
```

但不能作为 P0 主实验默认设置。

### 11.4 Parameter-matched Non-prompt Control

为了证明提升不是来自额外参数量，主实验或补充实验必须包含参数量匹配的非 prompt 对照。

建议对照：

```text
ParamMatchedResidualControl
```

形式：

```text
u_control_i = MLP_control(self_context_i)
z_control_i = z_i + gamma_control * u_control_i
```

其中：

```text
self_context_i:
  只包含节点自身表示或随机置换后的上下文；
  不使用 train labels；
  不使用 semantic prototypes；
  不使用 graph diffusion summaries。
```

参数量应尽量匹配：

```text
MLP_control parameter count ~= PromptModule parameter count
```

至少需要报告：

```text
NoPrompt
ParamMatchedResidualControl
Full-P0
```

如果：

```text
ParamMatchedResidualControl 与 Full-P0 提升接近
```

说明收益可能主要来自额外参数或 residual MLP，而不是多视角 prompt 设计。

## 12. Label Access Isolation

标签访问必须隔离：

```text
train labels:
  CE loss
  semantic prototype construction
  leave-one-out train prototype

validation labels:
  early stopping
  route budget selection
  hyperparameter selection

test labels:
  final reporting only
```

禁止：

```text
1. 使用 test labels 计算 rho / route budget。
2. 使用完整标签同配率设定数据集特定配置。
3. 使用 val/test labels 构造 semantic prototypes。
4. 使用 val/test labels 训练 gate 或 prompt assignment。
5. 在训练期间根据 test metrics 选择 checkpoint。
```

允许但必须标注为 post-hoc reporting：

```text
semantic_route_label_agreement
dominant_route_accuracy
route_usage_by_true_class
```

## 13. Zero-init Wake-up Tests

P0 要求 zero-init 后初始等价 NoPrompt，但也必须证明 prompt 可以被训练唤醒。

初始化等价测试：

```text
At initialization and eval mode:
  adapted_x == z
  max_abs(adapted_x - z) <= eps
  logits_p0 ~= logits_noprompt
```

推荐阈值：

```text
eps = 1e-6 for adapted_x
logit tolerance = 1e-5 to 1e-4 depending on dropout/eval mode
```

唤醒测试：

```text
After 1-2 optimizer steps on toy graph:
  ||gamma * u_prompt|| / ||z|| becomes > 0
  prompt output projection receives non-zero gradients
  forward/backward has no NaN
```

这避免两种失败：

```text
1. prompt 初始扰动破坏 NoPrompt baseline。
2. zero-init 后 prompt 永远无法被激活。
```

## 14. Data Flow

### 14.1 Prompt Module Inputs

```python
prompt_out = prompt_module(
    z=z,
    h_pre=h_pre,
    edge_index=edge_index,
    train_mask=train_mask,
    y=y,
    split="train" | "eval",
)
```

### 14.2 Prompt Module Outputs

```python
{
    "adapted_x": z_prompted,
    "u_sem": u_sem,
    "u_struct": u_struct,
    "u_prompt": u_prompt,
    "gate": gate,
    "gamma": gamma,
    "aux": aux,
}
```

其中：

```text
gate[:, 0] = semantic route weight
gate[:, 1] = structural route weight
gate[:, 2] = soft null route weight
```

### 14.3 Dual Branch Forward

当前 `FaithfulGP2F.forward()` 已支持 `adapted_x`，因此 P0 可以调用：

```python
logits, h_pre, h_adp, h_mix, alpha = model(
    z,
    edge_index,
    adapted_x=prompt_out["adapted_x"],
    adapted_edge_index=edge_index,
)
```

不修改：

```text
FaithfulGP2F 主体逻辑
pretrained backbone loading
frozen branch behavior
```

## 15. What to Keep

保留：

```text
1. frozen branch only original graph。
2. prompt only affects adapted branch。
3. small learnable gamma。
4. soft intervention budget。
5. semantic margin diagnostic。
6. no Gumbel in P0。
7. no pseudo-label graph construction。
8. no prompt edges in P0 main method。
9. CE-only paired comparison as main protocol。
```

## 16. What to Remove or Downgrade

从 P0 主方法中删除：

```text
1. 未标记节点依据 prototype similarity 直接连接 class prompt。
2. 高置信预测或伪标签决定 class prompt 连边。
3. 强制 labeled query 选择真实类别 prompt。
4. 按完整标签同配率手工设置不同数据集 rho。
5. 声称提升原图同配性。
6. 声称拓扑重构或连边学习。
```

降级为消融或后续扩展：

```text
Class Prototype Prompt Hub:
  Semantic / Same-support ablation。

Prompt Edges:
  Edge-based prompt ablation。

Bidirectional Prompt Edges:
  Prompt edge 消融，不作为主方法。

DFS / BFS Context:
  Structural Context View 之后的扩展消融。

Global Structural Role:
  P1/P2。

Class Compatibility Matrix:
  P2。

Hard Sparse Gate / Gumbel:
  P1/P2。
```

## 17. HeterGP Reference Boundary

HeterGP 的参考价值：

```text
图中可能同时存在同质和异质信息；
单一同配视角不足；
多视图结构上下文可能适合异配图。
```

不能写成：

```text
DFS 能找同类节点。
BFS 是可靠同配邻域。
低相似度就是有用异配连接。
```

P0 不直接实现 DFS。

推荐扩展顺序：

```text
P0:
  one-step and two-step structural diffusion summaries

P1:
  DFS / random-walk context view

P2:
  compare diffusion context vs DFS/random-walk context
```

## 18. Diagnostics

除 accuracy 和 macro-F1 外，必须记录：

```text
route_usage:
  mean(g_sem), mean(g_struct), mean(g_null)

route_usage_by_class:
  per-class route usage

route_usage_by_split:
  train / val / test route usage

gamma:
  prompt message strength

prompt_message_norm:
  ||gamma * u_prompt|| / ||z||

non_null_ratio:
  mean(g_sem + g_struct)

semantic_margin:
  top1-top2 semantic score margin

view_output_norm:
  ||u_sem||, ||u_struct||, ||u_prompt||

init_equivalence:
  max_abs(z_prompted - z) at initialization

connected_edge_count:
  P0 should be 0
```

实验后可用完整标签做 reporting，但不能参与训练或构图：

```text
semantic_route_label_agreement
dominant_route_accuracy
route_usage_by_true_class
```

## 19. Minimal Ablation Plan

第一轮消融：

```text
NoPrompt:
  CE-only GP2F dual-branch baseline。

SemanticOnly:
  Semantic View + soft Null route。

StructuralOnly:
  Structural Context View + soft Null route。

Semantic+Structural-NoNull:
  两路辅助，不启用 Null route。

Full-P0:
  Semantic View + Structural Context View + soft Null route。

NoBudget:
  移除 route budget。

NoZeroInit:
  检查初始化扰动影响。

NoLOOPrototype:
  检查 train-node self-support shortcut。

EdgePrototypeAblation:
  旧版 class prototype prompt edge。
```

需要观察：

```text
1. 同配图上 Null route 是否占比较高。
2. 异配图上 Structural route 是否更常被使用。
3. SemanticOnly 是否在异配图上不稳定。
4. Full-P0 是否比强制 prompt 更稳。
5. zero-init 是否保证初始等价 NoPrompt。
6. leave-one-out 是否降低 train-test mismatch。
```

## 20. Default Config Sketch

```yaml
prompt:
  enabled: true
  variant: unified_multiview_residual_p0

  gradient:
    h_pre_for_prompt: detach
    z_for_prompt: detach
    allow_prompt_to_aligner_grad: false

  semantic:
    enabled: true
    use_feature_proto: true
    use_hidden_proto: true
    beta_z: 0.5
    beta_h: 1.0
    tau_sem: 0.5
    use_leave_one_out_train_proto: true
    one_shot_train_semantic: mask
    use_margin_diagnostic: true

  structural:
    enabled: true
    base: h_pre_detached
    base_ablation_options: [h_pre_detached, z_detached]
    summary: two_step_diffusion
    hidden_dim: 128
    dropout: 0.3
    bottleneck: true
    zero_init_output: true

  gate:
    hidden_dim: 64
    dropout: 0.3
    null_bias_init: 1.0

  gamma_init: 0.05
  gamma_max: 0.5
  learnable_gamma: true
  zero_init_prompt_output: true

  route_budget_grid: [0.0, 0.05, 0.10, 0.20, 0.35, 0.45]
  route_budget: 0.10
  route_budget_selection: validation_only
  selection_metric: val_acc
  lambda_budget: 0.01

controls:
  parameter_matched_residual:
    enabled: true
    use_labels: false
    use_graph_diffusion: false
    match_prompt_parameter_count: approximate

labels:
  train_labels_for: [classification_loss, semantic_prototypes]
  val_labels_for: [early_stopping, route_budget_selection]
  test_labels_for: [final_reporting]

loss:
  use_original_contrastive: false
  use_original_topology_fusion: false
```

## 21. Implementation Notes

建议新增：

```text
models/prompt_module.py
```

类名：

```text
UnifiedMultiViewResidualPrompt
```

核心要求：

```text
1. 与 FaithfulGP2F 解耦。
2. 不修改 frozen branch。
3. 不默认生成 prompt edge。
4. 支持 train/eval prototype 行为差异。
5. 支持 leave-one-out train prototype。
6. 支持 1-shot semantic masking。
7. 支持 zero-init output projection。
8. 输出 route diagnostics。
9. 支持 validation-only route budget selection。
10. 支持 parameter-matched residual control。
11. 明确隔离 train / val / test label access。
```

测试要求：

```text
1. zero-init 时 adapted_x 等于 z。
2. zero-init 时 P0 logits 近似 NoPrompt logits。
3. gate shape 为 [N, 3]。
4. gate 每行 sum 为 1。
5. no prompt edge in P0。
6. train leave-one-out prototype 不包含自身。
7. 1-shot train semantic route 被 mask。
8. forward/backward 无 NaN。
9. PromptModule evidence inputs 默认 detach。
10. Structural View 默认基于 h_pre_detached。
11. route budget selection 不读取 test labels。
12. parameter-matched control 参数量近似匹配。
13. zero-init 后 1-2 步 prompt output 可被唤醒。
```

## 22. Final Summary

最终 P0 是：

```text
Unified Minimal Multi-view Residual Prompt
```

它通过：

```text
Semantic View:
  提供弱语义辅助，但不做 hard class routing。

Structural Context View:
  提供一跳与两步结构扩散摘要，不假设同类。

Soft Null Route:
  允许模型软抑制 prompt，避免无依据干预。
```

解决的问题是：

```text
在同配先验失效时，如何为 adapted branch 提供可控、可拒绝、可解释的辅助表示。
```

不解决或不主张：

```text
显式图重构。
学习新连边。
提升原图同配性。
伪标签构图。
强制同类 prompt 连接。
```

这版设计可以作为下一步代码实现的依据。
