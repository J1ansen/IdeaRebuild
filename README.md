# GraphPromptBoostHomophily

## 1. 项目动机

本项目研究 **跨域少样本节点分类** 场景下，如何将源域预训练图神经网络迁移到目标图，尤其是迁移到 **异配图（heterophilic graph）** 上。

已有的图提示方法通常假设图结构对节点分类是有帮助的，即相邻节点往往具有相似标签或相似语义。在同配图中，这一假设通常成立，因此基于原始邻接关系的预训练、对比学习和结构一致性约束能够有效提升下游分类性能。

但在异配图中，这一假设并不稳定。异配图中的相邻节点可能来自不同类别，甚至图结构本身会把节点推向错误的语义方向。如果继续直接使用原图邻接关系进行强结构约束，模型可能会把异类邻居的信息错误地传播给目标节点，从而削弱分类效果。

因此，本项目的核心问题是：

> 在不破坏原始图结构、不依赖伪标签、不使用测试标签的前提下，如何为目标图中的不稳定节点提供额外的、可控的 prompt 信息，使适应分支能够更好地处理异配图中的结构差异？

本项目不是试图把异配图改造成同配图，也不是通过伪标签重新构造图结构。相反，本项目将 prompt 信息看作一种 **可接收、可拒绝、可缩放、可诊断的残差校正信号**，只作用于目标域适应分支，用于辅助模型判断哪些节点需要额外信息、应该连接哪些 prompt 节点、以及 prompt message 是否真的有助于分类。

---

## 2. 总体设计思路

给定输入图：

[
G=(V,E)
]

首先通过输入对齐模块将目标图节点特征映射到源域预训练模型的输入空间，得到节点表示：

$$
z_i = \text{Align}(x_i)
$$

随后，模型分为两个分支：

1. **冻结预训练分支（frozen pretrained branch）**
2. **目标域适应分支（adapted branch）**

冻结分支始终在原始图 (G=(V,E)) 上计算：

[
h_i^{pre} = \text{FrozenGNN}(z_i, E)
]

该分支的作用是保留源域预训练模型学到的稳定知识。它不接收 prompt 节点，不接收 prompt 边，也不参与 prompt 图传播。这样可以避免 prompt 图对源域预训练知识造成破坏。

适应分支则接收由 prompt 模块构造的增强图。该增强图并不修改原始图结构，而是在原始节点后追加一组可学习 prompt 节点：

[
P = {p_1, p_2, ..., p_M}
]

并在部分原始节点与 prompt 节点之间构造额外的 prompt edges。原始节点始终保留在表示矩阵的前 (N) 行，最终分类也只在原始节点上进行。

整个模型的最终表示由冻结分支和适应分支融合得到：

[
h_i = \alpha h_i^{pre} + (1-\alpha)h_i^{adp}
]

其中，(\alpha) 是可学习的融合系数。冻结分支提供稳定的源域知识，适应分支负责学习目标图中的结构差异和 prompt-based residual correction。

---

## 3. 核心原则

本项目的设计必须遵守以下原则：

### 3.1 不修改原始图结构

原始图中的边 (E) 不被删除、不被重连、不被替换。prompt edges 只是适应分支中的额外辅助边。

也就是说，模型不是在做 graph rewiring，而是在 adapted branch 中引入额外的 prompt message 通道。

### 3.2 不使用伪标签构图

候选连接池 pool 的选择不能依赖伪标签，也不能依赖 validation/test 标签。pool 的选择只能依赖训练节点和无标签结构统计量。

### 3.3 Prompt 只进入适应分支

冻结分支必须始终只使用原始图。prompt 节点和 prompt 边只能影响 adapted branch，不能影响 frozen branch。

### 3.4 Prompt message 必须可控

异配图中 prompt message 可能有害。因此，prompt 信息不能被无条件注入所有节点。模型必须能够控制：

* 哪些节点进入 prompt connection pool；
* 每个节点连接哪些 prompt；
* 每个节点是否真的接收 prompt message；
* prompt message 的强度有多大；
* prompt message 是否在训练中真正降低了分类损失。

### 3.5 当前方法是实验性框架

当前代码实现的是一个 heterophily-oriented prompt adaptation framework。它的目标不是直接宣称已经稳定超过所有 baseline，而是用于验证：

> prompt message 是否能作为一种有效的 residual correction 信号，帮助 adapted branch 在异配目标图上获得更好的分类性能。

---

## 4. 模块设计与实现方案

### 4.1 输入对齐模块

#### 设计动机

源域预训练 GNN 的输入维度和目标图的原始特征维度可能不同。为了复用源域预训练模型，需要先将目标图节点特征映射到源域输入空间。

#### 实现方案

给定目标图节点特征 (x_i)，通过 input aligner 得到：

[
z_i = \text{Align}(x_i)
]

如果源域和目标域输入维度一致，可以使用 identity mapping；如果维度不同，则使用线性映射或其他可学习对齐方式。

#### 期望效果

输入对齐模块应当使目标图节点能够被源域预训练 GNN 正常处理，同时尽量减少由特征维度差异带来的迁移障碍。

---

### 4.2 冻结预训练分支

#### 设计动机

GP2F 类方法的核心优势来自源域预训练图神经网络。即使目标图是异配图，预训练模型中仍可能包含有用的节点特征编码能力。因此，需要保留一个稳定的 frozen branch，作为整个模型的知识锚点。

#### 实现方案

冻结分支使用 frozen pretrained GNN，在原始图上计算：

[
h^{pre} = \text{FrozenGNN}(z, E)
]

该分支满足：

* 不更新 backbone 参数；
* 不接收 prompt nodes；
* 不接收 prompt edges；
* 始终只在原始图上进行传播。

#### 期望效果

冻结分支提供稳定的源域知识，防止目标域少样本训练过程中过拟合或被 prompt message 破坏。

---

### 4.3 适应分支

#### 设计动机

冻结分支虽然稳定，但它无法充分适应目标图中的异配结构。目标域可能存在与源域不同的结构模式、类别关系和邻域分布。因此，需要一个 adapted branch 学习目标域特有的信息。

#### 实现方案

适应分支复用预训练 GNN 的结构，但引入 adapter 和 prompt-aware message passing。它可以接收 prompt 模块构造的增强图：

[
G^{adp} = (V \cup P, E \cup E_{prompt})
]

其中，(P) 是 prompt nodes，(E_{prompt}) 是原始节点与 prompt 节点之间的 prompt edges。

适应分支输出：

[
h^{adp} = \text{AdaptedGNN}(z, P, E, E_{prompt})
]

最终只取前 (N) 个原始节点的表示用于分类。

#### 期望效果

适应分支应当学习目标图中的结构差异，并利用 prompt message 对部分结构不稳定节点进行残差校正。

---

## 5. Prompt 模块设计

### 5.1 Prompt nodes

#### 设计动机

在异配图中，直接依赖原图邻居可能会引入错误信息。prompt nodes 被设计为一组可学习的辅助节点，用于提供额外的信息通道。

它们不是原图中的真实节点，也不改变原图结构，而是作为 adapted branch 的辅助表示单元。

#### 实现方案

模型维护一组可学习 prompt nodes：

[
P = {p_1, p_2, ..., p_M}
]

这些 prompt nodes 被追加到原始节点之后。原始节点仍然位于前 (N) 行，prompt nodes 位于后 (M) 行。

#### 期望效果

prompt nodes 应当学习到目标图中某些可复用的结构模式、语义模式或校正方向，为原始节点提供补充信息。

---

### 5.2 候选连接池 pool

#### 设计动机

不是所有节点都需要 prompt 信息。如果对所有节点都连接 prompt nodes，可能会导致过度干预，尤其是在异配图中，错误的 prompt message 可能会损害分类性能。

因此，需要先从原始节点中选择一个候选连接池 pool，只让部分节点有机会连接 prompt。

#### 实现方案

pool 的选择不依赖伪标签、验证标签或测试标签，而是基于无标签结构不稳定性。

对每个节点 (i)，计算：

* 节点自身表示 (z_i)；
* 一跳邻域摘要 (m_i^{(1)})；
* 两步扩散摘要 (m_i^{(2)})；
* 节点与邻域摘要之间的差异；
* 邻域方差；
* 度数和结构角色统计量。

结构不稳定性可以理解为：

[
s_i = f(
1-\cos(z_i, m_i^{(1)}),
1-\cos(m_i^{(1)}, m_i^{(2)}),
\text{Var}(\mathcal{N}(i))
)
]

训练节点默认进入 pool，同时结构不稳定性较高的节点也会进入 pool。

#### 选择原因

异配图中，结构不稳定节点更可能受到原始邻域传播的负面影响，因此更可能需要额外 prompt message 进行校正。

#### 期望效果

pool 模块应当筛选出“可能需要辅助信息”的节点，而不是让 prompt 对全图进行无差别干预。

---

### 5.3 多视角 prompt routing

#### 设计动机

节点是否应该连接某个 prompt，不能只看单一特征。异配图中的节点可能出现以下情况：

* 特征相似但不相邻；
* 相邻但类别不同；
* 局部结构不稳定；
* 具有相似结构角色但语义不同；
* 具有相似属性但处在不同邻域环境中。

因此，prompt routing 需要从多个视角判断节点与 prompt 之间的匹配关系。

#### 实现方案

当前设计包含以下 routing views：

#### 5.3.1 语义视角 semantic view

语义视角使用节点自身表示与 prompt key 进行匹配：

[
r_{sem}(i,p) = \langle q_{sem}(z_i), k_p^{sem} \rangle
]

该视角主要刻画节点自身特征与 prompt 的语义相似性。

#### 期望效果

语义视角希望捕获“虽然在原图中不相连，但特征或语义上相似”的节点模式。

---

#### 5.3.2 结构上下文视角 structural-context view

结构上下文视角使用节点自身表示、一跳邻域摘要、两步扩散摘要及其差异：

[
[z_i, m_i^{(1)}, m_i^{(2)}, z_i - m_i^{(1)}, m_i^{(1)} - m_i^{(2)}]
]

该视角描述节点所在的局部结构环境。

#### 期望效果

结构上下文视角希望识别节点是否处在不稳定或异配的局部结构中，从而为其分配合适的 prompt。

---

#### 5.3.3 结构角色视角 structural-role view

结构角色视角使用度数、邻域差异、邻域方差、结构不一致性等统计量。

#### 期望效果

结构角色视角希望识别具有相似结构角色的节点。例如，一些节点可能虽然特征不同，但都处在高异配、高方差或高结构不一致的位置，这类节点可能需要类似的 prompt correction。

---

#### 5.3.4 属性视角 attribute view

在当前较新的 attribute-role variant 中，模型还可以引入 attribute view，直接从原始或对齐后的属性特征中学习 routing signal。

#### 期望效果

attribute view 用于补充 semantic/structural/role view，尤其是在结构信号不可靠时，让模型仍能利用节点属性信息进行 prompt routing。

---

### 5.4 View gate

#### 设计动机

不同节点需要依赖的视角不同。有些节点更依赖语义特征，有些节点更依赖结构上下文，有些节点更依赖结构角色。因此，不能手动固定各个 view 的权重。

#### 实现方案

模型使用节点级 view gate 自适应融合多个 routing views：

[
r(i,p) =
g_{sem}(i)r_{sem}(i,p)
+
g_{str}(i)r_{str}(i,p)
+
g_{role}(i)r_{role}(i,p)
+
g_{attr}(i)r_{attr}(i,p)
]

其中：

[
\sum_v g_v(i)=1
]

每个节点根据自身特征和结构上下文动态决定依赖哪些视角。

#### 期望效果

view gate 应当让模型对不同节点采取不同 routing 策略，而不是使用统一的 prompt 匹配规则。

---

### 5.5 Prompt slot 设计

当前代码中存在两类 prompt slot 设计。

---

#### 5.5.1 Class-aware routing anchors

#### 设计动机

为了增强 prompt routing 的可解释性，可以将前 (C) 个 prompt slots 作为 class-aware routing anchors，其中 (C) 是类别数。

这些 class-aware slots 并不使用 validation/test 标签，而是只用 training labels 进行初始化或辅助监督。

#### 实现方案

前 (C) 个 prompt slots 对应 (C) 个类别。它们可以由训练节点的结构查询原型初始化：

[
k_c = \frac{1}{|\mathcal{D}*{train}^c|}\sum*{i \in \mathcal{D}_{train}^c} q_i
]

其余 prompt slots 作为 residual prompt slots。

#### 期望效果

class-aware anchors 希望让一部分 prompt slot 具有明确语义，使 routing 更可解释；residual slots 则用于吸收无法明确归入某个类别结构的节点。

---

#### 5.5.2 Pattern prompt bank

#### 设计动机

在异配图中，“结构模式”和“类别”不一定强绑定。某些结构角色可能跨类别共享。如果强行使用 class-aware prompt anchors，可能会让 prompt 学到过强的类别偏置。

因此，较新的 P12 variant 引入 pattern prompt bank，不再强制 prompt slots 与类别一一对应，而是使用 label-free pool medoids 初始化 prompt patterns。

#### 实现方案

从 pool 中基于无标签表示选择若干 medoids，用于初始化 pattern prompt keys 或 prompt nodes。

#### 期望效果

pattern prompt bank 希望让 prompt slots 表示结构模式、属性模式或局部上下文模式，而不是直接表示类别。

---

## 6. Prompt edge 构造

### 6.1 Top-k prompt selection

对于 pool 中的每个节点，模型根据多视角 routing score 选择 top-(k) 个 prompt nodes：

[
\text{TopK}*i = \operatorname{topk}*{p \in P} r(i,p)
]

然后构造节点与 prompt 之间的连接。

### 6.2 双向或单向 prompt edges

早期设计中，prompt edges 可以是双向的：

[
i \rightarrow p, \quad p \rightarrow i
]

后续设计中，也可以只保留 prompt-to-node message，即：

[
p \rightarrow i
]

这样可以避免原始节点的信息反向污染 prompt nodes，使 prompt 更像一种外部 correction source。

### 6.3 Edge type 区分

适应分支显式区分三类边：

1. 原始图边；
2. node-to-prompt 边；
3. prompt-to-node 边。

原始图边仍由 adapted GNN 处理，prompt edges 则由 prompt-aware receiver 处理。

#### 期望效果

通过区分 edge type，模型不会把 prompt edges 当作普通图边处理，从而避免 prompt 信息和原图信息混杂。

---

## 7. Prompt-aware receiver

### 7.1 设计动机

如果直接把 prompt edges 加入图中，并让普通 GNN 进行传播，prompt nodes 可能会像普通邻居一样影响节点表示。这在异配图中风险较高，因为错误的 prompt message 会进一步放大噪声。

因此，需要单独设计 prompt-aware receiver，使 prompt message 以受控的残差形式注入 adapted branch。

### 7.2 实现方案

适应分支首先通过原始图边得到基础表示：

[
h_i^{base} = \text{AdaptedGNN}(h_i, E)
]

然后通过 prompt-aware receiver 计算 prompt message：

[
m_i^{prompt} = \text{Receiver}(h_i, h_p, e_{p \rightarrow i})
]

最后以 residual correction 的方式注入：

[
h_i^{adp} = h_i^{base} + \lambda_i \cdot s \cdot m_i^{prompt}
]

其中：

* (\lambda_i) 是节点级 receive gate；
* (s) 是 message scale；
* (m_i^{prompt}) 是 prompt receiver 产生的消息；
* prompt update 可以经过 normalization 和 norm bound 约束。

### 7.3 当前 receiver 设计

当前代码中已经尝试过多种 receiver，包括：

* linear receiver；
* conditioned receiver；
* node residual receiver；
* multi-expert residual receiver；
* prototype-directional receiver；
* classifier-directional receiver。

较新的 P11/P12 设计倾向于使用 classifier-directional 或 directional residual correction，使 prompt message 更像是朝分类器有利方向的可控修正，而不是普通邻居聚合。

#### 期望效果

prompt-aware receiver 应当实现：

> prompt 不直接替代图传播，而是作为 adapted branch 的残差校正项，只在有用时对节点表示进行小幅、有界、可控的修正。

---

## 8. 节点级提示接收门控

### 8.1 设计动机

异配图中 prompt message 并不总是有用。如果 prompt message 对某些节点有害，模型应当允许这些节点拒绝 prompt，而不是强制所有 pool 节点接收 prompt。

### 8.2 实现方案

模型比较 prompt-on 分支和 no-prompt 分支在训练节点上的表现。

如果 prompt-on 相比 no-prompt 降低了分类交叉熵：

[
\Delta CE_i = CE_i^{no\ prompt} - CE_i^{prompt} > 0
]

则说明 prompt 对该节点可能有帮助，模型鼓励该节点接收 prompt。

如果：

[
\Delta CE_i < 0
]

则说明 prompt 可能有害，模型鼓励该节点拒绝或降低 prompt message。

在当前较新的代码中，这一思想主要通过 utility receive gate、prompt message help loss、query proto alignment 等辅助目标实现。

### 8.3 期望效果

接收门控应当实现：

* 对 prompt 有收益的节点打开 gate；
* 对 prompt 有害的节点关闭 gate；
* 避免 prompt message 对全图产生无差别干预；
* 让 prompt branch 在必要时退回接近 no-prompt 行为。

---

## 9. Support-query prompt supervision

### 9.1 设计动机

少样本场景中，直接用全部训练节点监督 prompt routing 容易过拟合。为了让 prompt 模块学习更具有泛化性的 routing 和 receive 策略，当前代码中引入了 train 内部的 support-query split。

### 9.2 实现方案

训练集内部被划分为：

* support nodes；
* query nodes。

prompt graph 可以只基于 support nodes 构造，而 query nodes 用于诊断 prompt 是否能泛化到未直接参与 prompt 构造的训练节点。

### 9.3 需要注意的边界

当前代码中的 support-query 机制主要是 train-internal diagnostic。由于主分类交叉熵仍可能在完整 train mask 上计算，因此它还不是严格意义上的 meta-learning support-query episode。

如果后续要证明 prompt 能从 support 泛化到 query，应当增加一个严格版本：

> 主 CE 只在 support nodes 上计算，query nodes 只用于内部验证 prompt utility。

### 9.4 期望效果

support-query 机制希望降低 prompt routing 对训练标签的过拟合，并观察 prompt message 是否能在训练内部表现出一定泛化能力。

---

## 10. 最终分类

最终模型融合冻结分支和适应分支：

[
h_i = \alpha h_i^{pre} + (1-\alpha)h_i^{adp}
]

然后使用分类器预测节点类别：

[
\hat{y}_i = \text{Classifier}(h_i)
]

其中：

* (h_i^{pre}) 来自冻结分支，只依赖原始图；
* (h_i^{adp}) 来自适应分支，可以接收 prompt message；
* (\alpha) 控制源域知识和目标域适应之间的平衡。

---

## 11. 当前实验效果

当前代码已经实现了以下设计：

1. GP2F-style frozen branch 与 adapted branch；
2. input alignment；
3. residual adapter；
4. prompt nodes；
5. label-free structural pool selection；
6. multi-view routing；
7. class-aware routing anchors；
8. residual prompt slots；
9. pattern prompt bank；
10. prompt edge type 区分；
11. prompt-aware receiver；
12. prompt receive gate；
13. prompt-on 与 no-prompt CE delta 诊断；
14. support-query split；
15. message scale grid；
16. 多种 prompt receiver ablation。

目前可以确认的是：

* prompt graph 构造逻辑已经基本完整；
* prompt nodes 不会进入 frozen branch；
* 原始节点顺序被保留；
* prompt edges 只影响 adapted branch；
* routing、receive gate、message norm、message scale 等诊断指标已经具备；
* 训练节点上的 routing consistency 可以被提高；
* prompt message 的作用可以被诊断和消融。

但当前结果尚未证明 prompt message 能在异配目标图上稳定带来 validation/test gain。部分实验中，验证集会选择较小的 message scale，甚至选择接近 no-prompt 的设置。这说明当前 prompt message 还没有稳定成为有效的分类增益来源。

因此，当前方法不能表述为“已经解决异配图跨域迁移问题”，而应表述为：

> 当前方法实现了一个面向异配图的 prompt-based adaptation framework，用于研究 prompt message 是否能够作为可控 residual correction，帮助 adapted branch 在跨域少样本节点分类中获得更好的泛化性能。

---

## 12. 当前方案可能失效的原因

### 12.1 Prompt routing 不等于 prompt message 有用

当前 routing 模块可以把节点分配给 prompt nodes，但“分配合理”并不必然意味着 prompt message 会降低分类损失。

也就是说，模型可能学到了稳定的 routing pattern，但 receiver 产生的 message 仍然不能有效改善节点表示。

### 12.2 异配图中的类别与结构模式不一致

在同配图中，类别、邻域和结构通常具有较强一致性。但在异配图中，相同类别节点可能不相邻，相邻节点也可能不同类。

因此，class-aware prompt anchors 可能会失效，因为类别原型和结构原型之间不一定一致。

### 12.3 Prompt message 容易引入噪声

如果 prompt message 被当作普通邻居信息传播，它可能进一步放大异配图中的错误结构信号。因此，即使 prompt graph 构造合理，message passing 仍可能损害分类性能。

这也是当前代码中引入 prompt-aware receiver、message scale、receive gate 和 bounded residual update 的原因。

### 12.4 Receive gate 容易退化

如果 prompt message 整体不稳定，receive gate 可能学会关闭大部分 prompt message。此时模型退化为 no-prompt 或 near-no-prompt 行为。

这种退化并不一定是坏事，因为它说明模型避免了有害 prompt；但它也说明 prompt branch 尚未提供稳定收益。

### 12.5 CE delta 监督可能过拟合训练节点

当前 prompt utility 的判断主要来自训练节点上的 prompt-on 与 no-prompt CE 差异。这个信号可能对训练节点有效，但不一定能泛化到 validation/test 节点。

因此，需要进一步验证：

* train pool 上 prompt 是否降低 CE；
* query pool 上 prompt 是否降低 CE；
* validation/test pool 上是否也有类似趋势；
* utility gate score 是否与真实 CE delta 相关。

### 12.6 模块过多导致优化困难

当前系统包含 routing、view gate、prompt keys、prompt nodes、receive gate、receiver、message scale、adapter、fusion alpha 以及多个辅助 loss。模块过多可能导致优化目标互相干扰。

因此，后续实验应优先验证最小有效机制，而不是继续增加模块复杂度。

---

## 13. 后续实验重点

后续实验应优先回答一个核心问题：

> 在固定 positive message scale 的情况下，prompt-on 是否能够稳定降低 pool nodes 上的分类 CE？

建议优先观察以下指标：

1. (CE^{no\ prompt} - CE^{prompt}) 的分布；
2. train/support/query/validation/test pool 上的 CE delta；
3. utility receive gate score 与 CE delta 的 Pearson/Spearman correlation；
4. 不同 message scale 下 prompt update norm 与 test accuracy 的关系；
5. prompt receiver 是否真正产生非零且有方向性的 residual correction；
6. class-aware anchors 与 pattern prompt bank 哪个更适合异配图；
7. support-only prompt graph 是否能泛化到 query nodes。

---

## 14. 后续 AI 修改代码时必须遵守的约束

如果后续使用 AI 继续修改本项目代码，必须遵守以下约束：

1. 不允许修改 frozen branch 的核心逻辑。frozen branch 必须始终在原始图上计算。
2. 不允许让 prompt nodes 或 prompt edges 进入 frozen branch。
3. 不允许删除、重连或替换原始图边。
4. 不允许使用 validation/test labels 构造 prompt graph。
5. 不允许使用 pseudo-labels 构造 prompt edges，除非明确新增为单独 ablation。
6. prompt edges 只能作为 adapted branch 的辅助边。
7. 原始节点必须始终保留在前 (N) 行，最终分类只能作用于原始节点。
8. prompt message 必须通过 receiver、gate、message scale 或 residual bound 控制，不能无约束注入。
9. 任何新的 prompt 模块都必须和 no-prompt baseline 对比。
10. 任何声称 prompt 有效的实验都必须报告 prompt-on 与 no-prompt 的 CE delta。
11. 如果引入 support-query 机制，必须明确 query 是否参与主 CE 训练。
12. README 中不能声称当前方法已经稳定解决异配图问题，除非有充分实验结果支持。

---

## 15. 当前方法的一句话总结

本项目的核心设计是：

> 在跨域少样本节点分类中，保留一个始终运行在原始图上的冻结预训练分支，同时为目标域适应分支构造一个由结构不稳定节点、多视角 prompt routing、可学习 prompt nodes、节点级 receive gate 和 prompt-aware residual receiver 组成的辅助 prompt graph，使 prompt message 作为可接收、可拒绝、可缩放、可诊断的 residual correction 信号，而不是无约束地改变原始图结构。
