
# p1:构造 prompt nodes / prompt edges
# p2:让 adapted branch 能“识别并专门处理 prompt 边”。

目标：完善 P2 prompt graph，使池内节点与提示节点的连接不仅由 learned structural similarity 决定，还受到 train-only 类别一致性约束，从而尽量让相同类别的池内节点连接到相同或相近的提示节点，同时保留异配图上的拒绝机制，避免有害 prompt 干预。
1. 原图约束
- frozen branch 始终只使用 original_edge_index。
- 原图节点之间的 original_edge_index 不修改、不重连、不删边。
- 原图边权重保持为 1，edge_type = 0。
- prompt graph 只作用于 adapted branch。

2. 池选择约束
- 只允许池内原图节点连接 prompt 节点。
- pool_mask 必须包含所有 train_mask 节点。
- 额外池节点由结构不可靠分数选择 top rho 节点，默认 rho = 0.20。
- 结构不可靠分数基于：
  base, m1, m2, base - m1, m1 - m2, neighbor variance
- 不使用 val/test label 选择池。
- 非池内原图节点不直接连接 prompt 节点。
- P2 默认开启 pool_only_prompt_update，使 prompt-to-original 更新只作用于池内原图节点。

3. prompt 节点约束
- prompt 节点追加在原图节点之后。
- 默认 num_prompt_nodes = 8。
- prompt 节点具有可学习 prompt_node_x 和 prompt_key。
- prompt_key 不再只作为自由 learned role，还应受到 train-only 类别路由约束。

4. 基础 top-k 路由约束
- 对每个池内节点 i 构造 query_i：
  query_i = MLP([base_i, m1_i, m2_i, base_i - m1_i, m1_i - m2_i])
- 计算 routing logits：
  logits_i = normalize(query_i) @ normalize(prompt_keys).T / tau
- 默认 tau = 0.5。
- 每个池内节点最多连接 topk_prompt_per_node 个 prompt，默认 topk = 2。
- k = min(topk_prompt_per_node, num_prompt_nodes)。
- 对 top-k logits 做 softmax 得到 assignment probability。
- 不使用伪标签、高置信预测或 val/test label 决定 prompt 连接。

5. 双向边约束
- 若池内节点 i 选择 prompt 节点 p，则同时添加：
  i -> p, edge_type = 1
  p -> i, edge_type = 2
- 两个方向共享同一个 soft edge weight。
- adapted_edge_index = original_edge_index + prompt_edges。
- adapted_edge_type 中：
  original edge = 0
  node-to-prompt edge = 1
  prompt-to-node edge = 2

6. 边权约束
- prompt edge weight 使用软权重：
  w_i,p = edge_scale * edge_scale_multiplier * acceptance_i * assignment_prob_i,p
- edge_scale 为可学习标量，有上限 edge_scale_max。
- edge_scale_multiplier 支持 warmup。
- acceptance_i 来自 rejection gate。
- 若 rejection gate 判断 prompt 对节点 i 可能有害，应降低 acceptance_i。
- 不强制所有池内节点都强接受 prompt。

7. 类别一致性路由约束
- 使用 train_mask 内的有标签池内节点构造类别路由监督。
- 为每个类别 c 指定一个或一组 class prompt slots S_c。
- 若训练池内节点 i 的标签为 y_i，则其 routing probability 应集中到 S_{y_i}：
  L_class_route = -log sum_{p in S_{y_i}} softmax(logits_i)[p]
- 该约束只作用于 train_mask & pool_mask 节点。
- 不对 val/test 节点使用真实标签。
- 目标是：同类别池内节点优先连接到相同或相近的 prompt 节点。

8. 同类聚合与异类分离约束
- 对 train_mask & pool_mask 节点的 routing distribution q_i = softmax(logits_i) 加 supervised contrastive 约束。
- 同类别节点的 q_i 应更接近。
- 不同类别节点的 q_i 应更远。
- 可使用 cosine(q_i, q_j) 或 KL/JS 距离。
- 该约束用于减少 prompt role 混杂，提升 prompt_key 的类别可解释性。

9. prompt_key 类别原型约束
- 使用 train_mask & pool_mask 节点的 query 构造每类 query prototype：
  mu_c = mean(query_i), where y_i = c
- 将对应类别 prompt_key 初始化为 mu_c，或训练时加入：
  L_key_proto = || normalize(prompt_key_c) - normalize(stopgrad(mu_c)) ||^2
- 若一个类别对应多个 prompt slots，则约束这些 slots 靠近该类别 query prototype，同时保留 slot 间 diversity。
- 不使用 val/test label 构造 prototype。

10. prompt role 多样性约束
- 对不同 prompt_key 加 diversity regularization，避免所有 prompt collapse。
- 但类别主 prompt 不应被 balance loss 强行均匀使用到破坏类别路由。
- capacity routing / usage balance 只能作为辅助，不能覆盖类别一致性约束。

11. rejection / null 安全约束
- 即使节点属于某一类别，也不强制它必须接受 prompt。
- 最终策略应是：
  如果节点接受 prompt，则同类节点尽量连到同类 prompt；
  如果 prompt 可能有害，则 rejection gate 降低边权或走 null。
- 这对 Actor / chameleon / squirrel 尤其重要，因为实验显示有用 prompt 信号是稀疏的，大量池内 prompt 消息可能有害。

12. 总损失
- 主损失保持 CE-only paired comparison 为基础。
- 新增约束项只使用 train labels：
  L_total =
    L_CE
    + lambda_class_route * L_class_route
    + lambda_route_supcon * L_route_supcon
    + lambda_key_proto * L_key_proto
    + lambda_role_diversity * L_role_diversity
    + lambda_acceptance * L_acceptance
    + lambda_edge_l1 * L_edge_l1
- 所有 lambda 通过 validation split 选择。
- test labels 不参与任何训练、路由、池选择或超参选择。