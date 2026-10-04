# PgmForge Model Card

作者：晨星 · 版本 0.1.0

---

## 系统标识

| 项 | 值 |
|---|---|
| 名称 | PgmForge |
| 类型 | 概率图模型（BN / MRF）推断引擎套件 |
| 版本 | 0.1.0 |
| 作者 | 晨星 |
| 许可证 | MIT |
| 输入 | 离散有限域 PGM：变量基数 + 因子（CPT 或势函数）+ 图结构 |
| 输出 | 变量边际分布、MAP 赋值、路由决策 |

## 预期用途

**适用**
- 离散贝叶斯网 / 马尔可夫随机场的边际推断与 MAP 查询
- 需要**可交叉验证正确性**的场景（有暴力穷举参照、有明确误差上界）
- 中小 treewidth 的精确推断；大 scale 图的近似推断
- 算法教学 / benchmark：同一模型上横向对比 6 类引擎

**不适用**
- 连续变量、混合型图（当前仅离散有限域）
- treewidth 极大且不能接受近似误差的场景（精确推断本质受 treewidth 限制）
- 在线/流式大规模服务（未做分布式或增量更新）

## 组件清单

| 引擎 | 类型 | 是否保证精确 |
|---|---|---|
| Variable Elimination | 精确 sum/max-product | 是 |
| Junction Tree | 精确 Shafer-Shenoy 校准 | 是 |
| Loopy Belief Propagation | 近似 | 树结构精确；有环近似 |
| Naive Mean Field | 近似 | 否 |
| Gibbs Sampling | 随机近似 | 否（渐进） |
| Brute-Force Gold | 金标准参照 | 是（≤2¹⁸ 状态） |
| PgmFuse | 路由编排 | treewidth ≤ 阈值时精确 |

## 训练数据

**本系统不进行学习/训练。** 所有评测模型来自 `data/synthetic.py` 的确定性合成生成器，seed 固定：

| 生成器 | 结构特征 | treewidth |
|---|---|---|
| `chain(n)` | 线性链 X0→X1→…→Xn | 1 |
| `tree(n)` | 随机生成树 | 1 |
| `grid_mrf(r,c)` | r×c 网格 MRF | min(r,c) |
| `multiplexer(k)` | k 个因 → 1 个果（道德化后 k+1 团） | k+1 |
| `random_loopy(n,p)` | 随机 DAG（每节点 ≤3 父） | 随 seed 变化 |
| `big_chain` / `big_grid` | 缩放测试（2³⁶ 状态） | 1 / 6 |

这是有意为之：推断正确性是**独立于训练数据**的性质，用受控合成图才能把 treewidth、环密度等维度分离开来验证。

## 量化性能

基准：26 模型 × seeds {1,2,3}，参照 = 暴力穷举（状态数 ≤ 2¹⁸）。

| 指标 | 结果 |
|---|---|
| VE/JT 边际 L∞ vs 金标准 | 6.88e-15（阈值 1e-9） |
| VE-MAP 与金标准一致率 | 24/24 |
| LBP 平均 L∞ | 2.58e-03 |
| MF 平均 L∞ | 1.02e-01 |
| Gibbs 平均 L∞ | 2.97e-03 |
| LBP 相对 MF 改善 | +93.8% |
| JT 处理 2³⁶ 状态模型 | ≈ 7 ms |
| 二次运行核心指标差 | 0.0（逐位一致） |

## 公平性 / 偏差考量

本系统执行概率推理，不涉及人的类别属性、不做预测性判定，因此不存在直接的公平性危害。需要留意的两点：

1. **近似误差不是均匀分布的**——Loopy BP 在紧密耦合的有环图上可能只在部分变量上失效。生产使用时应先和精确引擎在小规模子集上比对，或监控迭代 delta。
2. **有环图上 Loopy BP 可能不收敛**——实现中通过归一化与阻尼缓解，但不保证。若到达 `lbp_max_iters` 仍未低于 `lbp_tol`，应视为"未收敛"而非"已收敛"。

## 已知限制

- 超出 `treewidth` 阈值时，`PgmFuse` 静默回退到 LBP（可通过 `route()` 显式查询实际路径）。
- `gold_max_states` 之外的模型无法用金标准校验，此时基准改用团树作为参照，读者应注意参照源已切换（`ref` 字段标记 `brute_force` / `junction_tree`）。
- Mean Field 作为基线实现，坐标上升可能停在局部最优。
- 未实现：连续变量、动态贝叶斯网、在线增量推断、分布式推断。

## 复现方法

```bash
pip install -r requirements.lock.txt
python cli.py demo                     # 复现全部基准数字，落盘 benchmark.json
pytest -q                              # 复现全部硬不变量断言
```

完整结果见仓库内 `benchmark.json`（`gates` 字段为门禁结论，`models` 字段为逐模型明细）。

---

作者：晨星
