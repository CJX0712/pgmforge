# PgmForge 架构说明

作者：晨星

---

## 1. 设计目标

在**同一套 factor 代数**上提供 PGM 的精确与近似推断，并让每个引擎都能被独立参照物验证。三个约束驱动了全部设计决策：

1. **不能实例化全局张量**——`d^n` 状态空间在任何真实模型上都不可行。所有运算必须保持在局部作用域上。
2. **精确性必须可交叉验证**——小规模用例交给暴力穷举枚举，精确引擎必须与之逐位一致（L∞ ≤ 1e-9，实测 6.9e-15）。
3. **结果必须可复现**——单一 seed 入口，禁用非确定性随机源。

## 2. 分层

```
                ┌─────────────────────────────────────────┐
  应用层         │  cli.py          examples/run_demo.py    │
                └───────────────────┬─────────────────────┘
                                    │
                ┌───────────────────▼─────────────────────┐
  编排层         │  pipeline/pgm_pipeline.py               │
                │  PgmFuse 路由 + run_benchmark 评测        │
                └───────────────────┬─────────────────────┘
                                    │
        ┌───────────────┬───────────┴────────┬───────────────┐
        │               │                    │               │
   ┌────▼────┐   ┌──────▼──────┐   ┌─────────▼──┐   ┌────────▼──────┐
   │ VE      │   │ JunctionTree│   │ LoopyBP    │   │ MeanField     │
   │ 精确     │   │ 精确 SOTA   │   │ 近似 (有环) │   │ 近似 (可扩展)  │
   └────┬────┘   └──────┬──────┘   └─────────┬──┘   └────────┬──────┘
        │               │                    │               │
        └───────────────┴────────┬───────────┴───────────────┘
                                 │
                       ┌─────────▼──────────┐
                       │  Gibbs / BruteForce│  MCMC 近似 / 金标准
                       └─────────┬──────────┘
                                 │
                ┌────────────────▼─────────────────┐
  核心层         │  core/types.py  Factor 代数       │
                │  {scope, cards, table} + 运算     │
                │  core/config  seed  errors        │
                └──────────────────────────────────┘
```

## 3. 核心抽象：Factor

`Factor` 是全部推断的唯一数据结构：

```python
Factor(scope=(1, 0), cards=(2, 2), table=np.ndarray)
```

- `scope`：变量 id 元组，**顺序即 table 的轴序**。这是最关键的不变量。
- `cards`：各轴的维度。
- `table`：局部张量。

运算集合（全部自包含，不依赖全局张量）：

| 运算 | 语义 |
|---|---|
| `multiply` | 广播相乘，作用域取并集 |
| `sum_out` / `max_out` | sum-product / max-product 消去 |
| `argmax_axis` | 记录消去时的 argmax，供 MAP 回溯 |
| `restrict` | 固定证据，丢弃该轴 |
| `marginal` | 消去其余变量后归一化 |

### 轴对齐：`_align_to`

两个作用域不同的因子相乘时，需把各自张量轴搬到并集作用域的目标轴位。实现是 **expand_dims + transpose 两段式**：

```python
missing = [i for i in range(n) if i not in axes]
t = np.expand_dims(table, axis=tuple(missing))  # 补出缺失轴（单点）
sorted_axes = sorted(axes)
cur2tgt[axes[j]] = sorted_axes[j]  # 目标位 ← 当前位
return t.transpose(tuple(cur2tgt))
```

**为什么不能直接 `expand_dims` 广播？** `expand_dims` 的语义是把原张量轴按"剩余位置递增"依次填入非 axis 位——它无法表达非单调映射。对于 BN 的 CPT，`scope = (child, parent)` 常常是降序的（如 `(1,0)`、`(3,1,2)`）；若只做 `expand_dims`，变量轴会静默互换，得到**看起来合理、实则错误**的结果。

该缺陷在 2 轴情形与目标(url)排序恰好一致时会"蒙对"（`(2,1)` 在 `(1,0,2)` 中两种算法结果相同），到 3 轴才暴露——这也是为什么它以 L∞ ≈ 0.7 的形式残存了很久。回归测试 `test_multiply_non_monotonic_scope` 与 `test_multiply_three_axis_non_monotonic` 专门锁定这一点，参照物是"按 factor 真实轴序索引"的独立手算联合。

## 4. 精确推断

### 4.1 变量消去（VE）

```
order, tw = min_fill_order(moral_graph)     # 启发式消元序
for z in elim_vars:
    prod   = ∏ { factors containing z }
    reduce = sum_out(z) 或 max_out(z)       # max-product 时先存 argmax
answer = ∏ remaining factors → normalize
```

消去顺序只影响**效率**、不影响正确性——任何合法顺序都给出相同的精确结果，因为 sum-product 满足结合律与交换律。但它决定中间因子的团大小：糟糕的顺序会让一团膨胀到内存不可行。这里用 min-fill 启发式最小化 fill-in 边数。

### 4.2 团树（Junction Tree）

```
道德化 → min-fill 三角化 → 极大团 → 最大生成树（交界加权）
      → 每因子归入含其作用域的最小团 → 两遍消息传递校准
```

关键在于**消息的排除语义**：`i→j` 的消息必须排除 `j` 已发来的消息。

```python
bu = pot[u]
for k in adj[u]:
    if k == v:  # 排除 v 的贡献
        continue
    bu = bu.multiply(msg[(k, u)])
msg[(u, v)] = project(bu, sep)
```

若<｜hy_place▁holder▁no▁813｜>地把"已含 v 消息的全信念"投影回传给 v，分隔变量的边缘分布会在相邻团之间被重复乘入，得到 `φ·μ` 而非 `φ`——表现为一致性条件被破坏（`∑_{C_i\S} belief_i ≠ ∑_{C_j\S} belief_j`）。该 bug 在多团数、多连通分量时才明显暴露；单团或小树不显现。

实现采用显式消息字典 `msg[(from,to)]`，并在 `distribute` 时构造"剔除对方消息"的中间信念，从结构上杜绝泄漏。多连通分量的道德图会建立多棵树，各自独立校准，端到端仍精确。

### 4.3 消息延后与假收敛（LBP）

环形 BP 的一个隐蔽失效模式：`v→f` 消息若用**上一轮**的 `f→v` 计算，会滞后一步；此时 delta 在第 1 轮出现假 0，`delta < tol` 触发提前 break，消息根本没有传播出几个 hop——连树结构都不精确（曾实测 chain-10 L∞ = 1.26e-01）。

修法是让 `v→f` 使用**刚刚提交**的 `f→v`，使 delta 反映真实变化：

```python
msg_fv = blend(new_fv, msg_fv)      # 先提交 f→v
new_vf = ∏_{fj≠fi} msg_fv[(fj, v)]  # 再用新的 f→v 算 v→f
delta  = max(‖new_vf - msg_vf‖∞)
```

配合两端消息归一化（防数值漂移）与阻尼 `msg = λ·old + (1-λ)·new`（λ 默认 0.5）抑制有环振荡，树结构恢复精确，loopy 图上相对平均场的改善从 −95% 提升到 **+93.8%**。

## 5. 近似推断

| 引擎 | 原理 | 收敛性 | 何时用它 |
|---|---|---|---|
| **LoopyBP** | 因子图 sum-product | 树结构精确；有环可能振荡 | 有环但 treewidth 中等——精度/代价最优 |
| **MeanField** | `q(x)=∏q_v`，坐标上升，log 域更新 | 总收敛到局部最优 | 大规模，作为基线 |
| **Gibbs** | 单站点 MCMC，`exclusive scan` 的条件分布采样 | 渐进收敛 | treewidth 过大时的兜底 |

三者共用同一套 factor 代数，因此任何改进立即作用于全部近似路径。

## 6. 确定性

唯一入口 `core/seed.set_all(seed)` → `np.random.RandomState`。

- 使用 legacy `RandomState` 而非新 Generator：在 NEP-19 之后 `RandomState` 仍提供逐位稳定的兼容性，而 PCG64 的部分流抽样在未来版本可能变化。
- 所有生成器（合成数据、Gibbs）从该单一状态派生，Gibbs 用 `fresh_state(seed+1000)` 避免与数据生成互相干扰。
- 验证方式：demo 连续跑两遍 `run_benchmark`，对**核心指标**（门禁结果 + 各引擎 L∞）做逐位比对——实测最大差 0.0。

## 7. 错误体系

```
PgmError (E000)
├── FactorError   (E100) 张量形状/作用域/非有限值
├── GraphError    (E200) 图结构 / 团指派 / 连通性
├── InferenceError(E300) 引擎不可用 / 状态爆炸 / 变量不在团中
├── DataError     (E400) 数据
└── ConfigError   (E500) 配置解析
```

带稳定错误码，便于在 CI 中针对特定失败制作断言。

## 8. 复杂度

| 运算 | 复杂度 |
|---|---|
| Factor 乘法 | O(∏_{v∈union} d_v) |
| VE 单次查询 | O(n · d^(w+1))，w = treewidth |
| 团树校准（全部边际） | O(∑_团 d^{|团|}) —— 一次校准得所有边际 |
| LBP 每轮 | O(|F| · d²) |
| 暴力穷举 | O(d^n) |

团树的价值在于：**一次校准提供全部变量的边际**，而 VE 需要对每个查询变量重跑。

---

作者：晨星
