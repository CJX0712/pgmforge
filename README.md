# PgmForge

**概率图模型精确与近似推断系统** — 变量消去 / 团树 / 环形置信传播 / 平均场 / Gibbs，带暴力穷举金标准交叉验证。

作者：晨星

[![CI](https://github.com/CJX0712/pgmforge/actions/workflows/ci.yml/badge.svg)](https://github.com/CJX0712/pgmforge/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/python-3.11%20%7C%203.12%20%7C%203.13-blue)
![License](https://img.shields.io/badge/license-MIT-green)
![Determinism](https://img.shields.io/badge/determinism-bit--identical-brightgreen)
![Exactness](https://img.shields.io/badge/exactness-6.9e--15%20vs%20brute--force-orange)

---

## 它解决什么问题

概率图模型（贝叶斯网 / 马尔可夫随机场）的推断——求边际分布与最可能解释（MAP）——是许多 AI 系统的底座。难点在于：**朴素穷举的状态数随变量数指数爆炸**（36 个二值变量就是 2³⁶ ≈ 687 亿个状态）。

PgmForge 在同一套 factor 代数上同时提供精确与近似两类引擎，并由**暴力穷举金标准**逐位交叉验证，确保精确引擎不是"看起来对"，而是数学上对。

## 核心能力

| 引擎 | 类型 | 适用 | 复杂度 |
|---|---|---|---|
| **Variable Elimination（VE）** | 精确 sum/max-product | 通用 | 指数于 treewidth |
| **Junction Tree（团树）** | 精确 Shafer-Shenoy 校准 | 通用 SOTA | 指数于 treewidth，单次校准求所有边际 |
| **Loopy BP（LBP）** | 近似 sum-product | 有环图 | 每轮 O(因子·d²)，树结构精确 |
| **Naive Mean Field（MF）** | 近似坐标上升 | 大规模 | 每轮 O(因子·d²) |
| **Gibbs Sampling（MCMC）** | 随机近似 | 通用 | 每样本 O(邻居·d) |
| **Brute-Force Gold** | 金标准参照 | ≤ 2¹⁸ 状态 | 指数于变量数 |
| **PgmFuse（旗舰编排）** | 自适应路由 | treewidth ≤ 阈值走团树，否则 LBP 回退 | — |

## 快速开始

```bash
pip install -r requirements.txt

python cli.py demo                      # 端到端基准，落盘 benchmark.json
python cli.py marginals chain --n 12    # 单个模型的边际分布
python cli.py marginals grid --rows 4 --cols 4
```

Python API：

```python
from pgmforge.data.synthetic import grid_mrf
from pgmforge.pipeline.pgm_pipeline import PgmFuse

m = grid_mrf(4, 4, 2, seed=3)
print(PgmFuse().route(m))  # junction_tree / loopy_bp
marg = PgmFuse().query_marginals(m)
```

## 实验验证结果

26 个模型 × 3 seeds，全部由暴力穷举（≤2¹⁸ 状态）交叉校验：

| 门禁 | 判据 | 实测 |
|---|---|---|
| **精确性** | VE/JT 边际 vs 金标准 L∞ ≤ 1e-9 | **6.88e-15** |
| **MAP** | VE-MAP 与金标准 argmax 一致 | **24/24** |
| **近似优势** | LBP 相对 MF 平均误差改善 ≥ 30% | **+93.8%** |
| **MCMC** | Gibbs 收敛且同 seed 逐位可复现 | 误差 0.0030，Δ = 0 |
| **缩放** | JT 处理 chain-50 / grid-6×6（2³⁶ 状态） | ≈ 7 ms |
| **确定性** | 二次运行核心指标逐位一致 | Δ = 0 |

引擎平均 L∞（vs 参照）：

```
ve       8.05e-16      jt       7.84e-16      pgmfuse  7.84e-16
lbp      2.58e-03      mf       1.02e-01      gibbs    2.97e-03
```

## 架构

```
core/         Factor 代数（作用域+局部张量）/ 配置 / 确定性种子 / 错误码
data/         合成生成器：链 / 树 / 网格MRF / 多父汇聚 / 随机环图
inference/    ve · junction_tree · lbp · mean_field · gibbs · gold · elim_order
pipeline/     PgmFuse 自适应路由 + 基准评测
```

核心设计：**Factor 只存「作用域 + 局部张量表」**，从不实例化全局 d^n 张量，因此同一套引擎既能处理 10 变量的校验模型，也能处理 50 节点的链与 6×6 网格。详见 [docs/architecture.md](docs/architecture.md)。

## 一处值得一提的工程细节

所有乘法走 `Factor._align_to`：先 `expand_dims` 再 `transpose`，把每个因子的每张量轴**精确搬运**到并集作用域的目标轴位。早期版本用纯 `expand_dims` 广播，在非单调作用域（如 BN 的 `(child, parent)` = `(1,0)`）上会静默地把变量轴互换——2 轴情形巧合正确，3 轴才暴露。测试 `test_multiply_non_monotonic_scope` 与 `test_multiply_three_axis_non_monotonic` 是这条 bug 的回归栅栏。

## 测试

```bash
pytest -q --cov=pgmforge --cov-report=term-missing
ruff check . && ruff format --check .
```

每条引擎与每个不变量都有对应测试；所有精确推断结果均以暴力穷举为参照交叉验证。

## 限制

- 精确推断的复杂度仍 exponential in treewidth；treewidth > 阈值时 `PgmFuse` 自动回退到 LBP。
- 有环图上 Loopy BP 不保证收敛；实现中对 v→f 与 f→v 消息均做归一化并施加阻尼（`--lbp-damping`，默认 0.5）以抑制振荡。
- 金标准在超过 `gold_max_states`（默认 2¹⁸）时不可用，此时基准自动改用团树作为参照。
- 当前仅支持离散有限域变量。

## License

MIT — 见 [LICENSE](LICENSE)。

作者：晨星
