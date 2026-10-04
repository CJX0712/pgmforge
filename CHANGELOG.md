# 更新日志

本项目所有值得记录的变更将写入此文件。
格式参考 [Keep a Changelog](https://keepachangelog.com/zh-CN/1.1.0/)，语义化版本见 [SemVer](https://semver.org/lang/zh-CN/)。

作者：晨星

## [0.1.0] - 2026-10-05

### 新增
- `core/`：`Factor` 代数（作用域 + 局部张量表，从不实例化 d^n 全局张量）、配置、确定性种子、分层错误码体系（E000–E500）。
- `inference/`：
  - 变量消去（VE）：精确 sum/max-product，MAP 回溯
  - 团树（Junction Tree）：min-fill 三角化 + 极大团 + 最大生成树 + Shafer-Shenoy 两遍校准，多连通分量支持
  - Loopy BP：因子图 sum-product，带归一化与阻尼
  - Naive Mean Field：坐标上升基线
  - Gibbs：单站点 MCMC
  - Brute-Force Gold：≤2¹⁸ 状态的穷举金标准
- `pipeline/`：`PgmFuse` 按 treewidth 自适应路由（≤阈值走团树，否则 LBP 回退）+ `run_benchmark` 26 模型横向评测。
- `data/`：确定性合成生成器（链 / 树 / 网格 MRF / 多父汇聚 / 随机环图 / 缩放实例）。
- `cli.py`：`demo` / `bench` / `marginals` 子命令。
- 端到端基准：5 项门禁 + 确定性二次校验，结果落盘 `benchmark.json`。

### 修复
- **`Factor.multiply` 轴错位（正确性）**：原实现仅用 `expand_dims` 广播，无法表达非单调作用域的轴映射，导致 BN 的 CPT（`scope = (child, parent)` 常为降序）在相乘时静默互换变量轴。现改为 `expand_dims + transpose` 两段式 `_align_to`，并由"按 factor 真实轴序索引"的手算联合交叉验证。
- **`JunctionTree` 消息泄漏（正确性）**：`distribute` 阶段把"已含子节点消息的父信念"投影回传给该子节点，使分隔变量边缘被重复乘入（`φ·μ` 而非 `φ`），破坏团树一致性条件。现用显式消息字典，发送时剔除对方消息。
- **`LoopyBP` 假收敛（正确性）**：`v→f` 消息滞后一轮，使 delta 在第 1 轮出现假 0 并提前 break，消息未传播（连树结构都不精确，chain-10 L∞ 曾达 1.26e-01）。现改为用刚提交的 `f→v` 计算 `v→f`，并对两端消息归一化 + 施加阻尼。loopy 图上相对 MF 由 −95.2% 转为 +93.8%。
- `multiplexer` 生成器曾向 `edges` 写入自环 `(i,i)`，已移除。
- `benchmark.json` 曾落盘到仓库外（旧 `ROOT` 指向上级目录），现落在仓库内。
- 移除 `gold._to_full` 未被引用的死代码（其广播逻辑与已修复 bug 同源）。

### 验证
- VE/JT 边际 vs 暴力穷举 L∞ = 6.88e-15（阈值 ≤1e-9）
- VE-MAP 与金标准一致 24/24
- LBP 相对 MF 改善 +93.8%（阈值 ≥30%）
- Gibbs 误差 0.0030，同 seed 逐位可复现
- JT 处理 2³⁶ 状态模型 ≈7ms
- 二次运行核心指标逐位一致（Δ=0）

[0.1.0]: https://github.com/CJX0712/pgmforge/releases/tag/v0.1.0
