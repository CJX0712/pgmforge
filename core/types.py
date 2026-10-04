"""PgmForge 核心数据类型。

变量以整数 id 表示（0..n-1）。Factor 以「作用域 + 局部张量表」表达，所有运算
（乘/求和/取最大/限制/边际）均保持自包含，不依赖全局张量，从而精确推断可处理
大规模图（只实例化团，而非 d^n 全局张量）。

作者：晨星
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np

from .errors import FactorError


@dataclass
class Variable:
    vid: int
    name: str
    card: int  # 定义域大小（笛卡尔积维度）
    values: list[str] | None = None  # 可选取值标签


class Factor:
    """离散因子：scope（变量 id 元组，顺序即 table 轴序）+ cards + table。"""

    __slots__ = ("cards", "scope", "table")

    def __init__(
        self,
        scope: Sequence[int],
        table: np.ndarray,
        cards: Sequence[int] | None = None,
    ) -> None:
        self.scope = tuple(int(v) for v in scope)
        if cards is None:
            raise FactorError("Factor 必须显式提供 cards", "E100")
        self.cards = tuple(int(c) for c in cards)
        if len(self.cards) != len(self.scope):
            raise FactorError("cards 长度与作用域不一致", "E100")
        self.table = np.asarray(table, dtype=float)
        if self.table.shape != self.cards:
            # 允许传入展平数组
            if self.table.size == int(np.prod(self.cards)):
                self.table = self.table.reshape(self.cards)
            else:
                raise FactorError(
                    f"table.shape={self.table.shape} 与 cards={self.cards} 不符", "E100"
                )
        if np.any(~np.isfinite(self.table)):
            raise FactorError("Factor 含非有限值", "E100")

    # ---- 基础 ---------------------------------------------------------------
    def __repr__(self) -> str:
        return f"Factor(scope={self.scope}, cards={self.cards})"

    def copy(self) -> Factor:
        return Factor(self.scope, self.table.copy(), self.cards)

    def _union_cards(self, other: Factor) -> tuple:
        cards = []
        for v in self.scope:
            cards.append(self.cards[self.scope.index(v)])
        other_only = [v for v in other.scope if v not in self.scope]
        for v in other_only:
            cards.append(other.cards[other.scope.index(v)])
        return tuple(cards), other_only

    @staticmethod
    def _align_to(table: np.ndarray, axes: list[int], n: int) -> np.ndarray:
        """把一个张量广播到 n 维 union：table 的第 j 轴放到结果第 axes[j] 轴。

        关键修复：expand_dims 仅按「剩余位置递增」填充原轴，无法表达非单调映射
        （如 self.scope=(1,0) 在 union=(1,0,2) 中 self_axes=[0,1] 没问题，但
        other.scope=(2,1) 的 other_axes=[2,0] 会被错误填成 (var2→pos0, var1→pos2)）。
        这里先 expand 再 transpose，把每个原轴精确搬到 axes 指定位置。
        """
        missing = [i for i in range(n) if i not in axes]
        t = np.expand_dims(table, axis=tuple(missing))
        sorted_axes = sorted(axes)
        # 当前位置 -> 目标位置：原轴 j 当前在 sorted_axes[j]，应去 axes[j]。
        # 故 cur2tgt[axes[j]] = sorted_axes[j]（单点轴原地）。注意：索引对不能写反。
        cur2tgt = [0] * n
        for j, sa in enumerate(sorted_axes):
            cur2tgt[axes[j]] = sa
        for m in missing:
            cur2tgt[m] = m
        return t.transpose(tuple(cur2tgt))

    def multiply(self, other: Factor) -> Factor:
        """广播相乘，结果作用域 = self.scope ∪ other.scope。

        正确处理非单调作用域：每个因子表经 _align_to 精确搬到 union 的对应轴位，
        再逐元素相乘。
        """
        union_cards, other_only = self._union_cards(other)
        union = list(self.scope) + other_only
        n = len(union)
        self_axes = [union.index(v) for v in self.scope]
        other_axes = [union.index(v) for v in other.scope]
        a = self._align_to(self.table, self_axes, n)
        b = self._align_to(other.table, other_axes, n)
        return Factor(union, a * b, union_cards)

    def _drop(self, var: int) -> tuple:
        ax = self.scope.index(var)
        new_scope = self.scope[:ax] + self.scope[ax + 1 :]
        new_cards = self.cards[:ax] + self.cards[ax + 1 :]
        return ax, new_scope, new_cards

    def sum_out(self, var: int) -> Factor:
        """沿 var 轴求和（sum-product 消去）。"""
        if var not in self.scope:
            return self.copy()
        ax, ns, nc = self._drop(var)
        return Factor(ns, self.table.sum(axis=ax), nc)

    def max_out(self, var: int) -> Factor:
        """沿 var 轴取最大（max-product 消去，丢弃 argmax）。"""
        if var not in self.scope:
            return self.copy()
        ax, ns, nc = self._drop(var)
        return Factor(ns, self.table.max(axis=ax), nc)

    def argmax_axis(self, var: int) -> Factor:
        """沿 var 轴的 argmax 索引（形状 = 其余轴），用于 MAP 回溯。"""
        ax = self.scope.index(var)
        ns = self.scope[:ax] + self.scope[ax + 1 :]
        nc = self.cards[:ax] + self.cards[ax + 1 :]
        return Factor(ns, self.table.argmax(axis=ax).astype(float), nc)

    def restrict(self, var: int, val: int) -> Factor:
        """将 var 固定为 val（证据），丢弃该轴。"""
        if var not in self.scope:
            return self.copy()
        ax, ns, nc = self._drop(var)
        return Factor(ns, np.take(self.table, int(val), axis=ax), nc)

    def marginal(self, var: int) -> np.ndarray:
        """对 var 求归一化边际分布（1D 数组）。"""
        f = self
        for v in list(self.scope):
            if v != var:
                f = f.sum_out(v)
        s = float(f.table.sum())
        if s <= 0:
            return (
                np.ones(self.cards[self.scope.index(var)], dtype=float)
                / self.cards[self.scope.index(var)]
            )
        return f.table / s

    def normalize(self) -> Factor:
        s = float(self.table.sum())
        if s <= 0 or not np.isfinite(s):
            return self.copy()
        return Factor(self.scope, self.table / s, self.cards)

    def log_table(self) -> np.ndarray:
        """返回 log(table + eps)，避免 log(0)。"""
        return np.log(self.table + 1e-300)

    def linf(self, other: Factor) -> float:
        """与另一 factor（同作用域）的 L∞ 差。"""
        if self.scope != other.scope:
            raise FactorError("作用域不同无法比较 L∞", "E100")
        return float(np.max(np.abs(self.table - other.table)))


@dataclass
class DiscreteModel:
    """离散 PGM：BN（有向）或 MRF（无向）。推断只需 factors + cards。"""

    n_vars: int
    cards: list[int]
    factors: list[Factor] = field(default_factory=list)
    var_names: list[str] = field(default_factory=list)
    kind: str = "bn"  # "bn" | "mrf"
    edges: list[tuple[int, int]] = field(default_factory=list)  # 无向边（moral/原图）
    # 有向边 (child, parent)，由 parent 指向 child
    dag_edges: list[tuple[int, int]] = field(default_factory=list)
    source: str = "synthetic"

    def __post_init__(self) -> None:
        if not self.var_names:
            self.var_names = [f"X{i}" for i in range(self.n_vars)]
        if len(self.cards) != self.n_vars:
            raise FactorError("cards 长度与 n_vars 不符", "E100")

    def var(self, i: int) -> Variable:
        return Variable(i, self.var_names[i], self.cards[i])

    def full_state_count(self) -> int:
        return int(np.prod(self.cards))

    def joint(self) -> Factor:
        """所有因子乘积（未归一化联合 ∝ P）。"""
        if not self.factors:
            raise FactorError("模型无因子", "E100")
        f = self.factors[0]
        for g in self.factors[1:]:
            f = f.multiply(g)
        return f

    def true_card(self, var: int) -> int:
        return self.cards[var]
