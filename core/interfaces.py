"""推断引擎接口契约（统一评测语义：分数越大越异常/误差越小越好）。

所有引擎（精确/近似/金标准）实现同一 Protocol，pipeline 才能公平横向对比。

作者：晨星
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np

from .types import DiscreteModel


@runtime_checkable
class InferenceEngine(Protocol):
    name: str

    def available(self) -> bool:
        """后端是否可用（离线兜底应恒 True）。"""
        ...

    def query_marginals(
        self, model: DiscreteModel, query_vars: list[int] | None = None
    ) -> dict[int, np.ndarray]:
        """返回 {var: 归一化边际分布(ndarray)}；query_vars=None 表示全部变量。"""
        ...

    def map_query(
        self, model: DiscreteModel, evidence: dict[int, int] | None = None
    ) -> dict[int, int]:
        """返回 MAP/MPE 赋值 {var: value}。"""
        ...
