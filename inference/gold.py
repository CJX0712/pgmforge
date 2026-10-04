"""暴力穷举金标准（Brute-Force Enumeration）。

对小规模模型枚举全部赋值得精确联合，作为精确性硬不变量参照。
总状态数超过 gold_max_states 则不可用（benchmark 自动 skip）。

作者：晨星
"""

from __future__ import annotations

import numpy as np

from ..core.config import Config
from ..core.errors import InferenceError
from ..core.types import DiscreteModel


class BruteForceGold:
    name = "brute_force"

    def __init__(self, cfg: Config | None = None) -> None:
        self.cfg = cfg or Config()

    @staticmethod
    def available() -> bool:
        return True

    def _joint_vector(self, model: DiscreteModel) -> np.ndarray:
        total = model.full_state_count()
        if total > self.cfg.gold_max_states:
            raise InferenceError(
                f"暴力穷举状态数 {total} 超 gold_max_states={self.cfg.gold_max_states}", "E300"
            )
        n = model.n_vars
        cards = model.cards
        grids = np.indices(cards).reshape(n, -1)  # (n, total)
        full = np.ones(total, dtype=float)
        for g in model.factors:
            rows = [grids[v] for v in g.scope]
            vals = g.table[tuple(rows)]
            full = full * vals
        return full

    def query_marginals(
        self, model: DiscreteModel, query_vars: list[int] | None = None
    ) -> dict[int, np.ndarray]:
        n = model.n_vars
        if query_vars is None:
            query_vars = list(range(n))
        total = model.full_state_count()
        if total > self.cfg.gold_max_states:
            raise InferenceError(
                f"暴力穷举状态数 {total} 超 gold_max_states={self.cfg.gold_max_states}", "E300"
            )
        cards = model.cards
        joint = self._joint_vector(model)
        z = joint.sum()
        if z <= 0:
            z = 1.0
        joint = joint / z
        joint_full = joint.reshape(cards)
        out: dict[int, np.ndarray] = {}
        for v in query_vars:
            m = joint_full.sum(axis=tuple(u for u in range(n) if u != v))
            out[v] = m
        return out

    def map_query(
        self, model: DiscreteModel, evidence: dict[int, int] | None = None
    ) -> dict[int, int]:
        evidence = evidence or {}
        n = model.n_vars
        cards = model.cards
        grids = np.indices(cards).reshape(n, -1)  # (n, total)
        joint = self._joint_vector(model)
        best_idx = int(np.argmax(joint))
        assign = {v: int(grids[v, best_idx]) for v in range(n)}
        for v, val in evidence.items():
            assign[v] = val
        return assign
