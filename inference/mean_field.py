"""Naive Mean Field（坐标上升近似推断）。

q(x)=∏ q_v，作为可扩展近似基线；在 loopy 图上通常弱于 Loopy BP（低估相关性）。

作者：晨星
"""

from __future__ import annotations

import numpy as np

from ..core.config import Config
from ..core.types import DiscreteModel, Factor


class MeanField:
    name = "mean_field"

    def __init__(self, cfg: Config | None = None) -> None:
        self.cfg = cfg or Config()

    @staticmethod
    def available() -> bool:
        return True

    def query_marginals(
        self, model: DiscreteModel, query_vars: list[int] | None = None
    ) -> dict[int, np.ndarray]:
        n = model.n_vars
        factors = model.factors
        if query_vars is None:
            query_vars = list(range(n))

        fac_var: dict[int, list[int]] = {v: [] for v in range(n)}
        for fi, g in enumerate(factors):
            for v in g.scope:
                fac_var[v].append(fi)

        q = {v: np.full(model.cards[v], 1.0 / model.cards[v]) for v in range(n)}

        for _ in range(self.cfg.mf_max_iters):
            max_delta = 0.0
            for v in range(n):
                card = model.cards[v]
                logq = np.zeros(card)
                for fi in fac_var[v]:
                    g = factors[fi]
                    sub = g  # restrict over all other vars using q
                    for u in g.scope:
                        if u == v:
                            continue
                        qu = Factor((u,), q[u], (model.cards[u],))
                        sub = sub.multiply(qu)
                    for xv in range(card):
                        val = float(sub.restrict(v, xv).table.sum())
                        logq[xv] += np.log(val + 1e-300)
                logq -= np.max(logq)
                new_q = np.exp(logq)
                s = new_q.sum()
                if s <= 0:
                    new_q = np.ones(card) / card
                else:
                    new_q /= s
                max_delta = max(max_delta, float(np.max(np.abs(new_q - q[v]))))
                q[v] = new_q
            if max_delta < self.cfg.mf_tol:
                break

        return {v: q[v] for v in query_vars}
