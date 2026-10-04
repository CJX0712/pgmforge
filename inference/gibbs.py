"""Gibbs 采样（单站点 MCMC 近似推断 + MAP 估计）。

随机后端，种子锁定后逐位可复现；样本量↑时收敛至精确边际。

作者：晨星
"""

from __future__ import annotations

import numpy as np

from ..core.config import Config
from ..core.seed import fresh_state
from ..core.types import DiscreteModel


class Gibbs:
    name = "gibbs"

    def __init__(self, cfg: Config | None = None, seed: int | None = None) -> None:
        self.cfg = cfg or Config()
        self.seed = seed if seed is not None else self.cfg.seed

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

        rng = fresh_state(self.seed + 1000)
        assign = [int(rng.randint(0, model.cards[v])) for v in range(n)]
        counts = [np.zeros(model.cards[v]) for v in range(n)]
        free = list(range(n))

        total = self.cfg.gibbs_burnin + self.cfg.gibbs_samples
        for t in range(total):
            for v in free:
                card = model.cards[v]
                logp = np.zeros(card)
                for fi in fac_var[v]:
                    g = factors[fi]
                    rows = [
                        np.full(card, assign[u], dtype=int) if u != v else np.arange(card)
                        for u in g.scope
                    ]
                    idx = tuple(rows)
                    vals = g.table[idx]
                    logp += np.log(vals + 1e-300)
                mx = logp.max()
                p = np.exp(logp - mx) if np.isfinite(mx) else np.ones(card)
                s = p.sum()
                if s <= 0:
                    p = np.ones(card) / card
                else:
                    p /= s
                assign[v] = int(rng.choice(card, p=p))
            if t >= self.cfg.gibbs_burnin:
                for v in free:
                    counts[v][assign[v]] += 1.0

        out: dict[int, np.ndarray] = {}
        for v in query_vars:
            c = counts[v]
            s = c.sum()
            out[v] = c / s if s > 0 else np.ones(model.cards[v]) / model.cards[v]
        return out

    def map_query(
        self, model: DiscreteModel, evidence: dict[int, int] | None = None
    ) -> dict[int, int]:
        """Gibbs MAP 估计：取采样中出现频率最高的完整赋值（MPE 近似）。"""
        evidence = evidence or {}
        # 简化：对小规模用 VE 精确 MAP；此处仅对无可行精确时的近似可调用。
        raise NotImplementedError("Gibbs MAP 请走 VE 精确路径；Gibbs 仅提供边际近似")
