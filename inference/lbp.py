"""Loopy Belief Propagation（因子图 sum-product）。

通用高阶因子图消息传递；树结构上精确，loopy 图上为 SOTA 近似。

作者：晨星
"""

from __future__ import annotations

import numpy as np

from ..core.config import Config
from ..core.types import DiscreteModel, Factor


class LoopyBP:
    name = "loopy_bp"

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
        nf = len(factors)
        if query_vars is None:
            query_vars = list(range(n))

        fvars = [list(g.scope) for g in factors]
        var_factors: dict[int, list[int]] = {v: [] for v in range(n)}
        for fi, sv in enumerate(fvars):
            for v in sv:
                var_factors[v].append(fi)

        def ones_var(v: int) -> Factor:
            return Factor((v,), np.ones(model.cards[v]), (model.cards[v],))

        msg_vf: dict[tuple, Factor] = {}
        msg_fv: dict[tuple, Factor] = {}
        for fi in range(nf):
            for v in fvars[fi]:
                msg_vf[(fi, v)] = ones_var(v)
                msg_fv[(fi, v)] = ones_var(v)

        lam = float(self.cfg.lbp_damping)
        if not 0.0 <= lam < 1.0:
            lam = 0.0

        def blend(new_m: Factor, old_m: Factor) -> Factor:
            """阻尼：msg = λ·old + (1-λ)·new，抑制 loopy 振荡。"""
            if lam == 0.0 or old_m.scope != new_m.scope:
                return new_m
            tab = lam * old_m.table + (1.0 - lam) * new_m.table
            return Factor(new_m.scope, tab, new_m.cards)

        for _ in range(self.cfg.lbp_max_iters):
            # ---- factor -> variable（用上一轮 v->f） ----
            new_fv: dict[tuple, Factor] = {}
            for fi in range(nf):
                g = factors[fi]
                sv = fvars[fi]
                for v in sv:
                    acc = g
                    for u in sv:
                        if u != v:
                            acc = acc.multiply(msg_vf[(fi, u)])
                    for u in list(acc.scope):
                        if u != v:
                            acc = acc.sum_out(u)
                    s = float(acc.table.sum())
                    new_fv[(fi, v)] = acc if s <= 0 else acc.normalize()
            # 关键修复：立即提交 f->v，再据此算 v->f。若用旧的 msg_fv 会滞后一轮，
            # 使 delta 在第 1 轮出现假 0 而提前 break，消息根本没传播（树上都不精确）。
            msg_fv = {k: blend(m, msg_fv[k]) for k, m in new_fv.items()}

            # ---- variable -> factor（用刚更新的 f->v），并归一化防数值漂移 ----
            new_vf: dict[tuple, Factor] = {}
            for fi in range(nf):
                for v in fvars[fi]:
                    acc = ones_var(v)
                    for fj in var_factors[v]:
                        if fj != fi:
                            acc = acc.multiply(msg_fv[(fj, v)])
                    s = float(acc.table.sum())
                    new_vf[(fi, v)] = acc if s <= 0 else acc.normalize()

            delta = 0.0
            for key in new_vf:
                old = msg_vf.get(key)
                delta = max(delta, new_vf[key].linf(old) if old is not None else float("inf"))
            msg_vf = {k: blend(m, msg_vf[k]) for k, m in new_vf.items()}
            if delta < self.cfg.lbp_tol:
                break

        out: dict[int, np.ndarray] = {}
        for v in query_vars:
            acc = ones_var(v)
            for fj in var_factors[v]:
                acc = acc.multiply(msg_fv[(fj, v)])
            s = float(acc.table.sum())
            out[v] = acc.table / s if s > 0 else np.ones(model.cards[v]) / model.cards[v]
        return out
