"""变量消去（Variable Elimination）：精确 sum/max-product 推断（基线 + MAP 引擎）。

作者：晨星
"""

from __future__ import annotations

import numpy as np

from ..core.types import DiscreteModel, Factor
from .elim_order import build_undirected, min_fill_order


class VariableElimination:
    name = "ve"

    @staticmethod
    def available() -> bool:
        return True

    @staticmethod
    def _eliminate(model: DiscreteModel, elim_vars: list[int], mode: str) -> list[Factor]:
        factors = [f.copy() for f in model.factors]
        for z in elim_vars:
            relevant = [f for f in factors if z in f.scope]
            if not relevant:
                continue
            prod = relevant[0]
            for g in relevant[1:]:
                prod = prod.multiply(g)
            reduced = prod.sum_out(z) if mode == "sum" else prod.max_out(z)
            factors = [f for f in factors if z not in f.scope] + [reduced]
        return factors

    def query_marginals(
        self, model: DiscreteModel, query_vars: list[int] | None = None
    ) -> dict[int, np.ndarray]:
        n = model.n_vars
        if query_vars is None:
            query_vars = list(range(n))
        # 消除顺序：min-fill，每项只消去「除查询变量外」的所有变量
        order, _ = min_fill_order(build_undirected(model), n)
        out: dict[int, np.ndarray] = {}
        for v in query_vars:
            elim_vars = [u for u in order if u != v]
            fs = self._eliminate(model, elim_vars, "sum")
            acc = fs[0]
            for f in fs[1:]:
                acc = acc.multiply(f)
            out[v] = acc.marginal(v)
        return out

    def map_query(
        self, model: DiscreteModel, evidence: dict[int, int] | None = None
    ) -> dict[int, int]:
        evidence = evidence or {}
        # 证据限制
        factors: list[Factor] = []
        for f in model.factors:
            g = f
            for v, val in evidence.items():
                if v in g.scope:
                    g = g.restrict(v, val)
            factors.append(g)
        free = [v for v in range(model.n_vars) if v not in evidence]
        order, _ = min_fill_order(build_undirected(model), model.n_vars)
        order = [v for v in order if v in free]

        arg_store: dict[int, tuple[tuple, Factor]] = {}
        for z in order:
            relevant = [f for f in factors if z in f.scope]
            if not relevant:
                continue
            prod = relevant[0]
            for g in relevant[1:]:
                prod = prod.multiply(g)
            arg_store[z] = (prod.argmax_axis(z).scope, prod.argmax_axis(z))
            reduced = prod.max_out(z)
            factors = [f for f in factors if z not in f.scope] + [reduced]

        assignment: dict[int, int] = {}
        for z in reversed(order):
            if z in evidence:
                assignment[z] = evidence[z]
                continue
            if z not in arg_store:
                fac = next((f for f in factors if f.scope == (z,)), None)
                assignment[z] = int(np.argmax(fac.table)) if fac is not None else 0
                continue
            nscope, arg_f = arg_store[z]
            if not nscope:
                assignment[z] = int(arg_f.table)
            else:
                idx = tuple(assignment[v] for v in nscope)
                assignment[z] = int(arg_f.table[idx])
        for v, val in evidence.items():
            assignment[v] = val
        return assignment
