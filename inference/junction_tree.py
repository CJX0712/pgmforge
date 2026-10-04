"""团树精确推断（Junction Tree / Clique Tree, Shafer-Shenoy 校准）。

PGM 精确推断 SOTA：三角化 → 极大团 → 最大生成树（交界加权）→ 两遍消息传递。
支持多连通分量（对每个分量独立建树校准，端到端精确）。

作者：晨星
"""

from __future__ import annotations

import itertools
import sys

import networkx as nx
import numpy as np

from ..core.errors import GraphError, InferenceError
from ..core.types import DiscreteModel, Factor
from .elim_order import build_cliques, build_undirected, min_fill_order


def build_junction_tree(model: DiscreteModel):
    """返回 (trees, treewidth)。trees: List[(cliques_subset, tree_edges)]，每分量一棵。"""
    adj = build_undirected(model)
    order, tw = min_fill_order(adj, model.n_vars)
    cliques = build_cliques(adj, order)
    if not cliques:
        cliques = [frozenset(range(model.n_vars))]
    n = len(cliques)

    G = nx.Graph()
    G.add_nodes_from(range(n))
    for i, j in itertools.combinations(range(n), 2):
        inter = cliques[i] & cliques[j]
        if inter:
            G.add_edge(i, j, sep=inter, w=len(inter))

    trees: list[tuple[list[frozenset], list[tuple[int, int, frozenset]]]] = []
    for comp in nx.connected_components(G):
        comp = list(comp)
        local = {cg: li for li, cg in enumerate(comp)}
        sub_edges = sorted(
            ((d["w"], i, j) for (i, j, d) in G.edges(comp, data=True)),
            reverse=True,
        )
        parent = {v: v for v in comp}

        def find(x: int, par: dict[int, int]) -> int:
            """并查集查找（带路径压缩）。显式传参，避免闭包捕获循环变量。"""
            while par[x] != x:
                par[x] = par[par[x]]
                x = par[x]
            return x

        te: list[tuple[int, int, frozenset]] = []
        for _w, i, j in sub_edges:
            if find(i, parent) != find(j, parent):
                parent[find(i, parent)] = find(j, parent)
                te.append((local[i], local[j], frozenset(cliques[i] & cliques[j])))
        trees.append(([cliques[v] for v in comp], te))
    return trees, tw


class JunctionTree:
    name = "junction_tree"

    def __init__(self, model: DiscreteModel) -> None:
        self.model = model
        self.trees, self.treewidth = build_junction_tree(model)
        # 全局因子→团指派（每个因子归入含其 scope 的最小团），再按分量分组
        all_cliques = [
            (ci, k, c) for ci, (cs, _te) in enumerate(self.trees) for k, c in enumerate(cs)
        ]
        factor_to_clique: dict[tuple[int, int], list[Factor]] = {}
        for g in model.factors:
            gset = set(g.scope)
            best = None
            best_size = None
            for ci, k, c in all_cliques:
                if gset <= c and (best_size is None or len(c) < best_size):
                    best_size = len(c)
                    best = (ci, k)
            if best is None:
                raise GraphError(f"因子 {g.scope} 不被任何团包含", "E200")
            factor_to_clique.setdefault(best, []).append(g)
        self._belief_trees = []
        for ci, (cs, te) in enumerate(self.trees):
            assigned = [factor_to_clique.get((ci, k), []) for k in range(len(cs))]
            self._belief_trees.append(self._calibrate(cs, te, assigned))

    @staticmethod
    def available() -> bool:
        return True

    def _init_potentials(
        self, cliques_subset: list[frozenset], assigned: list[list[Factor]]
    ) -> list[Factor]:
        pots: list[Factor] = []
        for i, c in enumerate(cliques_subset):
            scope = tuple(sorted(c))
            cards = tuple(self.model.cards[v] for v in scope)
            f = Factor(scope, np.ones(cards), cards)
            for g in assigned[i]:
                f = f.multiply(g)
            pots.append(f)
        return pots

    def _calibrate(
        self,
        cliques_subset: list[frozenset],
        tree_edges: list[tuple[int, int, frozenset]],
        assigned: list[list[Factor]],
    ) -> list[Factor]:
        n = len(cliques_subset)
        pot = self._init_potentials(cliques_subset, assigned)

        if n == 1:
            return [p.copy() for p in pot]

        adj = {i: [] for i in range(n)}
        sep: dict[tuple[int, int], list[int]] = {}
        for i, j, s in tree_edges:
            adj[i].append(j)
            adj[j].append(i)
            sv = sorted(s)
            sep[(i, j)] = sv
            sep[(j, i)] = sv

        sys.setrecursionlimit(10000)

        # 显式消息字典 (from,to) -> 分隔变量上的 Factor。
        # 关键：i->j 的消息必须排除 j 已发来的消息，否则分隔变量会被重复乘入
        # （标准 Shafer-Shenoy 的 collect/distribute 泄漏 bug）。
        msg: dict[tuple[int, int], Factor] = {}

        def project(fac: Factor, keep: list[int]) -> Factor:
            out = fac
            for w in list(out.scope):
                if w not in keep:
                    out = out.sum_out(w)
            return out

        def belief_of(u: int) -> Factor:
            b = pot[u].copy()
            for k in adj[u]:
                if (k, u) in msg:
                    b = b.multiply(msg[(k, u)])
            return b

        def collect(u: int, par: int) -> None:
            for v in adj[u]:
                if v == par:
                    continue
                collect(v, u)
                # belief_of(v) 此时只含 v 的子节点消息（不含父 u），投影即得正确分隔消息
                msg[(v, u)] = project(belief_of(v), sep[(v, u)])

        def distribute(u: int, par: int) -> None:
            for v in adj[u]:
                if v == par:
                    continue
                # 构造 belief[u] 但剔除 v 已发来的消息，避免回灌泄漏
                bu = pot[u].copy()
                for k in adj[u]:
                    if k == v:
                        continue
                    if (k, u) in msg:
                        bu = bu.multiply(msg[(k, u)])
                msg[(u, v)] = project(bu, sep[(u, v)])
                distribute(v, u)

        collect(0, -1)
        distribute(0, -1)
        return [belief_of(u) for u in range(n)]

    def query_marginals(
        self, model: DiscreteModel | None = None, query_vars: list[int] | None = None
    ) -> dict[int, np.ndarray]:
        if query_vars is None:
            query_vars = list(range(self.model.n_vars))
        out: dict[int, np.ndarray] = {}
        for v in query_vars:
            found = False
            for ci, (cs, _te) in enumerate(self.trees):
                for k, c in enumerate(cs):
                    if v in c:
                        out[v] = self._belief_trees[ci][k].marginal(v)
                        found = True
                        break
                if found:
                    break
            if not found:
                raise InferenceError(f"变量 {v} 不在任何团中", "E300")
        return out
