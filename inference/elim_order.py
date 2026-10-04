"""消除顺序与三角化（min-fill / min-degree 启发式）。

作者：晨星
"""

from __future__ import annotations

from ..core.types import DiscreteModel


def build_undirected(model: DiscreteModel) -> dict[int, set[int]]:
    """构造（道德化后的）无向邻接表。

    - MRF：直接用 edges。
    - BN：child-parent 边 + 同一 child 的父节点两两相连（moralize）。
    """
    n = model.n_vars
    adj: dict[int, set[int]] = {i: set() for i in range(n)}
    for a, b in model.edges:
        adj[a].add(b)
        adj[b].add(a)
    if model.kind == "bn":
        for c, p in model.dag_edges:
            adj[c].add(p)
            adj[p].add(c)
        parents_of: dict[int, list[int]] = {}
        for c, p in model.dag_edges:
            parents_of.setdefault(c, []).append(p)
        for _child, ps in parents_of.items():
            for i in range(len(ps)):
                for j in range(i + 1, len(ps)):
                    adj[ps[i]].add(ps[j])
                    adj[ps[j]].add(ps[i])
    return adj


def min_fill_order(adj: dict[int, set[int]], n: int) -> tuple[list[int], int]:
    """返回 (消除顺序, treewidth)。min-fill = 每次选 fill-in 最少的变量。"""
    adj = {v: set(s) for v, s in adj.items()}
    order: list[int] = []
    max_clique = 0
    for _ in range(n):
        best = None
        best_fill = None
        for v in adj:
            fill = 0
            nb = list(adj[v])
            for i in range(len(nb)):
                for j in range(i + 1, len(nb)):
                    if nb[j] not in adj[nb[i]]:
                        fill += 1
            if best_fill is None or fill < best_fill:
                best_fill = fill
                best = v
        order.append(best)
        nb = list(adj[best])
        max_clique = max(max_clique, len(nb) + 1)
        for i in range(len(nb)):
            for j in range(i + 1, len(nb)):
                adj[nb[i]].add(nb[j])
                adj[nb[j]].add(nb[i])
        for w in nb:
            adj[w].discard(best)
        del adj[best]
    return order, max_clique - 1


def build_cliques(adj: dict[int, set[int]], order: list[int]) -> list[frozenset]:
    """从消除顺序提取团（每次消除的 {邻居}∪{z}），再做极大团约简。"""
    adj = {v: set(s) for v, s in adj.items()}
    all_cliques: list[frozenset] = []
    for z in order:
        nb = set(adj[z])
        all_cliques.append(frozenset(nb | {z}))
        nb_list = list(nb)
        for i in range(len(nb_list)):
            for j in range(i + 1, len(nb_list)):
                adj[nb_list[i]].add(nb_list[j])
                adj[nb_list[j]].add(nb_list[i])
        for w in nb_list:
            adj[w].discard(z)
    # 极大团约简（去掉被包含的子集）+ 去重
    uniq = list({c for c in all_cliques})
    maximal = [c for c in uniq if not any(c < o for o in uniq if o is not c)]
    return maximal
