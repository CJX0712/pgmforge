"""合成 PGM 生成器（固定 seed，可复现）。

覆盖：链 / 树 / 网格 MRF / 多父汇聚(loopy) / 随机有向图 / 大规模缩放实例。
数值约定：BN 的 CPT 在每种父配置下对子归一；MRF 势函数为正。

作者：晨星
"""

from __future__ import annotations

import itertools

import numpy as np

from ..core.types import DiscreteModel, Factor


def _rng(seed: int) -> np.random.RandomState:
    return np.random.RandomState(int(seed))


def _cpt(
    child: int,
    parents: tuple[int, ...],
    cards: list[int],
    rng: np.random.RandomState,
    alpha: float = 1.0,
) -> Factor:
    scope = (child, *tuple(parents))
    shape = (cards[child], *tuple(cards[p] for p in parents))
    table = np.ones(shape)
    parent_shape = shape[1:]
    for pc in itertools.product(*[range(s) for s in parent_shape]):
        dist = rng.dirichlet(np.ones(cards[child]) * alpha)
        table[slice(None), *pc] = dist
    return Factor(scope, table, shape)


def chain(n: int, cards: list[int] | int = 2, seed: int = 1) -> DiscreteModel:
    if isinstance(cards, int):
        cards = [cards] * n
    rng = _rng(seed)
    factors: list[Factor] = []
    edges: list[tuple[int, int]] = []
    dag: list[tuple[int, int]] = []
    factors.append(_cpt(0, (), cards, rng))
    for i in range(1, n):
        factors.append(_cpt(i, (i - 1,), cards, rng))
        edges.append((i, i - 1))
        dag.append((i, i - 1))
    return DiscreteModel(
        n,
        list(cards),
        factors,
        kind="bn",
        edges=edges,
        dag_edges=dag,
        source=f"chain(n={n},seed={seed})",
    )


def tree(n: int, cards: list[int] | int = 2, seed: int = 2) -> DiscreteModel:
    if isinstance(cards, int):
        cards = [cards] * n
    rng = _rng(seed)
    # 随机生成树（确保连通）
    parent = [-1] * n
    order = list(range(1, n))
    rng.shuffle(order)
    for i in order:
        p = rng.randint(0, i)
        parent[i] = p
    factors = [_cpt(0, (), cards, rng)]
    edges = []
    dag = []
    for i in range(1, n):
        factors.append(_cpt(i, (parent[i],), cards, rng))
        edges.append((i, parent[i]))
        dag.append((i, parent[i]))
    return DiscreteModel(
        n,
        list(cards),
        factors,
        kind="bn",
        edges=edges,
        dag_edges=dag,
        source=f"tree(n={n},seed={seed})",
    )


def grid_mrf(rows: int, cols: int, cards: list[int] | int = 2, seed: int = 3) -> DiscreteModel:
    n = rows * cols
    if isinstance(cards, int):
        cards = [cards] * n
    rng = _rng(seed)
    factors: list[Factor] = []
    edges: list[tuple[int, int]] = []

    def idx(r: int, c: int) -> int:
        return r * cols + c

    for v in range(n):
        # 单点势
        table = rng.random(cards[v]) + 0.2
        factors.append(Factor((v,), table, (cards[v],)))
    for r in range(rows):
        for c in range(cols):
            if c + 1 < cols:
                a, b = idx(r, c), idx(r, c + 1)
                edges.append((a, b))
                pot = rng.random((cards[a], cards[b])) + 0.2
                factors.append(Factor((a, b), pot, (cards[a], cards[b])))
            if r + 1 < rows:
                a, b = idx(r, c), idx(r + 1, c)
                edges.append((a, b))
                pot = rng.random((cards[a], cards[b])) + 0.2
                factors.append(Factor((a, b), pot, (cards[a], cards[b])))
    return DiscreteModel(
        n,
        list(cards),
        factors,
        kind="mrf",
        edges=edges,
        source=f"grid_mrf({rows}x{cols},seed={seed})",
    )


def multiplexer(
    n_causes: int, effect_card: int = 2, cause_card: int = 2, seed: int = 4
) -> DiscreteModel:
    """n_causes 个因 → 1 个果；道德化后父节点相互相连 → loopy（treewidth=n_causes+1）。"""
    n = n_causes + 1
    cards = [cause_card] * n_causes + [effect_card]
    rng = _rng(seed)
    factors: list[Factor] = []
    edges: list[tuple[int, int]] = []
    dag: list[tuple[int, int]] = []
    for i in range(n_causes):
        factors.append(_cpt(i, (), cards, rng))
    effect = n_causes
    parents = tuple(range(n_causes))
    factors.append(_cpt(effect, parents, cards, rng, alpha=0.6))
    for i in range(n_causes):
        edges.append((effect, i))
        dag.append((effect, i))
    # dag_edges 需含父关系；补全
    dag_full = [(effect, p) for p in parents]
    return DiscreteModel(
        n,
        list(cards),
        factors,
        kind="bn",
        edges=edges,
        dag_edges=dag_full,
        source=f"multiplexer(causes={n_causes},seed={seed})",
    )


def random_loopy(
    n: int, p_edge: float = 0.3, cards: list[int] | int = 2, seed: int = 5
) -> DiscreteModel:
    if isinstance(cards, int):
        cards = [cards] * n
    rng = _rng(seed)
    # 随机 DAG（拓扑序防环）
    edges: list[tuple[int, int]] = []
    dag: list[tuple[int, int]] = []
    adj: dict[int, set[int]] = {i: set() for i in range(n)}
    for i in range(n):
        ps = [j for j in range(i) if (rng.random() < p_edge) and (j not in adj[i])]
        # 限制父数避免爆树宽
        if len(ps) > 3:
            ps = ps[:3]
        for p in ps:
            adj[i].add(p)
            edges.append((i, p))
            dag.append((i, p))
    factors = [_cpt(0, (), cards, rng)]
    for i in range(1, n):
        ps = list(adj[i])
        if ps:
            factors.append(_cpt(i, tuple(ps), cards, rng, alpha=0.7))
        else:
            factors.append(_cpt(i, (), cards, rng))
    return DiscreteModel(
        n,
        list(cards),
        factors,
        kind="bn",
        edges=edges,
        dag_edges=dag,
        source=f"random_loopy(n={n},p={p_edge},seed={seed})",
    )


def big_chain(n: int, cards: list[int] | int = 2, seed: int = 7) -> DiscreteModel:
    return chain(n, cards, seed)


def big_grid(rows: int, cols: int, cards: list[int] | int = 2, seed: int = 8) -> DiscreteModel:
    return grid_mrf(rows, cols, cards, seed)
