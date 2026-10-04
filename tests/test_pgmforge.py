"""PgmForge 测试套件。

覆盖每条硬不变量：factor 代数轴对齐（含非单调作用域）、VE/JT 精确性、
MAP 命中、LBP 树精确性与 loopy 优势、Gibbs 可复现、CLI 冒烟。

作者：晨星
"""

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pgmforge.core.config import Config
from pgmforge.core.seed import set_all
from pgmforge.core.types import DiscreteModel, Factor
from pgmforge.data import synthetic as S
from pgmforge.inference.elim_order import (
    build_cliques,
    build_undirected,
    min_fill_order,
)
from pgmforge.inference.gibbs import Gibbs
from pgmforge.inference.gold import BruteForceGold
from pgmforge.inference.junction_tree import JunctionTree
from pgmforge.inference.lbp import LoopyBP
from pgmforge.inference.mean_field import MeanField
from pgmforge.inference.ve import VariableElimination
from pgmforge.pipeline.pgm_pipeline import PgmFuse, run_benchmark


# --------------------------------------------------------------------------
# Factor 代数：轴对齐（非单调作用域是关键回归点）
# --------------------------------------------------------------------------
def _joint_by_hand(scope_pairs, cards):
    """独立手算联合：按 factor 真实轴序索引，作为 factor 乘法的参照。"""
    n = len(cards)
    grids = np.indices(cards).reshape(n, -1)
    full = np.ones(grids.shape[1], dtype=float)
    for scope, table in scope_pairs:
        rows = [grids[v] for v in scope]
        full = full * table[tuple(rows)]
    return full.reshape(cards)


def test_multiply_monotonic_scope():
    a = Factor((0, 1), [[0.1, 0.2], [0.3, 0.4]], (2, 2))
    b = Factor((1,), [0.5, 2.0], (2,))
    prod = a.multiply(b)
    expect = np.array([[0.1 * 0.5, 0.2 * 2.0], [0.3 * 0.5, 0.4 * 2.0]])
    assert prod.scope == (0, 1)
    np.testing.assert_allclose(prod.table, expect)


def test_multiply_non_monotonic_scope():
    """回归：作用域非单调时曾因 expand_dims 填轴错位导致变量互换。"""
    a = Factor((1, 0), [[0.1, 0.2], [0.3, 0.4]], (2, 2))  # 轴0=var1, 轴1=var0
    b = Factor((2, 1), [[0.5, 0.6], [0.7, 0.8]], (2, 2))  # 轴0=var2, 轴1=var1
    prod = a.multiply(b)
    cards = [2, 2, 2]
    expect = _joint_by_hand([((1, 0), a.table), ((2, 1), b.table)], cards)
    order = list(prod.scope)
    got = np.transpose(prod.table, [order.index(v) for v in range(3)])
    np.testing.assert_allclose(got, expect)


def test_multiply_three_axis_non_monotonic():
    """回归：3 轴非单调作用域曾暴露 cur2tgt 索引写反的 bug。

    参照：按 factor 真实轴序逐 assignment 索引手算的联合，逐元素比对。
    """
    import itertools

    rng = np.random.RandomState(0)
    t1 = rng.rand(2, 2, 2)
    a = Factor((3, 1, 2), t1, (2, 2, 2))
    t2 = rng.rand(2, 2, 2)
    b = Factor((5, 2, 4), t2, (2, 2, 2))

    prod = a.multiply(b)
    joint = _joint_by_hand([((3, 1, 2), t1), ((5, 2, 4), t2)], [2] * 6)

    order = list(prod.scope)  # 期望 (3,1,2,5,4)
    assert order == [3, 1, 2, 5, 4]
    for assign in itertools.product(range(2), repeat=6):
        idx = tuple(assign[v] for v in order)
        assert abs(prod.table[idx] - joint[assign]) < 1e-12


def test_factor_ops():
    f = Factor((0, 1), [[1.0, 2.0], [3.0, 4.0]], (2, 2))
    # sum_out(var1)：沿轴1求和 -> [1+2, 3+4]
    np.testing.assert_allclose(f.sum_out(1).table, [3.0, 7.0])
    # max_out(var1)：沿轴1取最大 -> [2, 4]
    np.testing.assert_allclose(f.max_out(1).table, [2.0, 4.0])
    # restrict(var0=0)：固定 var0 -> [1, 2]
    np.testing.assert_allclose(f.restrict(0, 0).table, [1.0, 2.0])
    # marginal(var1)：消去 var0 -> [1+3, 2+4]=[4,6] 归一化 -> [0.4,0.6]
    np.testing.assert_allclose(f.marginal(1), [0.4, 0.6])
    assert abs(f.marginal(1).sum() - 1.0) < 1e-12
    assert f.multiply(Factor((1,), [1.0, 1.0], (2,))).scope == (0, 1)
    # 不在作用域内的变量：sum_out/restrict 应原样返回副本
    assert f.sum_out(7).scope == (0, 1)
    assert f.restrict(7, 0).scope == (0, 1)
    # argmax_axis：沿 var0 轴取 argmax -> 两列均为 1（4>2, 3>1）
    np.testing.assert_allclose(f.argmax_axis(0).table, [1.0, 1.0])


def test_factor_errors():
    with pytest.raises(Exception):
        Factor((0, 1), np.ones(5), (2, 2))  # shape 与 cards 不符
    with pytest.raises(Exception):
        Factor((0,), np.array([1.0, 2.0]))  # 缺 cards
    with pytest.raises(Exception):
        Factor((0,), np.array([np.inf, 1.0]), (2,))


# --------------------------------------------------------------------------
# 精确性硬不变量：VE / JT vs 暴力穷举金标准
# --------------------------------------------------------------------------
EXACT_MODELS = (
    [S.chain(10, 2, sd) for sd in (1, 2, 3)]
    + [S.tree(10, 2, sd) for sd in (1, 2, 3)]
    + [S.grid_mrf(3, 3, 2, sd) for sd in (1, 2, 3)]
    + [S.multiplexer(3, 2, 2, sd) for sd in (1, 2, 3)]
    + [S.multiplexer(4, 2, 2, sd) for sd in (1, 2, 3)]
    + [S.random_loopy(8, 0.22, 2, sd) for sd in (1, 2, 3)]
)


@pytest.mark.parametrize("model", EXACT_MODELS, ids=lambda m: m.source)
def test_ve_marginals_match_gold(model):
    gold = BruteForceGold().query_marginals(model)
    ve = VariableElimination().query_marginals(model)
    for v in gold:
        assert np.max(np.abs(ve[v] - gold[v])) <= 1e-9


@pytest.mark.parametrize("model", EXACT_MODELS, ids=lambda m: m.source)
def test_junction_tree_marginals_match_gold(model):
    gold = BruteForceGold().query_marginals(model)
    jt = JunctionTree(model).query_marginals(model)
    for v in gold:
        assert np.max(np.abs(jt[v] - gold[v])) <= 1e-9


@pytest.mark.parametrize("model", EXACT_MODELS, ids=lambda m: m.source)
def test_ve_map_matches_gold(model):
    gold = BruteForceGold().map_query(model)
    ve = VariableElimination().map_query(model)
    assert ve == gold


def test_pgmfuse_matches_exact_on_low_treewidth():
    m = S.chain(12, 2, 1)
    assert PgmFuse().route(m) == "junction_tree"
    gold = BruteForceGold().query_marginals(m)
    fuse = PgmFuse().query_marginals(m)
    for v in gold:
        assert np.max(np.abs(fuse[v] - gold[v])) <= 1e-9


# --------------------------------------------------------------------------
# Loopy BP
# --------------------------------------------------------------------------
def test_lbp_exact_on_trees():
    """硬不变量：BP 在树/链结构上必须精确（曾因提前 break 失效）。"""
    cfg = Config()
    cfg.lbp_tol = 1e-13
    for m in (S.chain(10, 2, 1), S.tree(10, 2, 1)):
        gold = BruteForceGold().query_marginals(m)
        lbp = LoopyBP(cfg).query_marginals(m)
        for v in gold:
            assert np.max(np.abs(lbp[v] - gold[v])) <= 1e-8


def test_lbp_beats_mean_field_on_loopy():
    """SOTA 近似优势：loopy 图上 LBP 应显著优于朴素平均场。"""
    gold = BruteForceGold()

    def mean_err(pred, ref):
        return float(np.mean([float(np.mean(np.abs(pred[v] - ref[v]))) for v in ref]))

    gains = []
    for m in (S.grid_mrf(4, 4, 2, sd) for sd in (1, 2, 3)):
        ref = gold.query_marginals(m)
        lbp_err = mean_err(LoopyBP().query_marginals(m), ref)
        mf_err = mean_err(MeanField().query_marginals(m), ref)
        assert lbp_err < mf_err, (m.source, lbp_err, mf_err)  # 每个 loopy 实例都必须更优
        gains.append(1.0 - lbp_err / mf_err)
    assert float(np.mean(gains)) >= 0.30


def test_mean_field_runs_and_normalized():
    m = S.grid_mrf(3, 3, 2, 1)
    mf = MeanField().query_marginals(m)
    for v in mf:
        assert abs(mf[v].sum() - 1.0) < 1e-9
        assert np.all(mf[v] >= 0)


# --------------------------------------------------------------------------
# Gibbs：收敛 + 硫定 seed 逐位可复现
# --------------------------------------------------------------------------
def test_gibbs_converges_and_reproducible():
    cfg = Config()
    cfg.gibbs_samples = 20000
    cfg.gibbs_burnin = 2000
    m = S.grid_mrf(3, 3, 2, 1)
    ref = BruteForceGold().query_marginals(m)
    g1 = Gibbs(cfg, seed=42).query_marginals(m)
    g2 = Gibbs(cfg, seed=42).query_marginals(m)
    for v in ref:
        assert abs(g1[v] - g2[v]).max() == 0.0  # 逐位一致
        assert np.max(np.abs(g1[v] - ref[v])) <= 0.05  # 收敛到小误差


# --------------------------------------------------------------------------
# 图论：道德化 / min-fill / 团提取
# --------------------------------------------------------------------------
def test_chain_treewidth_is_one():
    for n in (5, 10, 20):
        adj = build_undirected(S.chain(n, 2, 1))
        order, tw = min_fill_order(adj, n)
        assert tw == 1
        assert sorted(order) == list(range(n))


def test_multiplexer_moralization_creates_clique():
    """道德图母节点互连 -> 包含 effect 与全部 causes 的团。"""
    m = S.multiplexer(3, 2, 2, 1)
    adj = build_undirected(m)
    cliques = build_cliques(adj, list(range(m.n_vars)))
    assert any({0, 1, 2, 3} <= c for c in cliques)


def test_grid_treewidth():
    adj = build_undirected(S.grid_mrf(3, 3, 2, 1))
    _, tw = min_fill_order(adj, 9)
    assert tw >= 2


# --------------------------------------------------------------------------
# 缩放：穷举不可行的实例由团树处理
# --------------------------------------------------------------------------
def test_scaling_beyond_enumeration():
    cfg = Config()
    m = S.big_grid(6, 6, 2, 8)
    assert m.full_state_count() > cfg.gold_max_states  # 枚举不可行
    with pytest.raises(Exception):
        BruteForceGold(cfg).query_marginals(m)
    marg = JunctionTree(m).query_marginals(m)
    for v in marg:
        assert abs(marg[v].sum() - 1.0) < 1e-8


# --------------------------------------------------------------------------
# 确定性 / 配置 / CLI 冒烟
# --------------------------------------------------------------------------
def test_determinism_bit_identical():
    a = run_benchmark(Config(), seeds=[1])
    b = run_benchmark(Config(), seeds=[1])
    for key in ("mean_linf", "max_linf"):
        for eng, val in a["summary"][key].items():
            if val is None:
                continue
            assert val == b["summary"][key][eng]
    # 计时具备固有抖动，比对核心结果时剔除
    ga = {k: v for k, v in a["gates"].items() if k != "jt_times_s"}
    gb = {k: v for k, v in b["gates"].items() if k != "jt_times_s"}
    assert ga == gb
    assert a["gates"]["exact_linf_pass"] and a["gates"]["map_match"]
    assert a["gates"]["lbp_vs_mf_pass"] and a["gates"]["gibbs_reproducible"]


def test_config_env_override(monkeypatch):
    monkeypatch.setenv("PGMFORGE_SEED", "123")
    monkeypatch.setenv("PGMFORGE_LBP_DAMPING", "0.3")
    c = Config.load()
    assert c.seed == 123
    assert abs(c.lbp_damping - 0.3) < 1e-12
    monkeypatch.setenv("PGMFORGE_LBP_DAMPING", "1.5")
    with pytest.raises(ValueError):
        Config.load()


def test_seed_state():
    from pgmforge.core.seed import fresh_state, get_state

    set_all(7)
    a = get_state().rand(8)
    set_all(7)
    b = get_state().rand(8)
    np.testing.assert_array_equal(a, b)  # 同 seed 逐位一致
    set_all(9)
    c = get_state().rand(8)
    assert not np.array_equal(a, c)  # 不同 seed 应产生不同流
    # fresh_state 不污染全局状态
    np.testing.assert_array_equal(fresh_state(42).rand(5), fresh_state(42).rand(5))


def test_cli_demo_smoke(capsys):
    from pgmforge.cli import main

    rc = main(["demo"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "PgmForge" in out
    assert "门禁1" in out


def test_cli_marginals_unknown():
    from pgmforge.cli import main

    assert main(["marginals", "nosuchmodel"]) == 2


def test_discrete_model_post_init():
    m = DiscreteModel(3, [2, 2, 2], [], kind="mrf")
    assert m.var_names == ["X0", "X1", "X2"]
    assert m.var(1).card == 2
    with pytest.raises(Exception):
        DiscreteModel(3, [2], [])


def test_available_flags():
    from pgmforge.core.interfaces import InferenceEngine

    for eng in (
        VariableElimination,
        JunctionTree,
        LoopyBP,
        MeanField,
        Gibbs,
        BruteForceGold,
        PgmFuse,
    ):
        assert eng.available() is True  # 统一为 staticmethod：类上即可调用
    assert hasattr(InferenceEngine, "query_marginals")
    assert hasattr(InferenceEngine, "map_query")
