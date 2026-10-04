"""PgmFuse 自适应旗舰 + 基准评测。

PgmFuse：treewidth ≤ 阈值 → 团树精确；否则 → Loopy BP 回退。一处路由，best-of-both。
benchmark：横向对比 金标准/VE/JT/LBP/MF/Gibbs，计算 L∞ 误差、MAP 命中、确定性、缩放。

作者：晨星
"""

from __future__ import annotations

import time

import numpy as np

from ..core.config import Config
from ..core.seed import set_all
from ..core.types import DiscreteModel
from ..inference.elim_order import build_undirected, min_fill_order
from ..inference.gibbs import Gibbs
from ..inference.gold import BruteForceGold
from ..inference.junction_tree import JunctionTree
from ..inference.lbp import LoopyBP
from ..inference.mean_field import MeanField
from ..inference.ve import VariableElimination


def _linf_marginals(a: dict[int, np.ndarray], b: dict[int, np.ndarray]) -> float:
    errs = []
    for v in a:
        if v in b:
            errs.append(float(np.max(np.abs(a[v] - b[v]))))
    return max(errs) if errs else 0.0


def _mean_linf(a: dict[int, np.ndarray], b: dict[int, np.ndarray]) -> float:
    errs = []
    for v in a:
        if v in b:
            errs.append(float(np.mean(np.abs(a[v] - b[v]))))
    return float(np.mean(errs)) if errs else 0.0


class PgmFuse:
    name = "pgmfuse"

    def __init__(self, cfg: Config | None = None) -> None:
        self.cfg = cfg or Config()

    @staticmethod
    def available() -> bool:
        return True

    def route(self, model: DiscreteModel) -> str:
        _, tw = min_fill_order(build_undirected(model), model.n_vars)
        return "junction_tree" if tw <= self.cfg.jt_max_treewidth else "loopy_bp"

    def query_marginals(
        self, model: DiscreteModel, query_vars: list[int] | None = None
    ) -> dict[int, np.ndarray]:
        if self.route(model) == "junction_tree":
            return JunctionTree(model).query_marginals(model, query_vars)
        return LoopyBP(self.cfg).query_marginals(model, query_vars)


def build_benchmark_models(seeds: list[int]) -> list[DiscreteModel]:
    """构造基准模型集：精确可行 / loopy 可行 / 大规模缩放（穷举不可行）。"""
    from ..data import synthetic as S

    models: list[DiscreteModel] = []
    # 精确可行（金标准 ≤ gold_max_states）
    for sd in seeds:
        models.append(S.chain(10, 2, sd))
        models.append(S.tree(10, 2, sd))
    for sd in seeds:
        models.append(S.grid_mrf(3, 3, 2, sd))  # 9 vars, 512 states
        models.append(S.multiplexer(3, 2, 2, sd))  # 4 vars, 32 states
        models.append(S.multiplexer(4, 2, 2, sd))  # 5 vars, 32 states
        models.append(S.random_loopy(8, 0.22, 2, sd))  # 256 states
    # loopy 可行（金标准可行，用于 LBP vs MF 对比）
    for sd in seeds:
        models.append(S.grid_mrf(4, 4, 2, sd))  # 16 vars, 65536 states
        models.append(S.multiplexer(5, 2, 2, sd))  # 6 vars, 64 states
    # 缩放（穷举不可行，仅团树参照 + 计时）
    models.append(S.big_chain(50, 2, 1))
    models.append(S.big_grid(6, 6, 2, 8))  # 36 vars, 2^36 不可举
    return models


def run_benchmark(cfg: Config | None = None, seeds: list[int] | None = None) -> dict:
    cfg = cfg or Config()
    seeds = seeds or [1, 2, 3]
    set_all(cfg.seed)
    models = build_benchmark_models(seeds)
    gold = BruteForceGold(cfg)
    ve = VariableElimination()
    jt = JunctionTree
    lbp = LoopyBP(cfg)
    mf = MeanField(cfg)

    engines_marginal = {
        "ve": lambda m: ve.query_marginals(m),
        "jt": lambda m: jt(m).query_marginals(m),
        "lbp": lambda m: lbp.query_marginals(m),
        "mf": lambda m: mf.query_marginals(m),
        "pgmfuse": lambda m: PgmFuse(cfg).query_marginals(m),
    }

    model_rows = []
    agg_err: dict[str, list[float]] = {**{k: [] for k in engines_marginal}, "gibbs": []}
    agg_err_mean: dict[str, list[float]] = {**{k: [] for k in engines_marginal}, "gibbs": []}
    loopy_err: dict[str, list[float]] = {"lbp": [], "mf": []}
    map_mismatch = 0
    map_total = 0
    exact_max_err = 0.0
    scaling_ok = True
    jt_times: list[float] = []

    for m in models:
        _, tw = min_fill_order(build_undirected(m), m.n_vars)
        # 参照：金标准若可行否则团树
        gold_feasible = m.full_state_count() <= cfg.gold_max_states
        try:
            ref = gold.query_marginals(m) if gold_feasible else None
        except Exception:
            ref = None
        if ref is None:
            ref = jt(m).query_marginals(m)
        ref_src = "brute_force" if gold_feasible and ref is not None else "junction_tree"

        row: dict = {"name": m.source, "n": m.n_vars, "treewidth": tw, "ref": ref_src}
        for name, fn in engines_marginal.items():
            marg = fn(m)
            e = _linf_marginals(marg, ref)
            em = _mean_linf(marg, ref)
            agg_err[name].append(e)
            agg_err_mean[name].append(em)
            row[f"err_{name}"] = e
            if name in ("lbp", "mf") and tw >= 2 and gold_feasible:
                loopy_err[name].append(e)

        # 精确性硬不变量：VE/JT 对金标准 L∞ ≤ 1e-9
        if gold_feasible:
            exact_max_err = max(exact_max_err, row["err_ve"], row["err_jt"])

        # MAP：VE vs 金标准（仅金标准可行时）
        if gold_feasible:
            ve_map = ve.map_query(m)
            g_map = gold.map_query(m)
            map_total += 1
            if ve_map != g_map:
                map_mismatch += 1

        # 缩放计时
        if not gold_feasible:
            t0 = time.perf_counter()
            jt(m).query_marginals(m)
            dt = time.perf_counter() - t0
            jt_times.append(dt)
            row["jt_time_s"] = dt
            if dt > 60.0:
                scaling_ok = False

        model_rows.append(row)

    # Gibbs 收敛 + 确定性（取一个 loopy 可行模型）
    gibbs_cfg = Config()
    gibbs_cfg.gibbs_samples = 40000
    gibbs_cfg.gibbs_burnin = 4000
    gm = next(
        mm
        for mm in models
        if mm.source.startswith("grid_mrf(4x4") and mm.full_state_count() <= cfg.gold_max_states
    )
    ref_g = gold.query_marginals(gm)
    g1 = Gibbs(gibbs_cfg, seed=42).query_marginals(gm)
    g2 = Gibbs(gibbs_cfg, seed=42).query_marginals(gm)
    gibbs_err = _linf_marginals(g1, ref_g)
    gibbs_reproducible = bool(np.max(np.abs(np.concatenate([g1[v] - g2[v] for v in g1]))) == 0.0)
    agg_err["gibbs"].append(gibbs_err)
    agg_err_mean["gibbs"].append(_mean_linf(g1, ref_g))

    summary = {
        "mean_linf": {k: float(np.mean(v)) if v else None for k, v in agg_err.items()},
        "max_linf": {k: float(np.max(v)) if v else None for k, v in agg_err.items()},
        "mean_mean_linf": {k: float(np.mean(v)) if v else None for k, v in agg_err_mean.items()},
    }

    # LBP vs MF 相对改善（loopy 可行集）
    lbp_lo = float(np.mean(loopy_err["lbp"])) if loopy_err["lbp"] else None
    mf_lo = float(np.mean(loopy_err["mf"])) if loopy_err["mf"] else None
    lbp_rel_improve = None
    if lbp_lo is not None and mf_lo and mf_lo > 0:
        lbp_rel_improve = 1.0 - lbp_lo / mf_lo

    gates = {
        "exact_linf_vs_gold": exact_max_err,
        "exact_linf_pass": bool(exact_max_err <= 1e-9),
        "map_match": map_mismatch == 0,
        "map_mismatch": map_mismatch,
        "map_total": map_total,
        "lbp_vs_mf_rel_improve": lbp_rel_improve,
        "lbp_vs_mf_pass": bool(lbp_rel_improve is not None and lbp_rel_improve >= 0.30),
        "gibbs_err_vs_jt": gibbs_err,
        "gibbs_reproducible": gibbs_reproducible,
        "scaling_ok": scaling_ok,
        "jt_times_s": jt_times,
    }

    return {
        "meta": {
            "system": "PgmForge",
            "author": "晨星",
            "config": cfg.as_dict(),
            "seeds": seeds,
            "n_models": len(models),
        },
        "gates": gates,
        "summary": summary,
        "loopy_compare": {
            "lbp_mean_linf": lbp_lo,
            "mf_mean_linf": mf_lo,
            "lbp_rel_improve": lbp_rel_improve,
        },
        "models": model_rows,
        "determinism": {
            "gibbs_reproducible": gibbs_reproducible,
            "gibbs_max_abs_delta_run1_run2": 0.0 if gibbs_reproducible else None,
        },
    }
