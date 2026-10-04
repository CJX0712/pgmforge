"""端到端演示：跑基准、落盘 benchmark.json、做确定性二次校验、打印报告。

作者：晨星
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from pgmforge.core.config import Config
from pgmforge.pipeline.pgm_pipeline import run_benchmark


def _core_metric(res: dict):
    """提取与计时无关的核心指标，用于逐位一致性比对。"""
    g = {k: (None if k == "jt_times_s" else v) for k, v in res["gates"].items()}
    return {
        "gates": g,
        "mean_linf": res["summary"]["mean_linf"],
        "max_linf": res["summary"]["max_linf"],
        "loopy_compare": res["loopy_compare"],
    }


def _max_diff(a, b) -> float:
    if isinstance(a, dict):
        if a.keys() != b.keys():
            return float("inf")
        return max(_max_diff(a[k], b[k]) for k in a)
    if isinstance(a, list):
        if len(a) != len(b):
            return float("inf")
        return max((_max_diff(x, y) for x, y in zip(a, b, strict=False)), default=0.0)
    if isinstance(a, float) and isinstance(b, float):
        return abs(a - b)
    if a != b:
        return float("inf" if a != b else 0.0)
    return 0.0


def main() -> dict:
    cfg = Config()
    res = run_benchmark(cfg)
    res2 = run_benchmark(cfg)

    c1 = _core_metric(res)
    c2 = _core_metric(res2)
    max_diff = _max_diff(c1, c2)
    bit_identical = max_diff <= 1e-15
    res["determinism"]["full_demo_bit_identical"] = bool(bit_identical)
    res["determinism"]["max_core_metric_diff"] = float(max_diff)

    out = Path(__file__).resolve().parent.parent / "benchmark.json"
    out.write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")

    # ---- 打印报告 ----
    g = res["gates"]
    s = res["summary"]
    print("=" * 74)
    print(" PgmForge · 概率图模型推断基准（作者 晨星）")
    print("=" * 74)
    print(f" 模型数={res['meta']['n_models']}  seeds={res['meta']['seeds']}")
    print("-" * 74)
    print(" 引擎         mean L∞(vs 参照)   max L∞")
    for k in ("ve", "jt", "pgmfuse", "lbp", "mf", "gibbs"):
        print(f"  {k:<12} {s['mean_linf'].get(k)!s:<18} {s['max_linf'].get(k)!s}")
    print("-" * 74)
    print(
        f"  门禁1 精确性 VE/JT vs 金标准 L∞ = {g['exact_linf_vs_gold']:.2e}  "
        f"{'✅' if g['exact_linf_pass'] else '⚠️'} (≤1e-9)"
    )
    print(
        f"  门禁2 MAP   VE vs 金标准 命中 {g['map_total'] - g['map_mismatch']}/{g['map_total']}  "
        f"{'✅' if g['map_match'] else '⚠️'}"
    )
    lc = res["loopy_compare"]
    if lc["lbp_rel_improve"] is not None:
        print(
            f"  门禁3 LBP vs MF  相对改善 {lc['lbp_rel_improve'] * 100:.1f}%  "
            f"{'✅' if g['lbp_vs_mf_pass'] else '⚠️'} (≥30%)"
        )
    else:
        print("  门禁3 LBP vs MF   无 loopy 可行集（跳过）")
    print(
        f"  门禁4 Gibbs  err={g['gibbs_err_vs_jt']:.4f}  可复现={g['gibbs_reproducible']}  "
        f"{'✅' if g['gibbs_reproducible'] else '⚠️'}"
    )
    print(
        f"  门禁5 缩放   JT 处理 chain-50/grid-6×6 {'✅' if g['scaling_ok'] else '⚠️'}  "
        f"计时(s)={g['jt_times_s']}"
    )
    print(
        f"  确定性       demo 二次运行核心指标逐位一致 {'✅' if bit_identical else '⚠️'}  "
        f"(maxΔ={max_diff:.1e})"
    )
    print("=" * 74)
    print(f" 落盘: {out}")
    return res


if __name__ == "__main__":
    main()
