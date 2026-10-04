"""PgmForge 命令行入口。

用法:
  python cli.py demo            # 跑端到端基准 + 落盘 benchmark.json
  python cli.py bench           # 同 demo
  python cli.py marginals NAME  # 对指定合成模型跑 PgmFuse 边际（示例）

作者：晨星
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from pgmforge.core.config import Config
from pgmforge.pipeline.pgm_pipeline import PgmFuse


def _cmd_demo(_args: argparse.Namespace) -> int:
    from pgmforge.examples.run_demo import main

    main()
    return 0


def _cmd_marginals(args: argparse.Namespace) -> int:
    from pgmforge.data import synthetic as S

    spec = args.name
    rng_seed = args.seed
    if spec.startswith("chain"):
        m = S.chain(args.n, 2, rng_seed)
    elif spec.startswith("tree"):
        m = S.tree(args.n, 2, rng_seed)
    elif spec.startswith("grid"):
        m = S.grid_mrf(args.rows, args.cols, 2, rng_seed)
    else:
        print(f"未知模型 {spec}")
        return 2
    fuse = PgmFuse(Config())
    print(f"路由: {fuse.route(m)}")
    marg = fuse.query_marginals(m)
    for v in sorted(marg):
        print(f"  P({m.var_names[v]}) = {marg[v].round(4)}")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="pgmforge", description="PgmForge PGM 推断系统")
    sub = ap.add_subparsers(dest="cmd")
    sub.add_parser("demo", help="端到端基准 + 落盘")
    sub.add_parser("bench", help="同 demo")
    pm = sub.add_parser("marginals", help="跑指定模型边际")
    pm.add_argument("name", nargs="?", default="chain")
    pm.add_argument("--n", type=int, default=8)
    pm.add_argument("--rows", type=int, default=4)
    pm.add_argument("--cols", type=int, default=4)
    pm.add_argument("--seed", type=int, default=1)
    args = ap.parse_args(argv)
    if args.cmd in ("demo", "bench", None):
        return _cmd_demo(args)
    if args.cmd == "marginals":
        return _cmd_marginals(args)
    ap.print_help()
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
