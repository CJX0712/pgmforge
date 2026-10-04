"""配置：ENV_PGMFORGE_* 覆盖 + 字段校验。

作者：晨星
"""

from __future__ import annotations

import os
from dataclasses import dataclass, fields


@dataclass
class Config:
    seed: int = 7
    jt_max_treewidth: int = 12
    lbp_max_iters: int = 300
    lbp_tol: float = 1e-9
    lbp_damping: float = 0.5  # loopy 阻尼：msg = λ·old + (1-λ)·new，抑制振荡
    mf_max_iters: int = 800
    mf_tol: float = 1e-9
    gibbs_burnin: int = 2000
    gibbs_samples: int = 20000
    gold_max_states: int = 1 << 18  # 穷举金标准上限（2^18）

    @staticmethod
    def load() -> Config:
        c = Config()
        env_map = {
            "PGMFORGE_SEED": ("seed", int),
            "PGMFORGE_JT_MAX_TREEWIDTH": ("jt_max_treewidth", int),
            "PGMFORGE_LBP_MAX_ITERS": ("lbp_max_iters", int),
            "PGMFORGE_LBP_TOL": ("lbp_tol", float),
            "PGMFORGE_LBP_DAMPING": ("lbp_damping", float),
            "PGMFORGE_MF_MAX_ITERS": ("mf_max_iters", int),
            "PGMFORGE_MF_TOL": ("mf_tol", float),
            "PGMFORGE_GIBBS_BURNIN": ("gibbs_burnin", int),
            "PGMFORGE_GIBBS_SAMPLES": ("gibbs_samples", int),
            "PGMFORGE_GOLD_MAX_STATES": ("gold_max_states", int),
        }
        for env_key, (attr, caster) in env_map.items():
            raw = os.environ.get(env_key)
            if raw is None:
                continue
            try:
                setattr(c, attr, caster(raw))
            except (ValueError, TypeError) as exc:  # pragma: no cover
                from .errors import ConfigError

                raise ConfigError(f"{env_key}={raw!r} 解析失败: {exc}", "E500") from exc
        # 字段边界校验
        if c.jt_max_treewidth < 1:
            raise ValueError("jt_max_treewidth 必须 >= 1")
        if c.lbp_tol <= 0 or c.mf_tol <= 0:
            raise ValueError("tol 必须 > 0")
        if not 0.0 <= c.lbp_damping < 1.0:
            raise ValueError("lbp_damping 必须 ∈ [0, 1)")
        if c.gold_max_states < 1:
            raise ValueError("gold_max_states 必须 >= 1")
        return c

    def as_dict(self) -> dict:
        return {f.name: getattr(self, f.name) for f in fields(self)}
