"""全局确定性入口。

唯一 seed 入口 set_all(seed)：一次性设齐 random / numpy 全局 / 返回可复用的
legacy RandomState（NEP 19 保证跨 numpy 版本逐位一致，用于 Gibbs 等随机后端）。

作者：晨星
"""

from __future__ import annotations

import random

import numpy as np

_STATE: np.random.RandomState | None = None
_DEFAULT_SEED = 7


def set_all(seed: int = _DEFAULT_SEED) -> np.random.RandomState:
    """设齐所有随机源，返回受控的 RandomState 实例。"""
    global _STATE
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    # legacy RandomState：跨 numpy 版本可复现（NEP 19），优于 Generator
    _STATE = np.random.RandomState(seed)
    return _STATE


def get_state() -> np.random.RandomState:
    global _STATE
    if _STATE is None:
        _STATE = set_all(_DEFAULT_SEED)
    return _STATE


def fresh_state(seed: int) -> np.random.RandomState:
    """返回一个全新的 RandomState（不污染全局），用于每次独立运行。"""
    return np.random.RandomState(int(seed))
