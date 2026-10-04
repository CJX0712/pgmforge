"""PgmForge 错误体系（E100~E500）。

所有异常均为 PgmError 子类，携带稳定 code 便于定位与跨模块统一捕获。
作者：晨星
"""


class PgmError(Exception):
    code = "E000"
    """基类错误。"""

    def __init__(self, msg: str = "", code: str | None = None) -> None:
        super().__init__(msg)
        if code is not None:
            self.code = code


class FactorError(PgmError):
    code = "E100"
    """因子运算错误（形状/作用域不匹配、空表）。"""


class GraphError(PgmError):
    code = "E200"
    """图结构错误（非 DAG、不可三角化、团树非法）。"""


class InferenceError(PgmError):
    code = "E300"
    """推断错误（backend 不可用、消息不收敛、数值失效）。"""


class DataError(PgmError):
    code = "E400"
    """合成数据/载入错误（维度/分布非法、seed 越界）。"""


class ConfigError(PgmError):
    code = "E500"
    """配置错误（环境变量解析失败、字段越界）。"""
