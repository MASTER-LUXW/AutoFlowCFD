"""AutoFlowCFD V2.0 - 湍流模型的配置层枚举与求解器 / CLI 模型名之间的唯一映射。

此前 CLI（`cli/solve/physical_constants.py`）、`api/helpers.py`、`api_config.py`（稳态与瞬态各一份）
各写一份映射，稳态那份还把未知名字静默换成 SST。
"""

from .solver_config import TurbulenceModel

#: 配置层枚举 -> 求解器 / CLI 使用的模型名（`core/turbulence/registry.py` 判据用的是这些名字的大写形式）。
_SOLVER_NAME_BY_TURBULENCE = {
    TurbulenceModel.NONE: "none",
    TurbulenceModel.SST_KW: "sst",
    TurbulenceModel.SA: "sa",
    TurbulenceModel.DDES: "ddes",
    TurbulenceModel.IDDES: "iddes",
    TurbulenceModel.WMLES: "wmles",
    TurbulenceModel.LES: "les",
}


def turbulence_solver_name(model: "TurbulenceModel") -> str:
    """配置层枚举值 -> 求解器 / CLI 的模型名。"""
    return _SOLVER_NAME_BY_TURBULENCE[TurbulenceModel(model)]


def turbulence_from_name(name: str) -> "TurbulenceModel":
    """求解器 / CLI 的模型名或枚举值（"sst"、"sst_kw"、"sa" ...）-> 枚举；未知名字报错，
    不静默换成别的模型。"""
    key = str(name).lower()
    for model, solver_name in _SOLVER_NAME_BY_TURBULENCE.items():
        if key in (solver_name, model.value):
            return model
    valid = sorted(set(_SOLVER_NAME_BY_TURBULENCE.values()) | {m.value for m in _SOLVER_NAME_BY_TURBULENCE})
    raise ValueError(f"Unknown turbulence model '{name}', expected one of {valid}")
