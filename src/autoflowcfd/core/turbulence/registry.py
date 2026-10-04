"""AutoFlowCFD V2.0 - 湍流模型名的分类（唯一判据）。

此前各处各写一份 `("SST", "DDES", "IDDES")` / `["SST", "DDES", "IDDES", "WMLES", "LES"]`（二十多处，
单机、GPU、分布式各有），引入新模型时要逐处同步。这里给出全部分类，调用方只问"属于哪一类"。

    SST_FAMILY          k-omega SST 模型对象（DDES/IDDES 是 SST 加 DES 长度尺度）
    SA_FAMILY           SA-neg（`turbulence/sa`）
    TRANSPORT_MODELS    带输运方程、走隐式紧耦合 Newton 的模型（上面两族）
    ALGEBRAIC_MODELS    亚格子粘性是代数的（LES / 壁模化 LES）
    WALL_DISTANCE_MODELS 需要壁面距离场的模型
    SUPPORTED_MODELS    全部合法模型名（含层流 "NONE"）
"""

SST_FAMILY = ("SST", "DDES", "IDDES")
SA_FAMILY = ("SA",)
TRANSPORT_MODELS = SST_FAMILY + SA_FAMILY
ALGEBRAIC_MODELS = ("WMLES", "LES")
WALL_DISTANCE_MODELS = TRANSPORT_MODELS + ALGEBRAIC_MODELS
SUPPORTED_MODELS = ("NONE",) + TRANSPORT_MODELS + ALGEBRAIC_MODELS


def normalized(name) -> str:
    """模型名的规范形式（大写；None 视为层流）。"""
    return "NONE" if name is None else str(name).upper()


def is_sst_family(name) -> bool:
    return normalized(name) in SST_FAMILY


def is_sa_family(name) -> bool:
    return normalized(name) in SA_FAMILY


def has_transport_equations(name) -> bool:
    return normalized(name) in TRANSPORT_MODELS


def needs_wall_distance(name) -> bool:
    return normalized(name) in WALL_DISTANCE_MODELS


def n_state_vars(name) -> int:
    """求解器状态数组的变量数。湍流输运场一律存在模型对象上（`TransportedTurbulence`），状态
    数组只需要 5 个守恒变量；SST 族历史上另带两个 k/omega 槽位 `U[..., 5:7]`（全仓库无人读取、
    残差恒为零，见 `time_integration/implicit/mean_flow_step.py` 模块文档），SA-neg 不再携带。"""
    return 7 if is_sst_family(name) else 5


def require_supported(name) -> str:
    """规范化并校验模型名，非法时报错（列出全部合法值）。"""
    n = normalized(name)
    if n not in SUPPORTED_MODELS:
        raise ValueError(f"未知湍流模型 {name!r}，合法值：{', '.join(SUPPORTED_MODELS)}")
    return n
