"""RANS 湍流模型的近壁处理方式（`wall_treatment`）。

* ``resolved``：低雷诺数壁面处理——无滑移幽灵态、粘性底层由网格解析（要求第一层 y+ ~ 1）；SST 的 omega 壁面值取
  Menter 放大式 `10 x 6 nu / (beta1 d1^2)`（d1 = 贴壁单元离壁解点壁距的最小值），k 在壁面为 0。
* ``auto``：自动壁面处理（与 CFX/Fluent 的 automatic wall treatment 同一思路）——平衡壁面模型给出壁面剪应力
  （`model.py`：Spalding 全 y+ 壁面律，粘性底层、缓冲层、对数层连续，第一层落在粘性底层时退化为
  tau_w = mu u / y，即解析值），幽灵态取滑移镜像；SST 的 omega 壁面值取 Menter 二项混合式
  `sqrt(omega_vis^2 + omega_log^2)`（OpenFOAM `omegaWallFunction` 同式，d1 取贴壁单元形心壁距，与壁面模型的
  匹配高度同一口径），k 在壁面零通量；SA-neg 的 nu_tilde 壁面值仍为 0（对数层里 nu_t = kappa u_tau y 线性，
  多项式在贴壁单元里可以表示）。第一层 y+ 从 1 到对数层都能用，按当地 y+ 自动从解析过渡到壁面函数。

WMLES 恒用平衡壁面模型，与本选项无关。

隐式稳态的解析块 Jacobian 覆盖两档的全部壁面项：壁面模型剪应力（`fr_residual/jacobian/wall_stress.py`）与
auto 档 omega 壁面目标值对贴壁单元 k 的依赖（`turbulence/jacobian/assemble.py::add_wall_target_k_coupling`）。
求解收尾按贴壁单元形心高度报告壁面 y+，并按本选项检查适用范围（`postprocess/wall_yplus.py`）。
"""

WALL_TREATMENTS = ("resolved", "auto")

#: 默认近壁处理方式（全部求解器构造参数、CLI、配置的唯一来源）
DEFAULT_WALL_TREATMENT = "resolved"


def validate_wall_treatment(value) -> str:
    """规范化并校验取值；非法值报错。"""
    v = str(value).lower()
    if v not in WALL_TREATMENTS:
        raise ValueError(f"近壁处理方式 {value!r} 不受支持（取 {' | '.join(WALL_TREATMENTS)}）")
    return v


def build_wall_model(turb_model_name, wall_treatment):
    """需要时构造平衡壁面模型（`needs_wall_model`），否则 None。全部求解器的唯一构造点：必须先于边界幽灵态
    构造——`build_boundary_ghost_provider` 据它是否为 None 决定物理无滑移壁面取滑移镜像幽灵态。"""
    if not needs_wall_model(turb_model_name, wall_treatment):
        return None
    from .model import EquilibriumWallModel

    return EquilibriumWallModel()


def needs_wall_model(turb_model_name, wall_treatment) -> bool:
    """是否需要平衡壁面模型：WMLES，或 RANS 输运模型（SA-neg、SST 族）选 auto。"""
    from autoflowcfd.core.turbulence.registry import is_sa_family, is_sst_family

    name = str(turb_model_name).upper()
    if name == "WMLES":
        return True
    return validate_wall_treatment(wall_treatment) == "auto" and (is_sst_family(name) or is_sa_family(name))
