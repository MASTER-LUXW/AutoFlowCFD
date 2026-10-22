"""阶数切换（Order Continuation 换阶）中与后端无关的部分：单机 CPU / 单 GPU / CPU-MPI 两种加载 / 多 GPU 两种加载
六条路径共用，后端只负责主机与设备之间的搬运、分区与面几何的重建、壁面距离、幽灵态等几何量。

* `switched_fields`：主机数组上把守恒量、湍流输运场、涡粘换到新阶数；
* `install_turbulence`：把换好的湍流场写回模型（GPU 后端传上传函数）；
* `switch_mesh_order`：释放其他阶数的几何缓存、网格切到新阶数，返回新阶数的 FR 算子；
* `reset_order_dependent_state`：换阶后失效的跨步状态（SGS 涡粘、双时间步上一时间层、Newton 跨步状态）。

2026-10-22（I-107）以前这几段在六处各写一遍，已经分叉：GPU 两条路径换阶时不清 SGS 涡粘（下一步按旧阶数形状的涡粘
与新阶数状态相乘：P1->P2 广播失败，P0->P1 被静默广播成常数）；涡粘形状与旧阶数不符时各处都原样保留旧数组。
"""

from typing import Callable, List, NamedTuple, Optional

import numpy as np

from autoflowcfd.fr.native_padding import sps_width


class SwitchedFields(NamedTuple):
    """新阶数下的主机数组。`turbulence` / `nu_t` 在没有输运湍流模型时为 None。"""
    U: np.ndarray                              # (n, new_n_sps, n_vars)
    turbulence: Optional[List[np.ndarray]]     # 输运场（物理量形式，按模型 TRANSPORTED_FIELDS 顺序）
    nu_t: Optional[np.ndarray]                 # (n, new_n_sps)


def switched_fields(old_order: int, new_order: int, n_prism: int, freestream: dict, U: np.ndarray,
                    turb_model=None, turbulence=None, nu_t: Optional[np.ndarray] = None) -> SwitchedFields:
    """把一组单元（前 `n_prism` 行是棱柱）的场从 `old_order` 换到 `new_order`。

    * 升阶：旧阶数多项式在新解点上求值（`fr/order_interp.py::apply_order_interp`，原生四面体 / 棱柱各自的精确
      延拓）；输运场在未知量空间延拓（`TransportedTurbulence.mapped_fields`：SST 的多项式表示 ln omega，在
      物理 omega 上延拓会在单元内下冲出负值）；涡粘按同一算子延拓。
    * 降阶：只发生在全新求解器开始爬坡时从构造阶数重建到 P0（`run.py`），不是精确操作（粗化投影无意义），
      取均匀自由来流（速度方向取自攻角 / 侧滑角）、模型的来流湍流值、零涡粘。

    Args:
        U: (n, old_n_sps, n_vars) 守恒量
        turb_model: 输运湍流模型（SST 族 / SA-neg），没有时 None
        turbulence: 输运场的主机数组（与 `U` 同行数）；None 时取模型自己的当前场（CPU 后端）
        nu_t: 涡粘主机数组；形状与旧阶数不符（尚未按当前场算过）时按零重建，不保留旧形状的数组
    """
    n, old_n_sps, n_vars = U.shape
    new_n_sps = sps_width(new_order)
    if new_order > old_order:
        from autoflowcfd.fr.order_interp import apply_order_interp

        def lift(field):
            return apply_order_interp(field, n_prism, old_order, new_order)

        U_new = lift(U)
        if turb_model is None:
            return SwitchedFields(U_new, None, None)
        fields = list(turb_model.mapped_fields(lift, np, fields=turbulence))
        if nu_t is not None and nu_t.shape == (n, old_n_sps):
            nu_t_new = lift(nu_t)
        else:
            nu_t_new = np.zeros((n, new_n_sps))
        return SwitchedFields(U_new, fields, nu_t_new)

    from autoflowcfd.core.utils.flow_direction import freestream_conservative_state

    U_new = np.empty((n, new_n_sps, n_vars))
    U_new[:] = freestream_conservative_state(freestream, n_vars)
    if turb_model is None:
        return SwitchedFields(U_new, None, None)
    return SwitchedFields(U_new, list(turb_model.freestream_fields((n, new_n_sps), np)), np.zeros((n, new_n_sps)))


def install_turbulence(turb_model, switched: SwitchedFields, to_array: Callable = np.asarray) -> None:
    """把 `switched` 的输运场与涡粘写回 `turb_model`（GPU 后端传 `cp.asarray`，在目标设备上下文里调用）。"""
    if turb_model is None:
        return
    if switched.turbulence is None or switched.nu_t is None:
        raise ValueError("switched_fields 调用时没有给出湍流模型，换出的场里没有湍流部分")
    turb_model.set_transported_fields([to_array(f) for f in switched.turbulence])
    turb_model.nu_t = to_array(switched.nu_t)


def switch_mesh_order(mesh, order: int):
    """释放其他阶数的几何缓存（大网格上同时驻留多份阶数几何会 OOM）、网格切到 `order`，返回该阶的 FR 算子。"""
    from autoflowcfd.fr.operators import generate_fr_operators

    for o in [o for o in list(mesh._order_geometry_cache) if o != order]:
        del mesh._order_geometry_cache[o]
    mesh.set_order(order)
    return generate_fr_operators(order)


def reset_order_dependent_state(solver, sgs_model) -> None:
    """换阶后失效的跨步状态：SGS 涡粘（`sgs_model`，CPU 后端的 `sgs_model` / GPU 后端的 `sgs_model_gpu`；在残差之后
    才按当时的速度梯度算出，下一步按新维度重算，这一步只用分子粘度）、双时间步上一时间层（形状不再匹配、也不是
    同一离散空间的解，BDF2 干净地退化为 BDF1）、Newton 跨步状态（`reset_newton_state` 文档）。"""
    from autoflowcfd.core.time_integration.implicit.newton.mean_flow_step import reset_newton_state
    from autoflowcfd.core.time_integration.implicit.newton.physical_time import reset_dual_time_history

    if sgs_model is not None:
        sgs_model.nu_t = None
    reset_dual_time_history(solver)
    reset_newton_state(solver)
