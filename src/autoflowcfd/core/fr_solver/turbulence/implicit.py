"""AutoFlowCFD V2.0 - 隐式稳态（Newton-Krylov）下 k-omega 方程的隐式更新。

## 为什么显式那套更新在隐式路径上不能用

显式路径（`source.py::compute_turbulence_source`）每步做一次
`k += dt * (点隐式阻尼后的源项 + 输运)`，`dt` 取平均流同一个 CFL 下按物理
波速算出的局部步长。输运项（对流 + 扩散）是**显式**的，只在显式 CFL
（~0.03~0.06）下稳定。隐式路径上 SER 律会把 CFL 推到几十到上千，同一个
`dt` 让显式输运更新必然失稳；反过来若把湍流的步长钉在显式极限，平均流
每走一个 Newton 步湍流只走显式的一小步，耦合系统的收敛速度就被湍流拖回
显式量级，隐式白做。

真实观察（plate_demo P1 + SST，预处理 NK，CFL ~5，湍流仍走显式更新）：
10 步内板边单元 `k_max` 7.9 -> 15、坏单元 45 -> 129，而同一网格同一 CFL
的层流 NK 干净——这是分离求解里"一个方程隐式、另一个显式"的典型症状。

## 做法：分离式 PTC-Newton（工业 RANS 求解器的标准做法）

每个隐式步先在**冻结的平均流**上对 `(k, omega)` 做一个 PTC-Newton 步，
再对平均流做一个（用更新后的涡粘）：

    ( I/dtau + dR_t/d(k,w) ) d(k,w) = -R_t(k,w),   R_t = -(dk/dt, dw/dt)

* `R_t` 与显式路径**同一套**源项与输运求值（`source.py::
  evaluate_turbulence_rates`），不另写一份物理；
* 线性求解、SER、dtau 缩档、块 Jacobi 全部复用平均流那一套
  （`time_integration/implicit/`），物理性限幅换成"k、omega 单步相对下降
  不超过 50%"（`positive_fields_limited_step`）；
* 零填充槽位（原生基）不参与：它们的 `R_t` 置零（平均流残差在那里本来
  就恒为零），于是 Newton 不动它们；
* 更新之后的正性/上界限幅、模态滤波、omega 壁面松弛与显式路径**同一套**
  后处理（`finalize_turbulence_update`），最后在新场上求一次源项，让
  平均流这一步用到的 `nu_t` 是更新后的而不是滞后一步的。

## omega 壁面条件：残差内的强约束

显式路径每步做一次 `enforce_omega_wall_relaxation`：把壁面 owner 单元的
omega 拉向 Wilcox 解析值的一半，是**步后投影**、不在残差里。这条投影与
Newton 不相容，真实数据（棱柱通道 + SST，NK）：平均流残差 5 个量级
下降、CFL 到 1e4 之后，湍流残差停在 1.3e4、每个 Newton 步都被拒绝
（`theta = 0`）——残差全部集中在壁面单元，那里 omega 被投影钉在目标值，
而 `R_omega` 在那一点不为零，Newton 往根走、投影往回拉，永远在拉锯。

该投影自己的文档说它等价于 OpenFOAM `omegaWallFunction` 对近壁单元值的
直接设定。那在 OpenFOAM 里正是线性系统里的**强约束**（把近壁单元的
omega 方程换成"等于目标值"），隐式路径就这样做：壁面 owner 单元全部
真实解点的 omega 行换成

    R_omega = beta1 * omega_t * (omega - omega_t)

量纲与量级与该处的耗散项 `D_omega / rho = beta * omega^2` 一致，不依赖
dtau；大 dtau 下一个 Newton 步就落到目标值上。目标值与单元集合读的是
显式路径同一个函数（`transport/omega_wall.py::omega_wall_cell_targets`）。
"""

import numpy as np

from autoflowcfd.core.time_integration.implicit import (
    BlockJacobiCache,
    EisenstatWalkerForcing,
    step_newton_krylov,
)
from autoflowcfd.core.time_integration.implicit.jfnk import positive_fields_limited_step
from autoflowcfd.core.turbulence.transport import omega_wall_cell_targets
from autoflowcfd.fr.native_padding import real_sps_per_cell

from .init import _update_production_ramp
from .source import (
    evaluate_turbulence_rates,
    finalize_turbulence_update,
    prepare_turbulence_inputs,
)

#: 走隐式 k-omega 更新的湍流模型（带 k/omega 输运方程的那几个）。LES/WMLES
#: 的亚格子粘性是代数的，没有输运方程，继续走原有更新。
IMPLICIT_TURBULENCE_MODELS = ("SST", "DDES", "IDDES")

#: 模型上被 `compute_source_terms` 刷新的缓存属性：Newton 内部每次试探求值
#: 之后都要恢复，否则试探场会泄漏进平均流用的 `nu_t`。
_CACHED_ATTRS = ("nu_t", "_last_beta_blend", "_omega_realizability_min")


class TurbulenceResidual:
    """冻结平均流下的 `R_t(k, omega) = -(dk/dt, domega/dt)`，形状 `(N, 2)`。

    做成类而不是闭包：它在整个 Krylov 求解期间存活（项目规范）。
    """

    __slots__ = ("_solver", "_Q", "_grad_vel", "_d_wall", "_mu", "_shape", "_real_rows",
                 "_wall_rows", "_wall_target", "_wall_rate")

    def __init__(self, solver, Q, grad_vel, d_wall, mu):
        self._solver = solver
        self._Q, self._grad_vel, self._d_wall, self._mu = Q, grad_vel, d_wall, mu
        n_cells, n_sps = solver.turb_model.k_field.shape
        self._shape = (n_cells, n_sps)
        order = getattr(solver, "current_order", None)
        if order is None:
            order = solver.order
        n_real_prism, n_real_tet = real_sps_per_cell(int(order))
        is_prism = np.arange(n_cells) < int(solver.mesh.n_prism_cells)
        n_real = np.where(is_prism, n_real_prism, n_real_tet)
        self._real_rows = (np.arange(n_sps)[None, :] < n_real[:, None]).ravel()

        # omega 壁面强约束（见模块文档）：壁面 owner 单元的全部真实解点
        hit_cells, target = omega_wall_cell_targets(
            solver, getattr(solver, "_turbulence_flat_face_override", None))
        rows = (hit_cells[:, None] * n_sps + np.arange(n_sps)[None, :]).ravel()
        keep = self._real_rows[rows]
        self._wall_rows = rows[keep]
        self._wall_target = np.repeat(target, n_sps)[keep]
        self._wall_rate = float(solver.turb_model.beta1) * self._wall_target

    def __call__(self, kw_flat: np.ndarray) -> np.ndarray:
        m = self._solver.turb_model
        saved_fields = (m.k_field, m.omega_field)
        saved_cache = {a: getattr(m, a) for a in _CACHED_ATTRS if hasattr(m, a)}
        m.k_field = np.ascontiguousarray(kw_flat[:, 0]).reshape(self._shape)
        m.omega_field = np.ascontiguousarray(kw_flat[:, 1]).reshape(self._shape)
        try:
            _, _, dk_dt, domega_dt, tk, tw = evaluate_turbulence_rates(
                self._solver, self._Q, self._grad_vel, self._d_wall, self._mu, apply_des=False)
        finally:
            m.k_field, m.omega_field = saved_fields
            for a, v in saved_cache.items():
                setattr(m, a, v)
        rate_k = dk_dt if tk is None else dk_dt + tk
        rate_w = domega_dt if tw is None else domega_dt + tw
        r = -np.stack([rate_k.ravel(), rate_w.ravel()], axis=1)
        r[~self._real_rows] = 0.0
        r[self._wall_rows, 1] = self._wall_rate * (kw_flat[self._wall_rows, 1] - self._wall_target)
        return r


def _newton_state(solver):
    """隐式湍流步跨 Newton 步保持的状态（阶数变化时由 Order Continuation 清空）。"""
    st = getattr(solver, "_newton_turb_state", None)
    if st is None:
        from autoflowcfd.core.fr_operators.face_kernels import get_flat_face_geometry

        n_cells, n_sps = solver.turb_model.k_field.shape
        order = getattr(solver, "current_order", None)
        if order is None:
            order = solver.order
        n_real_prism, n_real_tet = real_sps_per_cell(int(order))
        ffg = get_flat_face_geometry(solver.mesh, solver.ops)
        st = {
            "forcing": EisenstatWalkerForcing(),
            "dtau_scale": 1.0,
            "block": BlockJacobiCache(
                owner_cell=ffg.owner_cell, neighbor_cell=ffg.neighbor_cell,
                cell_is_prism=np.arange(n_cells) < int(solver.mesh.n_prism_cells),
                n_sps=n_sps, n_real_prism=n_real_prism, n_real_tet=n_real_tet, n_var=2),
            "last_info": None,
        }
        solver._newton_turb_state = st
    return st


def step_turbulence_newton(solver, dtau) -> None:
    """对 `(k, omega)` 做一个分离式 PTC-Newton 步（平均流冻结）。

    Args:
        solver: FRSolver（`turb_model_name` 在 `IMPLICIT_TURBULENCE_MODELS` 内）
        dtau: 逐 SP 伪时间步长 `(n_cells, n_sps)`，与显式路径传给
            `update_fields` 的是同一个量（物理波速下的局部步长）。
    """
    m = solver.turb_model
    _update_production_ramp(solver)
    Q, grad_vel, d_wall, mu = prepare_turbulence_inputs(solver)
    residual = TurbulenceResidual(solver, Q, grad_vel, d_wall, mu)
    st = _newton_state(solver)

    kw0 = np.stack([m.k_field.ravel(), m.omega_field.ravel()], axis=1)
    scales = np.array([max(float(m.k_inf), 1e-30), max(float(m.omega_inf), 1e-30)])
    kw_new, info = step_newton_krylov(
        residual, kw0, np.asarray(dtau, dtype=np.float64).ravel(), scales,
        forcing=st["forcing"], dtau_scale=st["dtau_scale"], block_precond=st["block"],
        physicality=positive_fields_limited_step)
    st["dtau_scale"] = info["dtau_scale"]
    st["last_info"] = info

    n_cells, n_sps = m.k_field.shape
    m.k_field = np.ascontiguousarray(kw_new[:, 0]).reshape(n_cells, n_sps)
    m.omega_field = np.ascontiguousarray(kw_new[:, 1]).reshape(n_cells, n_sps)
    m.apply_positivity_limiter()
    finalize_turbulence_update(solver, dtau, omega_wall_relaxation=False)
    # 在最终场上刷新 nu_t / 混合系数 / DES 长度尺度，供平均流这一步使用
    evaluate_turbulence_rates(solver, Q, grad_vel, d_wall, mu, apply_des=True)
