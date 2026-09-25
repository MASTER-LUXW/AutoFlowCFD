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

## 后端

算法（本文件的 `TurbulenceResidual` / `step_turbulence_newton`）只有一份；
单机 CPU 与单机 GPU 各提供一个**适配器**，只回答"用哪一套求值件、在哪个
数组模块上"：

    CpuTurbulenceBackend   本文件，调 `source.py` 的 prepare/evaluate/finalize
    GpuTurbulenceBackend   `core/gpu/turbulence/gpu_implicit_turbulence.py`，
                           调 `GPUFRSolver` 上同名的三个 `_..._gpu` 方法

适配器接口：`model / xp / red / shape / n_prism / order / solver`（跨步状态
挂在 `solver._newton_turb_state` 上）、`prepare()`、`rates(apply_des)`、
`wall_targets()`、`positivity()`、`finalize(dtau)`、`face_adjacency()`。
"""

import numpy as np

from autoflowcfd.core.time_integration.implicit import (
    BlockJacobiCache,
    EisenstatWalkerForcing,
    step_newton_krylov,
)
from autoflowcfd.core.time_integration.implicit.jfnk import positive_fields_limited_step
from autoflowcfd.core.time_integration.implicit.reductions import LocalReductions
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

#: 模型上被源项求值刷新的缓存属性：Newton 内部每次试探求值之后都要恢复，
#: 否则试探场会泄漏进平均流用的 `nu_t`（CPU 与 GPU 的 SST 模型同名）。
_CACHED_ATTRS = ("nu_t", "_last_beta_blend", "_omega_realizability_min")


def _current_order(solver) -> int:
    order = getattr(solver, "current_order", None)
    return int(order if order is not None else solver.order)


class CpuTurbulenceBackend:
    """单机 CPU 适配器：`fr_solver/turbulence/source.py` 的三个求值件。"""

    __slots__ = ("solver", "model", "xp", "red", "shape", "n_prism", "order", "_inputs")

    def __init__(self, solver):
        self.solver = solver
        self.model = solver.turb_model
        self.xp = np
        self.red = LocalReductions(np)
        self.shape = self.model.k_field.shape
        self.n_prism = int(solver.mesh.n_prism_cells)
        self.order = _current_order(solver)
        self._inputs = None

    def prepare(self) -> None:
        _update_production_ramp(self.solver)
        self._inputs = prepare_turbulence_inputs(self.solver)

    def rates(self, apply_des: bool):
        _, _, dk, dw, tk, tw = evaluate_turbulence_rates(self.solver, *self._inputs, apply_des=apply_des)
        return (dk if tk is None else dk + tk), (dw if tw is None else dw + tw)

    def wall_targets(self):
        return omega_wall_cell_targets(
            self.solver, getattr(self.solver, "_turbulence_flat_face_override", None))

    def positivity(self) -> None:
        self.model.apply_positivity_limiter()

    def finalize(self, dtau) -> None:
        finalize_turbulence_update(self.solver, dtau, omega_wall_relaxation=False)

    def face_adjacency(self):
        from autoflowcfd.core.fr_operators.face_kernels import get_flat_face_geometry

        ffg = get_flat_face_geometry(self.solver.mesh, self.solver.ops)
        return np.asarray(ffg.owner_cell), np.asarray(ffg.neighbor_cell)


class TurbulenceResidual:
    """冻结平均流下的 `R_t(k, omega) = -(dk/dt, domega/dt)`，形状 `(N, 2)`。

    做成类而不是闭包：它在整个 Krylov 求解期间存活（项目规范）。调用前
    适配器必须已经 `prepare()`（平均流输入在整个 Newton 步内冻结）。
    """

    __slots__ = ("_be", "_real_rows", "_wall_rows", "_wall_target", "_wall_rate")

    def __init__(self, backend):
        self._be = backend
        xp = backend.xp
        n_cells, n_sps = backend.shape
        n_real_prism, n_real_tet = real_sps_per_cell(backend.order)
        is_prism = np.arange(n_cells) < backend.n_prism
        n_real = np.where(is_prism, n_real_prism, n_real_tet)
        real_rows = (np.arange(n_sps)[None, :] < n_real[:, None]).ravel()
        self._real_rows = xp.asarray(real_rows)

        # omega 壁面强约束（见模块文档）：壁面 owner 单元的全部真实解点
        hit_cells, target = backend.wall_targets()
        hit_cells = xp.asarray(hit_cells)
        rows = (hit_cells[:, None] * n_sps + xp.arange(n_sps)[None, :]).ravel()
        keep = self._real_rows[rows]
        self._wall_rows = rows[keep]
        self._wall_target = xp.repeat(xp.asarray(target), n_sps)[keep]
        self._wall_rate = float(backend.model.beta1) * self._wall_target

    def __call__(self, kw_flat):
        be = self._be
        xp, m = be.xp, be.model
        saved_fields = (m.k_field, m.omega_field)
        saved_cache = {a: getattr(m, a) for a in _CACHED_ATTRS if hasattr(m, a)}
        m.k_field = xp.ascontiguousarray(kw_flat[:, 0]).reshape(be.shape)
        m.omega_field = xp.ascontiguousarray(kw_flat[:, 1]).reshape(be.shape)
        try:
            rate_k, rate_w = be.rates(apply_des=False)
        finally:
            m.k_field, m.omega_field = saved_fields
            for a, v in saved_cache.items():
                setattr(m, a, v)
        r = -xp.stack([rate_k.ravel(), rate_w.ravel()], axis=1)
        r[~self._real_rows] = 0.0
        r[self._wall_rows, 1] = self._wall_rate * (kw_flat[self._wall_rows, 1] - self._wall_target)
        return r


def _newton_state(backend):
    """隐式湍流步跨 Newton 步保持的状态，挂在 `solver._newton_turb_state`
    （阶数变化时由 Order Continuation 清空）。"""
    solver = backend.solver
    st = getattr(solver, "_newton_turb_state", None)
    if st is None:
        n_cells, n_sps = backend.shape
        n_real_prism, n_real_tet = real_sps_per_cell(backend.order)
        owner, neighbor = backend.face_adjacency()
        st = {
            "forcing": EisenstatWalkerForcing(),
            "dtau_scale": 1.0,
            "block": BlockJacobiCache(
                owner_cell=owner, neighbor_cell=neighbor,
                cell_is_prism=np.arange(n_cells) < backend.n_prism,
                n_sps=n_sps, n_real_prism=n_real_prism, n_real_tet=n_real_tet, n_var=2,
                red=backend.red),
            "last_info": None,
        }
        solver._newton_turb_state = st
    return st


def step_turbulence_newton(backend, dtau) -> None:
    """对 `(k, omega)` 做一个分离式 PTC-Newton 步（平均流冻结）。

    Args:
        backend: `CpuTurbulenceBackend` / `GpuTurbulenceBackend`（湍流模型
            在 `IMPLICIT_TURBULENCE_MODELS` 内）。
        dtau: 逐 SP 伪时间步长 `(n_cells, n_sps)`（在 `backend.xp` 上），与
            显式路径传给 `update_fields` 的是同一个量（物理波速下的局部步长）。
    """
    xp, m = backend.xp, backend.model
    backend.prepare()
    residual = TurbulenceResidual(backend)
    st = _newton_state(backend)

    kw0 = xp.stack([m.k_field.ravel(), m.omega_field.ravel()], axis=1)
    scales = np.array([max(float(m.k_inf), 1e-30), max(float(m.omega_inf), 1e-30)])
    kw_new, info = step_newton_krylov(
        residual, kw0, xp.asarray(dtau, dtype=xp.float64).ravel(), scales,
        forcing=st["forcing"], dtau_scale=st["dtau_scale"], block_precond=st["block"],
        physicality=positive_fields_limited_step, red=backend.red)
    st["dtau_scale"] = info["dtau_scale"]
    st["last_info"] = info

    m.k_field = xp.ascontiguousarray(kw_new[:, 0]).reshape(backend.shape)
    m.omega_field = xp.ascontiguousarray(kw_new[:, 1]).reshape(backend.shape)
    backend.positivity()
    backend.finalize(dtau)
    # 在最终场上刷新 nu_t / 混合系数 / DES 长度尺度，供平均流这一步使用
    backend.rates(apply_des=True)
