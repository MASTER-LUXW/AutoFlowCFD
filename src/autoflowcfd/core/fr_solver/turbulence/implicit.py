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

## 做法：与平均流紧耦合的 PTC-Newton

`(k, w)`（`w = ln omega`，见 `core/turbulence/sst/log_omega.py`）与平均流 5 个守恒变量
进同一个 Newton 系统（`time_integration/implicit/coupled_step.py` 模块文档：为什么分离式
不行、残差与预处理如何组合）。湍流这一半：

* `R_t = -(dk/dt, dw/dt)` 与显式路径**同一套**源项与输运求值（`source.py::
  evaluate_turbulence_rates`），不另写一份物理；
* 物理性限幅"k 单步变化不超过 `max(|k|, 尺度下限)` 的 50%、`|dw| <= ln 2`（omega 单步
  最多减半/加倍）"，逐**解点**（`physicality.ScaledFieldRowLimits` 的 `log_columns`；被输运的
  k 不裁剪、可越过下限，见 `turbulence/sst/bounds.py`；omega = exp(w) 恒为正）；
* 零填充槽位（原生基）的 `R_t` 置零，Newton 不动它们；
* 更新之后的上界限幅与显式路径同一套后处理（`finalize_turbulence_update`），最后在新场
  上刷新一次涡粘与 DES 长度尺度。

## omega 壁面条件：只在残差里弱施加

壁面上的 omega 由扩散残差的面 Dirichlet 罚项施加（`turbulence/transport/diffusion.py`
"边界条件"一节：目标值 `transport/omega_wall.py::_compute_omega_wall_target`），与
显式路径是同一个残差、同一个事实来源。隐式路径不再额外改写方程。

**2026-09-30 删除的整单元强约束**：此前壁面 owner 单元**全部**真实解点的 w 行被换成
`beta1 omega_t (w - ln omega_t)`（09-25 加入，当时扩散残差里还没有面 Dirichlet，
是为了替代与 Newton 不相容的显式步后投影）。高阶下它把离壁很远的解点也钉在壁面
值上：P3 贴壁单元在法向覆盖 4 排解点（直到约 0.23 倍单元外的位置），第一排解点
的 omega 被钉在 ~3900、生成/耗散比 1.9，k 在第一排长成尖峰并与平均流形成正反馈——
槽道 SST P3 在任何 CFL（固定 20 亦然）下都发散，冻结任一子系统则各自收敛。去掉
强约束后同一算例 P1/P2/P3 分别 55/59/70 步收敛（残差降 2.5e10），k 全场为正。

## 后端

湍流求值件的适配器（每个后端一个，只回答"用哪一套求值件、在哪个数组模块上"）：

    CpuTurbulenceBackend   本文件，调 `source.py` 的 prepare/evaluate/finalize
    GpuTurbulenceBackend   `core/gpu/turbulence/gpu_implicit_turbulence.py`
    分布式 / 多 GPU        `core/mpi/distributed_implicit.py`、`core/gpu/distributed/gpu_distributed_implicit.py`

适配器接口：`model / xp / red / shape / cell_is_prism / order / solver`、`advance_ramp()`
（产生项斜坡，每个 Newton 步一次）、`prepare_inputs()`（按**当前**平均流准备冻结输入与标量
对流几何，耦合残差每次求值都调）、`rates(apply_des)`、`positivity()`、`finalize(dtau)`、
`cell_colors()`、`block_assembler()`、`coarse_context()`。耦合 Newton 步再要一个包住它的
耦合适配器（`coupled_step.py` 模块文档"后端适配器"；单机 CPU 是本文件的 `CpuCoupledBackend`）。
"""

from functools import partial

import numpy as np

from autoflowcfd.core.time_integration.implicit.reductions import LocalReductions
from autoflowcfd.core.turbulence.jacobian.pointwise import CACHED_MODEL_ATTRS
from autoflowcfd.core.turbulence.sst.log_omega import log_omega, omega_from_log
from autoflowcfd.core.turbulence.transport import prepare_convection_geometry

from .init import _update_production_ramp
from .source import (
    evaluate_turbulence_rates,
    finalize_turbulence_update,
    prepare_turbulence_inputs,
)

#: 走隐式 k-omega 更新的湍流模型（带 k/omega 输运方程的那几个）。LES/WMLES
#: 的亚格子粘性是代数的，没有输运方程，继续走原有更新。
IMPLICIT_TURBULENCE_MODELS = ("SST", "DDES", "IDDES")



def _current_order(solver) -> int:
    order = getattr(solver, "current_order", None)
    return int(order if order is not None else solver.order)


def single_machine_cell_colors(solver) -> np.ndarray:
    """单机块 Jacobi 着色：残差本身用的同一份展平面几何上的贪心距离 1 着色。"""
    from autoflowcfd.core.fr_operators.face_kernels import get_flat_face_geometry
    from autoflowcfd.core.time_integration.implicit.coloring import greedy_cell_coloring

    ffg = get_flat_face_geometry(solver.mesh, solver.ops)
    return greedy_cell_coloring(ffg.owner_cell, ffg.neighbor_cell, int(solver.mesh.n_cells))


def single_machine_coupling_graph(solver):
    """单机 P0 差分耦合图（距离 2 着色 + 模板单元对），同一份展平面几何。"""
    from autoflowcfd.core.fr_operators.face_kernels import get_flat_face_geometry
    from autoflowcfd.core.time_integration.implicit.coloring import coupling_graph_from_faces

    ffg = get_flat_face_geometry(solver.mesh, solver.ops)
    return coupling_graph_from_faces(ffg.owner_cell, ffg.neighbor_cell, int(solver.mesh.n_cells))


class CpuTurbulenceBackend:
    """单机 CPU 适配器：`fr_solver/turbulence/source.py` 的三个求值件。"""

    __slots__ = ("solver", "model", "xp", "red", "shape", "cell_is_prism", "order", "_inputs",
                 "_conv_geom")

    def __init__(self, solver):
        self.solver = solver
        self.model = solver.turb_model
        self.xp = np
        self.red = LocalReductions(np)
        self.shape = self.model.k_field.shape
        self.cell_is_prism = np.arange(self.shape[0]) < int(solver.mesh.n_prism_cells)
        self.order = _current_order(solver)
        self._inputs = None
        self._conv_geom = None

    def advance_ramp(self) -> None:
        _update_production_ramp(self.solver)

    def prepare_inputs(self) -> None:
        self._inputs = prepare_turbulence_inputs(self.solver)
        self._conv_geom = prepare_convection_geometry(
            self.solver, getattr(self.solver, "_turbulence_flat_face_override", None))

    def rates(self, apply_des: bool):
        _, _, dk, dw, tk, tw = evaluate_turbulence_rates(
            self.solver, *self._inputs, apply_des=apply_des, conv_geom=self._conv_geom)
        return (dk if tk is None else dk + tk), (dw if tw is None else dw + tw)

    def positivity(self) -> None:
        self.model.apply_positivity_limiter()

    def finalize(self, dtau) -> None:
        finalize_turbulence_update(self.solver)

    def cell_colors(self):
        return single_machine_cell_colors(self.solver)

    def coarse_context(self):
        return None             # 单机：本地多层预处理的最粗层就是全局的

    def block_assembler(self):
        """解析单元块装配器（`core/turbulence/jacobian`）；输入与残差同一份冻结量（最近一次
        `prepare_inputs`）。"""
        from autoflowcfd.core.fr_operators.face_kernels import get_flat_face_geometry
        from autoflowcfd.core.turbulence.jacobian import TurbulenceBlockAssembler, turbulence_linearization

        s = self.solver
        flat = getattr(s, "_turbulence_flat_face_override", None) or get_flat_face_geometry(s.mesh, s.ops)
        ctx = turbulence_linearization(s, self.model, self._inputs, self._conv_geom, flat)
        return TurbulenceBlockAssembler(ctx, self.shape[1])


class CpuCoupledBackend:
    """单机 CPU 的耦合 Newton 适配器（`time_integration/implicit/coupled_step.py` 模块文档
    "后端适配器"）。`nu_av`：本步冻结的人工扩散系数（未启用时 None）。"""

    __slots__ = ("solver", "turb", "red", "cell_is_prism", "order", "n_sps", "scales_mean", "positivity",
                 "coupling_graph", "_nu_av", "_n_rows")

    def __init__(self, solver, nu_av=None):
        from autoflowcfd.core.fr_solver.residual_diagnostics import _reference_scales
        from autoflowcfd.core.time_integration.positivity import get_positivity_limiter

        self.solver = solver
        self.turb = CpuTurbulenceBackend(solver)
        self.red = self.turb.red
        self.cell_is_prism = self.turb.cell_is_prism
        self.order = self.turb.order
        n_cells, self.n_sps = self.turb.shape
        self._n_rows = n_cells * self.n_sps
        self.scales_mean = _reference_scales(solver.freestream, 5)
        self.positivity = get_positivity_limiter(solver)
        self.coupling_graph = partial(single_machine_coupling_graph, solver)
        self._nu_av = nu_av

    def state(self):
        s, m = self.solver, self.turb.model
        x = np.empty((self._n_rows, 7))
        x[:, :5] = np.asarray(s.state.U).reshape(self._n_rows, -1)[:, :5]
        x[:, 5] = m.k_field.ravel()
        x[:, 6] = log_omega(m.omega_field, np).ravel()
        return x

    def snapshot(self):
        m = self.turb.model
        return (self.solver.state.U, m.k_field, m.omega_field,
                {a: getattr(m, a) for a in CACHED_MODEL_ATTRS if hasattr(m, a)})

    def restore(self, snap) -> None:
        m = self.turb.model
        self.solver.state.U, m.k_field, m.omega_field = snap[0], snap[1], snap[2]
        for a, v in snap[3].items():
            setattr(m, a, v)
        self.solver.state._update_primitives()

    def set_trial(self, x) -> None:
        s, m = self.solver, self.turb.model
        U = np.array(s.state.U, copy=True)
        U.reshape(self._n_rows, -1)[:, :5] = x[:, :5]
        s.state.U = U
        s.state._update_primitives()
        m.k_field = np.ascontiguousarray(x[:, 5]).reshape(self.turb.shape)
        m.omega_field = omega_from_log(np.ascontiguousarray(x[:, 6]), m.omega_max, np).reshape(self.turb.shape)

    def trial_mu_t(self):
        from .corrections import get_turbulent_viscosity_field

        return get_turbulent_viscosity_field(self.solver)

    def mean_residual(self, mu_t):
        """平均流 `Gamma R`（`fr_solver/step.py::mean_flow_residual` 同一约定，`dU/dt = -R`）。"""
        s = self.solver
        res = s.compute_viscous_residual(mu_t_turb=mu_t, nu_av=self._nu_av)
        res += s.compute_inviscid_residual()
        res *= -1
        if s.low_mach_precond_enabled:
            from autoflowcfd.core.utils.preconditioning import apply_low_mach_preconditioner

            res = apply_low_mach_preconditioner(res, s.state.Q, s.freestream["mach_ref"], out=res)
        return res.reshape(self._n_rows, -1)[:, :5]

    def mean_assembler(self, mu_t):
        from autoflowcfd.core.fr_residual.jacobian.backend import MeanFlowBlockAssembler, unsupported_reason

        s = self.solver
        if unsupported_reason(order=self.order, entropy_stable_volume=s.entropy_stable_volume_enabled,
                              wmles=s.wmles_model is not None):
            return None
        return MeanFlowBlockAssembler(
            mesh=s.mesh, ops=s.ops, ghost_provider=s.boundary_ghost_provider, mu=s.mu_molecular,
            mach_ref=s.freestream["mach_ref"], low_mach=s.low_mach_precond_enabled, mu_t=mu_t,
            nu_av=self._nu_av, n_sps=self.n_sps)

    def cell_colors(self):
        return self.turb.cell_colors()

    def coarse_context(self):
        return None

    def install(self, x, dtau) -> None:
        """写入 Newton 步的新状态，再做湍流步后收尾（上界、finalize），并在新的平均流与
        湍流场上刷新涡粘与 DES 长度尺度（供本步之后的一切消费方使用）。"""
        self.set_trial(x)
        turb = self.turb
        turb.positivity()
        turb.finalize(dtau.reshape(turb.shape))
        turb.prepare_inputs()
        turb.rates(apply_des=True)
