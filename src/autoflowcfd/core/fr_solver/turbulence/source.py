"""AutoFlowCFD V2.0 - 湍流源项与输运的每步求值。"""

from typing import Optional

import numpy as np

from autoflowcfd.core.turbulence.registry import is_sa_family, is_sst_family
from autoflowcfd.core.turbulence.transported import TurbulenceRates
from autoflowcfd.core.fr_operators.corrected_gradient import source_velocity_gradient
from autoflowcfd.core.fr_operators.face_kernels import get_flat_face_geometry
from autoflowcfd.core.turbulence.des import des_length_for
from .init import (
    _update_production_ramp,
)


def compute_turbulence_source(solver, dt) -> Optional[tuple]:
    """湍流场的一次更新（对应 FRSolver.compute_turbulence_source）。

    Args:
        dt: 一次更新的（伪）时间步长，转发给 `turb_model.update_fields`：逐 SP 的局部步长
            （形状 (n_cells, n_sps)，按物理波速，cfl.py 的阶数/粘性/几何刚性收紧都在里面）。
    """
    if solver.turb_model is None:
        return None

    # 更新湍流产项渐变因子（每个物理步一次，production_factor 从 0 渐增到 1）
    _update_production_ramp(solver)

    Q, grad_vel, d_wall, mu = prepare_turbulence_inputs(solver)
    rates = evaluate_turbulence_rates(solver, Q, grad_vel, d_wall, mu)

    from autoflowcfd.core.turbulence.transported import advance_transported_fields

    advance_transported_fields(solver.turb_model, dt, rates, np)
    return rates.raw


def turbulence_velocity_gradient(solver, Q):
    """湍流模型（SST/DDES/IDDES 源项、LES 亚格子涡粘）用的速度梯度 `(n_cells, n_sps, 3, 3)`。

    取原始变量的速度（不是守恒变量动量的梯度：grad(rho u) != rho grad(u)，2026-09-03
    cube_demo 跨阶发散的真因）。P0 带上界面跳变的提升修正（单元内梯度恒为零，SST
    产生项随之为零、整个 P0 阶段湍流惰性），P>=1 用单元内多项式导数——理由与实测见
    `fr_operators/corrected_gradient.py` 模块文档。CPU 分布式的紧凑空间适配器带
    `halo_refresh`（`mpi/compact_halo.py`），单机求解器没有这个属性。
    """
    flat = (getattr(solver, "_turbulence_flat_face_override", None)
            or get_flat_face_geometry(solver.mesh, solver.ops))
    return source_velocity_gradient(Q, solver.mesh, solver.ops, flat, solver.boundary_ghost_provider,
                                    halo_refresh=getattr(solver, "halo_refresh", None))


def prepare_turbulence_inputs(solver):
    """一步之内只依赖平均流的输入：`(Q, grad_vel, d_wall, mu)`。

    显式路径每步调用一次；隐式路径（`implicit.py`）在整个湍流 Newton 步内
    冻结它们（平均流不动），只让 `k/omega` 变化。
    """
    Q = solver.state.Q
    grad_vel = turbulence_velocity_gradient(solver, Q)

    # 壁距是纯几何量，每个阶数都在当时的解点上由来源重新查询（`wall_distance.py`）。缺失或
    # 形状不符一律报错：此前形状不符时取单元平均再广播（2026-09-06 修掉的同类缺陷，近壁
    # 分辨率被抹平），缺失时退回"单元特征长度"——到达这里的模型全部需要壁距，两条兜底只会
    # 让近壁湍流静默变坏。
    d_wall = solver.wall_distance
    expected_shape = (solver.state.n_cells, solver.state.n_sps)
    if d_wall is None:
        raise RuntimeError(
            f"湍流模型 '{solver.turb_model_name}' 需要壁面距离场，但求解器没有施加过"
            f"（apply_wall_distance_source）——不能用估计值代替。")
    if d_wall.shape != expected_shape:
        raise RuntimeError(
            f"壁面距离场形状 {d_wall.shape} 与当前解点 {expected_shape} 不符：换阶后必须由来源"
            f"重新查询（recompute_wall_distance_for_current_order），不能插值或平均。")
    # 分子粘度：湍流方程自身扩散系数用分子粘度，与平均流粘性应力张量所用的有效粘度是两个不同量
    return Q, grad_vel, d_wall, solver.mu_molecular


def evaluate_turbulence_rates(solver, Q, grad_vel, d_wall, mu, *, conv_geom=None):
    """在 `turb_model` 当前的输运场上求各 Newton 未知量的时间导数。

    SA-neg 走 `turbulence/sa/rates.py::evaluate_sa_rates`（后端无关的一份算法，这里注入 CPU 的
    标量输运原语）；SST 族在 `k_field/omega_field` 上求 `dk/dt` 与 `dw/dt`（`w = ln omega`，见
    `core/turbulence/sst/log_omega.py`），下面各段说明都是 SST 的。

    返回 `TurbulenceRates`（`turbulence/transported.py`）：`raw = (Sk, S_omega)` 是模型源项
    （带 rho），`source = (Sk/rho, S_omega/(rho omega))` 是**源项部分**，`transport` 是输运部分
    ——显式路径的 `update_fields` 只对源项做点隐式阻尼，所以两者必须分开。

    副作用：`compute_source_terms` 会刷新模型上的 `nu_t`、混合 `beta` 与
    realizability 下限（都是当前场的函数，DES 长度尺度也在同一次求值里算出、不跨步保存）。

    `conv_geom`：标量对流的共享几何（只依赖平均流），隐式路径在 Newton 步起点
    算一次传入（`core/turbulence/transport/` 的 `prepare_convection_geometry`）。
    """
    if is_sa_family(solver.turb_model_name):
        from autoflowcfd.core.turbulence.sa import evaluate_sa_rates
        from autoflowcfd.core.turbulence.transport.primitives import CpuScalarTransport

        transport = CpuScalarTransport(solver, conv_geom, getattr(solver, "_turbulence_flat_face_override", None))
        return evaluate_sa_rates(solver.turb_model, transport, Q, grad_vel, d_wall, mu)

    grad_k = None
    grad_w = None
    grad_omega = None
    omega = solver.turb_model.omega_field
    if is_sst_family(solver.turb_model_name):
        from autoflowcfd.core.fr_residual.viscous import compute_scalar_gradient
        from autoflowcfd.core.turbulence.sst.log_omega import log_omega

        # 梯度对 k 与 w = ln(omega) 求（被输运的量）；模型项用物理梯度 grad(omega) = omega grad(w)。
        # 2026-10-17 以前两者都按有量纲绝对上限 1e6 截断：近壁 |grad ln omega| ~ 2/y 在 y < 2e-6 m 时被截，
        # 改变交叉扩散与 ln omega 方程的 |grad w|^2 源项（I-42）
        grad_k = compute_scalar_gradient(solver.turb_model.k_field[:, :, np.newaxis], solver.ops, solver.mesh)
        grad_w = compute_scalar_gradient(log_omega(omega, np)[:, :, np.newaxis], solver.ops, solver.mesh)
        grad_omega = omega[:, :, np.newaxis] * grad_w

    # DES 族：破坏项长度尺度在同一次源项求值里用本次的 F1 与 nu_t 算出（`turbulence/sst/source.py`）
    des = des_length_for(solver.ddes_model, getattr(solver, "_iddes_h_max", None), getattr(solver, "_iddes_h_wn", None))
    Sk, S_omega = solver.turb_model.compute_source_terms(Q, grad_vel, d_wall, mu, grad_k=grad_k, grad_omega=grad_omega,
                                                         des=des)

    # Sk/S_omega 是 compute_source_terms 按标准 SST 公式算出的 rho*k、
    # rho*omega 方程源项（P_k/D_k/P_omega/D_omega/CD_omega 都显式带 rho
    # 因子），但 turb_model.k_field/omega_field 存的是 k、omega 本身
    # （不是 rho*k/rho*omega，初值 1e-6/1.0 也是 k/omega 量级而非 rho*k/
    # rho*omega 量级）——直接 self.k_field += dt*Sk 会缺一个 1/rho，
    # 量纲不对。这里换算成 dk/dt ≈ Sk/rho（对缓变 rho 的标准近似：
    # d(rho*k)/dt = rho*dk/dt + k*drho/dt ≈ rho*dk/dt）再传给
    # update_fields。
    rho = Q[:, :, 0]
    dk_dt = Sk / np.maximum(rho, 1e-10)
    # w = ln(omega) 方程的源项部分（omega 由 exp(w) 产生、恒为正）
    dw_dt = S_omega / (np.maximum(rho, 1e-10) * omega)

    # 完整输运项（对流+扩散）：对 SST/DDES 模型计算 k/omega 的 FR 空间输运
    # 残差，使 k/omega 不再仅是逐点 ODE 源项弛豫，而是真正随流场对流、
    # 跨单元扩散。见 core/turbulence/transport/ 包文档。
    transport_k = None
    transport_w = None
    if is_sst_family(solver.turb_model_name):
        from autoflowcfd.core.turbulence.transport import compute_turbulence_transport_residual
        # 扩散系数读 compute_source_terms 刚在同一组 (k, omega) 上刷新的 nu_t 与 F1；
        # grad_k / grad_w 复用上面为源项算好的同一份（79 万单元 P2 冗余梯度
        # 约 1 GB，2026-08-28 OOM 追查）。输运失败直接抛出，不降级为"仅源项更新"。
        transport_k, transport_w = compute_turbulence_transport_residual(
            solver, grad_k=grad_k, grad_log_omega=grad_w,
            flat_face_override=getattr(solver, "_turbulence_flat_face_override", None),
            conv_geom=conv_geom,
        )

    return TurbulenceRates((Sk, S_omega), (dk_dt, dw_dt), (transport_k, transport_w))
