"""AutoFlowCFD V2.0 - 湍流源项与输运的每步求值。

从 `core/fr_solver/turbulence.py` 拆出（2026-09-24）。纯搬家，逻辑未改。
"""

from typing import Optional

import numpy as np

from autoflowcfd.core.turbulence.des import IDDESModel
from autoflowcfd.core.turbulence.registry import is_sa_family, is_sst_family
from autoflowcfd.core.turbulence.transported import TurbulenceRates
from autoflowcfd.core.fr_operators.corrected_gradient import source_velocity_gradient
from autoflowcfd.core.fr_operators.face_kernels import get_flat_face_geometry
from .init import (
    _filter_matrices_are_identity,
    _update_production_ramp,
)


def compute_turbulence_source(solver, dt) -> Optional[tuple]:
    """计算湍流模型源项（对应 FRSolver.compute_turbulence_source）。

    Args:
        dt: k/omega 场显式更新使用的时间步长，直接转发给
            `turb_model.update_fields`。调用方（fr_solver/step.py）
            按 scheme 传入不同的量：稳态加速模式（SSP-RK/IMEX）传
            逐 SP 的局部 CFL 步长数组 dt_local（形状 (n_cells, n_sps)，
            与 dk_total/domega_total 广播兼容），DUAL_TIME 模式传标量
            物理 dt。之前这里统一收到的是原始物理 dt 标量，未经
            cfl.py 的阶数/粘性/几何刚性收紧，真实复现过在合成 Couette
            +SST 算例与 cube_demo 生产网格上都会让 omega 场显式积分
            失稳（一步内放大几十倍，Order Continuation 升阶后几步内
            发散至 inf/NaN）——修复见 step.py::step 文档。
    """
    if solver.turb_model is None:
        return None

    # 更新湍流产项渐变因子（每步调用，production_factor 从 0 渐增到 1）
    _update_production_ramp(solver)

    Q, grad_vel, d_wall, mu = prepare_turbulence_inputs(solver)
    rates = evaluate_turbulence_rates(solver, Q, grad_vel, d_wall, mu, apply_des=True)

    solver.turb_model.update_fields(dt, rates.source, rates.transport)
    finalize_turbulence_update(solver)
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


def evaluate_turbulence_rates(solver, Q, grad_vel, d_wall, mu, *, apply_des: bool, conv_geom=None):
    """在 `turb_model` 当前的输运场上求各 Newton 未知量的时间导数。

    SA-neg 走 `turbulence/sa/rates.py::evaluate_sa_rates`（后端无关的一份算法，这里注入 CPU 的
    标量输运原语）；SST 族在 `k_field/omega_field` 上求 `dk/dt` 与 `dw/dt`（`w = ln omega`，见
    `core/turbulence/sst/log_omega.py`），下面各段说明都是 SST 的。

    返回 `TurbulenceRates`（`turbulence/transported.py`）：`raw = (Sk, S_omega)` 是模型源项
    （带 rho），`source = (Sk/rho, S_omega/(rho omega))` 是**源项部分**，`transport` 是输运部分
    ——显式路径的 `update_fields` 只对源项做点隐式阻尼，所以两者必须分开。

    副作用：`compute_source_terms` 会刷新模型上的 `nu_t`、混合 `beta` 与
    realizability 下限（都是当前场的函数）。`apply_des=True` 时还按刚算出的
    `nu_t` 刷新 DES 长度尺度（供**下一步**用，见下方原注释）——隐式路径的
    每次残差求值必须传 `False`，否则 Newton 内部的试探场会改写它。

    `conv_geom`：标量对流的共享几何（只依赖平均流），隐式路径在 Newton 步起点
    算一次传入（`transport/residual.py::prepare_convection_geometry`）。
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
        from autoflowcfd.core.turbulence.limits import clip_gradient_magnitude
        from autoflowcfd.core.turbulence.sst.log_omega import log_omega

        # 梯度对 k 与 w = ln(omega) 求（被输运的量），模长上限只作用在这两者上（退化单元上
        # 的度量噪声放大，见 `turbulence/limits.py::MAX_GRADIENT_MAGNITUDE`）；模型项用物理梯度
        # grad(omega) = omega grad(w)
        grad_k = clip_gradient_magnitude(
            compute_scalar_gradient(solver.turb_model.k_field[:, :, np.newaxis], solver.ops, solver.mesh), np)
        grad_w = clip_gradient_magnitude(
            compute_scalar_gradient(log_omega(omega, np)[:, :, np.newaxis], solver.ops, solver.mesh), np)
        grad_omega = omega[:, :, np.newaxis] * grad_w

    # DDES 的有效长度尺度 (sst_model.des_length_scale) 依赖涡粘 nu_t，而
    # nu_t 只在 compute_source_terms 内部才会被重新计算（sst_model.nu_t 是
    # 上一次调用留下的缓存），两者互相依赖对方的输出，天然只能"慢一拍"：
    # apply_to_sst_model 必须排在 compute_source_terms 之后，用这一步刚算
    # 出的 nu_t 算出 des_length_scale，供*下一步*使用——这是有意为之的
    # 近似（同阶数内连续迭代时物理上完全合理），不是疏忽，因此调用顺序
    # 本身不能颠倒（真实网格已验证：颠倒后 nu_t 变成读取上一步的旧维度
    # 缓存，问题只是从 des_length_scale 转移到 nu_t，没有解决）。
    #
    # 真正需要处理的是"跨阶数切换"这一刻：des_length_scale 是按上一个阶数
    # 的 SPs 维度算出的，阶数切换后与已经正确插值过的 k_field 形状不匹配。
    # 修复见 order_continuation.py：阶数切换时显式清空 des_length_scale，
    # 让切换后的第一步自动退回标准 RANS 耗散项（不依赖过期维度的缓存），
    # 而不是在这里颠倒调用顺序。
    Sk, S_omega = solver.turb_model.compute_source_terms(Q, grad_vel, d_wall, mu, grad_k=grad_k, grad_omega=grad_omega)

    if apply_des and solver.ddes_model is not None:
        rho = Q[:, :, 0]
        nu_field = mu / np.maximum(rho, 1e-10)
        if isinstance(solver.ddes_model, IDDESModel):
            # IDDES 的 Δ/f_B/f_e 公式需要逐单元 h_max/h_wn（见
            # init_turbulence_models 里 IDDES 分支的一次性缓存），与
            # DDESModel 基类只需要 cell_volumes 的 cube_root(V) 网格
            # 尺度公式结构不同，走独立的 apply_to_sst_model_iddes。
            solver.ddes_model.apply_to_sst_model_iddes(
                solver.turb_model, d_wall, solver._iddes_h_max, solver._iddes_h_wn,
                nu_field, grad_vel,
            )
        else:
            cell_volumes = solver._get_cell_volumes()
            # h_max（2026-09-02）：`init_turbulence_models` 的 DDES 分支
            # 现在也会设置 `solver._iddes_h_max`（与 IDDES 同一处几何量、
            # 同一个一次性缓存，见该分支文档）——传入后 apply_to_sst_
            # model 改用各向异性感知的 max_edge 网格尺度，不再是
            # cube_root(V)。`getattr` 兜底：极少数不经过 init_
            # turbulence_models（例如脱离 solver 直接构造 DDESModel 的
            # 测试场景）没有这个属性时，退化为 cube_root，不报错。
            solver.ddes_model.apply_to_sst_model(
                solver.turb_model, d_wall, cell_volumes, nu_field, grad_vel,
                h_max=getattr(solver, '_iddes_h_max', None),
            )

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
    # 跨单元扩散。见 core/turbulence_transport.py 模块文档。
    transport_k = None
    transport_w = None
    if is_sst_family(solver.turb_model_name):
        from autoflowcfd.core.turbulence.transport import compute_turbulence_transport_residual
        # 扩散系数读 compute_source_terms 刚在同一组 (k, omega) 上刷新的 nu_t 与 F1；
        # grad_k / grad_w 复用上面为源项算好、裁剪过的同一份（79 万单元 P2 冗余梯度
        # 约 1 GB，2026-08-28 OOM 追查）。输运失败直接抛出，不降级为"仅源项更新"。
        transport_k, transport_w = compute_turbulence_transport_residual(
            solver, grad_k=grad_k, grad_log_omega=grad_w,
            flat_face_override=getattr(solver, "_turbulence_flat_face_override", None),
            conv_geom=conv_geom,
        )

    return TurbulenceRates((Sk, S_omega), (dk_dt, dw_dt), (transport_k, transport_w))


def finalize_turbulence_update(solver) -> None:
    """湍流场更新之后的后处理：模态滤波（非恒等滤波矩阵时）。显式与隐式路径共用。

    滤波作用在模型的 Newton 未知量上（被多项式表示、被输运的量：SST 为 k 与 `ln omega`、
    SA-neg 为 `nu_tilde`，`TransportedTurbulence.newton_unknowns`），逐列施加后经模型写回。

    **壁面 omega 不在这里做步后松弛**（2026-10-01 删除）：壁面条件只由扩散残差里
    的面 Dirichlet 施加（显式与隐式同一离散）。此前显式路径每步把壁面 owner 单元
    **全部**解点的 ln(omega) 往壁面目标值拉一半——隐式 NK 收敛到 R≈1e-8 的槽道定常
    解只做一次就被改动 2.13 倍（P1）/1.58 倍（P3），即正确的离散定常解不是显式
    路径的不动点（不动点随伪时间步变化），显式槽道 40000 步残差停在 1e4；它的存在
    理由（扩散残差缺 omega 壁面条件）09-26 起已不成立。隐式路径的同类整单元强约束
    09-30 已因同一原因删除。"""
    # 真实 bug 修复（2026-09-12，cube_demo 791,492 单元真实网格 P1 直连
    # 长程测试发现）：k/omega 场同样需要与平均流一致的模态滤波，见
    # fr_solver/filter.py::filter_scalar_field 完整推导——此前"湍流走
    # 单步显式更新、不经过多级 RK 因此不会积累混叠"的排除理由已被真实
    # 数据证伪（P1 独立发散，omega 8.6% 单元逼近安全上限，定位到具体
    # 单元内部相邻解点间出现数量级跳变，外插到面后被上风格式放大成
    # 巨大虚假对流残差）。用与平均流完全同一套 filter_prism/filter_tet
    # 矩阵，P0（n_sps=1）下矩阵退化为单位矩阵，天然是无操作。
    if solver.mesh.n_sps_per_cell > 1 and not _filter_matrices_are_identity(solver.ops):
        from autoflowcfd.core.fr_solver.filter import (
            compute_turb_troubled_mask, filter_scalar_field,
            filter_scalar_field_gated, resolve_turb_filter_gate,
        )
        model = solver.turb_model
        shape = model.transported_fields()[0].shape
        x = model.newton_unknowns(np)
        columns = [np.ascontiguousarray(x[:, j]).reshape(shape) for j in range(x.shape[1])]
        n_prism = solver.mesh.n_prism_cells
        # 门控维度与平均流的 `AFCFD_FILTER_MODE` **独立**（2026-09-15）：
        # 真实网格 250 步对照决定性证明两维的效果可以完全分离——off 与
        # sensor 两档的平均流轨迹几乎逐位相同，om_max 却差一个量级，差异
        # 全部来自这里。理由与实测数据见 filter.py::resolve_turb_filter_gate。
        # 默认 "all" 与此前行为逐位一致。
        if resolve_turb_filter_gate() == "sensor":
            order = getattr(solver, "current_order", None)
            if order is None:
                order = solver.order
            # 传感器量的是多项式表示的光滑度，被表示的是未知量本身（SST 为 w = ln(omega)）
            troubled = compute_turb_troubled_mask(columns, int(order), n_prism=n_prism)
            solver._turb_filter_troubled_frac = float(np.mean(troubled))
            filtered = [filter_scalar_field_gated(c, solver.ops.filter_prism, solver.ops.filter_tet, troubled,
                                                  n_prism=n_prism) for c in columns]
        else:
            filtered = [filter_scalar_field(c, n_prism, solver.ops.filter_prism, solver.ops.filter_tet)
                        for c in columns]
        model.set_newton_unknowns(np.stack([np.asarray(f).ravel() for f in filtered], axis=1), np)
        # 滤波可能把场值推到正性下限以下（滤波器系数含负权重，理论上
        # 可能），滤波后必须重新过一遍正性/上界限制器，不能假设滤波
        # 输出天然满足这些约束。
        solver.turb_model.apply_positivity_limiter()
