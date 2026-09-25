"""AutoFlowCFD V2.0 - 湍流输运残差的顶层编排。

从 `core/turbulence/transport.py` 拆出（2026-09-24）。纯搬家，逻辑未改。

只做一件事：把对流（`convection.py`）与扩散（`diffusion.py`）两条残差
按包 `__init__.py` 记录的符号约定相加。
"""

import numpy as np
from typing import Tuple


from autoflowcfd.core.fr_operators.gradients import (
    compute_physical_scalar_gradient, compute_physical_gradient,
)
from autoflowcfd.core.fr_operators.face_kernels import get_flat_face_geometry

from .faces import precompute_scalar_convection_geometry
from .convection import compute_scalar_convection_residual
from .diffusion import compute_scalar_diffusion_residual
from .omega_wall import (
    _compute_omega_wall_target,
    _compute_open_boundary_face_mask,
    _compute_wall_dirichlet_face_mask,
)


def compute_turbulence_transport_residual(
    solver,
    grad_vel: np.ndarray = None,
    grad_k: np.ndarray = None,
    grad_omega: np.ndarray = None,
    flat_face_override=None,
) -> Tuple[np.ndarray, np.ndarray]:
    """计算 k/omega 的完整输运残差（对流 + 扩散）。

    入口函数：从 solver 获取流场和湍流场信息，分别计算 k 和 omega 的
    对流+扩散残差，返回 dk/dt 和 domega/dt 的输运贡献（已除以密度）。

    Args:
        solver: FRSolver 实例（需要已初始化 SST/DDES 湍流模型）
        grad_vel, grad_k, grad_omega: 可选，调用方（`fr_solver_
            turbulence.compute_turbulence_source`）如果已经算过这三个量，
            直接传进来复用，跳过内部重新计算——性能优化：唯一真实调用方
            `compute_turbulence_source` 在调用本函数*之前*就已经为
            `compute_source_terms` 算过完全相同的 grad_vel/grad_k/
            grad_omega（同一个 solver.state.U/turb_model.k_field/
            omega_field，同一套 mesh/ops，数学上是同一个量），此前这里
            总是无条件重新算一遍——`compute_physical_gradient` 是本项目
            profile 过的真实热点（79万单元 P1 阶段单步 7.5s 累计），这里
            的重复调用是三次里的一次，真实测得省下约 1.6s/步。三者任一
            为 None 时退回原来的内部计算（保持本函数可独立调用的公开
            API 行为不变，不依赖调用方一定会传）。
        flat_face_override: 显式传入时优先使用，透传给内部四次
            `compute_scalar_convection_residual`/`compute_scalar_
            diffusion_residual` 调用（2026-09-02 分布式湍流移植新增，
            见这两个函数同名参数文档）——分布式路径下 `solver` 是
            `DistributedTurbulenceSolverAdapter`（`solver.mesh` 是
            `DistributedMeshAdapter`），必须传入 `dist_fc.base_flat`，
            否则会尝试从压缩索引空间的适配器重新构建全局面几何。

    Returns:
        (dk_dt_transport, domega_dt_transport): 各自 (n_cells, n_sps)，
        输运项对 dk/dt 和 domega/dt 的贡献
    """
    if solver.turb_model is None or not hasattr(solver.turb_model, 'k_field'):
        n_cells, n_sps = solver.state.U.shape[:2]
        return np.zeros((n_cells, n_sps)), np.zeros((n_cells, n_sps))

    turb = solver.turb_model
    Q = solver.state.Q
    rho = Q[:, :, 0]  # (n_cells, n_sps)
    vel = Q[:, :, 1:4]  # (n_cells, n_sps, 3)

    mu = solver.mu_molecular
    rho_nu_t = rho * turb.nu_t  # 动力涡粘度 mu_t = rho * nu_t

    # 计算有效扩散系数 Gamma_k, Gamma_omega
    # 需要 F1 blending 来确定 sigma_k, sigma_omega
    if grad_vel is None:
        # 真实 bug 修复（2026-09-03）：同 fr_solver/turbulence.py::
        # compute_turbulence_source 里的 grad_vel 修复——不能对*守恒*
        # 变量 U 求梯度再切片动量分量冒充速度梯度，见该处文档。这里
        # `Q`/`vel`（上面已经从 solver.state.Q 取出的原始变量）本来就是
        # 正确的速度，直接对它求梯度。
        grad_vel = compute_physical_gradient(vel, solver.mesh, solver.ops)
    S_mag = turb.compute_strain_rate_magnitude(grad_vel)
    nu = mu / np.maximum(rho, 1e-10)

    # 交叉扩散项（F1 计算需要）
    if grad_k is None:
        grad_k = compute_physical_scalar_gradient(turb.k_field, solver.mesh, solver.ops)
    if grad_omega is None:
        grad_omega = compute_physical_scalar_gradient(turb.omega_field, solver.mesh, solver.ops)

    # 梯度幅值裁剪（真实 bug，已修复，2026-08-21）：这里的 grad_k/grad_omega
    # 此前完全没有上限保护——`fr_solver/turbulence.py::compute_turbulence_
    # source` 里给 compute_source_terms 用的那一份 grad_k/grad_omega 早就有
    # 同样的 max_grad_mag=1e6 裁剪（见该文件"正性保持检查"注释），但本函数
    # 参数文档明确说明这里*刻意*不复用那份裁剪后的值、自己独立重新计算，
    # 于是这份独立计算的副本一直没有对应的裁剪。真实复现（cube_demo 生产
    # 网格，P1 阶数，DDES）：mesh 在坍缩坐标+度量退化单元（troubled_cell.py
    # 诊断此网格 P1 阶段 95.15% 单元面法向失配>1度）上，对*理论上处处为
    # 常数*的初始 k/omega 场求梯度，参考空间导数本应恰好为 0，但浮点舍入
    # 误差量级的非零值被 adj(J)/det(J) 这个在退化单元上可以任意大的度量
    # 比值放大到 >1e150（np.linalg.norm 内部计算 x*x 时溢出到 inf，py-spy
    # 采样证实的真实复现）——多数为普通浮点噪声，但间或有值落入次正规数
    # （denormal/subnormal）区间，x86 硬件处理这类数值要走慢得多的微码
    # 路径：单次 `np.sum(grad_k*grad_omega, axis=-1)`（下面这一行）在
    # ~19M 元素规模上因此实测卡住数分钟，而不是正常的毫秒级——是一次
    # "看起来像死锁、实际是每个浮点算子被拖慢几十~上百倍"的真实性能故障，
    # py-spy 对卡住进程的调用栈采样直接定位到本行。与 fr_solver/
    # turbulence.py 用完全相同的裁剪公式（不是发明新阈值，是把已经在
    # 别处验证过、这里唯一遗漏的同一道安全网补齐）。
    # np.linalg.norm 内部对每个分量求平方——在同一类退化单元上分量本身
    # 就已经是溢出级别的量，平方会先于这里的裁剪逻辑触发一次 inf；
    # errstate 只是抑制这一步的警告噪音，紧接着的 np.maximum(...,1e-10)/
    # np.clip(...,0,1) 已经能正确处理 inf 输入（inf>max_grad_mag 恒真，
    # scale=max_grad_mag/inf=0，裁剪结果趋于 0，不是 nan），不依赖这个
    # errstate 才能得到正确结果。
    with np.errstate(over='ignore', invalid='ignore'):
        max_grad_mag = 1e6
        grad_k_mag = np.linalg.norm(grad_k, axis=-1)
        grad_omega_mag = np.linalg.norm(grad_omega, axis=-1)
        if np.any(grad_k_mag > max_grad_mag):
            scale_k = max_grad_mag / np.maximum(grad_k_mag, 1e-10)
            grad_k = grad_k * np.clip(scale_k, 0, 1)[..., None]
        if np.any(grad_omega_mag > max_grad_mag):
            scale_omega = max_grad_mag / np.maximum(grad_omega_mag, 1e-10)
            grad_omega = grad_omega * np.clip(scale_omega, 0, 1)[..., None]

        grad_dot = np.sum(grad_k * grad_omega, axis=-1)
        omega_safe = np.maximum(turb.omega_field, 1e-10)
        CD_kw = np.maximum(2.0 * rho * turb.sigma_w2 / omega_safe * grad_dot, 1e-10)

    F1 = turb.compute_blending_function_F1(
        turb.k_field, turb.omega_field, solver.wall_distance, nu, S_mag, rho, CD_kw
    )

    sigma_k = F1 * turb.sigma_k1 + (1.0 - F1) * turb.sigma_k2
    sigma_w = F1 * turb.sigma_w1 + (1.0 - F1) * turb.sigma_w2

    gamma_k = mu + sigma_k * rho_nu_t    # (n_cells, n_sps)
    gamma_w = mu + sigma_w * rho_nu_t    # (n_cells, n_sps)

    # WALL 上 k=0 的 Dirichlet 掩码（真实修复，2026-08-21，见
    # transport_kernel.py::extrapolate_scalar_to_faces_kernel 文档）。
    # 数值作用点：对流项的上风 phi ghost（镜像成 -owner 强制壁面 k=0）；
    # 扩散项自 2026-08-25 校正改梯度差形式后该掩码不再有数值影响（奇镜像
    # 对梯度对称，见 compute_scalar_diffusion_residual 参数文档），仍传入
    # 以保持接口一致。
    wall_mask_k = _compute_wall_dirichlet_face_mask(solver)

    # 共享几何量（性能优化 2026-09-13，见 `ScalarConvectionGeometry` 文档）：
    # k 与 omega 的对流调用此前各自重复算了一遍与标量无关的逆变质量通量
    # 和面上 mass_flux，这里统一算一次传给两者。
    _flat_conv = (flat_face_override if flat_face_override is not None
                  else get_flat_face_geometry(solver.mesh, solver.ops))
    conv_geom = precompute_scalar_convection_geometry(
        rho, vel, solver.mesh, solver.ops, _flat_conv,
    )

    # 开放边界（流入/流出）掩码：k/omega 的来流条件（见
    # compute_scalar_convection_residual 的 open_boundary_face 参数文档）
    open_mask = _compute_open_boundary_face_mask(solver, _flat_conv)

    # 计算 k 的对流 + 扩散残差
    conv_k = compute_scalar_convection_residual(
        turb.k_field, rho, vel, solver.mesh, solver.ops, wall_dirichlet_zero_face=wall_mask_k,
        flat_face_override=flat_face_override, conv_geom=conv_geom,
        open_boundary_face=open_mask, freestream_value=float(turb.k_inf),
    )
    diff_k = compute_scalar_diffusion_residual(
        turb.k_field, gamma_k, solver.mesh, solver.ops, wall_dirichlet_zero_face=wall_mask_k,
        flat_face_override=flat_face_override,
    )
    with np.errstate(over='ignore', invalid='ignore'):
        dk_dt_transport = (conv_k + diff_k) / np.maximum(rho, 1e-10)

    # WALL 上 omega 解析壁面值的 Dirichlet 目标（真实修复，V2.0 专家组
    # 盲审发现，2026-08-28，见 _compute_omega_wall_target 文档）：此前
    # omega 恒用 Neumann（零梯度）默认，是明确记录过的已知限制——现在
    # 用 Wilcox 解析式 60*nu/(beta1*d1^2) 代替。d1 需要 solver.wall_
    # distance 已经计算好（SST/DDES/WMLES 初始化时必然如此，见
    # fr_solver/turbulence.py），否则 _compute_omega_wall_target 里的
    # np.min(solver.wall_distance[...]) 会直接因 wall_distance 为 None
    # 报错——这是有意的（没有壁面距离场，压根不该假装能算出解析壁面值）。
    omega_wall_value_face, has_omega_wall = _compute_omega_wall_target(
        solver, wall_mask_k, mu, rho, flat_face_override=flat_face_override,
    )

    # 计算 omega 的对流 + 扩散残差
    conv_w = compute_scalar_convection_residual(
        turb.omega_field, rho, vel, solver.mesh, solver.ops,
        wall_dirichlet_value_face=omega_wall_value_face, has_wall_dirichlet_value=has_omega_wall,
        flat_face_override=flat_face_override, conv_geom=conv_geom,
        open_boundary_face=open_mask, freestream_value=float(turb.omega_inf),
    )
    diff_w = compute_scalar_diffusion_residual(
        turb.omega_field, gamma_w, solver.mesh, solver.ops,
        wall_dirichlet_value_face=omega_wall_value_face, has_wall_dirichlet_value=has_omega_wall,
        flat_face_override=flat_face_override,
    )
    with np.errstate(over='ignore', invalid='ignore'):
        domega_dt_transport = (conv_w + diff_w) / np.maximum(rho, 1e-10)

    # 机制3（按 (cell,SP,变量) 粒度检测残差量级异常并清零）**已于
    # 2026-09-19 整体删除**，本函数曾经在这里调用它。删除依据是真实网格
    # （plate_demo_volume_les，179,237 单元，P1 + LES）上的消融对照：
    # 机制3 触发 15 次、清零 729 个槽位，而残差轨迹与关掉它的那条只差
    # ~1e-10 相对量，两条都在 146~148 步发散。完整记录见
    # `fr_residual/inviscid.py` 同一处。
    #
    # 所以本函数现在只保留下面那道 isfinite 归零作为唯一防线 —— 那是
    # 真正必要的（退化网格上梯度/Jacobian 会产生非有限值，非有限值一旦
    # 进入 SST 的场更新就不可恢复），而"明显偏大但还是有限值"这一类
    # 由机制3 处理的情形，实测它处理与不处理没有可观测差别。
    # NaN/Inf 隔离（最后一道防线）：退化网格上梯度/Jacobian 可能产生非
    # 有限值，归零后由 SST.update_fields 的二次防护和 positivity
    # limiter 接管
    dk_dt_transport = np.where(np.isfinite(dk_dt_transport), dk_dt_transport, 0.0)
    domega_dt_transport = np.where(np.isfinite(domega_dt_transport), domega_dt_transport, 0.0)

    return dk_dt_transport, domega_dt_transport
