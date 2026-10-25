"""AutoFlowCFD V2.0 - 粘性残差顶层：体积项 + 界面项 + IP 罚项

体积项一律过积分、用与无粘同一个体积算子 K（修正项本侧通量迹已并入，2026-10-01，见
`fr/face_flux_trace.py`）；界面核只施加公共通量与内罚项。
"""


import numpy as np

from autoflowcfd.core.fr_operators.gradients import compute_physical_gradient

from autoflowcfd.core.fr_operators.flux_kernels import (
    resolve_viscous_ip_constant,
)

from autoflowcfd.core.fr_operators.volume_contract import get_overintegration_context
from .overintegration import viscous_volume_term
from .pointwise import compute_temperature
from .constants import PRANDTL_TURBULENT


def compute_viscous_residual_fr(U: np.ndarray, mesh, ops, mu: float, Pr: float,
                                mu_t_field=None, Pr_t: float = PRANDTL_TURBULENT,
                                boundary_ghost_provider=None,
                                flat_face_override=None, primitives=None) -> np.ndarray:
    """计算真实面耦合的 FR 粘性残差 dU/dt（物理空间，已除以 det(J)）。

    Args:
        U: 守恒变量 (n_cells,n_sps,n_vars)，只使用前5个欧拉变量
        mesh: HighOrderMesh（需要 face_connectivity, face_flux_points, jacobians）
        ops: FROperators
        mu: 分子动力粘度（标量）
        Pr: 分子普朗特数
        mu_t_field: 湍流动力涡粘 `rho*nu_t`，形状 (n_cells, n_sps)；None（层流 / 未启用湍流模型）视为全零。
            湍流模型与平均流的耦合点：调用方（`core/fr_solver/`）把模型给出的涡粘传进来，进入应力张量与热传导。
        Pr_t: 湍流普朗特数
        boundary_ghost_provider: 边界面幽灵态提供者，与无粘残差（`core/fr_residual/inviscid.py`）共用同一个实例、
            同一套 WALL / INLET / OUTLET / FARFIELD / SYMMETRY 逻辑（`boundary/fr_ghost_state.py`），签名
            `(face_idx, Q_owner_fp, true_normal) -> Q_ghost_fp`。必须显式给出（`require_ghost_provider`；零梯度外插须
            显式传 `DefaultGhostProvider()`）。边界面上 BR1 平均态取幽灵态（例如无滑移壁的平均速度为零），没有它
            粘性方程在边界上不施加任何条件。

            **边界"另一侧"梯度与罚项按粘性边界种类分派**：无滑移壁速度梯度取本侧（壁面切向速度的法向导数就是
            壁面剪应力，不能镜像掉）、∇T 法向镜像（绝热）；对称面 / 滑移壁取镜像场的梯度（切向牵引与法向热通量
            恰为零）；远场与入口流入点按 Dirichlet 加速度与温度罚项；出口取充分发展条件。延拓分量上若不施加边界
            条件，扩散算子失去强制性，论证与实测见 `core/fr_operators/flux_kernels/viscous_bc.py` 模块文档。非
            `BoundaryGhostStateProvider` 的 provider（`DefaultGhostProvider`、测试 stub）没有 BC 语义、幽灵态即本侧
            延拓，按零法向粘性通量处理。

        flat_face_override: 分布式路径预先按 local+halo 压缩索引空间构造的面几何（见
            `fr_residual/inviscid.py::compute_inviscid_residual_fr` 同名参数）。
        primitives: 与无粘项共用的原始变量与边界幽灵态（`fr_residual/inviscid.py::mean_flow_primitives`）；
            None 时在这里算。

    Returns:
        residual: (n_cells, n_sps, 5)
    """
    from autoflowcfd.core.fr_residual.inviscid import conserved_to_primitive
    from autoflowcfd.boundary.fr_ghost_state import require_ghost_provider

    ghost_provider = require_ghost_provider(boundary_ghost_provider, "compute_viscous_residual_fr")

    n_cells = mesh.n_cells
    n_sps = mesh.n_sps_per_cell

    mu_t_field = np.zeros((n_cells, n_sps)) if mu_t_field is None else mu_t_field

    Q = conserved_to_primitive(U[..., :5]) if primitives is None else primitives.Q  # (n_cells,n_sps,5)
    T = compute_temperature(Q)  # (n_cells,n_sps)

    grad_Q = compute_physical_gradient(Q, mesh, ops)  # (n_cells,n_sps,5,3)
    grad_vel = grad_Q[:, :, 1:4, :]  # (n_cells,n_sps,3,3): grad_vel[...,i,j]=d(u_i)/dx_j
    grad_T = compute_physical_gradient(T[:, :, None], mesh, ops)[:, :, 0, :]  # (n_cells,n_sps,3)

    det_jacs = mesh.jacobians["det_jacs"].reshape(n_cells, n_sps)

    # IP 罚项常数按阶数解析一次（trace 不等式常数 ~ (p+1)(p+3)/3，
    # 见 `flux_kernels.resolve_viscous_ip_constant`）。
    _c_ip = resolve_viscous_ip_constant(int(mesh.order))
    if n_sps == 1:
        # P0：分片常数场的体积散度恒为零；修正项的本侧通量由 P0 核自己减去（P0 没有
        # 体积算子可以并入，见 viscous_p0_kernel.py）
        residual = np.zeros((n_cells, n_sps, 5))
    else:
        # 粘性通量（tau、u·tau、k grad T 再乘 adj(J)）在细点上重新求值（去混叠），体积
        # 算子 K 同时减去修正项的本侧通量迹（与无粘同一个 K，fr/face_flux_trace.py）。
        # 粘性项是 +div(G)（见模块文档的符号约定）。
        _oi = get_overintegration_context(mesh, ops)
        if _oi is None:
            raise RuntimeError(
                "P>=1 粘性残差需要过积分细点度量（mesh.jacobians_fine）与 ops.overint_* 算子，"
                "这里缺失：网格或算子是按不完整的阶数几何构造的")
        residual = viscous_volume_term(Q, grad_vel, grad_T, mu_t_field, mu, Pr, Pr_t, _oi, n_sps)
        np.divide(residual, det_jacs[..., None], out=residual)      # 原地（全尺寸，见无粘残差同一处）

    # --- 界面项：numba 逐点标量 kernel（性能优化，替代原纯 Python
    # `for f in range(fc.n_faces)` 逐面循环，理由/验证方式与
    # core/fr_residual/inviscid.py 的同类改动完全一致，见
    # core/fr_residual/viscous_flux_kernel.py 模块文档、
    # tests/unit/test_fr_viscous_flux_kernel_crosscheck.py 的新旧实现
    # 逐位对比验证）。
    from autoflowcfd.core.fr_operators.face_kernels import get_flat_face_geometry
    from autoflowcfd.core.fr_residual.inviscid_kernel import compute_boundary_ghost_states

    # #2（2026-08-28）：见 core/fr_residual/inviscid.py::
    # compute_inviscid_residual_fr 同名参数文档——CPU 分布式路径必须传入
    # 预先按 local+halo 压缩索引空间构造好的 flat_face_override，不能让
    # 这里对 DistributedMeshAdapter 重新调用 get_flat_face_geometry。
    flat = flat_face_override if flat_face_override is not None else get_flat_face_geometry(mesh, ops)
    Q_ghost = compute_boundary_ghost_states(flat, Q, ghost_provider) if primitives is None else primitives.Q_ghost
    # 边界面的公共粘性通量按粘性边界种类分派（Dirichlet / 无滑移壁 / 镜像 /
    # Neumann / 入口逐点判定），见 core/fr_operators/flux_kernels/viscous_bc.py。
    from autoflowcfd.boundary.fr_ghost_state import build_viscous_boundary_kind
    vbc_kind = build_viscous_boundary_kind(flat.n_faces, flat.is_boundary, ghost_provider)

    if n_sps == 1:
        # P0 专用路径：使用 n_sps=1 特化 kernel（消除 SP 循环，外插写成
        # 标量乘——n_sps=1 下与矩阵乘是同一个运算，不是近似，见
        # viscous_p0_kernel.py 模块文档的用词更正）
        # 性能优化：Order Continuation P0 阶段 n_sps=1，通用 kernel 的
        # for s in range(n_sps) 循环虽只有 1 次迭代但仍有分支/索引开销，
        # P0 专用 kernel 在编译期消除所有 SP 循环。
        from autoflowcfd.core.fr_residual.viscous_p0_kernel import (
            compute_viscous_interface_correction_p0_kernel,
        )
        correction = np.zeros((n_cells, 1, 5))
        for face_indices in flat.color_face_indices:
            if len(face_indices) == 0:
                continue
            compute_viscous_interface_correction_p0_kernel(
                Q, grad_vel, grad_T, mu_t_field,
                det_jacs, mu, Pr, Pr_t,
                flat.owner_cell, flat.neighbor_cell, flat.is_boundary,
                flat.owner_is_primary, flat.neighbor_is_primary,
                flat.owner_adj_row_exact, flat.neighbor_adj_row_exact,
                flat.neighbor_src0_cell, flat.neighbor_src1_idx, flat.neighbor_src1_cell,
                flat.owner_src0_cell, flat.owner_src1_idx, flat.owner_src1_cell,
                flat.mixed_nb_partner, flat.mixed_ow_partner, flat.p0_second_half_frac,
                Q_ghost, vbc_kind,
                flat.owner_face_op, flat.neighbor_face_op,
                flat.ref_area_weight,
                flat.boundary_extrap_native, flat.lift_native,
                flat.ip_length,
                _c_ip,
                face_indices, correction,
            )
    else:
        # P≥1 通用路径：按面着色
        from autoflowcfd.core.fr_residual.viscous_flux_kernel import (
            compute_viscous_interface_correction_kernel,
        )
        # 图着色方案：同色面无 owner_cell 冲突，直接写入共享 buffer
        # 内存从 O(n_threads * n_cells * n_sps * 5) 降至 O(n_cells * n_sps * 5)
        # 着色结果已缓存在 flat 中（build 时一次性计算，不再重复着色）

        correction = np.zeros((n_cells, n_sps, 5))
        for c in range(flat.n_colors):
            face_indices = flat.color_face_indices[c]
            if len(face_indices) == 0:
                continue
            compute_viscous_interface_correction_kernel(
                Q, grad_vel, grad_T, mu_t_field,
                det_jacs, mu, Pr, Pr_t,
                flat.owner_cell, flat.neighbor_cell, flat.is_boundary,
                flat.owner_is_primary, flat.neighbor_is_primary,
                flat.owner_adj_row_exact, flat.neighbor_adj_row_exact,
                flat.neighbor_src0_cell, flat.neighbor_src0_tpl, flat.neighbor_src0_tid,
                flat.neighbor_src1_idx, flat.neighbor_src1_cell, flat.neighbor_src1_mat,
                flat.owner_src0_cell, flat.owner_src0_tpl, flat.owner_src0_tid,
                flat.owner_src1_idx, flat.owner_src1_cell, flat.owner_src1_mat,
                flat.mixed_nb_partner, flat.mixed_nb_mask,
                flat.mixed_ow_partner, flat.mixed_ow_mask,
                Q_ghost, vbc_kind,
                face_indices, correction,
                flat.owner_face_op, flat.neighbor_face_op,
                flat.ref_area_weight,
                flat.boundary_extrap_native, flat.lift_native,
                flat.ip_length,
                _c_ip,
            )

    residual += correction
    del correction
    return residual
