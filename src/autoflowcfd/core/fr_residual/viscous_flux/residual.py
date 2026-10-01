"""AutoFlowCFD V2.0 - 粘性残差顶层：体积项 + 界面项 + IP 罚项

从 `src/autoflowcfd/core/fr_residual/viscous_flux.py`(原 560 行)拆出(2026-09-24, 项目"单文件不超 500 行"规范)。
体积项一律过积分、用与无粘同一个体积算子 K（修正项本侧通量迹已并入，2026-10-01，见
`fr/face_flux_trace.py`）；界面核只施加公共通量与内罚项。
"""

import os

import numpy as np

from autoflowcfd.core.fr_operators.gradients import compute_physical_gradient

from autoflowcfd.core.fr_operators.flux_kernels import (
    resolve_viscous_ip_constant,
)

from autoflowcfd.core.fr_operators.volume_contract import compute_adj_j, get_overintegration_context
from .overintegration import viscous_volume_term
from .pointwise import compute_temperature
from .constants import PRANDTL_TURBULENT


def compute_viscous_residual_fr(U: np.ndarray, mesh, ops, mu: float, Pr: float,
                                 mu_t_field=None, Pr_t: float = PRANDTL_TURBULENT,
                                 boundary_ghost_provider=None,
                                 flat_face_override=None) -> np.ndarray:
    """计算真实面耦合的 FR 粘性残差 dU/dt（物理空间，已除以 det(J)）。

    Args:
        U: 守恒变量 (n_cells,n_sps,n_vars)，只使用前5个欧拉变量
        mesh: HighOrderMesh（需要 face_connectivity, face_flux_points, jacobians）
        ops: FROperators
        mu: 分子动力粘度（标量）
        Pr: 分子普朗特数
        mu_t_field: 湍流涡粘度场，形状 (n_cells, n_sps) 或 None（层流/未激活
            湍流模型时视为全零）。这是 T-01/T-04/T-06 湍流-平均流耦合的
            接入点——调用方（core/fr_solver.py）负责把 SST/DDES/WALE 算出
            的 nu_t 乘以密度后传入，见该模块修复说明。
        Pr_t: 湍流普朗特数
        boundary_ghost_provider: 边界面幽灵态提供者，与无粘残差
            （fr_residual_inviscid.py）共用同一个实例/同一套 WALL/INLET/
            OUTLET/FARFIELD/SYMMETRY 逻辑（见 boundary/fr_ghost_state.py），
            签名 (face_idx, Q_owner_fp, true_normal) -> Q_ghost_fp。
            None 时退化为 DefaultGhostProvider（零梯度外插，Q_ghost=Q_owner，
            BR1 跳跃恒为零）。

            此前这里完全不使用该参数、边界面统一取 Q_n=Q_o（内部值原样
            镜像），导致 BR1 平均 Q_avg 恒等于 Q_o、跳跃项 jump_owner 恒为
            零——即固壁上不存在任何由粘性方程施加的边界约束，无滑移
            剪应力不存在（真实数值验证：真实网格上 WALL 面的粘性校正项
            逐面精确为 0，与是否退化/网格质量无关，是恒等式）。现在真正
            调用 boundary_ghost_provider 取得反映边界条件的幽灵原始变量
            （例如 WALL 用速度镜像取反构造 Q_avg 速度=0，真正的无滑移）。

            **边界"另一侧"梯度与罚项按粘性边界种类分派（2026-09-30）**：
            无滑移壁速度梯度取本侧（壁面切向速度的法向导数就是壁面剪应力，
            **不能**镜像掉）、∇T 法向镜像（绝热）；对称面/滑移壁取镜像场的
            梯度（切向牵引与法向热通量恰为零）；远场与入口流入点按 Dirichlet
            加速度与温度罚项；出口与入口回流点取零法向粘性通量。此前除绝热类
            镜像 ∇T 外一律"本侧梯度、只罚速度"，延拓分量上扩散算子没有任何
            边界条件、失去强制性，完整论证与实测见
            core/fr_operators/flux_kernels/viscous_bc.py 模块文档。非
            `BoundaryGhostStateProvider` 的 provider（DefaultGhostProvider、
            测试 stub）没有 BC 语义、幽灵态即本侧延拓，按零法向粘性通量处理。

            **修复前的实测量级，以及为什么那个数字不能再被引用**：
            79 万单元 cube_demo P1 iter=300 检查点上，温度场全域变化
            5.086 K 但**胞内** |∇T| 只有 mean=5.9e-12 / p99=6.4e-11 /
            max=3.7e-10 K/m，据此虚假壁面热通量密度
            |q_n| = k|∇T| ≈ 6.1e-14 W/m^2，相对来流焓通量密度
            rho*U*cp*T = 1.18e7 W/m^2 约 5e-21。**这个 5e-21 是被另一个
            缺陷压出来的**：legacy 模态滤波器每个 RK stage 清掉一整阶模态
            （见 fr/modal_filter.py 模块文档），P1 的解实际退化成分片常数，
            胞内梯度当然是机器零。滤波器一旦放开，实测胞内 |∇u|/(U/h) 从
            7.7e-16 跳到 O(1)，这一项的真实影响必须重新测量。
            （校验说明：上述 |∇T| 数字所用的 `compute_physical_gradient`
            调用方式已在已知线性场 T=x / T=3y 上验证到机器精度
            7.6e-16，不是算子用错导致的虚低。）

    Returns:
        residual: (n_cells, n_sps, 5)
    """
    from autoflowcfd.core.fr_residual.inviscid import conserved_to_primitive, DefaultGhostProvider

    ghost_provider = boundary_ghost_provider if boundary_ghost_provider is not None else DefaultGhostProvider()

    n_cells = mesh.n_cells
    n_sps = mesh.n_sps_per_cell

    mu_t_field = np.zeros((n_cells, n_sps)) if mu_t_field is None else mu_t_field

    Q = conserved_to_primitive(U[..., :5])  # (n_cells,n_sps,5)
    T = compute_temperature(Q)  # (n_cells,n_sps)

    grad_Q = compute_physical_gradient(Q, mesh, ops)  # (n_cells,n_sps,5,3)
    grad_vel = grad_Q[:, :, 1:4, :]  # (n_cells,n_sps,3,3): grad_vel[...,i,j]=d(u_i)/dx_j
    grad_T = compute_physical_gradient(T[:, :, None], mesh, ops)[:, :, 0, :]  # (n_cells,n_sps,3)

    det_jacs = mesh.jacobians["det_jacs"].reshape(n_cells, n_sps)
    inv_jacs = mesh.jacobians["inv_jacs"].reshape(n_cells, n_sps, 3, 3)
    adj_j = compute_adj_j(det_jacs, inv_jacs)

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
        residual = viscous_volume_term(
            Q, grad_vel, grad_T, mu_t_field, mu, Pr, Pr_t, _oi, n_sps) / det_jacs[..., None]

    # --- 界面项：numba 逐点标量 kernel（性能优化，替代原纯 Python
    # `for f in range(fc.n_faces)` 逐面循环，理由/验证方式与
    # fr_residual_inviscid.py 的同类改动完全一致，见
    # fr_viscous_flux_kernel.py 模块文档、
    # tests/unit/test_fr_viscous_flux_kernel_crosscheck.py 的新旧实现
    # 逐位对比验证）。
    from autoflowcfd.core.fr_operators.face_kernels import get_flat_face_geometry
    from autoflowcfd.core.fr_residual.inviscid_kernel import compute_boundary_ghost_states

    # #2（2026-08-28）：见 core/fr_residual/inviscid.py::
    # compute_inviscid_residual_fr 同名参数文档——CPU 分布式路径必须传入
    # 预先按 local+halo 压缩索引空间构造好的 flat_face_override，不能让
    # 这里对 DistributedMeshAdapter 重新调用 get_flat_face_geometry。
    flat = flat_face_override if flat_face_override is not None else get_flat_face_geometry(mesh, ops)
    Q_ghost = compute_boundary_ghost_states(flat, Q, adj_j, ghost_provider)
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
        import numba
        n_threads = numba.get_num_threads()
        correction = compute_viscous_interface_correction_p0_kernel(
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
            n_threads,
            flat.owner_cube_face, flat.neighbor_cube_face,
            flat.ref_area_weight,
            flat.boundary_extrap_native, flat.lift_native,
            flat.ip_length,
            _c_ip,
        )
    else:
        # P≥1 通用路径：图着色或 per-thread buffer
        from autoflowcfd.core.fr_residual.viscous_flux_kernel import (
            compute_viscous_interface_correction_kernel,
        )
        from autoflowcfd.core.fr_residual.viscous_flux_kernel_colored import (
            compute_viscous_interface_correction_kernel_colored,
        )
        # 图着色方案：同色面无 owner_cell 冲突，直接写入共享 buffer
        # 内存从 O(n_threads * n_cells * n_sps * 5) 降至 O(n_cells * n_sps * 5)
        # 着色结果已缓存在 flat 中（build 时一次性计算，不再重复着色）
        # 通过环境变量或配置可切换回 per-thread buffer 方案
        use_coloring = os.environ.get("AFCFD_USE_COLORING", "1") == "1"
        
        if use_coloring:
            correction = np.zeros((n_cells, n_sps, 5))
            for c in range(flat.n_colors):
                face_indices = flat.color_face_indices[c]
                if len(face_indices) == 0:
                    continue
                compute_viscous_interface_correction_kernel_colored(
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
                    flat.owner_cube_face, flat.neighbor_cube_face,
                    flat.ref_area_weight,
                    flat.boundary_extrap_native, flat.lift_native,
                    flat.ip_length,
                    _c_ip,
                )
        else:
            # 回退到 per-thread buffer 方案（小网格 + 低线程数可能更快）
            import numba
            n_threads = numba.get_num_threads()
            correction = compute_viscous_interface_correction_kernel(
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
                n_threads,
                flat.owner_cube_face, flat.neighbor_cube_face,
                flat.ref_area_weight,
                flat.boundary_extrap_native, flat.lift_native,
                flat.ip_length,
                _c_ip,
            )
    residual = residual + correction

    # 机制3 已删除，依据见 `fr_residual/inviscid.py` 同一处的完整记录
    # （真实网格消融对照：触发 15 次但残差轨迹只差 ~1e-10、结局不变）。
    return residual
