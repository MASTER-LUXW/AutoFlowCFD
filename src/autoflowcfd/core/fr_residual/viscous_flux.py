"""
AutoFlowCFD V2.0 - FR 粘性物理通量与界面耦合 (Tier-0 重建版, 对应 S-03)

牛顿流体应力张量 + 傅里叶热传导的物理通量函数，以及基于真实单元-面连接
关系的界面耦合（BR1 格式：界面处的原始变量与梯度取相邻单元外插值的平均，
这是规范文档 3_系统实现方式-算法流程.md §2.3 明确允许的做法——
"在单元界面处，Θ̂ 取左右单元的平均值（或加权平均值）"）。

取代旧版本 fr_residual_viscous.py 中：
1. 从未被满足的 `hasattr(mesh,'face_connectivity')` 分支（死代码，从未执行）
2. 唯一实际执行的 fallback——用**单元内部梯度模长**冒充界面跳跃
   （`jump_estimate = h_local * |grad_u|`），这不是任何邻居信息，纯粹是
   同一个单元自己的局部量，物理上不构成"界面耦合"
3. 体积项用 D_3d 直接当物理导数使用（缺少度量项变换，见
   core/fr_gradients.py 文档），对本代码库的每个曲边/坍缩坐标单元都是
   错误导数

正确性通过「均匀常数流场（零梯度）粘性残差应严格为零」验证——牛顿粘性
应力和热传导对常数场恒为零，这是比自由流场保持性更基础但同样严格的
判据，见 tests/unit/test_fr_residual_viscous.py。

问题单元保护：`compute_physical_gradient` 用 `inv_jac`（近似正比于
adj(J)/det(J)）把参考空间导数转成物理梯度，坍缩坐标退化 SP 处 det(J)
极小，`inv_jac` 对应地极大——梯度本身在这类点先被放大一次，随后
`residual = div_comp/det(J)`（体积项）与 `correction/det(J)`（界面项）
在同一个退化 det(J) 上再放大一次，是比无粘残差更严重的*双重*放大（真实
Couette 合成算例复现：粘性残差 3 步内从 4e-2 量级放大到 1.16e7）。

此前曾仿照无粘那边"先用 det(J)/法向失配几何量预判、按整个单元降阶"的
机制1/2 实现过一版保护，但发现该判据有两个真实缺陷（见
fr_troubled_cell.py 模块文档"机制3"一节）：(1) 绝对 det(J) 阈值是照一个
特定网格的绝对尺度标定的，换个尺度就可能失效（真实复现：det(J) 比阈值
高 828 倍仍被放大到灾难量级）；(2) 按整个单元降阶，会在网格所有单元
恰好同一绝对尺度、以至于机制1对*每个*单元都命中时（合成验证网格常见），
把全网格的粘性物理都拍平成零梯度，等于关掉了粘性扩散本身。现改用机制3
（`suppress_residual_outliers`）：直接对*算出的最终残差*做 (cell,SP,变量)
粒度的量级异常检测并清零，不依赖网格绝对尺度，也只清零真正异常的
那几个 SP，同一单元其余 SP 保留完整梯度耦合。
"""

import os
import numpy as np

from autoflowcfd.core.fr_operators.gradients import compute_physical_gradient
from autoflowcfd.core.fr_operators.troubled_cell import suppress_residual_outliers
from autoflowcfd.core.fr_operators.flux_kernels import viscous_physical_flux_batch
from autoflowcfd.core.fr_operators.volume_contract import contract_shared_operator_2axis

GAMMA = 1.4
R_AIR = 287.0  # 空气比气体常数 J/(kg*K)


def compute_temperature(Q: np.ndarray) -> np.ndarray:
    """T = p/(rho*R)。"""
    rho = np.maximum(Q[..., 0], 1e-10)
    return Q[..., 4] / (rho * R_AIR)


def viscous_physical_flux(
    Q: np.ndarray,
    grad_vel: np.ndarray,
    grad_T: np.ndarray,
    mu: float,
    Pr: float,
    mu_t=0.0,
    Pr_t: float = 0.9,
) -> np.ndarray:
    """计算粘性物理通量张量 G_i，与 euler_physical_flux 同样的 (...,3,5) 约定。

    Args:
        Q: (...,5) 原始变量 (rho,u,v,w,p)
        grad_vel: (...,3,3) 速度梯度，grad_vel[...,i,j] = d(u_i)/d(x_j)
        grad_T: (...,3) 温度梯度
        mu: 分子动力粘度（标量）
        Pr: 分子普朗特数
        mu_t: 湍流涡粘度（标量或可广播到 Q.shape[:-1] 的数组），默认0
            （层流/未提供湍流模型时）。应力张量按 Boussinesq 假设用
            mu_total=mu+mu_t 统一处理；热传导的湍流贡献用湍流普朗特数
            Pr_t（标准值0.9，非分子普朗特数）单独换算，两者不能共用同一
            个 Pr——这是本次修复把湍流涡粘度真正耦合进粘性应力张量
            （T-01/T-04/T-06）的核心：此前调用方从不传湍流粘度，
            粘性通量永远只用分子粘度。
        Pr_t: 湍流普朗特数

    Returns:
        G: (...,3,5)，G[...,i,:] 是方向 i 的粘性通量向量
           （质量分量恒为0；动量分量 G[...,i,1+j]=tau_ij；能量分量含粘性功+热传导）
    """
    mu_total = mu + mu_t
    mu_total = mu_total * np.ones(Q.shape[:-1]) if np.isscalar(mu_total) else mu_total

    S = 0.5 * (grad_vel + np.swapaxes(grad_vel, -1, -2))  # (...,3,3)
    div_u = grad_vel[..., 0, 0] + grad_vel[..., 1, 1] + grad_vel[..., 2, 2]
    lam = -2.0 / 3.0 * mu_total

    eye3 = np.eye(3)
    tau = 2.0 * mu_total[..., None, None] * S + lam[..., None, None] * div_u[..., None, None] * eye3  # (...,3,3)

    cp = GAMMA * R_AIR / (GAMMA - 1.0)
    k_cond = mu * cp / Pr + mu_t * cp / Pr_t
    q = -k_cond * grad_T if np.isscalar(k_cond) else -k_cond[..., None] * grad_T  # (...,3)

    vel = Q[..., 1:4]  # (...,3)
    work = np.einsum("...i,...ij->...j", vel, tau)  # work[...,j] = sum_i u_i*tau_ij

    shape = Q.shape[:-1]
    G = np.zeros(shape + (3, 5))
    G[..., :, 1:4] = np.swapaxes(tau, -1, -2)  # G[...,i,1+j] = tau[...,j,i] = tau[...,i,j] (对称)
    G[..., :, 4] = work + q
    return G


def compute_viscous_residual_fr(U: np.ndarray, mesh, ops, mu: float, Pr: float,
                                 mu_t_field=None, Pr_t: float = 0.9,
                                 boundary_ghost_provider=None) -> np.ndarray:
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
            （例如 WALL 用速度镜像取反构造 Q_avg 速度=0，真正的无滑移），
            只有梯度（gv_n/gT_n）仍取内部值镜像（标准 BR1/LDG 简化处理：
            梯度本身没有独立的边界"真值"可用，用内部外插值是常见做法，
            边界约束通过状态跳跃在通量里体现，不需要也没有单独的梯度
            幽灵值）。

    Returns:
        residual: (n_cells, n_sps, 5)
    """
    from autoflowcfd.core.fr_residual.inviscid import conserved_to_primitive, DefaultGhostProvider

    ghost_provider = boundary_ghost_provider if boundary_ghost_provider is not None else DefaultGhostProvider()

    n_cells = mesh.n_cells
    n_sps = mesh.n_sps_per_cell
    n1d = mesh.n_points_1d

    mu_t_field = np.zeros((n_cells, n_sps)) if mu_t_field is None else mu_t_field

    Q = conserved_to_primitive(U[..., :5])  # (n_cells,n_sps,5)
    T = compute_temperature(Q)  # (n_cells,n_sps)

    grad_Q = compute_physical_gradient(Q, mesh, ops)  # (n_cells,n_sps,5,3)
    grad_vel = grad_Q[:, :, 1:4, :]  # (n_cells,n_sps,3,3): grad_vel[...,i,j]=d(u_i)/dx_j
    grad_T = compute_physical_gradient(T[:, :, None], mesh, ops)[:, :, 0, :]  # (n_cells,n_sps,3)

    det_jacs = mesh.jacobians["det_jacs"].reshape(n_cells, n_sps)
    inv_jacs = mesh.jacobians["inv_jacs"].reshape(n_cells, n_sps, 3, 3)
    adj_j = det_jacs[..., None, None] * inv_jacs

    # 体积项性能优化：与 fr_residual_inviscid.py 的同类改动理由/验证方式
    # 完全一致（py-spy 对生产网格的采样证实 `viscous_physical_flux`/
    # `einsum` 是界面项 numba 化之后新暴露出的主导耗时）——
    # `viscous_physical_flux_batch` 复用已逐位验证过的 numba 逐点 kernel，
    # `np.matmul`/`contract_shared_operator_2axis` 与原 einsum 公式严格
    # 等价，只是换一条计算路径，见 fr_volume_contract.py 模块文档。
    Q_flat = np.ascontiguousarray(Q.reshape(-1, 5))
    grad_vel_flat = np.ascontiguousarray(grad_vel.reshape(-1, 3, 3))
    grad_T_flat = np.ascontiguousarray(grad_T.reshape(-1, 3))
    mu_t_flat = np.ascontiguousarray(mu_t_field.reshape(-1))
    G_phys = viscous_physical_flux_batch(
        Q_flat, grad_vel_flat, grad_T_flat, mu, Pr, mu_t_flat, Pr_t
    ).reshape(n_cells, n_sps, 3, 5)

    G_tilde = np.matmul(adj_j, G_phys)  # (n_cells,n_sps,3,5)
    # 四面体/棱柱专用坍缩坐标微分矩阵，理由同 fr_residual_inviscid.py 的
    # 同类改动——见 FROperators.D_3d_tet/D_3d_prism 文档。
    n_prism = mesh.n_prism_cells
    div_comp = np.zeros((n_cells, n_sps, 5))
    if n_prism > 0:
        div_comp[:n_prism] = contract_shared_operator_2axis(ops.D_3d_prism, G_tilde[:n_prism])
    if n_cells > n_prism:
        div_comp[n_prism:] = contract_shared_operator_2axis(ops.D_3d_tet, G_tilde[n_prism:])
    residual = div_comp / det_jacs[..., None]  # 注意：粘性项是 +div(G)（见模块文档的符号约定）

    # --- 界面项：numba 逐点标量 kernel（性能优化，替代原纯 Python
    # `for f in range(fc.n_faces)` 逐面循环，理由/验证方式与
    # fr_residual_inviscid.py 的同类改动完全一致，见
    # fr_viscous_flux_kernel.py 模块文档、
    # tests/unit/test_fr_viscous_flux_kernel_crosscheck.py 的新旧实现
    # 逐位对比验证）。
    from autoflowcfd.core.fr_operators.face_kernels import get_flat_face_geometry
    from autoflowcfd.core.fr_residual.inviscid_kernel import compute_boundary_ghost_states
    
    flat = get_flat_face_geometry(mesh, ops)
    Q_ghost = compute_boundary_ghost_states(flat, Q, adj_j, ghost_provider)

    if n_sps == 1:
        # P0 专用路径：使用简化 kernel（消除 SP 循环，外插简化为标量乘）
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
            adj_j, det_jacs, mu, Pr, Pr_t,
            flat.owner_cell, flat.neighbor_cell, flat.is_boundary,
            flat.owner_axis, flat.owner_side, flat.neighbor_axis, flat.neighbor_side,
            flat.owner_is_primary, flat.neighbor_is_primary,
            flat.neighbor_src0_cell, flat.neighbor_src0_mat,
            flat.neighbor_src1_idx, flat.neighbor_src1_cell, flat.neighbor_src1_mat,
            flat.owner_src0_cell, flat.owner_src0_mat,
            flat.owner_src1_idx, flat.owner_src1_cell, flat.owner_src1_mat,
            flat.boundary_extrap, flat.g_left, flat.g_right, Q_ghost,
            flat.dist_fp_of_sp, flat.dist_axis_coord_of_sp,
            n_prism, n_threads,
        )
    else:
        # P≥1 通用路径：图着色或 per-thread buffer
        from autoflowcfd.core.fr_residual.viscous_flux_kernel import (
            compute_viscous_interface_correction_kernel,
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
                    adj_j, det_jacs, mu, Pr, Pr_t,
                    flat.owner_cell, flat.neighbor_cell, flat.is_boundary,
                    flat.owner_axis, flat.owner_side, flat.neighbor_axis, flat.neighbor_side,
                    flat.owner_is_primary, flat.neighbor_is_primary,
                    flat.neighbor_src0_cell, flat.neighbor_src0_mat,
                    flat.neighbor_src1_idx, flat.neighbor_src1_cell, flat.neighbor_src1_mat,
                    flat.owner_src0_cell, flat.owner_src0_mat,
                    flat.owner_src1_idx, flat.owner_src1_cell, flat.owner_src1_mat,
                    flat.boundary_extrap, flat.g_left, flat.g_right, Q_ghost,
                    flat.dist_fp_of_sp, flat.dist_axis_coord_of_sp,
                    n_prism, face_indices, correction,
                )
        else:
            # 回退到 per-thread buffer 方案（小网格 + 低线程数可能更快）
            import numba
            n_threads = numba.get_num_threads()
            correction = compute_viscous_interface_correction_kernel(
                Q, grad_vel, grad_T, mu_t_field,
                adj_j, det_jacs, mu, Pr, Pr_t,
                flat.owner_cell, flat.neighbor_cell, flat.is_boundary,
                flat.owner_axis, flat.owner_side, flat.neighbor_axis, flat.neighbor_side,
                flat.owner_is_primary, flat.neighbor_is_primary,
                flat.neighbor_src0_cell, flat.neighbor_src0_mat,
                flat.neighbor_src1_idx, flat.neighbor_src1_cell, flat.neighbor_src1_mat,
                flat.owner_src0_cell, flat.owner_src0_mat,
                flat.owner_src1_idx, flat.owner_src1_cell, flat.owner_src1_mat,
                flat.boundary_extrap, flat.g_left, flat.g_right, Q_ghost,
                flat.dist_fp_of_sp, flat.dist_axis_coord_of_sp,
                n_prism, n_threads,
            )
    residual = residual + correction

    # 机制3（症状检测，见 fr_troubled_cell.py 模块文档）：直接对算出的
    # 最终粘性残差做 (cell,SP,变量) 粒度的量级异常检测并清零，取代按
    # 整个单元降阶的旧机制1/2 门控，理由同 fr_residual_inviscid.py 的
    # 同名改动。
    return suppress_residual_outliers(residual, U[..., :5])
