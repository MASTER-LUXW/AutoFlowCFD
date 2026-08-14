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

import numpy as np

from autoflowcfd.core.fr_gradients import compute_physical_gradient
from autoflowcfd.core.fr_troubled_cell import suppress_residual_outliers

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


def _distribute_from_face(fp_data: np.ndarray, n1d: int, axis: int, g_prime: np.ndarray) -> np.ndarray:
    trailing_shape = fp_data.shape[1:]
    fp_grid = fp_data.reshape((n1d, n1d) + trailing_shape)
    expanded = np.tensordot(g_prime, fp_grid, axes=0)
    result = np.moveaxis(expanded, 0, axis)
    return result.reshape((n1d**3,) + trailing_shape)


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
        boundary_ghost_provider: 边界面统一取内部值外插（零梯度延拓），
            边界层剪应力由 WMLES 壁面应力模型单独提供，见函数早期版本
            文档说明（未删除以保持接口一致，当前实现不使用该参数，
            保留位置供未来扩展绝热壁面镜像梯度时使用）。

    Returns:
        residual: (n_cells, n_sps, 5)
    """
    from autoflowcfd.core.fr_residual_inviscid import conserved_to_primitive

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

    G_phys = viscous_physical_flux(Q, grad_vel, grad_T, mu, Pr, mu_t=mu_t_field, Pr_t=Pr_t)  # (n_cells,n_sps,3,5)

    G_tilde = np.einsum("csmi,csiv->csmv", adj_j, G_phys)  # (n_cells,n_sps,3,5)
    # 四面体/棱柱专用坍缩坐标微分矩阵，理由同 fr_residual_inviscid.py 的
    # 同类改动——见 FROperators.D_3d_tet/D_3d_prism 文档。
    n_prism = mesh.n_prism_cells
    div_comp = np.zeros((n_cells, n_sps, 5))
    if n_prism > 0:
        div_comp[:n_prism] = np.einsum("sjm,cjmv->csv", ops.D_3d_prism, G_tilde[:n_prism])
    if n_cells > n_prism:
        div_comp[n_prism:] = np.einsum("sjm,cjmv->csv", ops.D_3d_tet, G_tilde[n_prism:])
    residual = div_comp / det_jacs[..., None]  # 注意：粘性项是 +div(G)（见模块文档的符号约定）

    fc = mesh.face_connectivity
    ffp_list = mesh.face_flux_points
    correction = np.zeros_like(residual)

    def extrap_to_face(cell: int, field: np.ndarray, axis: int, side: float) -> np.ndarray:
        """四面体/棱柱专用坍缩坐标模态基外插（不是朴素 1D 张量积
        extrapolate_to_face），理由同 fr_residual_inviscid.py 的同名
        改动——见 FROperators.boundary_extrap_tet/boundary_extrap_prism
        文档：朴素外插在坍缩坐标退化边附近算出的等效法向方向与真实
        几何法向偏差可达近 30°，被同处偏小的 Jacobian 放大到灾难量级。
        """
        E = ops.boundary_extrap_prism[(axis, side)] if cell < n_prism else ops.boundary_extrap_tet[(axis, side)]
        trailing = field.shape[1:]
        flat = E @ field.reshape(field.shape[0], -1)
        return flat.reshape((E.shape[0],) + trailing)

    def extrapolate_state_and_grad(cell, axis, side):
        Q_fp = extrap_to_face(cell, Q[cell], axis, side)
        gradvel_fp = extrap_to_face(cell, grad_vel[cell], axis, side)
        gradT_fp = extrap_to_face(cell, grad_T[cell], axis, side)
        adj_row_fp = extrap_to_face(cell, adj_j[cell][:, axis, :], axis, side)
        mut_fp = extrap_to_face(cell, mu_t_field[cell][:, None], axis, side)[:, 0]
        return Q_fp, gradvel_fp, gradT_fp, adj_row_fp, mut_fp

    def _apply_interp(interp_matrix: np.ndarray, field: np.ndarray) -> np.ndarray:
        """interp_matrix: (n_fp, n_sps); field: (n_sps, ...) -> (n_fp, ...)。"""
        trailing = field.shape[1:]
        flat = interp_matrix @ field.reshape(field.shape[0], -1)
        return flat.reshape((interp_matrix.shape[0],) + trailing)

    def _apply_sources(sources, field_of_cell) -> np.ndarray:
        """sources: List[(cell_id, (n_fp,n_sps)矩阵)]；field_of_cell(cell_id)
        返回该 cell 的 (n_sps, ...) 场。按物理位置精确来源求和组装
        （见 fr/face_flux_points.py::FaceFluxPointGeometry 文档，1~2 个
        真实相邻单元的情形统一处理）。"""
        total = None
        for cell_id, mat in sources:
            contrib = _apply_interp(mat, field_of_cell(cell_id))
            total = contrib if total is None else total + contrib
        return total

    for f in range(fc.n_faces):
        ffp = ffp_list[f]
        owner_cell = int(fc.owner_cell[f])
        owner_axis, owner_side = ffp.owner_axis, ffp.owner_side

        # 棱柱四边形侧面被拆分成 2 个三角形子面记录的情形（见
        # fr/face_flux_points.py 文档）：owner/neighbor 各自的自身外插+
        # 校正投影只能由分组内的一条 primary 记录触发一次。
        if ffp.owner_is_primary:
            Q_o, gv_o, gT_o, adjrow_o, mut_o = extrapolate_state_and_grad(owner_cell, owner_axis, owner_side)

            if fc.is_boundary[f]:
                Q_n, gv_n, gT_n, mut_n = Q_o, gv_o, gT_o, mut_o
            else:
                # neighbor（1~2 个真实相邻单元）的场在 owner FP 精确物理
                # 位置处的取值（精确点位插值）
                Q_n = _apply_sources(ffp.neighbor_sources, lambda c: Q[c])
                gv_n = _apply_sources(ffp.neighbor_sources, lambda c: grad_vel[c])
                gT_n = _apply_sources(ffp.neighbor_sources, lambda c: grad_T[c])
                mut_n = _apply_sources(ffp.neighbor_sources, lambda c: mu_t_field[c][:, None])[:, 0]

            # BR1: 界面态取相邻两侧（同一物理位置处）取值的算术平均
            Q_avg = 0.5 * (Q_o + Q_n)
            gv_avg = 0.5 * (gv_o + gv_n)
            gT_avg = 0.5 * (gT_o + gT_n)
            mut_avg = 0.5 * (mut_o + mut_n)

            G_common = viscous_physical_flux(Q_avg, gv_avg, gT_avg, mu, Pr, mu_t=mut_avg, Pr_t=Pr_t)  # (n_fp,3,5)
            G_tilde_common_owner = np.einsum("fi,fiv->fv", adjrow_o, G_common)

            G_phys_owner_fp = viscous_physical_flux(Q_o, gv_o, gT_o, mu, Pr, mu_t=mut_o, Pr_t=Pr_t)
            G_tilde_own_boundary = np.einsum("fi,fiv->fv", adjrow_o, G_phys_owner_fp)

            jump_owner = G_tilde_common_owner - G_tilde_own_boundary
            g_prime_owner = ops.g_left if owner_side < 0 else ops.g_right
            contrib_owner = _distribute_from_face(jump_owner, n1d, owner_axis, g_prime_owner)
            correction[owner_cell] += contrib_owner / det_jacs[owner_cell][:, None]

        if not fc.is_boundary[f] and ffp.neighbor_is_primary:
            # neighbor 侧：用 neighbor 自己原生 FP 网格 + owner 的场在这些
            # 精确物理位置处的取值，独立重做一次 BR1 平均与校正投影
            # （原生 FP 位置一般不同于 owner 的 FP 位置，不能直接复用上面
            # 算好的 Q_avg 等量）。
            neighbor_cell = int(fc.neighbor_cell[f])
            n_axis, n_side = ffp.neighbor_axis, ffp.neighbor_side
            Q_n_native, gv_n_native, gT_n_native, adjrow_n_native, mut_n_native = extrapolate_state_and_grad(
                neighbor_cell, n_axis, n_side
            )

            Q_o_at_n = _apply_sources(ffp.owner_sources, lambda c: Q[c])
            gv_o_at_n = _apply_sources(ffp.owner_sources, lambda c: grad_vel[c])
            gT_o_at_n = _apply_sources(ffp.owner_sources, lambda c: grad_T[c])
            mut_o_at_n = _apply_sources(ffp.owner_sources, lambda c: mu_t_field[c][:, None])[:, 0]

            Q_avg_n = 0.5 * (Q_n_native + Q_o_at_n)
            gv_avg_n = 0.5 * (gv_n_native + gv_o_at_n)
            gT_avg_n = 0.5 * (gT_n_native + gT_o_at_n)
            mut_avg_n = 0.5 * (mut_n_native + mut_o_at_n)

            G_common_native = viscous_physical_flux(
                Q_avg_n, gv_avg_n, gT_avg_n, mu, Pr, mu_t=mut_avg_n, Pr_t=Pr_t
            )
            G_tilde_common_neighbor = np.einsum("fi,fiv->fv", adjrow_n_native, G_common_native)

            G_phys_neighbor_fp = viscous_physical_flux(
                Q_n_native, gv_n_native, gT_n_native, mu, Pr, mu_t=mut_n_native, Pr_t=Pr_t
            )
            G_tilde_own_boundary_neighbor = np.einsum("fi,fiv->fv", adjrow_n_native, G_phys_neighbor_fp)

            jump_neighbor = G_tilde_common_neighbor - G_tilde_own_boundary_neighbor
            g_prime_neighbor = ops.g_left if n_side < 0 else ops.g_right
            contrib_neighbor = _distribute_from_face(jump_neighbor, n1d, n_axis, g_prime_neighbor)
            correction[neighbor_cell] += contrib_neighbor / det_jacs[neighbor_cell][:, None]

    residual += correction
    # 机制3（症状检测，见 fr_troubled_cell.py 模块文档）：直接对算出的
    # 最终粘性残差做 (cell,SP,变量) 粒度的量级异常检测并清零，取代按
    # 整个单元降阶的旧机制1/2 门控，理由同 fr_residual_inviscid.py 的
    # 同名改动。
    return suppress_residual_outliers(residual, U[..., :5])
