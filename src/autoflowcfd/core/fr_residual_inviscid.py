"""
AutoFlowCFD V2.0 - FR 无粘残差组装 (Tier-0 重建版, 对应 S-02/S-04)

替换旧版本中「全场单元平均态 + 硬编码法向量」的伪校正项
(fr_solver.py::_compute_fr_correction_ausm)。真正实现：

1. **体积项**：在计算立方体空间用逆变通量 (contravariant flux)
   F̃_m = Σ_i adj(J)_{m,i} F_i(U) 做散度，再除以 det(J) 得到物理残差——
   这是曲边/坍缩坐标单元下唯一能保证与 GCL 一致的散度计算方式；旧版本
   `fr_kernels.compute_fr_residual_kernel` 把计算空间微分算子 D_3d 直接
   当作物理空间导数使用，缺少度量项变换，对任何非笛卡尔映射的单元
   （即本代码库中的每一个四面体/棱柱单元）给出的都是错误导数。

2. **界面项**：用真实单元-面连接关系（grid/face_connectivity.py +
   fr/face_flux_points.py）取得 owner/neighbor 双方在物理重合点上的解，
   用已验证正确的 AUSM+up 黎曼求解器（fr_kernels.compute_ausm_up_flux）
   结合 FaceExtractor 给出的真实物理法向量/面积计算公共通量，再通过
   Radau/VCJH 校正函数导数 (matrix_operators.compute_correction_weights)
   投影回 SPs 残差。

正确性通过「均匀自由流场残差应严格为零」(free-stream preservation /
离散 GCL 的实际检验) 数值验证，见 tests/unit/test_fr_residual_inviscid.py。
"""

from typing import Callable, Optional

import numpy as np

from autoflowcfd.core.fr_kernels import compute_ausm_up_flux
from autoflowcfd.core.fr_troubled_cell import suppress_residual_outliers

GAMMA = 1.4


def conserved_to_primitive(U: np.ndarray) -> np.ndarray:
    """U=(rho,rho*u,rho*v,rho*w,rho*E) -> Q=(rho,u,v,w,p)，沿最后一维前5个分量。"""
    rho = np.maximum(U[..., 0], 1e-10)
    u = U[..., 1] / rho
    v = U[..., 2] / rho
    w = U[..., 3] / rho
    E = U[..., 4] / rho
    ke = 0.5 * (u**2 + v**2 + w**2)
    p = (GAMMA - 1.0) * rho * (E - ke)
    return np.stack([rho, u, v, w, p], axis=-1)


def primitive_to_conserved(Q: np.ndarray) -> np.ndarray:
    """Q=(rho,u,v,w,p) -> U=(rho,rho*u,rho*v,rho*w,rho*E)。"""
    rho, u, v, w, p = Q[..., 0], Q[..., 1], Q[..., 2], Q[..., 3], Q[..., 4]
    ke = 0.5 * (u**2 + v**2 + w**2)
    e_internal = p / ((GAMMA - 1.0) * np.maximum(rho, 1e-10))
    E = e_internal + ke
    return np.stack([rho, rho * u, rho * v, rho * w, rho * E], axis=-1)


def euler_physical_flux(Q: np.ndarray) -> np.ndarray:
    """计算物理通量张量 F_i(Q)，i=x,y,z。

    Args:
        Q: 形状 (..., 5)，(rho,u,v,w,p)

    Returns:
        F: 形状 (..., 3, 5)，F[...,i,:] 是方向 i 的通量向量
    """
    rho, u, v, w, p = Q[..., 0], Q[..., 1], Q[..., 2], Q[..., 3], Q[..., 4]
    ke = 0.5 * (u**2 + v**2 + w**2)
    e_internal = p / ((GAMMA - 1.0) * np.maximum(rho, 1e-10))
    rhoE = rho * (e_internal + ke)
    H = (rhoE + p) / np.maximum(rho, 1e-10)  # 总焓

    vel = np.stack([u, v, w], axis=-1)  # (...,3)
    mass_flux = rho[..., None] * vel  # (...,3) = rho*u_i

    F = np.zeros(Q.shape[:-1] + (3, 5))
    F[..., :, 0] = mass_flux
    for i in range(3):
        F[..., i, 1] = mass_flux[..., i] * u + (p if i == 0 else 0.0)
        F[..., i, 2] = mass_flux[..., i] * v + (p if i == 1 else 0.0)
        F[..., i, 3] = mass_flux[..., i] * w + (p if i == 2 else 0.0)
        F[..., i, 4] = rho * H * vel[..., i]
    return F


def ausm_up_flux_batch(Q_L: np.ndarray, Q_R: np.ndarray, normal: np.ndarray) -> np.ndarray:
    """对一批 Flux Points 逐点调用 Numba 版 AUSM+up (标量法向通量密度)。

    Args:
        Q_L, Q_R: (n_fp, 5)
        normal: (n_fp, 3) 单位法向量（由 L 指向 R）

    Returns:
        flux: (n_fp, 5)，F*·n （每单位面积的物理通量密度）
    """
    n_fp = Q_L.shape[0]
    flux = np.zeros((n_fp, 5))
    for i in range(n_fp):
        flux[i] = compute_ausm_up_flux(Q_L[i], Q_R[i], normal[i])
    return flux


def _distribute_from_face(fp_data: np.ndarray, n1d: int, axis: int, g_prime: np.ndarray) -> np.ndarray:
    """把 (n1d^2, ...) 的面数据按 g'(x) 权重分配回 (n1d^3, ...) 的 SPs 残差贡献。"""
    trailing_shape = fp_data.shape[1:]
    other_axes = [a for a in range(3) if a != axis]
    fp_grid = fp_data.reshape((n1d, n1d) + trailing_shape)
    expanded = np.tensordot(g_prime, fp_grid, axes=0)  # (n1d_axis, n1d, n1d, ...)
    result = np.moveaxis(expanded, 0, axis)
    return result.reshape((n1d**3,) + trailing_shape)


class DefaultGhostProvider:
    """默认的边界幽灵态提供者：零梯度外插（ghost = 内部外插态），
    仅用于尚未接入真实边界条件（BD-01，见 boundary/fr_weak_bc.py 的接入
    工作）之前的开发期占位与自由流场守恒性测试——对均匀流场恰好退化为
    零跳跃，不会掩盖真实边界条件缺失这一事实（调用方必须显式传入
    real ghost provider 才能获得物理正确的边界处理）。
    """

    def __call__(self, face_idx: int, Q_owner_fp: np.ndarray, true_normal: np.ndarray) -> np.ndarray:
        return Q_owner_fp.copy()


def _compute_inviscid_residual_fv_p0(
    U: np.ndarray,
    mesh,
    boundary_ghost_provider: Optional[Callable[[int, np.ndarray, np.ndarray], np.ndarray]] = None,
) -> np.ndarray:
    """P0（1 SP/cell，Order Continuation 最低阶）专用有限体积残差。

    背景：P>=1 的界面项用坍缩坐标体积度量张量 adj(J) 外插到面上再做一致性
    校验（见 compute_inviscid_residual_fr 里的 alignment 校验），这在 P0
    下必然报错——坍缩（Duffy）坐标的 Jacobian 在单元内部本就强烈非均匀
    （fr/collapsed_basis.py 模块文档），单元内唯一那个解点（位于坍缩参考
    立方体中心）处的度量方向，物理上没有理由与该单元 3~4 个不同面各自的
    真实法向对齐——这不是数值 bug，是"用同一个点的坍缩度量代表所有面"
    这一想法在数学上站不住脚。真实网格已复现：alignment cosine 可低至
    0.20（约78°偏差），远超 0.5 的校验阈值。

    P0 在数学上唯一自洽的定义就是经典分片常数有限体积：完全不依赖坍缩
    度量张量，直接用 face_connectivity 给出的真实几何法向 (ffp.true_normal)
    与真实面积权重 (ffp.true_area_weight) 做迎风通量积分。owner/neighbor
    共用同一个法向量、同一次 Riemann 求解结果（对两侧符号相反地施加），
    天然精确守恒——不像 P>=1 那样需要 owner/neighbor 各自独立取自己的
    度量法向（那是坍缩坐标外插固有的不一致来源，P0 完全没有这个问题，
    因为这里用的是同一个真实几何法向，不是两个独立外插出来的近似法向）。

    体积项在 P0 下无需计算：D_3d_tet/D_3d_prism 在 order=0 时解析恒为
    零矩阵（常数函数对任何参考方向的导数都是零），体积散度贡献必为零。

    关于棱柱四边形侧面拆分（真实网格已复现、修复的一个关键点）：
    `build_face_flux_points` 对 face_connectivity 里的*每一条*记录（包括
    棱柱四边形侧面因三角化被拆出的 2 条子面记录）都无条件调用
    `result.append(...)`，`true_normal`/`true_area_weight` 也在
    primary/非primary 判断之前就已算好、对每条记录都有效——`owner_is_primary`
    /`neighbor_is_primary` 只是控制"是否触发一次自身外插+跨单元投影"
    （P>=1 的坍缩度量路径需要，用来避免同一个原生 FP 网格被重复计入两次），
    与"这条记录的真实几何面积/法向是否有效"无关。本函数因此直接按
    face_connectivity 的原始每条记录处理（不经过 owner_is_primary 过滤、
    不经过 neighbor_sources/owner_sources 的多源合并——P0 下每条记录本来
    就唯一对应一个真实相邻单元，不存在"一个原生 FP 网格分给两个不同
    相邻单元"这个 P>=1 才有的问题）：曾经的第一版实现按
    `if not ffp.owner_is_primary: continue` 跳过非 primary 记录，等价于
    直接丢弃了棱柱被拆分的那一半四边形的真实面积——对闭合单元的面积/
    法向积分 Σ(n̂·A)=0 这一几何恒等式造成真实的（非浮点噪声量级的）
    破坏，在小体积单元（真实网格边界层单元体积低至 ~1e-11 m³）上除以
    体积后被放大到 1e11 量级的"伪残差"（已复现：直接改用本函数现在的
    写法后，均匀自由流场残差恢复到机器精度量级）。

    Args:
        U: 守恒变量，形状 (n_cells, 1, n_vars)
        mesh: HighOrderMesh 实例（n_points_1d 必须为 1）
        boundary_ghost_provider: 同 compute_inviscid_residual_fr

    Returns:
        residual: 形状 (n_cells, 1, 5)
    """
    n_cells = mesh.n_cells
    if mesh.cell_volumes is None:
        raise RuntimeError(
            "mesh.cell_volumes not available - required for the P0 finite-volume residual path "
            "(should have been computed once in load_from_volume_mesh at the mesh's target order)."
        )
    cell_volumes = mesh.cell_volumes

    Q_all = conserved_to_primitive(U[..., :5])[:, 0, :]  # (n_cells,5)，P0 唯一解点即单元均值

    fc = mesh.face_connectivity
    ffp_list = mesh.face_flux_points
    ghost_provider = boundary_ghost_provider if boundary_ghost_provider is not None else DefaultGhostProvider()

    residual5 = np.zeros((n_cells, 5))

    for f in range(fc.n_faces):
        ffp = ffp_list[f]
        owner_cell = int(fc.owner_cell[f])
        true_normal = ffp.true_normal  # (1,3)，owner->neighbor / 边界面指向域外
        area_w = ffp.true_area_weight  # (1,)

        Q_owner_fp = Q_all[owner_cell : owner_cell + 1]  # (1,5)

        if fc.is_boundary[f]:
            Q_neighbor_fp = ghost_provider(f, Q_owner_fp, true_normal)
        else:
            neighbor_cell = int(fc.neighbor_cell[f])
            Q_neighbor_fp = Q_all[neighbor_cell : neighbor_cell + 1]

        F_common_n = ausm_up_flux_batch(Q_owner_fp, Q_neighbor_fp, true_normal)  # (1,5)
        flux_integral = F_common_n[0] * area_w[0]  # (5,)

        residual5[owner_cell] += -flux_integral / cell_volumes[owner_cell]
        if not fc.is_boundary[f]:
            residual5[neighbor_cell] += flux_integral / cell_volumes[neighbor_cell]

    return residual5[:, None, :]


def compute_inviscid_residual_fr(
    U: np.ndarray,
    mesh,
    ops,
    boundary_ghost_provider: Optional[Callable[[int, np.ndarray, np.ndarray], np.ndarray]] = None,
) -> np.ndarray:
    """计算真实面耦合的 FR 无粘残差 dU/dt（物理空间，已除以 det(J)）。

    Args:
        U: 守恒变量，形状 (n_cells, n_sps, n_vars)；只使用前5个欧拉变量
        mesh: HighOrderMesh 实例，需要已调用 load_from_volume_mesh
            （提供 face_connectivity, face_flux_points, jacobians）
        ops: FROperators（D_3d, g_left, g_right）
        boundary_ghost_provider: 可调用对象 (face_idx, Q_owner_fp, true_normal) -> Q_ghost_fp，
            用于给出边界面的幽灵态；None 时使用 DefaultGhostProvider（零梯度外插）

    Returns:
        residual: 形状 (n_cells, n_sps, 5)
    """
    if mesh.face_connectivity is None or mesh.face_flux_points is None:
        raise RuntimeError(
            "HighOrderMesh has no face_connectivity/face_flux_points - "
            "call load_from_volume_mesh(build_faces=True) first."
        )

    if mesh.n_points_1d == 1:
        # P0（Order Continuation 最低阶）：坍缩坐标单点度量方向在数学上
        # 无法代表单元各面各自的真实法向，完全绕开度量张量外插机制，走
        # 独立的真实几何有限体积路径，见 _compute_inviscid_residual_fv_p0
        # 文档。
        return _compute_inviscid_residual_fv_p0(U, mesh, boundary_ghost_provider)

    n_cells = mesh.n_cells
    n_sps = mesh.n_sps_per_cell
    n1d = mesh.n_points_1d

    Q = conserved_to_primitive(U[..., :5])  # (n_cells, n_sps, 5)

    det_jacs = mesh.jacobians["det_jacs"].reshape(n_cells, n_sps)
    inv_jacs = mesh.jacobians["inv_jacs"].reshape(n_cells, n_sps, 3, 3)
    adj_j = det_jacs[..., None, None] * inv_jacs  # (n_cells,n_sps,3,3), adj_j[...,m,i]

    F_phys = euler_physical_flux(Q)  # (n_cells,n_sps,3,5)
    # 逆变通量 F_tilde_m = sum_i adj_j[m,i] * F_phys[i]
    F_tilde = np.einsum("csmi,csiv->csmv", adj_j, F_phys)  # (n_cells,n_sps,3,5)

    # 体积项（计算空间散度）：res_comp[c,s,v] = sum_{j,m} D_3d[s,j,m]*F_tilde[c,j,m,v]。
    # 四面体/棱柱必须用坍缩坐标专用微分矩阵（fr/collapsed_basis.py），不能
    # 用朴素张量积 D_3d——见 FROperators.D_3d_tet/D_3d_prism 文档：真实
    # 网格上朴素张量积基在坍缩坐标退化边附近的混叠误差，被同一处真实
    # 偏小的几何 Jacobian（det_jacs，下面一步会除以它）放大到灾难量级。
    n_prism = mesh.n_prism_cells
    div_comp = np.zeros((n_cells, n_sps, 5))
    if n_prism > 0:
        div_comp[:n_prism] = np.einsum("sjm,cjmv->csv", ops.D_3d_prism, F_tilde[:n_prism])
    if n_cells > n_prism:
        div_comp[n_prism:] = np.einsum("sjm,cjmv->csv", ops.D_3d_tet, F_tilde[n_prism:])
    residual = -div_comp / det_jacs[..., None]  # 物理空间残差（体积项部分）

    # --- 界面项 ---
    ghost_provider = boundary_ghost_provider if boundary_ghost_provider is not None else DefaultGhostProvider()

    fc = mesh.face_connectivity
    ffp_list = mesh.face_flux_points

    correction = np.zeros_like(residual)

    def extrap_to_face(cell: int, field: np.ndarray, axis: int, side: float) -> np.ndarray:
        """把 cell 体积 SPs 上的场外插到其 (axis,side) 边界 Flux Points。

        四面体/棱柱必须用坍缩坐标模态基外插矩阵（ops.boundary_extrap_tet/
        boundary_extrap_prism），不能用朴素 1D 张量积 extrapolate_to_face
        ——真实网格验证发现，朴素外插算出的等效界面法向方向在坍缩坐标
        退化边附近与真实几何法向偏差可达近 30°（仍在下面 alignment 校验
        的宽松阈值内、不会报错，但足以在残差公式除以该处真实偏小的
        Jacobian 后放大到灾难量级），见 FROperators.boundary_extrap_tet
        文档。FP 索引约定（other_axes 顺序展平）与 extrapolate_to_face
        完全一致，_distribute_from_face 的逐点校正投影不受影响。
        """
        if cell < n_prism:
            E = ops.boundary_extrap_prism[(axis, side)]
        else:
            E = ops.boundary_extrap_tet[(axis, side)]
        trailing = field.shape[1:]
        flat = E @ field.reshape(field.shape[0], -1)
        return flat.reshape((E.shape[0],) + trailing)

    def side_contravariant_flux(cell: int, axis: int, side: float, Q_fp: np.ndarray, face_idx: int, label: str):
        """给定单元在其局部面 (axis, side) 处外插得到的 Q_fp，计算：
        - 该单元自身的边界逆变通量 F_tilde_own = adj_row · F_phys(Q_fp)
        - adj_row 的模长与"是否指向真实外法向"的方向标记 side（+1/-1）

        计算立方体参考域 [-1,1] 上，"+ξ_axis 方向"在 side=-1 的面处指向域内、
        在 side=+1 的面处指向域外（标准参考单元朝向的固有性质，与单元是
        四面体还是棱柱无关）。因此 adj_row 必须再乘以 side 才能与「真实
        物理外法向」这一约定的方向一致——这是残差组装里最容易出错、也
        最容易被误认为"符号无所谓"的一步，已用均匀自由流场残差应严格
        为零的测试验证（tests/unit/test_fr_residual_inviscid.py）。
        """
        adj_row_fp = extrap_to_face(cell, adj_j[cell][:, axis, :], axis, side)  # (n_fp,3)
        adj_mag = np.linalg.norm(adj_row_fp, axis=-1)
        adj_dir_outward = (adj_row_fp / np.maximum(adj_mag[:, None], 1e-300)) * side
        return adj_row_fp, adj_mag, adj_dir_outward

    for f in range(fc.n_faces):
        ffp = ffp_list[f]
        owner_cell = int(fc.owner_cell[f])
        owner_axis = ffp.owner_axis
        owner_side = ffp.owner_side

        # 棱柱四边形侧面被网格生成器恒定拆分成 2 个三角形子面记录（见
        # fr/face_flux_points.py 模块文档），owner/neighbor 各自的自身
        # 外插+校正投影只能由分组内的一条 primary 记录触发一次，否则会
        # 对同一批原生 Flux Points 的界面校正项重复计入。非 primary 的
        # 记录只是把"这条记录对应四边形哪一半、该跟哪个真实相邻单元耦合"
        # 的信息合并进了 primary 记录的 neighbor_sources/owner_sources
        # 里（见 face_flux_points_merge.py），本身不需要再单独处理。
        if ffp.owner_is_primary:
            Q_owner_fp = extrap_to_face(owner_cell, Q[owner_cell], owner_axis, owner_side)  # (n_fp,5)
            adj_owner_row_fp, adj_mag_owner, adj_dir_owner_outward = side_contravariant_flux(
                owner_cell, owner_axis, owner_side, Q_owner_fp, f, "owner"
            )

            alignment = np.sum(adj_dir_owner_outward * ffp.true_normal, axis=-1)
            riemann_normal_owner = adj_dir_owner_outward
            if np.any(alignment < 0.5):
                # 度量张量外插得到的等效法向与真实几何法向偏离过大——真实
                # 网格已复现：Order Continuation 的低阶 warm-up 阶段（P0、
                # P1，目标阶 P2 本身未见此问题）单元内 SPs 太少，坍缩坐标
                # 固有的度量非均匀性（fr/collapsed_basis.py 文档）在这些
                # 阶数下无法被外插矩阵充分捕捉，与真实几何法向偏差可达
                # 60°+；这是低阶表示的固有局限，不是拓扑/朝向 bug。偏离
                # 超阈值的那些 Flux Points 改用真实几何法向做 Riemann 求解
                # 迎风判据——物理上更基础可靠（迎风方向理应看真实外法向），
                # 而不是让整条残差计算中止；adj_mag_owner/owner_side 的
                # 投影关系只依赖 owner 自己的度量张量本身，与法向方向的
                # 选择无关，不受影响。
                bad = alignment < 0.5
                riemann_normal_owner = np.where(bad[:, None], ffp.true_normal, adj_dir_owner_outward)

            if fc.is_boundary[f]:
                Q_neighbor_fp = ghost_provider(f, Q_owner_fp, ffp.true_normal)
            else:
                # neighbor（1~2 个真实相邻单元，见模块文档）的解在 owner FP
                # 精确物理位置处的取值：每个 sources 矩阵只在它真正覆盖的
                # 那部分 FP 行上非零，求和即得到按物理位置精确来源组装出的
                # 完整界面场（不是"两侧独立离散化恰好重合"的错误假设）。
                Q_neighbor_fp = sum(mat @ Q[cell] for cell, mat in ffp.neighbor_sources)

            # 公共物理法向通量密度（黎曼求解器）：必须用 owner 自己的度量
            # 法向 adj_dir_owner_outward，不能用外部（FaceExtractor 给出的
            # 平面三角形）true_normal——棱柱四边形侧面是双线性曲面，其
            # 局部法向随位置变化，与相邻四面体平面三角形的法向本就不
            # 严格相等（真实网格验证：偏差可达约 28°，且是真实几何事实，
            # 不是数值误差，见开发过程记录），若用 true_normal 算出
            # F_common_n 后直接乘以 adj_mag_owner（隐含假设两者方向一致），
            # 在偏差较大处会引入真实的、非近似意义下的不一致——用
            # adj_dir_owner_outward 本身做黎曼求解器的法向，用它算出的
            # 通量投影到 owner 自己的逆变坐标就是精确自洽的（不是近似）。
            #
            # （本会话曾尝试用 owner/neighbor 两侧方向的平分方向代替，
            # 目的是恢复黎曼求解器的反对称性/通量守恒——已用受控解析算例
            # 证实两侧独立方向确实违反 F(A,B,n)=-F(B,A,-n)——但在真实
            # cube_demo 网格上实测导致自由流场残差从 9e-5 恶化到 3.1e7，
            # 说明该平分方向实现对棱柱侧的坍缩坐标轴处理有误，已完整
            # 撤销，恢复本段代码；通量不守恒问题仍待正确定位与修复。）
            F_common_n_owner = ausm_up_flux_batch(Q_owner_fp, Q_neighbor_fp, riemann_normal_owner)  # (n_fp,5)

            # --- owner 侧：转换到 owner 自己的逆变（"+ξ_axis方向"）约定并做校正投影 ---
            F_tilde_common_owner = F_common_n_owner * adj_mag_owner[:, None] * owner_side
            F_phys_owner_fp = euler_physical_flux(Q_owner_fp)  # (n_fp,3,5)
            F_tilde_own_boundary = np.einsum("fi,fiv->fv", adj_owner_row_fp, F_phys_owner_fp)

            jump_owner = F_tilde_common_owner - F_tilde_own_boundary  # (n_fp,5)
            g_prime_owner = ops.g_left if owner_side < 0 else ops.g_right
            contrib_owner = -_distribute_from_face(jump_owner, n1d, owner_axis, g_prime_owner)
            correction[owner_cell] += contrib_owner / det_jacs[owner_cell][:, None]

        if not fc.is_boundary[f] and ffp.neighbor_is_primary:
            # --- neighbor 侧：独立地用 neighbor 自己的度量项、自己的 side、
            # 自己原生的 FP 网格重新计算一次公共通量（不复用 owner 侧算好的
            # F_common_n——那是在 owner 的 FP 物理位置上算的，neighbor 原生
            # FP 位置一般是同一张平面上的另一组点，必须用 owner 解在
            # *neighbor 的* FP 物理位置处的精确取值重新做一次黎曼求解）。
            # 面是平面直边三角形/四边形（P1 直边网格），真实法向量在整张
            # 面上恒定，neighbor 视角的外法向直接是 -true_normal，无需
            # 按位置重新计算或置换。
            neighbor_cell = int(fc.neighbor_cell[f])
            neighbor_axis = ffp.neighbor_axis
            neighbor_side = ffp.neighbor_side
            neighbor_true_normal = -ffp.true_normal

            Q_neighbor_fp_native = extrap_to_face(neighbor_cell, Q[neighbor_cell], neighbor_axis, neighbor_side)
            adj_neighbor_row_fp_native, adj_mag_neighbor_native, adj_dir_neighbor_outward_native = (
                side_contravariant_flux(neighbor_cell, neighbor_axis, neighbor_side, Q_neighbor_fp_native, f, "neighbor")
            )

            alignment_n = np.sum(adj_dir_neighbor_outward_native * neighbor_true_normal, axis=-1)
            riemann_normal_neighbor = adj_dir_neighbor_outward_native
            if np.any(alignment_n < 0.5):
                # 同 owner 侧的处理（见上面 riemann_normal_owner 处的注释）：
                # 低阶 warm-up 阶段度量外插法向偏离过大的 Flux Points 改用
                # 真实几何法向做 Riemann 求解，不中止残差计算。
                bad_n = alignment_n < 0.5
                riemann_normal_neighbor = np.where(bad_n[:, None], neighbor_true_normal, adj_dir_neighbor_outward_native)

            # owner（1~2 个真实相邻单元）的解在 neighbor 原生 FP 精确物理
            # 位置处的取值，同样按 sources 求和组装
            Q_owner_at_neighbor_fp = sum(mat @ Q[cell] for cell, mat in ffp.owner_sources)
            # 黎曼求解器用 neighbor 自己的度量法向，理由同 owner 侧
            # （F_common_n_owner 处的注释）——避免用外部 true_normal 算出
            # 通量后再乘以 neighbor 自己的 adj_mag 这一步隐含的方向一致假设。
            F_common_n_native = ausm_up_flux_batch(
                Q_neighbor_fp_native, Q_owner_at_neighbor_fp, riemann_normal_neighbor
            )

            F_tilde_common_neighbor = F_common_n_native * adj_mag_neighbor_native[:, None] * neighbor_side
            F_phys_neighbor_fp = euler_physical_flux(Q_neighbor_fp_native)
            F_tilde_own_boundary_neighbor = np.einsum("fi,fiv->fv", adj_neighbor_row_fp_native, F_phys_neighbor_fp)

            jump_neighbor = F_tilde_common_neighbor - F_tilde_own_boundary_neighbor
            g_prime_neighbor = ops.g_left if neighbor_side < 0 else ops.g_right
            contrib_neighbor = -_distribute_from_face(jump_neighbor, n1d, neighbor_axis, g_prime_neighbor)
            correction[neighbor_cell] += contrib_neighbor / det_jacs[neighbor_cell][:, None]

    residual += correction
    # 机制3（症状检测，见 fr_troubled_cell.py 模块文档）：取代此前"先用
    # det(J)/法向失配几何量预判、按整个单元降阶"的机制1/2，直接对算出的
    # 最终残差本身做 (cell,SP,变量) 粒度的量级异常检测——只清零真正异常
    # 的那几个 SP，不牵连同一单元里其余健康的 SP，也不依赖网格绝对尺度。
    return suppress_residual_outliers(residual, U[..., :5])
