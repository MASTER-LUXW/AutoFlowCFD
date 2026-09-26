"""AutoFlowCFD V2.0 - 湍流标量的**扩散**残差（含去混叠体积项）。

从 `core/turbulence/transport.py` 拆出（2026-09-24）；界面项 2026-09-26 改为 IIPG 内罚、两侧各自坐标系。

残差约定：`+div(Gamma*grad(phi))/det(J)`（含 IIPG 内罚界面项），与
`viscous_flux.py` 的"粘性项是 +div(G)"完全同一约定 —— 这个符号曾经写错成
反扩散、把 k/omega 场两极分化到正性限制器的上下界，见包 `__init__.py`
"符号约定"一节记录的那次修复。
"""

import numpy as np


from autoflowcfd.core.fr_operators.gradients import (
    compute_physical_scalar_gradient,
)
from autoflowcfd.core.fr_operators.volume_contract import (
    contract_shared_operator_1axis,
    contract_shared_operator_2axis,
    contravariant_flux_from_metric,
)
from autoflowcfd.core.fr_operators.face_kernels import get_flat_face_geometry
from autoflowcfd.core.turbulence.transport_kernel import (
    scalar_volume_divergence_kernel,
)

from autoflowcfd.core.fr_operators.flux_kernels import resolve_viscous_ip_constant

from .face_frames import diffusion_face_jumps_kernel
from .faces import (
    _lift_side_jumps,
    boundary_diffusion_targets,
    penalty_length,
    unit_normals,
)
from .convection import (
    _TURB_OVERINT_CHUNK_CELLS,
    _turb_overint_ops,
    resolve_turb_overintegration,
)


def _scalar_diffusion_volume_overintegrated(gamma_field, grad_phi, oi, n_sps):
    """扩散体积项 `div(adj(J)*Gamma*grad_phi)` 的去混叠版，返回
    (n_cells,n_sps)。

    与对流版同一条链路：Gamma 与 grad_phi 各自精确插值到 FINE 点后**在
    FINE 点相乘**，用 FINE 点度量算逆变通量、FINE 微分矩阵求散度，再
    精确限制回 coarse。

    **`Gamma` 自身的混叠：已量化、刻意不实施（2026-09-15 结论）**。
    `Gamma = mu + sigma*rho*nu_t` 里 `nu_t` 是**商**、根本不是多项式，
    它在 coarse 点上的节点值本身已经是一个投影结果——本函数只能把这份
    节点表示精确插值到 FINE 点，无法像平均流那样"在 FINE 点重新求值
    非线性通量函数"（`fr_residual/inviscid.py` 能那样做是因为它手里有 Q）。
    所以这里去掉的是 **Gamma×grad_phi 乘积以及与度量项乘积**的混叠。

    要不要把 Gamma 自身那层也去掉，做过受控测量（`Gamma = mu +
    s*rho*k/omega`，rho/k/omega 全取一次场，与解析散度比较，相对 L-inf）：

        order=1 prism: 插值 Gamma 1.0019e-02 -> 细点精确 3.2491e-05  (308x 更好)
        order=1 tet  : 插值 Gamma 9.3413e-03 -> 细点精确 1.4530e-04  ( 64x 更好)
        order=2 prism: 插值 Gamma 1.7408e-02 -> 细点精确 3.4978e-02  (0.50x **更差**)
        order=2 tet  : 插值 Gamma 1.8643e-03 -> 细点精确 1.6869e-03  (1.11x)

    **order=2 上"细点精确求值"反而更差**，不是测量噪声：在细点精确求值
    一个有理函数、再对它的 degree-over_order 插值求导，会把更多高频内容
    带进微分算子；而插值过的 Gamma 本身更平滑。也就是说这条改动**不是
    单调有益**的。

    代价侧同样不小：生产里 `sigma` 来自 SST 混合函数 `F1`，而 `F1` 依赖
    `wall_distance`——那是纯几何量，**必须重新 KD-Tree 查询、不能插值**
    （2026-09-05 真实 bug 修复：阶数切换时把 wall_distance 当解多项式场
    插值导致 d1 系统性偏大），细点是 791492x64 ≈ 5070 万个查询点；此外
    每步还要在 8 倍点数上重算 `F1`/`nu_t`/`CD_kw`。

    综合判断：在 order=2 净负、order=1 的收益又落在一个相对误差已经只有
    1e-2 的项上（对比本轮去掉的 200%~498%），不值这个代价。**这是一条
    有数据支撑的结论，不是待办项**——若将来工作阶数或精度诉求变化需要
    重新评估，上面的数字与代价分析可以直接复用。
    """
    n_cells = gamma_field.shape[0]
    # 每段自带自己的 n_fine 与已切好的细点度量（2026-09-17）：native
    # 四面体的过积分细网格轴不再填充到棱柱的 (oo+1)^3 宽度，两段的
    # n_fine 不同了。度量按**段内局部**索引切（`i0 = c0 - seg_lo`）——
    # 用全局 c0 去切段内数组会静默取到错误的单元。
    div_G = np.empty((n_cells, n_sps))
    for (seg_lo, seg_hi, n_fine, det_seg, inv_seg,
         op_c2f, op_D_fine, op_f2c) in oi["segs"]:
        for c0 in range(seg_lo, seg_hi, _TURB_OVERINT_CHUNK_CELLS):
            c1 = min(c0 + _TURB_OVERINT_CHUNK_CELLS, seg_hi)
            i0, i1 = c0 - seg_lo, c1 - seg_lo
            gam_f = contract_shared_operator_1axis(
                op_c2f, np.ascontiguousarray(gamma_field[c0:c1, :, None]))
            grad_f = contract_shared_operator_1axis(
                op_c2f, np.ascontiguousarray(grad_phi[c0:c1]))        # (n,n_fine,3)
            G_phys_f = (gam_f * grad_f)[..., None]                    # (n,n_fine,3,1)
            del gam_f, grad_f
            G_tilde_f = contravariant_flux_from_metric(
                det_seg[i0:i1], inv_seg[i0:i1], G_phys_f)  # 度量视图直接传（见 contravariant_flux_from_metric）
            del G_phys_f
            div_f = contract_shared_operator_2axis(op_D_fine, G_tilde_f)
            del G_tilde_f
            div_G[c0:c1] = contract_shared_operator_1axis(op_f2c, div_f)[..., 0]
            del div_f
    return div_G


def compute_scalar_diffusion_residual(
    scalar_field: np.ndarray,
    gamma_field: np.ndarray,
    mesh,
    ops,
    wall_dirichlet_zero_face: np.ndarray = None,
    wall_dirichlet_value_face: np.ndarray = None,
    has_wall_dirichlet_value: np.ndarray = None,
    flat_face_override=None,
) -> np.ndarray:
    """标量扩散 FR 残差 `+div(Gamma*grad(phi))/det(J)`（体积项 + IIPG 内罚界面项），
    返回 `d(rho*phi)/dt` 的扩散部分（调用方再除以 rho）。

    ## 符号

    `dphi/dt = +div(Gamma*grad(phi))`（与平均流粘性残差同一约定；曾误写成
    `-div(...)` 反扩散，指数放大棋盘模态）。界面项按 `face_frames.py` 的统一
    约定以 `sign=+1` 提升。**2026-09-26**：此前界面跳变量在 owner 顺序里算好
    后按"owner -= / neighbor +="分配，而扩散跳变量两侧对称，neighbor 侧因此是
    **反扩散**，且用错了通量点顺序——均匀 Gamma 下显式推进 100 步扰动放大 1e9
    倍；修正后扰动单调衰减。

    ## 边界条件（2026-09-26）

    此前真边界面上 ghost 取零梯度、扩散跳变量为零，等于边界通量取单元内梯度的
    单侧值——**没有施加任何边界条件**：无滑移壁上质量通量为零、对流上风 ghost
    不起作用，k=0 实际上无处施加；其余边界也不是零扩散通量。纯 Neumann 小网格上
    这表现为算子零特征值成了 3 阶 Jordan 块（`z^2/2` 一类场的拉普拉斯是常数、
    边界通量却原样流出）。现在：

    * 壁面（`wall_dirichlet_zero_face` 为 k=0、`has_wall_dirichlet_value` 为 omega
      解析值）：内罚 Dirichlet `G*.n = G_self.n - eta (phi_self - g)`；
    * 其余边界（对称、出口、远场、入口）：齐次 Neumann `G*.n = 0`（来流值由对流
      上风 ghost 给出，扩散通量在这些边界上不需要另一条 Dirichlet 条件）。

    罚项与内部面同一个常数与长度尺度（边界面 `h = V_owner / A_face`）。
    2026-09-05 曾加过一次显式壁面罚项（系数用 `C/d1`，d1 为壁面到第一个解点的
    距离，比 `V/A` 小一半以上），显式推进 20 步内全域 omega_mean 2.8e4 -> 5.4e11；
    隐式路径的 omega 壁面单元已是强约束，显式路径的时间步含粘性稳定性限制（与
    平均流内罚项同一刚性量级）。

    Args:
        scalar_field: (n_cells, n_sps) 标量场
        gamma_field: (n_cells, n_sps) 有效扩散系数 Gamma（动力量纲）
        wall_dirichlet_zero_face, wall_dirichlet_value_face, has_wall_dirichlet_value:
            壁面 Dirichlet 掩码与目标值，与对流残差同一组参数（见上文"边界条件"）。
        flat_face_override: 见 `compute_scalar_convection_residual` 同名参数。
    """
    n_cells = mesh.n_cells
    n_sps = mesh.n_sps_per_cell
    n_prism = mesh.n_prism_cells

    det_jacs = mesh.jacobians["det_jacs"].reshape(n_cells, n_sps)
    inv_jacs = mesh.jacobians["inv_jacs"].reshape(n_cells, n_sps, 3, 3)

    # === 体积项 ===
    # 计算标量梯度（度量项一致）——grad_phi 下面界面项的逐分量外插还要
    # 复用，不能纳入分块（只能分块处理它之后、只在体积项内部一次性
    # 使用的 adj_j/G_phys/G_tilde）。
    grad_phi = compute_physical_scalar_gradient(scalar_field, mesh, ops)  # (n_cells, n_sps, 3)

    # 按单元分块执行 adj_j 构造→扩散通量→逆变通量→散度 全链路（真实
    # 内存修复，2026-09-01，理由与 compute_scalar_convection_residual
    # 体积项分块同一处文档：cube_demo 79万单元 P2+SST 组合内存峰值实测
    # 约需 37GB，超过常见 32GB 工作站配置）。
    _tet_op_D = ops.D_native_tet_padded if getattr(ops, "D_native_tet_padded", None) is not None else ops.D_3d_tet
    # 性能优化（2026-09-13，与对流项体积项同一次剖析、同一手法）：整条
    # "度量×扩散通量→散度"链换成 `contravariant_flux_from_metric` +
    # `scalar_volume_divergence_kernel` 两个 numba prange kernel，取代
    # 原来 Python 层分块 + 逐点 3x3@3x1 微型 gemm + 3 次 tensordot 的
    # numpy 链路（见两个 kernel 各自的文档）。
    # 去混叠（AFCFD_TURB_OVERINT=on），理由见 `resolve_turb_overintegration`
    # 与 `_scalar_diffusion_volume_overintegrated`（含那里明确写出的、
    # Gamma 自身混叠仍在的局限）。默认 off，行为逐位不变。
    _oi = _turb_overint_ops(mesh, ops) if resolve_turb_overintegration() == "on" else None
    if _oi is not None:
        div_G = _scalar_diffusion_volume_overintegrated(
            gamma_field, grad_phi, _oi, n_sps)
    else:
        G_phys = gamma_field[:, :, None] * grad_phi                      # (n_cells,n_sps,3)
        G_tilde = contravariant_flux_from_metric(det_jacs, inv_jacs, G_phys[..., None])[..., 0]
        del G_phys
        div_G = np.empty((n_cells, n_sps))
        if n_prism > 0:
            scalar_volume_divergence_kernel(
                np.ascontiguousarray(G_tilde[:n_prism]),
                np.ascontiguousarray(ops.D_3d_prism), div_G[:n_prism],
            )
        if n_cells > n_prism:
            scalar_volume_divergence_kernel(
                np.ascontiguousarray(G_tilde[n_prism:]),
                np.ascontiguousarray(_tet_op_D), div_G[n_prism:],
            )
        del G_tilde

    # 扩散对 dphi/dt 的贡献是 +div(G)/det(J)（见本函数文档符号约定，
    # 与 viscous_flux.py::"residual = div_comp / det_jacs"同一约定）。
    # 退化单元溢出保护：理由/验证方式同 compute_scalar_convection_
    # residual 里对应的 errstate（见该函数文档），同一类已知、已在
    # compute_turbulence_transport_residual 末尾被下游清零处理的溢出。
    with np.errstate(over='ignore', invalid='ignore'):
        residual = div_G / det_jacs
    del div_G

    # === 界面项（BR1 平均通量校正）===
    # === 界面项：IIPG + 内罚项，两侧各自坐标系 ===
    # 与平均流粘性 kernel 同一个格式（`flux_kernels/viscous.py::viscous_ip_penalty_tilde`
    # 文档"为什么内部面也必须加"）：`grad_phi` 是纯单元内局部梯度（没有 BR1 的
    # 提升项），界面耦合若只有"通量取两侧平均"就不控制跨面跳跃——不满足强制性，
    # P0 下内部面扩散恒为零。罚项常数与长度尺度取同一处定义：
    #     G*.n = {G}.n - eta [[phi]],  eta = c_ip * {Gamma} / h_face（见 `faces.penalty_length`）
    #     J_side = (G* - G_side).n_side = 1/2 (G_other - G_side).n_side - eta_side (phi_side - phi_other)
    # 边界点（真边界面、混合拆分面的边界半区）：壁面为内罚 Dirichlet
    # `G*.n = G_self.n - eta (phi_self - g)`，其余为齐次 Neumann `G*.n = 0`，见本函数
    # 文档"边界条件"。
    flat = flat_face_override if flat_face_override is not None else get_flat_face_geometry(mesh, ops)
    c_ip = resolve_viscous_ip_constant(int(mesh.order))
    h_face = np.ascontiguousarray(penalty_length(np, flat), dtype=np.float64)
    masks = (wall_dirichlet_zero_face, wall_dirichlet_value_face, has_wall_dirichlet_value)
    grad_c = np.ascontiguousarray(grad_phi)
    phi_c = np.ascontiguousarray(scalar_field)
    gamma_c = np.ascontiguousarray(gamma_field)
    jumps = []
    for frame, self_cell, self_code, src, adj_row in (
            ("owner", flat.owner_cell, flat.owner_cube_face,
             (flat.neighbor_src0_cell, flat.neighbor_src0_mat, flat.neighbor_src1_idx,
              flat.neighbor_src1_cell, flat.neighbor_src1_mat), flat.owner_adj_row_exact),
            ("neighbor", flat.neighbor_cell, flat.neighbor_cube_face,
             (flat.owner_src0_cell, flat.owner_src0_mat, flat.owner_src1_idx,
              flat.owner_src1_cell, flat.owner_src1_mat), flat.neighbor_adj_row_exact)):
        is_bnd, is_dir, target = boundary_diffusion_targets(np, flat, frame, *masks)
        with np.errstate(over='ignore', invalid='ignore'):
            jumps.append(diffusion_face_jumps_kernel(
                phi_c, gamma_c, grad_c, self_cell, self_code, flat.boundary_extrap_native, *src,
                np.ascontiguousarray(unit_normals(adj_row)), h_face, float(c_ip),
                np.ascontiguousarray(is_bnd), np.ascontiguousarray(is_dir),
                np.ascontiguousarray(target, dtype=np.float64)))
    del grad_phi

    with np.errstate(over='ignore', invalid='ignore'):
        return residual + _lift_side_jumps(jumps[0], jumps[1], +1.0, flat, mesh)
