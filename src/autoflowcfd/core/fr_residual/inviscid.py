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
import os

import numpy as np

from autoflowcfd.core.fr_operators.kernels import compute_ausm_up_flux
from autoflowcfd.core.fr_operators.troubled_cell import suppress_residual_outliers
from autoflowcfd.core.fr_operators.flux_kernels import euler_physical_flux_batch, entropy_stable_volume_divergence_batch
from autoflowcfd.core.fr_operators.volume_contract import contract_shared_operator_1axis, contract_shared_operator_2axis, compute_adj_j

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


def ausm_up_flux_batch(Q_L: np.ndarray, Q_R: np.ndarray, normal: np.ndarray, mach_ref: float = 0.1) -> np.ndarray:
    """对一批 Flux Points 逐点调用 Numba 版 AUSM+up (标量法向通量密度)。

    Args:
        Q_L, Q_R: (n_fp, 5)
        normal: (n_fp, 3) 单位法向量（由 L 指向 R）
        mach_ref: 见 kernels.py::compute_ausm_up_flux 文档

    Returns:
        flux: (n_fp, 5)，F*·n （每单位面积的物理通量密度）
    """
    n_fp = Q_L.shape[0]
    flux = np.zeros((n_fp, 5))
    for i in range(n_fp):
        flux[i] = compute_ausm_up_flux(Q_L[i], Q_R[i], normal[i], mach_ref)
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
    mach_ref: float = 0.1,
) -> np.ndarray:
    """P0 专用有限体积无粘残差。实现见
    inviscid_p0.py::compute_inviscid_residual_fv_p0（从本
    文件拆出，控制单文件行数），文档字符串也在那里。"""
    from .inviscid_p0 import compute_inviscid_residual_fv_p0

    return compute_inviscid_residual_fv_p0(U, mesh, boundary_ghost_provider, mach_ref)


def compute_inviscid_residual_fr(
    U: np.ndarray,
    mesh,
    ops,
    boundary_ghost_provider: Optional[Callable[[int, np.ndarray, np.ndarray], np.ndarray]] = None,
    mach_ref: float = 0.1,
    flat_face_override=None,
    entropy_stable_volume: bool = False,
) -> np.ndarray:
    """计算真实面耦合的 FR 无粘残差 dU/dt（物理空间，已除以 det(J)）。

    Args:
        U: 守恒变量，形状 (n_cells, n_sps, n_vars)；只使用前5个欧拉变量
        mesh: HighOrderMesh 实例，需要已调用 load_from_volume_mesh
            （提供 face_connectivity, face_flux_points, jacobians）
        ops: FROperators（D_3d, g_left, g_right）
        boundary_ghost_provider: 可调用对象 (face_idx, Q_owner_fp, true_normal) -> Q_ghost_fp，
            用于给出边界面的幽灵态；None 时使用 DefaultGhostProvider（零梯度外插）
        mach_ref: AUSM+up Weiss-Smith 预处理用的参考（自由来流）马赫数
            （见 kernels.py::compute_ausm_up_flux 文档）。默认值 0.1 只是
            保留旧硬编码值以免所有非求解器主循环调用点（CLI 工具/MPI
            冒烟测试/测试文件）都被迫改动；求解器真正的残差求值路径
            （fr_solver/solver.py、mpi/distributed_compute.py）必须显式
            传入 `solver.freestream["mach_ref"]`（真实自由来流马赫数），
            不能依赖这个默认值——CFL 步长估计（cfl.py）用的就是这同一个
            真实值，两者不同步正是 2026-08-14 那次失稳的根因，见 cfl.py
            模块文档"已撤销"一节。
        entropy_stable_volume: 体积项非线性通量混叠优化开关（默认关闭，
            行为与此前完全一致）。开启后过积分分支（`mesh.jacobians_fine
            is not None`）改用 Chandrashekar (2013) 熵守恒两点通量 +
            对称平均度量项替代逐点通量代入（见
            `core/fr_operators/flux_kernels.py::entropy_stable_volume_
            divergence_batch` 与 `8_算法重构-Entropy-Stable_Split-Form
            通量重构-Part1/2.md` 完整推导/验证）——真实决定性测试确认
            方向一致、幅度真实但有限的改善（P2 中位数额外再改善约
            3.5 倍，在已经用了过积分的基础上），代价是体积项计算复杂度
            从 O(n_fine) 升到 O(n_fine^2)（两点通量需要遍历 SP 对，是
            entropy-stable 方案的固有代价），默认关闭以避免无条件拖慢
            现有全部生产用例；无过积分分支（P0）或用户显式设为 False
            时行为完全不变。

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
        return _compute_inviscid_residual_fv_p0(U, mesh, boundary_ghost_provider, mach_ref)

    n_cells = mesh.n_cells
    n_sps = mesh.n_sps_per_cell
    n1d = mesh.n_points_1d

    Q = conserved_to_primitive(U[..., :5])  # (n_cells, n_sps, 5)

    det_jacs = mesh.jacobians["det_jacs"].reshape(n_cells, n_sps)
    inv_jacs = mesh.jacobians["inv_jacs"].reshape(n_cells, n_sps, 3, 3)
    adj_j = compute_adj_j(det_jacs, inv_jacs)  # (n_cells,n_sps,3,3), adj_j[...,m,i]
    # adj_j（coarse）在下面界面/校正项里仍要用（side_contravariant_flux 等
    # 闭包捕获），体积项散度改走 over-integration（去混叠）路径，两者不
    # 是同一段计算，coarse adj_j 不能删。

    n_prism = mesh.n_prism_cells

    if mesh.jacobians_fine is not None:
        # 体积项去混叠（over-integration，V2.0 二次评审 Tier 0 #2）：直接
        # 在 coarse SPs 上对 adj(J)*F_phys(Q) 做 D_3d_tet/prism 散度会
        # 把这个非线性乘积（真实多项式次数远高于 order）混叠到
        # degree-order 空间再求导——真实数值实验：对解析残差恒为 0 的
        # 线性剪切场，P2 算出的残差是真值的 43~62 倍。改为：① 把 Q
        # 精确插值到更细的 FINE 参考点集（Q 本身次数 <= order，插值不
        # 引入误差）；② 用解析精确的 FINE 点度量（mesh.jacobians_fine，
        # 与 coarse 版同源，见 HighOrderMesh._build_order_geometry）和
        # 在 FINE 点上重新算的物理通量算出逆变通量；③ 用 FINE 网格自己
        # 的微分矩阵求散度（差分的是更接近真实非线性次数的插值多项式）；
        # ④ 把结果插值限制回 coarse SPs。见
        # fr/collapsed_basis.py::build_overintegration_operators 文档。
        n_fine = mesh.n_sps_per_cell_fine
        det_jacs_fine = mesh.jacobians_fine["det_jacs"].reshape(n_cells, n_fine)
        inv_jacs_fine = mesh.jacobians_fine["inv_jacs"].reshape(n_cells, n_fine, 3, 3)

        # 按单元分块执行整条 插值→物理通量→逆变通量→散度 链（B-12 P2
        # OOM 修复，2026-08-26）：原实现把四个全场临时数组（adj_j_fine
        # ~3.4GiB、Q_fine ~1.9GiB、F_phys_fine ~5.7GiB、F_tilde_fine ~5.7GiB，
        # 79万单元×64过积分点）先后全量分配，峰值时刻三者共存——仪表化
        # 实测（_tmp_review/diag_p2_13.py，31万单元网格）体积项链路自身峰值
        # RSS≈16.6GB，按单元数比例外推 79万单元生产网格峰值≈38GB，超过本机
        # commit limit 43GB 减去系统其余进程占用后的剩余额度，P1→P2 切换后
        # 首次残差求值在 F_phys_fine 处分配失败崩溃（三次独立复现）。这条链上
        # 每一步都是 cell 局部的（批量矩阵乘按 cell 广播、numba kernel 按点纯
        # gather，cell 之间零数据依赖），把 cell 轴切成块、块内走完五步再进下一
        # 块，同一块内的每步计算形状/求和顺序与全场版逐位一致，数值结果不变；
        # 峰值降为常驻的 div_comp_fine（~1.9GiB）+ 块内瞬态（32768 单元块约
        # 0.7GiB）。prism/tet 两段分开切块是因为两者用不同的算子，且单元存储
        # 本来就是 prism 在前 tet 在后，块不会跨类型。
        div_comp_fine = np.zeros((n_cells, n_fine, 5))
        # entropy-stable 路径是 O(n_fine^2)（两点通量遍历 SP 对），用比
        # 强形式更小的分块降低单块瞬态峰值/便于 numba prange 调度粒度，
        # 强形式路径块大小不变（沿用既有 P2 OOM 修复的取值）。
        _OVERINT_CHUNK_CELLS = 4096 if entropy_stable_volume else 32768
        for seg_lo, seg_hi, op_c2f, op_D_fine in (
            (0, n_prism, ops.overint_interp_c2f_prism, ops.overint_D_fine_prism),
            (n_prism, n_cells, ops.overint_interp_c2f_tet, ops.overint_D_fine_tet),
        ):
            for c0 in range(seg_lo, seg_hi, _OVERINT_CHUNK_CELLS):
                c1 = min(c0 + _OVERINT_CHUNK_CELLS, seg_hi)
                Q_fine = contract_shared_operator_1axis(op_c2f, Q[c0:c1])
                adj_j_fine = compute_adj_j(det_jacs_fine[c0:c1], inv_jacs_fine[c0:c1])
                if entropy_stable_volume:
                    # Chandrashekar 两点熵守恒通量 + 对称平均度量项，见
                    # entropy_stable_volume_divergence_batch 文档；该函数
                    # 内部已经把 "-2*div_comp/det_jacs" 里的 2.0 折进
                    # 返回值（与 8_算法重构-Entropy-Stable_Split-Form
                    # 通量重构-Part2.md 决定性验证脚本同一约定），下游
                    # `residual = -div_comp/det_jacs` 不需要再乘 2。
                    div_comp_fine[c0:c1] = entropy_stable_volume_divergence_batch(Q_fine, adj_j_fine, op_D_fine)
                    del adj_j_fine
                else:
                    F_phys_fine = euler_physical_flux_batch(
                        Q_fine.reshape(-1, 5)
                    ).reshape(c1 - c0, n_fine, 3, 5)
                    F_tilde_fine = np.matmul(adj_j_fine, F_phys_fine)  # (块长,n_fine,3,5)
                    del adj_j_fine, F_phys_fine
                    div_comp_fine[c0:c1] = contract_shared_operator_2axis(op_D_fine, F_tilde_fine)
                    del F_tilde_fine
                del Q_fine  # 块内用完即弃，下一轮迭代变量重新绑定

        div_comp = np.zeros((n_cells, n_sps, 5))
        if n_prism > 0:
            div_comp[:n_prism] = contract_shared_operator_1axis(ops.overint_restrict_f2c_prism, div_comp_fine[:n_prism])
        if n_cells > n_prism:
            div_comp[n_prism:] = contract_shared_operator_1axis(ops.overint_restrict_f2c_tet, div_comp_fine[n_prism:])
        del div_comp_fine  # ~1.9GiB，用完即弃
    else:
        # 没有 fine 几何——理论上有两种情形：(a) order==0，但 P0 在函数
        # 入口就已经短路到 _compute_inviscid_residual_fv_p0，不会走到
        # 这里；(b) tet_basis_mode=="native"（Part8 文档"四、明确未做
        # 的后续工作"第4条：native 单纯形基的过积分算子尚未设计，
        # `build_order_geometry` 因此故意不为 native 网格构建
        # jacobians_fine，见该函数文档），四面体部分需要改用
        # `D_native_tet_padded`（Part8 文档"零填充块对角"不变量，填充
        # 行的散度贡献恒为 0）——棱柱部分不受影响，仍是坍缩坐标
        # `D_3d_prism`（棱柱没有 native 概念）。这条分支对普通坍缩坐标
        # P1+ 网格理论上不会被触发（保留只是为了任何未预见的
        # jacobians_fine 缺失场景不静默得到错误答案，而是仍用未去混叠
        # 的朴素路径，不崩溃）。
        Q_flat = np.ascontiguousarray(Q.reshape(-1, 5))
        F_phys = euler_physical_flux_batch(Q_flat).reshape(n_cells, n_sps, 3, 5)
        F_tilde = np.matmul(adj_j, F_phys)  # (n_cells,n_sps,3,5)
        div_comp = np.zeros((n_cells, n_sps, 5))
        if n_prism > 0:
            div_comp[:n_prism] = contract_shared_operator_2axis(ops.D_3d_prism, F_tilde[:n_prism])
        if n_cells > n_prism:
            D_tet_op = (
                ops.D_native_tet_padded
                if getattr(mesh, "tet_basis_mode", "collapsed") == "native"
                else ops.D_3d_tet
            )
            div_comp[n_prism:] = contract_shared_operator_2axis(D_tet_op, F_tilde[n_prism:])

    residual = -div_comp / det_jacs[..., None]  # 物理空间残差（体积项部分）
    # div_comp（~854MiB）用完即弃，理由同上（over-integration 分支/朴素
    # 分支的 div_comp 是同一个局部变量名，del 同样安全，之后不再被引用）。
    del div_comp

    # --- 界面项：numba 逐点标量 kernel（性能优化，替代原纯 Python
    # `for f in range(fc.n_faces)` 逐面循环——生产规模网格上 130 万个面、
    # 每次残差求值都要跑一遍，纯 Python 解释器 + 逐次小 numpy 调用的
    # 开销是实测 1546s/次残差求值的主因。控制流/数学公式与原实现逐字
    # 对应，只是执行方式换成编译后的原生代码，见
    # fr_residual_inviscid_kernel.py 模块文档、
    # tests/unit/test_fr_residual_inviscid_kernel_crosscheck.py 的新旧
    # 实现逐位对比验证）。
    ghost_provider = boundary_ghost_provider if boundary_ghost_provider is not None else DefaultGhostProvider()

    from autoflowcfd.core.fr_operators.face_kernels import get_flat_face_geometry
    from autoflowcfd.core.fr_residual.inviscid_kernel import (
        compute_inviscid_interface_correction_kernel,
        compute_boundary_ghost_states,
    )
    from autoflowcfd.core.fr_residual.inviscid_kernel_colored import (
        compute_inviscid_interface_correction_kernel_colored,
    )

    # #2（V2.0 专家组盲审第4轮，2026-08-28）：flat_face_override 显式传入
    # 时优先使用，不再无条件调用 get_flat_face_geometry(mesh, ops)——CPU
    # 分布式路径（core/mpi/distributed_compute.py）下 `mesh` 是
    # `DistributedMeshAdapter`，其 `face_connectivity`/`face_flux_points`
    # 是 local+halo 压缩索引空间下的分布式几何数据（`DistributedFlatFaceGeometry`
    # /局部 HighOrderMesh 片段），不是 `get_flat_face_geometry` 内部
    # `build_flat_face_geometry` 期望的完整全局 FRFaceConnectivity——用它
    # 重新构建一遍等于试图从不完整/类型不匹配的数据重新推导面几何，
    # 必然出错或产出错误结果。调用方需要直接传入已经按同一套 local+halo
    # 压缩索引空间构造好的 `DistributedFlatFaceGeometry.base_flat`（与
    # gpu_inviscid.py::compute_inviscid_residual_fr_gpu 的 flat_face_cpu
    # 参数同一个道理）。单机路径不传，行为完全不变。
    flat = flat_face_override if flat_face_override is not None else get_flat_face_geometry(mesh, ops)
    Q_ghost = compute_boundary_ghost_states(flat, Q, adj_j, ghost_provider)

    # 图着色方案：同色面无 owner_cell 冲突，直接写入共享 buffer
    # 内存从 O(n_threads * n_cells * n_sps * 5) 降至 O(n_cells * n_sps * 5)
    # 着色结果已缓存在 flat 中（build 时一次性计算，不再重复着色）
    # 通过环境变量或配置可切换回 per-thread buffer 方案
    use_coloring = os.environ.get("AFCFD_USE_COLORING", "1") == "1"
    
    if use_coloring:
        n_cells = Q.shape[0]
        n_sps = Q.shape[1]
        correction = np.zeros((n_cells, n_sps, 5))
        for c in range(flat.n_colors):
            face_indices = flat.color_face_indices[c]
            if len(face_indices) == 0:
                continue
            compute_inviscid_interface_correction_kernel_colored(
                Q, det_jacs,
                flat.owner_cell, flat.neighbor_cell, flat.is_boundary,
                flat.owner_axis, flat.owner_side, flat.neighbor_axis, flat.neighbor_side,
                flat.owner_is_primary, flat.neighbor_is_primary,
                flat.true_normal,
                flat.owner_adj_row_exact, flat.neighbor_adj_row_exact,
                flat.neighbor_src0_cell, flat.neighbor_src0_mat,
                flat.neighbor_src1_idx, flat.neighbor_src1_cell, flat.neighbor_src1_mat,
                flat.owner_src0_cell, flat.owner_src0_mat,
                flat.owner_src1_idx, flat.owner_src1_cell, flat.owner_src1_mat,
                flat.mixed_nb_partner, flat.mixed_nb_mask,
                flat.mixed_ow_partner, flat.mixed_ow_mask,
                flat.boundary_extrap, flat.g_left, flat.g_right, Q_ghost,
                flat.dist_fp_of_sp, flat.dist_axis_coord_of_sp,
                n_prism, face_indices, correction, mach_ref,
                flat.owner_cube_face, flat.neighbor_cube_face,
                flat.true_area_weight,
                flat.boundary_extrap_native, flat.lift_native,
            )
    else:
        # 回退到 per-thread buffer 方案（小网格 + 低线程数可能更快）
        import numba
        n_threads = numba.get_num_threads()
        correction = compute_inviscid_interface_correction_kernel(
            Q, det_jacs,
            flat.owner_cell, flat.neighbor_cell, flat.is_boundary,
            flat.owner_axis, flat.owner_side, flat.neighbor_axis, flat.neighbor_side,
            flat.owner_is_primary, flat.neighbor_is_primary,
            flat.true_normal,
            flat.owner_adj_row_exact, flat.neighbor_adj_row_exact,
            flat.neighbor_src0_cell, flat.neighbor_src0_mat,
            flat.neighbor_src1_idx, flat.neighbor_src1_cell, flat.neighbor_src1_mat,
            flat.owner_src0_cell, flat.owner_src0_mat,
            flat.owner_src1_idx, flat.owner_src1_cell, flat.owner_src1_mat,
            flat.mixed_nb_partner, flat.mixed_nb_mask,
            flat.mixed_ow_partner, flat.mixed_ow_mask,
            flat.boundary_extrap, flat.g_left, flat.g_right, Q_ghost,
            flat.dist_fp_of_sp, flat.dist_axis_coord_of_sp,
            n_prism, n_threads, mach_ref,
            flat.owner_cube_face, flat.neighbor_cube_face,
            flat.true_area_weight,
            flat.boundary_extrap_native, flat.lift_native,
        )
    residual = residual + correction

    # 机制3（症状检测，见 fr_troubled_cell.py 模块文档）：取代此前"先用
    # det(J)/法向失配几何量预判、按整个单元降阶"的机制1/2，直接对算出的
    # 最终残差本身做 (cell,SP,变量) 粒度的量级异常检测——只清零真正异常
    # 的那几个 SP，不牵连同一单元里其余健康的 SP，也不依赖网格绝对尺度。
    return suppress_residual_outliers(residual, U[..., :5])
