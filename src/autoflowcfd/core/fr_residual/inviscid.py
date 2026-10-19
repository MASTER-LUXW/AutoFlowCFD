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
   fr/face_flux_points/geometry.py）取得 owner/neighbor 双方在物理重合点上的解，
   用已验证正确的 AUSM+up 黎曼求解器（fr_kernels.compute_ausm_up_flux）
   结合 FaceExtractor 给出的真实物理法向量/面积计算公共通量，再通过
   Radau/VCJH 校正函数导数 (matrix_operators.compute_correction_weights)
   投影回 SPs 残差。

正确性通过「均匀自由流场残差应严格为零」(free-stream preservation /
离散 GCL 的实际检验) 数值验证，见 tests/unit/test_fr_residual_inviscid.py。
"""

from typing import Callable, NamedTuple, Optional

import numpy as np
from numba import njit, prange

from autoflowcfd.boundary.fr_ghost_state import require_ghost_provider
from autoflowcfd.core.fr_operators.kernels import compute_ausm_up_flux
from autoflowcfd.core.fr_operators.flux_kernels import euler_physical_flux_point
from autoflowcfd.core.fr_operators.volume_contract import (
    cell_blocks, contract_local, contravariant_point, get_overintegration_context, lifted_segments,
)

GAMMA = 1.4


def conserved_to_primitive(U: np.ndarray) -> np.ndarray:
    """U=(rho,rho*u,rho*v,rho*w,rho*E) -> Q=(rho,u,v,w,p)，沿最后一维前5个分量。

    numba 并行实现（2026-09-25）：此前的 numpy 版本（5 个中间数组 + `np.stack`）
    在 plate_demo P1 上每次约 45 ms、单线程，一次残差求值要做两次，是 16 线程下
    残差里最大的串行项之一。算式与运算顺序和原版逐项相同（`rho` 下限用比较而不是
    `max`，保证 NaN 照样传播——`np.maximum(NaN, x)` 是 NaN，发散检测依赖它）。
    """
    U = np.asarray(U)
    lead = U.shape[:-1]
    U2 = U.reshape(-1, U.shape[-1])
    Q = np.empty((U2.shape[0], 5))
    _conserved_to_primitive_rows(U2, Q)
    return Q.reshape(lead + (5,))


@njit(cache=True, parallel=True)
def _conserved_to_primitive_rows(U, Q):
    gm1 = GAMMA - 1.0
    for i in prange(U.shape[0]):
        rho = U[i, 0]
        if rho < 1e-10:
            rho = 1e-10
        u = U[i, 1] / rho
        v = U[i, 2] / rho
        w = U[i, 3] / rho
        E = U[i, 4] / rho
        ke = 0.5 * (u * u + v * v + w * w)
        Q[i, 0] = rho
        Q[i, 1] = u
        Q[i, 2] = v
        Q[i, 3] = w
        Q[i, 4] = gm1 * rho * (E - ke)


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


def ausm_up_flux_batch(Q_L: np.ndarray, Q_R: np.ndarray, normal: np.ndarray,
                       mach_ref: float = 0.1) -> np.ndarray:
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


class MeanFlowPrimitives(NamedTuple):
    """一次平均流残差求值里无粘与粘性两项共用的原始变量与边界幽灵态（`mean_flow_primitives`）。"""
    Q: np.ndarray          # (n_cells, n_sps, 5)
    Q_ghost: np.ndarray    # (n_faces, n_fp, 5)，只有边界面的行有意义


def mean_flow_primitives(U: np.ndarray, flat, ghost_provider) -> MeanFlowPrimitives:
    """`U` 上的原始变量与边界幽灵态，各算一次供无粘、粘性两项共用
    （`FRSolver.compute_mean_flow_residual`）。两项此前各自从 `U` 重算这两样，结果逐位相同。"""
    from autoflowcfd.core.fr_residual.inviscid_kernel import compute_boundary_ghost_states

    Q = conserved_to_primitive(U[..., :5])
    provider = require_ghost_provider(ghost_provider, "mean_flow_primitives")
    return MeanFlowPrimitives(Q, compute_boundary_ghost_states(flat, Q, provider))


@njit(cache=True, parallel=True)
def _inviscid_volume_kernel(Q, lo, c2f, det_f, inv_f, K2, combo, out, n_blocks):
    """段内单元 `lo + i`：`out[c] = K2[combo[i]] : adj(J) F(Q_fine)`（`inviscid_volume_divergence`）。"""
    n = combo.shape[0]
    n_fine = c2f.shape[0]
    for b in prange(n_blocks):
        q_f = np.empty((n_fine, 5))
        f_t = np.empty((3 * n_fine, 5))
        for i in range(b * n // n_blocks, (b + 1) * n // n_blocks):
            c = lo + i
            contract_local(c2f, Q[c], q_f)
            for p in range(n_fine):
                contravariant_point(det_f[i, p], inv_f[i, p], euler_physical_flux_point(q_f[p]),
                                    f_t[3 * p:3 * p + 3])
            contract_local(K2[combo[i]], f_t, out[c])


def inviscid_volume_divergence(Q, oi):
    """无粘体积项 `K F~`（参考空间、除 det 之前），`(n_cells, n_sps, 5)`；`oi` 为
    `get_overintegration_context` 的结果。

    强形式：`K = f2c·D_fine - Σ_面 lift·W·Tn` 一次收缩——体积散度的 L2 投影减去修正项的本侧通量（细层通量
    多项式在面通量点上的法向迹），界面核只施加公共通量（离散守恒，`fr/face_flux_trace.py`）。`K` 按单元
    三角形面的坍缩顶点槽位组合取（与界面核同一组通量点）。

    **逐单元融合（2026-10-15）**：一个单元的整条链在块内局部工作数组里走完，细点数组不再落成块大小的
    临时数组。此前按 32768 单元分块、每步各自分配（Q_fine / F_phys_fine / F_tilde_fine）：Windows 上新
    分配 1 MB 以上的数组约 0.21 ms/MB 花在缺页上，是复用同一块内存的 10 倍以上；cube_demo 粗网格（28.6 万
    单元）P2 一次残差求值的无粘体积项要分配约 4 GB。逐点公式与求和顺序取自批量核同一份内联定义
    （`volume_contract.contract_local` / `contravariant_point`、`euler_physical_flux_point`），与分块版
    逐位相同（`tests/unit/test_fused_overintegration.py`）。分块那一版本身是为了去掉全场细点数组（79 万
    单元 P2 体积项峰值约 38 GB 的 OOM，2026-08-26）；逐单元执行同样不需要它们。
    """
    Q = np.ascontiguousarray(Q)
    out = np.empty(Q.shape[:2] + (5,))
    for lo, hi, c2f, det_f, inv_f, K2, combo in lifted_segments(oi):
        _inviscid_volume_kernel(Q, lo, c2f, det_f, inv_f, K2, combo, out, cell_blocks(hi - lo))
    return out


def compute_inviscid_residual_fr(
    U: np.ndarray,
    mesh,
    ops,
    boundary_ghost_provider: Callable[[int, np.ndarray, np.ndarray], np.ndarray],
    mach_ref: float = 0.1,
    flat_face_override=None,
    primitives: Optional[MeanFlowPrimitives] = None,
) -> np.ndarray:
    """计算真实面耦合的 FR 无粘残差 dU/dt（物理空间，已除以 det(J)）。

    Args:
        U: 守恒变量，形状 (n_cells, n_sps, n_vars)；只使用前5个欧拉变量
        mesh: HighOrderMesh 实例，需要已调用 load_from_volume_mesh
            （提供 face_connectivity, face_flux_points, jacobians）
        ops: FROperators（D_3d_prism/D_3d_tet 等体积算子）
        boundary_ghost_provider: 可调用对象 (face_idx, Q_owner_fp, true_normal) -> Q_ghost_fp，
            用于给出边界面的幽灵态，必须显式给出（`require_ghost_provider`；零梯度外插传 DefaultGhostProvider()）
        mach_ref: AUSM+up Weiss-Smith 预处理用的参考（自由来流）马赫数
            （见 kernels.py::compute_ausm_up_flux 文档）。默认值 0.1 只是
            保留旧硬编码值以免所有非求解器主循环调用点（CLI 工具/MPI
            冒烟测试/测试文件）都被迫改动；求解器真正的残差求值路径
            （fr_solver/solver.py、mpi/distributed_compute.py）必须显式
            传入 `solver.freestream["mach_ref"]`（真实自由来流马赫数），
            不能依赖这个默认值——CFL 步长估计（cfl.py）用的就是这同一个
            真实值，两者不同步正是 2026-08-14 那次失稳的根因，见 cfl.py
            模块文档"已撤销"一节。
        primitives: 与粘性项共用的原始变量与边界幽灵态（`mean_flow_primitives`，必须由同一个 `U`、同一个
            幽灵态提供者算出）；None 时在这里算。P0 走有限体积路径，不读它。

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
        # 独立的真实几何有限体积路径，见 inviscid_p0.py::compute_inviscid_residual_fv_p0
        # 文档。面着色取自与 P>=1 同一份面几何（CPU 分布式为 flat_face_override）。
        from autoflowcfd.core.fr_operators.face_kernels import get_flat_face_geometry
        from .inviscid_p0 import compute_inviscid_residual_fv_p0

        flat = flat_face_override if flat_face_override is not None else get_flat_face_geometry(mesh, ops)
        return compute_inviscid_residual_fv_p0(U, mesh, flat, boundary_ghost_provider, mach_ref)

    n_cells = mesh.n_cells
    n_sps = mesh.n_sps_per_cell
    Q = conserved_to_primitive(U[..., :5]) if primitives is None else primitives.Q  # (n_cells, n_sps, 5)

    det_jacs = mesh.jacobians["det_jacs"].reshape(n_cells, n_sps)

    # 过积分上下文改用共享 helper（2026-09-17）：此前本函数自己直读
    # `mesh.jacobians_fine` / `mesh.n_sps_per_cell_fine` 并硬编码两段
    # 算子，与 `viscous_flux.py` / `turbulence/transport.py` 那两个消费方
    # 各写一份。native 四面体的过积分细网格轴不再填充到棱柱宽度之后，
    # "每段的 n_fine 和度量切片"这件事必须三处完全一致，所以收敛到
    # `volume_contract.get_overintegration_context` 这一个来源。
    _oi = get_overintegration_context(mesh, ops)
    if _oi is not None:
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

        # 逐单元执行整条 插值→物理通量→逆变通量→体积算子 K 链（`inviscid_volume_divergence`）。
        div_comp = inviscid_volume_divergence(Q, _oi)
    else:
        # order>=1 时 `build_order_geometry` 恒构造细点度量与过积分算子；没有它们说明
        # 网格/算子构造不完整。旧版这里有一条"直接在解点上微分"的兜底分支，它的
        # 修正项本侧通量与体积项不一致（不守恒），且生产上不可达——不保留第二套离散。
        raise RuntimeError(
            "P>=1 无粘残差需要过积分细点度量（mesh.jacobians_fine）与 ops.overint_* 算子，"
            "这里缺失：网格或算子是按不完整的阶数几何构造的")
    # 物理空间残差（体积项部分）`-div_comp / det`，原地算（取负与除法都不改变舍入，与 `-a / b` 逐位相同）：
    # 每个全尺寸临时数组在 cube_demo 粗网格（28.6 万单元）P2 上是 0.31 GB、新分配约 0.07 s 缺页
    residual = np.divide(div_comp, det_jacs[..., None], out=div_comp)
    np.negative(residual, out=residual)
    del div_comp

    # --- 界面项：numba 逐点标量 kernel（性能优化，替代原纯 Python
    # `for f in range(fc.n_faces)` 逐面循环——生产规模网格上 130 万个面、
    # 每次残差求值都要跑一遍，纯 Python 解释器 + 逐次小 numpy 调用的
    # 开销是实测 1546s/次残差求值的主因。控制流/数学公式与原实现逐字
    # 对应，只是执行方式换成编译后的原生代码，见
    # fr_residual_inviscid_kernel.py 模块文档、
    # tests/unit/test_fr_residual_inviscid_kernel_crosscheck.py 的新旧
    # 实现逐位对比验证）。
    ghost_provider = require_ghost_provider(boundary_ghost_provider, "compute_inviscid_residual_fr")

    from autoflowcfd.core.fr_operators.face_kernels import get_flat_face_geometry
    from autoflowcfd.core.fr_residual.inviscid_kernel import (
        compute_boundary_ghost_states,
        compute_inviscid_interface_correction_kernel,
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
    Q_ghost = compute_boundary_ghost_states(flat, Q, ghost_provider) if primitives is None else primitives.Q_ghost

    # 按面着色：同色面无 owner/neighbor 冲突，直接写入共享 buffer（着色结果缓存在 flat 里）

    n_cells = Q.shape[0]
    n_sps = Q.shape[1]
    correction = np.zeros((n_cells, n_sps, 5))
    for c in range(flat.n_colors):
        face_indices = flat.color_face_indices[c]
        if len(face_indices) == 0:
            continue
        compute_inviscid_interface_correction_kernel(
            Q, det_jacs,
            flat.owner_cell, flat.neighbor_cell, flat.is_boundary,
            flat.owner_is_primary, flat.neighbor_is_primary,
            flat.owner_adj_row_exact, flat.neighbor_adj_row_exact,
            flat.neighbor_src0_cell, flat.neighbor_src0_tpl, flat.neighbor_src0_tid,
            flat.neighbor_src1_idx, flat.neighbor_src1_cell, flat.neighbor_src1_mat,
            flat.owner_src0_cell, flat.owner_src0_tpl, flat.owner_src0_tid,
            flat.owner_src1_idx, flat.owner_src1_cell, flat.owner_src1_mat,
            flat.mixed_nb_partner, flat.mixed_nb_mask,
            flat.mixed_ow_partner, flat.mixed_ow_mask,
            Q_ghost,
            face_indices, correction, mach_ref,
            flat.owner_face_op, flat.neighbor_face_op,
            flat.ref_area_weight,
            flat.boundary_extrap_native, flat.lift_native,
        )

    residual += correction
    del correction

    # 机制3（`suppress_residual_outliers`，按 (cell,SP,变量) 粒度检测残差
    # 量级异常并清零）**已于 2026-09-19 删除**。删除依据是真实网格上的
    # 消融对照（plate_demo_volume_les，179,237 单元，P1 + LES）：
    #
    #     机制3 开：888 次调用，触发 15 次，清零 729 个槽位，末残差 nan
    #     机制3 关：876 次调用，触发  0 次，清零   0 个槽位，末残差 nan
    #
    #     step      开                    关                  相对差
    #        1      9.9435206851e+08      9.9435206857e+08    6e-11
    #      100      7.1528723012e+08      7.1528722999e+08    1.8e-10
    #      140      9.0997334909e+08      9.0997334913e+08    4e-11
    #
    # 也就是它**触发了、清零了、但把残差轨迹只改变了 ~1e-10 相对量，
    # 并且没有改变发散这个结局**（两条都在 146~148 步 nan）。健康运行上
    # 它从不触发（真实 LES 网格前 40 步 240 次调用 0 次、检查 17.2 亿个
    # 槽位；合成 Couette/Blasius 两档棱柱基 x 两个阶数 0 次）。
    #
    # 删除后在**健康**运行上做了逐位对照（同一份 plate_demo_volume_les
    # 第 400 步 checkpoint 续算，固定 CFL 0.03、sensor+bounds 默认档，
    # 删除前/删除后各跑 5 步）：`U` 与 `Q` 两个 (179237, 8, 7) 场
    # **逐位完全相同**；另一组 20 步对照的逐步残差也在日志的 7 位
    # 有效数字上全部相同。也就是删除它对健康运行零影响。
    # 单步耗时 11.117 s -> 10.634 s（去掉前 2 步 JIT 预热后的均值，
    # 20 步），省 4.34%。
    #
    # 它的设立理由是**坍缩坐标四面体基**的各向异性放大（1/3 的四面体
    # 单元把残差放大 6~7 个量级），而那套基已于 2026-09-03 删除；原生基
    # 的 `max|D|` 是 O(1~5)，"同单元内残差比达 1e4" 这个触发条件已不可
    # 达。代价是实测 79 万单元 P1 约 2.5 s/步（约占单步 11%），GPU 三条
    # 路径还要为它做一次 GPU->CPU->GPU 往返。
    # 加上 F2 记录的那个"整单元均匀放大逃过检测"的固有盲区与一次
    # 灾难性误判史（P2/P3 四面体残差被整体清零），结论是删除。
    return residual
