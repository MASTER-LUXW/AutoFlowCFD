"""
AutoFlowCFD V2.0 - P>=1 高阶 FR 无粘残差 GPU 实现

完整的高阶 FR 无粘残差 GPU 版本，对应 core/fr_residual_inviscid.py。
包含两部分：
1. 体积项：CuPy 向量化（物理通量 + 张量收缩 + 度量项）
2. 界面项：CuPy kernel（AUSM+up + 校正分配，按图着色逐色处理）

设计：
- 体积项完全用 CuPy 向量化操作（cp.matmul, cp.tensordot），底层走 cuBLAS
- 界面项使用 CuPy ElementwiseKernel 逐面计算 AUSM+up 通量
- 校正分配使用图着色保证无冲突写入（同色面无 owner_cell 冲突）
- 数据全部常驻 GPU，避免 CPU↔GPU 传输
"""

import numpy as np
from typing import Callable, Optional
from loguru import logger

from autoflowcfd.core.fr_operators.kernels import (
    PRECOND_LEGACY,
    PRECOND_PHYSICAL,
    PRECOND_PRESSURE_PHYSICAL,
    resolve_ausm_precond_mode,
)
from autoflowcfd.core.gpu import get_cupy
from autoflowcfd.core.gpu.residual.gpu_flux import (
    euler_physical_flux_gpu,
    conserved_to_primitive_gpu,
)
from autoflowcfd.core.gpu.residual.gpu_inviscid_volume import (
    prepare_mesh_data as _prepare_mesh_data,
    prepare_ops_data as _prepare_ops_data,
    compute_volume_term_gpu as _compute_volume_term_gpu_impl,
    distribute_face_correction_to_sps,
)


def _compute_volume_term_gpu(U, mesh_data, ops_data, n_cells, n_sps, n_prism):
    """体积项计算，实现已拆分到 gpu_inviscid_volume.py（控制单文件行数）。"""
    return _compute_volume_term_gpu_impl(
        get_cupy(), U, mesh_data, ops_data, n_cells, n_sps, n_prism,
    )


def compute_inviscid_residual_fr_gpu(
    U,
    mesh,
    ops,
    boundary_ghost_provider=None,
    mesh_data=None,
    ops_data=None,
    flat_face_gpu=None,
    flat_face_cpu=None,
    device_id=0,
    mach_ref=0.1,
    precond_mode=None,
):
    """P>=1 高阶 FR 无粘残差的 GPU 实现。

    与 core/fr_residual_inviscid.py::compute_inviscid_residual_fr 公式完全一致。

    Args:
        U: CuPy 数组 (n_cells, n_sps, n_vars) 或 numpy 数组（自动上传）
        mesh: HighOrderMesh
        ops: FROperators
        boundary_ghost_provider: 边界幽灵态提供者
        mesh_data: 预上传的网格数据（可选，None 时自动上传）
        ops_data: 预上传的算子数据（可选）
        flat_face_gpu: 预构建的 GPU 面几何（可选）
        device_id: GPU 设备 ID
        mach_ref: AUSM+up Weiss-Smith 预处理参考马赫数（见
            kernels.py::compute_ausm_up_flux 文档）。默认值 0.1 只是
            保留旧硬编码值，真正的求解器路径（gpu_solver.py）必须显式
            传入 `solver.freestream["mach_ref"]`，不能依赖这个默认值。
        precond_mode: AUSM+up 预处理声速作用域，None 表示按环境变量
            `AFCFD_AUSM_PRECOND_MODE` 解析（与 CPU 端同一个解析器，
            保证同一次运行里 CPU/GPU 两条路径取到同一档）。

    Returns:
        residual: CuPy 数组 (n_cells, n_sps, 5) 或 numpy 数组（与输入同类型）
    """
    cp = get_cupy()
    if cp is None:
        raise RuntimeError("CuPy is not available")

    input_is_numpy = isinstance(U, np.ndarray)
    if input_is_numpy:
        with cp.cuda.Device(device_id):
            U = cp.asarray(U)

    # #1（V2.0 专家组盲审第4轮，2026-08-28）：n_cells/n_prism 此前恒从
    # `mesh` 读取——分布式多 GPU 路径下，`mesh` 传入的是完整全局网格
    # （`get_flat_face_geometry(mesh, ops)` 下面几行需要它提供真实
    # face_connectivity 来算边界幽灵态，不能替换成压缩后的 local+halo
    # 子集），但残差/体积项数组的真实尺寸是 local+halo 压缩索引空间
    # 大小（`mesh_data`/`flat_face_gpu` 已经是按这个压缩空间预先构造
    # 好、显式传入的），与 `mesh.n_cells`/`mesh.n_prism_cells`（全局
    # 尺寸）不一致——用全局尺寸给残差数组分配形状、给体积项当切片
    # 阈值，会得到形状不匹配（IndexError/崩溃）或更隐蔽地用错误阈值
    # 切分棱柱/四面体。`mesh_data`/`ops_data` 由调用方显式传入时，
    # 优先从其中读取（array_manager.py::upload_mesh_data 自己就会写
    # `mesh_data['n_cells']`/`['n_prism']`，分布式调用方构造压缩版
    # mesh_data 时同样要写这两个 key，见 gpu_distributed.py 的构造处）；
    # 未显式传入时（单机 GPUFRSolver 路径）保持原有行为，直接读 mesh。
    if mesh_data is not None and 'n_cells' in mesh_data:
        n_cells = mesh_data['n_cells']
        n_prism = mesh_data.get('n_prism', mesh.n_prism_cells)
    else:
        n_cells = mesh.n_cells
        n_prism = mesh.n_prism_cells
    n_sps = mesh.n_sps_per_cell
    n1d = mesh.n_points_1d

    # ── 准备网格数据（如果未预上传）──
    if mesh_data is None:
        mesh_data = _prepare_mesh_data(cp, mesh, device_id)
    if ops_data is None:
        ops_data = _prepare_ops_data(cp, ops, device_id)

    # ── 1. 体积项 ──
    residual = _compute_volume_term_gpu(
        U, mesh_data, ops_data, n_cells, n_sps, n_prism,
    )

    # ── 2. 界面项 ──
    # flat_face（CPU 侧）恒需要获取——即便调用方已经预构建好
    # flat_face_gpu，边界幽灵态计算仍要用 CPU 侧 flat_face（见下方
    # _compute_boundary_ghost_states_gpu 文档）；get_flat_face_geometry
    # 有单进程单槽缓存（face_kernels.py），同一 mesh/ops 再次调用是
    # 缓存命中，不会重复构建。
    #
    # #1（2026-08-28）：`flat_face_cpu` 显式传入时优先使用，不再无条件
    # 调用 get_flat_face_geometry(mesh, ops)——分布式多 GPU 路径下这里
    # 的 `mesh` 是完整全局网格，get_flat_face_geometry 会返回全局尺寸的
    # face 几何（owner_cell 等是全局单元编号），但 Q_gpu 现在是
    # local+halo 压缩索引空间大小；用全局 flat_face 去算边界幽灵态会
    # 用全局单元编号误当压缩索引去读 Q_gpu（要么越界崩溃，要么读到
    # 毫不相关单元的数据），且会为不属于本 rank 的边界面白白计算一遍。
    # 调用方（MultiGPUDistributedSolver）需要传入 dist_flat_face.base_flat
    # ——那是已经按同一套 local+halo 压缩索引重映射过、且只包含本 rank
    # 负责的面的 FlatFaceGeometry，与 flat_face_gpu 出自同一次
    # build_distributed_flat_face 调用，两者索引空间天然一致。单机路径
    # 不传这个参数，行为完全不变。
    if flat_face_cpu is not None:
        flat_face = flat_face_cpu
    else:
        from autoflowcfd.core.fr_operators.face_kernels import get_flat_face_geometry
        flat_face = get_flat_face_geometry(mesh, ops)
    if flat_face_gpu is None:
        from autoflowcfd.core.gpu.gpu_face_geometry import build_gpu_flat_face
        flat_face_gpu = build_gpu_flat_face(flat_face, device_id)

    Q_gpu = conserved_to_primitive_gpu(U[..., :5])
    adj_j = mesh_data['adj_j']
    det_jacs = mesh_data['det_jacs']

    # 边界幽灵态：必须调用真实 ghost_provider 才能正确处理 WALL 无滑移/
    # INLET/FARFIELD/SYMMETRY 边界（CPU 上计算，然后上传到 GPU——见
    # _compute_boundary_ghost_states_gpu 文档）。
    Q_ghost_gpu = _compute_boundary_ghost_states_gpu(
        Q_gpu, flat_face, boundary_ghost_provider, device_id,
    )

    # 界面校正（按图着色逐色处理）
    correction = _compute_interface_correction_gpu(
        Q_gpu, adj_j, det_jacs, flat_face_gpu, Q_ghost_gpu,
        n_cells, n_sps, n_prism, device_id, mach_ref, precond_mode,
    )

    residual = residual + correction

    # ── 3. 异常残差抑制 ──
    # 第四次评审修复：此前 if/else 两分支代码逐字相同，都以 cp.asarray(result)
    # 结尾——不管 input_is_numpy 是 True 还是 False 都返回 CuPy 数组，
    # 与函数文档承诺的"与输入同类型（numpy 输入返回 numpy）"矛盾，是一处
    # 补丁堆叠后未清理的重复分支。
    from autoflowcfd.core.fr_operators.troubled_cell import suppress_residual_outliers
    residual_np = cp.asnumpy(residual)
    U_np = cp.asnumpy(U)
    result = suppress_residual_outliers(residual_np, U_np[..., :5])
    return result if input_is_numpy else cp.asarray(result)


def _compute_boundary_ghost_states_gpu(
    Q_gpu, flat_face, ghost_provider, device_id,
):
    """计算边界面每个 FP 各自的真实边界条件幽灵态。

    真实 bug 修复（V2.0 专家组盲审发现，2026-08-27）：此前这里完全
    没有调用 `ghost_provider`——无论调用方传入 `None` 还是真实的
    `BoundaryGhostStateProvider`，都恒定用"owner cell SP0 值零梯度
    外插、对该面全部 FP 广播同一个值"代替（参数被静默丢弃）。等价于
    让 WALL/INLET/OUTLET/FARFIELD/SYMMETRY 全部边界面在无粘通量计算
    里"对该边界不可见"（镜像内部值），P>=1 阶数的 GPU 无粘残差路径上
    车身壁面、来流边界、出口条件形同虚设——CLI `solve steady/transient
    --backend gpu` 默认 order=2，恒定触发这条路径。

    `ghost_provider` 是任意 Python 可调用对象，numba/CuPy 都调不了，
    只能在 CPU 上跑（复用已验证的 P1+ CPU 实现
    `inviscid_kernel.py::compute_boundary_ghost_states`，只循环边界面，
    约占全部面的 3%，性能可接受）——与 `gpu_viscous.py` 里粘性残差
    的同类边界幽灵态计算用的是同一个 CPU round-trip 模式。

    Args:
        Q_gpu: (n_cells, n_sps, 5) 当前原变量（GPU）
        flat_face: CPU 侧 FlatFaceGeometry（get_flat_face_geometry 返回）
        ghost_provider: 边界幽灵态提供者，None 时退化为
            DefaultGhostProvider（镜像 CPU 路径的默认行为）
        device_id: GPU 设备 ID

    Returns:
        Q_ghost_gpu: (n_faces, n_fp, 5)，每个边界面每个 FP 各自的
            幽灵态（只有边界面对应的行有意义），与 CPU 版
            `compute_boundary_ghost_states` 返回形状一致
    """
    cp = get_cupy()
    from autoflowcfd.core.fr_residual.inviscid import DefaultGhostProvider
    from autoflowcfd.core.fr_residual.inviscid_kernel import compute_boundary_ghost_states

    provider = ghost_provider if ghost_provider is not None else DefaultGhostProvider()
    Q_cpu = cp.asnumpy(Q_gpu)
    Q_ghost_np = compute_boundary_ghost_states(flat_face, Q_cpu, None, provider)
    with cp.cuda.Device(device_id):
        return cp.asarray(Q_ghost_np)


def _extrap_q_to_fp(cp, mat, src_cell, Q_gpu):
    """(nF,n_fp,n_sps) @ Q_gpu[src_cell] (nF,n_sps,5) -> (nF,n_fp,5)。"""
    return cp.matmul(mat, Q_gpu[src_cell])


def _add_q_src1_to_fp(cp, out, src1_idx, src1_cell, src1_mat, Q_gpu):
    """叠加稀疏第二来源（分裂面场景），Q 专用版本，见
    gpu_viscous.py::_add_src1_to_fp 同名通用版本的文档（这里内联一份
    Q-only 版本，避免 gpu_inviscid.py<->gpu_viscous.py 产生循环 import：
    两者都从 gpu_inviscid_volume.py 导入 prepare_mesh_data/prepare_ops_data，
    不再互相 import）。"""
    has1 = src1_idx >= 0
    if not bool(cp.any(has1)):
        return out
    sel = cp.where(has1)[0]
    idx1 = src1_idx[sel]
    c1 = src1_cell[idx1]
    m1 = src1_mat[idx1]
    out[sel] = out[sel] + cp.matmul(m1, Q_gpu[c1])
    return out


def _ausm_direction_with_fallback(cp, adjrow, side, true_normal_ref):
    """按 CPU 版 inviscid_kernel.py 的"自洽方向 + true_normal 对齐安全阀"
    逻辑构造 AUSM+up 用的法向：adjrow 精确方向与 true_normal_ref 夹角
    过大（alignment<0.5）时回退到 true_normal_ref 本身。

    Args:
        adjrow: (n, n_fp, 3) 未归一化 adj(J) 行
        side: (n,) ±1（collapsed 面）或恒为 1.0（native 面，见调用方
            `_native_aware_side_factor` 文档——native 面用固定 +1 方向
            系数，不能沿用 owner_side/neighbor_side 复用槽位的哑值）
        true_normal_ref: (n, n_fp, 3) 对齐基准（owner 侧用 true_normal，
            neighbor 侧用 -true_normal，见 CPU kernel"neighbor 视角外
            法向恒为 -true_normal"）

    Returns:
        (direction, adj_mag)：direction (n,n_fp,3)，adj_mag (n,n_fp)
    """
    a0 = adjrow[..., 0]
    a1 = adjrow[..., 1]
    a2 = adjrow[..., 2]
    adj_mag = cp.sqrt(a0 * a0 + a1 * a1 + a2 * a2)
    adj_mag_safe = cp.maximum(adj_mag, 1e-300)
    s = side[:, None]
    dirx = a0 / adj_mag_safe * s
    diry = a1 / adj_mag_safe * s
    dirz = a2 / adj_mag_safe * s
    alignment = dirx * true_normal_ref[..., 0] + diry * true_normal_ref[..., 1] + dirz * true_normal_ref[..., 2]
    use_fallback = alignment < 0.5
    dirx = cp.where(use_fallback, true_normal_ref[..., 0], dirx)
    diry = cp.where(use_fallback, true_normal_ref[..., 1], diry)
    dirz = cp.where(use_fallback, true_normal_ref[..., 2], dirz)
    direction = cp.stack([dirx, diry, dirz], axis=-1)
    return direction, adj_mag


def _native_aware_side_factor(cp, is_native, side):
    """native 面方向系数固定 +1，collapsed 面沿用 owner_side/neighbor_side
    ——与 CPU 版 inviscid_kernel.py `side_factor_o = 1.0 if o_is_native
    else oside` 逐字对应（原理见该文件同名注释："_native_tet_adj_row_
    batched 已给出正确 outward 定向，不能沿用 oside 那个复用槽位的哑值"）。
    """
    return cp.where(is_native, cp.float64(1.0), side.astype(cp.float64))


def _native_self_extrap(cp, is_native, cube_face_code, boundary_extrap_native, E_collapsed):
    """自身面外插矩阵：native 面用 `boundary_extrap_native[excluded_vertex]`
    查表，collapsed 面用调用方已经按 (celltype,axis,side_idx) 索引好的
    `E_collapsed`——与 CPU 版 `E_o = boundary_extrap_native[oc_code-6] if
    o_is_native else boundary_extrap[celltype_o,oax,oside_idx]` 逐字对应。

    Args:
        is_native: (n,) bool
        cube_face_code: (n,) int，原始 cube face 编码（>=6 时 -6 才是
            合法的 excluded_vertex 索引；<6 时这个表达式本身无意义，但
            仍会被算出来去 gather boundary_extrap_native——用 clip 确保
            gather 本身不越界，实际值由下面 cp.where 按 is_native 丢弃，
            不会污染结果，与 CPU 版"native 分支只在 o_is_native 为真时
            才使用这个索引"的分派逻辑等价）。
        boundary_extrap_native: (4, n_fp, n_sps)
        E_collapsed: (n, n_fp, n_sps) 调用方已经用 celltype/axis/side_idx
            gather 好的 collapsed 外插矩阵

    Returns:
        E: (n, n_fp, n_sps)
    """
    # 真实 bug 修复（2026-09-02，实现单 GPU Order Continuation 时首次
    # 真正端到端构造+调用 `GPUFRSolver.step()` 才发现——此前既有的
    # native/collapsed crosscheck 测试恒用 `tet_basis_mode="native"`
    # 构造网格，从未覆盖过默认的 `tet_basis_mode="collapsed"`
    # + 网格里确实存在四面体这个组合）：`tet_basis_mode="collapsed"`
    # 时整个网格没有任何单元走 native 分支，`boundary_extrap_native`
    # 是形状 `(0, n_fp, n_sps)` 的空数组（`generate_fr_operators` 压根
    # 不会为 collapsed 模式生成任何 native 表）——`cp.clip(x, 0,
    # shape[0]-1)` 在 `shape[0]==0` 时退化成 `cp.clip(x, 0, -1)`
    # （下界>上界的病态区间），`boundary_extrap_native[excluded_vertex]`
    # 无论如何都会用某个越界下标去取一个长度为 0 的数组——不管
    # `is_native` 掩码本身是否恒为 False（这正是 collapsed 模式下的
    # 真实情形），`cp.where` 的两个分支都会被**提前**无条件求值，
    # `E_native` 这一步本身就已经崩溃，`cp.where` 根本没有机会把它
    # 丢弃。真实网格上任何 `--tet-basis-mode collapsed`（CLI 默认值）
    # + 单 GPU 后端 + 阶数>=1 + 网格含四面体的组合都会在第一次调用
    # `compute_inviscid_residual_gpu()` 时崩溃，与 Order Continuation
    # 本身无关，是一个此前从未被任何测试捕捉到的独立真实 bug（既有
    # crosscheck 测试从未测过这个组合）。修复：完全没有 native 单元时
    # （`boundary_extrap_native.shape[0] == 0`，等价于 `is_native`
    # 全为 False）直接短路返回 `E_collapsed`，不进入这条会越界的
    # gather 路径——`is_native` 全 False 时结果本来就恒等于
    # `E_collapsed`，这不是近似，是这个分支在该前提下唯一可能的取值。
    if boundary_extrap_native.shape[0] == 0:
        return E_collapsed
    excluded_vertex = cp.clip(cube_face_code - 6, 0, boundary_extrap_native.shape[0] - 1)
    E_native = boundary_extrap_native[excluded_vertex]  # (n,n_fp,n_sps)
    return cp.where(is_native[:, None, None], E_native, E_collapsed)


def _native_or_collapsed_contrib(
    cp, is_native, cube_face_code, lift_native, true_area_weight_face, jump, contrib_collapsed,
):
    """面校正分配到体积节点：native 面用 DG 提升算子
    `lift_native[excluded_vertex] @ (true_area_weight ⊙ jump)`，collapsed
    面用调用方已经算好的 `contrib_collapsed`（1D 修正函数分布，见
    `distribute_face_correction_to_sps`）——与 CPU 版
    `contrib_owner = lift_native[oc_code-6] @ weighted_jump_o if
    o_is_native else _distribute_point(...)` 逐字对应。

    Args:
        cube_face_code: (n,)
        lift_native: (4, n_sps, n_fp)
        true_area_weight_face: (n, n_fp)
        jump: (n, n_fp, 5)
        contrib_collapsed: (n, n_sps, 5)

    Returns:
        contrib: (n, n_sps, 5)
    """
    # 同一处真实 bug 修复，见 `_native_self_extrap` 文档——
    # `tet_basis_mode="collapsed"` 时 `lift_native` 是空数组
    # （`(0, n_sps, n_fp)`），`is_native` 恒为 False，短路直接返回
    # `contrib_collapsed`，避免对空数组做越界 gather。
    if lift_native.shape[0] == 0:
        return contrib_collapsed
    excluded_vertex = cp.clip(cube_face_code - 6, 0, lift_native.shape[0] - 1)
    lift = lift_native[excluded_vertex]  # (n, n_sps, n_fp)
    weighted_jump = true_area_weight_face[..., None] * jump  # (n, n_fp, 5)
    contrib_native = cp.matmul(lift, weighted_jump)  # (n, n_sps, 5)
    return cp.where(is_native[:, None, None], contrib_native, contrib_collapsed)


def _compute_interface_correction_gpu(
    Q_gpu, adj_j, det_jacs, flat_face_gpu, Q_ghost_gpu,
    n_cells, n_sps, n_prism, device_id, mach_ref, precond_mode=None,
):
    """GPU 界面校正计算（按图着色逐色处理）。

    真实 bug 修复（2026-08-23，本次移植 GPU 粘性界面项时顺带发现并
    修复，用户明确要求本轮一并处理）：此前的实现有两个复合缺陷：

    1. **无 owner_is_primary/neighbor_is_primary 过滤**：`owner_src0`/
       `neighbor_src0` 字段的真实语义是"跨单元交叉引用数据，只在对应
       角色 primary 时才被写入"（见 face_flux_points_merge.py
       "kernel 只在 owner_primary[f] 为真...时写入 _nb_interp[f]，其余
       情形保持全零"的说明）——此前这里把它们当"这条记录自己的原生值"
       无条件读取。对约 5% 因棱柱四边形侧面拆分成 2 条记录的面：非
       owner_primary 记录的 `neighbor_src0` 全零/cell=-1，AUSM+up 会用
       一个全零假邻居态算出真实错误的通量（不只是重复计数），owner 侧
       的真实贡献还会被两条记录各计入一次。
    2. **owner/neighbor 两侧共享同一个 AUSM+up 通量分别分配**：
       inviscid_kernel.py 模块文档"关键正确性约束"明确记录过这个
       "优化"在真实网格上被验证证伪——自由流场残差从 9e-5 恶化到
       3.1e7，owner/neighbor 两侧必须用各自的度量法向、各自原生 FP
       位置插值出的状态独立各调用一次 AUSM+up，不能合并复用。
    3. **符号约定**：CPU 版每一侧的贡献是 `correction[...] += -contrib
       /dj`（带负号，配合无粘残差 `-div(F)` 的整体约定），此前这里
       scatter 的是不带负号的 `contrib_o`/`contrib_n`。

    修复：逐字仿照 inviscid_kernel.py 的两段独立结构（owner-primary
    过滤块 + neighbor-primary 过滤块），owner/neighbor 各自的"自身
    原生值"改用 `boundary_extrap` 查表外插（与 primary 状态无关、对
    任何记录都有效，不依赖 src0），"跨单元值"才用 src0/src1（只在
    对应角色 primary 时读取有意义）；两侧各自独立调用一次批量 AUSM+up；
    accumulate 时统一带负号。

    本机没有 CUDA/cupy，这处修复无法在本地实际执行验证（全仓库此前
    也没有任何 `compute_inviscid_residual_fr_gpu` 的 crosscheck 测试，
    见 test_gpu_p1_inviscid_interface_crosscheck.py 模块文档——这是
    这次连带发现的又一个"从未被验证过"的 GPU 生产路径），只能靠对照
    已经过详细验证的 CPU kernel 逐字核对；请在有真实 GPU 的环境上跑
    该新增测试文件做最终确认。
    """
    if precond_mode is None:
        precond_mode = resolve_ausm_precond_mode()
    precond_mode = int(precond_mode)

    cp = get_cupy()
    ff = flat_face_gpu
    correction = cp.zeros((n_cells, n_sps, 5), dtype=cp.float64)

    for c in range(ff.n_colors):
        face_idx = ff.color_face_indices[c]
        if face_idx.shape[0] == 0:
            continue

        is_bnd = ff.is_boundary[face_idx]
        owner_primary = ff.owner_is_primary[face_idx]
        neighbor_primary = ff.neighbor_is_primary[face_idx]

        # ── owner-primary 贡献块 ──
        mask_o = owner_primary
        if bool(cp.any(mask_o)):
            idx_o = face_idx[mask_o]
            oc = ff.owner_cell[idx_o]
            oax = ff.owner_axis[idx_o]
            oside = ff.owner_side[idx_o]
            is_bnd_o = ff.is_boundary[idx_o]

            # #1（2026-08-28）：分布式 local+halo 扩展索引空间下，
            # `oc < n_prism` 这个单一阈值判据不成立（见
            # distributed_flat_face.py::DistributedFlatFaceGeometry.
            # compact_cell_type 文档）——ff.compact_cell_type 存在时
            # （分布式路径）改用逐位置查表；单机路径 ff 没有这个属性，
            # getattr 回退到原有阈值判据，行为完全不变。
            compact_cell_type = getattr(ff, 'compact_cell_type', None)
            if compact_cell_type is not None:
                celltype_o = compact_cell_type[oc]
            else:
                celltype_o = cp.where(oc < n_prism, 0, 1)
            oside_idx = cp.where(oside < 0, 0, 1)

            # native 四面体（路径C）GPU 移植（2026-09-02）：`owner_cube_face`
            # >=6 即 native 真实面，自身面外插改用 `boundary_extrap_native`
            # 查表，与 CPU 版 inviscid_kernel.py 逐字对应；纯坍缩坐标网格下
            # `oc_code_o` 恒 <6，`is_native_o` 恒为 False，行为完全不变。
            oc_code_o = ff.owner_cube_face[idx_o]
            is_native_o = oc_code_o >= 6

            # 真实 bug 修复（2026-09-03，首次用完整 numpy-as-cupy 替身
            # 真正端到端跑通 `tet_basis_mode="native"` + GPU 才发现——
            # 此前"native/collapsed crosscheck"测试文件整体
            # `pytest.importorskip("cupy")` 跳过，从未真正执行过这个
            # 组合）：`oax`（owner_axis）对 native 面存的是复用的
            # excluded_vertex（取值 0~3，见 face_flux_points_merge.py
            # 模块文档"owner_axis 对 native 面存的是复用的 excluded_
            # vertex"一节），不是坍缩坐标的真实轴（0~2）——
            # `ff.boundary_extrap` 的轴维度只有 3（0/1/2），native 面
            # `oax==3` 时这行无条件 gather 会直接 `IndexError`，不管
            # 下面 `_native_self_extrap` 最终是否会丢弃这个结果（与
            # `_native_self_extrap` 自己那处"`boundary_extrap_native`
            # 空数组 + `cp.clip(x,0,-1)`"是同一类"提前无条件求值导致
            # 越界"问题，只是这里是另一张表）。修复：native 面统一把
            # `oax` clip 到 0（一个恒安全的哑值，结果本来就会被
            # `is_native_o` 丢弃，不影响 collapsed 面的真实行为）。
            oax_safe = cp.where(is_native_o, 0, oax)
            E_o_collapsed = ff.boundary_extrap[celltype_o, oax_safe, oside_idx]  # (nO,n_fp,n_sps)
            E_o = _native_self_extrap(cp, is_native_o, oc_code_o, ff.boundary_extrap_native, E_o_collapsed)
            Q_o = cp.matmul(E_o, Q_gpu[oc])  # (nO,n_fp,5)

            Q_n = _extrap_q_to_fp(cp, ff.neighbor_src0_mat[idx_o], ff.neighbor_src0_cell[idx_o], Q_gpu)
            Q_n = _add_q_src1_to_fp(
                cp, Q_n, ff.neighbor_src1_idx[idx_o], ff.neighbor_src1_cell, ff.neighbor_src1_mat, Q_gpu,
            )

            n_fp = Q_o.shape[1]
            # Q_ghost_gpu 现在是逐 FP 幽灵态 (n_faces,n_fp,5)（见
            # _compute_boundary_ghost_states_gpu 文档），直接按 FP 对齐
            # 使用，不再对该面全部 FP 广播同一个值。
            Q_ghost_face = Q_ghost_gpu[idx_o]  # (nO, n_fp, 5)
            Q_n = cp.where(
                is_bnd_o[:, None, None],
                Q_ghost_face,
                Q_n,
            )

            # 混合拆分面（B-8，镜像 CPU inviscid_kernel.py 同名分支）：混合配对的内部面在边界半区
            # 逐 FP 取配对边界面的幽灵态。
            mp_o = ff.mixed_nb_partner[idx_o]
            mixed_sel_o = (mp_o[:, None] >= 0) & ff.mixed_nb_mask[idx_o]  # (nO, n_fp)
            Q_ghost_partner_o = Q_ghost_gpu[cp.maximum(mp_o, 0)]  # (nO, n_fp, 5)
            Q_n = cp.where(
                mixed_sel_o[..., None],
                Q_ghost_partner_o,
                Q_n,
            )

            side_factor_o = _native_aware_side_factor(cp, is_native_o, oside)
            adjrow_o = ff.owner_adj_row_exact[idx_o]
            direction_o, adj_mag_o = _ausm_direction_with_fallback(
                cp, adjrow_o, side_factor_o, ff.true_normal[idx_o],
            )

            nO = Q_o.shape[0]
            flux_o = _ausm_up_flux_batch_gpu(
                Q_o.reshape(nO, n_fp, 5), Q_n.reshape(nO, n_fp, 5), direction_o, mach_ref,
                precond_mode,
            )
            F_tilde_common_o = flux_o * adj_mag_o[..., None] * side_factor_o[:, None, None]

            a0 = adjrow_o[..., 0]
            a1 = adjrow_o[..., 1]
            a2 = adjrow_o[..., 2]
            F_phys_o = euler_physical_flux_gpu(Q_o.reshape(nO * n_fp, 5)).reshape(nO, n_fp, 3, 5)
            F_tilde_own_o = (
                a0[..., None] * F_phys_o[..., 0, :]
                + a1[..., None] * F_phys_o[..., 1, :]
                + a2[..., None] * F_phys_o[..., 2, :]
            )

            jump_owner = F_tilde_common_o - F_tilde_own_o

            # 真实 bug 修复（V2.0 专家组盲审第四轮，2026-08-28）：分配
            # 改用 dist_fp_of_sp/dist_axis_coord_of_sp gather，不再用
            # `ff.g_left[idx_o]` 按面索引去索引这个长度仅 n1d 的数组
            # （会在真实 GPU 上 IndexError），见
            # gpu_inviscid_volume.py::distribute_face_correction_to_sps
            # 文档的完整推导。
            #
            # 真实 bug 修复（2026-09-03）：这里必须用上面已经算好的
            # `oax_safe`（native 面 clip 到 0），不能用原始 `oax`——
            # `dist_fp_of_sp`/`dist_axis_coord_of_sp` 同样只有 3 个轴，
            # 原始 `oax`（native 面上是 excluded_vertex，可达 3）会在
            # 这里 `IndexError`，即便下面 `_native_or_collapsed_contrib`
            # 最终会丢弃 collapsed 分支的结果（同一类"提前无条件求值"
            # 问题，与本文件其余同类修复同一根因）。
            contrib_o_collapsed = distribute_face_correction_to_sps(
                cp, jump_owner, oax_safe, oside, ff.dist_fp_of_sp, ff.dist_axis_coord_of_sp,
                ff.g_left, ff.g_right,
            )
            contrib_o = _native_or_collapsed_contrib(
                cp, is_native_o, oc_code_o, ff.lift_native, ff.true_area_weight[idx_o],
                jump_owner, contrib_o_collapsed,
            )
            contrib_o = contrib_o / det_jacs[oc][..., None]
            _scatter_add_to_correction(correction, -contrib_o, oc, n_cells, n_sps)

        # ── neighbor-primary 贡献块（仅内部面）──
        mask_n = neighbor_primary & (~is_bnd)
        if bool(cp.any(mask_n)):
            idx_n = face_idx[mask_n]
            nc = ff.neighbor_cell[idx_n]
            nax = ff.neighbor_axis[idx_n]
            nside = ff.neighbor_side[idx_n]

            # #1（2026-08-28）：见上方 owner-primary 块同名注释，同一处修复。
            compact_cell_type = getattr(ff, 'compact_cell_type', None)
            if compact_cell_type is not None:
                celltype_n = compact_cell_type[nc]
            else:
                celltype_n = cp.where(nc < n_prism, 0, 1)
            nside_idx = cp.where(nside < 0, 0, 1)

            # native 四面体（路径C）GPU 移植（2026-09-02）：见上方
            # owner-primary 块同名注释，同一处修复。
            nc_code_n = ff.neighbor_cube_face[idx_n]
            is_native_n = nc_code_n >= 6

            # 真实 bug 修复（2026-09-03）：见上方 owner-primary 块同名
            # 注释，同一处修复——`nax` 对 native 面同样存的是复用的
            # excluded_vertex（0~3），不能无条件拿去 gather 只有 3 个
            # 轴的 `boundary_extrap`。
            nax_safe = cp.where(is_native_n, 0, nax)
            E_n_collapsed = ff.boundary_extrap[celltype_n, nax_safe, nside_idx]
            E_n = _native_self_extrap(cp, is_native_n, nc_code_n, ff.boundary_extrap_native, E_n_collapsed)
            Q_n_native = cp.matmul(E_n, Q_gpu[nc])  # (nN,n_fp,5)

            Q_o_at_n = _extrap_q_to_fp(cp, ff.owner_src0_mat[idx_n], ff.owner_src0_cell[idx_n], Q_gpu)
            Q_o_at_n = _add_q_src1_to_fp(
                cp, Q_o_at_n, ff.owner_src1_idx[idx_n], ff.owner_src1_cell, ff.owner_src1_mat, Q_gpu,
            )

            # 混合拆分面（B-8）：neighbor 侧对称处理——边界半区对侧状态
            # 逐 FP 取配对面幽灵态（Q_ghost_gpu 为逐 FP (n_faces,n_fp,5)）。
            mp_n = ff.mixed_ow_partner[idx_n]
            mixed_sel_n = (mp_n[:, None] >= 0) & ff.mixed_ow_mask[idx_n]
            Q_ghost_partner_n = Q_ghost_gpu[cp.maximum(mp_n, 0)]  # (nN, n_fp, 5)
            Q_o_at_n = cp.where(
                mixed_sel_n[..., None],
                Q_ghost_partner_n,
                Q_o_at_n,
            )

            n_fp = Q_n_native.shape[1]
            nN = Q_n_native.shape[0]

            # neighbor 视角外法向恒为 -true_normal（见 CPU kernel 同名注释）
            side_factor_n = _native_aware_side_factor(cp, is_native_n, nside)
            adjrow_n = ff.neighbor_adj_row_exact[idx_n]
            tn_neg = -ff.true_normal[idx_n]
            direction_n, adj_mag_n = _ausm_direction_with_fallback(
                cp, adjrow_n, side_factor_n, tn_neg,
            )

            flux_n = _ausm_up_flux_batch_gpu(
                Q_n_native.reshape(nN, n_fp, 5), Q_o_at_n.reshape(nN, n_fp, 5), direction_n, mach_ref,
                precond_mode,
            )
            F_tilde_common_n = flux_n * adj_mag_n[..., None] * side_factor_n[:, None, None]

            a0n = adjrow_n[..., 0]
            a1n = adjrow_n[..., 1]
            a2n = adjrow_n[..., 2]
            F_phys_n = euler_physical_flux_gpu(Q_n_native.reshape(nN * n_fp, 5)).reshape(nN, n_fp, 3, 5)
            F_tilde_own_n = (
                a0n[..., None] * F_phys_n[..., 0, :]
                + a1n[..., None] * F_phys_n[..., 1, :]
                + a2n[..., None] * F_phys_n[..., 2, :]
            )

            jump_neighbor = F_tilde_common_n - F_tilde_own_n

            # 见上方 owner-primary 块同名注释，同一处修复——用
            # `nax_safe`（native 面 clip 到 0），不能用原始 `nax`。
            contrib_n_collapsed = distribute_face_correction_to_sps(
                cp, jump_neighbor, nax_safe, nside, ff.dist_fp_of_sp, ff.dist_axis_coord_of_sp,
                ff.g_left, ff.g_right,
            )
            contrib_n = _native_or_collapsed_contrib(
                cp, is_native_n, nc_code_n, ff.lift_native, ff.true_area_weight[idx_n],
                jump_neighbor, contrib_n_collapsed,
            )
            contrib_n = contrib_n / det_jacs[nc][..., None]
            _scatter_add_to_correction(correction, -contrib_n, nc, n_cells, n_sps)

    return correction


def _ausm_up_flux_batch_gpu(Q_L, Q_R, normal, mach_ref, precond_mode):
    """GPU 批量 AUSM+up 通量计算（CuPy 向量化版本，含 Weiss-Smith 低马赫
    数预处理）。与 kernels.py::compute_ausm_up_flux 逐字对应，理由/推导
    见该函数文档，这里不重复；两处必须同步修改（该文件模块文档要求
    "逐字对应"）。

    真实 bug 修复（问题清单 #5，2026-09-02，用真实含多源/混合拆分面的
    合成网格 + numpy-as-cupy 替身 + 完整 `GPUFRSolver.step()` 首次真正
    走到这里才发现——`test_gpu_solver_order_continuation.py`"发现但
    本次未修复"一节记录过这个缺口，本次专项排查修复）：`normal` 形参
    此前文档标注/实现都当作 `(N, 3)`（逐面一个法向，`nx=normal[...,
    0:1]` 故意保留末尾长度 1 的维度，为的是广播到 `uL` 等 `(N,n_fp)`
    形状——对应 CPU kernel 里"同一面内全部 FP 共用同一个法向"这个
    从未真正成立的假设）——但两个真实调用点（本文件
    `_compute_interface_correction_gpu` 的 owner/neighbor 两个分支）
    传入的 `direction_o`/`direction_n` 来自 `_ausm_direction_with_
    fallback`，形状恒为 `(N, n_fp, 3)`（逐 FP 各自独立的方向，因为
    对齐安全阀是逐 FP 用该 FP 自己的 `adjrow`/`true_normal_ref` 判定
    的，不是全面共享同一个值——与 CPU numba kernel 逐 FP 独立调用
    `compute_ausm_up_flux(qL, qR, normal_at_this_fp, ...)` 完全对应）。
    `nx=normal[...,0:1]` 对 `(N,n_fp,3)` 输入会产出 `(N,n_fp,1)`，
    与 `uL`（`(N,n_fp)`）相乘时 NumPy/CuPy 广播规则按从右往左对齐，
    `uL` 的 `n_fp` 维度会被错误地对上 `nx` 的长度-1 维度、`uL` 隐式
    补出的前导维度又被要求匹配 `nx` 的 `n_fp` 维——只有 `N==n_fp`（本
    次合成测试网格的巧合，两者都恰好是 4）时这个错误广播才会"成功"
    但产出一个多出一维、内容完全错误的结果，`N!=n_fp` 的一般网格上
    会在这里直接 `ValueError`（这正是该测试文件此前报告的"P1/P2 直接
    构造时同样复现"的崩溃点，只是那次调查停在了更下游的
    `distribute_face_correction_to_sps`）。修复：`nx=normal[...,0]`
    （不保留末尾维度）——这样在真实的 `(N,n_fp,3)` 输入下 `nx` 形状
    恰好是 `(N,n_fp)`，与 `uL` 等逐 FP 量逐元素精确匹配，不再依赖任何
    隐式广播。

    Args:
        Q_L, Q_R: (N, n_fp, 5) 左右状态
        normal: (N, n_fp, 3) 单位法向量——逐 FP 各自独立（不是逐面共享
            同一个值，见上方"真实 bug 修复"说明）
        precond_mode: AUSM+up 预处理声速在通量内部的作用域（
            kernels.py 的 PRECOND_PHYSICAL/PRECOND_PRESSURE_PHYSICAL/
            PRECOND_LEGACY，语义与 CPU 端逐字对应，推导见该文件模块级
            常量上方的长注释）。GPU 侧不存在 numba 磁盘缓存冻结全局量的
            问题，但仍然按实参传入，以保证 CPU/GPU 交叉一致性测试能对
            同一档逐项比对。
        mach_ref: 参考（自由来流）马赫数，见 kernels.py::
            compute_ausm_up_flux 文档

    Returns:
        flux: (N, n_fp, 5) 数值通量
    """
    cp = get_cupy()
    gamma = 1.4

    rhoL = cp.maximum(Q_L[..., 0], 1e-6)
    uL, vL, wL = Q_L[..., 1], Q_L[..., 2], Q_L[..., 3]
    pL = cp.maximum(Q_L[..., 4], 10.0)

    rhoR = cp.maximum(Q_R[..., 0], 1e-6)
    uR, vR, wR = Q_R[..., 1], Q_R[..., 2], Q_R[..., 3]
    pR = cp.maximum(Q_R[..., 4], 10.0)

    nx = normal[..., 0]
    ny = normal[..., 1]
    nz = normal[..., 2]

    unL = uL * nx + vL * ny + wL * nz
    unR = uR * nx + vR * ny + wR * nz

    aL = cp.sqrt(cp.maximum(gamma * pL / rhoL, 1e-10))
    aR = cp.sqrt(cp.maximum(gamma * pR / rhoR, 1e-10))

    a_half = 0.5 * (aL + aR)
    rho_half = 0.5 * (rhoL + rhoR)
    Mbar2 = (unL**2 + unR**2) / (2.0 * a_half**2)

    M0_sq = cp.minimum(1.0, cp.maximum(Mbar2, mach_ref**2))
    sqrt_M0_sq = cp.sqrt(M0_sq)
    fa = sqrt_M0_sq * (2.0 - sqrt_M0_sq)
    fa = cp.maximum(fa, 1e-6)

    # M4±/P5± 耗散系数——真实 bug 修复（V2.0 专家组盲审第四次评审，
    # 2026-08-28，#12），与 kernels.py::compute_ausm_up_flux 逐字对应，
    # 完整推导/文献交叉核实见该文件同名注释，这里不重复。
    beta_mass = 1.0 / 8.0
    alpha_pressure = 3.0 / 16.0 * (-4.0 + 5.0 * fa * fa)

    # Weiss-Smith 预处理声速（与 kernels.py::compute_ausm_up_flux 的
    # _WEISS_SMITH_K=1.1 同一个安全裕度常数、同一套 beta2 公式）。
    beta2 = cp.minimum(1.0, cp.maximum(cp.maximum(Mbar2, 1.1 * mach_ref**2), 1e-10))
    sqrt_beta2 = cp.sqrt(beta2)

    # 预处理声速的作用域按 precond_mode 分派，与 kernels.py::
    # compute_ausm_up_flux 的 s_mass/s_pres 逐字对应（PRECOND_PHYSICAL
    # 是默认值，此时两个因子都是 1.0、本函数精确退化为标准 AUSM+up）。
    if precond_mode == PRECOND_PHYSICAL:
        s_mass = 1.0
        s_pres = 1.0
    elif precond_mode == PRECOND_PRESSURE_PHYSICAL:
        s_mass = sqrt_beta2
        s_pres = 1.0
    elif precond_mode == PRECOND_LEGACY:
        s_mass = sqrt_beta2
        s_pres = sqrt_beta2
    else:
        raise ValueError(f'未知 precond_mode: {precond_mode!r}')

    aL_m = s_mass * aL
    aR_m = s_mass * aR
    a_half_m = s_mass * a_half
    a_half_pr = s_pres * a_half

    M_L = unL / cp.maximum(aL_m, 1e-10)
    M_R = unR / cp.maximum(aR_m, 1e-10)
    M_L_pr = unL / cp.maximum(s_pres * aL, 1e-10)
    M_R_pr = unR / cp.maximum(s_pres * aR, 1e-10)

    # M+ / M-
    abs_ML = cp.abs(M_L)
    abs_MR = cp.abs(M_R)
    Mp_L = cp.where(
        abs_ML >= 1.0,
        0.5 * (M_L + abs_ML),
        0.25 * (M_L + 1.0)**2 + beta_mass * (M_L**2 - 1.0)**2,
    )
    Mm_R = cp.where(
        abs_MR >= 1.0,
        0.5 * (M_R - abs_MR),
        -0.25 * (M_R - 1.0)**2 - beta_mass * (M_R**2 - 1.0)**2,
    )
    M_half = Mp_L + Mm_R

    # Mp 压力扩散
    Kp = 0.25
    sigma_p = 1.0
    one_minus_sigma = cp.maximum(1.0 - sigma_p * Mbar2, 0.0)
    Mp = -(Kp / fa) * one_minus_sigma * (pR - pL) / (rho_half * a_half_m**2)
    mass_flux = 0.5 * (rhoL * aL_m + rhoR * aR_m) * (M_half + Mp)

    # P+ / P-（用压力分裂专属的马赫数 M_*_pr，见上方 s_pres 分派）
    abs_ML_pr = cp.abs(M_L_pr)
    abs_MR_pr = cp.abs(M_R_pr)
    Pp_L = cp.where(
        abs_ML_pr >= 1.0,
        0.5 * (1.0 + cp.sign(M_L_pr)),
        0.25 * ((M_L_pr + 1.0)**2 * (2.0 - M_L_pr)
                + alpha_pressure * M_L_pr * (M_L_pr**2 - 1.0)**2),
    )
    Pm_R = cp.where(
        abs_MR_pr >= 1.0,
        0.5 * (1.0 - cp.sign(M_R_pr)),
        0.25 * ((M_R_pr - 1.0)**2 * (2.0 + M_R_pr)
                - alpha_pressure * M_R_pr * (M_R_pr**2 - 1.0)**2),
    )

    # pu 速度扩散
    Ku = 0.75
    p_half = (Pp_L * pL + Pm_R * pR
              - Ku * Pp_L * Pm_R * (rhoL + rhoR) * fa * a_half_pr * (unR - unL))

    # 上风通量
    upwind_L = (mass_flux >= 0.0)
    u_up = cp.where(upwind_L, uL, uR)
    v_up = cp.where(upwind_L, vL, vR)
    w_up = cp.where(upwind_L, wL, wR)

    hL = gamma / (gamma - 1.0) * pL / rhoL + 0.5 * (uL**2 + vL**2 + wL**2)
    hR = gamma / (gamma - 1.0) * pR / rhoR + 0.5 * (uR**2 + vR**2 + wR**2)
    h_up = cp.where(upwind_L, hL, hR)

    flux = cp.stack([
        mass_flux,
        mass_flux * u_up + p_half * nx,
        mass_flux * v_up + p_half * ny,
        mass_flux * w_up + p_half * nz,
        mass_flux * h_up,
    ], axis=-1)

    return flux


def _scatter_add_to_correction(correction, contrib, cell_indices, n_cells, n_sps):
    """将面的校正贡献写入全局 correction 数组。

    同色面无 owner_cell 冲突，但不同面可能写同一个 cell（虽然同色面之间不会），
    所以这里使用 CuPy 的 scatter add 模式。

    由于同色面保证无冲突，可以直接用索引赋值。
    """
    cp = get_cupy()
    # 同色面无冲突，直接用 advanced indexing 写入
    # contrib: (n_color_faces, n_sps, 5), cell_indices: (n_color_faces,)
    # 需要处理多个面写同一个 cell 的情况（虽然同色面不冲突，但保险起见用 add）
    cp.scatter_add(correction, (cell_indices, slice(None), slice(None)), contrib)
