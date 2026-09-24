"""AutoFlowCFD V2.0 - 体积项与残差入口(GPU)

从 `src/autoflowcfd/core/gpu/residual/gpu_inviscid.py`(原 710 行)拆出(2026-09-24, 项目"单文件不超 500 行"规范)。**纯搬家, 逻辑未改**。
"""

import numpy as np


from autoflowcfd.core.gpu import get_cupy

from autoflowcfd.core.gpu.residual.gpu_flux import conserved_to_primitive_gpu

from autoflowcfd.core.gpu.residual.gpu_inviscid_volume import (
    prepare_mesh_data as _prepare_mesh_data,
    prepare_ops_data as _prepare_ops_data,
    compute_volume_term_gpu as _compute_volume_term_gpu_impl,
)
from .interface import _compute_interface_correction_gpu


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
        n_cells, n_sps, device_id, mach_ref, precond_mode,
    )

    residual = residual + correction

    # 机制3（异常残差抑制）已于 2026-09-19 删除，完整依据见
    # `fr_residual/inviscid.py` 同一处。**对 GPU 路径还额外省掉一次
    # `GPU -> CPU -> GPU` 往返**：那一步是为了复用 CPU 的 numba 实现而
    # 把整个残差场与守恒场都拷回主机再拷回来。
    return cp.asnumpy(residual) if input_is_numpy else residual


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
