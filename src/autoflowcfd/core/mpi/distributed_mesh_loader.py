"""
AutoFlowCFD V2.0 - 完全分布式网格加载

优化内存使用：只有 root rank 加载完整网格，然后通过 MPI 将每个 rank 的
局部数据分发出去。非 root rank 不需要持有完整网格数据。

流程:
1. Root rank 加载完整网格 → 构建面连接关系 → 分区
2. Root rank 提取每个 rank 的局部数据（mesh geometry + face connectivity）
3. Root rank 通过 MPI 发送各 rank 的局部数据
4. 非 root rank 接收并构建局部网格对象

关键设计:
- 非 root rank 不再调用 load_mesh_for_solver()
- 局部网格使用重映射的 cell 索引（从 0 开始）
- 面连接关系的 owner_cell/neighbor_cell 使用局部索引
- halo cell 信息通过 partition 的 recv_lists 获取
"""

import copy
import numpy as np
from typing import Optional, Dict, Tuple

from loguru import logger

from autoflowcfd.core.mpi import get_comm, get_mpi, mpi_available, get_rank, get_size
from autoflowcfd.core.time_integration.base import TimeIntegrationScheme


class PrecompactedMeshData:
    """真正的"完全分布式网格加载"（2026-09-02）——mesh-like 鸭子类型
    对象，持有的全部几何数组都**已经**是 local+halo 压缩索引空间大小
    （不是全局 `n_global_cells` 大小），供 `DistributedMeshAdapter` 直接
    使用，不需要再做一次 `[compact_global_ids]` 切片。

    与"传统模式"（`local_mesh` 是完整全局网格，`DistributedMeshAdapter`
    自己按 `compact_global_ids` 从全局数组里切片，见该类文档"真实 bug
    修复"一节）的关键区别：本对象由 **root rank** 预先算好（root 手握
    完整全局网格+完整 ops，能正确调用 `build_distributed_flat_face` 拿到
    每个 rank 的 `compact_global_ids`，见 `build_fully_distributed_rank_
    package` 文档），非 root rank 只接收这个已经切好的紧凑包，从未持有
    过完整全局网格——这才是"完全分布式加载"名副其实的内存优化（大网格
    的主要内存开销是 `jacobians`/`sps_coords` 这类与 n_cells*n_sps 同
    量级的数组，本设计让每个非 root rank 只持有自己 local+halo 那一份，
    不是全局那一份）。

    face_connectivity 故意留空（`None`）；`face_flux_points` 是一个
    非 None 的哨兵值，不是真实数据（见下方"哨兵值说明"）：本次范围只
    覆盖"均值流残差计算"这一条主线（`DistributedMeshAdapter`/
    `distributed_compute_inviscid_residual`/`_viscous_residual` 全部
    只通过 `dist_fc`（压缩面几何，由 root 预先构建、随本对象一起下发）
    访问面数据，从未读取 `mesh.face_connectivity`/`mesh.face_flux_points`
    本身——这两个属性只在 `boundary_ghost_provider`/湍流模型壁面距离
    这类需要"完整全局面拓扑/节点坐标"的辅助路径里才用得到，这两类目前
    仍需要真实全局网格（湍流模型在这个模式下未接入，见
    `build_fully_distributed_rank_package` 文档"范围边界"一节）。

    哨兵值说明：`compute_inviscid_residual_fr`/`compute_viscous_
    residual` 入口处各有一个 `mesh.face_connectivity is None or mesh.
    face_flux_points is None` 的"存在性"检查（不是内容检查——已核实
    这两个函数被 `DistributedMeshAdapter` 包裹调用时，真正的面几何数据
    全部来自显式传入的 `flat_face_override`/`dist_fc`，从未真正读取
    `mesh.face_flux_points` 的内容，"传统模式"下这个检查能通过纯粹是
    因为 `local_mesh` 恰好是带着真实 `face_flux_points` 的完整全局
    网格，不代表这个属性的内容被使用）——`face_connectivity` 走
    `DistributedMeshAdapter.face_connectivity` 属性单独返回
    `self.dist_fc`（非 None，真实数据），只有 `face_flux_points` 这个
    检查还需要一个非 None 值才能通过；这里给它一个明确标注"仅哨兵、
    非真实数据"的占位字符串，不给 `None`（会让上面这个良性检查失败）
    也不给看起来像真实数据的对象（避免误导未来的读者以为这里有真实
    面通量点几何）。
    """

    _FACE_FLUX_POINTS_SENTINEL = "PrecompactedMeshData: no real face_flux_points, see class docstring"

    def __init__(
        self, det_jacs, inv_jacs, det_jacs_fine, inv_jacs_fine,
        cell_volumes, sps_coords, cell_types,
        n_prism_cells, n_points_1d, n_sps_per_cell, n_sps_per_cell_fine,
    ):
        self.n_cells = det_jacs.shape[0]  # compact（local+halo）大小
        self.n_prism_cells = n_prism_cells
        self.n_points_1d = n_points_1d
        self.n_sps_per_cell = n_sps_per_cell
        self.n_sps_per_cell_fine = n_sps_per_cell_fine

        self.jacobians = {'det_jacs': det_jacs, 'inv_jacs': inv_jacs}
        self.jacobians_fine = (
            {'det_jacs': det_jacs_fine, 'inv_jacs': inv_jacs_fine}
            if det_jacs_fine is not None else None
        )
        self.cell_volumes = cell_volumes
        self.sps_coords = sps_coords
        self.cell_types = cell_types

        # 见类文档"哨兵值说明"一节。
        self.face_connectivity = None
        self.face_flux_points = self._FACE_FLUX_POINTS_SENTINEL


def build_fully_distributed_rank_package(
    mesh, ops, face_connectivity, cell_partition: np.ndarray,
    rank: int, n_ranks: int,
    boundary_ghost_provider_global,
    freestream: dict, mu_molecular: float, mach_ref: float,
    order: int, enable_viscous: bool,
    turb_model_name: str = "NONE",
    wall_node_indices: Optional[np.ndarray] = None,
    h_max_global: Optional[np.ndarray] = None,
    h_wn_global: Optional[np.ndarray] = None,
    turbulence_intensity: float = 0.01,
    viscosity_ratio: float = 5.0,
    time_scheme=None,
    dual_time_inner_iter: int = 20,
) -> dict:
    """Root rank 专用：为指定 rank 算好它需要的全部紧凑数据（不需要该
    rank 自己持有完整全局网格）。

    纯函数（不做任何 MPI 通信）——`build_distributed_partition`/
    `build_distributed_flat_face` 本身都是纯函数（只依赖传入的完整
    `mesh`/`ops`/`face_connectivity`/`cell_partition`，不读任何 MPI
    状态），root 可以对 `rank=0..n_ranks-1` 依次调用本函数、分别取得
    每个 rank 的紧凑包，再各自 pickle 发送出去——通信本身（Send/Recv）
    留给调用方（`distributed_mesh_load_v2`），本函数只管"算"。

    `get_flat_face_geometry(mesh, ops)`（`build_distributed_flat_face`
    内部调用）按 `mesh.face_flux_points` 对象身份缓存（见
    `face_kernels.py::get_flat_face_geometry` 文档），root 对同一个
    `mesh` 反复调用本函数（每个 rank 一次）时这个全局面几何只会真正
    计算一次，不会因为循环 n_ranks 次而重复付出这个昂贵的 Newton
    迭代开销。

    范围边界（本次实现覆盖的范围，如实标注不覆盖的部分）：
    - 覆盖：inviscid/viscous 均值流残差计算所需的全部数据
      （`DistributedMeshAdapter` 需要的紧凑几何 + `boundary_ghost_
      provider`，两者共同决定 `distributed_compute_inviscid_residual`/
      `_viscous_residual` 能否正确工作）；SST/DDES/IDDES 湍流模型
      （2026-09-02 补齐——`compute_distributed_wall_distance`/
      `compute_h_max_and_h_wn` 都只需要**完整全局网格**（root 有，
      非 root rank 不需要），root 侧算好、切成 compact 索引空间随
      紧凑包一起发送即可，此前"范围不覆盖"的理由本身就不成立，只是
      当时没有一并做）；WMLES/LES（同日续接）——WMLES 复用与 SST 完全
      同一套 `wall_distance_compact`（y+ 计算需要），LES（WALE）不需要
      root 预计算任何额外几何量，纯代数每步现算。DUAL_TIME
      （2026-09-02 续接）——`time_scheme`/`dual_time_inner_iter` 只是
      纯配置量（不依赖 root 的完整网格），随包一起透传给
      `DistributedFRSolver.from_fully_distributed_package`（该方法早已
      读取这两个字段，见其文档，此前只是本函数从未真正把它们塞进
      package，`--fully-distributed --time-method dual-time` 因此
      恒被 CLI 拒绝——不是设计上不支持，只是没人接上这两个参数）。
      Order Continuation（2026-09-02 同日续接）已通过
      `redistribute_fully_distributed_for_new_order`（阶数切换时 root
      重新调用本函数）接入，不再是"不覆盖"的特性。

    Args:
        mesh, ops, face_connectivity: root rank 持有的完整全局网格/
            算子/面连接关系
        cell_partition: (n_global_cells,) 全局分区数组
        rank, n_ranks: 目标 rank 编号与总数
        boundary_ghost_provider_global: root 用完整全局网格构建好的
            **一份**边界幽灵态提供者（只需要构建一次，供全部 rank
            共享底层配置，各自只需要重新切片 `group_code`，见下方
            "边界条件"一节）
        freestream, mu_molecular, mach_ref, order, enable_viscous:
            纯配置量（标量/小 dict），直接透传
        turb_model_name: "NONE"/"SST"/"DDES"/"IDDES"/"WMLES"/"LES"
            （大写），决定是否需要计算 wall_distance/h_max/h_wn
        wall_node_indices: WALL 边界节点索引（全局节点编号），root 只
            需要算一次（不依赖 rank），由调用方（`distributed_mesh_
            load_v2`）传入
        h_max_global, h_wn_global: (n_global_cells,) IDDES 专用，root
            只需要对同一个 mesh 算一次（`compute_h_max_and_h_wn` 纯
            几何、与 rank 无关），由调用方缓存后传入，避免每个 rank
            重复计算

    Returns:
        dict：{'partition', 'dist_fc', 'precompacted_mesh',
        'boundary_ghost_provider', 'freestream', 'mu_molecular',
        'mach_ref', 'order', 'enable_viscous', 'turb_model_name',
        'wall_distance_compact', 'iddes_h_max_compact',
        'iddes_h_wn_compact', 'time_scheme', 'dual_time_inner_iter'}，
        全部可安全 pickle。
    """
    # fail-fast 护栏（真实 bug 修复，2026-09-02）：拒绝任何拼写错误/
    # 未知的 turb_model_name，而不是静默产出一个 `wall_distance_
    # compact=None`、但 `'turb_model_name'` 仍标记为该值的包（会被原样
    # pickle 发给非 root rank，物理不完整地跑出结果而不报错）。在 root
    # 侧构造期就拒绝，比等 MPI 分发之后再报错更早、更便宜。
    if turb_model_name not in ("NONE", "SST", "DDES", "IDDES", "WMLES", "LES"):
        raise NotImplementedError(
            f"build_fully_distributed_rank_package: 完全分布式加载模式"
            f"目前只支持 turb_model_name='NONE'/'SST'/'DDES'/'IDDES'/"
            f"'WMLES'/'LES'，收到的是 '{turb_model_name}'。"
        )
    if time_scheme is None:
        time_scheme = TimeIntegrationScheme.SSP_RK3

    from autoflowcfd.core.mpi.partition import build_distributed_partition
    from autoflowcfd.core.mpi.distributed_flat_face import build_distributed_flat_face
    from autoflowcfd.core.mpi.distributed_turbulence import compute_distributed_wall_distance

    partition = build_distributed_partition(face_connectivity, cell_partition, rank=rank, n_ranks=n_ranks)
    dist_fc = build_distributed_flat_face(mesh, ops, partition, cell_partition=cell_partition)
    compact_ids = dist_fc.compact_global_ids

    n_sps = mesh.n_sps_per_cell
    det_jacs = mesh.jacobians['det_jacs'].reshape(mesh.n_cells, n_sps)[compact_ids].copy()
    inv_jacs = mesh.jacobians['inv_jacs'].reshape(mesh.n_cells, n_sps, 3, 3)[compact_ids].copy()

    det_jacs_fine = inv_jacs_fine = None
    n_sps_fine = getattr(mesh, 'n_sps_per_cell_fine', None)
    if getattr(mesh, 'jacobians_fine', None) is not None:
        det_jacs_fine = mesh.jacobians_fine['det_jacs'].reshape(mesh.n_cells, n_sps_fine)[compact_ids].copy()
        inv_jacs_fine = mesh.jacobians_fine['inv_jacs'].reshape(mesh.n_cells, n_sps_fine, 3, 3)[compact_ids].copy()

    cell_volumes = mesh.cell_volumes[compact_ids].copy() if getattr(mesh, 'cell_volumes', None) is not None else None
    sps_coords = mesh.sps_coords[compact_ids].copy() if getattr(mesh, 'sps_coords', None) is not None else None
    cell_types = (
        mesh.cell_types[compact_ids].copy()
        if getattr(mesh, 'cell_types', None) is not None else None
    )

    precompacted_mesh = PrecompactedMeshData(
        det_jacs=det_jacs, inv_jacs=inv_jacs,
        det_jacs_fine=det_jacs_fine, inv_jacs_fine=inv_jacs_fine,
        cell_volumes=cell_volumes, sps_coords=sps_coords, cell_types=cell_types,
        n_prism_cells=dist_fc.base_flat.n_prism,
        n_points_1d=mesh.n_points_1d, n_sps_per_cell=n_sps, n_sps_per_cell_fine=n_sps_fine,
    )

    # 边界条件：`boundary_ghost_provider_global` 只需要构建一次（root
    # 用完整全局网格+完整边界组信息构建，见调用方 distributed_mesh_
    # load_v2 文档），这里只需要把它的 `group_code`（(n_faces_global,)
    # 数组）重新切成本 rank 的 `partition.local_faces`（全局面编号）——
    # 与 `distributed_solver.py::local_solver`/`gpu_distributed.py` 里
    # 已经验证过的同一个"group_code 重映射"修复完全同一个模式，唯一
    # 区别是这里在 root 侧、构造时就切好，而不是非 root rank 自己收到
    # 完整 provider 后再切（那样仍然需要先有完整 provider 才能切，
    # provider 本身不大，但如果 code_to_config/其余标量属性之外还有
    # 别的大数组字段就会浪费——目前只有 group_code 是数组字段）。
    boundary_ghost_provider = None
    if boundary_ghost_provider_global is not None:
        boundary_ghost_provider = copy.copy(boundary_ghost_provider_global)
        if hasattr(boundary_ghost_provider, 'group_code'):
            boundary_ghost_provider.group_code = (
                boundary_ghost_provider_global.group_code[partition.local_faces]
            )

    # SST/DDES/IDDES/WMLES（2026-09-02，WMLES 同日续接——y+ 计算同样
    # 需要 wall_distance）：wall_distance/h_max/h_wn 都只需要 root 手上
    # 的**完整**全局网格（`compute_distributed_wall_distance`/
    # `compute_h_max_and_h_wn` 的输入都是 `mesh`，不是某个 rank 的局部
    # 数据），root 算好后按本 rank 的 compact_global_ids 切片即可，与
    # 其余紧凑几何数据同一个模式——不需要新的跨 rank 几何交换。LES
    # （WALE）不需要 wall_distance，不在这个分支里。
    wall_distance_compact = None
    iddes_h_max_compact = None
    iddes_h_wn_compact = None
    if turb_model_name in ("SST", "DDES", "IDDES", "WMLES"):
        wall_distance_compact = compute_distributed_wall_distance(
            partition, dist_fc, mesh, wall_node_indices,
        )
        if turb_model_name == "IDDES":
            if h_max_global is None or h_wn_global is None:
                raise RuntimeError(
                    "build_fully_distributed_rank_package: turb_model_name='IDDES' "
                    "需要调用方提供 h_max_global/h_wn_global（compute_h_max_and_h_wn(mesh) "
                    "的输出，只需要对同一个 mesh 算一次，见本函数 Args 文档）。"
                )
            iddes_h_max_compact = h_max_global[compact_ids].copy()
            iddes_h_wn_compact = h_wn_global[compact_ids].copy()
        elif turb_model_name == "DDES":
            # DDES（2026-09-02 补齐，与 IDDES 同一处理）：`apply_to_sst_
            # model` 现在优先用各向异性感知的 max_edge 网格尺度（见
            # des.py 对应方法文档），需要 h_max_global——不需要 h_wn
            # （那是 IDDES 专属的近壁法向间距量）。
            if h_max_global is None:
                raise RuntimeError(
                    "build_fully_distributed_rank_package: turb_model_name='DDES' "
                    "需要调用方提供 h_max_global（compute_h_max_and_h_wn(mesh) 的第一个"
                    "返回值，只需要对同一个 mesh 算一次，见本函数 Args 文档）。"
                )
            iddes_h_max_compact = h_max_global[compact_ids].copy()

    return {
        'partition': partition,
        'dist_fc': dist_fc,
        'precompacted_mesh': precompacted_mesh,
        'boundary_ghost_provider': boundary_ghost_provider,
        'freestream': freestream,
        'mu_molecular': mu_molecular,
        'mach_ref': mach_ref,
        'order': order,
        'enable_viscous': enable_viscous,
        'turb_model_name': turb_model_name,
        'wall_distance_compact': wall_distance_compact,
        'iddes_h_max_compact': iddes_h_max_compact,
        'iddes_h_wn_compact': iddes_h_wn_compact,
        'turbulence_intensity': turbulence_intensity,
        'viscosity_ratio': viscosity_ratio,
        'time_scheme': time_scheme,
        'dual_time_inner_iter': dual_time_inner_iter,
    }


def distributed_mesh_load_v2(
    input_file: str,
    order: int,
    surface_mesh: Optional[str],
    n_ranks: int,
    freestream: dict,
    mu_molecular: float,
    mach_ref: float,
    enable_viscous: bool = True,
    bc_overrides: Optional[dict] = None,
    skip_quality_check: bool = False,
    turb_model_name: str = "NONE",
    turbulence_intensity: float = 0.01,
    viscosity_ratio: float = 5.0,
    time_scheme=None,
    dual_time_inner_iter: int = 20,
):
    """真正的完全分布式网格加载（2026-09-02）——只有 root rank 加载
    完整网格并对每个 rank 分别调用 `build_fully_distributed_rank_
    package`，其余 rank 只通过 MPI 接收自己的那一份紧凑包，从未持有
    完整全局网格。取代此前从未真正跑通过的 `distributed_mesh_load`
    （见该函数/`build_local_mesh_from_data` 文档"注意"一节——那条路径
    产出的 `local_mesh` 缺 `face_connectivity`/`face_flux_points`，
    `DistributedMeshAdapter` 需要完整全局网格重新切片 jacobians，两个
    前提在那条路径下都不成立，构造期必然出错，此前从未被 CLI 实际
    调用过，见 `solve_steady_command.py` 对应注释）。

    Order Continuation 支持（2026-09-02 续接，见 core/mpi/
    distributed_order_continuation.py 模块文档"完全分布式加载"一节）：
    返回值从单一的 `my_package` 改为 `(my_package, root_context)` 二元
    组——`root_context` 只有 root rank 非 None（其余 rank 恒为
    None，它们从未持有、也不需要持有完整全局网格），持有本函数内部
    算好的 `mesh`/`ops`/`fc`/`cell_partition`/`boundary_ghost_provider_
    global`/`wall_node_indices`/`h_max_global`/`h_wn_global` 等，供
    `redistribute_fully_distributed_for_new_order` 在阶数切换时复用
    （重新构造完整网格/重新分区都是不必要的重复开销——这些量本身除了
    `boundary_ghost_provider_global`（阶数相关的 FP 几何）之外全部与
    阶数无关）。调用方（CLI）需要把 `root_context` 原样传给
    `DistributedFRSolver.from_fully_distributed_package(package,
    n_ranks=n_ranks, root_context=root_context)`。

    Args:
        turb_model_name: "NONE"/"SST"/"DDES"/"IDDES"/"WMLES"/"LES"
            （大写）。root 用真实全局网格算 wall_distance/h_max/h_wn
            （`compute_distributed_wall_distance`/`compute_h_max_and_
            h_wn` 都只需要完整网格，root 有，只算一次，见下方调用点；
            WMLES 同样需要 wall_distance——y+ 计算依赖它，LES 不需要
            任何 root 预计算的几何量），非 root rank 从未需要这些几何
            量的计算依赖。
        time_scheme, dual_time_inner_iter: DUAL_TIME 支持（2026-09-02
            续接）——透传给 `build_fully_distributed_rank_package`，
            `time_scheme` 是 `TimeIntegrationScheme` 枚举值（不是
            字符串），None（默认）时回退到 `TimeIntegrationScheme.
            SSP_RK3`。

    Returns:
        (my_package, root_context)：`my_package` 是本 rank 的紧凑包
        （见 `build_fully_distributed_rank_package` Returns 文档，
        字段完全一致），`root_context` 见上方"Order Continuation 支持"
        一节。
    """
    from autoflowcfd.cli.solve_helpers import load_mesh_for_solver
    from autoflowcfd.core.mpi.partition import partition_mesh
    from autoflowcfd.core.mpi.comm import bcast_from_root
    from autoflowcfd.fr.operators import generate_fr_operators
    from autoflowcfd.core.fr_solver.boundary import build_boundary_ghost_provider

    rank = get_rank()
    comm = get_comm()

    if rank == 0:
        logger.info("Root rank loading full mesh (fully-distributed mode)...")
        mesh, _volume_data = load_mesh_for_solver(
            input_file, order, surface_mesh=surface_mesh,
            skip_quality_check=skip_quality_check,
        )
        ops = generate_fr_operators(order)
        fc = mesh.face_connectivity

        cell_partition = partition_mesh(fc, n_ranks, n_cells=mesh.n_cells)

        # 边界幽灵态提供者只需要完整全局网格构建一次（root 独有）——用
        # 一个最小鸭子类型对象满足 build_boundary_ghost_provider 的接口
        # 要求（`.mesh`/`.freestream`/`.turb_model_name`/`.wmles_model`，
        # 见该函数文档），不需要构造真正的 FRSolver。
        import types
        # 真实 bug 修复（2026-09-02，接入 SST/DDES/IDDES 时发现）：此前
        # 这里硬编码 `turb_model_name="NONE"`——`build_boundary_ghost_
        # provider` 用它判断 `use_sem`（DDES/IDDES 应该在 VELOCITY_INLET
        # 自动接入 BD-02 合成湍流入口，见 core/fr_solver/boundary.py
        # 文档），硬编码 "NONE" 会让"完全分布式加载"模式下的 DDES/IDDES
        # 永远拿不到 SEM 入口，与单机/CPU MPI 传统模式行为不一致。
        # 真实 bug 修复（2026-09-02，接入 WMLES 时发现，与上面 DDES/IDDES
        # 的 turb_model_name 硬编码是同一类问题）：`wmles_model` 也硬编码
        # None——`build_boundary_ghost_provider` 用 `getattr(solver,
        # "wmles_model",None) is None` 判断 WALL 组是否要切换
        # `is_no_slip=False`（见该函数文档 #9 修复说明），WMLES 激活时
        # 若这里恒为 None 会让"完全分布式加载"模式下的 WMLES 恒得到
        # `is_no_slip=True`，与单机/CPU MPI 传统模式行为不一致（这里只
        # 需要一个非 None 的哨兵值，不需要真正的 WMLESModel 实例——
        # `build_boundary_ghost_provider` 只检查 is None）。
        root_solver_stub = types.SimpleNamespace(
            mesh=mesh, freestream=freestream, turb_model_name=turb_model_name,
            wmles_model=(object() if turb_model_name == "WMLES" else None),
        )
        boundary_ghost_provider_global = build_boundary_ghost_provider(
            root_solver_stub, bc_overrides=bc_overrides or {},
        )

        # SST/DDES/IDDES/WMLES：wall_node_indices/h_max/h_wn 只依赖完整
        # 全局网格，只需要算一次（不随 rank 变化），见
        # build_fully_distributed_rank_package 文档对应参数说明。
        wall_node_indices = None
        h_max_global = h_wn_global = None
        if turb_model_name in ("SST", "DDES", "IDDES", "WMLES"):
            boundary_groups = getattr(mesh, 'boundary_groups', None)
            if boundary_groups is not None:
                for bg_name, bg in boundary_groups.items():
                    if 'WALL' in bg_name.upper() or bg.get('type', '').upper() == 'WALL':
                        wall_node_indices = bg.get('node_indices')
                        break
            if turb_model_name in ("DDES", "IDDES"):
                # DDES（2026-09-02 补齐）：apply_to_sst_model 现在优先用
                # h_max（max_edge 网格尺度）而不是 cube_root(V)，见
                # build_fully_distributed_rank_package 对应分支文档。
                from autoflowcfd.core.turbulence.des import compute_h_max_and_h_wn
                h_max_global, h_wn_global = compute_h_max_and_h_wn(mesh)

        packages = [
            build_fully_distributed_rank_package(
                mesh, ops, fc, cell_partition, r, n_ranks,
                boundary_ghost_provider_global, freestream, mu_molecular, mach_ref,
                order, enable_viscous,
                turb_model_name=turb_model_name, wall_node_indices=wall_node_indices,
                h_max_global=h_max_global, h_wn_global=h_wn_global,
                turbulence_intensity=turbulence_intensity, viscosity_ratio=viscosity_ratio,
                time_scheme=time_scheme, dual_time_inner_iter=dual_time_inner_iter,
            )
            for r in range(n_ranks)
        ]
        my_package = packages[0]

        if n_ranks > 1:
            import pickle
            for r in range(1, n_ranks):
                buf = pickle.dumps(packages[r])
                buf_size = np.array([len(buf)], dtype=np.int64)
                comm.Send(buf_size, dest=r, tag=310)
                comm.Send(buf, dest=r, tag=311)

        # root_context（见函数文档"Order Continuation 支持"一节）：
        # 只有 root 保留，供后续阶数切换重新计算+重新分发紧凑包用，
        # 不随本次分发的 packages 一起发送给非 root rank（它们不需要，
        # 也不应该持有完整全局网格）。
        root_context = {
            'mesh': mesh, 'ops': ops, 'fc': fc, 'cell_partition': cell_partition,
            'boundary_ghost_provider_global': boundary_ghost_provider_global,
            'freestream': freestream, 'mu_molecular': mu_molecular, 'mach_ref': mach_ref,
            'enable_viscous': enable_viscous, 'turb_model_name': turb_model_name,
            'wall_node_indices': wall_node_indices,
            'h_max_global': h_max_global, 'h_wn_global': h_wn_global,
            'turbulence_intensity': turbulence_intensity, 'viscosity_ratio': viscosity_ratio,
            'bc_overrides': bc_overrides or {}, 'n_ranks': n_ranks,
            'time_scheme': time_scheme, 'dual_time_inner_iter': dual_time_inner_iter,
        }
    else:
        import pickle
        buf_size = np.empty(1, dtype=np.int64)
        comm.Recv(buf_size, source=0, tag=310)
        buf = np.empty(int(buf_size[0]), dtype=np.uint8)
        comm.Recv(buf, source=0, tag=311)
        my_package = pickle.loads(buf.tobytes())
        root_context = None

    return my_package, root_context


def redistribute_fully_distributed_for_new_order(solver, target_p: int) -> None:
    """"完全分布式加载"模式的阶数切换（2026-09-02，见 core/mpi/
    distributed_order_continuation.py 模块文档"完全分布式加载"一节）。

    与"传统模式"（`cpu_traditional_interpolate_to_new_order`）的关键
    区别：本 rank 从未持有完整全局网格，`PrecompactedMeshData`/
    `DistributedFlatFaceGeometry` 都是 root 预先按 compact 索引空间
    切好、通过 MPI 发过来的——阶数切换后这些几何量的形状/内容整体
    改变，必须由 root 用它持续持有的完整全局网格（`solver._root_
    context`，见 `distributed_mesh_load_v2` 文档）重新计算、重新分发
    一份新的紧凑包，不是本 rank 自己能算出来的。

    协议（与 `distributed_mesh_load_v2` 的初次分发同一套 Send/Recv
    机制，只是复用已有的 `root_context` 而不是重新加载/重新分区）：
    1. 每个 rank 独立在本地插值/重置自己的 local U + 湍流场（纯本地
       操作，`n_local`——同一个 `cell_partition`——阶数切换不变，不
       需要通信，见下方"local 状态"一节）。
    2. Root：`root_context['mesh'].set_order(target_p)` + 重新生成
       `ops` + 重新构建 `boundary_ghost_provider_global`（阶数相关的
       FP 几何）+ 对每个 rank 调用 `build_fully_distributed_rank_
       package` 算出新紧凑包，Send 给对应 rank（root 自己直接用）。
    3. 每个 rank：接收新包，把上一步算好的插值/重置后 local U 写入
       新包对应 partition 布局的 state，替换 `solver` 的全部 compact
       相关属性（partition/dist_flat_face/mesh/ops/wall_distance_
       compact/iddes 字段/boundary_ghost_provider），不重新构造
       `solver` 实例本身（保留 `turb_model`/`sgs_model`/`wmles_model`
       对象引用，只替换它们的 k_field/omega_field/nu_t 数组）。

    只支持通过 `root_context` 非 None 构造的实例（`DistributedFRSolver.
    from_fully_distributed_package(package, n_ranks, root_context=...)`
    ——root_context 未提供时 fail-fast，不静默产生错误结果）。
    """
    from autoflowcfd.core.mpi.comm import get_comm
    from autoflowcfd.core.mpi import get_rank
    from autoflowcfd.fr.operators import generate_fr_operators
    from autoflowcfd.fr.quadrature_points import gauss_legendre
    from autoflowcfd.core.mpi.distributed_state import DistributedFRState
    from autoflowcfd.core.mpi.halo import HaloExchange
    from autoflowcfd.core.fr_residual.inviscid import conserved_to_primitive
    from autoflowcfd.core.fr_solver.turbulence import _set_freestream_turbulence
    from autoflowcfd.core.utils.order_continuation import _build_linear_interp_matrix_3d

    old_order = solver.current_order
    if old_order == target_p:
        return

    is_root_rank = get_rank() == 0
    root_context = getattr(solver, '_root_context', None) if is_root_rank else None
    if is_root_rank and root_context is None:
        raise NotImplementedError(
            "redistribute_fully_distributed_for_new_order: root rank 没有 "
            "_root_context（本实例不是通过 distributed_mesh_load_v2 返回的 "
            "root_context 构造的）——Order Continuation 在'完全分布式加载' "
            "模式下要求 root 持续持有完整全局网格用于重新分发，不支持从 "
            "缺少这份上下文的实例继续爬坡（如实报告，不是假装能用）。"
        )

    n_local = solver.partition.n_local_cells
    n_vars = solver.state.n_vars

    # --- 1. 每个 rank 独立插值/重置自己的 local U + 湍流场（纯本地，
    # 数学与 CPU"传统模式"完全一致）---
    #
    # 自由来流条件：`self.freestream` 只在 turb_model_name 为
    # SST/DDES/IDDES 时才被 `from_fully_distributed_package` 设置（见
    # 该方法对应代码），NONE/LES/WMLES 分支没有这个属性——`self.
    # _package_freestream` 是本函数与 `from_fully_distributed_package`
    # 共同维护的、对全部 turb_model_name 都存在的通用存储点（见
    # `from_fully_distributed_package` 里的同名赋值、以及本函数末尾
    # "应用新包"一节的同名更新）。
    freestream = getattr(solver, '_package_freestream', None) or getattr(solver, 'freestream', None) \
        or {'rho_inf': 1.225, 'vel_inf': 33.33, 'p_inf': 101325.0}
    rho_inf = freestream.get('rho_inf', 1.225)
    vel_inf = freestream.get('vel_inf', 33.33)
    p_inf = freestream.get('p_inf', 101325.0)

    if target_p > old_order:
        old_sps_1d, _ = gauss_legendre(old_order + 1)
        new_sps_1d, _ = gauss_legendre(target_p + 1)
        W = _build_linear_interp_matrix_3d(old_sps_1d, new_sps_1d)
        old_local_U = solver.state.get_local_U()[:n_local]
        new_local_U = np.einsum('ab,cbv->cav', W, old_local_U)

        if solver.turb_model is not None and hasattr(solver.turb_model, 'k_field'):
            solver.turb_model.k_field = np.einsum('ab,cb->ca', W, solver.turb_model.k_field[:n_local])
            solver.turb_model.omega_field = np.einsum('ab,cb->ca', W, solver.turb_model.omega_field[:n_local])
            old_nu_t = getattr(solver.turb_model, 'nu_t', None)
            if old_nu_t is not None and old_nu_t.shape[1] == old_local_U.shape[1]:
                solver.turb_model.nu_t = np.einsum('ab,cb->ca', W, old_nu_t[:n_local])
            if hasattr(solver.turb_model, 'des_length_scale'):
                solver.turb_model.des_length_scale = None
        if getattr(solver, 'sgs_model', None) is not None and hasattr(solver.sgs_model, 'nu_t'):
            solver.sgs_model.nu_t = None
    else:
        new_n_sps = (target_p + 1) ** 3
        gamma = 1.4
        e = p_inf / ((gamma - 1.0) * rho_inf) + 0.5 * vel_inf ** 2
        new_local_U = np.zeros((n_local, new_n_sps, n_vars))
        new_local_U[:, :, 0] = rho_inf
        new_local_U[:, :, 1] = rho_inf * vel_inf
        new_local_U[:, :, 4] = rho_inf * e

        if solver.turb_model is not None and hasattr(solver.turb_model, 'k_field'):
            k_inf, omega_inf = _set_freestream_turbulence(solver)
            solver.turb_model.k_field = np.ones((n_local, new_n_sps)) * k_inf
            solver.turb_model.omega_field = np.ones((n_local, new_n_sps)) * omega_inf
            if hasattr(solver.turb_model, 'nu_t'):
                solver.turb_model.nu_t = np.zeros((n_local, new_n_sps))
            if hasattr(solver.turb_model, 'des_length_scale'):
                solver.turb_model.des_length_scale = None
        if getattr(solver, 'sgs_model', None) is not None and hasattr(solver.sgs_model, 'nu_t'):
            solver.sgs_model.nu_t = None

    # --- 2. Root 重新计算 + 分发新紧凑包 ---
    comm = get_comm()
    n_ranks = solver.n_ranks

    if is_root_rank:
        from autoflowcfd.core.fr_solver.boundary import build_boundary_ghost_provider
        import types

        mesh = root_context['mesh']
        stale_orders = [o for o in list(mesh._order_geometry_cache) if o != target_p]
        for o in stale_orders:
            del mesh._order_geometry_cache[o]
        mesh.set_order(target_p)
        ops = generate_fr_operators(target_p, flux_point_type=getattr(solver, 'flux_type', 'radau'))
        root_context['ops'] = ops

        turb_model_name = root_context['turb_model_name']
        root_solver_stub = types.SimpleNamespace(
            mesh=mesh, freestream=root_context['freestream'], turb_model_name=turb_model_name,
            wmles_model=(object() if turb_model_name == "WMLES" else None),
        )
        boundary_ghost_provider_global = build_boundary_ghost_provider(
            root_solver_stub, bc_overrides=root_context.get('bc_overrides', {}),
        )
        root_context['boundary_ghost_provider_global'] = boundary_ghost_provider_global

        packages = [
            build_fully_distributed_rank_package(
                mesh, ops, root_context['fc'], root_context['cell_partition'], r, n_ranks,
                boundary_ghost_provider_global, root_context['freestream'],
                root_context['mu_molecular'], root_context['mach_ref'],
                target_p, root_context['enable_viscous'],
                turb_model_name=turb_model_name, wall_node_indices=root_context['wall_node_indices'],
                h_max_global=root_context['h_max_global'], h_wn_global=root_context['h_wn_global'],
                turbulence_intensity=root_context['turbulence_intensity'],
                viscosity_ratio=root_context['viscosity_ratio'],
                time_scheme=root_context.get('time_scheme'),
                dual_time_inner_iter=root_context.get('dual_time_inner_iter', 20),
            )
            for r in range(n_ranks)
        ]
        my_package = packages[0]
        if n_ranks > 1:
            import pickle
            for r in range(1, n_ranks):
                buf = pickle.dumps(packages[r])
                buf_size = np.array([len(buf)], dtype=np.int64)
                comm.Send(buf_size, dest=r, tag=320)
                comm.Send(buf, dest=r, tag=321)
    else:
        import pickle
        buf_size = np.empty(1, dtype=np.int64)
        comm.Recv(buf_size, source=0, tag=320)
        buf = np.empty(int(buf_size[0]), dtype=np.uint8)
        comm.Recv(buf, source=0, tag=321)
        my_package = pickle.loads(buf.tobytes())

    # --- 3. 应用新包：替换 compact 相关属性，保留 turb_model/sgs_model/
    # wmles_model 对象本身（只是上一步已经替换过它们的数组）---
    solver.ops = generate_fr_operators(target_p, flux_point_type=getattr(solver, 'flux_type', 'radau'))
    solver.mesh = my_package['precompacted_mesh']
    solver.partition = my_package['partition']
    solver.dist_flat_face = my_package['dist_fc']

    new_n_sps = solver.mesh.n_sps_per_cell
    new_state = DistributedFRState(solver.partition, new_n_sps, n_vars)
    new_state.U[:n_local] = new_local_U
    new_state.Q[:n_local] = conserved_to_primitive(new_local_U[..., :5])
    solver.state = new_state
    solver.halo_exchange = HaloExchange(solver.partition, new_n_sps, n_vars)

    if solver.turb_model is not None:
        solver.turb_halo_exchange = HaloExchange(solver.partition, new_n_sps, 2)
    solver.wall_distance_compact = my_package.get('wall_distance_compact')
    solver.iddes_h_max_compact = my_package.get('iddes_h_max_compact')
    solver.iddes_h_wn_compact = my_package.get('iddes_h_wn_compact')
    if solver.ddes_model is not None:
        solver.des_length_scale_halo_exchange = HaloExchange(solver.partition, new_n_sps, 1)

    # `_local_solver` 在"完全分布式加载"模式下从构造起就是这个
    # `types.SimpleNamespace` 替身（不是"传统模式"的懒加载真实
    # `FRSolver`，见 `from_fully_distributed_package` 构造处同一段
    # 代码）——阶数切换后 `boundary_ghost_provider`/`enable_viscous`
    # 都是新阶数的值，必须真正重新赋值，不能保留旧引用。
    import types
    solver._local_solver = types.SimpleNamespace(
        config=types.SimpleNamespace(
            physics=types.SimpleNamespace(enable_viscous=my_package['enable_viscous'])
        ),
        mu_molecular=my_package['mu_molecular'],
        boundary_ghost_provider=my_package['boundary_ghost_provider'],
        freestream={**my_package['freestream'], "mach_ref": my_package['mach_ref']},
    )
    solver._package_freestream = my_package['freestream']

    if hasattr(solver, '_dual_time_U_prev'):
        solver._dual_time_U_prev = None

    solver.current_order = target_p


def extract_local_mesh_data(
    mesh,
    face_connectivity,
    local_cells: np.ndarray,
    n_global_cells: int,
) -> dict:
    """从完整网格中提取指定 cell 子集的局部数据。

    Args:
        mesh: HighOrderMesh（完整网格）
        face_connectivity: FRFaceConnectivity（全局面连接关系）
        local_cells: (n_local,) 本 rank 拥有的 cell 全局索引
        n_global_cells: 全局 cell 总数

    Returns:
        local_data: dict，包含局部网格的所有必要数据
    """
    n_local = len(local_cells)

    # 构建全局→局部映射
    global_to_local = np.full(n_global_cells, -1, dtype=np.int64)
    global_to_local[local_cells] = np.arange(n_local)

    # 1. 提取 SP 坐标
    sps_coords_local = mesh.sps_coords[local_cells].copy()

    # 2. 提取 Jacobian 数据
    jacobians_local = {}
    if mesh.jacobians is not None:
        for key, arr in mesh.jacobians.items():
            if arr.shape[0] == mesh.n_cells:
                jacobians_local[key] = arr[local_cells].copy()
            else:
                jacobians_local[key] = arr.copy()

    # 3. 提取 fine Jacobian（如果有）
    jacobians_fine_local = None
    if mesh.jacobians_fine is not None:
        jacobians_fine_local = {}
        for key, arr in mesh.jacobians_fine.items():
            if arr.shape[0] == mesh.n_cells:
                jacobians_fine_local[key] = arr[local_cells].copy()
            else:
                jacobians_fine_local[key] = arr.copy()

    # 4. 提取 cell volumes
    cell_volumes_local = None
    if mesh.cell_volumes is not None:
        cell_volumes_local = mesh.cell_volumes[local_cells].copy()

    # 5. 统计 local prism cells
    n_prism_local = 0
    if hasattr(mesh, 'cell_types'):
        # cell_types 数组：0=tet, 1=prism
        # local_cells 中 prism 的数量
        n_prism_local = int(np.sum(mesh.cell_types[local_cells] == 1))

    # 6. 提取本 rank 拥有的面（owner 是 local cell 的面）
    owner_is_local = np.isin(face_connectivity.owner_cell, local_cells)
    face_indices = np.where(owner_is_local)[0]

    # 提取面连接关系数据，重映射 cell 索引
    fc_data = {}
    fc_data['n_faces'] = len(face_indices)
    fc_data['owner_cell'] = global_to_local[face_connectivity.owner_cell[face_indices]].copy()
    
    # neighbor_cell: 边界面为 -1，内部面重映射
    neighbor_global = face_connectivity.neighbor_cell[face_indices]
    neighbor_local = np.where(neighbor_global >= 0, global_to_local[np.maximum(neighbor_global, 0)], -1)
    fc_data['neighbor_cell'] = neighbor_local
    
    fc_data['is_boundary'] = face_connectivity.is_boundary[face_indices].copy()
    fc_data['owner_cube_face'] = face_connectivity.owner_cube_face[face_indices].copy()
    fc_data['neighbor_cube_face'] = face_connectivity.neighbor_cube_face[face_indices].copy()
    fc_data['normal'] = face_connectivity.normal[face_indices].copy()
    fc_data['area'] = face_connectivity.area[face_indices].copy()
    fc_data['center'] = face_connectivity.center[face_indices].copy()
    fc_data['face_node_ids'] = face_connectivity.face_node_ids[face_indices].copy()

    # 7. 提取 cell_types（如果有）
    cell_types_local = None
    if hasattr(mesh, 'cell_types') and mesh.cell_types is not None:
        cell_types_local = mesh.cell_types[local_cells].copy()

    return {
        'n_cells': n_local,
        'n_prism_cells': n_prism_local,
        'n_points_1d': mesh.n_points_1d,
        'n_sps_per_cell': mesh.n_sps_per_cell,
        'sps_coords': sps_coords_local,
        'jacobians': jacobians_local,
        'jacobians_fine': jacobians_fine_local,
        'cell_volumes': cell_volumes_local,
        'cell_types': cell_types_local,
        'face_connectivity': fc_data,
        'order': mesh.order,
    }


def build_local_mesh_from_data(local_data: dict):
    """从局部数据构建局部 HighOrderMesh 对象。

    Args:
        local_data: extract_local_mesh_data 返回的数据字典

    Returns:
        local_mesh: 部分初始化的 HighOrderMesh（只包含局部数据）
    """
    from autoflowcfd.grid.high_order.high_order_mesh import HighOrderMesh

    mesh = HighOrderMesh(order=local_data['order'])
    mesh.n_cells = local_data['n_cells']
    mesh.n_prism_cells = local_data['n_prism_cells']
    mesh.sps_coords = local_data['sps_coords']
    mesh.jacobians = local_data['jacobians']
    mesh.jacobians_fine = local_data['jacobians_fine']
    mesh.cell_volumes = local_data['cell_volumes']

    if local_data['cell_types'] is not None:
        mesh.cell_types = local_data['cell_types']

    # 注意：face_connectivity 和 face_flux_points 需要单独构建
    # 这里先存储原始数据，由调用方构建完整的 FRFaceConnectivity
    mesh._local_fc_data = local_data['face_connectivity']

    return mesh


def distribute_mesh_data(
    mesh,
    face_connectivity,
    cell_partition: np.ndarray,
    n_ranks: int,
) -> Optional[dict]:
    """Root rank 将各 rank 的局部网格数据分发出去。

    Args:
        mesh: HighOrderMesh（完整网格，仅 root rank 需要）
        face_connectivity: FRFaceConnectivity（全局面连接关系，仅 root rank 需要）
        cell_partition: (n_global_cells,) cell_partition[i] = cell i 所属 rank
        n_ranks: MPI rank 总数

    Returns:
        local_data: 本 rank 的局部网格数据（所有 rank 都有值）
    """
    rank = get_rank()
    comm = get_comm()
    MPI = get_mpi()

    n_global_cells = mesh.n_cells if rank == 0 else None

    # 广播全局 cell 数
    if n_ranks > 1:
        n_global_buf = np.array([n_global_cells if n_global_cells is not None else 0], dtype=np.int64)
        comm.Bcast(n_global_buf, root=0)
        n_global_cells = int(n_global_buf[0])
    
    # Root rank: 提取并发送各 rank 的局部数据
    if rank == 0:
        # 提取 root 自己的数据
        local_cells_root = np.where(cell_partition == 0)[0]
        local_data = extract_local_mesh_data(mesh, face_connectivity, local_cells_root, n_global_cells)

        # 发送给其他 rank
        if n_ranks > 1:
            for r in range(1, n_ranks):
                local_cells_r = np.where(cell_partition == r)[0]
                data_r = extract_local_mesh_data(mesh, face_connectivity, local_cells_r, n_global_cells)
                
                # 使用 pickle 序列化发送
                import pickle
                buf = pickle.dumps(data_r)
                buf_size = np.array([len(buf)], dtype=np.int64)
                comm.Send(buf_size, dest=r, tag=300)
                comm.Send(buf, dest=r, tag=301)
    else:
        # 非 root rank: 接收数据
        import pickle
        buf_size = np.empty(1, dtype=np.int64)
        comm.Recv(buf_size, source=0, tag=300)
        buf = np.empty(int(buf_size[0]), dtype=np.uint8)
        comm.Recv(buf, source=0, tag=301)
        local_data = pickle.loads(buf.tobytes())

    return local_data


def distributed_mesh_load(
    input_file: str,
    order: int,
    surface_mesh: Optional[str],
    n_ranks: int,
    skip_quality_check: bool = False,
) -> Tuple:
    """完全分布式网格加载。

    只有 root rank 加载完整网格并执行质量检查，然后分发各 rank 的局部数据。

    Args:
        input_file: 体网格文件路径
        order: FR 阶数
        surface_mesh: 原始面网格路径（.nas 格式需要）
        n_ranks: MPI rank 数
        skip_quality_check: 是否跳过质量检查

    Returns:
        (local_mesh, local_fc_data, partition_info):
            local_mesh: 本 rank 的局部 HighOrderMesh
            local_fc_data: 本 rank 的局部面连接关系数据
            partition_info: 分区信息（cell_partition 等）
    """
    from autoflowcfd.cli.solve_helpers import load_mesh_for_solver
    from autoflowcfd.core.mpi.partition import partition_mesh
    from autoflowcfd.core.mpi.comm import bcast_from_root

    rank = get_rank()

    if rank == 0:
        # Root rank: 加载完整网格
        logger.info("Root rank loading full mesh...")
        mesh, volume_data = load_mesh_for_solver(
            input_file, order, surface_mesh=surface_mesh,
            skip_quality_check=skip_quality_check,
        )

        # 面连接关系已由 load_mesh_for_solver -> HighOrderMesh.load_from_volume_mesh
        # 构建好挂在 mesh 上——此前这里写的是 `FRFaceConnectivity(mesh, ops)`，
        # 但 FRFaceConnectivity 是一个 dataclass，字段是 owner_cell/
        # neighbor_cell/... 等 numpy 数组，不是 (mesh, ops)，会把 mesh 对象
        # 本身错误地绑定到 owner_cell 字段——第四次评审发现的又一个
        # 独立崩溃 bug，MPI 分布式网格加载路径此前从未真正跑通过。
        from autoflowcfd.fr.operators import generate_fr_operators
        ops = generate_fr_operators(order)
        fc = mesh.face_connectivity

        # 分区
        logger.info(f"Root rank partitioning mesh into {n_ranks} parts...")
        cell_partition = partition_mesh(fc, n_ranks, n_cells=mesh.n_cells)

        # 分发网格数据
        local_data = distribute_mesh_data(mesh, fc, cell_partition, n_ranks)

        # 广播分区信息（非 root rank 需要知道 cell_partition 来构建 partition）。
        # 同时广播**全局**面连接关系的 owner_cell/neighbor_cell/is_boundary——
        # build_distributed_partition 的 halo 探测（哪些面跨越分区边界）
        # 必须在全局、未裁剪的面连接关系上做：extract_local_mesh_data
        # 产出的 local_fc_data 已经把跨 rank 的 neighbor 重映射成 -1
        # （与真正的边界面用同一个哨兵值，二者在本 rank 视角下无法区分），
        # 用它调用 build_distributed_partition 只会产出全空的 halo 列表——
        # 残差在分区边界上完全得不到邻居数据（V2.0 专家组评审逐行核实，
        # DistributedFRSolver 的 face_connectivity_data 路径此前从未被
        # 真正跑通过）。这三个全局数组量级与 cell_partition 相同
        # （均为 O(n_faces)/O(n_cells) 的整数/布尔数组），随 partition_info
        # 一起广播的开销可忽略。
        partition_info = {
            'cell_partition': cell_partition,
            'n_global_cells': mesh.n_cells,
            'global_owner_cell': fc.owner_cell,
            'global_neighbor_cell': fc.neighbor_cell,
            'global_is_boundary': fc.is_boundary,
        }
    else:
        # 非 root rank: 接收数据
        local_data = distribute_mesh_data(None, None, None, n_ranks)
        partition_info = None

    # 广播分区信息
    if n_ranks > 1:
        partition_info = bcast_from_root(partition_info)

    # 构建局部网格对象
    local_mesh = build_local_mesh_from_data(local_data)

    return local_mesh, local_data['face_connectivity'], partition_info
