"""AutoFlowCFD V2.0 - 完全分布式加载的紧凑包（root 端预切）。

从 `core/mpi/distributed_mesh_loader.py`（原 1030 行）拆出（2026-09-24，
项目"单文件不超 500 行"规范）。**纯搬家，逻辑未改**。

这一层做的是 root rank 上的事：按各 rank 的 compact 索引空间**预先**切好
紧凑包再发送，取代此前"发全局网格、各 rank 自己切"的做法。
"""

import copy
import numpy as np
from typing import Optional


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
        order, face_area=None, face_normal=None,
    ):
        self.n_cells = det_jacs.shape[0]  # compact（local+halo）大小
        self.n_prism_cells = n_prism_cells
        self.n_points_1d = n_points_1d
        # 多项式阶数：残差链路真实会读它（粘性 IP 罚项常数按阶数解析，见
        # `fr_operators/flux_kernels.resolve_viscous_ip_constant`），而本
        # 对象是 mesh-like 鸭子类型，凡残差链路读的 mesh 属性都必须持有。
        # 2026-09-23 补上；不给默认值，因为"猜错阶数"会静默用错罚项常数。
        self.order = int(order)
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

        # 逐面 area/normal（2026-09-14 新增）：本 rank 局部面那一段，由
        # root 按 `partition.local_faces` 从全局 `FRFaceConnectivity` 切好
        # 后随包下发。**这是逐单元局部 CFL 步长的必需输入**——局部 dt 的
        # 谱半径求和是逐面的，而本对象刻意不持有全局 face_connectivity。
        # 这两个数组很小（n_local_faces×1 + n_local_faces×3 个 double），
        # 与本类"只持有 local+halo 那一份"的内存目标不冲突。
        # 为什么必须用与单机同一个几何量（而不是从 base_flat 的逐 flux
        # point 量现算）：见 core/mpi/distributed_cfl.py::
        # _DistributedFaceConnectivityView 文档。
        self.face_area = face_area
        self.face_normal = face_normal


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
    cfl_start: Optional[float] = None,
    cfl_max: Optional[float] = None,
    cfl_min: Optional[float] = None,
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

    # 逐面 area/normal（2026-09-14）：按 partition.local_faces 从全局
    # FRFaceConnectivity 切出本 rank 那一段，随包下发——逐单元局部 CFL
    # 步长的必需输入（本对象刻意不持有全局 face_connectivity），见
    # PrecompactedMeshData.face_area 字段注释。
    _lf = partition.local_faces
    _fc_global = face_connectivity if face_connectivity is not None else getattr(
        mesh, 'face_connectivity', None)
    if _fc_global is None or getattr(_fc_global, 'area', None) is None:
        raise ValueError(
            "root 侧需要完整的 face_connectivity（含 area/normal）才能为各 "
            "rank 切出局部面几何——这是逐单元局部 CFL 步长的必需输入，"
            "不能静默回退到全局固定步长（被禁止的简化）")
    face_area_local = np.ascontiguousarray(_fc_global.area[_lf])
    face_normal_local = np.ascontiguousarray(_fc_global.normal[_lf])

    precompacted_mesh = PrecompactedMeshData(
        det_jacs=det_jacs, inv_jacs=inv_jacs,
        det_jacs_fine=det_jacs_fine, inv_jacs_fine=inv_jacs_fine,
        cell_volumes=cell_volumes, sps_coords=sps_coords, cell_types=cell_types,
        n_prism_cells=dist_fc.base_flat.n_prism,
        n_points_1d=mesh.n_points_1d, n_sps_per_cell=n_sps, n_sps_per_cell_fine=n_sps_fine,
        order=mesh.order,
        face_area=face_area_local, face_normal=face_normal_local,
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
        # 见本函数 Args 里 cfl_* 一节：这三个量此前从未进入 package，
        # 使 CLI 的 --cfl-start/--cfl-max 在完全分布式路径上被静默丢弃。
        'cfl_start': cfl_start,
        'cfl_max': cfl_max,
        'cfl_min': cfl_min,
    }
