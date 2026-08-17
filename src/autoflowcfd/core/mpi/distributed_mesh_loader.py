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

import numpy as np
from typing import Optional, Dict, Tuple

from loguru import logger

from autoflowcfd.core.mpi import get_comm, get_mpi, mpi_available, get_rank, get_size


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
    from autoflowcfd.grid.high_order_mesh import HighOrderMesh

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
    from autoflowcfd.grid.face_connectivity import FRFaceConnectivity
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

        # 构建面连接关系
        from autoflowcfd.core.operators import FROperators
        ops = FROperators(order=order, n_points_1d=mesh.n_points_1d)
        fc = FRFaceConnectivity(mesh, ops)

        # 分区
        logger.info(f"Root rank partitioning mesh into {n_ranks} parts...")
        cell_partition = partition_mesh(fc, n_ranks)

        # 分发网格数据
        local_data = distribute_mesh_data(mesh, fc, cell_partition, n_ranks)

        # 广播分区信息（非 root rank 需要知道 cell_partition 来构建 partition）
        partition_info = {
            'cell_partition': cell_partition,
            'n_global_cells': mesh.n_cells,
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
