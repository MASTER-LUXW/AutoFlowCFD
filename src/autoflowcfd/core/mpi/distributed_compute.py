"""
AutoFlowCFD V2.0 - 分布式残差计算

将现有的残差计算函数改造为分布式版本，支持 MPI 域分解并行计算。

核心设计:
1. DistributedMeshAdapter: 将分布式数据包装为 mesh 接口，复用现有残差函数
2. 分布式残差计算: halo 交换 → 调用现有函数 → 返回 local cells 残差
3. 分区边界处理: 邻居 cell 数据从 halo 层读取

关键约束:
- 体积项只在 local cells 上计算（halo cells 的残差被忽略）
- 界面项在 local cells 拥有的所有面上计算（包括 partition boundary 面）
- 残差范数需要 MPI Allreduce（全局 L2 范数）
"""

import numpy as np
from typing import Optional, Callable

from loguru import logger

from autoflowcfd.core.mpi.partition import DistributedPartition
from autoflowcfd.core.mpi.halo import HaloExchange
from autoflowcfd.core.mpi.distributed_flat_face import DistributedFlatFaceGeometry


class DistributedMeshAdapter:
    """分布式网格适配器。

    将分布式数据结构包装为与 HighOrderMesh 相同的接口，
    使得现有的残差计算函数可以直接使用，无需修改。

    Attributes:
        partition: 分区信息
        dist_fc: 分布式面连接关系
        local_mesh: 本地网格对象（提供 jacobians 等）
        n_local_cells: 本 rank 的 local cell 数
        n_halo_cells: 本 rank 的 halo cell 数
    """

    def __init__(
        self,
        partition: DistributedPartition,
        dist_fc: DistributedFlatFaceGeometry,
        local_mesh,
        ops,
    ):
        """初始化分布式网格适配器。

        Args:
            partition: 本 rank 的分区信息
            dist_fc: 分布式面连接关系
            local_mesh: 本地网格对象（提供 jacobians、face_flux_points 等）
            ops: FR 算子
        """
        self.partition = partition
        self.dist_fc = dist_fc
        self.local_mesh = local_mesh
        self.ops = ops

        self.n_cells = partition.n_local_cells
        self.n_halo_cells = partition.n_halo
        self.n_points_1d = local_mesh.n_points_1d
        self.n_sps_per_cell = local_mesh.n_sps_per_cell

        # 统计 local prism cells
        # 假设 cell_types 只包含 local cells
        if hasattr(local_mesh, 'cell_types'):
            self.n_prism_cells = int(np.sum(local_mesh.cell_types == 1))
        else:
            self.n_prism_cells = 0

    @property
    def face_connectivity(self):
        """返回分布式面连接关系（接口兼容）。"""
        return self.dist_fc

    @property
    def face_flux_points(self):
        """返回面通量点（从本地网格获取）。"""
        return self.local_mesh.face_flux_points

    @property
    def jacobians(self):
        """返回 Jacobian 信息（只包含 local cells）。"""
        return self.local_mesh.jacobians

    @property
    def jacobians_fine(self):
        """返回 fine Jacobian 信息（用于 over-integration）。"""
        return self.local_mesh.jacobians_fine


def distributed_compute_inviscid_residual(
    U_local: np.ndarray,
    partition: DistributedPartition,
    halo_exchange: HaloExchange,
    dist_fc: DistributedFlatFaceGeometry,
    local_mesh,
    ops,
    boundary_ghost_provider: Optional[Callable] = None,
) -> np.ndarray:
    """分布式无粘残差计算。

    1. 执行 halo 交换，获取 local + halo 的扩展状态
    2. 创建网格适配器
    3. 调用现有残差函数（自动处理分区边界）
    4. 返回 local cells 的残差

    Args:
        U_local: (n_local_cells, n_sps, 5) 本 rank 的 local cell 守恒变量
        partition: 分区信息
        halo_exchange: halo 交换管理器
        dist_fc: 分布式面连接关系
        local_mesh: 本地网格对象
        ops: FR 算子
        boundary_ghost_provider: 边界幽灵态提供者

    Returns:
        residual: (n_local_cells, n_sps, 5) local cells 的残差
    """
    from autoflowcfd.core.fr_residual_inviscid import compute_inviscid_residual_fr

    # 1. Halo 交换：获取 local + halo 的扩展状态
    U_extended = halo_exchange.exchange(U_local)

    # 2. 创建网格适配器
    adapter = DistributedMeshAdapter(partition, dist_fc, local_mesh, ops)

    # 3. 调用现有残差函数
    # 注意：残差函数会访问 adapter.face_connectivity（即 dist_fc）
    # dist_fc 的 neighbor_cell 对于 partition boundary 面指向 halo cells
    # 残差函数会自动从 U_extended 中读取 halo cells 的数据
    residual_extended = compute_inviscid_residual_fr(
        U_extended, adapter, ops, boundary_ghost_provider
    )

    # 4. 只返回 local cells 的残差
    residual_local = residual_extended[:partition.n_local_cells]

    return residual_local


def distributed_compute_viscous_residual(
    U_local: np.ndarray,
    grad_U_local: np.ndarray,
    partition: DistributedPartition,
    halo_exchange: HaloExchange,
    dist_fc: DistributedFlatFaceGeometry,
    local_mesh,
    ops,
    config,
) -> np.ndarray:
    """分布式粘性残差计算。

    Args:
        U_local: (n_local_cells, n_sps, 5) 本 rank 的 local cell 守恒变量
        grad_U_local: (n_local_cells, n_sps, 3, 5) 本 rank 的 local cell 梯度
        partition: 分区信息
        halo_exchange: halo 交换管理器
        dist_fc: 分布式面连接关系
        local_mesh: 本地网格对象
        ops: FR 算子
        config: 求解器配置

    Returns:
        residual: (n_local_cells, n_sps, 5) local cells 的粘性残差
    """
    from autoflowcfd.core.fr_viscous_flux import compute_viscous_residual_fr

    # 1. Halo 交换（守恒变量和梯度）
    U_extended = halo_exchange.exchange(U_local)
    grad_U_extended = halo_exchange.exchange(grad_U_local)

    # 2. 创建网格适配器
    adapter = DistributedMeshAdapter(partition, dist_fc, local_mesh, ops)

    # 3. 调用现有残差函数
    residual_extended = compute_viscous_residual_fr(
        U_extended, grad_U_extended, adapter, ops, config
    )

    # 4. 只返回 local cells 的残差
    residual_local = residual_extended[:partition.n_local_cells]

    return residual_local


def distributed_compute_physical_gradient(
    U_local: np.ndarray,
    partition: DistributedPartition,
    halo_exchange: HaloExchange,
    local_mesh,
    ops,
) -> np.ndarray:
    """分布式物理梯度计算。

    Args:
        U_local: (n_local_cells, n_sps, 5) 本 rank 的 local cell 守恒变量
        partition: 分区信息
        halo_exchange: halo 交换管理器
        local_mesh: 本地网格对象
        ops: FR 算子

    Returns:
        grad_U: (n_local_cells, n_sps, 3, 5) local cells 的梯度
    """
    from autoflowcfd.core.fr_gradient import compute_physical_gradient

    # 1. Halo 交换
    U_extended = halo_exchange.exchange(U_local)

    # 2. 创建网格适配器
    adapter = DistributedMeshAdapter(partition, None, local_mesh, ops)

    # 3. 调用现有梯度函数
    grad_U_extended = compute_physical_gradient(U_extended, adapter, ops)

    # 4. 只返回 local cells 的梯度
    grad_U_local = grad_U_extended[:partition.n_local_cells]

    return grad_U_local


def distributed_turbulence_transport(
    U_local: np.ndarray,
    turb_local: np.ndarray,
    grad_turb_local: np.ndarray,
    partition: DistributedPartition,
    halo_exchange: HaloExchange,
    local_mesh,
    ops,
    config,
    boundary_provider=None,
) -> np.ndarray:
    """分布式湍流输运方程计算。

    Args:
        U_local: (n_local_cells, n_sps, 5) 本 rank 的 local cell 守恒变量
        turb_local: (n_local_cells, n_sps, 2) 本 rank 的 local cell 湍流变量 (k, omega)
        grad_turb_local: (n_local_cells, n_sps, 3, 2) 本 rank 的 local cell 湍流梯度
        partition: 分区信息
        halo_exchange: halo 交换管理器
        local_mesh: 本地网格对象
        ops: FR 算子
        config: 求解器配置
        boundary_provider: 湍流边界条件提供者

    Returns:
        d_turb_dt: (n_local_cells, n_sps, 2) local cells 的湍流残差
    """
    from autoflowcfd.core.turbulence_transport import turbulence_transport

    # 1. Halo 交换（守恒变量、湍流变量、湍流梯度）
    U_extended = halo_exchange.exchange(U_local)
    turb_extended = halo_exchange.exchange_scalar(turb_local)
    grad_turb_extended = halo_exchange.exchange(grad_turb_local)

    # 2. 创建网格适配器
    adapter = DistributedMeshAdapter(partition, None, local_mesh, ops)

    # 3. 调用现有湍流输运函数
    d_turb_dt_extended = turbulence_transport(
        U_extended, turb_extended, grad_turb_extended,
        adapter, ops, config, boundary_provider
    )

    # 4. 只返回 local cells 的残差
    d_turb_dt_local = d_turb_dt_extended[:partition.n_local_cells]

    return d_turb_dt_local
