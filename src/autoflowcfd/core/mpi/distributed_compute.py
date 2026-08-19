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
    from autoflowcfd.core.fr_residual.inviscid import compute_inviscid_residual_fr

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
    partition: DistributedPartition,
    halo_exchange: HaloExchange,
    dist_fc: DistributedFlatFaceGeometry,
    local_mesh,
    ops,
    mu: float,
    boundary_ghost_provider=None,
) -> np.ndarray:
    """分布式粘性残差计算。

    此前这里错误地按 `compute_viscous_residual_fr(U, grad_U, adapter, ops,
    config)` 的参数顺序调用，但该函数真实签名是
    `compute_viscous_residual_fr(U, mesh, ops, mu, Pr, mu_t_field=None,
    Pr_t=0.9, boundary_ghost_provider=None)`（见 fr_residual/viscous_flux.py）
    ——等价于把 `grad_U`（一个数组）当 `mesh` 传、把 `adapter` 当 `ops`
    传、把 `ops` 当 `mu`（标量）传、把 `config` 当 `Pr`（标量）传，只要
    `config.physics.enable_viscous=True`（真实粘性算例的默认配置）就会
    在第一次残差求值时立刻因属性访问失败而崩溃——分布式求解器实际上
    从未跑通过任何粘性算例。改用单机路径同一个入口
    `core.fr_residual.viscous.compute_viscous_residual`（该函数内部会
    重新计算 primitive 变量与梯度，不需要调用方预先算好并传入，`grad_U`
    参数因此整个不再需要）。

    尚未支持湍流涡粘度耦合（`mu_t_field` 恒为 None，等价于纯层流粘性
    应力）——`DistributedFRSolver.__init__` 已经在构造时拒绝
    `turbulence_model != 'none'`，所以这里不会在有湍流模型的场景下
    被静默调用。

    Args:
        U_local: (n_local_cells, n_sps, 5) 本 rank 的 local cell 守恒变量
        partition: 分区信息
        halo_exchange: halo 交换管理器
        dist_fc: 分布式面连接关系
        local_mesh: 本地网格对象
        ops: FR 算子
        mu: 分子动力粘度（标量）
        boundary_ghost_provider: 边界幽灵态提供者

    Returns:
        residual: (n_local_cells, n_sps, 5) local cells 的粘性残差
    """
    from autoflowcfd.core.fr_residual.viscous import compute_viscous_residual as compute_viscous_residual_ldg
    from autoflowcfd.core.fr_residual.inviscid import conserved_to_primitive

    # 1. Halo 交换
    U_extended = halo_exchange.exchange(U_local)

    # 2. 创建网格适配器
    adapter = DistributedMeshAdapter(partition, dist_fc, local_mesh, ops)

    # 3. 调用现有残差函数（state_Q 参数只为兼容旧签名保留，函数内部
    # 从 state_U 自行重新计算 primitive 变量，见该函数文档）
    Q_extended = conserved_to_primitive(U_extended[..., :5])
    residual_extended = compute_viscous_residual_ldg(
        U_extended, Q_extended, ops, adapter, mu=mu,
        boundary_ghost_provider=boundary_ghost_provider,
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
    from autoflowcfd.core.fr_operators.gradients import compute_physical_gradient

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
    """分布式湍流输运方程计算——尚未实现，故意报错而不是静默跑错误物理。

    此前这里调用一个不存在的 `turbulence_transport(U, turb, grad_turb,
    mesh, ops, config, boundary_provider)` 函数（导入即失败）。真实的
    单机实现是 `core/turbulence/transport.py::compute_turbulence_
    transport_residual(solver)`——它的入参不是这几个松散数组，而是一个
    完整的 `FRSolver` 实例：要从 `solver.turb_model`（`k_field`/
    `omega_field`/`nu_t`/SST 各项系数）、`solver.state.Q`、
    `solver.wall_distance`、`solver._compute_gradients()` 里读一整套
    仍在自增长的湍流模型内部状态，不是几个可以从调用方直接拼出来的
    独立数组。

    要让这个函数真正可用，需要先把湍流模型实例接入
    `DistributedFRSolver`（分布式状态扩展到 7 vars、`turb_model` 与分布
    式 U 的 k/omega 分量同步、wall_distance 分发到 local cells、每个 RK
    子步之间 k/omega 也要 halo 交换)——这是一次完整的分布式湍流耦合
    实现，不是修一处调用签名能带出来的。`DistributedFRSolver.__init__`
    已经在构造时直接拒绝 `turbulence_model != 'none'`，所以这个函数
    在当前代码库里没有任何调用方；保留函数签名与文档是为了未来接入
    时有明确的落脚点，调用它本身应该失败得清楚，而不是被绕过或悄悄
    退化为忽略湍流输运。

    Raises:
        NotImplementedError: 恒为此——分布式湍流输运尚未实现。
    """
    raise NotImplementedError(
        "分布式湍流输运（DDES/SST 的 k/omega 对流+扩散）尚未实现："
        "需要先把 turb_model 实例接入 DistributedFRSolver 的分布式状态"
        "（7-var 状态、wall_distance 分发、逐 RK 子步的 k/omega halo "
        "交换），不是简单的函数签名修复。MPI 分布式求解目前只支持 "
        "turbulence_model='none'；单机模式已完整支持 SST/DDES/WMLES/LES。"
    )
