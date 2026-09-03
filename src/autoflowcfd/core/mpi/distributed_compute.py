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

    真实 bug 修复（#2，V2.0 专家组盲审第4轮，2026-08-28，与 GPU 侧 #1
    同一个根因，见 core/gpu/gpu_distributed.py::_CompactMeshDataView
    文档）：此前 `self.n_cells = partition.n_local_cells`、
    `self.jacobians = local_mesh.jacobians` 直接原样转发——但
    `dist_fc`（`DistributedFlatFaceGeometry`）的 `owner_cell`/
    `neighbor_cell` 用的是 local+halo 压缩索引空间（且现在是"棱柱在前、
    四面体在后"排列，见 distributed_flat_face.py 模块文档），既不是
    `n_local_cells`（缺 halo 部分，界面项读 halo 邻居数据会越界/读错），
    也不是 `local_mesh.jacobians` 的原始单元编号（`local_mesh` 若是完整
    全局网格，其 jacobians 按*全局*单元编号索引，与压缩索引空间是两套
    不同的编号）。修复为按 `dist_fc.compact_global_ids`（"棱柱在前"
    local+halo 压缩索引空间每个位置对应的全局单元编号）从 `local_mesh`
    的全局 jacobians/cell_volumes 里重新抽取+重排，`n_cells`/
    `n_prism_cells` 也改用这套压缩索引空间对应的值（`dist_fc.base_flat.
    n_prism`）。

    要求 `local_mesh` 是**完整全局网格**（`build_distributed_flat_face`
    构造 `dist_fc` 时本来就要求这一点，见该函数文档"传统模式"一节）——
    "完全分布式加载"（只有 root 持有完整网格）模式下这个前提尚不成立，
    是仍然未解决的架构缺口，不在本次修复范围内。

    Attributes:
        partition: 分区信息
        dist_fc: 分布式面连接关系（"棱柱在前"压缩索引空间）
        local_mesh: 完整全局网格对象（提供 jacobians 等，见上方说明）
        n_cells: local+halo 压缩索引空间大小（不是 n_local_cells）
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
            local_mesh: 完整全局网格对象（提供 jacobians、face_flux_points
                等，见类文档"要求 local_mesh 是完整全局网格"一节）
            ops: FR 算子
        """
        self.partition = partition
        self.dist_fc = dist_fc
        self.local_mesh = local_mesh
        self.ops = ops

        compact_global_ids = dist_fc.compact_global_ids
        self.n_cells = len(compact_global_ids)
        self.n_halo_cells = partition.n_halo
        self.n_points_1d = local_mesh.n_points_1d
        self.n_sps_per_cell = local_mesh.n_sps_per_cell
        self.n_prism_cells = dist_fc.base_flat.n_prism

        n_sps = self.n_sps_per_cell

        # "完全分布式加载"模式（2026-09-02，见 distributed_mesh_loader.py::
        # PrecompactedMeshData 文档）：`local_mesh` 已经是 root rank 预先
        # 按 compact_global_ids 切好的紧凑数据，形状已经是
        # (n_compact, n_sps, ...)，不能也不需要再按 compact_global_ids
        # 二次索引（那会用 0..n_compact-1 的紧凑局部编号去索引一个已经
        # 只有 n_compact 行的数组，读到完全不对应的单元，且大多数情况下
        # compact_global_ids 的取值范围（真实全局单元编号，可能远大于
        # n_compact）会直接越界崩溃——用是否已经是 PrecompactedMeshData
        # 实例判断走哪条路径，两条路径产出的 self._jacobians 等最终形状
        # 完全一致，只是"传统模式"多一步"从全局数组按 compact_global_ids
        # 抽取"，这里已经在 root 侧做过了。
        from autoflowcfd.core.mpi.distributed_mesh_loader import PrecompactedMeshData
        is_precompacted = isinstance(local_mesh, PrecompactedMeshData)

        self._jacobians = None
        if getattr(local_mesh, 'jacobians', None) is not None:
            if is_precompacted:
                self._jacobians = local_mesh.jacobians
            else:
                det_jacs = local_mesh.jacobians['det_jacs'].reshape(local_mesh.n_cells, n_sps)
                inv_jacs = local_mesh.jacobians['inv_jacs'].reshape(local_mesh.n_cells, n_sps, 3, 3)
                self._jacobians = {
                    'det_jacs': det_jacs[compact_global_ids],
                    'inv_jacs': inv_jacs[compact_global_ids],
                }

        # #2 补充修复：compute_inviscid_residual_fr 的 over-integration
        # 分支（order>=1 时恒会走到，见该函数文档）读 `mesh.n_sps_per_cell_
        # fine`——这是与单元数量无关的标量（阶数决定），直接透传，不需要
        # 像 jacobians 那样按 compact_global_ids 重排。
        self.n_sps_per_cell_fine = getattr(local_mesh, 'n_sps_per_cell_fine', None)

        self._jacobians_fine = None
        if getattr(local_mesh, 'jacobians_fine', None) is not None:
            if is_precompacted:
                self._jacobians_fine = local_mesh.jacobians_fine
            else:
                n_fine = local_mesh.n_sps_per_cell_fine
                det_jacs_fine = local_mesh.jacobians_fine['det_jacs'].reshape(local_mesh.n_cells, n_fine)
                inv_jacs_fine = local_mesh.jacobians_fine['inv_jacs'].reshape(local_mesh.n_cells, n_fine, 3, 3)
                self._jacobians_fine = {
                    'det_jacs': det_jacs_fine[compact_global_ids],
                    'inv_jacs': inv_jacs_fine[compact_global_ids],
                }

        self._cell_volumes = None
        if getattr(local_mesh, 'cell_volumes', None) is not None:
            if is_precompacted:
                self._cell_volumes = local_mesh.cell_volumes
            else:
                self._cell_volumes = local_mesh.cell_volumes[compact_global_ids]

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
        """返回 Jacobian 信息，已按 local+halo 压缩索引空间重排（见类文档）。"""
        return self._jacobians

    @property
    def jacobians_fine(self):
        """返回 fine Jacobian 信息（用于 over-integration），已重排。"""
        return self._jacobians_fine

    @property
    def cell_volumes(self):
        """返回单元体积，已按 local+halo 压缩索引空间重排。"""
        return self._cell_volumes


def distributed_compute_inviscid_residual(
    U_local: np.ndarray,
    partition: DistributedPartition,
    halo_exchange: HaloExchange,
    dist_fc: DistributedFlatFaceGeometry,
    local_mesh,
    ops,
    boundary_ghost_provider: Optional[Callable] = None,
    mach_ref: float = 0.1,
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
        mach_ref: AUSM+up Weiss-Smith 预处理参考马赫数，调用方必须传入
            与单进程路径同一个 `solver.freestream["mach_ref"]`（各 rank
            必须用同一个值，否则分区边界两侧算出的 beta2 不一致，通量
            反对称性会被破坏——理由与本函数入参必须显式传递而非隐式
            读取全局状态的一般原则相同）。

    Returns:
        residual: (n_local_cells, n_sps, 5) local cells 的残差
    """
    from autoflowcfd.core.fr_residual.inviscid import compute_inviscid_residual_fr

    # 1. Halo 交换：获取 local + halo 的扩展状态（halo 交换协议原生排列，
    # local 在前、halo 在后——见 partition.py/halo.py，本函数不改动这个
    # 协议本身）
    U_extended = halo_exchange.exchange(U_local)

    # 2. 创建网格适配器（#2，2026-08-28：n_cells/jacobians 现在是
    # "棱柱在前"local+halo 压缩索引空间，见 DistributedMeshAdapter 文档）
    adapter = DistributedMeshAdapter(partition, dist_fc, local_mesh, ops)

    # 3. 把 halo 交换协议原生排列的 U_extended 重排到 adapter 实际使用的
    # "棱柱在前"压缩索引空间（与 GPU 侧 gpu_distributed.py::
    # compute_inviscid_residual_gpu 的 perm/inv_perm 重排同一个道理，见
    # distributed_flat_face.py::DistributedFlatFaceGeometry.perm 文档）。
    U_compact = U_extended[dist_fc.perm]

    # 4. 调用现有残差函数（flat_face_override=dist_fc.base_flat，不让
    # 它对 adapter 重新调用 get_flat_face_geometry——见 compute_
    # inviscid_residual_fr 该参数文档）
    residual_compact = compute_inviscid_residual_fr(
        U_compact, adapter, ops, boundary_ghost_provider, mach_ref=mach_ref,
        flat_face_override=dist_fc.base_flat,
    )

    # 5. 换回原生排列，再只取 local cells 的残差（与 self.U_gpu/U_local
    # 对齐的顺序，即 partition.local_cells 自身顺序）
    residual_native = residual_compact[dist_fc.inv_perm]
    residual_local = residual_native[:partition.n_local_cells]

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
    mu_t_field_compact: Optional[np.ndarray] = None,
    wmles_model=None,
    wall_distance_compact: Optional[np.ndarray] = None,
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

    湍流涡粘度耦合（2026-09-02 新增，见 core/mpi/distributed_turbulence.py
    模块文档）：`mu_t_field_compact` 非 None 时传给底层
    `compute_viscous_residual_ldg` 的 `mu_t_field` 参数——必须已经是
    compact 索引空间（local+halo，"棱柱在前"排列，与本函数内部的
    `U_compact` 同一个索引空间），由调用方（`distributed_compute_
    turbulence_source_and_viscosity`）在 halo 交换 k/omega 后算出。
    `DistributedFRSolver.__init__` 现在只接受 'none'/'SST'。

    Args:
        U_local: (n_local_cells, n_sps, 5) 本 rank 的 local cell 守恒变量
        partition: 分区信息
        halo_exchange: halo 交换管理器
        dist_fc: 分布式面连接关系
        local_mesh: 本地网格对象
        ops: FR 算子
        mu: 分子动力粘度（标量）
        boundary_ghost_provider: 边界幽灵态提供者
        mu_t_field_compact: (n_compact, n_sps) 湍流动力涡粘度场（compact
            索引空间），None 时等价于纯层流（此前唯一行为）
        wmles_model: WMLESModel 实例（2026-09-02 新增，见 `core/utils/
            solver_helpers.py::compute_wmles_wall_stress_correction`
            文档），None 时跳过壁面剪应力修正（此前唯一行为）
        wall_distance_compact: (n_compact, n_sps) 壁面距离场（compact
            索引空间，与 `mu_t_field_compact` 同一个索引空间），WMLES
            激活时必须提供

    Returns:
        residual: (n_local_cells, n_sps, 5) local cells 的粘性残差
    """
    import types
    from autoflowcfd.core.fr_residual.viscous import compute_viscous_residual as compute_viscous_residual_ldg
    from autoflowcfd.core.fr_residual.inviscid import conserved_to_primitive

    # 1. Halo 交换（原生排列，local 在前 halo 在后）
    U_extended = halo_exchange.exchange(U_local)

    # 2. 创建网格适配器（#2，2026-08-28：见 DistributedMeshAdapter 文档
    # "棱柱在前"压缩索引空间说明）
    adapter = DistributedMeshAdapter(partition, dist_fc, local_mesh, ops)

    # 3. 重排到 adapter 实际使用的"棱柱在前"压缩索引空间，与
    # distributed_compute_inviscid_residual 同一处理，见该函数文档。
    U_compact = U_extended[dist_fc.perm]

    # 4. 调用现有残差函数（state_Q 参数只为兼容旧签名保留，函数内部
    # 从 state_U 自行重新计算 primitive 变量，见该函数文档；
    # flat_face_override=dist_fc.base_flat 避免对 adapter 重新调用
    # get_flat_face_geometry，见该参数文档）
    Q_compact = conserved_to_primitive(U_compact[..., :5])
    residual_compact = compute_viscous_residual_ldg(
        U_compact, Q_compact, ops, adapter, mu=mu,
        mu_t_field=mu_t_field_compact,
        boundary_ghost_provider=boundary_ghost_provider,
        flat_face_override=dist_fc.base_flat,
    )

    # 4b. WMLES 壁面剪应力修正（2026-09-02）：与单机路径
    # `FRSolver.compute_viscous_residual` 同一个施加时机（残差组装
    # 阶段，时间积分之前）——复用同一个 CPU 核心函数，
    # `flat_face_override=dist_fc.base_flat` 与上面粘性残差调用同一个
    # 对象，避免它退回到对 `adapter` 重新调用
    # `get_flat_face_geometry`（那样会因 `adapter.face_flux_points`
    # 是"传统模式"下真实全局 mesh 的 face_flux_points 对象、但这里的
    # 面索引是 compact 空间而语义错位——与 `_compute_omega_wall_target`
    # 此前的同一类 bug，见该函数文档）。
    if wmles_model is not None:
        facade = types.SimpleNamespace(
            wmles_model=wmles_model, mesh=adapter, ops=ops,
            wall_distance=wall_distance_compact,
            state=types.SimpleNamespace(U=U_compact, Q=Q_compact),
            boundary_ghost_provider=boundary_ghost_provider,
        )
        from autoflowcfd.core.utils.solver_helpers import compute_wmles_wall_stress_correction
        correction_compact = compute_wmles_wall_stress_correction(facade, flat_face_override=dist_fc.base_flat)
        if correction_compact is not None:
            residual_compact = residual_compact + correction_compact[..., :residual_compact.shape[-1]]

    # 5. 换回原生排列，再只返回 local cells 的残差
    residual_native = residual_compact[dist_fc.inv_perm]
    residual_local = residual_native[:partition.n_local_cells]

    return residual_local


def distributed_compute_physical_gradient(
    U_local: np.ndarray,
    partition: DistributedPartition,
    halo_exchange: HaloExchange,
    dist_fc: DistributedFlatFaceGeometry,
    local_mesh,
    ops,
) -> np.ndarray:
    """分布式物理梯度计算。

    真实 bug 修复（#2，2026-08-28）：此前 `DistributedMeshAdapter(partition,
    None, local_mesh, ops)` 传 `dist_fc=None`——这个函数目前在代码库里
    没有任何调用点（死代码），但 `DistributedMeshAdapter` 现在的构造
    （见该类文档）需要从 `dist_fc.compact_global_ids` 取"棱柱在前"
    local+halo 压缩索引空间的单元编号来重排 jacobians，传 None 会在
    `None.compact_global_ids` 直接 AttributeError。补上 `dist_fc` 参数，
    与另外两个 `distributed_compute_*_residual` 函数保持同一套接口，
    不留這个未来调用点必现崩溃的隐患。

    Args:
        U_local: (n_local_cells, n_sps, 5) 本 rank 的 local cell 守恒变量
        partition: 分区信息
        halo_exchange: halo 交换管理器
        dist_fc: 分布式面连接关系
        local_mesh: 完整全局网格对象（见 DistributedMeshAdapter 文档）
        ops: FR 算子

    Returns:
        grad_U: (n_local_cells, n_sps, 3, 5) local cells 的梯度
    """
    from autoflowcfd.core.fr_operators.gradients import compute_physical_gradient

    # 1. Halo 交换（原生排列）
    U_extended = halo_exchange.exchange(U_local)

    # 2. 创建网格适配器 + 重排到压缩索引空间，与其余两个
    # distributed_compute_*_residual 函数同一处理。
    adapter = DistributedMeshAdapter(partition, dist_fc, local_mesh, ops)
    U_compact = U_extended[dist_fc.perm]

    # 3. 调用现有梯度函数
    grad_U_compact = compute_physical_gradient(U_compact, adapter, ops)

    # 4. 换回原生排列，只返回 local cells 的梯度
    grad_U_native = grad_U_compact[dist_fc.inv_perm]
    grad_U_local = grad_U_native[:partition.n_local_cells]

    return grad_U_local


# 分布式湍流输运（k/omega 对流+扩散）已接入（2026-09-02，SST 模型）：
# 见 core/mpi/distributed_turbulence.py::
# distributed_compute_turbulence_source_and_viscosity——独立于本文件
# （k/omega 走自己的 2-var halo 交换，不是 7-var 分布式状态；本文件的
# `distributed_compute_viscous_residual` 新增 `mu_t_field_compact` 参数
# 消费其产出）。DDES/IDDES/WMLES/LES 分布式支持仍未实现，见该模块文档
# "范围边界"一节。
