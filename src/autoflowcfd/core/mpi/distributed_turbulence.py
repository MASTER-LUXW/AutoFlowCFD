"""AutoFlowCFD V2.0 - 分布式 SST/DDES/IDDES 湍流模型（CPU MPI 路径）。

真正接入分布式状态与残差计算（2026-09-02）——此前 `DistributedFRSolver`/
`MultiGPUDistributedSolver` 都在构造时对非 'none' 湍流模型 fail-fast
拒绝（见 distributed_solver.py/gpu_distributed.py 构造器文档），本模块
是真正的实现。

范围更新（2026-09-02 续，"过时信息源"教训——见
ProjectFiles/V2.0/19_重大问题修复-湍流模型跨后端逐项排查与完全分布式加载补齐.md）：
本文件只覆盖 SST/DDES/IDDES 的 k/omega 输运本身；WMLES 不需要这套
k/omega 输运（没有 ODE 状态），它的壁面剪应力修正
（`core/utils/solver_helpers.py::compute_wmles_wall_stress_correction`）
已独立接入 `DistributedFRSolver.__init__`/`MultiGPUDistributedSolver.
__init__`（"传统模式"两条路径均已支持，"完全分布式加载"仍不支持，见
`distributed_solver.py::from_fully_distributed_package` 文档），不经过
本文件——本文件此前"WMLES 需要额外的分布式面外插"的表述已经过时（
真正的障碍是 `compute_wmles_wall_stress_correction` 没有跟随
`flat_face_override` 约定，不是需要全新的面外插基础设施，现已修复）。
LES（WALE）同样不经过本文件的 SST/DDES/IDDES 主函数（没有 ODE 状态，
纯代数现算），走独立的 `distributed_compute_les_viscosity`——已在
`MultiGPUDistributedSolver`（多GPU分布式）与本文件（CPU MPI，2026-
09-02 续接）两条路径都接入。

设计原则：**复用**而不是重新实现单机 SST 的完整数值逻辑
（`core/fr_solver/turbulence.py::compute_turbulence_source`、
`core/turbulence/sst.py::SSTModelFR.compute_source_terms/update_fields`、
`core/turbulence/transport.py::compute_turbulence_transport_residual`）——
这些函数本身已经是纯 (状态, 网格, 算子) -> 残差的函数式接口（唯一状态
来自传入的 `solver`-like 对象），分布式版本只需要：
1. 对 k/omega 做与平均流 U 同一套 halo 交换 + compact 索引空间重排
   （复用 `DistributedMeshAdapter`，与 distributed_compute.py 的
   `distributed_compute_inviscid_residual` 等函数同一个模式）。
2. 构造一个满足 `compute_turbulence_source` 所需鸭子类型接口的适配器
   对象（`DistributedTurbulenceSolverAdapter`），把 compact 索引空间
   的 mesh/state/wall_distance/turb_model 喂给它。
3. 调用完全相同的 `compute_turbulence_source`，得到更新后的 k/omega
   （compact 索引空间）与 dk/dt、domega/dt——只把 **local cells** 的
   结果写回真正的（只有 n_local 大小的）`turb_model`。
4. 额外产出 `mu_t_field`（compact 索引空间，供平均流粘性残差的 BR1
   界面项使用邻居侧涡粘）。

这样不需要重新审查/复刻 SST 那套精细的正性限制、点隐式阻尼、DDES 长度
尺度耦合时序等已经过大量真实网格调试的数值细节——分布式路径与单机路径
在这些数值算法层面逐字共享同一份代码，只有"怎么把 halo 数据喂进去"
不同。
"""

import numpy as np
from types import SimpleNamespace
from typing import Optional

from autoflowcfd.core.mpi.partition import DistributedPartition
from autoflowcfd.core.mpi.halo import HaloExchange
from autoflowcfd.core.mpi.distributed_flat_face import DistributedFlatFaceGeometry
from autoflowcfd.core.mpi.distributed_compute import DistributedMeshAdapter


def compute_distributed_wall_distance(dist_fc, global_mesh, source) -> np.ndarray:
    """compact 索引空间（local + halo，"棱柱在前"）解点上的壁面距离。

    与单机/GPU 同一个来源（`core/utils/wall_distance_source.py`）在本 rank
    compact 单元的解点坐标上查询——halo 单元也要有真实壁距，SST 涡粘与混合
    函数在 halo 上的值要供平均流粘性残差的 BR1 界面项读取。

    **2026-09-25 修复**：此前这里自己从 `mesh.boundary_groups` 找壁面节点，
    把 `BoundaryMap.groups` 的数组当字典读（真实网格上构造即 AttributeError），
    且找不到壁面时静默退回"单元特征长度"。现在来源由 CLI 从体网格的 WALL
    边界面构造、一路传进来，没有来源就报错。

    Args:
        global_mesh: 完整全局网格（传统模式每个 rank 都有；完全分布式加载时
            只有 root 有，root 算好按 compact 切片放进各 rank 的包里）。
    """
    if source is None:
        raise RuntimeError(
            "分布式湍流求解需要壁面距离来源（WallDistanceSource），调用方没有提供"
            "——不退化为特征长度估计。CLI 各分布式分支都应由体网格构造并传入。")
    sps = getattr(global_mesh, "sps_coords", None)
    if sps is None:
        raise RuntimeError("全局网格没有 sps_coords，无法在解点上查询壁面距离")
    return source.query(np.asarray(sps)[dist_fc.compact_global_ids])


class DistributedTurbulenceSolverAdapter:
    """满足 `compute_turbulence_source`/`compute_turbulence_transport_
    residual` 鸭子类型接口的最小适配器（compact 索引空间）。

    只暴露这两个函数实际读取的属性/方法（`mesh`、`ops`、`turb_model`、
    `turb_model_name`、`ddes_model`、`wall_distance`、`mu_molecular`、
    `state.Q`/`state.n_cells`/`state.n_sps`、`state.U`、
    `_compute_gradients()`、`_get_cell_volumes()`、
    `_turbulence_flat_face_override`），不是完整 FRSolver 接口。

    DDES/IDDES 支持（2026-09-02 补齐）：`compute_turbulence_source`
    对 SST/DDES/IDDES 三者复用完全相同的 k/omega 梯度/输运代码路径
    （只按 `solver.turb_model_name in ["SST","DDES","IDDES"]` 判断，
    三者结果相同），DES 特有的长度尺度替换是单独一段、只按
    `solver.ddes_model is not None`（及 `isinstance(..., IDDESModel)`
    进一步区分 DDES/IDDES）触发——与本适配器已有的"复用单机函数式接口，
    不重新实现数值逻辑"设计完全一致，只需要真正传入 `ddes_model`（而
    不是硬编码 `None`）即可。IDDES 额外需要的 `_iddes_h_max`/
    `_iddes_h_wn`（逐单元网格边长几何量，见 `des.py::
    compute_h_max_and_h_wn` 文档——纯粹是每个单元自身节点坐标的函数，
    与相邻单元/分区无关，本来就可以直接对"传统模式"下每个 rank 都持有
    的完整全局网格算一次、再按 compact_global_ids 切片，不需要新的
    跨 rank 几何交换基础设施）同样通过构造参数传入。
    """

    def __init__(
        self, mesh_adapter, ops, turb_model, wall_distance, mu_molecular, flat_face_override,
        turb_ramp_step=10 ** 9, turb_ramp_steps=0,
        turb_model_name="SST", ddes_model=None, iddes_h_max=None, iddes_h_wn=None,
        boundary_ghost_provider=None,
    ):
        """`boundary_ghost_provider` 必须是 `group_code` 已重切到 compact 面空间
        的那一份（`DistributedFRSolver.local_solver.boundary_ghost_provider`）。

        **真实缺陷（2026-09-25）**：此前适配器没有这个属性，于是湍流输运里
        一切按边界组类型取的条件——壁面 k=0、omega 的 Wilcox 壁面值（对流
        ghost + 显式壁面松弛）、以及来流条件——在 CPU 分布式路径上全部按
        "没有边界分组"退化成零梯度，静默地与单机不一致。单机与分布式的对照
        测试此前只覆盖层流，所以一直没有暴露。
        """
        self.mesh = mesh_adapter
        self.boundary_ghost_provider = boundary_ghost_provider
        self.ops = ops
        self.turb_model = turb_model
        self.turb_model_name = turb_model_name
        self.ddes_model = ddes_model
        self._iddes_h_max = iddes_h_max
        self._iddes_h_wn = iddes_h_wn
        self.wall_distance = wall_distance
        self.mu_molecular = mu_molecular
        self._turbulence_flat_face_override = flat_face_override
        # 产项渐变因子状态（见 fr_solver/turbulence.py::_update_production_ramp）
        # ——调用方（distributed_compute_turbulence_source_and_viscosity）
        # 从真正的分布式求解器读取当前迭代步数/渐变总步数并传入，默认值
        # （ramp_step 远大于 ramp_steps=0）等价于"渐变已完成、production_
        # factor=1.0"，只在调用方没有显式提供时才是这个默认行为。
        self._turb_ramp_step = turb_ramp_step
        self._turb_production_ramp_steps = turb_ramp_steps

    def set_state(self, U_compact: np.ndarray):
        from autoflowcfd.core.fr_residual.inviscid import conserved_to_primitive
        n_cells, n_sps = U_compact.shape[:2]
        Q = conserved_to_primitive(U_compact[..., :5])
        self.state = SimpleNamespace(U=U_compact, Q=Q, n_cells=n_cells, n_sps=n_sps)

    def _compute_gradients(self):
        from autoflowcfd.core.fr_operators.gradients import compute_physical_gradient
        return compute_physical_gradient(self.state.U[..., :5], self.mesh, self.ops)

    def _get_cell_volumes(self):
        return self.mesh.cell_volumes


def build_distributed_turbulence_view(
    U_local, partition, halo_exchange, turb_halo_exchange, dist_fc, local_mesh, ops,
    turb_model, mu, wall_distance_compact, *, turb_ramp_step, turb_ramp_steps,
    turb_model_name, ddes_model, iddes_h_max_compact, iddes_h_wn_compact,
    des_length_scale_halo_exchange, boundary_ghost_provider,
):
    """compact 索引空间（local + halo）上的湍流求值"视图"，返回 `(adapter, turb_view)`。

    `adapter` 满足单机 `fr_solver/turbulence/source.py` 求值件的鸭子类型接口，
    `turb_view` 是 n_compact 大小的临时 `SSTModelFR`（k/omega 经 halo 交换填好）。
    显式更新（`distributed_compute_turbulence_source_and_viscosity`）与隐式
    k-omega Newton（`distributed_implicit_turbulence.py`）共用这一份构造。
    """
    n_sps = local_mesh.n_sps_per_cell

    # 1. 平均流 halo 交换（与 distributed_compute_inviscid_residual 同一
    # 个模式）——只为了拿 compact 索引空间的 rho/velocity 用于 grad_vel。
    U_extended = halo_exchange.exchange(U_local)
    U_compact = U_extended[dist_fc.perm]

    # 2. k/omega halo 交换（独立于平均流的 2-var 交换）。
    k_omega_local = np.stack([turb_model.k_field, turb_model.omega_field], axis=-1)  # (n_local,n_sps,2)
    k_omega_extended = turb_halo_exchange.exchange(k_omega_local)
    k_omega_compact = k_omega_extended[dist_fc.perm]

    mesh_adapter = DistributedMeshAdapter(partition, dist_fc, local_mesh, ops)
    n_compact = mesh_adapter.n_cells

    # 3. 构造 compact 索引空间的临时 SSTModelFR"视图"——不能直接复用
    # 真正的 turb_model（那个只有 n_local 大小），需要一个同形状为
    # n_compact 的临时对象供 compute_source_terms/update_fields 读写，
    # 用完只把 local 部分写回真正的 turb_model。
    from autoflowcfd.core.turbulence.sst import SSTModelFR
    # k_inf/omega_inf 不只是初值：开边界来流 ghost 取它们作为来流值
    # （`transport/residual.py`），必须与真正的模型一致——2026-09-25 前视图
    # 用构造默认值 1e-6/1.0，分布式来流面上的 k/omega 被当成近乎零的来流。
    turb_view = SSTModelFR(n_compact, n_sps, k_inf=turb_model.k_inf, omega_inf=turb_model.omega_inf)
    # 复制真正模型当前的可调常数/状态标记（k_field/omega_field 下面整体覆盖）。
    for attr in (
        "sigma_k1", "sigma_k2", "sigma_w1", "sigma_w2", "beta1", "beta2",
        "a1", "kappa", "beta_star", "k_max", "omega_max", "production_factor",
    ):
        if hasattr(turb_model, attr):
            setattr(turb_view, attr, getattr(turb_model, attr))

    # 真实 bug 修复（2026-09-02，续查"完全分布式加载模式下的DDES/IDDES"
    # 时，用两次连续调用才测出来——第一次调用之前从未被本模块任何测试
    # 覆盖过）：`des_length_scale` 是跨步持久状态（本步 apply_to_sst_
    # model[_iddes] 用本步 nu_t 算出、写回 turb_model.des_length_scale
    # 供*下一步*读取，见下方"写回"一节），此前这里直接把它（n_local
    # 大小）原样 setattr 到 turb_view（n_compact 大小）——第一次调用时
    # `des_length_scale` 还是 None，`setattr` 不会出错，掩盖了这个问题；
    # 从第二次调用开始，`compute_source_terms` 内部 `rho*k_safe**1.5/
    # des_length_scale` 这类逐元素运算会因 (n_local,n_sps) 与
    # (n_compact,n_sps) 形状不匹配直接 ValueError（真实复现：2-rank
    # 合成网格上第二次调用必现）。修复为与 k_field/omega_field 完全
    # 同一套"先 native 顺序、halo 交换、再 permute 到 compact"处理——
    # `des_length_scale` 本身只有 1 个分量，不能直接复用 2-var 的
    # `turb_halo_exchange`，调用方（`distributed_solver.py`）需要额外
    # 提供一个 1-var 的 halo 交换器（None 时表示尚未配置，退化为
    # "没有跨步长度尺度记忆"，与 CPU 单机路径首次调用的行为一致，不是
    # 新的简化）。
    if getattr(turb_model, "des_length_scale", None) is not None and des_length_scale_halo_exchange is not None:
        des_length_scale_extended = des_length_scale_halo_exchange.exchange(
            turb_model.des_length_scale[:, :, None]
        )[:, :, 0]
        turb_view.des_length_scale = des_length_scale_extended[dist_fc.perm]
    else:
        turb_view.des_length_scale = None

    turb_view.k_field = k_omega_compact[..., 0].copy()
    turb_view.omega_field = k_omega_compact[..., 1].copy()

    if ddes_model is not None:
        from autoflowcfd.core.turbulence.des import IDDESModel
        if isinstance(ddes_model, IDDESModel):
            if iddes_h_max_compact is None or iddes_h_wn_compact is None:
                raise RuntimeError(
                    "distributed_compute_turbulence_source_and_viscosity: "
                    "ddes_model 是 IDDESModel 实例，但 iddes_h_max_compact/"
                    "iddes_h_wn_compact 未提供——IDDES 的长度尺度公式离不开"
                    "这两个几何量，不接受静默退化。"
                )

    adapter = DistributedTurbulenceSolverAdapter(
        mesh_adapter, ops, turb_view, wall_distance_compact, mu, dist_fc.base_flat,
        turb_ramp_step=turb_ramp_step, turb_ramp_steps=turb_ramp_steps,
        turb_model_name=turb_model_name, ddes_model=ddes_model,
        iddes_h_max=iddes_h_max_compact, iddes_h_wn=iddes_h_wn_compact,
        boundary_ghost_provider=boundary_ghost_provider,
    )
    adapter.set_state(U_compact)
    return adapter, turb_view


def set_view_k_omega(turb_view, turb_halo_exchange, dist_fc, k_local, omega_local) -> None:
    """把 local 的 k/omega（native 排列）经 halo 交换写进 compact 视图。"""
    k_omega_local = np.stack([k_local, omega_local], axis=-1)
    k_omega_compact = turb_halo_exchange.exchange(k_omega_local)[dist_fc.perm]
    turb_view.k_field = k_omega_compact[..., 0].copy()
    turb_view.omega_field = k_omega_compact[..., 1].copy()


def distributed_compute_turbulence_source_and_viscosity(
    U_local: np.ndarray,
    partition: DistributedPartition,
    halo_exchange: HaloExchange,
    turb_halo_exchange: HaloExchange,
    dist_fc: DistributedFlatFaceGeometry,
    local_mesh,
    ops,
    turb_model,
    mu: float,
    wall_distance_compact: np.ndarray,
    dt_local: np.ndarray,
    turb_ramp_step: int = 10 ** 9,
    turb_ramp_steps: int = 0,
    turb_model_name: str = "SST",
    ddes_model=None,
    iddes_h_max_compact: Optional[np.ndarray] = None,
    iddes_h_wn_compact: Optional[np.ndarray] = None,
    des_length_scale_halo_exchange: Optional[HaloExchange] = None,
    boundary_ghost_provider=None,
) -> "tuple[np.ndarray, int]":
    """分布式 SST/DDES/IDDES 源项+输运计算，就地更新 `turb_model`
    （local cells），返回 compact 索引空间的 `mu_t_field`（供平均流
    粘性残差消费）。

    Args:
        U_local: (n_local, n_sps, 5) 本 rank 的平均流守恒变量（用于算
            grad_vel 与 rho/velocity，不需要湍流分量）
        halo_exchange: 5-var halo 交换器（与平均流残差共用同一个）
        turb_halo_exchange: 2-var halo 交换器（k, omega），独立于平均流
            的 halo 交换——k/omega 是 turb_model 自己的状态，不在 U 里
        turb_model: 本 rank 的 SSTModelFR 实例（n_local 大小），本函数
            会就地更新它的 k_field/omega_field/nu_t
        dt_local: (n_local, n_sps) 湍流场显式更新用的局部时间步长（与
            平均流共用同一套 cfl.py 逻辑算出的值，调用方负责提供——
            单机路径的对应设计见 step.py 模块文档"关键修复"一节，分布式
            路径目前用全局固定 dt 广播成 (n_local,n_sps)，与
            DistributedFRSolver.step() 现有的"分布式路径目前用全局固定
            步长"简化一致，不是本次新引入的简化）
        turb_model_name: "SST"/"DDES"/"IDDES"（2026-09-02 新增）——三者
            在 `compute_turbulence_source` 内部走的是完全相同的 k/omega
            梯度/输运代码路径，只有 DES 长度尺度替换那一段单独按
            `ddes_model is not None` 触发，这个参数本身对计算结果没有
            直接影响，只是如实传给 adapter（保持鸭子类型属性语义正确，
            不是死参数）。
        ddes_model: `DDESModel`/`IDDESModel` 实例（None 时退化为纯 SST，
            与此前行为一致）——调用方（`DistributedFRSolver`）持有的
            真实对象，本函数只是每步复用它，不重新构造。
        iddes_h_max_compact, iddes_h_wn_compact: (n_compact,) IDDES
            专用的逐单元网格边长几何量（`des.py::compute_h_max_and_h_wn`
            的输出按 `dist_fc.compact_global_ids` 切片），`ddes_model`
            是 `IDDESModel` 实例时必须提供，否则 `compute_turbulence_
            source` 内部 `isinstance(solver.ddes_model, IDDESModel)`
            分支会读到 `None` 直接崩溃（有意不做静默兜底，缺几何数据
            不该悄悄退化成别的行为）。

    Returns:
        (mu_t_field_compact, next_turb_ramp_step)：`mu_t_field_compact`
        是 (n_compact, n_sps) 供 distributed_compute_viscous_residual
        的 mu_t_field 参数使用；`next_turb_ramp_step` 是调用方需要写回
        `self._turb_ramp_step`（供下一步调用）的递增计数（见
        `_update_production_ramp` 文档"递增计数器"一节——本函数内部的
        adapter 对象每步都重新构造，不会自动持久化这个计数）。
    """
    n_local = partition.n_local_cells
    n_sps = local_mesh.n_sps_per_cell
    adapter, turb_view = build_distributed_turbulence_view(
        U_local, partition, halo_exchange, turb_halo_exchange, dist_fc, local_mesh, ops,
        turb_model, mu, wall_distance_compact, turb_ramp_step=turb_ramp_step,
        turb_ramp_steps=turb_ramp_steps, turb_model_name=turb_model_name, ddes_model=ddes_model,
        iddes_h_max_compact=iddes_h_max_compact, iddes_h_wn_compact=iddes_h_wn_compact,
        des_length_scale_halo_exchange=des_length_scale_halo_exchange,
        boundary_ghost_provider=boundary_ghost_provider,
    )

    # dt_local 是 local cells 的局部步长；compact 索引空间里 halo cells
    # 的"更新"结果本来就会被丢弃（只写回 local 部分），halo 位置的 dt
    # 取值本身不影响 local cells 的结果（update_fields 是逐 cell 独立的
    # 点态更新，不同 cell 之间没有耦合）——用 1.0 占位即可，不需要真实
    # halo dt_local。
    #
    # 关键：`dist_fc.perm` 把 native 顺序（local_cells 在前、halo_cells
    # 在后拼接，即 native_ids = concat([local_cells, halo_cells])）重排
    # 成 compact 顺序（按单元类型"棱柱在前"分组，不保留 local/halo 的
    # 分界！）。因此 compact 位置 i<n_local 并不保证是 local cell——
    # 之前这里直接写 `dt_local_compact[:n_local] = dt_local` 是一个真实
    # bug：一旦某个 local cell（如四面体）在 compact 排序里落到了
    # >=n_local 的位置（因为某个 halo 棱柱单元占据了前面的位置），这个
    # local cell 就会被错误地分配到 dt=1.0 占位值而不是真实的
    # dt=1e-5——对显式 Euler 更新来说相当于步长放大 10 万倍，导致该 cell
    # 的 k/omega 灾难性发散（触发正性下限 1e-12 或飙出巨大值），这正是
    # 本模块 SST k/omega 输运"残留数值差异"长期未解决的根因（2026-09-02
    # 定位修复）。正确做法：先在 native 顺序里赋值（与
    # U_compact/k_omega_compact 完全同一套构造方式），再套用同一个
    # `perm` 重排到 compact 顺序。
    n_native = partition.n_local_cells + partition.n_halo
    dt_native = np.ones((n_native, n_sps))
    dt_native[:n_local] = dt_local  # halo_cells 部分的占位值随后被丢弃，只是避免未初始化内存参与计算
    dt_local_compact = dt_native[dist_fc.perm]

    from autoflowcfd.core.fr_solver.turbulence import compute_turbulence_source
    compute_turbulence_source(adapter, dt_local_compact)

    # 4. 只把 local cells 的更新结果写回真正的 turb_model（native 排列，
    # 见 dist_fc.inv_perm 文档——turb_view.k_field 是 compact 排列，需要
    # 先换回原生排列再切 [:n_local]，与平均流残差同一处理）。
    k_native = turb_view.k_field[dist_fc.inv_perm]
    omega_native = turb_view.omega_field[dist_fc.inv_perm]
    turb_model.k_field[:] = k_native[:n_local]
    turb_model.omega_field[:] = omega_native[:n_local]
    nu_t_native = turb_view.nu_t[dist_fc.inv_perm]
    turb_model.nu_t[:] = nu_t_native[:n_local]

    # DDES/IDDES：des_length_scale 是跨步持久状态（本步 apply_to_sst_
    # model/_iddes 用本步 nu_t/grad_vel 算出的 l_eff，供*下一步*
    # compute_source_terms 读取，见类文档"计算顺序"一节）——上面的
    # k_field/omega_field/nu_t 写回都处理了，这个此前遗漏（2026-09-02
    # 补齐，与那两个同一个 native/compact 重排写回模式）。
    if ddes_model is not None and getattr(turb_view, "des_length_scale", None) is not None:
        des_length_scale_native = turb_view.des_length_scale[dist_fc.inv_perm]
        turb_model.des_length_scale = des_length_scale_native[:n_local].copy()

    # mu_t_field 保持 compact 排列（平均流粘性残差直接消费 compact 索引
    # 空间的场，见 distributed_compute_viscous_residual 里 Q_compact 的
    # 同一处理）。
    rho_compact = adapter.state.Q[..., 0]
    mu_t_field_compact = rho_compact * turb_view.nu_t
    return mu_t_field_compact, adapter._turb_ramp_step


def distributed_compute_les_viscosity(
    U_local: np.ndarray,
    partition: DistributedPartition,
    halo_exchange: HaloExchange,
    dist_fc: DistributedFlatFaceGeometry,
    local_mesh,
    ops,
    sgs_model,
) -> np.ndarray:
    """分布式 LES（WALE）涡粘度计算（2026-09-02，CPU MPI 路径补齐——
    此前只有多GPU分布式接入了 LES，见本文件模块文档"范围更新"一节）。

    与 SST/DDES/IDDES 不同，WALE 是纯代数模型（没有跨步 k/omega 那样的
    ODE 积分状态），"每步开场用当前状态现算"与单机路径"缓存自上一步
    末尾"在数学上完全等价（见 `multi_gpu_sst_turbulence` 记忆里同一个
    结论）——因此这里不需要构造类似 SST 的临时 compact"视图"对象、
    不需要跨步持久状态的 halo 交换，只需要：halo 交换平均流 U -> compact
    索引空间重排 -> 算物理速度梯度 -> 调用 `WALEModel.compute_eddy_
    viscosity` -> 换算成 `mu_t = rho * nu_t` 直接返回。

    Args:
        U_local: (n_local, n_sps, 5) 本 rank 的平均流守恒变量
        halo_exchange: 5-var halo 交换器（与平均流残差共用同一个）
        sgs_model: `WALEModel`（或其他 SGS 模型）实例，调用方持有的
            真实对象——本函数只调用它的 `compute_eddy_viscosity`，不
            修改它的状态（WALE 本身也没有需要持久化的状态）。

    Returns:
        mu_t_field_compact: (n_compact, n_sps)，供 `distributed_compute_
        viscous_residual` 的 `mu_t_field_compact` 参数直接使用。
    """
    from autoflowcfd.core.fr_residual.viscous import compute_gradients
    from autoflowcfd.core.fr_residual.inviscid import conserved_to_primitive

    U_extended = halo_exchange.exchange(U_local)
    U_compact = U_extended[dist_fc.perm]

    mesh_adapter = DistributedMeshAdapter(partition, dist_fc, local_mesh, ops)
    n_sps = local_mesh.n_sps_per_cell

    # 真实 bug 修复（2026-09-03，同 fr_solver/turbulence.py::
    # compute_turbulence_source 文档同一处）：此前对*守恒*变量 U_compact
    # 求梯度再切片动量分量冒充速度梯度——grad(rho*u) != rho*grad(u)，
    # 除非密度梯度处处为零。改为先转成原始变量 Q_compact 再对速度分量
    # 求梯度（提前到这里，供下面 rho_compact 复用同一份）。
    Q_compact = conserved_to_primitive(U_compact[..., :5])
    grad_vel = compute_gradients(Q_compact[..., 1:4], ops, mesh_adapter)

    cell_volumes = mesh_adapter.cell_volumes
    delta = np.power(np.abs(cell_volumes), 1.0 / 3.0)
    delta = np.tile(delta[:, None], (1, n_sps))

    nu_t = sgs_model.compute_eddy_viscosity(grad_vel, delta)
    rho_compact = Q_compact[..., 0]
    return rho_compact * nu_t
