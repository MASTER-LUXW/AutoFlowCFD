"""分布式路径的逐单元局部 CFL 时间步长（2026-09-14）。

## 为什么有这个模块

在此之前，两条分布式路径（CPU MPI"传统模式"/"完全分布式加载"，以及
多 GPU）都用**全局固定 dt**：`DistributedFRSolver.step(dt)` 与
`MultiGPUDistributedSolver.step(dt)` 直接把调用方传进来的物理时间步长
铺满所有单元，不做单机路径那种逐 cell 局部 CFL。源码里当时把它记作
"已接受的简化"。

这条"简化"有三重真实代价，不是风格问题：

1. **步长被全场最苛刻的单元卡死**。局部 CFL 的全部意义就是让每个单元
   按自己的谱半径取步长；改成全局统一值后，只要有一个薄边界层棱柱或
   一个退化坍缩坐标 SP，整个计算域都被拖到那个单元的步长上。79 万单元
   cube_demo 实测局部 dt 的 min/max 相差**上千倍**，全局固定步长等于把
   绝大多数单元的推进效率按最坏单元打折。
2. **几何/度量 CFL 限制（`dt_geometric`）完全失效**。那一项是逐 SP 的
   （坍缩坐标下同一单元内不同 SP 的 det(J) 可差几百倍），它存在的目的
   正是防住项目记忆 `tet_collapsed_coord_anisotropy` 记录的那类退化 SP
   局部刚性失稳。没有局部步长就没有这一层保护。
3. **自适应 CFL 与低马赫数预处理在分布式路径上都无从生效**。前者按
   残差历史调节 CFL 数，后者的全部收益来自"可以按预处理波速取更大的
   dt"——两者都以"存在一个由 CFL 数决定的局部步长"为前提。

## 实现方式：复用单机那一份公式，不另写一套

本模块**不重新实现** CFL 公式，而是构造一个"solver 视图"
（`DistributedCFLView`），把分布式紧凑索引空间（local+halo，"棱柱在前"
排列，见 `distributed_flat_face.py`）包装成 `cfl.py::
compute_local_time_step` 期望的那 11 个属性，然后调用**同一个函数**。
这样做的理由与 `DistributedMeshAdapter` 把分布式数据包装成
`HighOrderMesh` 接口、让残差函数原样复用是同一个：公式一致性由构造
保证，而不是靠两份实现之间的人工同步（那正是本项目反复踩过的坑——
2026-08-25 GPU 侧 CFL 用预处理声速的 bug 就是"CPU 修了、GPU 那份没跟上"）。

## 为什么在 local+halo 紧凑空间上算，再切回 local

一个 local 单元的 dt 需要：
- 它自己的体积/det(J)/度量标度/rho/mu_eff —— 全是本地量；
- **它所有面的谱半径之和**，而这需要邻居单元的速度与声速——包括位于
  halo 的邻居。halo 交换产出的扩展状态正好提供这些。
`partition.local_faces` 包含"任一侧为 local"的全部面（见
`partition.py` 的 `halo_owner_mask` 修复），所以在紧凑空间上算完之后，
落在 local 单元上的谱半径求和是**完整**的。落在 halo 单元上的那些 dt
值是不完整的（它们的部分面归别的 rank），但一律被丢弃。

**切片必须先换回原生排列**（这是本模块最容易写错的一处，第一版就错了、
被逐单元比对测试当场抓住）：紧凑排列是"棱柱在前、四面体在后"，它把
local 与 halo 单元**混在一起**重排——实测 4 单元算例上
`compact_global_ids=[0,1,2,3]` 而 `local_cells=[0,2]`，local 单元根本
不在紧凑空间的前 `n_local` 段。正确做法是 `dt_compact[inv_perm]` 换回
原生排列（[0,n_local)=local_cells 自身顺序）之后再切 `[:n_local]`，
与残差路径"算完用 inv_perm 换回原生排列"完全一致。

粘性限制（`dt_visc`）与几何限制（`dt_geometric`）都是纯逐单元的、不含
邻居耦合，因此涡粘场只需要 local 部分：halo 段用分子粘度填充即可
（那段结果本来就要丢弃）。这一点让本模块不需要为 mu_t 再加一次
halo 交换。

## 本机验证能力

本机没有 mpi4py/CUDA，无法跑真正的多 rank 通信。验证方式与本项目既有
分布式测试一致（见 `tests/unit/test_distributed_compute_residual.py`
模块文档）：用假 halo 交换器提供已知正确的扩展状态，然后把分布式路径
算出的"某个 rank 的 local cells 的 dt"与**单机路径**
`compute_local_time_step` 在同一网格上算出的、对应同一批全局单元的 dt
逐单元比对。这个判据比"都是正数"强得多——它能抓出索引空间重排、面
归属、halo 对齐上的任何错误。见
`tests/unit/test_distributed_cfl.py`。
"""
from typing import Optional

import numpy as np


class _DistributedFaceConnectivityView:
    """把分布式面几何包装成 `cfl.py` 需要的 face_connectivity 接口。

    `cfl.py::compute_local_time_step` 只用到 5 个字段：`area`/`normal`/
    `owner_cell`/`neighbor_cell`/`is_boundary`，全部是逐**面**的量。

    **面积/法向必须与单机路径用同一个几何定义**：单机那边用的是
    `FRFaceConnectivity` 的逐面 `area`/`normal`（三角化半面几何——棱柱
    四边形侧面被拆成 2 条记录、各自半个面积，求和为整张面，而 cfl.py
    正是对所有面记录求和，所以这是自洽的）。不能改用
    `base_flat.true_area_weight`：Part1 精度修复后那个量对四边形侧面的
    两条重复记录都存"整张四边形面"的值（不再是各自的真实半面），直接
    求和会把该面算两次。也不能走 `_extract_p0_face_geometry`：它要求
    `_KernelFaceData` 类型（`base_flat` 是 `FlatFaceGeometry`，会掉进
    只对逐面对象列表有效的慢速路径），且它取的是 fp 0 的值、只对 P0
    成立。（顺带说明：多 GPU 路径原先那份 `_compute_local_time_step_gpu`
    正是这么调的——用 `dist_fc` 当 `fc` 传给 `_extract_p0_face_geometry`，
    而 `dist_fc` 没有 `.normal`/`.area`，一旦被调用必定 AttributeError；
    它从未被 `step()` 调用过，所以是个潜伏 bug，本轮一并修掉。）

    `is_boundary` 的语义要点：只有**物理**边界面才算边界。分区边界面
    在本 rank 看来有一个真实邻居（位于 halo），必须按内部面处理、两侧
    都累加谱半径，否则 owner 侧会丢掉这个面的贡献、dt 被高估。
    """

    def __init__(self, dist_fc, face_area: np.ndarray, face_normal: np.ndarray):
        n_faces = dist_fc.n_faces
        if face_area.shape[0] != n_faces or face_normal.shape[0] != n_faces:
            raise ValueError(
                f"面积/法向数组长度 {face_area.shape[0]}/{face_normal.shape[0]} "
                f"与本 rank 的局部面数 {n_faces} 不符——必须按 "
                f"partition.local_faces 切片后传入")
        self.n_faces = n_faces
        self.normal = np.ascontiguousarray(face_normal)
        self.area = np.ascontiguousarray(face_area)
        self.owner_cell = np.ascontiguousarray(dist_fc.owner_cell_local)
        self.neighbor_cell = np.ascontiguousarray(dist_fc.neighbor_cell_local)
        # 物理边界面 = 没有真实邻居；分区边界面有（在 halo 里），算内部面。
        # 另外把 neighbor 索引无效（-1）的面也一并算作边界面：紧凑重映射
        # 里拿不到对应 halo 槽位的面会是 -1，当成内部面会用 -1 去索引数组
        # （读到最后一个单元，静默错误）。
        self.is_boundary = np.ascontiguousarray(
            dist_fc.physical_boundary_mask | (self.neighbor_cell < 0))


def extract_local_face_area_normal(dist_fc, local_mesh):
    """取本 rank 局部面的逐面 area/normal，两种加载模式都支持。

    - **传统模式**（`local_mesh` 是完整全局网格）：按
      `partition.local_faces` 从全局 `FRFaceConnectivity` 切片。
    - **完全分布式加载**：`local_mesh.face_connectivity` 是 None，改读
      root 预先切好并随 rank 包下发的 `face_area`/`face_normal`
      （见 `distributed_mesh_loader.PrecompactedMeshData`）。两条路径
      给出的是**同一个几何量**，因此 dt 与单机逐位一致。
    """
    fc_global = getattr(local_mesh, "face_connectivity", None)
    if fc_global is not None and getattr(fc_global, "area", None) is not None:
        lf = dist_fc.partition.local_faces
        return fc_global.area[lf], fc_global.normal[lf]

    area = getattr(local_mesh, "face_area", None)
    normal = getattr(local_mesh, "face_normal", None)
    if area is None or normal is None:
        raise ValueError(
            "完全分布式加载模式下需要 root 下发的 face_area/face_normal "
            "（PrecompactedMeshData 的同名字段）才能算逐单元局部 CFL 步长；"
            "静默回退到全局固定步长是被禁止的简化")
    return area, normal


class DistributedCFLView:
    """`cfl.py::compute_local_time_step` 所需 solver 接口的分布式实现。

    刻意只实现那 11 个被真正读取的属性（`solver.state.U.shape`/
    `solver.state.Q`/`solver.mesh.{face_connectivity,get_all_cell_volumes,
    jacobians,n_sps_per_cell}`/`solver.{mu_molecular,freestream}`/
    `solver.{_get_metric_flux_scale,_get_turbulent_viscosity_field}`，
    以及 getattr 读取的 `_cfl_controller`/`current_order`/
    `low_mach_precond_enabled`），不做"通用 solver 替身"——属性面越小，
    上游改动时越容易被测试发现而不是静默走错分支。
    """

    class _MeshView:
        def __init__(self, fc_view, cell_volumes, jacobians, n_sps_per_cell, n_cells):
            self.face_connectivity = fc_view
            self.jacobians = jacobians
            self.n_sps_per_cell = n_sps_per_cell
            self.n_cells = n_cells
            self._cell_volumes = cell_volumes

        def get_all_cell_volumes(self):
            return self._cell_volumes

    class _StateView:
        def __init__(self, U, Q):
            self.U = U
            self.Q = Q

    def __init__(self, U_compact, Q_compact, dist_fc, local_mesh, *,
                 mu_molecular, freestream, cfl_controller, current_order,
                 low_mach_precond_enabled, mu_t_local=None, n_local_cells=None,
                 jacobians=None, cell_volumes=None, fixed_cfl_number=None):
        n_compact, n_sps = U_compact.shape[0], U_compact.shape[1]
        face_area, face_normal = extract_local_face_area_normal(dist_fc, local_mesh)
        fc_view = _DistributedFaceConnectivityView(dist_fc, face_area, face_normal)

        if jacobians is None or cell_volumes is None:
            raise ValueError(
                "jacobians/cell_volumes 必须由调用方按 dist_fc.compact_global_ids "
                "抽取好后传入——本视图刻意不自己去猜 local_mesh 是全局网格还是"
                "已经紧凑化的包（两种模式的索引语义不同，见 "
                "DistributedMeshAdapter 文档）"
            )

        self.state = self._StateView(U_compact, Q_compact)
        self.mesh = self._MeshView(fc_view, cell_volumes, jacobians, n_sps, n_compact)
        self.mu_molecular = mu_molecular
        self.freestream = freestream
        self._cfl_controller = cfl_controller
        self.fixed_cfl_number = fixed_cfl_number
        self.current_order = current_order
        self.low_mach_precond_enabled = low_mach_precond_enabled
        self._n_local_cells = (n_compact if n_local_cells is None else n_local_cells)
        self._mu_t_local = mu_t_local
        perm = getattr(dist_fc, "perm", None)
        inv_perm = getattr(dist_fc, "inv_perm", None)
        if perm is None or inv_perm is None:
            raise ValueError(
                "dist_fc 缺少 perm/inv_perm——本视图需要它们在原生排列与"
                "紧凑排列之间转换（见模块文档）")
        self._perm = np.asarray(perm)
        self._inv_perm = np.asarray(inv_perm)
        self._metric_flux_scale_cache = None

    def _get_metric_flux_scale(self) -> np.ndarray:
        """逐 SP 的度量"通量面积"标度，与单机 `_get_metric_flux_scale`
        同一公式（sum_m ||adj(J)[:,m,:]||，adj(J) = det(J)*inv(J)）。

        缓存判据同样比较**完整形状**（不只 shape[0]）——单机那一份曾因
        只比 shape[0] 而在阶数切换后静默返回陈旧形状，见
        `solver_geometry.py::_get_metric_flux_scale` 的 bug 记录。
        """
        cached = self._metric_flux_scale_cache
        if cached is not None and cached.shape == self.state.U.shape[:2]:
            return cached
        n_cells = self.mesh.n_cells
        n_sps = self.mesh.n_sps_per_cell
        det_jacs = self.mesh.jacobians["det_jacs"].reshape(n_cells, n_sps)
        inv_jacs = self.mesh.jacobians["inv_jacs"].reshape(n_cells, n_sps, 3, 3)
        adj_j = det_jacs[..., None, None] * inv_jacs
        scale = np.sum(np.linalg.norm(adj_j, axis=-1), axis=-1)
        self._metric_flux_scale_cache = scale
        return scale

    def _get_turbulent_viscosity_field(self) -> Optional[np.ndarray]:
        """涡粘场，按紧凑索引空间给出；halo 段用 0 填充。

        为什么 halo 段填 0 是正确的（而不是"少做了一次 halo 交换"）：
        涡粘只通过 `dt_visc = 0.25*CFL*rho*V^(2/3)/mu_eff` 进入，而那是
        **纯逐单元**的量、不含任何邻居耦合。halo 单元算出来的 dt 本来就
        不完整（它们的部分面归别的 rank）、一律被丢弃，所以 halo 段填什么
        都不影响任何被使用的结果。见模块文档"为什么在 local+halo 紧凑
        空间上算"。
        """
        if self._mu_t_local is None:
            return None
        n_compact, n_sps = self.state.U.shape[:2]
        src = self._mu_t_local
        if src.shape[1] != n_sps:
            rep = int(np.ceil(n_sps / src.shape[1]))
            src = np.tile(src, (1, rep))[:, :n_sps]
        # 先在**原生**排列（local 在前、halo 在后）里填好，再用 perm 换到
        # 紧凑排列——不能直接写紧凑空间的前 n_local 段，那一段不是 local
        # 单元（见模块文档"切片必须先换回原生排列"）。
        mu_t_native = np.zeros((n_compact, n_sps), dtype=np.float64)
        n_l = min(self._n_local_cells, src.shape[0])
        mu_t_native[:n_l] = src[:n_l]
        return mu_t_native[self._perm]


def compute_distributed_local_time_step(
    U_compact: np.ndarray,
    Q_compact: np.ndarray,
    dist_fc,
    local_mesh,
    *,
    jacobians,
    cell_volumes,
    n_local_cells: int,
    mu_molecular: float,
    freestream: dict,
    cfl_controller=None,
    fixed_cfl_number=None,
    current_order: int = 0,
    low_mach_precond_enabled: bool = False,
    mu_t_local: Optional[np.ndarray] = None,
    return_physical_too: bool = False,
):
    """分布式路径的逐单元局部 CFL 步长，只返回 local cells 那一段。

    Args:
        U_compact: (n_local+n_halo, n_sps, n_vars) 紧凑索引空间的守恒变量
            （halo 交换产出并已按 dist_fc.perm 重排到"棱柱在前"排列）。
        Q_compact: 同形状的原始变量 (rho,u,v,w,p)。
        dist_fc: DistributedFlatFaceGeometry。
        local_mesh: 提供 face_flux_points 的网格对象。
        jacobians: 已按 dist_fc.compact_global_ids 抽取/重排好的
            {'det_jacs','inv_jacs'}（与 DistributedMeshAdapter 同一来源）。
        cell_volumes: 同上，(n_local+n_halo,)。
        n_local_cells: 本 rank 的 local cell 数（切片长度）。
        mu_molecular, freestream, cfl_controller, fixed_cfl_number,
        current_order, low_mach_precond_enabled: 与单机 solver 上的同名量语义
            完全一致（CFL 取值规则见 adaptive_cfl/policy.py）。
        mu_t_local: (n_local, n_sps) 本地涡粘场，可选。
        return_physical_too: True 时返回 (dt_mean_flow, dt_physical)，
            前者可能是按预处理波速放大过的（湍流标量用后者）。

    Returns:
        (n_local_cells, n_sps) 的 dt；或该形状的二元组。
    """
    from autoflowcfd.core.fr_solver.cfl import compute_local_time_step

    view = DistributedCFLView(
        U_compact, Q_compact, dist_fc, local_mesh,
        mu_molecular=mu_molecular, freestream=freestream,
        cfl_controller=cfl_controller, fixed_cfl_number=fixed_cfl_number,
        current_order=current_order,
        low_mach_precond_enabled=low_mach_precond_enabled,
        mu_t_local=mu_t_local, n_local_cells=n_local_cells,
        jacobians=jacobians, cell_volumes=cell_volumes,
    )
    out = compute_local_time_step(view, return_physical_too=return_physical_too)
    inv_perm = view._inv_perm

    def _to_local(arr):
        # 紧凑排列 -> 原生排列（local 在前）-> 切出 local 段
        return arr[inv_perm][:n_local_cells]

    if return_physical_too:
        dt_mean, dt_phys = out
        return _to_local(dt_mean), _to_local(dt_phys)
    return _to_local(out)
