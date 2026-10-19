"""
AutoFlowCFD V2.0 - P0 阶专用有限体积无粘残差 (S-02)

从 fr_residual_inviscid.py 拆出来（控制单文件行数，>400 行需拆分的
项目规范）。`compute_inviscid_residual_fr` 在 `mesh.n_points_1d==1`
时委托到这里，见该函数文档。

性能优化：将原纯 Python 逐面循环替换为 numba 并行 kernel
(inviscid_p0_kernel.py)，791K 单元 / 188 万面网格上从 ~25s 降至 ~1-2s。
"""

from typing import Callable

import numpy as np


def compute_inviscid_residual_fv_p0(
    U: np.ndarray,
    mesh,
    flat,
    boundary_ghost_provider: Callable[[int, np.ndarray, np.ndarray], np.ndarray],
    mach_ref: float = 0.1,
) -> np.ndarray:
    """P0（1 SP/cell，Order Continuation 最低阶）专用有限体积残差。

    算法与原 Python 版本完全一致（逐位等价，仅浮点重排顺序不同）：
    1. 提取 flat 面几何数组（单位法向、面积权重、面连接关系）
    2. 预计算边界幽灵态（Python 端，仅 ~40K 边界面）
    3. numba 并行 kernel 执行 AUSM+up 黎曼求解，按面着色逐颜色组累加（颜色分组取自 `flat`，与 P>=1 界面核
       同一份）

    关于棱柱四边形侧面拆分的处理（2026-08-23 修复，见下方
    `p0_face_geometry` 文档）：此前这里的说法——"按
    face_connectivity 的原始每条记录处理（不过滤 owner_is_primary），
    保证闭合单元面积/法向积分 Σ(n̂·A)=0"——只在旧的 true_normal/
    true_area_weight 实现（每条三角化子面记录各自是真实的局部半面
    几何）下成立。当天引入的"逐 Flux Point 精确法向"修复
    （face_flux_points/exact_normal.py）把 true_normal/true_area_weight
    改成了 (owner_cell,owner_axis,owner_side) 的纯函数，棱柱四边形
    侧面的 2 条拆分记录因此变成完全相同的"整张四边形面"的值——不再
    过滤 owner_is_primary 就会让几乎所有棱柱四边形侧面的通量被
    scatter-add 两次（真实复现：cube_demo 791k 网格 P0 阶 resume，
    单元 17790 处 Σ(n̂·A) 相对误差达 6.76%，几乎全部动量残差来自
    这个未闭合的法向量残差乘以绝对压力，与马赫数/CFL 无关）。现在
    `p0_face_geometry` 负责按 owner_is_primary/
    neighbor_is_primary 去重（重复记录只保留一条），对真正 multi-source
    的拆分面（~5%，两条记录指向 2 个不同真实相邻四面体）改用旧的
    三角化真实半面 face_conn.normal/area（P0 每面只有 1 个 Flux
    Point，无法像 P1+ kernel 那样在 FP 级别按对角线混合两个真实
    邻居，只能退回真实几何半面）。

    Args:
        U: 守恒变量，形状 (n_cells, 1, n_vars)
        mesh: HighOrderMesh 实例（n_points_1d 必须为 1）
        flat: 同一网格的面几何（`get_flat_face_geometry`；CPU 分布式为预先按 local+halo 压缩索引构造的
            `flat_face_override`），这里只用它的面着色 `color_face_indices`，面索引与 `face_connectivity` 一致
        boundary_ghost_provider: 同 compute_inviscid_residual_fr

    Returns:
        residual: 形状 (n_cells, 1, 5)
    """
    from autoflowcfd.boundary.fr_ghost_state import require_ghost_provider
    from .inviscid import conserved_to_primitive
    from .inviscid_p0_kernel import _p0_inviscid_kernel


    n_cells = mesh.n_cells
    if mesh.cell_volumes is None:
        raise RuntimeError(
            "mesh.cell_volumes not available - required for the P0 finite-volume residual path "
            "(should have been computed once in load_from_volume_mesh at the mesh's target order)."
        )
    cell_volumes = mesh.cell_volumes

    Q_all = conserved_to_primitive(U[..., :5])[:, 0, :]  # (n_cells,5)

    n_faces = flat.n_faces

    # --- 面几何：连接关系与着色取 `flat`，P0 法向 / 面积取 `p0_face_arrays`（两者是同一个面表）---
    unit_normals, area_weights = p0_face_arrays(mesh)
    if unit_normals.shape[0] != n_faces:
        raise ValueError(f"P0 面几何有 {unit_normals.shape[0]} 个面，面连接关系有 {n_faces} 个（不是同一个面表）")

    # --- 预计算边界幽灵态 ---
    ghost_provider = require_ghost_provider(boundary_ghost_provider, "compute_inviscid_residual_fv_p0")
    Q_ghost = _precompute_ghost_states(flat, flat, ghost_provider, Q_all, n_faces, unit_normals)

    # --- numba 并行 kernel：按面着色逐颜色组累加 ---
    owner_cell = np.asarray(flat.owner_cell, dtype=np.int64)
    neighbor_cell = np.asarray(flat.neighbor_cell, dtype=np.int64)
    is_boundary = np.asarray(flat.is_boundary, dtype=np.bool_)
    residual5 = np.zeros((n_cells, 5))
    for face_indices in flat.color_face_indices:
        if len(face_indices) == 0:
            continue
        _p0_inviscid_kernel(
            owner_cell, neighbor_cell, is_boundary,
            unit_normals, area_weights,
            Q_all, Q_ghost, cell_volumes,
            face_indices, residual5, mach_ref,
        )
    return residual5[:, None, :]


def p0_face_arrays(mesh):
    """`mesh` 的面表上的 P0 `(unit_normals, area_weights)`。

    单机：`p0_face_geometry(mesh.face_flux_points)`（全局面表）。CPU 分布式（`mesh` 为 `DistributedMeshAdapter`）：
    分布式面几何构造时从全局 P0 面几何按本 rank 面表切出的那一份（`distributed_flat_face/build.py`，完全分布式
    加载下由 root 算好随包下发）。2026-10-19（I-108）以前分布式 P0 无粘残差不在分区上算：每个 rank（完全分布式
    下只有 root）把全局状态汇集起来算一遍全局残差再取自己那段，零并行加速、root 内存 O(全局)。
    """
    from autoflowcfd.core.mpi.distributed_flat_face import DistributedFlatFaceGeometry

    fc = mesh.face_connectivity
    if isinstance(fc, DistributedFlatFaceGeometry):
        if fc.p0_unit_normals is None:
            raise RuntimeError("分布式面几何没有 P0 面几何（构造时网格不是 P0 阶）")
        return fc.p0_unit_normals, fc.p0_area_weights
    return p0_face_geometry(mesh.face_flux_points)


def p0_face_geometry(ffp):
    """P0 有限体积通量用的逐面单位法向与面积权重 `(unit_normals (n_faces, 3), area_weights (n_faces,))`。

    只依赖网格，第一次调用时算好缓存在面通量点数据上（`_KernelFaceData._p0_geometry`），CPU 残差、GPU 残差的设备
    缓存与测试共用这一份；返回只读数组。2026-10-19（I-8）以前 CPU 每次残差求值都重新提取（含逐组 Python 去重循环）。
    非 `_KernelFaceData` 的逐面提取分支同时删除（生产不可达，且不做下述去重，会给出错误的面积）。

    棱柱四边形侧面去重（2026-08-23 修复，见 compute_inviscid_residual_
    fv_p0 文档"关于棱柱四边形侧面拆分的处理"一节的完整原理）：
    每个棱柱四边形侧面被网格生成器恒定拆成 2 条三角化子面记录
    （face_connectivity.py 文档），Part1 精度修复后这 2 条记录的
    true_normal/true_area_weight 变成完全相同的"整张四边形面"精确值
    （不再是各自的真实半面）。P0 kernel 对每条面记录都独立贡献一次
    scatter-add（不像 P1+ kernel 那样按 owner_is_primary/
    neighbor_is_primary 过滤），如果不做处理就会把这类面的通量重复
    计算两次。按两种情形处理：
    - 组内记录指向同一个真实相邻单元（约 95%，含全部边界面重复记录）：
      只保留 owner_is_primary/neighbor_is_primary 为真的那一条（此时
      就是整张面相对唯一真实邻居的精确值），其余记录面积清零——面积
      为零即可让该记录对 kernel 的贡献恒为零，不需要改 kernel 本身。
    - 组内记录指向 2 个不同的真实相邻单元（约 5%，genuine multi-source，
      按文档只会发生在棱柱四边形一侧对多个不同真实四面体的非协调
      情形）：两条记录都保留参与 kernel，但改用 face_connectivity 的
      原始三角化半面几何 `fc.normal`/`fc.area`（P0 每面只有 1 个 Flux
      Point，无法像 P1+ kernel 那样在 FP 级别按对角线把两个真实邻居
      混合进同一条记录，只能退回到两条记录各自真实、精确求和为整张
      面的三角化半面）。

    Args:
        ffp: 网格的面通量点数据（`mesh.face_flux_points`，`_KernelFaceData`）

    Returns:
        unit_normals: (n_faces, 3) float64，只读
        area_weights: (n_faces,) float64，只读
    """
    from autoflowcfd.fr.face_flux_points.data import _KernelFaceData

    if not isinstance(ffp, _KernelFaceData):
        raise TypeError(f"P0 面几何需要 _KernelFaceData，收到 {type(ffp).__name__}")
    if ffp._p0_geometry is None:
        ffp._p0_geometry = _build_p0_face_geometry(ffp)
    return ffp._p0_geometry


def _build_p0_face_geometry(ffp):
    unit_normals = np.array(ffp.true_normal[:, 0, :], dtype=np.float64, copy=True)
    area_weights = np.array(ffp.true_area_weight[:, 0], dtype=np.float64, copy=True)

    owner_is_primary = ffp.owner_is_primary
    neighbor_is_primary = ffp.neighbor_is_primary
    fc = ffp._face_conn
    old_normal = fc.normal
    old_area = fc.area
    owner_cell = fc.owner_cell
    neighbor_cell = fc.neighbor_cell

    def _dedupe_or_fallback(groups, is_primary, other_cell_arr):
        for flist in groups.values():
            if len(flist) < 2:
                continue
            members = sorted(flist)
            other_cells = {int(other_cell_arr[f]) for f in members}
            if len(other_cells) > 1:
                # genuine multi-source：每条记录各自真实半面几何，
                # 天然精确求和为整张面，各自对各自真实邻居生效。
                for f in members:
                    unit_normals[f] = old_normal[f]
                    area_weights[f] = old_area[f]
            else:
                # 重复记录、同一个真实邻居：只留 primary 那一条。
                for f in members:
                    if not is_primary[f]:
                        area_weights[f] = 0.0

    _dedupe_or_fallback(ffp._owner_groups, owner_is_primary, neighbor_cell)
    _dedupe_or_fallback(ffp._neighbor_groups, neighbor_is_primary, owner_cell)

    # 边界面的重复记录（同一个棱柱四边形面在边界上被拆成 2 条记录）不会
    # 出现在 _owner_groups/_neighbor_groups 里（build_face_flux_points
    # 单独用 boundary_owner_groups 分组，未存到 _KernelFaceData 上），
    # 但 owner_is_primary 已经对它们正确标记过——幽灵态对整张面统一
    # 取值（不依赖拆分子面各自覆盖对角线哪一半），不存在多真实邻居的
    # 情形，直接按 primary 去重即可。
    is_bnd = fc.is_boundary
    area_weights[is_bnd & ~owner_is_primary] = 0.0

    # 混合拆分面（棱柱四边形侧面一半挨内部单元、一半在边界上；`fr/face_flux_points/grouping.py`）：P>=1 由
    # 内部子面记录担任整张面 primary、逐通量点按半区取邻居或幽灵态。P0 每面一个通量点，两半直接当作两个
    # 独立的面——内部半区是普通内部面，边界半区是其所属棱柱（边界子面记录的 owner）的普通边界面——各用自己
    # 的真实三角形几何（与上面 genuine multi-source 的回退同一思路）。2026-10-17 以前改为在内部记录上按面积
    # 占比混合两份通量：边界半区的通量用了内部记录上从未计算过的全零"幽灵态"（均匀来流残差 5.6e7），邻居
    # 一侧还多加了 (1-b) 倍的边界半区通量（不守恒，I-7）
    for partner in (ffp.mixed_nb_partner, ffp.mixed_ow_partner):
        f_int = np.flatnonzero(np.asarray(partner) >= 0)
        for f in (f_int, np.asarray(partner)[f_int]):
            unit_normals[f] = old_normal[f]
            area_weights[f] = old_area[f]

    unit_normals.setflags(write=False)
    area_weights.setflags(write=False)
    return unit_normals, area_weights


def _precompute_ghost_states(face_data, fc, ghost_provider, Q_all, n_faces: int, unit_normals):
    """需要幽灵态的面（真边界面 + 混合拆分面的边界子面记录）的 P0 幽灵态，按**面**索引。

    按面而不是按 owner 单元索引：同一个 P0 单元可同时挨着多个不同类型/法向的边界面（角点单元）。

    Args:
        face_data: 提供 `mixed_bnd_face`（混合拆分面的边界子面记录）的面数据（`mesh.face_flux_points` 或面几何）
        fc: 提供 `is_boundary` / `owner_cell` 的面连接关系（与 `face_data` 同一个面表）
        Q_all: (n_cells, 5) 单元原始变量
        unit_normals: (n_faces, 3)，`p0_face_geometry` 给出的面法向

    Returns:
        Q_ghost: (n_faces, 5)，只有需要幽灵态的面的行有意义，其余为零（kernel 不读）。
    """
    Q_ghost = np.zeros((n_faces, 5), dtype=Q_all.dtype)
    need = np.asarray(fc.is_boundary, dtype=bool) | np.asarray(face_data.mixed_bnd_face, dtype=bool)
    faces = np.flatnonzero(need)
    if faces.size:
        Q_owner = Q_all[fc.owner_cell[faces]][:, np.newaxis, :]
        Q_ghost[faces] = ghost_provider.ghost_states(faces, Q_owner, unit_normals[faces][:, np.newaxis, :])[:, 0, :]
    return Q_ghost
