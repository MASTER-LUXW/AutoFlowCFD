"""
AutoFlowCFD V2.0 - P0 阶专用有限体积无粘残差 (S-02)

从 fr_residual_inviscid.py 拆出来（控制单文件行数，>400 行需拆分的
项目规范）。`compute_inviscid_residual_fr` 在 `mesh.n_points_1d==1`
时委托到这里，见该函数文档。

性能优化：将原纯 Python 逐面循环替换为 numba 并行 kernel
(inviscid_p0_kernel.py)，791K 单元 / 188 万面网格上从 ~25s 降至 ~1-2s。
"""

from typing import Callable, Optional

import numpy as np
import numba


def compute_inviscid_residual_fv_p0(
    U: np.ndarray,
    mesh,
    boundary_ghost_provider: Optional[Callable[[int, np.ndarray, np.ndarray], np.ndarray]] = None,
    mach_ref: float = 0.1,
    precond_mode: Optional[int] = None,
) -> np.ndarray:
    """P0（1 SP/cell，Order Continuation 最低阶）专用有限体积残差。

    算法与原 Python 版本完全一致（逐位等价，仅浮点重排顺序不同）：
    1. 提取 flat 面几何数组（单位法向、面积权重、面连接关系）
    2. 预计算边界幽灵态（Python 端，仅 ~40K 边界面）
    3. numba 并行 kernel 执行 AUSM+up 黎曼求解 + scatter-add
    4. per-thread buffer 归约得到最终残差

    关于棱柱四边形侧面拆分的处理（2026-08-23 修复，见下方
    `_extract_p0_face_geometry` 文档）：此前这里的说法——"按
    face_connectivity 的原始每条记录处理（不过滤 owner_is_primary），
    保证闭合单元面积/法向积分 Σ(n̂·A)=0"——只在旧的 true_normal/
    true_area_weight 实现（每条三角化子面记录各自是真实的局部半面
    几何）下成立。当天引入的"逐 Flux Point 精确法向"修复
    （face_flux_points_exact_normal.py）把 true_normal/true_area_weight
    改成了 (owner_cell,owner_axis,owner_side) 的纯函数，棱柱四边形
    侧面的 2 条拆分记录因此变成完全相同的"整张四边形面"的值——不再
    过滤 owner_is_primary 就会让几乎所有棱柱四边形侧面的通量被
    scatter-add 两次（真实复现：cube_demo 791k 网格 P0 阶 resume，
    单元 17790 处 Σ(n̂·A) 相对误差达 6.76%，几乎全部动量残差来自
    这个未闭合的法向量残差乘以绝对压力，与马赫数/CFL 无关）。现在
    `_extract_p0_face_geometry` 负责按 owner_is_primary/
    neighbor_is_primary 去重（重复记录只保留一条），对真正 multi-source
    的拆分面（~5%，两条记录指向 2 个不同真实相邻四面体）改用旧的
    三角化真实半面 face_conn.normal/area（P0 每面只有 1 个 Flux
    Point，无法像 P1+ kernel 那样在 FP 级别按对角线混合两个真实
    邻居，只能退回真实几何半面）。

    Args:
        U: 守恒变量，形状 (n_cells, 1, n_vars)
        mesh: HighOrderMesh 实例（n_points_1d 必须为 1）
        boundary_ghost_provider: 同 compute_inviscid_residual_fr
        precond_mode: AUSM+up 预处理声速作用域，None 表示按环境变量
            `AFCFD_AUSM_PRECOND_MODE` 解析（见 kernels.py::
            resolve_ausm_precond_mode）。在**纯 Python 层**解析、以
            实参形式传进 njit kernel 是硬性要求，不能改成 njit 里读
            全局量。

    Returns:
        residual: 形状 (n_cells, 1, 5)
    """
    from .inviscid import conserved_to_primitive, DefaultGhostProvider
    from .inviscid_p0_kernel import _p0_inviscid_kernel
    from autoflowcfd.core.fr_operators.kernels import resolve_ausm_precond_mode

    if precond_mode is None:
        precond_mode = resolve_ausm_precond_mode()

    n_cells = mesh.n_cells
    if mesh.cell_volumes is None:
        raise RuntimeError(
            "mesh.cell_volumes not available - required for the P0 finite-volume residual path "
            "(should have been computed once in load_from_volume_mesh at the mesh's target order)."
        )
    cell_volumes = mesh.cell_volumes

    Q_all = conserved_to_primitive(U[..., :5])[:, 0, :]  # (n_cells,5)

    fc = mesh.face_connectivity
    ffp_list = mesh.face_flux_points
    n_faces = fc.n_faces

    # --- 提取 flat 面几何数组 ---
    unit_normals, area_weights = _extract_p0_face_geometry(ffp_list, fc, n_faces)

    # --- 预计算边界幽灵态 ---
    ghost_provider = boundary_ghost_provider if boundary_ghost_provider is not None else DefaultGhostProvider()
    Q_ghost = _precompute_ghost_states(ffp_list, fc, ghost_provider, Q_all, n_faces, unit_normals)

    # --- numba 并行 kernel ---
    n_threads = numba.get_num_threads()
    owner_cell = fc.owner_cell.astype(np.int64)
    neighbor_cell = fc.neighbor_cell.astype(np.int64)
    is_boundary = fc.is_boundary.astype(np.bool_)
    # 混合拆分面（B-8）面积占比权重，非混合面恒 0；慢速路径无混合面概念，补零占位。
    mixed_p0_bnd_frac = getattr(ffp_list, "mixed_p0_bnd_frac", None)
    if mixed_p0_bnd_frac is None:
        mixed_p0_bnd_frac = np.zeros(n_faces, dtype=np.float64)

    residual_per_thread = _p0_inviscid_kernel(
        owner_cell, neighbor_cell, is_boundary,
        unit_normals, area_weights,
        Q_all, Q_ghost, cell_volumes,
        mixed_p0_bnd_frac,
        n_cells, n_threads, mach_ref, int(precond_mode),
    )

    # --- per-thread buffer 归约 ---
    residual5 = residual_per_thread.sum(axis=0)

    return residual5[:, None, :]


def _extract_p0_face_geometry(ffp_list, fc, n_faces: int):
    """从 face_flux_points 提取 P0 需要的 flat 数组。

    支持两种数据源：
    - _KernelFaceData（快速路径）：直接读取 flat 数组，并去重/回退
      三角化半面几何（见下方"棱柱四边形侧面去重"一节）
    - list of FaceFluxPointGeometry（慢速路径）：逐面提取（不做去重——
      仅在非 _KernelFaceData 场景使用，当前生产代码路径不会触发这个
      分支，见 face_kernels.py::build_flat_face_geometry 对慢速路径的
      说明）

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

    Returns:
        unit_normals: (n_faces, 3) float64
        area_weights: (n_faces,) float64
    """
    from autoflowcfd.fr.face_flux_points_data import _KernelFaceData

    if not isinstance(ffp_list, _KernelFaceData):
        # 慢速路径：逐面提取（仅在非 _KernelFaceData 时，不做棱柱四边形
        # 去重——当前生产代码路径不会触发，见上方文档）
        unit_normals = np.empty((n_faces, 3), dtype=np.float64)
        area_weights = np.empty(n_faces, dtype=np.float64)
        for f in range(n_faces):
            ffp = ffp_list[f]
            unit_normals[f] = ffp.true_normal[0]
            area_weights[f] = ffp.true_area_weight[0]
        return unit_normals, area_weights

    # 快速路径：直接使用 flat 数组（P0: n_fp=1），再按上方文档去重
    unit_normals = np.array(ffp_list.true_normal[:, 0, :], dtype=np.float64, copy=True)
    area_weights = np.array(ffp_list.true_area_weight[:, 0], dtype=np.float64, copy=True)

    owner_is_primary = ffp_list.owner_is_primary
    neighbor_is_primary = ffp_list.neighbor_is_primary
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

    _dedupe_or_fallback(ffp_list._owner_groups, owner_is_primary, neighbor_cell)
    _dedupe_or_fallback(ffp_list._neighbor_groups, neighbor_is_primary, owner_cell)

    # 边界面的重复记录（同一个棱柱四边形面在边界上被拆成 2 条记录）不会
    # 出现在 _owner_groups/_neighbor_groups 里（build_face_flux_points
    # 单独用 boundary_owner_groups 分组，未存到 _KernelFaceData 上），
    # 但 owner_is_primary 已经对它们正确标记过——幽灵态对整张面统一
    # 取值（不依赖拆分子面各自覆盖对角线哪一半），不存在多真实邻居的
    # 情形，直接按 primary 去重即可。
    is_bnd = fc.is_boundary
    area_weights[is_bnd & ~owner_is_primary] = 0.0

    return unit_normals, area_weights


def _precompute_ghost_states(ffp_list, fc, ghost_provider, Q_all, n_faces: int, unit_normals):
    """预计算需要幽灵态的面（真边界面 + 混合拆分面的内部记录，B-8）的
    幽灵态，按**面**索引（不是按 owner 单元索引）。

    真实 bug 修复（V2.0 专家组盲审第四次评审，2026-08-28）：此前这里
    按 owner 单元索引成 (n_cells,5) 数组、numba kernel 也按
    `Q_ghost[owner_cell]` 读取——如果同一个 P0 单元同时挨着 2 个以上
    边界面（角点单元很常见，真实复现：本文件下方交叉验证用的
    `build_channel_mesh_prism` 测试网格上，单元 3 就同时挨着 3 个边界
    面），后处理的边界面会把先处理的边界面的幽灵态覆盖掉，导致有些
    边界面在通量计算时用了另一个边界面（可能是不同 BC 类型/不同法向）
    算出的幽灵态。用 `DefaultGhostProvider`（ghost=owner 状态，覆盖谁
    都一样）时这个问题被完全掩盖，只有真实的 WALL/INLET/OUTLET 等
    幽灵态（不同面算出不同值）才会暴露。改为按面索引的 (n_faces,5)
    数组，与 GPU 版
    `core/gpu/residual/gpu_p0_inviscid.py::compute_inviscid_residual_p0_gpu`
    的既有正确做法保持一致；numba kernel（`inviscid_p0_kernel.py::
    _p0_inviscid_kernel`）同步改为按面索引读取。

    Args:
        unit_normals: (n_faces,3)，`_extract_p0_face_geometry` 已去重
            过的面法向——真边界面的法向不受多源去重逻辑影响（那只处理
            内部面的歧义），与原来读 `ffp.true_normal` 数值相同，直接
            复用避免重复访问 `ffp_list[f]`。

    Returns:
        Q_ghost: (n_faces, 5)，只有 is_boundary 或混合拆分面对应的行
            有意义，其余行为零占位（kernel 不会读取）。
    """
    from .inviscid import DefaultGhostProvider

    Q_ghost = np.zeros((n_faces, 5), dtype=Q_all.dtype)
    is_bnd = fc.is_boundary
    mixed_bnd_face = getattr(ffp_list, "mixed_bnd_face", None)
    if mixed_bnd_face is None:
        mixed_bnd_face = np.zeros(n_faces, dtype=np.bool_)
    need_ghost = is_bnd | mixed_bnd_face

    if isinstance(ghost_provider, DefaultGhostProvider):
        # 快速路径：ghost = owner 状态，向量化赋值，无需逐面调用。
        owner_of_needed = fc.owner_cell[need_ghost]
        Q_ghost[need_ghost] = Q_all[owner_of_needed]
        return Q_ghost

    # 通用路径：逐面调用 ghost_provider（真实边界条件依赖各面自己的
    # 法向/位置，不能向量化）。
    for f in np.nonzero(need_ghost)[0]:
        oc = int(fc.owner_cell[f])
        Q_owner_fp = Q_all[oc:oc + 1]  # (1,5)
        true_normal = unit_normals[f:f + 1]  # (1,3)
        Q_ghost[f, :] = ghost_provider(f, Q_owner_fp, true_normal)[0]

    return Q_ghost
