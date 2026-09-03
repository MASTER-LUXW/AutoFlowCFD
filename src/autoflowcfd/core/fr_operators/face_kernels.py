"""FR 面几何展平缓存 (性能优化：逐面 Python 循环 -> numba 逐点标量 kernel)。

背景：`core/fr_residual_inviscid.py`/`core/fr_viscous_flux.py` 里各有一个
纯 Python 的 `for f in range(fc.n_faces):` 逐面循环，在生产规模网格
（545,597 单元、1,326,110 面）上实测单次残差求值耗时约 1546 秒——瓶颈是
Python 解释器 + 每次循环体内十几个小 numpy 调用的调度开销，不是真正的
浮点运算量。修复思路：不改算法/控制流本身（这两处循环是本项目本轮
评审反复修复过的最核心正确性代码），只把执行方式换成 numba 编译的原生
代码。numba 的 nopython 模式不能直接消费 `mesh.face_flux_points`（一个
由 `FaceFluxPointGeometry` dataclass 组成的 Python list，见
`fr/face_flux_points.py`，其中 `owner_sources`/`neighbor_sources` 是
变长的 `(cell_id, matrix)` 元组列表），本模块负责把它一次性展平成 numba
可以直接读的定长 numpy 数组，缓存后供无粘/粘性两个 kernel 共用（面几何
本身在无粘/粘性残差之间是共享的，且在同一批 RK 子迭代内不变）。

缓存失效：Order Continuation（`grid/order_continuation.py`）会在阶数切换
时于同一个 mesh 对象上整体替换 `mesh.face_flux_points`（重新构建或从
缓存恢复，见 `grid/high_order_mesh_order.py`），所以缓存键必须是
`mesh.face_flux_points` 这个 list 对象本身的身份，不能是"mesh 上有没有
这个属性"——否则阶数切换后会静默复用上一个阶数的展平数组，不报错，只是
算出错的残差。

缓存键危险陷阱（真实复现过一次，不是假设）：最初实现用 `id(mesh.
face_flux_points)`（一个裸 int）当字典键。CPython 的 `id()` 只在对象存活
期间保证唯一——旧 mesh 对象被垃圾回收后，它的 `face_flux_points` list
腾出的内存完全可能被之后新建的、**另一个不相关 mesh** 的 `face_flux_
points` list 复用，导致两个不同 mesh 拿到相同的 `id()`。本模块的缓存
只保留最近一条记录，一旦发生这种 id 撞车，新 mesh 会命中缓存、静默拿到
另一个 mesh（可能阶数、单元数、prism/tet 构成完全不同）的展平几何——
真实复现：连续跑全量测试套件时（大量小 mesh 对象被创建/销毁，是 id 复用
最容易发生的场景）第一次触发了 `ValueError: incompatible array sizes for
np.dot`（形状对不上直接崩溃，运气好被抓住了）；同样的两次全量测试套件
连续跑，另一次完全没有触发——是概率性的内存分配时机问题，不是随机噪声，
更不能指望"多数时候不出现"就当它不存在：形状恰好碰巧兼容时，这个 bug
不会崩溃，而是**静默算出错误物理量**，比崩溃更危险。修复：缓存键换成
持有该 list 对象的**强引用**本身（不是它的 id()），查找时用 `is` 做身份
比较——只要这个引用被缓存持有着，它就不可能被垃圾回收，也就不可能有
任何其他对象复用到同一个身份，从根上消除这类 id 复用竞争。

内存设计：`owner_sources`/`neighbor_sources` 长度恒为 1 或 2（网格生成器
把棱柱四边形侧面恒定拆分成 2 个三角子面，不会更多，见
`fr/face_flux_points_merge.py::_resolve_multi_source` 文档），但绝大多数
面（普通四面体-四面体内部面、未拆分的棱柱面）只有 1 个来源。如果统一按
2 槽稠密填充，多出来的一半矩阵纯粹是浪费——在 P2、n_fp=9、n_sps=27 下，
1.3M 面 × 2 角色(owner/neighbor) × 2 槽 × (9×27×8字节) 约 10GB，这在真实
生产网格上不是可以忽略的开销。因此设计成"稠密槽 0 + 稀疏槽 1"：槽 0
对每个面恒定存在（大小 n_faces），槽 1 只对真正有第 2 个来源的那一小
部分面额外分配一个更小的紧凑数组，用 -1 表示的下标数组做重定向。
"""

from dataclasses import dataclass
from typing import Dict

import numpy as np

# 单槽缓存：(强引用住的 key list 对象, 对应的 FlatFaceGeometry)。用对象
# 本身的强引用而不是 id() 当键，理由见模块文档"缓存键危险陷阱"一节——
# 持有强引用能保证该对象在缓存存活期间不被 GC，其"身份"也就不可能被
# 另一个无关对象复用。
_FLAT_CACHE_KEY: object = None
_FLAT_CACHE_VALUE: "FlatFaceGeometry" = None

# n1d -> (dist_fp_of_sp (3,n_sps), dist_axis_coord_of_sp (3,n_sps))，与网格
# 无关，只依赖阶数，缓存不需要按 mesh 失效。
_DIST_MAP_CACHE: Dict[int, tuple] = {}


def _derive_distribute_mapping(n1d: int) -> tuple:
    """把 `fr_residual_inviscid.py::_distribute_from_face`（reshape+
    tensordot+moveaxis，numba 不支持这两个 numpy 函数）等价地表达成一个
    纯索引映射：每个输出 SP 只从唯一一个 (fp 行, g_prime 分量) 组合取值
    （`_distribute_from_face` 本质是外积，任意基向量探针只会点亮一个
    输出位置）。

    不手工重新推导 moveaxis 的下标代数（容易出转录错误），而是直接用
    one-hot 基向量喂给现有、已经过自由流场保持性等测试验证过的
    `_distribute_from_face` 本身，机械地读出映射关系——映射的正确性
    100% 继承自那个函数的正确性，不引入新的推导风险。

    Returns:
        (fp_of_sp, axis_coord_of_sp)，各自形状 (3, n_sps)：
        对 axis in {0,1,2}、SP 下标 s，
        `contrib[s,:] = g_prime[axis_coord_of_sp[axis,s]] * fp_data[fp_of_sp[axis,s],:]`
        与 `_distribute_from_face(fp_data, n1d, axis, g_prime)[s,:]` 逐位相等。
    """
    cached = _DIST_MAP_CACHE.get(n1d)
    if cached is not None:
        return cached

    from autoflowcfd.core.fr_residual.inviscid import _distribute_from_face

    n_fp = n1d * n1d
    n_sps = n1d ** 3
    fp_of_sp = np.full((3, n_sps), -1, dtype=np.int64)
    axis_coord_of_sp = np.full((3, n_sps), -1, dtype=np.int64)

    for axis in range(3):
        for i in range(n_fp):
            for p in range(n1d):
                fp_probe = np.zeros((n_fp, 1))
                fp_probe[i, 0] = 1.0
                g_probe = np.zeros(n1d)
                g_probe[p] = 1.0
                result = _distribute_from_face(fp_probe, n1d, axis, g_probe)[:, 0]
                hits = np.flatnonzero(np.abs(result - 1.0) < 1e-12)
                if len(hits) != 1:
                    raise RuntimeError(
                        f"_derive_distribute_mapping: n1d={n1d} axis={axis} i={i} p={p} "
                        f"探针命中 {len(hits)} 个输出位置（应恰好 1 个）——"
                        f"_distribute_from_face 的外积结构假设不成立，必须先查清原因。"
                    )
                s = hits[0]
                fp_of_sp[axis, s] = i
                axis_coord_of_sp[axis, s] = p

    if np.any(fp_of_sp < 0) or np.any(axis_coord_of_sp < 0):
        raise RuntimeError(f"_derive_distribute_mapping: n1d={n1d} 存在未被任何探针覆盖的 SP。")

    result = (fp_of_sp, axis_coord_of_sp)
    _DIST_MAP_CACHE[n1d] = result
    return result


@dataclass
class FlatFaceGeometry:
    """`mesh.face_flux_points` + `mesh.face_connectivity` 的展平数组表示。

    严格保持与 `range(fc.n_faces)` 相同的面序（不能为了"cache 友好"按
    axis/side 重新分组排序）——`correction[cell] += ...` 的累加顺序会
    影响退化 Jacobian 单元处的舍入误差量级（真实复现过 3 步内从 4e-2
    放大到 1.16e7 的案例，见 fr_viscous_flux.py 模块文档），顺序变了会让
    "新旧实现逐位对比"这个验证判据失去意义。
    """

    n_faces: int
    n_fp: int
    n_sps: int
    n_prism: int

    owner_cell: np.ndarray       # int64 (n_faces,)
    neighbor_cell: np.ndarray    # int64 (n_faces,)，边界面为 -1
    is_boundary: np.ndarray      # bool (n_faces,)
    owner_axis: np.ndarray       # int64 (n_faces,)
    owner_side: np.ndarray       # float64 (n_faces,)
    neighbor_axis: np.ndarray    # int64 (n_faces,)
    neighbor_side: np.ndarray    # float64 (n_faces,)
    owner_is_primary: np.ndarray     # bool (n_faces,)
    neighbor_is_primary: np.ndarray  # bool (n_faces,)
    true_normal: np.ndarray      # float64 (n_faces, n_fp, 3)
    # 真实 bug 修复（2026-08-23，见 fr/face_flux_points_exact_normal.py
    # 模块文档）：owner/neighbor 各自的精确 adj(J) 行（未归一化、未按
    # side 定向），取代此前 inviscid_kernel.py 内部对 SP 网格 adj_j 做
    # Lagrange 外插得到"自洽方向"的做法——外插对坍缩坐标下本质是有理
    # 函数的 adj(J) 有截断误差，直接在 FP 精确坐标求值消除这部分误差。
    # neighbor 侧对边界面为全零占位（无意义，kernel 内 is_boundary 分支
    # 不会读取）。
    owner_adj_row_exact: np.ndarray     # float64 (n_faces, n_fp, 3)
    neighbor_adj_row_exact: np.ndarray  # float64 (n_faces, n_fp, 3)
    # native 四面体（路径C，Part8 文档"三、本次会话实现范围"）支持新增：
    # 原始 cube face 编码（0~5 坍缩坐标，6~9 native，见
    # grid/connectivity/face_connectivity.py::CUBE_FACE_CODES），
    # `owner_axis`/`owner_side`（上面两个字段）对 native 面存的是复用
    # 的 excluded_vertex/哑值，不能用来判断是否 native、也不能安全地
    # 当 axis/side 语义使用——必须用这两个原始编码字段消除歧义。
    owner_cube_face: np.ndarray     # int64 (n_faces,)
    neighbor_cube_face: np.ndarray  # int64 (n_faces,)
    # 物理面积权重（`fr/face_flux_points_exact_normal.py::compute_exact_
    # face_normals_and_weights` 已经算好、验证过的量）——坍缩坐标的
    # 1D Radau/VCJH 修正函数 + 微分矩阵机制不需要它（那套数学结构本身
    # 不含物理面积因子），但 native 四面体的 DG 提升算子（`native_
    # simplex_basis.py::build_native_tet_lift` 文档"弱形式提升定义"）
    # 需要它按面点物理面积加权跳跃量，因此新增这个字段——对坍缩坐标
    # 面同样有意义（本来就已经算好），只是此前从未被这个 kernel 消费过。
    true_area_weight: np.ndarray    # float64 (n_faces, n_fp)

    # --- neighbor_sources（owner 侧用来组装 Q_neighbor 的来源）---
    neighbor_src0_cell: np.ndarray   # int64 (n_faces,)，-1 表示无来源
    neighbor_src0_mat: np.ndarray    # float64 (n_faces, n_fp, n_sps)
    neighbor_src1_idx: np.ndarray    # int64 (n_faces,)，-1 表示没有第2个来源
    neighbor_src1_cell: np.ndarray   # int64 (n_extra,) 紧凑数组
    neighbor_src1_mat: np.ndarray    # float64 (n_extra, n_fp, n_sps) 紧凑数组

    # --- owner_sources（neighbor 侧用来组装 Q_owner_at_n 的来源）---
    owner_src0_cell: np.ndarray
    owner_src0_mat: np.ndarray
    owner_src1_idx: np.ndarray
    owner_src1_cell: np.ndarray
    owner_src1_mat: np.ndarray

    # --- 混合分组面（B-8：棱柱四边形侧面三角化拆分后一条子面落在域边界、
    #     另一条为内部界面；见 fr/face_flux_points_merge.py 混合分组检测块文档）---
    # 内部界面侧：mixed_nb_partner[f_int] = 配对的边界面索引（-1 表示非混合面）；
    # mixed_nb_mask[f_int] 逐 FP 标记边界半区（True 处 Q_neighbor 应取配对面幽灵态）。
    mixed_nb_partner: np.ndarray   # int64 (n_faces,)
    mixed_nb_mask: np.ndarray      # bool (n_faces, n_fp)
    # neighbor 侧对称：mixed_ow_partner[f_int] = 配对的边界面索引，供
    # neighbor-primary 分支组装 Q_owner_at_n 时覆盖边界半区。
    mixed_ow_partner: np.ndarray   # int64 (n_faces,)
    mixed_ow_mask: np.ndarray      # bool (n_faces, n_fp)
    # 边界面侧：mixed_bnd_face[bf] = True 表示该边界面是混合配对的边界半区。
    # 这类面 owner_primary 已被置 False（不参与残差累加，整张面由配对的内部面记录），
    # 但幽灵态仍需计算（compute_boundary_ghost_states 据此保留）。
    mixed_bnd_face: np.ndarray     # bool (n_faces,)
    # P0 专用：混合面中边界子面的面积占比（面积加权混合通量用），非混合面为 0。
    mixed_p0_bnd_frac: np.ndarray  # float64 (n_faces,)

    # --- 外插算子（重堆叠自 ops.boundary_extrap_tet/prism 这两个
    #     Dict[(axis:int,side:float), ndarray]，numba nopython 模式不支持
    #     这种 float 键的 dict）---
    # 形状 (2, 3, 2, n_fp, n_sps)：[celltype(0=prism,1=tet), axis, side_idx(0:-1,1:+1)]
    boundary_extrap: np.ndarray

    # --- native 四面体（路径C）专属算子（Part8 文档），键是 excluded_
    #     vertex（0~3），不含坍缩坐标网格时是零长度占位（对应分支永远
    #     不会被 code>=6 触发，见 native_mode_active 说明）---
    # 体积->自身面外插矩阵，(4, n_fp, n_sps)（列已填充到全局 n_sps 宽度，
    # 与坍缩坐标 boundary_extrap 消费方式一致：E @ Q_volume_nodal）。
    boundary_extrap_native: np.ndarray
    # DG 提升算子，(4, n_sps, n_fp)（行已填充到 n_sps，见
    # native_tet_padding.py::pad_native_tet_matrix_to_global 与
    # native_simplex_basis.py::build_native_tet_lift 文档）。
    lift_native: np.ndarray

    # --- g_left/g_right（Radau/VCJH 校正函数导数，(n1d,) 向量，随 side 选择）---
    g_left: np.ndarray
    g_right: np.ndarray
    n1d: int

    # --- _distribute_from_face 等价的索引映射（见 _derive_distribute_mapping
    #     文档），形状 (3, n_sps)：[axis] -> (fp_of_sp, axis_coord_of_sp)
    dist_fp_of_sp: np.ndarray        # int64 (3, n_sps)
    dist_axis_coord_of_sp: np.ndarray  # int64 (3, n_sps)

    # --- 面图着色（消除 scatter-add 写冲突，替代 per-thread buffer）---
    # 在 build 时一次性计算，后续残差求值直接复用，不再重复着色。
    # color_face_indices[c] = 颜色 c 的面索引数组（int64），
    # n_colors = 总颜色数。同色面之间无 owner_cell 冲突，可安全直接
    # prange + 写入共享 buffer。
    color_face_indices: list  # list of np.ndarray (int64), length = n_colors
    n_colors: int


def build_flat_face_geometry(mesh, ops) -> FlatFaceGeometry:
    """把 `mesh.face_flux_points` + `mesh.face_connectivity` 展平成
    `FlatFaceGeometry`。不缓存（缓存由 `get_flat_face_geometry` 负责），
    每次调用都重新构建——调用方必须通过 `get_flat_face_geometry` 走缓存。

    `mesh.face_flux_points` 恒为 `_KernelFaceData`（numba kernel 直接输出
    的 flat 数组容器，见 `fr/face_flux_points_merge.py::build_face_flux_
    points` 唯一的 return 语句）——本函数不再有"逐面 Python 对象访问"的
    慢速路径分支（2026-09-03 删除，全仓库确认过该分支自 2026-08-30 起
    从未有任何调用方触发过，见该次删除记录）。
    """
    fc = mesh.face_connectivity
    ffp_data = mesh.face_flux_points
    n_faces = fc.n_faces
    n1d = mesh.n_points_1d
    n_fp = n1d * n1d
    n_sps = n1d ** 3
    n_prism = mesh.n_prism_cells

    # 原始 cube face 编码：`fc` 本身就带着，native 分支据此判断（code>=6，
    # 见 FlatFaceGeometry.owner_cube_face 文档），不依赖 owner_axis/
    # owner_side 那套可能有歧义的复用槽位。
    owner_cube_face = fc.owner_cube_face.astype(np.int64)
    neighbor_cube_face = fc.neighbor_cube_face.astype(np.int64)

    from autoflowcfd.fr.face_flux_points_merge import _KernelFaceData
    if not isinstance(ffp_data, _KernelFaceData):
        raise TypeError(
            f"build_flat_face_geometry: mesh.face_flux_points 必须是 "
            f"_KernelFaceData（build_face_flux_points 的唯一产出类型），"
            f"收到的是 {type(ffp_data).__name__}——这不是一个已知的合法"
            f"构造方式，之前支持过的逐面对象慢速路径已确认没有任何调用方"
            f"后删除（2026-09-03），如果这里真的需要一种新的 mesh.face_"
            f"flux_points 构造方式，需要先补上对应的展平逻辑，不能假装"
            f"可以正确产出结果。"
        )
    owner_axis = ffp_data.owner_axis
    owner_side = ffp_data.owner_side
    neighbor_axis = ffp_data.neighbor_axis
    neighbor_side = ffp_data.neighbor_side
    owner_is_primary = ffp_data.owner_is_primary
    neighbor_is_primary = ffp_data.neighbor_is_primary
    true_normal = ffp_data.true_normal
    neighbor_src0_cell = ffp_data.nb_src0_cell
    neighbor_src0_mat = ffp_data.nb_src0_mat
    neighbor_src1_idx = ffp_data.nb_src1_idx
    owner_src0_cell = ffp_data.ow_src0_cell
    owner_src0_mat = ffp_data.ow_src0_mat
    owner_src1_idx = ffp_data.ow_src1_idx
    # extra sources → src1（紧凑数组，通过 src1_idx 索引）
    neighbor_src1_cell = ffp_data.nb_extra_cell
    neighbor_src1_mat = ffp_data.nb_extra_mat
    owner_src1_cell = ffp_data.ow_extra_cell
    owner_src1_mat = ffp_data.ow_extra_mat
    owner_adj_row_exact = ffp_data.owner_adj_row_exact
    neighbor_adj_row_exact = ffp_data.neighbor_adj_row_exact
    true_area_weight = ffp_data.true_area_weight
    # 混合分组面（B-8）：merge 层检测后填入，_KernelFaceData 恒定提供这 6 个数组。
    mixed_nb_partner = ffp_data.mixed_nb_partner
    mixed_nb_mask = ffp_data.mixed_nb_mask
    mixed_ow_partner = ffp_data.mixed_ow_partner
    mixed_ow_mask = ffp_data.mixed_ow_mask
    mixed_bnd_face = ffp_data.mixed_bnd_face
    mixed_p0_bnd_frac = ffp_data.mixed_p0_bnd_frac

    # boundary_extrap_tet/prism: Dict[(axis:int,side:float), (n_fp,n_sps)矩阵]
    # -> (2,3,2,n_fp,n_sps)，[celltype(0=prism,1=tet), axis, side_idx]
    boundary_extrap = np.zeros((2, 3, 2, n_fp, n_sps), dtype=np.float64)
    for axis in range(3):
        for side_idx, side in enumerate((-1.0, 1.0)):
            boundary_extrap[0, axis, side_idx] = ops.boundary_extrap_prism[(axis, side)]
            boundary_extrap[1, axis, side_idx] = ops.boundary_extrap_tet[(axis, side)]

    # native 四面体（路径C）专属算子（Part8 文档）：`ops.boundary_extrap_
    # native_tet`/`ops.lift_native_tet_padded` 只在 `tet_basis_mode==
    # "native"` 时非 None——不含 native 四面体的既有网格传零长度占位
    # 数组，下游 kernel 对应分支（判据同样是 code>=6）永远不会被执行，
    # 不改变任何现有行为（与 face_flux_points_merge.py 里同一个"自动
    # 探测/零占位"原则一致）。boundary_extrap_native 的列同样需要填充
    # 到全局 n_sps 宽度（native_tet_boundary_extrap 原始形状是
    # (n_fp,n_native)，不像 D_native_tet_padded/lift_native_tet_padded
    # 那样已经在 fr/operators.py 里填充过）。
    if ops.boundary_extrap_native_tet is not None:
        from autoflowcfd.fr.native_tet_padding import pad_native_tet_matrix_to_global

        boundary_extrap_native = np.zeros((4, n_fp, n_sps), dtype=np.float64)
        lift_native = np.zeros((4, n_sps, n_fp), dtype=np.float64)
        for ev in range(4):
            boundary_extrap_native[ev] = pad_native_tet_matrix_to_global(
                ops.boundary_extrap_native_tet[ev], n_sps, pad_axes=(1,)
            )
            lift_native[ev] = ops.lift_native_tet_padded[ev]
    else:
        boundary_extrap_native = np.zeros((0, n_fp, n_sps), dtype=np.float64)
        lift_native = np.zeros((0, n_sps, n_fp), dtype=np.float64)

    dist_fp_of_sp, dist_axis_coord_of_sp = _derive_distribute_mapping(n1d)

    # 面图着色：一次性计算，后续残差求值直接复用（不再重复着色）。
    # 贪心着色覆盖 owner_cell 与非边界面 neighbor_cell 两侧的写冲突——
    # 图着色 kernel 对内部面会分别向 owner_cell 与 neighbor_cell 做非原子
    # scatter-add，只按 owner_cell 分组会漏掉跨面的 owner/neighbor 交叉
    # 冲突（面 A 的 owner 是面 B 的 neighbor），必须一并传入才能保证同色
    # 面之间真正无写冲突。
    from autoflowcfd.core.utils.face_coloring import greedy_face_coloring
    n_cells_est = int(np.max(fc.owner_cell)) + 1
    colors = greedy_face_coloring(
        fc.owner_cell.astype(np.int64), n_cells_est,
        fc.neighbor_cell.astype(np.int64), fc.is_boundary.astype(np.bool_),
    )
    n_colors = int(np.max(colors)) + 1
    color_face_indices = [
        np.where(colors == c)[0].astype(np.int64) for c in range(n_colors)
    ]

    return FlatFaceGeometry(
        n_faces=n_faces, n_fp=n_fp, n_sps=n_sps, n_prism=n_prism,
        owner_cell=fc.owner_cell.astype(np.int64),
        neighbor_cell=fc.neighbor_cell.astype(np.int64),
        is_boundary=fc.is_boundary.astype(np.bool_),
        owner_axis=owner_axis, owner_side=owner_side,
        neighbor_axis=neighbor_axis, neighbor_side=neighbor_side,
        owner_is_primary=owner_is_primary, neighbor_is_primary=neighbor_is_primary,
        true_normal=true_normal,
        owner_cube_face=owner_cube_face, neighbor_cube_face=neighbor_cube_face,
        true_area_weight=true_area_weight,
        owner_adj_row_exact=owner_adj_row_exact, neighbor_adj_row_exact=neighbor_adj_row_exact,
        neighbor_src0_cell=neighbor_src0_cell, neighbor_src0_mat=neighbor_src0_mat,
        neighbor_src1_idx=neighbor_src1_idx, neighbor_src1_cell=neighbor_src1_cell,
        neighbor_src1_mat=neighbor_src1_mat,
        owner_src0_cell=owner_src0_cell, owner_src0_mat=owner_src0_mat,
        owner_src1_idx=owner_src1_idx, owner_src1_cell=owner_src1_cell,
        owner_src1_mat=owner_src1_mat,
        mixed_nb_partner=mixed_nb_partner, mixed_nb_mask=mixed_nb_mask,
        mixed_ow_partner=mixed_ow_partner, mixed_ow_mask=mixed_ow_mask,
        mixed_bnd_face=mixed_bnd_face, mixed_p0_bnd_frac=mixed_p0_bnd_frac,
        boundary_extrap=boundary_extrap,
        boundary_extrap_native=boundary_extrap_native,
        lift_native=lift_native,
        g_left=np.asarray(ops.g_left, dtype=np.float64),
        g_right=np.asarray(ops.g_right, dtype=np.float64),
        n1d=n1d,
        dist_fp_of_sp=dist_fp_of_sp,
        dist_axis_coord_of_sp=dist_axis_coord_of_sp,
        color_face_indices=color_face_indices,
        n_colors=n_colors,
    )


def get_flat_face_geometry(mesh, ops) -> FlatFaceGeometry:
    """缓存版本，键为 `mesh.face_flux_points` 这个 list 对象本身的身份
    （不是 mesh 本身的身份——Order Continuation 在同一个 mesh 对象上原地
    替换这个 list，见模块文档"缓存失效"一节；也不是裸 `id()` 整数——
    见模块文档"缓存键危险陷阱"一节，裸 id() 有被另一个无关 mesh 复用
    撞车的真实风险）。"""
    global _FLAT_CACHE_KEY, _FLAT_CACHE_VALUE
    if _FLAT_CACHE_KEY is mesh.face_flux_points and _FLAT_CACHE_VALUE is not None:
        return _FLAT_CACHE_VALUE
    flat = build_flat_face_geometry(mesh, ops)
    # 只保留最近一次几何，避免阶数切换间反复累积旧缓存占用内存；同时
    # 持有 mesh.face_flux_points 的强引用本身作为键，防止其被 GC 后
    # 另一个无关对象复用同一身份。
    _FLAT_CACHE_KEY = mesh.face_flux_points
    _FLAT_CACHE_VALUE = flat
    return flat
