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

    from autoflowcfd.core.fr_residual_inviscid import _distribute_from_face

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

    # --- 外插算子（重堆叠自 ops.boundary_extrap_tet/prism 这两个
    #     Dict[(axis:int,side:float), ndarray]，numba nopython 模式不支持
    #     这种 float 键的 dict）---
    # 形状 (2, 3, 2, n_fp, n_sps)：[celltype(0=prism,1=tet), axis, side_idx(0:-1,1:+1)]
    boundary_extrap: np.ndarray

    # --- g_left/g_right（Radau/VCJH 校正函数导数，(n1d,) 向量，随 side 选择）---
    g_left: np.ndarray
    g_right: np.ndarray
    n1d: int

    # --- _distribute_from_face 等价的索引映射（见 _derive_distribute_mapping
    #     文档），形状 (3, n_sps)：[axis] -> (fp_of_sp, axis_coord_of_sp)
    dist_fp_of_sp: np.ndarray        # int64 (3, n_sps)
    dist_axis_coord_of_sp: np.ndarray  # int64 (3, n_sps)


def _build_source_arrays(sources_per_face, n_faces: int, n_fp: int, n_sps: int):
    """把每个面最多 2 个 `(cell_id, matrix)` 的变长列表，拆成稠密槽 0 +
    稀疏槽 1 两部分。sources_per_face[f] 是该面的 sources 列表（长度 0/1/2）。
    """
    src0_cell = np.full(n_faces, -1, dtype=np.int64)
    src0_mat = np.zeros((n_faces, n_fp, n_sps), dtype=np.float64)
    src1_idx = np.full(n_faces, -1, dtype=np.int64)

    extra_cells = []
    extra_mats = []

    for f in range(n_faces):
        sources = sources_per_face[f]
        n_src = len(sources)
        if n_src == 0:
            continue
        if n_src > 2:
            raise ValueError(
                f"face {f}: {n_src} sources，超出网格生成器保证的 1~2 个来源不变量"
                f"（见 fr/face_flux_points_merge.py::_resolve_multi_source 文档）——"
                f"说明该不变量已被破坏，必须先查清原因，不能静默截断/忽略多出来的来源。"
            )
        cell0, mat0 = sources[0]
        src0_cell[f] = cell0
        src0_mat[f] = mat0
        if n_src == 2:
            cell1, mat1 = sources[1]
            src1_idx[f] = len(extra_cells)
            extra_cells.append(cell1)
            extra_mats.append(mat1)

    src1_cell = np.asarray(extra_cells, dtype=np.int64) if extra_cells else np.empty((0,), dtype=np.int64)
    src1_mat = (
        np.stack(extra_mats, axis=0) if extra_mats else np.empty((0, n_fp, n_sps), dtype=np.float64)
    )
    return src0_cell, src0_mat, src1_idx, src1_cell, src1_mat


def build_flat_face_geometry(mesh, ops) -> FlatFaceGeometry:
    """把 `mesh.face_flux_points` + `mesh.face_connectivity` 展平成
    `FlatFaceGeometry`。不缓存（缓存由 `get_flat_face_geometry` 负责），
    每次调用都重新构建——调用方必须通过 `get_flat_face_geometry` 走缓存。
    """
    fc = mesh.face_connectivity
    ffp_list = mesh.face_flux_points
    n_faces = fc.n_faces
    n1d = mesh.n_points_1d
    n_fp = n1d * n1d
    n_sps = n1d ** 3
    n_prism = mesh.n_prism_cells

    owner_axis = np.empty(n_faces, dtype=np.int64)
    owner_side = np.empty(n_faces, dtype=np.float64)
    neighbor_axis = np.empty(n_faces, dtype=np.int64)
    neighbor_side = np.empty(n_faces, dtype=np.float64)
    owner_is_primary = np.empty(n_faces, dtype=np.bool_)
    neighbor_is_primary = np.empty(n_faces, dtype=np.bool_)
    true_normal = np.empty((n_faces, n_fp, 3), dtype=np.float64)

    neighbor_sources_per_face = [None] * n_faces
    owner_sources_per_face = [None] * n_faces

    for f in range(n_faces):
        ffp = ffp_list[f]
        owner_axis[f] = ffp.owner_axis
        owner_side[f] = ffp.owner_side
        neighbor_axis[f] = ffp.neighbor_axis
        neighbor_side[f] = ffp.neighbor_side
        owner_is_primary[f] = ffp.owner_is_primary
        neighbor_is_primary[f] = ffp.neighbor_is_primary
        true_normal[f] = ffp.true_normal
        neighbor_sources_per_face[f] = ffp.neighbor_sources
        owner_sources_per_face[f] = ffp.owner_sources

    (neighbor_src0_cell, neighbor_src0_mat, neighbor_src1_idx,
     neighbor_src1_cell, neighbor_src1_mat) = _build_source_arrays(
        neighbor_sources_per_face, n_faces, n_fp, n_sps
    )
    (owner_src0_cell, owner_src0_mat, owner_src1_idx,
     owner_src1_cell, owner_src1_mat) = _build_source_arrays(
        owner_sources_per_face, n_faces, n_fp, n_sps
    )

    # boundary_extrap_tet/prism: Dict[(axis:int,side:float), (n_fp,n_sps)矩阵]
    # -> (2,3,2,n_fp,n_sps)，[celltype(0=prism,1=tet), axis, side_idx]
    boundary_extrap = np.zeros((2, 3, 2, n_fp, n_sps), dtype=np.float64)
    for axis in range(3):
        for side_idx, side in enumerate((-1.0, 1.0)):
            boundary_extrap[0, axis, side_idx] = ops.boundary_extrap_prism[(axis, side)]
            boundary_extrap[1, axis, side_idx] = ops.boundary_extrap_tet[(axis, side)]

    dist_fp_of_sp, dist_axis_coord_of_sp = _derive_distribute_mapping(n1d)

    return FlatFaceGeometry(
        n_faces=n_faces, n_fp=n_fp, n_sps=n_sps, n_prism=n_prism,
        owner_cell=fc.owner_cell.astype(np.int64),
        neighbor_cell=fc.neighbor_cell.astype(np.int64),
        is_boundary=fc.is_boundary.astype(np.bool_),
        owner_axis=owner_axis, owner_side=owner_side,
        neighbor_axis=neighbor_axis, neighbor_side=neighbor_side,
        owner_is_primary=owner_is_primary, neighbor_is_primary=neighbor_is_primary,
        true_normal=true_normal,
        neighbor_src0_cell=neighbor_src0_cell, neighbor_src0_mat=neighbor_src0_mat,
        neighbor_src1_idx=neighbor_src1_idx, neighbor_src1_cell=neighbor_src1_cell,
        neighbor_src1_mat=neighbor_src1_mat,
        owner_src0_cell=owner_src0_cell, owner_src0_mat=owner_src0_mat,
        owner_src1_idx=owner_src1_idx, owner_src1_cell=owner_src1_cell,
        owner_src1_mat=owner_src1_mat,
        boundary_extrap=boundary_extrap,
        g_left=np.asarray(ops.g_left, dtype=np.float64),
        g_right=np.asarray(ops.g_right, dtype=np.float64),
        n1d=n1d,
        dist_fp_of_sp=dist_fp_of_sp,
        dist_axis_coord_of_sp=dist_axis_coord_of_sp,
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
