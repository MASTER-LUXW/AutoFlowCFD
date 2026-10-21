"""FR 面几何展平缓存：把面通量点数据展平成 numba kernel 可直接读的定长数组。

`mesh.face_flux_points`（`fr/face_flux_points/data.py::_KernelFaceData`）里 `owner_sources`/`neighbor_sources` 是
变长的 `(cell_id, matrix)` 元组列表，numba nopython 模式不能直接消费；本模块把它一次性展平成定长 numpy 数组，
供无粘 / 粘性界面 kernel（`core/fr_residual/inviscid_kernel.py`、`core/fr_residual/viscous_flux_kernel.py`）共用
（面几何在无粘 / 粘性残差之间共享，且在同一批 RK 子迭代内不变）。

缓存：展平几何只依赖网格（同一阶数），挂在 `mesh.face_flux_points` 自己的 `_flat_geometry` 上，与它同生命周期。
Order Continuation 换阶时整体替换 `mesh.face_flux_points`（`grid/high_order/high_order_mesh_order.py`），新对象
没有缓存，自然重建。缓存放在被缓存对象自身上：网格释放后不滞留，两张网格交替求值也不会命中对方的几何。

内存设计：`owner_sources`/`neighbor_sources` 长度恒为 1 或 2（网格生成器
把棱柱四边形侧面恒定拆分成 2 个三角子面，不会更多，见
`fr/face_flux_points/merge.py` 模块文档），但绝大多数
面（普通四面体-四面体内部面、未拆分的棱柱面）只有 1 个来源。如果统一按
2 槽稠密填充，多出来的一半矩阵纯粹是浪费——在 P2、n_fp=9、n_sps=18（原生棱柱）下，
1.3M 面 × 2 角色(owner/neighbor) × 2 槽 × (9×18×8字节) 约 6.7GB，这在真实
生产网格上不是可以忽略的开销。因此设计成"稠密槽 0 + 稀疏槽 1"：槽 0
对每个面恒定存在（大小 n_faces），槽 1 只对真正有第 2 个来源的那一小
部分面额外分配一个更小的紧凑数组，用 -1 表示的下标数组做重定向。
"""

from dataclasses import dataclass
from functools import cached_property

import numpy as np

from .face_points import p0_flux_points




@dataclass
class FlatFaceGeometry:
    """`mesh.face_flux_points` + `mesh.face_connectivity` 的展平数组表示。

    严格保持与 `range(fc.n_faces)` 相同的面序（不能为了"cache 友好"按
    axis/side 重新分组排序）——`correction[cell] += ...` 的累加顺序会
    影响舍入误差，顺序变了会让与逐面参考实现逐位对比的验证判据失去意义。
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
    # owner/neighbor 各自在通量点精确坐标上求值的 adj(J) 行（未归一化、原生面已外向定向，
    # `fr/face_flux_points/exact_normal.py`）。neighbor 侧对边界面为全零占位（kernel 内 is_boundary
    # 分支不读取）。
    owner_adj_row_exact: np.ndarray     # float64 (n_faces, n_fp, 3)
    neighbor_adj_row_exact: np.ndarray  # float64 (n_faces, n_fp, 3)
    # 面编码（四面体面 6~9、原生棱柱面 10~14，`grid/connectivity/face_connectivity.py`）。
    # `owner_axis`/`owner_side`（上面两个字段）对原生面存的是复用的 excluded_vertex / 哑值，不能当
    # axis/side 语义使用——面的类别一律按这两个编码字段判断。
    owner_cube_face: np.ndarray     # int64 (n_faces,)
    neighbor_cube_face: np.ndarray  # int64 (n_faces,)
    # 物理面积权重（`fr/face_flux_points/exact_normal.py::compute_exact_face_normals_and_weights`），
    # `|adj_row| * w_ref`，沿 FP 轴求和即物理面积（用于 `ip_length` 等几何量；DG 提升用下面的参考权重）。
    true_area_weight: np.ndarray    # float64 (n_faces, n_fp)
    # 面记录 P0 通量点的物理坐标，float64 (n_faces, 3)：P0 每面一个通量点，解点外插只能得到单元中心，
    # 由网格节点直接算出（`face_points.py::p0_flux_points`；拆分四边形的两条子面记录都取四边形中心）。
    # 通量点坐标一律经 `face_points.py::flux_point_positions` 取（P>=1 外插、P0 取这里）
    face_point: np.ndarray          # float64 (n_faces, 3)
    # **参考**面求积权重，float64 (n_fp,) —— 逐面相同（所有面都用同一套
    # `[-1,1]^2` 上的张量积 Gauss-Legendre 网格），所以只存一份。
    #
    # DG 提升算子的权重是**参考**求积权重，不是 `true_area_weight`：无粘核施加的 `|adj_row| F*` 已经是
    # **参考空间**的法向通量（Nanson 关系 `n_hat dA_phys = adj_row dA_ref` 把物理面积因子吃进去了），
    # 再乘物理面积权重等于多乘一个 `|adj_row| ~ h^2`，界面项从 `~1/h` 变成 `~h`。判据：坐标整体缩放
    # 减半，界面残差应放大 2.000 倍；常数跳跃 `F* = F_own + delta*n_hat` 对单元总量的贡献应恰为
    # `-delta * 总表面积`。
    ref_area_weight: np.ndarray     # float64 (n_fp,)

    # 逐面**内罚项长度尺度** `h_f = min(V_owner, V_neighbor) / A_face`（边界面取
    # `V_owner / A_face`），float64 (n_faces,)。平均流粘性与湍流扩散的 IP 罚项都读它
    # （唯一定义）。
    #
    # * `A_face` = `true_area_weight` 沿 FP 轴求和（与解析面积之比在平板算例六个
    #   边界平面上全部 1.000000，含原生棱柱的两个三角形封盖）；
    # * `V` = `mesh.get_all_cell_volumes()` 的精确体积。**不能**用 `mean(det_jacs)`：那是
    #   "体积 ÷ 参考单元体积"，随单元类型而变（原生棱柱 4、原生四面体 4/3）；也不能用几何平均
    #   `V**(1/3)`：各向异性贴壁单元上比壁法向厚度大 2.48 倍（实测平板）。
    #   `V/A` 是面法向的单元厚度（直棱柱贴壁单元上恰好等于 dy）；
    # * **两侧单值**：罚项是公共通量的一部分，两侧若各用自己的
    #   `V/A`，相邻体积不等时同一个面上两侧收到的罚通量不等，离散扩散不守恒。
    #   取较薄的一侧：对两侧都满足迹不等式的下界（强制性）。
    ip_length: np.ndarray           # float64 (n_faces,)

    # --- neighbor_sources（owner 侧用来组装 Q_neighbor 的来源）---
    neighbor_src0_cell: np.ndarray   # int64 (n_faces,)，-1 表示无来源
    # 第 f 个面的矩阵是 neighbor_src0_tpl[neighbor_src0_tid[f]]（模板表，见
    # fr/face_flux_points/templates.py）
    neighbor_src0_tpl: np.ndarray    # float64 (n_tpl, n_fp, n_sps)
    neighbor_src0_tid: np.ndarray    # int32 (n_faces,)
    neighbor_src1_idx: np.ndarray    # int64 (n_faces,)，-1 表示没有第2个来源
    neighbor_src1_cell: np.ndarray   # int64 (n_extra,) 紧凑数组
    neighbor_src1_mat: np.ndarray    # float64 (n_extra, n_fp, n_sps) 紧凑数组

    # --- owner_sources（neighbor 侧用来组装 Q_owner_at_n 的来源）---
    owner_src0_cell: np.ndarray
    owner_src0_tpl: np.ndarray
    owner_src0_tid: np.ndarray
    owner_src1_idx: np.ndarray
    owner_src1_cell: np.ndarray
    owner_src1_mat: np.ndarray

    # --- 混合分组面（B-8：棱柱四边形侧面三角化拆分后一条子面落在域边界、
    #     另一条为内部界面；见 fr/face_flux_points/merge.py 混合分组检测块文档）---
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
    # 四边形面第二个半区（混合面的边界半区、真多源面的次源半区）的面积占比（P0 粘性核按它加权两半的跳量，
    # 见 viscous_p0_kernel.py），其余面为 0。
    p0_second_half_frac: np.ndarray  # float64 (n_faces,)

    # --- 原生基面算子：按面算子索引 `op = (code - 6) + 9 * slot` 排好的整表
    #     （`fr/operators/face_ops.py`；四面体面 6~9、棱柱面 10~14，三角形面再按三角形采样的
    #     坍缩顶点槽位 `fr/triangle_apex.py` 分 3 份）。逐面取 `[owner_face_op[f]]` /
    #     `[neighbor_face_op[f]]`。
    # 体积->自身面外插矩阵，(27, n_fp, n_sps)（列已填充到全局 n_sps 宽度：
    # 消费方式恒为 `E @ Q_volume_nodal`；四边形面不存在的槽位行是 NaN）。
    boundary_extrap_native: np.ndarray
    # DG 提升算子，(27, n_sps, n_fp)（行已填充到 n_sps，见
    # native_padding.py::pad_native_matrix_to_global 与
    # native_tet/basis.py::build_native_tet_lift 文档）。
    lift_native: np.ndarray
    # 两侧的面算子索引，int64 (n_faces,)；边界面的 neighbor 侧为 0（不被读取）。
    owner_face_op: np.ndarray
    neighbor_face_op: np.ndarray

    n1d: int

    # --- 面图着色（消除 scatter-add 写冲突，替代 per-thread buffer）---
    # 在 build 时一次性计算，后续残差求值直接复用，不再重复着色。
    # color_face_indices[c] = 颜色 c 的面索引数组（int64），
    # n_colors = 总颜色数。同色面之间无 owner_cell 冲突，可安全直接
    # prange + 写入共享 buffer。
    color_face_indices: list  # list of np.ndarray (int64), length = n_colors
    n_colors: int

    # 逐通量点的单位外法向（adj 行归一化，与 `true_normal` 那个面平面法向不是一回事：扭曲面上
    # 两者最多差 37°，见 `troubled_cell.py`）。只依赖几何，每个几何对象算一次（湍流对流/扩散
    # 残差与其解析 Jacobian 每次求值都要用；此前每次现算，plate P1 每次湍流速率约 0.2 s）。
    @cached_property
    def owner_unit_normal(self) -> np.ndarray:
        return unit_normals(self.owner_adj_row_exact)

    @cached_property
    def neighbor_unit_normal(self) -> np.ndarray:
        return unit_normals(self.neighbor_adj_row_exact)


def unit_normals(adj_row: np.ndarray) -> np.ndarray:
    """`(n_faces, n_fp, 3)` 的 adj 行归一化成该侧的单位外法向（GPU 上传这份结果，见
    `gpu/gpu_face_geometry.py`）。"""
    adj_row = np.asarray(adj_row)
    mag = np.sqrt(np.sum(adj_row * adj_row, axis=-1))
    return np.ascontiguousarray(adj_row / np.maximum(mag, 1e-300)[..., None])


def _check_one_primary_side_per_cell_face(owner, neighbor, is_boundary, owner_primary, neighbor_primary,
                                          owner_code, neighbor_code, n_cells: int, n_prism: int) -> None:
    """每个单元的每个几何面恰有一条"本单元为 primary 侧"的面记录（四面体 4 个、棱柱
    5 个面编码各一次）。无粘修正项的本侧通量迹并进了体积算子
    （`fr/face_flux_trace.py`），按单元的全部面一次性减去——这条不成立时残差会
    静默多减或漏减一个面，所以在构造期核查。"""
    owner, neighbor = np.asarray(owner, np.int64), np.asarray(neighbor, np.int64)
    nb_side = (~np.asarray(is_boundary, bool)) & np.asarray(neighbor_primary, bool)
    ow_side = np.asarray(owner_primary, bool)
    cells = np.concatenate([owner[ow_side], neighbor[nb_side]])
    codes = np.concatenate([np.asarray(owner_code, np.int64)[ow_side], np.asarray(neighbor_code, np.int64)[nb_side]])
    key = cells * 16 + codes
    uniq, counts = np.unique(key, return_counts=True)
    per_cell = np.bincount(uniq // 16, minlength=n_cells)
    expected = np.where(np.arange(n_cells) < n_prism, 5, 4)
    if counts.max(initial=1) > 1 or np.any(per_cell != expected):
        bad = np.nonzero(per_cell != expected)[0]
        raise ValueError(
            f"面记录不满足'每个单元每个面恰一条 primary 侧'：重复 {int((counts > 1).sum())} 处，"
            f"面数不符的单元 {bad.size} 个（例如 {bad[:5].tolist()}）。无粘体积算子按单元全部面"
            f"减去修正项的本侧通量迹，这条不成立时残差会静默错误")


def build_flat_face_geometry(mesh, ops) -> FlatFaceGeometry:
    """把 `mesh.face_flux_points` + `mesh.face_connectivity` 展平成
    `FlatFaceGeometry`。不缓存（缓存由 `get_flat_face_geometry` 负责），
    每次调用都重新构建——调用方必须通过 `get_flat_face_geometry` 走缓存。

    `mesh.face_flux_points` 恒为 `_KernelFaceData`（numba kernel 直接输出
    的 flat 数组容器，见 `fr/face_flux_points/merge.py::build_face_flux_
    points` 唯一的 return 语句）——本函数不再有"逐面 Python 对象访问"的
    慢速路径分支（2026-09-03 删除，全仓库确认过该分支自 2026-08-30 起
    从未有任何调用方触发过，见该次删除记录）。
    """
    fc = mesh.face_connectivity
    ffp_data = mesh.face_flux_points
    n_faces = fc.n_faces
    n1d = mesh.n_points_1d
    n_fp = n1d * n1d
    n_sps = mesh.n_sps_per_cell
    n_prism = mesh.n_prism_cells

    # 原始 cube face 编码：`fc` 本身就带着，native 分支据此判断（四面体
    # native 面是 [6,10)，见 FlatFaceGeometry.owner_cube_face 文档），
    # 不依赖 owner_axis/owner_side 那套可能有歧义的复用槽位。
    owner_cube_face = fc.owner_cube_face.astype(np.int64)
    neighbor_cube_face = fc.neighbor_cube_face.astype(np.int64)


    from autoflowcfd.fr.face_flux_points.merge import _KernelFaceData
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
    neighbor_src0_tpl, neighbor_src0_tid = ffp_data.nb_src0_tpl, ffp_data.nb_src0_tid
    neighbor_src1_idx = ffp_data.nb_src1_idx
    owner_src0_cell = ffp_data.ow_src0_cell
    owner_src0_tpl, owner_src0_tid = ffp_data.ow_src0_tpl, ffp_data.ow_src0_tid
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
    p0_second_half_frac = ffp_data.p0_second_half_frac

    # 原生面外插 / 提升整表与逐面算子索引（见 `FlatFaceGeometry.boundary_extrap_native`
    # 字段文档）。槽位由单元连接关系唯一决定（`fr/face_flux_points/merge.py`）。
    from autoflowcfd.fr.operators.face_ops import face_op_index

    if ops.face_extrap_by_op is None:
        raise ValueError("算子集里没有原生面算子整表 —— 算子构造本身失败了")
    boundary_extrap_native = ops.face_extrap_by_op
    lift_native = ops.face_lift_by_op
    is_bnd = np.asarray(fc.is_boundary, dtype=bool)
    owner_face_op = np.ascontiguousarray(face_op_index(owner_cube_face, ffp_data.owner_tri_slot), dtype=np.int64)
    neighbor_face_op = np.ascontiguousarray(
        np.where(is_bnd, 0, face_op_index(neighbor_cube_face, ffp_data.neighbor_tri_slot)), dtype=np.int64)

    # 参考面求积权重（见 `ref_area_weight` 字段文档）：所有面共用同一套
    # `[-1,1]^2` 张量积 Gauss-Legendre 网格，与 `fr/face_flux_points/merge.py`
    # 里生成面通量点用的是**同一个** `gauss_legendre(n1d)`；无粘体积算子 `K` 构造
    # 时用的也是同一个函数（`fr/face_flux_trace.py`）。
    from autoflowcfd.fr.face_flux_trace import face_reference_weights

    ref_area_weight = face_reference_weights(n1d - 1)
    _check_one_primary_side_per_cell_face(
        fc.owner_cell, fc.neighbor_cell, fc.is_boundary, owner_is_primary, neighbor_is_primary,
        owner_cube_face, neighbor_cube_face, int(mesh.n_cells), n_prism)

    # 内罚项长度尺度（见同名字段文档）：只依赖几何，随 flat 几何一次性构造并缓存。
    face_area = np.asarray(true_area_weight, dtype=np.float64).sum(axis=1)
    cell_volume = np.asarray(mesh.get_all_cell_volumes(), dtype=np.float64)
    if cell_volume.shape[0] != int(mesh.n_cells):
        raise ValueError(
            f"cell_volume 长度 {cell_volume.shape[0]} 与网格单元数 "
            f"{int(mesh.n_cells)} 不一致 —— 按 owner/neighbor 单元下标索引它，"
            f"长度不对会静默取到错误的单元")
    _nb = np.asarray(fc.neighbor_cell, dtype=np.int64)
    _v = cell_volume[np.asarray(fc.owner_cell, dtype=np.int64)]
    _v = np.where(_nb >= 0, np.minimum(_v, cell_volume[np.maximum(_nb, 0)]), _v)
    ip_length = np.ascontiguousarray(_v / np.maximum(face_area, 1e-300))

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
        face_point=p0_flux_points(fc.owner_cell, fc.owner_cube_face, fc.face_node_ids, mesh._node_coords),
        ref_area_weight=ref_area_weight,
        ip_length=ip_length,
        owner_adj_row_exact=owner_adj_row_exact, neighbor_adj_row_exact=neighbor_adj_row_exact,
        neighbor_src0_cell=neighbor_src0_cell, neighbor_src0_tpl=neighbor_src0_tpl,
        neighbor_src0_tid=neighbor_src0_tid,
        neighbor_src1_idx=neighbor_src1_idx, neighbor_src1_cell=neighbor_src1_cell,
        neighbor_src1_mat=neighbor_src1_mat,
        owner_src0_cell=owner_src0_cell, owner_src0_tpl=owner_src0_tpl, owner_src0_tid=owner_src0_tid,
        owner_src1_idx=owner_src1_idx, owner_src1_cell=owner_src1_cell,
        owner_src1_mat=owner_src1_mat,
        mixed_nb_partner=mixed_nb_partner, mixed_nb_mask=mixed_nb_mask,
        mixed_ow_partner=mixed_ow_partner, mixed_ow_mask=mixed_ow_mask,
        mixed_bnd_face=mixed_bnd_face, p0_second_half_frac=p0_second_half_frac,
        boundary_extrap_native=boundary_extrap_native,
        lift_native=lift_native,
        owner_face_op=owner_face_op,
        neighbor_face_op=neighbor_face_op,
        n1d=n1d,
        color_face_indices=color_face_indices,
        n_colors=n_colors,
    )


def get_flat_face_geometry(mesh, ops) -> FlatFaceGeometry:
    """`mesh` 当前阶数的展平面几何，缓存在 `mesh.face_flux_points._flat_geometry` 上（见模块文档"缓存"）。"""
    ffp = mesh.face_flux_points
    if ffp._flat_geometry is None:
        ffp._flat_geometry = build_flat_face_geometry(mesh, ops)
    return ffp._flat_geometry
