"""AutoFlowCFD V2.0 - BJ 型邻居极值判据所需的分布式连接关系。

从 `core/fr_solver/filter.py` 拆出（2026-09-24）。纯搬家，逻辑未改。

`bounds` 传感器要拿"邻居单元的极值包络"，在分布式下这需要跨 rank 的面
邻接信息；这份构造是那条路径专属的，与滤波本身正交。
"""


import numpy as np
from loguru import logger


def build_distributed_bounds_conn(dist_fc, n_total_cells, get_halo,
                                 get_provider, freestream, to_device=None,
                                 ascontiguous=None):
    """构造 BJ 越界判据在**分布式**后端上的连接参数（CPU MPI 与多 GPU 共用）。

    两条分布式后端此前各有一份逐字相同的实现。它们唯一的差别是数组模块
    （numpy / CuPy）与 halo 交换对象，其余——索引换算、两条自洽性护栏、
    halo 扩展的时机、两张边界表的惰性构造——完全一样。本项目已多次因为
    "两份实现只改了一份"出真实缺陷，而这一处尤其危险：漏改一条后端不会
    报错，只会让那条后端的掩码悄悄不同。

    ## 三条约束（两条后端逐字相同）

    1. **分区边界面不能当边界面排除**。否则掩码随 rank 数变化，同一个
       算例换分区数得到不同的解。halo 单元的均值必须真的读到。
    2. **halo 扩展必须用当前 stage 的解**。滤波在正定性投影之后施加，
       复用残差求值时缓存的扩展场会比单机路径滞后半个 stage，两条后端
       就不再逐位可比。代价是每 stage 多一次交换
       （`n_halo * n_sps * n_vars`），与 `_compute_distributed_local_
       time_step` 的既有权衡相同。
    3. **索引空间**。场处在 halo 交换的**原生**排列（local 在前、halo
       在后），而 `dist_fc.owner_cell_local`/`neighbor_cell_local` 处在
       "棱柱在前"的**紧凑**排列，用 `perm` 换算：紧凑下标 k 对应原生
       下标 `perm[k]`（`array_native[perm] == array_permuted`），所以
       `owner_native = perm[owner_cell_local]`。掩码在原生扩展空间上算、
       取前 `n_local` 项；halo 行的邻域不完整，算出的掩码丢弃。

    Args:
        dist_fc: `DistributedFlatFaceGeometry`（或同结构对象）
        n_total_cells: `partition.n_total_cells`，用于校验 `perm` 长度
        get_halo: **零参**可调用，返回 halo 交换对象（它的 `.exchange(U_3d)`
            给出 `(n_total, n_sps, n_vars)` 原生排列的扩展场；连续化由
            本函数负责）。之所以是"取值器"而不是对象本身：多 GPU 的
            `gpu_halo` 在滤波初始化时**还是 None**，构造顺序决定了它必须
            到首次施加滤波时才解析（与 `get_provider` 同一个原因）。
        get_provider: 零参可调用，返回 `boundary_ghost_provider`（惰性：
            多 GPU 的滤波初始化在 provider 构造之前）
        freestream: `solver.freestream`
        to_device: 可选，把 numpy 数组搬到计算设备的可调用
        ascontiguous: 可选，按数组自身模块分派的 `ascontiguousarray`
            （GPU 后端传 `core/gpu/device_context.py::ascontiguous_like`）；
            默认 `np.ascontiguousarray`

    Returns:
        可直接展开给 `build_sensor_gated_filter_func_arrays` 的 dict。

    Raises:
        RuntimeError: `perm` 缺失/长度不符，或 `(neighbor_cell_local < 0)`
            与 `is_boundary` 不重合 —— 两者都会让 BJ 包络读到**错误的
            单元**，而那种错误在残差日志里完全看不出来，所以必须硬失败。
    """
    from autoflowcfd.core.fr_solver.boundary import make_bj_boundary_tables

    if ascontiguous is None:
        ascontiguous = np.ascontiguousarray
    if to_device is None:
        def to_device(a):
            return np.asarray(a)

    perm = getattr(dist_fc, "perm", None)
    if perm is None or np.asarray(perm).size == 0:
        raise RuntimeError(
            "sensor+bounds 需要 dist_flat_face.perm 做紧凑->原生索引换算，"
            "当前分布式面几何没有它")
    perm = np.asarray(perm)
    n_total = int(n_total_cells)
    if perm.size != n_total:
        raise RuntimeError(
            f"dist_flat_face.perm 长度 {perm.size} 与 n_total_cells "
            f"{n_total} 不符，索引换算不可靠")

    oc = np.asarray(dist_fc.owner_cell_local)
    nc = np.asarray(dist_fc.neighbor_cell_local)
    bnd = np.asarray(dist_fc.is_boundary, dtype=bool)
    # 边界面的 neighbor_cell_local 是 -1。这两个集合必须严格重合：不重合
    # 意味着"某条边界面带着真实邻居"或"某条内部面没有邻居"，任一情形下
    # BJ 包络都会读到错误的单元。
    if not np.array_equal(nc < 0, bnd):
        n_mismatch = int(np.count_nonzero((nc < 0) != bnd))
        raise RuntimeError(
            f"分布式面几何自洽性失败：{n_mismatch} 条面的 "
            f"(neighbor_cell_local < 0) 与 is_boundary 不一致。"
            f"BJ 越界判据靠 is_boundary 排除没有邻居单元的面。")

    # 紧凑 -> 原生。边界面的 -1 先填 0（占位），它们被 is_boundary 排除，
    # 不会被读到。
    owner_native = perm[oc]
    neigh_native = perm[np.where(nc >= 0, nc, 0)]

    def halo_extend(U_local_3d):
        # 无 MPI / 单 rank 时 `exchange` 只是把 local 拷进扩展数组，
        # 于是掩码与单机路径逐位相同。
        halo = get_halo()
        if halo is None:
            raise RuntimeError(
                "sensor+bounds 的分布式掩码需要 halo 交换对象，但施加滤波"
                "时它仍然是 None。分区边界上的 BJ 包络必须读到 halo 单元"
                "的均值，否则同一个算例换 rank 数会得到不同的掩码——那不能"
                "静默发生。")
        return halo.exchange(ascontiguous(U_local_3d))

    # 扩展场每一行是否是棱柱（BJ 判据只在真实槽位上统计，见
    # `build_sensor_gated_filter_func_arrays` 的 `row_is_prism_extended`）。
    # `compact_cell_type` 处在"棱柱在前"的**紧凑**排列，而掩码场处在
    # halo 交换的**原生**排列 —— 用 `perm` 换回去（`array_native[perm]
    # == array_permuted`，所以 `native[perm[k]] = permuted[k]`）。
    cct = getattr(dist_fc, "compact_cell_type", None)
    if cct is None:
        raise RuntimeError(
            "sensor+bounds 需要 dist_flat_face.compact_cell_type 才能知道"
            "扩展场每一行有多少真实自由度槽位（原生基的零填充槽位冻结在"
            "初值，不排除它们就是在馊值上统计单元极值）")
    from autoflowcfd.core.mpi.distributed_flat_face import native_cell_is_prism

    row_is_prism_native = native_cell_is_prism(dist_fc)

    # `true_normal` 是单位外法向、与 dist_fc 同一（local 面）索引空间，
    # 逐通量点形状由 `make_bj_boundary_tables` 归约成逐面。
    nrm = getattr(dist_fc, "true_normal", None)
    return dict(owner_cell=to_device(owner_native),
                neighbor_cell=to_device(neigh_native),
                is_boundary=to_device(bnd),
                freestream=freestream,
                halo_extend=halo_extend,
                row_is_prism_extended=to_device(row_is_prism_native),
                bnd_tables=make_bj_boundary_tables(
                    get_provider, int(bnd.size),
                    None if nrm is None else np.asarray(nrm),
                    to_device=to_device))


_DIST_FACE_STENCIL_WARNED = [False]


def _warn_distributed_face_stencil(sensor: str) -> None:
    """分布式路径用 BJ 判据时提示"顶点模板尚不可用"，只提示一次。

    一次性：这个函数在每次构造滤波回调时被调用（Order Continuation
    换阶数会重建），逐次刷屏会把真正的日志淹掉。
    """
    if sensor not in ("bounds", "both") or _DIST_FACE_STENCIL_WARNED[0]:
        return
    _DIST_FACE_STENCIL_WARNED[0] = True
    logger.warning(
        "分布式路径的 BJ 越界判据仍用**面邻居**模板：顶点邻域模板需要"
        "按顶点的归约交换（现有 halo 是按单元的 1 层面邻居），尚未实现。"
        "面模板在欠解析光滑场上过度标记（实测标记比例在三档网格加密上"
        "恒为 100%、不收敛；顶点模板是 100%->25%->6.18%），所以分布式上"
        "这个门控会比单机保守得多。需要精确门控请用单机路径，或用 "
        "AFCFD_TROUBLED_SENSOR=persson / AFCFD_FILTER_MODE=off（默认值）。"
        "详见 core/fr_operators/vertex_stencil.py 模块文档。")
