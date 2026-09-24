"""AutoFlowCFD V2.0 - 传感器门控滤波回调的构造。

从 `core/fr_solver/filter.py` 拆出（2026-09-24）。纯搬家，逻辑未改。

`sensor` 档的核心：只对被传感器标记为"欠分辨"的单元施加滤波，其余单元
原样通过。数组版（`..._arrays`）供 GPU/分布式路径直接消费展平数组，
对象版（`build_sensor_gated_filter_func`）是单机 CPU 的薄包装。
"""

from typing import Callable

import numpy as np

from autoflowcfd.core.utils.array_module import array_module as _array_module

from .apply import _filter_leading_vars_inplace_kernel


def build_sensor_gated_filter_func_arrays(
    n_cells: int, n_sps: int, order: int, filter_prism, filter_tet,
    *, n_prism=None, cell_is_prism=None,
    sensor: str = "persson",
    owner_cell=None, neighbor_cell=None, is_boundary=None,
    freestream=None, halo_extend=None, bnd_tables=None,
    row_is_prism_extended=None, vertex_stencil=None,
) -> Callable[[np.ndarray], np.ndarray]:
    """传感器门控模态滤波的**后端无关**实现（只吃数组，不吃 solver）。

    这是 `build_sensor_gated_filter_func` 的内核。拆出来的理由见
    `fr_operators/artificial_viscosity.py::compute_troubled_cell_mask`：
    第一版门控靠"把当前 stage 的解临时塞进 `solver.state.U` 再调用
    solver 版传感器"实现，那是个副作用 hack，而且让门控只能用在单机
    CPU 推进循环里。现在传感器是纯数组接口，CPU MPI 那条路径（local
    排列、棱柱/四面体交错，用 `cell_is_prism`）可以用同一个内核。

    门控判据：对**守恒密度**求 Persson-Peraire 指示器。守恒密度与原始
    密度只差一个整体缩放，而 S_e 是能量比值、对缩放不变，所以两者给出
    同一个判据——不需要先反算原始变量。

    施加方式：欠分辨的单元走完整的滤波矩阵（该单元确实有需要压制的
    混叠内容），其余单元**完全不动**（保留全部已解析的多项式内容）。
    本函数不改变滤波矩阵本身（仍用传入的 filter_prism/filter_tet），
    只改变"对哪些单元施加"——`AFCFD_FILTER_MODE=off/mild` 那两档改的是
    矩阵，两个维度可以独立组合，便于受控 A/B。

    Args:
        n_cells, n_sps: 单元数与每单元解点数
        order: 当前多项式阶数（order==0 时传感器恒不触发，等价于不滤波）
        filter_prism, filter_tet: 滤波矩阵
        n_prism / cell_is_prism: 单元类型划分，恰好给一个，语义同
            `compute_troubled_cell_mask`
        sensor: 门控判据（`AFCFD_TROUBLED_SENSOR`，见
            `fr_operators/bounds_sensor.py::resolve_troubled_sensor`）：

              persson  Persson-Peraire 模态能量指示器（默认，既有行为）
              bounds   邻居极值越界（BJ 型）
              both     两者取并集

            **为什么需要第二个判据**：Persson-Peraire 的 `s0 =
            -4*log10(order)` 在 order=1 时为 0，门限退化成"顶模态能量
            占比 >= 10%"，而 P1 的顶模态就是全部非常数模态——它在生产
            阶数 P1 上原理上不适用（实测 A/B 前 51 步逐字符相同）。
            BJ 型判据不依赖模态分解，没有这个退化。完整推导与真实网格
            实测见 `fr_operators/bounds_sensor.py` 模块文档。
        owner_cell / neighbor_cell / is_boundary: 面连接数组，`sensor`
            含 "bounds" 时**必须**给出（BJ 判据要邻居均值）。索引空间
            必须与**掩码场的行数**一致：不给 `halo_extend` 时就是
            `n_cells`（单机、GPU 单机）；给了 `halo_extend` 时是它返回的
            `n_total = n_cells + n_halo`（CPU MPI / 多 GPU）。
        bnd_tables: BJ 判据的两张边界表 `(bnd_dirichlet, bnd_mirror_normal)`，
            **或**一个返回该二元组的零参可调用（惰性求值）。四条后端统一
            用 `fr_solver/boundary.py::make_bj_boundary_tables` 构造它 ——
            惰性的理由、两张表各自的含义、以及"不给它贴壁单元会被结构性
            误判、壁面剪应力被压掉 14 倍"的实测，全部见那边与
            `bounds_sensor.compute_bounds_violation_mask` 的同名参数。
        vertex_stencil: 可选 `fr_operators/vertex_stencil.VertexStencil`
            —— 给了它 BJ 的邻域包络就**额外**计入顶点邻居（共享任一顶点
            的全部单元）的均值。

            **必须给**（生产路径）：面邻居在三维四面体上只有 4 个，其
            单元均值不能把本单元夹住，于是 O(h|grad u|) 的合法光滑变化
            被当成越界 —— 实测在欠解析光滑场（TGV 解析初场）上标记
            **100%** 的单元，换顶点模板后降到 **6.18%**、中位越界降到
            恰好 0。完整数据（含"经典 TVB 的 M h^2 修不了它"那条被测量
            否掉的方案）见 `fr_operators/vertex_stencil.py` 模块文档。

            `None` 时只用面邻居（2026-09-19 之前的行为），供没有单元-
            顶点连接的合成单元测试网格与历史对照使用。
        row_is_prism_extended: 可选 (n_total,) 布尔 —— **扩展场每一行**
            是否是棱柱单元，供 BJ 判据只在**真实**自由度槽位上统计单元
            极值/均值。不给时按 `cell_is_prism`/`n_prism` 推（只在没有
            `halo_extend` 时够用）。

            为什么必需：原生基的零填充槽位残差恒为零、解**冻结在初值**，
            而真实槽位在演化 —— 实测 TGV（P2 四面体）30 步后填充值已越出
            真实槽位区间 14% 区间宽。不排除它们就是在冻结的馊值上统计。
            完整说明见 `bounds_sensor.compute_bounds_violation_mask` 的
            `row_is_prism` 参数文档。
        halo_extend: 可选回调 `(n_cells, n_sps, n_vars) -> (n_total,
            n_sps, n_vars)`，把本 rank 的场扩展成含 halo 的场。**分区
            边界上的 BJ 包络必须读到 halo 单元的均值**，否则同一个算例
            换 rank 数会得到不同的掩码——那是求解器不可接受的（结果依赖
            分区）。把 halo 面当边界面排除也不行：分区边界并不是物理
            边界，排除它等于在任意位置人为切断包络。

            为什么是回调而不是让调用方直接传扩展场：滤波回调每个 RK
            stage 被调用一次，扩展必须用**当前 stage** 的解做（用上一次
            残差求值缓存的扩展场会比单机路径滞后半个 stage，两条后端就
            不再逐位可比）。`core/mpi/distributed_solver.py` 传的就是
            `halo_exchange.exchange` 的薄封装——与
            `_compute_distributed_local_time_step` 同一个既有模式
            （局部 dt 的谱半径同样需要额外一次 halo 交换）。

            只影响**掩码**的计算域：滤波矩阵仍然只施加在 `[0, n_cells)`
            的 local 单元上，halo 行算出来的掩码被丢弃（它们的邻域不
            完整）。
        freestream: `solver.freestream` 字典，`sensor` 含 "bounds" 时
            **必须**给出——BJ 判据的绝对地板要用来流参考量级
            （见 `bounds_sensor.py` 模块文档"第一版用全场 RMS 做尺度
            为什么不行"一节：用全场 RMS 时实测标记了 11.08% 的单元）。
    """
    from autoflowcfd.core.fr_operators.artificial_viscosity import (
        compute_troubled_cell_mask,
    )
    from autoflowcfd.core.fr_operators.bounds_sensor import (
        compute_bounds_violation_mask,
    )
    from autoflowcfd.core.fr_solver.residual_diagnostics import (
        _reference_scales,
    )
    if (n_prism is None) == (cell_is_prism is None):
        raise ValueError("n_prism 与 cell_is_prism 必须且只能给一个")
    sensor = str(sensor).lower()
    if sensor not in ("persson", "bounds", "both"):
        raise ValueError(f"未知 sensor: {sensor!r}")
    need_conn = sensor in ("bounds", "both")
    if need_conn and freestream is None:
        raise ValueError(
            f"sensor={sensor!r} 需要 freestream（BJ 判据的绝对地板用来流"
            f"参考量级），不能静默退回用全场 RMS——实测那会标记 11% 的单元"
        )
    if halo_extend is not None and not need_conn:
        # 不静默忽略：给了 halo_extend 说明调用方以为门控要跨分区，
        # 而 persson 档是纯单元局部的、根本不会用到它——静默忽略会
        # 让"我已经接线了分布式"这个错误认知留在调用方。
        raise ValueError(
            f"halo_extend 只对含 bounds 的 sensor 有意义（Persson-Peraire "
            f"是纯单元局部判据，不读邻居），当前 sensor={sensor!r}"
        )
    if need_conn and (owner_cell is None or neighbor_cell is None
                      or is_boundary is None):
        # 不静默退回 persson：那会让"我明明开了 bounds 档"与实际行为
        # 不一致，而这种不一致在日志里完全看不出来。
        raise ValueError(
            f"sensor={sensor!r} 需要 owner_cell/neighbor_cell/is_boundary "
            f"三个面连接数组，缺失的不能静默忽略"
        )
    # 数组模块由**滤波矩阵**决定：GPU 调用方传的是 CuPy 矩阵，于是整个
    # 回调走 CuPy 的同名函数；CPU 调用方传 NumPy，走原来那条路。判据内核
    # （Persson / BJ）本身已经是数组模块无关的，所以四条后端共用这一份
    # 门控实现，而不是各抄一份（本项目的重复实现历来只改一份，见项目
    # 记忆 `feedback-prefer-deleting-redundant-code`）。
    xp = _array_module(filter_prism, filter_tet)
    if cell_is_prism is not None:
        cip = xp.asarray(cell_is_prism).astype(bool)
        # 掩码按单元类型分派时还要用它本身（Persson 的 tet/prism 两条
        # 分支），保持与 xp 一致。
        cell_is_prism = cip
    else:
        if not 0 <= n_prism <= n_cells:
            # 不静默钳：`U[:n_prism]` 这种切片在 n_prism > n_cells 时会
            # 被静默钳到 n_cells，后果是**四面体也被施加棱柱矩阵**而
            # 不报任何错（多 GPU 路径 2026-09-18 真实踩到：传的是全局
            # mesh.n_prism_cells，而数组只有 n_local 行）。
            raise ValueError(
                f"n_prism={n_prism} 超出 [0, n_cells={n_cells}]。"
                f"分布式 local 排列里棱柱与四面体交错、且棱柱数是本 rank "
                f"的局部量，必须用 cell_is_prism 而不是 n_prism。")
        cip = xp.arange(n_cells) < n_prism
    prism_idx_all = xp.flatnonzero(cip)
    tet_idx_all = xp.flatnonzero(~cip)

    # 真实自由度槽位（见 `row_is_prism_extended` 文档）。两类单元各自的
    # 真实槽位数来自 `real_sps_per_cell` —— "哪些槽位是真的"的唯一判据
    # 来源，不在这里重算公式。
    from autoflowcfd.fr.native_padding import real_sps_per_cell

    # 零填充布局**只在** SP 轴等于全局统一宽度 `(order+1)^3` 时存在 ——
    # 这是构造上的事实，不是兜底：宽度不等于它的数组（例如只关心门控
    # 逻辑的合成布局）根本没有填充槽位可言。
    if n_sps == (order + 1) ** 3:
        _n_real_prism, _n_real_tet = real_sps_per_cell(order)
    else:
        _n_real_prism = _n_real_tet = n_sps

    if _n_real_prism == _n_real_tet:
        # 两类单元真实槽位数相同 -> 行类型与统计无关，不需要行掩码
        # （纯坍缩棱柱网格、或上面那种合成布局）。
        _row_is_prism = None
    elif row_is_prism_extended is not None:
        _row_is_prism = xp.asarray(row_is_prism_extended, dtype=bool)
    elif halo_extend is None:
        # 单机：掩码场的行数就是 n_cells，`cip` 正好覆盖
        _row_is_prism = cip
    else:
        # 分布式且调用方没给扩展版 -> 硬失败。静默退回"全槽位统计"会让
        # halo 行在冻结的填充值上参与包络，而那在日志里完全看不出来。
        raise ValueError(
            "给了 halo_extend（分布式掩码在扩展场上算）却没给 "
            "row_is_prism_extended —— 扩展场的 halo 行也要知道自己有多少"
            "真实槽位，否则 BJ 包络会读到冻结的零填充值。"
            "见 `core/fr_solver/filter.py::build_distributed_bounds_conn`。")

    # 惰性求值与缓存都在 `make_bj_boundary_tables` 里（见那边文档）；
    # 这里只在"直接传了二元组"时补一个同形状的取值器，让下游只有一条
    # 取值路径。
    _resolve_bnd_tables = (bnd_tables if callable(bnd_tables)
                           else (lambda: bnd_tables or (None, None)))

    def filter_func(U_flat: np.ndarray) -> np.ndarray:
        U = U_flat.reshape(n_cells, n_sps, -1)
        # Persson-Peraire 只探**守恒密度**（S_e 是能量比值、对整体缩放
        # 不变，所以守恒密度与原始密度给出同一个判据）。BJ 判据则探
        # 全部 5 个守恒变量并取并集：2026-09-16 的真实 checkpoint 实测
        # 里越界量最大的是横向动量与压力，只探密度会漏掉它们
        # （同一类问题：人工粘性的 DEFAULT_SENSOR_VAR_INDEX = 0 只探
        # 密度，而 P2 的失效模态在能量上）。
        troubled = xp.zeros(n_cells, dtype=bool)
        if sensor in ("persson", "both"):
            troubled |= compute_troubled_cell_mask(
                xp.ascontiguousarray(U[:, :, 0]), order,
                n_prism=n_prism, cell_is_prism=cell_is_prism)
        if sensor in ("bounds", "both"):
            # 分区边界上的 BJ 包络要读 halo 单元的均值，所以掩码在
            # **扩展场**上算（见 halo_extend 文档）；扩展场多出来的
            # halo 行邻域不完整，算出的掩码丢弃，只取前 n_cells 项。
            field = U if halo_extend is None else halo_extend(U)
            bd, bmn = _resolve_bnd_tables()
            troubled |= compute_bounds_violation_mask(
                xp.ascontiguousarray(field[:, :, :5]),
                owner_cell, neighbor_cell, is_boundary,
                ref_scales=_reference_scales(freestream, 5),
                bnd_dirichlet=bd, bnd_mirror_normal=bmn,
                row_is_prism=_row_is_prism,
                n_real_prism=_n_real_prism if _row_is_prism is not None else None,
                n_real_tet=_n_real_tet if _row_is_prism is not None else None,
                vertex_stencil=vertex_stencil,
                )[:n_cells]
        if not bool(xp.any(troubled)):
            return U_flat
        if xp is np:
            # CPU：只在被标记的单元上做，走 numba prange kernel。
            # **不能**换成 einsum——那正是 2026-09-13 剖析掉的热点
            # （79 万单元 P1 每次调用约 0.55s，每步 3 次），见
            # `_filter_leading_vars_inplace_kernel` 文档。
            for sel_all, mat in ((prism_idx_all, filter_prism),
                                 (tet_idx_all, filter_tet)):
                sel = sel_all[troubled[sel_all]]
                if sel.size == 0:
                    continue
                sub = np.ascontiguousarray(U[sel])
                _filter_leading_vars_inplace_kernel(
                    sub, np.ascontiguousarray(mat), 5)
                U[sel] = sub
        else:
            # GPU：「两个矩阵都对全场算一遍、再按掩码选」而不是花式索引。
            # 理由与 `gpu_modal_filter.py::filter_scalar_field_gated_gpu`
            # 同一条实测结论：设备上布尔/整数索引要触发额外的
            # gather/scatter 与同步，而滤波矩阵是 (n_sps,n_sps) 的小矩阵
            # （n_sps<=64）、einsum 落到 cuBLAS 批量 gemm，多算一遍的
            # 成本远低于索引开销。数值上完全等价：被选中的单元取滤波
            # 结果、未选中的取原值。
            # 与 CPU 的差异只在求和顺序（einsum vs 顺序循环），因此两条
            # 后端是机器精度一致而非逐位一致——与本项目其它 CPU/GPU
            # 交叉验证同一口径（见 test_gpu_*_crosscheck.py）。
            lead = U[:, :, :5]
            filt = xp.where(cip[:, None, None],
                            xp.einsum("sj,cjv->csv", filter_prism, lead),
                            xp.einsum("sj,cjv->csv", filter_tet, lead))
            U[:, :, :5] = xp.where(troubled[:, None, None], filt, lead)
        return U.reshape(U_flat.shape)

    return filter_func


def build_sensor_gated_filter_func(solver) -> Callable[[np.ndarray], np.ndarray]:
    """按 Persson-Peraire 传感器**逐单元门控**的模态滤波（2026-09-15）。

    为什么需要它：全局每-stage 施加的滤波器在本项目的工作阶数上会精确
    抹掉一整阶（order=1 -> 只剩常数，order=2 -> 只剩双线性，见
    `fr/modal_filter.py` 顶部的实测记录）。而且因为它每个 RK stage 都施加，
    任何 sigma<1 都会随步数复合累积，单纯调小 alpha 只是把清零推迟——
    正确的方向是**只在确实需要的单元上施加**。

    本函数只是 `build_sensor_gated_filter_func_arrays` 的单机适配层
    （"棱柱在前"排列，order 取 `current_order`/`order`），全部说明见
    那边与 `fr_operators/artificial_viscosity.py::compute_troubled_cell_mask`。

    **重要范围说明**：本函数（以及 `AFCFD_FILTER_MODE=sensor`）只门控
    **平均流**滤波。k/omega 走的是 `filter_scalar_field`，由
    `fr_solver/turbulence.py` 单独调用、有**独立**的门控开关，理由见
    `filter_scalar_field` 文档"为什么 k/omega 的门控必须独立判定"一节。
    """
    from autoflowcfd.core.fr_operators.bounds_sensor import (
        resolve_troubled_sensor,
    )

    mesh = solver.mesh
    ops = solver.ops
    order = getattr(solver, "current_order", None)
    if order is None:
        order = solver.order
    sensor = resolve_troubled_sensor()
    conn = {}
    if sensor in ("bounds", "both"):
        fc = mesh.face_connectivity
        if fc is None:
            raise RuntimeError(
                "AFCFD_TROUBLED_SENSOR=bounds/both 需要 mesh.face_connectivity"
                "（BJ 判据要面邻居均值），当前网格没有构建面连接"
            )
        from autoflowcfd.core.fr_solver.boundary import (
            make_bj_boundary_tables,
        )
        _n_faces = int(np.asarray(fc.owner_cell).size)
        # `fc.normal` 是单位外法向，与 owner_cell 同一索引空间 —— 对称面
        # 与滑移壁的镜像包络贡献要用它，见 make_bj_boundary_tables。
        nrm = getattr(fc, "normal", None)
        conn = dict(owner_cell=np.asarray(fc.owner_cell),
                    neighbor_cell=np.asarray(fc.neighbor_cell),
                    is_boundary=np.asarray(fc.is_boundary, dtype=bool),
                    freestream=solver.freestream,
                    bnd_tables=make_bj_boundary_tables(
                        lambda: getattr(solver, "boundary_ghost_provider",
                                        None),
                        _n_faces,
                        None if nrm is None else np.asarray(nrm)))
    # 顶点邻域模板（BJ 判据用；面邻居在三维四面体上不能把本单元夹住，
    # 实测在欠解析光滑场上标记 100%，见
    # `fr_operators/vertex_stencil.py` 模块文档）。只在真的要用 bounds
    # 判据时才建 —— 它要遍历一遍单元-顶点连接，persson 档不需要。
    vstencil = None
    if sensor in ("bounds", "both"):
        from autoflowcfd.core.fr_operators.vertex_stencil import (
            build_vertex_stencil,
        )

        vstencil = build_vertex_stencil(mesh)
    return build_sensor_gated_filter_func_arrays(
        mesh.n_cells, mesh.n_sps_per_cell, int(order),
        ops.filter_prism, ops.filter_tet, n_prism=mesh.n_prism_cells,
        sensor=sensor, vertex_stencil=vstencil, **conn)
