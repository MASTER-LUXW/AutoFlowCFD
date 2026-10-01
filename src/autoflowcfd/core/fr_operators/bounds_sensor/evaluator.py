"""AutoFlowCFD V2.0 - BJ 越界比的后端无关求值器（滤波门控与人工粘性共用）。

`compute_bounds_violation_ratio` 只吃"已经准备好的"参数；真正调用它之前还有
四件与消费方无关、却容易各写一份的事：

1. 真实槽位数：原生四面体 / 棱柱在统一宽度 `(order+1)^3` 的 SP 轴上有零填充
   槽位，统计极值时必须跳过（那些槽位冻结在初值，会让包络读到假值）；
2. 行类型：两类单元真实槽位数不同时，判据要知道每一行是哪一类；分布式下掩码
   在扩展场（本地 + halo）上算，halo 行也要有类型；
3. 边界表：两张表（Dirichlet 值与镜像法向）是惰性的，取值器统一成一个函数；
4. halo 扩展：分区边界单元的包络要读 halo 单元均值，扩展场多出来的 halo 行
   邻域不完整，结果只取前 `n_cells` 项。

这四件事此前写在模态滤波门控里（`fr_solver/filter/sensor_gate.py`），
2026-10-01 收拢到这里，让门控只表达"按越界比施加滤波"这一件事。
"""

from typing import Callable

import numpy as np

from autoflowcfd.core.utils.array_module import array_module as _array_module

from .mask import compute_bounds_violation_ratio


def make_bounds_ratio_evaluator(
    xp, n_cells: int, n_sps: int, order: int, cell_is_prism, *,
    owner_cell, neighbor_cell, is_boundary, ref_scales,
    bnd_tables=None, vertex_stencil=None, halo_extend=None,
    row_is_prism_extended=None,
) -> Callable[[np.ndarray], np.ndarray]:
    """返回 `ratio_fn(U) -> (n_cells,)`：5 个守恒变量上的 BJ 越界比（逐变量取最大）。

    Args:
        xp: 数组模块（NumPy / CuPy），由调用方决定（与其滤波矩阵或解同一模块）
        n_cells: 本地单元数（结果长度）
        n_sps: SP 轴宽度
        order: 当前阶数
        cell_is_prism: (n_cells,) 布尔，True = 棱柱（单机"棱柱在前"与分布式
            交错排列都用它表达）
        owner_cell, neighbor_cell, is_boundary: 面连接（扩展场索引空间）
        ref_scales: 5 个守恒变量的来流参考量级（绝对地板，见 `constants.py`）
        bnd_tables: `(bnd_dirichlet, bnd_mirror_normal)` 二元组、返回它的无参
            函数，或 None
        vertex_stencil: 顶点邻域模板（见 `fr_operators/vertex_stencil.py`）
        halo_extend: 分布式下把 (n_cells, n_sps, n_var) 扩成含 halo 的场的函数
        row_is_prism_extended: 扩展场每一行的类型；两类单元真实槽位数不同且给了
            `halo_extend` 时必须提供
    """
    from autoflowcfd.fr.native_padding import real_sps_per_cell

    cip = xp.asarray(cell_is_prism).astype(bool)
    # 零填充布局**只在** SP 轴等于全局统一宽度 `(order+1)^3` 时存在 —— 这是
    # 构造上的事实，不是兜底：宽度不等于它的数组（例如只关心判据逻辑的合成
    # 布局）根本没有填充槽位可言。
    if n_sps == (order + 1) ** 3:
        n_real_prism, n_real_tet = real_sps_per_cell(order)
    else:
        n_real_prism = n_real_tet = n_sps

    if n_real_prism == n_real_tet:
        # 两类单元真实槽位数相同 -> 行类型与统计无关，不需要行掩码
        row_is_prism = None
    elif row_is_prism_extended is not None:
        row_is_prism = xp.asarray(row_is_prism_extended, dtype=bool)
    elif halo_extend is None:
        # 单机：判据场的行数就是 n_cells，`cip` 正好覆盖
        row_is_prism = cip
    else:
        # 分布式且调用方没给扩展版 -> 硬失败。静默退回"全槽位统计"会让 halo
        # 行在冻结的填充值上参与包络，而那在日志里完全看不出来。
        raise ValueError(
            "给了 halo_extend（分布式判据在扩展场上算）却没给 "
            "row_is_prism_extended —— 扩展场的 halo 行也要知道自己有多少"
            "真实槽位，否则 BJ 包络会读到冻结的零填充值。"
            "见 `core/fr_solver/filter/bounds_conn.py::build_distributed_bounds_conn`。")
    n_real_kw = (dict(n_real_prism=n_real_prism, n_real_tet=n_real_tet)
                 if row_is_prism is not None else {})

    resolve_tables: Callable[[], tuple] = (
        bnd_tables if callable(bnd_tables) else (lambda: bnd_tables or (None, None)))

    def ratio_fn(U: np.ndarray) -> np.ndarray:
        U3 = U.reshape(n_cells, n_sps, -1)
        field = U3 if halo_extend is None else halo_extend(U3)
        bd, bmn = resolve_tables()
        return compute_bounds_violation_ratio(
            xp.ascontiguousarray(field[:, :, :5]),
            owner_cell, neighbor_cell, is_boundary,
            ref_scales=ref_scales, bnd_dirichlet=bd, bnd_mirror_normal=bmn,
            row_is_prism=row_is_prism, vertex_stencil=vertex_stencil,
            **n_real_kw)[:n_cells]

    return ratio_fn

