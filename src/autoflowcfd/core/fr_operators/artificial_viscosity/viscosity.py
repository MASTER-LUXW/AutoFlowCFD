"""AutoFlowCFD V2.0 - Persson-Peraire 斜坡映射与 troubled-cell 掩码（滤波门控用）。

从 `artificial_viscosity.py` 拆出（2026-09-20）。人工粘性系数场 2026-10-01 起改由
熵残差判据给出（`entropy_viscosity.py`），本模块只剩滤波门控用的 Persson-Peraire
判据。
"""

from typing import Optional

import numpy as np

from autoflowcfd.core.utils.array_module import array_module as _array_module

from .sensor import (
    compute_persson_peraire_sensor_native_prism,
    compute_persson_peraire_sensor_native_tet,
)
from .sensor_operators import DEFAULT_SENSOR_VAR_INDEX, SENSOR_KAPPA


def compute_artificial_viscosity_ramp(s_e: np.ndarray, order: int, kappa: float = SENSOR_KAPPA) -> np.ndarray:
    """把传感器值 s_e 映射到 [0,1] 的人工粘性强度斜坡（mirgecom 公式）：

        epsilon = 0                                          if s_e < s0-kappa
        epsilon = 0.5*(1+sin(pi*(s_e-s0)/(2*kappa)))         if s0-kappa <= s_e <= s0+kappa
        epsilon = 1                                          if s_e > s0+kappa

    s0 = -4*log10(order)（光滑函数模态系数按 1/N^4 衰减的理论预期，
    Persson & Peraire 原始论文的判据），order<=0 时无意义（调用方保证
    不会以 order==0 调用本函数，见 compute_persson_peraire_sensor 的
    order==0 短路：s_e=-inf 恒小于任意有限 s0-kappa，ramp 天然为 0，
    这里不需要重复特判）。
    """
    xp = _array_module(s_e)
    s0 = -4.0 * np.log10(max(order, 1))     # 标量，与数组模块无关
    ramp = xp.zeros_like(s_e)
    mid = (s_e >= s0 - kappa) & (s_e <= s0 + kappa)
    high = s_e > s0 + kappa
    ramp[mid] = 0.5 * (1.0 + xp.sin(np.pi * (s_e[mid] - s0) / (2.0 * kappa)))
    ramp[high] = 1.0
    return ramp


def compute_troubled_cell_mask(
    field_nodal: np.ndarray, order: int, *,
    n_prism: Optional[int] = None,
    cell_is_prism: Optional[np.ndarray] = None,
    kappa: float = SENSOR_KAPPA,
) -> np.ndarray:
    """逐单元判定"该单元的这个标量场欠分辨" —— **纯数组接口**。

    Persson-Peraire 传感器 + `compute_artificial_viscosity_ramp` 判据，只返回
    布尔掩码，**不需要 solver 也不需要 ops**——只要 (场, order, 单元类型划分)。

    为什么需要这个接口（2026-09-15）：模态滤波器的传感器门控
    （`core/fr_solver/filter.py`）此前只能靠"把场临时塞进
    `solver.state.U` 再调用 solver 版传感器"来实现，那既是一个副作用
    hack，也让门控**只能在单机 CPU 推进循环里用**（CPU MPI / GPU 那些
    路径没有同构的 solver 对象）。改成纯数组接口后：

    - 平均流可以直接传守恒密度（Persson-Peraire 的 S_e 是能量比值、
      对场的整体缩放不变，所以守恒密度与原始密度给出同一个判据）；
    - **k/omega 同样可以用**——这是关键，`filter_scalar_field` 一直
      直接用 `ops.filter_prism`、完全不经过任何门控，于是湍流场在
      legacy/sensor 两档下都被清掉一整阶（即 k/omega 实际是 P0）。
      那是与平均流同一类的静默降阶，只是换了个场。

    两种单元各走自己的原生传感器（只需要 `(field, order)`，坍缩族用过的
    张量积参考点集已随坍缩基删除）。

    Args:
        field_nodal: (n_cells, n_sps) 待探测的标量场
        order: 当前多项式阶数；order==0 时没有可截断的最高阶模态，
            直接返回全 False（与传感器 order==0 短路返回 -inf 一致）
        n_prism: 单机"棱柱在前"排列下的棱柱单元数。与 `cell_is_prism`
            **必须且只能给一个**。
        cell_is_prism: (n_cells,) 布尔数组，True=棱柱。分布式的 local
            排列里棱柱与四面体是**交错**的（见 core/fr_solver/filter.py::
            build_filter_func_by_cell_type 同一处理由），不能用 n_prism
            切片表达，必须走这条。
        kappa: ramp 过渡带宽度，默认与人工粘性同一个 SENSOR_KAPPA

    Returns:
        (n_cells,) 布尔掩码，True = 该单元欠分辨（ramp > 0）
    """
    if (n_prism is None) == (cell_is_prism is None):
        raise ValueError(
            "compute_troubled_cell_mask 需要 n_prism 与 cell_is_prism 中"
            "恰好一个：前者是单机'棱柱在前'排列，后者是分布式交错排列。"
            "同时给或都不给都是调用方对索引空间没有明确认知的信号。")

    xp = _array_module(field_nodal, cell_is_prism)
    n_cells = field_nodal.shape[0]
    mask = xp.zeros(n_cells, dtype=xp.bool_)
    if order == 0 or n_cells == 0:
        return mask

    if n_prism is not None:
        groups = ((xp.arange(0, n_prism), "prism"),
                  (xp.arange(n_prism, n_cells), "tet"))
    else:
        cip = xp.asarray(cell_is_prism).astype(bool)
        if cip.shape != (n_cells,):
            raise ValueError(
                f"cell_is_prism 形状 {cip.shape} 与场的单元数 {n_cells} 不符")
        groups = ((xp.flatnonzero(cip), "prism"),
                  (xp.flatnonzero(~cip), "tet"))

    for sel, cell_type in groups:
        if sel.size == 0:
            continue
        # 四面体走 native 专属传感器（正交归一 PKD 基 + 只用真实自由度），
        # 棱柱走张量积族 + GL 求积权重——两条分支各自精确，理由见
        # `_build_native_tet_sensor_operators`。
        # 两种单元类型都只有原生基一种实现（坍缩四面体 2026-09-03 删除、
        # 坍缩棱柱 2026-09-23 删除），所以这里只按单元类型分派。
        if cell_type == "tet":
            s_e = compute_persson_peraire_sensor_native_tet(
                xp.ascontiguousarray(field_nodal[sel]), order)
        else:
            s_e = compute_persson_peraire_sensor_native_prism(
                xp.ascontiguousarray(field_nodal[sel]), order)
        mask[sel] = compute_artificial_viscosity_ramp(s_e, order, kappa) > 0.0
    return mask
