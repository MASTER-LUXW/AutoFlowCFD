"""
AutoFlowCFD V2.0 - GPU 版 Persson-Peraire 欠分辨单元判定（传感器门控用）

存在的理由（2026-09-15 系统性审计的 A 类发现）：`AFCFD_FILTER_TURB_GATE`
这一维此前只在单机 CPU（`fr_solver/turbulence.py`）与 CPU MPI（经同一个
`compute_turbulence_source` 自动获得）上接线过，单 GPU
（`gpu_solver_io.py`）与多 GPU 分布式（`gpu_distributed_init.py`）两条
路径**直接调用 `filter_scalar_field_gpu` 无条件全场滤波**，于是同一个
环境变量在不同后端意味着不同的数值方案，而且没有任何提示。本项目对这
类静默分歧的处理方式见 `fr_solver/filter.py::resolve_filter_mode`——但
那里只能"显式报错"，因为 `sensor` 档的平均流门控需要在 RK stage 内部
逐 stage 求指标；k/omega 这一维不同：它只在每步的湍流源项之后施加一次，
补齐成本很低，所以这里**真正实现**而不是拒绝。

数值上与 CPU 版逐位等价：算子（Vandermonde 逆、最高阶模态掩码、棱柱的
张量积求积权重）直接复用 CPU 侧 `fr_operators/artificial_viscosity.py`
的构造函数与缓存，只把结果上传一次；判据同样是
`compute_artificial_viscosity_ramp(s_e) > 0`，等价于 `s_e > s0 - kappa`
（见那边的实现：`mid` 区间左端点 `s_e == s0-kappa` 处 `sin(-pi/2) = -1`
使 ramp 恰为 0，所以是**严格**大于）。

棱柱与四面体两条分支的内积算法**刻意不同**（棱柱用 GL 张量积权重、
native 四面体用正交归一 PKD 基不需要权重），完整论证见
`artificial_viscosity.py::_build_native_tet_sensor_operators`。
"""

import numpy as np

from autoflowcfd.core.gpu import get_cupy

#: 每个 (order) 一份上传后的算子，键是 order，值见 `_gpu_sensor_operators`
_gpu_sensor_cache = {}


def _gpu_sensor_operators(order: int):
    """上传并缓存该阶数的传感器算子。

    Returns:
        dict，键：
          'prism_V_inv' / 'prism_mass' / 'prism_top' / 'prism_n_native'
          'tet_V_inv' / 'tet_top' / 'tet_n_native'
    """
    if order in _gpu_sensor_cache:
        return _gpu_sensor_cache[order]

    cp = get_cupy()
    # 私有构造函数从它们**真正的**所在子模块导入（人工粘性模块
    # 2026-09-20 拆成子包，包 `__init__` 只 re-export 公开名）。
    from autoflowcfd.core.fr_operators.artificial_viscosity.sensor_operators import (  # noqa: E501
        _build_native_prism_sensor_operators,
        _build_native_tet_sensor_operators,
    )

    # **2026-09-23 修复的真实缺陷**：这里此前调
    # `_build_sensor_operators("prism", order, ref_cube_sps)` —— 那是
    # **坍缩坐标模态族**的算子（`(order+1)^3` 个张量积模态 + 把张量积
    # Gauss 点当解点）。CPU 侧早在 2026-09-20 就迁到了原生棱柱专属算子
    # （见 `_build_native_prism_sensor_operators` 文档"与四面体同一类的
    # 真实缺陷"一节），**GPU 这份拷贝被漏掉了**：对原生基的解等于
    #   (a) 用一个不是解所在空间的基做模态分解；
    #   (b) 把冻结在初值的零填充槽位当自由度喂进指标。
    # 这正是本项目反复出现的"同一语义两份实现、只改了一份"。
    p_V_inv, p_mass, p_top, p_n_native = (
        _build_native_prism_sensor_operators(order))
    t_V_inv, t_top, t_n_native = _build_native_tet_sensor_operators(order)

    ops = {
        'prism_V_inv': cp.asarray(np.ascontiguousarray(p_V_inv)),
        'prism_mass': cp.asarray(np.ascontiguousarray(p_mass)),
        'prism_top': cp.asarray(np.ascontiguousarray(p_top)),
        'prism_n_native': int(p_n_native),
        'tet_V_inv': cp.asarray(np.ascontiguousarray(t_V_inv)),
        'tet_top': cp.asarray(np.ascontiguousarray(t_top)),
        'tet_n_native': int(t_n_native),
    }
    _gpu_sensor_cache[order] = ops
    return ops


def _sensor_prism_gpu(field, ops):
    """原生棱柱分支：PKD(三角形)xLegendre(挤出) 模态能量比，返回 `s_e`。

    与 CPU 侧 `compute_persson_peraire_sensor_native_prism` 逐项对应：

        S_e = sum_{top} M_mm chat_m^2 / sum_all M_mm chat_m^2

    `M_mm` 是**对角质量**——原生棱柱基在参考棱柱上正交但**不归一**
    （P1 实测对角为 8/2.667/4/1.333/5.333/1.778），所以 L2 能量必须带上
    它；不能像四面体那样直接 `sum(chat^2)`（PKD 正交归一），也不能像坍缩
    那样用节点求积权重（那条成立只因为坍缩解点=求积点）。

    只取**前 n_native 列**：其余是冻结在初值的零填充槽位、不是自由度。
    """
    cp = get_cupy()
    n_native = ops['prism_n_native']
    if field.shape[1] < n_native:
        raise ValueError(
            f"field 每单元只有 {field.shape[1]} 个解点，少于 native 棱柱"
            f"所需的 {n_native} 个真实自由度")
    real = cp.ascontiguousarray(field[:, :n_native])
    modal = cp.einsum("ij,cj->ci", ops['prism_V_inv'], real)
    energy = modal * modal * ops['prism_mass'][cp.newaxis, :]
    energy_all = cp.sum(energy, axis=1)
    energy_top = cp.sum(
        cp.where(ops['prism_top'][cp.newaxis, :], energy, 0.0), axis=1)
    S_e = energy_top / cp.maximum(energy_all, 1e-300)
    return cp.log10(cp.maximum(S_e, 1e-300))


def _sensor_native_tet_gpu(field, ops):
    """native 四面体分支：正交归一 PKD 基下的模态能量比，只用真实自由度。"""
    cp = get_cupy()
    n_native = ops['tet_n_native']
    if field.shape[1] < n_native:
        raise ValueError(
            f"field 每单元只有 {field.shape[1]} 个解点，少于 native 四面体"
            f"所需的 {n_native} 个真实自由度")
    real = field[:, :n_native]
    modal = cp.einsum("ij,cj->ci", ops['tet_V_inv'], real)
    energy_all = cp.einsum("ci,ci->c", modal, modal)
    modal_top = cp.where(ops['tet_top'][cp.newaxis, :], modal, 0.0)
    energy_top = cp.einsum("ci,ci->c", modal_top, modal_top)
    S_e = energy_top / cp.maximum(energy_all, 1e-300)
    return cp.log10(cp.maximum(S_e, 1e-300))


def compute_troubled_cell_mask_gpu(field, n_prism: int, order: int,
                                   kappa: float = None):
    """逐单元判定"该单元的这个标量场欠分辨"——GPU 版，"棱柱在前"排列。

    与 CPU 版 `fr_operators/artificial_viscosity.py::
    compute_troubled_cell_mask` 同一套算子与同一条判据。GPU 的两条调用
    路径（单机 GPU 与多 GPU 分布式的 compact 索引空间）都是棱柱在前，
    所以这里不需要 CPU 版那条 `cell_is_prism` 交错分支。

    Args:
        field: (n_cells, n_sps) CuPy 数组
        n_prism: 棱柱单元数（前 n_prism 个）
        order: 当前多项式阶数；order==0 时恒返回全 False
        kappa: ramp 过渡带宽度，None 时用 CPU 侧同一个 `SENSOR_KAPPA`

    Returns:
        (n_cells,) CuPy 布尔数组，True = 欠分辨
    """
    cp = get_cupy()
    from autoflowcfd.core.fr_operators.artificial_viscosity import SENSOR_KAPPA
    if kappa is None:
        kappa = SENSOR_KAPPA
    n_cells = field.shape[0]
    mask = cp.zeros(n_cells, dtype=cp.bool_)
    if order == 0 or n_cells == 0:
        return mask

    ops = _gpu_sensor_operators(order)
    # 判据：ramp > 0 <=> s_e > s0 - kappa（严格），见模块文档。
    s0 = -4.0 * float(np.log10(max(order, 1)))
    thr = s0 - float(kappa)
    if n_prism > 0:
        mask[:n_prism] = _sensor_prism_gpu(
            cp.ascontiguousarray(field[:n_prism]), ops) > thr
    if n_cells > n_prism:
        mask[n_prism:] = _sensor_native_tet_gpu(
            cp.ascontiguousarray(field[n_prism:]), ops) > thr
    return mask


def compute_turb_troubled_mask_gpu(k_field, omega_field, n_prism: int,
                                   order: int):
    """k/omega 门控掩码：对两个场**分别**求指标后取**并集**。

    取并集而不是交集、以及不复用平均流掩码的理由，完整记录在 CPU 版
    `fr_solver/filter.py::compute_turb_troubled_mask` 的文档里（简述：
    2026-09-12 记录的真实发散发生在 omega，更早一轮失控的是 k，任一
    出问题该单元都需要被滤波；而一个单元完全可以密度光滑而 omega 有
    尖峰——实测 off/sensor 两档平均流轨迹几乎逐位相同、om_max 却差一个
    量级，正是这件事的直接证据）。
    """
    mk = compute_troubled_cell_mask_gpu(k_field, n_prism, order)
    mo = compute_troubled_cell_mask_gpu(omega_field, n_prism, order)
    return mk | mo
