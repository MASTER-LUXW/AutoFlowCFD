"""AutoFlowCFD V2.0 - 残差量级离群抑制（troubled-cell "机制3"）。

从 `troubled_cell.py`（原 591 行）拆出（2026-09-19，项目"单文件不超
500 行"规范）。两段功能上本来就无关：那边留的是**几何**退化诊断
（scaled Jacobian、面法向失配），这里是**残差**量级的逐 (cell,SP,var)
离群检测与清零。纯搬家，逻辑未改。

机制3 的完整设计背景、它已知的"整单元均匀放大逃过检测"缺口、以及一
条被真实网格数据证伪的修法（全局中位数参照），见
`suppress_residual_outliers` 的函数文档。
"""

import numpy as np
from numba import njit, prange


# 机制3（RESIDUAL_OUTLIER_FACTOR）：真实复现的灾难放大比同单元内正常
# SP 间残差差异高出 7~10 个数量级（3.14e5 vs ~1e-8 量级），任何合理
# 物理场在单个（微小）单元内部的残差变化不会到 1e4 倍这个量级，取值
# 留有充分安全边际，不会误伤真实的局部大梯度。
RESIDUAL_OUTLIER_FACTOR = 1e4
# 场值相对下限：低于"该变量自身场值量级 * 此下限"的残差差异一律视为
# 噪声，不参与异常判定——避免在健康单元（残差普遍已经很小，中位数本身
# 逼近浮点噪声）里把噪声当异常清零。
RESIDUAL_OUTLIER_FIELD_REL_FLOOR = 1e-9


@njit(cache=True, parallel=True)
def _median_abs_over_sps_kernel(residual: np.ndarray) -> np.ndarray:
    """等价于 `np.median(np.abs(residual), axis=1)`，residual 形状
    (n_cells, n_sps, n_vars) -> 返回 (n_cells, n_vars)。

    性能优化：`suppress_residual_outliers` 每次残差求值调用 2 次（无粘+
    粘性各一次），每步 SSP-RK3 又调用 3 次子级，真实生产网格（79万单元）
    P1 阶段单步 6 次调用里，`np.median` 自身（内部落到 `numpy.partition`
    的通用 n 维归约路径）实测占约 2.1s——但每次归约只是在极小的 n_sps
    （P1=8/P2=27）范围内找中位数，被 79 万这个外层 cell 数放大成瓶颈，
    是 numpy 通用分派开销主导、不是算法本身复杂。换成 numba 并行 kernel
    对每个 (cell,var) 独立排序这一小段定长数组直接取中位数，消除通用
    n 维归约的分派开销——已用随机数据在 n_sps∈{1,8,27,64}（覆盖 P0-P3
    的奇偶两种中位数定义：奇数取中间值、偶数取两个中间值平均，与
    np.median 定义完全一致）、真实网格规模上做过逐位对比（最大误差
    0.0，机器精度意义上的恰好相等），79万单元×8SPs×5变量规模下实测
    3 倍提速（0.496s -> 0.166s）。
    """
    n_cells, n_sps, n_vars = residual.shape
    out = np.empty((n_cells, n_vars))
    half = n_sps // 2
    even = (n_sps % 2 == 0)
    for c in prange(n_cells):
        buf = np.empty(n_sps)
        for v in range(n_vars):
            for s in range(n_sps):
                x = residual[c, s, v]
                buf[s] = x if x >= 0.0 else -x
            buf_sorted = np.sort(buf)
            if even:
                out[c, v] = 0.5 * (buf_sorted[half - 1] + buf_sorted[half])
            else:
                out[c, v] = buf_sorted[half]
    return out


@njit(cache=True, parallel=True)
def _outlier_ref_and_flag_kernel(residual, reference_field, factor,
                                 field_rel_floor, n_prism,
                                 n_real_prism, n_real_tet):
    """一趟算出逐 (cell,var) 的异常判据参照量 `ref`，并给出全场是否存在异常值。

    `ref[c,v] = max( median_{s<n_real}(|residual[c,s,v]|),
                     field_rel_floor * mean_{s<n_real}(|reference_field[c,s,v]|),
                     1e-300 )`
    中位数定义与 `np.median` 一致（偶数个取中间两个的平均）。

    ## 只统计**真实**槽位（2026-09-18 修掉的真实生产缺陷）

    原生基的零填充槽位**残差恒为零**（"零填充块对角"不变量刻意保证：
    填充槽位不被时间推进改写）。如果把它们一起排进中位数：

        order   n_sps   真实   零填充   中位数落点
          P1      8       4      4      sorted[3],sorted[4] 均值 = 最小真实值/2 > 0
          P2     27      10     17      sorted[13] 落在零区                    = 0
          P3     64      20     44      sorted[32] 落在零区                    = 0

    P2/P3 上参照量因此塌到 `field_rel_floor * mean|U|` 这个地板（密度约
    1.2e-9），阈值 `1e4 * 1.2e-9 = 1.2e-5`，而真实残差是 1e5 量级 ——
    **整个四面体单元的残差被全部清零、单元完全不演化**。P1 侥幸逃过
    （中位数非零），所以 P1 的生产运行看起来正常、掩盖了这条缺陷。

    真实槽位数来自 `fr/native_padding.py::real_sps_per_cell`（"哪些槽位
    是真的"的唯一判据来源），由调用方取好传进来 —— numba nopython 不能
    调那个函数。

    Args:
        n_prism: 棱柱单元数（"棱柱在前"排列下的分界）
        n_real_prism / n_real_tet: 两段各自的真实槽位数

    Returns:
        (ref, has_outlier)：ref 形状 (n_cells, n_vars)；has_outlier 为
        bool（等价于原实现的 `np.any(outlier)`）。
    """
    n_cells, n_sps, n_vars = residual.shape
    ref = np.empty((n_cells, n_vars))
    flags = np.zeros(n_cells, dtype=np.bool_)
    for c in prange(n_cells):
        n_real = n_real_prism if c < n_prism else n_real_tet
        half = n_real // 2
        even = (n_real % 2 == 0)
        buf = np.empty(n_real)
        local_flag = False
        for v in range(n_vars):
            acc = 0.0
            for s in range(n_real):
                x = residual[c, s, v]
                buf[s] = x if x >= 0.0 else -x
                y = reference_field[c, s, v]
                acc += y if y >= 0.0 else -y
            buf_sorted = np.sort(buf)
            if even:
                med = 0.5 * (buf_sorted[half - 1] + buf_sorted[half])
            else:
                med = buf_sorted[half]
            r = med
            rf = field_rel_floor * (acc / n_real)
            if rf > r:
                r = rf
            if r < 1e-300:
                r = 1e-300
            ref[c, v] = r
            thresh = factor * r
            for s in range(n_real):
                a = buf[s]
                if a > thresh:
                    local_flag = True
        flags[c] = local_flag
    return ref, bool(np.any(flags))


@njit(cache=True, parallel=True)
def _outlier_zero_kernel(residual, ref, factor, out, n_prism,
                         n_real_prism, n_real_tet) -> None:
    """按 `_outlier_ref_and_flag_kernel` 给出的参照量清零异常 (cell,SP,var)。

    只在**真实**槽位上判定（填充槽位残差恒为零、原样拷过去），理由见
    `_outlier_ref_and_flag_kernel` 文档那节。
    """
    n_cells, n_sps, n_vars = residual.shape
    for c in prange(n_cells):
        n_real = n_real_prism if c < n_prism else n_real_tet
        for v in range(n_vars):
            thresh = factor * ref[c, v]
            for s in range(n_real):
                x = residual[c, s, v]
                a = x if x >= 0.0 else -x
                out[c, s, v] = 0.0 if a > thresh else x
            for s in range(n_real, n_sps):
                out[c, s, v] = residual[c, s, v]


def suppress_residual_outliers(
    residual: np.ndarray,
    reference_field: np.ndarray,
    n_prism: int,
    factor: float = RESIDUAL_OUTLIER_FACTOR,
    field_rel_floor: float = RESIDUAL_OUTLIER_FIELD_REL_FLOOR,
) -> np.ndarray:
    """机制3：按 (cell, SP, 变量) 粒度检测残差量级异常并清零，见模块文档
    "机制3"一节。

    Args:
        residual: 已算出的（无粘或粘性）残差，形状 (n_cells,n_sps,n_vars)
        reference_field: 对应的场值（如 Q 或 U），同形状，用于建立与
            该变量自身量级挂钩的绝对下限（质量/动量/能量分量的自然
            量级可以相差好几个数量级，不能共用同一个绝对阈值）
        factor: 相对同单元其余 SP 中位数的放大倍数阈值
        field_rel_floor: 场值量级的相对下限系数

    Returns:
        清零异常 SP 后的残差，形状不变

    2026-08-29 调查记录（尝试过但已放弃的改法，供后续参考）：确认过
    一个真实缺陷——`ref_sibling` 只在*同一个单元内*比较，对"整个单元
    所有 SP 均匀、连续地被放大"这类情形（四面体坍缩坐标各向异性，见
    tet_collapsed_coord_anisotropy 项目记忆）结构性失明：合成验证里，
    单点凸出型异常能被现有判据抓住，但让同一个单元全部 SP 均匀放大到
    1e8（其余单元正常）时，`ref_sibling` 对该单元本身也同步被拖到
    1e8 量级，判据完全放行。

    曾尝试修复：新增 `ref_global`（所有单元 `ref_sibling` 的全局中位数）
    作为不依赖"同单元"的独立参照，与局部判据做"或"关系。这个改法通过
    了合成负控制测试与 tests/validation/test_couette.py / test_tgv.py
    两个稳定性回归测试，但用在真实 cube_demo 网格、从均匀自由流场初场
    起步的第 1 步残差评估时被证伪：真实流场里绝大多数单元深处远场、
    残差天然接近零（自由流场保持性），只有边界附近少数单元有真实的大
    残差（这正是边界条件驱动物理演化所必需的、合法的大梯度）——全局
    中位数被这批"沉默的大多数"远场单元拖到接近机器噪声量级，导致边界
    附近合法的大残差被误判成"全局异常"整片清零：真实复现，RMS 残差从
    3.5e8（原始行为）骤降到 4.28e-4，气动力积分 F_pressure≈1.6e-11、
    Cd=0.000000——不是收敛，是把边界条件驱动的真实物理当异常打掉，
    求解器实质上被冻结在初始均匀流场附近，完全没有真实演化。这暴露了
    "用全网格中心趋势统计量做参照"这个思路本身的结构性缺陷：真实流场
    的残差分布天然、合理地高度不均匀（边界层/尾迹/驻点相对静止远场
    残差大出好几个数量级是物理本身要求的，不是需要抑制的异常），任何
    形式的"全局典型尺度"参照都无法可靠区分"合法的局部强物理"与"真正
    的退化伪影"。已回退到本函数原始实现（只保留同单元内的局部判据）；
    这个"整单元均匀放大逃过检测"的缺陷本身仍然真实存在、未被修复，
    但目前没有已知的安全解法——需要的是一个不依赖任何全网格统计量、
    真正独立于四面体坍缩坐标各向异性污染的参照（项目记忆里提到的
    "真实几何法向的 P0 有限体积残差"是唯一有理论依据但尚未实现、且
    有明显额外计算成本的方向），不在这次调查范围内解决。
    """
    # 性能优化（2026-09-13，真实剖析：本函数每步被调用 8 次——无粘/粘性
    # 残差各 3 次 RK stage + k/omega 输运各 1 次，79 万单元 P1 实测合计
    # 约 2.5s/步）：原实现是 "numba 中位数 kernel + 5~7 趟 numpy 全场
    # 遍历"（np.mean、np.abs、比较、np.any、np.where 各自一趟，每趟读写
    # 253MiB 的 (79万,8,5) 数组，且 numpy 逐元素运算**全部单线程**）。
    # 现在把参照量计算与异常检测合并进一个按 cell prange 的 kernel
    # （`_outlier_ref_and_flag_kernel`），只有真的存在异常值时才走第二个
    # kernel 写出清零后的数组——"无异常值时原样返回同一个数组对象"这条
    # 既有语义完全保留。数学判据逐项对应原实现（同一个中位数定义、同一个
    # `max(median, floor*mean, 1e-300)` 参照、同一个 `|res| > factor*ref`
    # 比较），结果逐位相同（见 tests/unit/test_troubled_cell_*.py）。
    # 真实槽位数（**必须**只在真实槽位上统计，否则原生基的零填充会把
    # 中位数拖到 0、把整个单元的残差判成异常清零 —— 见
    # `_outlier_ref_and_flag_kernel` 文档那节的量级分析）。
    from autoflowcfd.fr.native_padding import (
        order_from_n_sps,
        real_sps_per_cell,
    )

    n_cells, n_sps = residual.shape[0], residual.shape[1]
    if not (0 <= n_prism <= n_cells):
        raise ValueError(
            f"n_prism={n_prism} 超出 [0, n_cells={n_cells}] —— 它是"
            f'"棱柱在前"排列下的分界，越界说明调用方传错了参数，'
            f"而按错的分界统计真实槽位会静默把残差判成异常清零")
    from autoflowcfd.fr.native_prism.mode import prism_basis_is_native

    if n_prism == n_cells and not prism_basis_is_native():
        # 没有四面体、且棱柱走坍缩基 -> 全网格没有任何填充槽位，全部
        # SP 都是真实自由度。这条短路同时让"合成形状"（`n_sps` 不是
        # 某个 `(p+1)^3`，例如只关心归约语义的单元测试）不必先反解阶数。
        n_real_prism = n_real_tet = n_sps
    else:
        order = order_from_n_sps(n_sps)
        n_real_prism, n_real_tet = real_sps_per_cell(order)

    ref, has_outlier = _outlier_ref_and_flag_kernel(
        np.ascontiguousarray(residual), np.ascontiguousarray(reference_field),
        factor, field_rel_floor, n_prism, n_real_prism, n_real_tet,
    )
    if not has_outlier:
        return residual
    out = np.empty_like(residual)
    _outlier_zero_kernel(np.ascontiguousarray(residual), ref, factor, out,
                         n_prism, n_real_prism, n_real_tet)
    return out
