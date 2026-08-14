"""
AutoFlowCFD V2.0 - 问题单元（troubled cell）探测与局部限制

背景：坍缩坐标下 P>=2 的四面体/棱柱，单元内不同解点（SP）的 det(J) 天然
可以相差数百倍（即使是完美形状的正四面体也一样，是 Duffy 坍缩变换本身的
固有性质，与单元形状无关）；真实网格棱柱-四面体过渡区的偏斜单元会把
这个比值进一步推向极端（可低至 1e-14）。core/fr_residual_inviscid.py 的
无粘残差公式 residual = -div_comp/det(J) 中，参考空间散度 div_comp 对
*非均匀*流场存在无法随 det(J) 一起缩小的截断/混叠误差；这个误差经黎曼
求解器 owner/neighbor 两侧各自独立取自己度量法向（对*非均匀*流场违反
反对称性 F(A,B,n)=-F(B,A,-n)，产生真实通量不守恒，详见
ProjectFiles/V2.0/5_重大问题修复-Part1.md）进一步放大后，在 det(J) 极小
的单元上会被放大到灾难量级——但这个不守恒的数学根源本身无法在不重新
设计通量构造（entropy-stable/split-form，工业界均未采用，属研究级工作）
的前提下根治。

本模块实现行业实际路线（Fluent/STAR-CCM+/OpenFOAM 等对棱柱-tet 过渡区
退化单元的通用做法）：不消除不守恒本身，把它的影响锁死在局部。

两个独立触发机制（都是"或"关系，任一满足就对该单元做保护）：

1. **体积项混叠**：放大因子是残差公式里的 *原始* det(J)（而不是任何
   归一化/形状相关的量）——真实网格与合成算例都验证过：det(J) 很小但
   *形状本身完全正常*（只是物理尺寸小，如细密边界层网格里的普通单元）
   的单元，一样会出现体积项残差灾难性放大；本模块开发过程中一度尝试
   过把判据换成与物理尺寸无关的"缩放雅可比"形状质量（理由是绝对
   det(J) 在很多网格上因物理尺度普遍偏小而失去判别力），但这个尝试
   被数值验证证伪——合成 Couette 算例里一个形状质量完全正常、只是
   det(J) 天然偏小（坍缩坐标 SP 间 det(J) 比值恒为 1/488，是 Duffy
   变换固有性质，见下）的单元，在改用形状质量判据后残差从已修复的
   ~2e-1 打回未修复的 ~3.9e8，证实体积项混叠的风险只取决于原始
   det(J) 绝对值，不取决于单元形状是否退化。因此本机制的判据就是最
   直接的原始 det(J) 阈值。
2. **面校正项不守恒**：这个机制需要两侧法向*确实*不一致才会触发，与
   机制1相互独立——真实网格实测：两侧法向失配（本单元自己的度量方向
   相对该面平面几何法向 true_normal 的偏离，最大可达 37°）主要出现在
   棱柱四边形侧面（双线性曲面 vs 相邻四面体平面三角形的几何差异），
   这类棱柱的*形状本身*往往完全正常；用"det(J) 小"去筛选机制2会因为
   两个集合基本不相交而让机制2形同虚设（真实网格验证：命中 0 单元），
   因此机制2改用与法向失配独立的判据，只要检测到真实存在的失配就
   触发保护，不要求 det(J) 同时很小。

真实网格数值验证（cube_demo，1% 幅度非均匀扰动）：
- 单个最差单元（cell 509974，det(J)低至 ~2e-14）体积项残差
  从 ~6.3e10 降到 ~2.1e-1（约11个数量级）
- Couette 合成算例端到端初始全局残差从 8.99e8 降到 283（约7个数量级），
  发散步数从 2~4 步推迟到 300+ 步
- 机制1（det(J)<1e-9）命中 545597 单元中的 102858 个（18.85%）；机制2
  （法向失配>1°）命中 6833 个（1.25%），与机制1的交集很小（两者基本
  是不同的单元群体），按"或"合并后面校正冻结共命中约 2.7% 单元。

机制3（症状检测，取代机制1/2 在无粘/粘性残差里"先几何预判、按整个
单元降阶"的角色，机制1/2 本身保留作 log_degenerate_cell_report 的
诊断统计与 mesh 加载期的一次性报告）：

背景（真实复现，2026-08-14 Couette 合成算例定量验证过程中发现）：
1. 机制1 用*绝对* det(J) 阈值，量纲上等价于"物理体积小于某个绝对值就
   危险"，是照着一个特定真实网格（cube_demo，米级尺度）反推标定出来
   的，换一个绝对尺度的网格（比如缩小的合成验证网格）这个标定就不
   成立——网格越细，退化 SP 的 det(J) 绝对值越容易跌破这同一个绝对
   阈值（det(J)~L³，跟网格尺度强相关），跟单元是否"真的危险"其实
   没有必然联系。真实复现：det(J)=8.28e-7（比阈值 1e-9 高出 828 倍，
   机制1判定"安全"）的 SP，无粘残差仍被放大到 3.14e5——阈值以上不
   代表安全，这个绝对判据本身不完备。
2. 机制1/2 按*整个单元*降阶（Q 全部替换成体积平均/梯度全部清零），
   而坍缩坐标的退化只出现在单元内一小部分 SP（通常 27 个 SP 里 1~9
   个），当一个网格的所有单元恰好都在同一绝对尺度、以至于机制1对
   *每个*单元都命中时（合成验证网格常见——真实工业网格因单元尺寸
   跨越多个数量级，机制1只命中一部分，不会有这个问题），会把*整个*
   网格的无粘/粘性物理都拍平成局部零阶，等于关掉了真实物理。

机制3改成直接检测*已经算出的*残差本身的量级异常（症状），而不是用
det(J)/法向失配这类间接几何量事先"预测"危险：同一单元、同一变量下，
用其余 SP 残差的中位数（对最多约一半 SP 同时异常仍稳健）作参照，
外加一个与该变量自身场值量级挂钩的下限（避免在健康、残差已经普遍
接近零的单元里把纯浮点噪声也当异常打掉）——两者同时满足才判定为
异常并清零（等价于用局部常数场假设去顶替这一个 SP 的贡献，与机制1
的"Q 平均化"同一物理语义，只是把干预粒度收紧到单个 SP）。这个判据
不含任何绝对网格尺度假设，任意网格尺度下都成立，且天然只清零真正
异常的那几个 SP，同一单元其余 SP 保留完整 P2 精度与真实物理耦合。
"""

from typing import Dict, Optional

import numpy as np
from loguru import logger

# 机制1（体积项）硬保护阈值：单元内最小*原始* det(J) 低于此值时，Q 场
# 局部降为 P=0。真实 cube_demo 网格实测（自由流场残差分布分析）：残差
# 超标单元与 det(J)<1e-9 完全重合（0 个反例），det(J)>=1e-9 的单元残差
# 全部正常——不是随意选取的安全余量，是直接测得的分界点。这个判据只
# 依赖原始 det(J)，不依赖单元形状是否退化（见模块文档机制1说明）。
TROUBLED_CELL_HARD_DET_JAC = 1e-9

# 机制2（面校正项）硬保护阈值（度）：本单元自己的度量方向相对该面
# true_normal 的偏离角度超过此值时，该面的校正跳跃项置零。真实网格
# 实测最大偏离 37°，主要出现在棱柱四边形侧面。
FACE_MISALIGNMENT_HARD_THRESHOLD_DEG = 1.0
_FACE_MISALIGNMENT_HARD_THRESHOLD = 1.0 - np.cos(np.radians(FACE_MISALIGNMENT_HARD_THRESHOLD_DEG))

# 形状质量预警阈值：纯诊断量（缩放雅可比，与物理尺寸无关），不参与任何
# 保护判据的触发，只用于诊断报告里区分"det(J) 小是因为形状真退化"还是
# "只是物理尺寸小"，帮助判断是否需要改善网格。
SHAPE_QUALITY_WARN_THRESHOLD = 0.1


def compute_scaled_jacobian_quality(jacobians: np.ndarray, det_jacs: np.ndarray) -> np.ndarray:
    """缩放雅可比形状质量（诊断量，不参与保护判据），形状 (n_sps,)（或
    更高维，只要最后一维是 SP）。quality=1 对应完美正交映射，越接近 0
    越退化——与物理尺寸、坍缩坐标 SP 间的固有非均匀性都无关，能真正
    区分"形状退化"与"只是物理尺寸小"（真实网格验证：正常单元
    quality~0.55~0.92，网格生成产生的畸形单元 quality~1.7e-4~1e-2）。

    Args:
        jacobians: 雅可比矩阵，形状 (...,3,3)，jacobians[...,i,m] = dx_i/dxi_m
            （见 curved_mapping.py::tet_exact_jacobian/prism_exact_jacobian
            的返回值约定：列 m 是参考方向 m 的物理切向量）
        det_jacs: 形状 (...,)，与 jacobians 对应的行列式
    """
    col_norms = np.linalg.norm(jacobians, axis=-2)  # (...,3): 每个参考方向切向量的模长
    prod = np.prod(col_norms, axis=-1)
    return det_jacs / np.maximum(prod, 1e-300)


def cell_min_det_jac(det_jacs: np.ndarray) -> np.ndarray:
    """每个单元内的最小 det(J)，形状 (n_cells,)。det_jacs 形状 (n_cells,n_sps)。"""
    return det_jacs.min(axis=1)


def cell_min_shape_quality(scaled_quality: np.ndarray) -> np.ndarray:
    """每个单元内的最小缩放雅可比质量（诊断量），形状 (n_cells,)。"""
    return scaled_quality.min(axis=1)


def troubled_cell_mask(det_jacs: np.ndarray, threshold: float = TROUBLED_CELL_HARD_DET_JAC) -> np.ndarray:
    """机制1判据：单元内最小*原始* det(J) 低于 threshold 的单元掩码，形状 (n_cells,)。"""
    return cell_min_det_jac(det_jacs) < threshold


def limit_troubled_cells(Q: np.ndarray, det_jacs: np.ndarray) -> np.ndarray:
    """机制1缓解：硬保护阈值单元局部降为 P=0（单元内用体积平均值代替
    真实的高阶多项式解）参与无粘通量计算；不改变该单元真实存储的高阶解
    本身（调用方传入的 Q 是残差组装用的临时副本，U 不受影响）。

    对*均匀*流场，本函数是恒等操作（体积平均本来就等于常数本身），不
    影响已验证的自由流场保持性。

    Args:
        Q: 原始变量场，形状 (n_cells, n_sps, 5)
        det_jacs: 形状 (n_cells, n_sps)

    Returns:
        Q_limited: 硬保护阈值单元被替换为体积平均值后的场，其余单元不变
    """
    troubled = troubled_cell_mask(det_jacs, TROUBLED_CELL_HARD_DET_JAC)
    if not np.any(troubled):
        return Q
    Q_limited = Q.copy()
    Q_avg = Q[troubled].mean(axis=1, keepdims=True)  # (n_troubled,1,5)
    Q_limited[troubled] = Q_avg
    return Q_limited


def face_needs_correction_freeze(
    cell: int, troubled: np.ndarray, cell_face_misalignment: Optional[np.ndarray]
) -> bool:
    """判断某单元自身的面校正跳跃项是否需要冻结：机制1（`troubled`，
    原始 det(J) 判据）*或* 机制2（`cell_face_misalignment`，法向失配
    判据）任一满足即可——两者是相互独立的触发条件，见模块文档。

    Args:
        cell: 单元索引
        troubled: troubled_cell_mask 的结果（机制1），形状 (n_cells,)
        cell_face_misalignment: 每个单元自身连接的所有面中，自己方向
            相对该面 true_normal 的最大偏离量（1-cos(夹角)，机制2），
            形状 (n_cells,)；None 时退化为只用机制1（未预计算失配信息
            的场景，如单元测试用的合成小网格）。
    """
    if bool(troubled[cell]):
        return True
    if cell_face_misalignment is None:
        return False
    return bool(cell_face_misalignment[cell] > _FACE_MISALIGNMENT_HARD_THRESHOLD)


def precompute_cell_face_misalignment(mesh) -> np.ndarray:
    """一次性（几何相关，与流场状态无关，网格加载时调用一次并缓存）计算
    每个单元自身连接的所有面中，自己方向相对该面平面几何法向
    `true_normal` 的最大偏离量 1-cos(夹角)，用于 face_needs_correction_freeze
    的机制2判据。

    每一侧独立计算自己相对 true_normal 的偏离（不需要跨单元查找对侧
    自己的坍缩坐标轴信息），天然对棱柱四边形侧面被拆成 2 个真实相邻
    单元的情形（约 5% 的棱柱）与普通情形一视同仁：owner_is_primary 的
    每条记录（含拆分子面）独立贡献 owner 侧的偏离，neighbor_is_primary
    的每条记录独立贡献 neighbor 侧的偏离。

    Returns:
        cell_face_misalignment: 形状 (n_cells,)
    """
    fc = mesh.face_connectivity
    ffp_list = mesh.face_flux_points
    n_prism = mesh.n_prism_cells
    ops = mesh.operators
    det_jacs = mesh.jacobians["det_jacs"].reshape(mesh.n_cells, -1)
    inv_jacs = mesh.jacobians["inv_jacs"].reshape(mesh.n_cells, -1, 3, 3)
    adj_j = det_jacs[..., None, None] * inv_jacs

    def extrap_to_face(cell, field, axis, side):
        E = ops.boundary_extrap_prism[(axis, side)] if cell < n_prism else ops.boundary_extrap_tet[(axis, side)]
        trailing = field.shape[1:]
        flat = E @ field.reshape(field.shape[0], -1)
        return flat.reshape((E.shape[0],) + trailing)

    def own_dir_outward(cell, axis, side):
        row = extrap_to_face(cell, adj_j[cell][:, axis, :], axis, side)
        mag = np.linalg.norm(row, axis=-1)
        return (row / np.maximum(mag[:, None], 1e-300)) * side

    cell_misalign = np.zeros(mesh.n_cells)
    for f in range(fc.n_faces):
        ffp = ffp_list[f]
        if ffp.owner_is_primary:
            owner_cell = int(fc.owner_cell[f])
            d = own_dir_outward(owner_cell, ffp.owner_axis, ffp.owner_side)
            misalign = 1.0 - np.sum(d * ffp.true_normal, axis=-1)
            cell_misalign[owner_cell] = max(cell_misalign[owner_cell], float(misalign.max()))
        if (not fc.is_boundary[f]) and ffp.neighbor_is_primary:
            neighbor_cell = int(fc.neighbor_cell[f])
            d = own_dir_outward(neighbor_cell, ffp.neighbor_axis, ffp.neighbor_side)
            misalign = 1.0 - np.sum(d * (-ffp.true_normal), axis=-1)
            cell_misalign[neighbor_cell] = max(cell_misalign[neighbor_cell], float(misalign.max()))
    return cell_misalign


def summarize_degenerate_cells(det_jacs: np.ndarray, scaled_quality: Optional[np.ndarray] = None) -> Dict[str, int]:
    """统计问题单元数量，供网格加载/求解器初始化时打印一次性诊断报告
    （不改变任何数值行为，纯诊断）。

    Returns:
        {"n_cells": 总单元数, "n_hard": 机制1（det(J)<阈值）命中单元数,
         "n_shape_warn": 形状质量预警单元数（诊断量，可选）}
    """
    n_cells = det_jacs.shape[0]
    n_hard = int(np.sum(troubled_cell_mask(det_jacs)))
    result = {"n_cells": n_cells, "n_hard": n_hard}
    if scaled_quality is not None:
        result["n_shape_warn"] = int(np.sum(cell_min_shape_quality(scaled_quality) < SHAPE_QUALITY_WARN_THRESHOLD))
    return result


def log_degenerate_cell_report(
    det_jacs: np.ndarray,
    cell_face_misalignment: Optional[np.ndarray] = None,
    scaled_quality: Optional[np.ndarray] = None,
) -> Dict[str, int]:
    """打印问题单元诊断报告并返回统计结果（见 summarize_degenerate_cells）。

    典型来源：棱柱-四面体过渡区、边界层内细小单元。这些单元在
    core/fr_residual_inviscid.py 里会被局部降阶（体积项，机制1）+ 视
    情况冻结面校正项（机制2）以防止灾难性发散，不会导致求解崩溃，但
    局部精度退化到一阶——如果占比明显偏高，建议改善网格（棱柱-四面体
    过渡区尺寸梯度约束、tet→poly 转换等），而不是仅依赖求解器兜底。
    """
    stats = summarize_degenerate_cells(det_jacs, scaled_quality)
    n_cells, n_hard = stats["n_cells"], stats["n_hard"]
    if n_hard == 0 and (cell_face_misalignment is None or not np.any(cell_face_misalignment > _FACE_MISALIGNMENT_HARD_THRESHOLD)):
        logger.info(f"Degenerate-cell check: 0/{n_cells} cells trigger mechanism-1/2 protection - mesh quality OK.")
        return stats

    shape_extra = ""
    if "n_shape_warn" in stats:
        shape_extra = f" (of which {stats['n_shape_warn']} ({100*stats['n_shape_warn']/n_cells:.3f}%) also have " \
                       f"genuinely degraded shape quality<{SHAPE_QUALITY_WARN_THRESHOLD}, rest are just small)"

    misalign_extra = ""
    n_face_frozen = n_hard
    if cell_face_misalignment is not None:
        troubled = troubled_cell_mask(det_jacs)
        n_misaligned = int(np.sum(cell_face_misalignment > _FACE_MISALIGNMENT_HARD_THRESHOLD))
        n_face_frozen = int(np.sum(troubled | (cell_face_misalignment > _FACE_MISALIGNMENT_HARD_THRESHOLD)))
        misalign_extra = (
            f"; separately, {n_misaligned} ({100*n_misaligned/n_cells:.3f}%) cells have face-normal "
            f"misalignment>{FACE_MISALIGNMENT_HARD_THRESHOLD_DEG:.1f}deg (mechanism 2, largely a different "
            f"population - typically prism quad faces)"
        )

    logger.warning(
        f"Degenerate-cell check: {n_hard}/{n_cells} ({100*n_hard/n_cells:.2f}%) cells with det(J)<"
        f"{TROUBLED_CELL_HARD_DET_JAC:.0e} (mechanism 1: volume flux locally degraded to P=0){shape_extra}"
        f"{misalign_extra}. Face correction is frozen for {n_face_frozen} ({100*n_face_frozen/n_cells:.3f}%) "
        f"cells in total (union of both mechanisms). See fr_troubled_cell.py; consider mesh improvement if "
        f"this fraction is large."
    )
    return stats


# 机制3（RESIDUAL_OUTLIER_FACTOR）：真实复现的灾难放大比同单元内正常
# SP 间残差差异高出 7~10 个数量级（3.14e5 vs ~1e-8 量级），任何合理
# 物理场在单个（微小）单元内部的残差变化不会到 1e4 倍这个量级，取值
# 留有充分安全边际，不会误伤真实的局部大梯度。
RESIDUAL_OUTLIER_FACTOR = 1e4
# 场值相对下限：低于"该变量自身场值量级 * 此下限"的残差差异一律视为
# 噪声，不参与异常判定——避免在健康单元（残差普遍已经很小，中位数本身
# 逼近浮点噪声）里把噪声当异常清零。
RESIDUAL_OUTLIER_FIELD_REL_FLOOR = 1e-9


def suppress_residual_outliers(
    residual: np.ndarray,
    reference_field: np.ndarray,
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
    """
    ref_sibling = np.median(np.abs(residual), axis=1, keepdims=True)  # (n_cells,1,n_vars)
    ref_field = field_rel_floor * np.mean(np.abs(reference_field), axis=1, keepdims=True)
    ref = np.maximum(np.maximum(ref_sibling, ref_field), 1e-300)
    outlier = np.abs(residual) > factor * ref
    if not np.any(outlier):
        return residual
    return np.where(outlier, 0.0, residual)
