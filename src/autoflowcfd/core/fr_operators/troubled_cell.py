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
from numba import njit, prange

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
    """机制1判据：单元内最小*原始* det(J) 低于 threshold 的单元掩码，形状 (n_cells,)。

    只用作 log_degenerate_cell_report 的诊断统计（见模块文档"机制3"
    一节：机制1/2 的*检测*判据保留用于诊断报告，但对残差本身的实际
    干预已经全部由机制3——suppress_residual_outliers——取代）。
    """
    return cell_min_det_jac(det_jacs) < threshold


@njit(cache=True)
def _cell_face_misalignment_kernel(
    det_jacs: np.ndarray, inv_jacs: np.ndarray,
    owner_cell: np.ndarray, neighbor_cell: np.ndarray, is_boundary: np.ndarray,
    owner_axis: np.ndarray, owner_side: np.ndarray, owner_is_primary: np.ndarray,
    neighbor_axis: np.ndarray, neighbor_side: np.ndarray, neighbor_is_primary: np.ndarray,
    true_normal: np.ndarray, boundary_extrap: np.ndarray,
    n_prism: int, n_faces: int, n_fp: int, n_sps: int, n_cells: int,
) -> np.ndarray:
    """`precompute_cell_face_misalignment` 的数值核心，逐点等价于原
    `own_dir_outward`/`extrap_to_face`（矩阵乘 `E @ adj_j[cell][:,axis,:]`
    + 逐行归一化 + 与 `true_normal` 点积），只是把 `E @ field` 展开成
    显式三重循环、`adj_j = det_jacs[...,None,None]*inv_jacs` 内联，避免
    对每个 (cell,axis) 组合重新构造整个 adj_j 数组切片。
    """
    cell_misalign = np.zeros(n_cells)
    for f in range(n_faces):
        if owner_is_primary[f]:
            oc = owner_cell[f]
            oax = owner_axis[f]
            oside = owner_side[f]
            oside_idx = 0 if oside <= 0.0 else 1
            celltype_o = 0 if oc < n_prism else 1
            E = boundary_extrap[celltype_o, oax, oside_idx]  # (n_fp, n_sps)
            worst = 0.0
            for i in range(n_fp):
                rx = 0.0
                ry = 0.0
                rz = 0.0
                for s in range(n_sps):
                    ed = E[i, s] * det_jacs[oc, s]
                    rx += ed * inv_jacs[oc, s, oax, 0]
                    ry += ed * inv_jacs[oc, s, oax, 1]
                    rz += ed * inv_jacs[oc, s, oax, 2]
                mag = np.sqrt(rx * rx + ry * ry + rz * rz)
                mag = mag if mag > 1e-300 else 1e-300
                dx = (rx / mag) * oside
                dy = (ry / mag) * oside
                dz = (rz / mag) * oside
                dot = dx * true_normal[f, i, 0] + dy * true_normal[f, i, 1] + dz * true_normal[f, i, 2]
                m = 1.0 - dot
                if m > worst:
                    worst = m
            if worst > cell_misalign[oc]:
                cell_misalign[oc] = worst
        if (not is_boundary[f]) and neighbor_is_primary[f]:
            nc = neighbor_cell[f]
            nax = neighbor_axis[f]
            nside = neighbor_side[f]
            nside_idx = 0 if nside <= 0.0 else 1
            celltype_n = 0 if nc < n_prism else 1
            E = boundary_extrap[celltype_n, nax, nside_idx]
            worst = 0.0
            for i in range(n_fp):
                rx = 0.0
                ry = 0.0
                rz = 0.0
                for s in range(n_sps):
                    ed = E[i, s] * det_jacs[nc, s]
                    rx += ed * inv_jacs[nc, s, nax, 0]
                    ry += ed * inv_jacs[nc, s, nax, 1]
                    rz += ed * inv_jacs[nc, s, nax, 2]
                mag = np.sqrt(rx * rx + ry * ry + rz * rz)
                mag = mag if mag > 1e-300 else 1e-300
                dx = (rx / mag) * nside
                dy = (ry / mag) * nside
                dz = (rz / mag) * nside
                dot = dx * (-true_normal[f, i, 0]) + dy * (-true_normal[f, i, 1]) + dz * (-true_normal[f, i, 2])
                m = 1.0 - dot
                if m > worst:
                    worst = m
            if worst > cell_misalign[nc]:
                cell_misalign[nc] = worst
    return cell_misalign


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

    性能修复（真实复现，2026-08-21，79万单元/187万面生产网格）：此前
    这里 `for f in range(fc.n_faces): ffp = ffp_list[f]` 逐面索引
    `mesh.face_flux_points`——自 face_flux_points_merge.py 的"flat array
    format"改造后，`mesh.face_flux_points` 已经是 `_KernelFaceData`（数值
    仍在扁平数组里，不是逐面对象），`_KernelFaceData.__getitem__` 为兼容
    后处理代码按需*构造*一个完整 `FaceFluxPointGeometry` 对象——187 万个
    面全部访问一遍等于触发 187 万次这种构造，是本函数（进而是每次
    `set_order`/Order Continuation 阶数切换、每次求解器初始化）实测耗时
    数分钟的直接原因，而残差求值热路径（`get_flat_face_geometry` 的
    `_KernelFaceData` 快速路径）早已绕开了这个问题，只有这个诊断量
    预计算函数遗漏。改为直接读取 `_KernelFaceData`/`FlatFaceGeometry`
    已经存好的扁平数组（`get_flat_face_geometry` 走的正是同一条已验证
    的快速路径），把 `E @ field` 矩阵乘与逐行归一化交给 numba kernel，
    数学上与原实现完全一致（同一组 `own_dir_outward`/`misalign` 公式，
    只是从"每面构造对象+逐面 numpy 矩阵乘"换成"直接读扁平数组+numba
    内联三重循环"），不引入近似。
    """
    from autoflowcfd.core.fr_operators.face_kernels import get_flat_face_geometry

    fc = mesh.face_connectivity
    ops = mesh.operators
    flat = get_flat_face_geometry(mesh, ops)
    det_jacs = mesh.jacobians["det_jacs"].reshape(mesh.n_cells, -1)
    inv_jacs = mesh.jacobians["inv_jacs"].reshape(mesh.n_cells, -1, 3, 3)

    return _cell_face_misalignment_kernel(
        det_jacs, inv_jacs,
        flat.owner_cell, flat.neighbor_cell, flat.is_boundary,
        flat.owner_axis, flat.owner_side, flat.owner_is_primary,
        flat.neighbor_axis, flat.neighbor_side, flat.neighbor_is_primary,
        flat.true_normal, flat.boundary_extrap,
        flat.n_prism, flat.n_faces, flat.n_fp, flat.n_sps, mesh.n_cells,
    )


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

    典型来源：棱柱-四面体过渡区、边界层内细小单元。机制1（det(J) 阈值）/
    机制2（法向失配）判据本身只用来*诊断*这类单元的占比，不再是实际
    干预残差的机制——对残差的实际保护现在由机制3（suppress_residual_
    outliers，症状检测，直接对算出的残差按 (cell,SP,变量) 粒度做统计
    异常清零）承担，见模块文档"机制3"一节。占比明显偏高仍然值得关注：
    意味着这部分区域的残差经常需要机制3介入清零，局部精度退化到一阶，
    建议改善网格（棱柱-四面体过渡区尺寸梯度约束、tet→poly 转换等），
    而不是仅依赖求解器兜底。
    """
    stats = summarize_degenerate_cells(det_jacs, scaled_quality)
    n_cells, n_hard = stats["n_cells"], stats["n_hard"]
    if n_hard == 0 and (cell_face_misalignment is None or not np.any(cell_face_misalignment > _FACE_MISALIGNMENT_HARD_THRESHOLD)):
        logger.info(f"Degenerate-cell check: 0/{n_cells} cells trigger mechanism-1/2 diagnostic criteria - mesh quality OK.")
        return stats

    shape_extra = ""
    if "n_shape_warn" in stats:
        shape_extra = f" (of which {stats['n_shape_warn']} ({100*stats['n_shape_warn']/n_cells:.3f}%) also have " \
                       f"genuinely degraded shape quality<{SHAPE_QUALITY_WARN_THRESHOLD}, rest are just small)"

    misalign_extra = ""
    n_flagged = n_hard
    if cell_face_misalignment is not None:
        troubled = troubled_cell_mask(det_jacs)
        n_misaligned = int(np.sum(cell_face_misalignment > _FACE_MISALIGNMENT_HARD_THRESHOLD))
        n_flagged = int(np.sum(troubled | (cell_face_misalignment > _FACE_MISALIGNMENT_HARD_THRESHOLD)))
        misalign_extra = (
            f"; separately, {n_misaligned} ({100*n_misaligned/n_cells:.3f}%) cells have face-normal "
            f"misalignment>{FACE_MISALIGNMENT_HARD_THRESHOLD_DEG:.1f}deg (mechanism 2, largely a different "
            f"population - typically prism quad faces)"
        )

    logger.warning(
        f"Degenerate-cell check: {n_hard}/{n_cells} ({100*n_hard/n_cells:.2f}%) cells with det(J)<"
        f"{TROUBLED_CELL_HARD_DET_JAC:.0e} (mechanism 1 diagnostic criterion){shape_extra}"
        f"{misalign_extra}. {n_flagged} ({100*n_flagged/n_cells:.3f}%) "
        f"cells flagged by mechanism 1 and/or 2's diagnostic criteria in total (union) - actual residual "
        f"protection for these cells is handled at runtime by mechanism 3 (suppress_residual_outliers), "
        f"not by this diagnostic. See fr_troubled_cell.py; consider mesh improvement if this fraction is large."
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
    ref_sibling = _median_abs_over_sps_kernel(residual)[:, np.newaxis, :]  # (n_cells,1,n_vars)
    ref_field = field_rel_floor * np.mean(np.abs(reference_field), axis=1, keepdims=True)
    ref = np.maximum(np.maximum(ref_sibling, ref_field), 1e-300)
    outlier = np.abs(residual) > factor * ref
    if not np.any(outlier):
        return residual
    return np.where(outlier, 0.0, residual)
