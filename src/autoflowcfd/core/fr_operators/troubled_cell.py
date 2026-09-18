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
    owner_cell: np.ndarray, neighbor_cell: np.ndarray, is_boundary: np.ndarray,
    owner_side: np.ndarray, owner_is_primary: np.ndarray,
    neighbor_side: np.ndarray, neighbor_is_primary: np.ndarray,
    true_normal: np.ndarray,
    owner_adj_row_exact: np.ndarray, neighbor_adj_row_exact: np.ndarray,
    n_faces: int, n_fp: int, n_cells: int,
    owner_cube_face: np.ndarray, neighbor_cube_face: np.ndarray,
) -> np.ndarray:
    """`precompute_cell_face_misalignment` 的数值核心：`own_dir_outward`
    与 `true_normal` 逐行归一化后点积，取 `1-dot` 的单元内最大值。

    真实 bug 修复（2026-08-23，见 fr/face_flux_points_exact_normal.py
    模块文档）：`owner_adj_row_exact`/`neighbor_adj_row_exact` 取代了
    此前这里对 `det_jacs`/`inv_jacs` 用 `boundary_extrap` 做 Lagrange
    外插到 FP 得到 `own_dir_outward` 的做法（`rx,ry,rz` 曾经是
    `E @ (det_jacs[cell]*inv_jacs[cell,:,axis,:])` 的展开三重循环）——
    外插本身的截断误差是"机制2失配率此前只能部分改善、不能收敛到
    接近零"的直接原因；直接读取已经在 mesh 加载阶段一次性算好的逐 FP
    精确值，不再需要这个内核自己重新做外插，函数也因此不再需要
    `det_jacs`/`inv_jacs`/`boundary_extrap`/`owner_axis`/`neighbor_axis`/
    `n_sps`/`n_prism` 这些只是为了做外插才需要的参数。

    native 四面体（路径C）真实 bug 修复（2026-08-30，真实 cube_demo
    生产网格 P1 native 模式跑通后发现——`log_degenerate_cell_report`
    打印"87.736% 单元 face-normal misalignment"，与坍缩坐标模式同一
    份网格上的 5.042% 形成巨大反差，排查后确认是本函数遗漏 native
    分支，不是真实几何/残差问题）：`oside`/`nside`（`owner_side`/
    `neighbor_side`）对 native 面是复用槽位哑值，不代表真正的
    ±1 定向语义（见 face_flux_points_merge.py"轴槽位复用"说明、
    inviscid_kernel.py 的 `side_factor` 判据文档）——用它去乘一个
    `owner_adj_row_exact`/`neighbor_adj_row_exact` 已经自带正确 outward
    定向的 native 面精确 adj 行，等于随机翻转方向，几乎必然把 `dot`
    从接近 1 翻成接近 -1，`m=1-dot` 因此几乎恒为约 2、远超
    `misalignment>1deg` 阈值——这是纯诊断层面的误报（`suppress_
    residual_outliers`——真正在残差计算路径里生效的机制3——不消费
    这个诊断量，所以不影响任何实际残差/收敛行为，仅仅是打印出来的
    报告具有严重误导性）。修复：native 面的 side 因子固定为 +1（与
    inviscid_kernel.py 的 `side_factor` 同一原则），不使用
    `oside`/`nside` 复用槽位值。
    """
    cell_misalign = np.zeros(n_cells)
    for f in range(n_faces):
        if owner_is_primary[f]:
            oc = owner_cell[f]
            oc_code = owner_cube_face[f]
            side_factor_o = 1.0 if oc_code >= 6 else owner_side[f]
            worst = 0.0
            for i in range(n_fp):
                rx = owner_adj_row_exact[f, i, 0]
                ry = owner_adj_row_exact[f, i, 1]
                rz = owner_adj_row_exact[f, i, 2]
                mag = np.sqrt(rx * rx + ry * ry + rz * rz)
                mag = mag if mag > 1e-300 else 1e-300
                dx = (rx / mag) * side_factor_o
                dy = (ry / mag) * side_factor_o
                dz = (rz / mag) * side_factor_o
                dot = dx * true_normal[f, i, 0] + dy * true_normal[f, i, 1] + dz * true_normal[f, i, 2]
                m = 1.0 - dot
                if m > worst:
                    worst = m
            if worst > cell_misalign[oc]:
                cell_misalign[oc] = worst
        if (not is_boundary[f]) and neighbor_is_primary[f]:
            nc = neighbor_cell[f]
            nc_code = neighbor_cube_face[f]
            side_factor_n = 1.0 if nc_code >= 6 else neighbor_side[f]
            worst = 0.0
            for i in range(n_fp):
                rx = neighbor_adj_row_exact[f, i, 0]
                ry = neighbor_adj_row_exact[f, i, 1]
                rz = neighbor_adj_row_exact[f, i, 2]
                mag = np.sqrt(rx * rx + ry * ry + rz * rz)
                mag = mag if mag > 1e-300 else 1e-300
                dx = (rx / mag) * side_factor_n
                dy = (ry / mag) * side_factor_n
                dz = (rz / mag) * side_factor_n
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
    已经存好的扁平数组，交给 numba kernel。

    精度修复（2026-08-23，见 fr/face_flux_points_exact_normal.py 模块
    文档）：`own_dir_outward` 现在直接读取 `flat.owner_adj_row_exact`/
    `flat.neighbor_adj_row_exact`（mesh 加载阶段一次性算好的逐 FP 精确
    adj(J) 行），不再对 `det_jacs`/`inv_jacs` 做 Lagrange 外插——这也
    意味着 owner 侧比较（`owner_adj_row_exact` 归一化后 vs 同样来自
    owner 侧精确值的 `true_normal`）现在恒等（数值上 `1-dot≈0`，浮点
    舍入级），真正非零的失配只会来自 neighbor 侧比较——这才是这个诊断
    真正应该测量的量：owner/neighbor 两侧*各自独立*的局部度量方向是否
    一致，即 `5_重大问题修复-黎曼求解器法向.md` 记载的、仍然"未根治"
    的内部面通量守恒性问题的直接几何体现，不再混杂外插截断误差。
    """
    from autoflowcfd.core.fr_operators.face_kernels import get_flat_face_geometry

    ops = mesh.operators
    flat = get_flat_face_geometry(mesh, ops)

    return _cell_face_misalignment_kernel(
        flat.owner_cell, flat.neighbor_cell, flat.is_boundary,
        flat.owner_side, flat.owner_is_primary,
        flat.neighbor_side, flat.neighbor_is_primary,
        flat.true_normal,
        flat.owner_adj_row_exact, flat.neighbor_adj_row_exact,
        flat.n_faces, flat.n_fp, mesh.n_cells,
        flat.owner_cube_face, flat.neighbor_cube_face,
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
    from autoflowcfd.fr.prism_basis_mode import prism_basis_is_native

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
