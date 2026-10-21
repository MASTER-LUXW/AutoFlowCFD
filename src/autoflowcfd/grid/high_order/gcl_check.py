"""AutoFlowCFD V2.0 - 几何守恒律（GCL）自检（`HighOrderMesh.verify_gcl` 委托到这里）。"""

import numpy as np
from loguru import logger

from ..curved_mapping.curved_mapping import CurvedMapping


def verify_gcl(mesh, tolerance: float = 1e-8) -> bool:
    """验证几何守恒律 (GCL)：对每个单元做 Kopriva 度量恒等式检验。

    Args:
        tolerance: 度量恒等式残差容差（P>=2 原生检验即可达到机器精度
            量级；P1 用下面的过积分检验后同样能达到，见该分支文档）

    Returns:
        bool: 全部单元 GCL 是否通过

    检验量是 Kopriva 度量恒等式 `sum_m d/dxi_m adj(J)_{m,i}`，用与残差组装同一套原生基微分矩阵求散度
    （`CurvedMapping.compute_metric_identity_residual`）。直边四面体的 adj(J) 是常数；棱柱有双线性侧面时
    adj(J) 在 `(r,s,t)` 里是 2 次多项式——P>=2 的微分矩阵能精确微分它，P1 不能（给出的是诊断自身的截断误差，
    不是求解器的 GCL 缺陷）。所以 P1 改在过积分细点上求值并用细网格微分矩阵求散度
    （`verify_gcl_overintegrated`，与体积项组装同一套算子与度量）。
    """
    if mesh.sps_coords is None:
        return False

    if mesh.order == 1 and mesh.jacobians_fine is not None:
        return verify_gcl_overintegrated(mesh, tolerance)

    mapper = CurvedMapping(mesh.order)
    max_residual = 0.0
    n_failed = 0
    for i in range(mesh.n_cells):
        cell_type = "prism" if i < mesh.n_prism_cells else "tet"
        if cell_type == "prism":
            cell_nodes = mesh._node_coords[mesh._fixed_prism_conn[i]]
        else:
            cell_nodes = mesh._node_coords[mesh._fixed_tet_conn[i - mesh.n_prism_cells]]
        residual = mapper.compute_metric_identity_residual(cell_type, cell_nodes)
        cell_max = float(np.max(np.abs(residual)))
        max_residual = max(max_residual, cell_max)
        if cell_max >= tolerance:
            n_failed += 1

    logger.info(f"GCL check: max metric-identity residual = {max_residual:.6e} (tolerance={tolerance:.1e})")
    if n_failed > 0:
        logger.warning(f"GCL check failed for {n_failed}/{mesh.n_cells} cells")
    return n_failed == 0


def verify_gcl_overintegrated(mesh, tolerance: float) -> bool:
    """`verify_gcl` 的过积分实现，见该方法文档"度量阶数与解阶数
    解耦"一节。散度算子与真实体积项组装（core/fr_residual/inviscid.py）
    用的是同一个 `contract_shared_operator_2axis`，不是重新实现一遍
    同一个数学操作的第二份独立代码。
    """
    from autoflowcfd.core.fr_operators.volume_contract import (
        contract_shared_operator_2axis, get_overintegration_context,
    )

    # 按段取细点度量：棱柱与四面体两段的 n_fine 不同，用与真实体积项组装同一个上下文 helper
    # （诊断与生产走同一份算子 / 度量）。
    oi = get_overintegration_context(mesh, mesh.operators)
    if oi is None:
        # 与 verify_gcl 的分派保持一致：没有过积分上下文就不该走到这里
        raise RuntimeError(
            "_verify_gcl_overintegrated 被调用但过积分上下文不可用"
            "（jacobians_fine 或 overint 算子缺失）——调用方的分派条件"
            "与实际可用性不一致"
        )

    cell_max = np.empty(mesh.n_cells)
    for (seg_lo, seg_hi, n_fine, det_seg, inv_seg,
         _c2f, op_D_fine, _f2c) in oi["segs"]:
        if seg_hi <= seg_lo:
            continue
        adj_seg = (np.ascontiguousarray(det_seg)[..., None, None]
                   * np.ascontiguousarray(inv_seg))   # adj[c,j,m,i]
        res_seg = contract_shared_operator_2axis(op_D_fine, adj_seg)
        cell_max[seg_lo:seg_hi] = np.max(np.abs(res_seg), axis=(1, 2))

    max_residual = float(np.max(cell_max))
    n_failed = int(np.sum(cell_max >= tolerance))

    logger.info(
        f"GCL check (over-integrated, order={mesh.order}): max metric-identity "
        f"residual = {max_residual:.6e} (tolerance={tolerance:.1e})"
    )
    if n_failed > 0:
        logger.warning(f"GCL check failed for {n_failed}/{mesh.n_cells} cells")
    return n_failed == 0
