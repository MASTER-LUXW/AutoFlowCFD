"""AutoFlowCFD V2.0 - 几何守恒律（GCL）自检。

从 `high_order_mesh.py`（原 510 行）拆出（2026-09-19，项目"单文件不超
500 行"规范）。纯搬家，逻辑未改；只把方法改成以 `mesh` 为第一参数的
模块级函数，`HighOrderMesh` 上保留同名薄委托方法，调用方式不变。
"""

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

    度量阶数与解阶数解耦（2026-08-23，此前"P1 GCL=0.105"被记录为
    已知、搁置的诊断局限，用户明确要求修复）：原实现对 adj(J) 精确
    求值后，用与当前求解阶数*相同*的坍缩坐标微分矩阵 D_3d_tet/prism
    求散度——但 adj(J) 在 Duffy 坍缩坐标下是有理函数、不是多项式，
    P1（degree=1）的微分矩阵次数不足以精确微分它，给出一个纯属
    诊断函数自身局限的 0.105（P2=1.6e-13、P3=1.2e-8 因为微分矩阵
    次数够高，问题不明显），不代表真实求解器残差有问题（真实求解器
    路径 compute_inviscid_residual_fr 的 P1 均匀自由流场残差实测
    1.25e-10，是 P1/P2/P3 三者里最好的，见 order_continuation.py
    模块文档）。

    修复思路：不再让"用来微分 adj(J)"的算子阶数与"求解阶数"绑死，
    复用真实求解器体积项本来就已经在用的过积分（over-integration）
    基础设施本身（`mesh.jacobians_fine`/`mesh.operators.
    overint_D_fine_tet/prism`，over_order=min(2*order,
    OVERINTEGRATION_MAX_ORDER)，见 high_order_mesh_order.py::
    build_order_geometry 文档）：adj(J) 在过积分细网格上精确求值
    （tet_exact_jacobian/prism_exact_jacobian 本身与阶数无关，多少
    个点都能精确求值），散度改用细网格自己更高次、更准确的微分矩阵，
    与体积项组装用的是完全同一套算子。

    真实验证结果（同一份合成混合网格）：P1 从 0.105 降到 1.64e-13
    （12 个数量级），确认原判据是纯粹的诊断局限。但同一次验证也
    发现：P2/P3 切到过积分反而从原生的 1.6e-13/1.2e-8 变差到
    1.17e-8（两者的 over_order 都被 OVERINTEGRATION_MAX_ORDER=3
    封顶，退化到同一个网格）——不是过积分本身有 bug，是本项目已经
    记录过的坍缩坐标高阶模态基条件数问题（Vandermonde 矩阵条件数
    随阶数增长，见 fr/collapsed_basis.py 相关文档）：degree-3 微分
    矩阵的截断误差改善不足以抵消它更差的舍入误差特性，P2 原生
    degree-2 矩阵在这个光滑合成网格上恰好已经落在"截断/舍入都小"
    的甜点区。因此只对 P1（唯一有真实、大幅度诊断局限的阶数）切换
    到过积分路径，P2/P3 保留已经很好的原生检验，不用一个"理论上
    更精确"但实测更差的路径替换一个已经工作良好的路径。
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
        residual = mapper.compute_metric_identity_residual(
            mesh.sps_coords[i], cell_type=cell_type, cell_nodes=cell_nodes, ref_cube_sps=mesh._ref_cube_sps
        )
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
    解耦"一节。散度算子与真实体积项组装（fr_residual_inviscid.py）
    用的是同一个 `contract_shared_operator_2axis`，不是重新实现一遍
    同一个数学操作的第二份独立代码。
    """
    from autoflowcfd.core.fr_operators.volume_contract import (
        contract_shared_operator_2axis, get_overintegration_context,
    )

    # 按段取细点度量（2026-09-17）：native 四面体过积分的细网格轴不再
    # 填充到棱柱的 (oo+1)^3 宽度，两段的 n_fine 不同了，不能再共用一份
    # `(n_cells, n_fine, ...)` 的整场数组。改用与真实体积项组装同一个
    # 上下文 helper，保持"诊断与生产走同一份算子/度量"这条既定设计。
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
        adj_seg = np.ascontiguousarray(det_seg)[..., None, None]                 * np.ascontiguousarray(inv_seg)   # adj[c,j,m,i]
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
