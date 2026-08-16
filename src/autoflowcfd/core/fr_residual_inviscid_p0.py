"""
AutoFlowCFD V2.0 - P0 阶专用有限体积无粘残差 (S-02)

从 fr_residual_inviscid.py 拆出来（控制单文件行数，>400 行需拆分的
项目规范）。`compute_inviscid_residual_fr` 在 `mesh.n_points_1d==1`
时委托到这里，见该函数文档。
"""

from typing import Callable, Optional

import numpy as np


def compute_inviscid_residual_fv_p0(
    U: np.ndarray,
    mesh,
    boundary_ghost_provider: Optional[Callable[[int, np.ndarray, np.ndarray], np.ndarray]] = None,
) -> np.ndarray:
    """P0（1 SP/cell，Order Continuation 最低阶）专用有限体积残差。

    背景：P>=1 的界面项用坍缩坐标体积度量张量 adj(J) 外插到面上再做一致性
    校验（见 compute_inviscid_residual_fr 里的 alignment 校验），这在 P0
    下必然报错——坍缩（Duffy）坐标的 Jacobian 在单元内部本就强烈非均匀
    （fr/collapsed_basis.py 模块文档），单元内唯一那个解点（位于坍缩参考
    立方体中心）处的度量方向，物理上没有理由与该单元 3~4 个不同面各自的
    真实法向对齐——这不是数值 bug，是"用同一个点的坍缩度量代表所有面"
    这一想法在数学上站不住脚。真实网格已复现：alignment cosine 可低至
    0.20（约78°偏差），远超 0.5 的校验阈值。

    P0 在数学上唯一自洽的定义就是经典分片常数有限体积：完全不依赖坍缩
    度量张量，直接用 face_connectivity 给出的真实几何法向 (ffp.true_normal)
    与真实面积权重 (ffp.true_area_weight) 做迎风通量积分。owner/neighbor
    共用同一个法向量、同一次 Riemann 求解结果（对两侧符号相反地施加），
    天然精确守恒——不像 P>=1 那样需要 owner/neighbor 各自独立取自己的
    度量法向（那是坍缩坐标外插固有的不一致来源，P0 完全没有这个问题，
    因为这里用的是同一个真实几何法向，不是两个独立外插出来的近似法向）。

    体积项在 P0 下无需计算：D_3d_tet/D_3d_prism 在 order=0 时解析恒为
    零矩阵（常数函数对任何参考方向的导数都是零），体积散度贡献必为零。

    关于棱柱四边形侧面拆分（真实网格已复现、修复的一个关键点）：
    `build_face_flux_points` 对 face_connectivity 里的*每一条*记录（包括
    棱柱四边形侧面因三角化被拆出的 2 条子面记录）都无条件调用
    `result.append(...)`，`true_normal`/`true_area_weight` 也在
    primary/非primary 判断之前就已算好、对每条记录都有效——`owner_is_primary`
    /`neighbor_is_primary` 只是控制"是否触发一次自身外插+跨单元投影"
    （P>=1 的坍缩度量路径需要，用来避免同一个原生 FP 网格被重复计入两次），
    与"这条记录的真实几何面积/法向是否有效"无关。本函数因此直接按
    face_connectivity 的原始每条记录处理（不经过 owner_is_primary 过滤、
    不经过 neighbor_sources/owner_sources 的多源合并——P0 下每条记录本来
    就唯一对应一个真实相邻单元，不存在"一个原生 FP 网格分给两个不同
    相邻单元"这个 P>=1 才有的问题）：曾经的第一版实现按
    `if not ffp.owner_is_primary: continue` 跳过非 primary 记录，等价于
    直接丢弃了棱柱被拆分的那一半四边形的真实面积——对闭合单元的面积/
    法向积分 Σ(n̂·A)=0 这一几何恒等式造成真实的（非浮点噪声量级的）
    破坏，在小体积单元（真实网格边界层单元体积低至 ~1e-11 m³）上除以
    体积后被放大到 1e11 量级的"伪残差"（已复现：直接改用本函数现在的
    写法后，均匀自由流场残差恢复到机器精度量级）。

    Args:
        U: 守恒变量，形状 (n_cells, 1, n_vars)
        mesh: HighOrderMesh 实例（n_points_1d 必须为 1）
        boundary_ghost_provider: 同 compute_inviscid_residual_fr

    Returns:
        residual: 形状 (n_cells, 1, 5)
    """
    from .fr_residual_inviscid import conserved_to_primitive, ausm_up_flux_batch, DefaultGhostProvider

    n_cells = mesh.n_cells
    if mesh.cell_volumes is None:
        raise RuntimeError(
            "mesh.cell_volumes not available - required for the P0 finite-volume residual path "
            "(should have been computed once in load_from_volume_mesh at the mesh's target order)."
        )
    cell_volumes = mesh.cell_volumes

    Q_all = conserved_to_primitive(U[..., :5])[:, 0, :]  # (n_cells,5)，P0 唯一解点即单元均值

    fc = mesh.face_connectivity
    ffp_list = mesh.face_flux_points
    ghost_provider = boundary_ghost_provider if boundary_ghost_provider is not None else DefaultGhostProvider()

    residual5 = np.zeros((n_cells, 5))

    for f in range(fc.n_faces):
        ffp = ffp_list[f]
        owner_cell = int(fc.owner_cell[f])
        true_normal = ffp.true_normal  # (1,3)，owner->neighbor / 边界面指向域外
        area_w = ffp.true_area_weight  # (1,)

        Q_owner_fp = Q_all[owner_cell : owner_cell + 1]  # (1,5)

        if fc.is_boundary[f]:
            Q_neighbor_fp = ghost_provider(f, Q_owner_fp, true_normal)
        else:
            neighbor_cell = int(fc.neighbor_cell[f])
            Q_neighbor_fp = Q_all[neighbor_cell : neighbor_cell + 1]

        F_common_n = ausm_up_flux_batch(Q_owner_fp, Q_neighbor_fp, true_normal)  # (1,5)
        flux_integral = F_common_n[0] * area_w[0]  # (5,)

        residual5[owner_cell] += -flux_integral / cell_volumes[owner_cell]
        if not fc.is_boundary[f]:
            residual5[neighbor_cell] += flux_integral / cell_volumes[neighbor_cell]

    return residual5[:, None, :]
