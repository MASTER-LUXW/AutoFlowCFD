"""AutoFlowCFD V2.0 - 面记录分组与"混合边界/内部分组"检测。

从 `merge.py`（原 670 行）拆出（2026-09-19，项目"单文件不超 500 行"
规范）。整段**原样搬运**，逻辑一字未改 —— 原来是
`build_face_flux_points` 里的一段内联代码，与后续步骤之间只通过下面这
个 `FaceGroups` 里的量耦合，本身自成体系。

## 这一段在做什么

棱柱的三个**四边形**侧面在网格生成时被对侧三角化成两条面记录（三角形
封盖不受影响）。所以同一个 `(cell, cube_face)` 可能对应 1~2 条记录：

  * 两条都是内部界面 -> multi-source（两个真实相邻单元，逐 FP 按对角线
    分半区）；
  * 两条都在域边界 -> 只让第一条担任 primary，否则边界校正项翻倍
    （真实复现：Couette 棱柱网格上贴壁单元前几步残差被放大到 0.1~0.93）；
  * **一条内部、一条边界**（"混合分组"，B-8）-> BL 挤出在几何尖角棱处
    产生拓扑缝隙的固有产物。内部那条担任整张面 primary，另半区逐 FP 取
    边界那条的幽灵态。旧版按"组内必有 2 条"取 `group[1]` 直接越界崩溃。
"""

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
from loguru import logger

from .data import _PRISM_QUAD_CODES, _classify_half, prism_quad_local_idx
from autoflowcfd.grid.connectivity.face_connectivity import FRFaceConnectivity


@dataclass
class FaceGroups:
    """`build_face_groups` 的产物。字段名与 `merge.py` 里原来的局部变量
    **同名**，便于逐行核对搬家前后一致。"""

    owner_groups: Dict[Tuple[int, int], List[int]]
    neighbor_groups: Dict[Tuple[int, int], List[int]]
    boundary_owner_groups: Dict[Tuple[int, int], List[int]]
    owner_primary: np.ndarray        # bool (n_faces,)
    neighbor_primary: np.ndarray     # bool (n_faces,)
    mixed_nb_partner: np.ndarray     # int64 (n_faces,)
    mixed_ow_partner: np.ndarray     # int64 (n_faces,)
    mixed_nb_mask: np.ndarray        # bool (n_faces, n_fp)
    mixed_ow_mask: np.ndarray        # bool (n_faces, n_fp)
    mixed_bnd_face: np.ndarray       # bool (n_faces,)
    mixed_p0_bnd_frac: np.ndarray    # float64 (n_faces,)


def build_face_groups(
    face_conn: FRFaceConnectivity,
    mesh,
    n_faces: int,
    n_fp: int,
    n_prism: int,
    is_lower_fp_standard: np.ndarray,
    is_lower_fp_flipped: np.ndarray,
) -> FaceGroups:
    """按 `(cell, cube_face)` 给面记录分组，并检测混合分组。

    Args:
        is_lower_fp_standard / is_lower_fp_flipped: 两种对角线走向下
            "FP 落在下半区"的掩码，由调用方按 `face_ref_grid` 同一套
            顺序预先算好（见 `merge.py` 里那段说明）。
    """
    owner_groups: Dict[Tuple[int, int], List[int]] = {}
    neighbor_groups: Dict[Tuple[int, int], List[int]] = {}
    # 边界面单独分组（不需要 neighbor 侧、不需要按对角线拆分 sources——
    # 幽灵态对整张四边形面统一取值，与两条子面记录各自覆盖对角线哪一半
    # 无关），只是为了标记同一 (owner_cell, 立方体面) 的重复子面记录，
    # 避免下面主循环对每条记录都各自完整跑一遍原生 FP 外插+校正投影导致
    # 边界校正项翻倍——与非边界分支的 owner_groups 是同一个 bug 的另一半，
    # 此前只修了 non-boundary 分支，边界分支被漏掉了（真实复现：Couette
    # 合成算例棱柱网格上，几乎每个贴着 z_min/z_max 的单元在时间推进的
    # 前几步内边界校正项残差就被放大到 0.1~0.93 量级，根因就是这里）。
    boundary_owner_groups: Dict[Tuple[int, int], List[int]] = {}
    for f in range(n_faces):
        if face_conn.is_boundary[f]:
            oc_cell, oc_code = int(face_conn.owner_cell[f]), int(face_conn.owner_cube_face[f])
            if oc_cell < n_prism and oc_code in _PRISM_QUAD_CODES:
                boundary_owner_groups.setdefault((oc_cell, oc_code), []).append(f)
            continue
        oc_cell, oc_code = int(face_conn.owner_cell[f]), int(face_conn.owner_cube_face[f])
        if oc_cell < n_prism and oc_code in _PRISM_QUAD_CODES:
            owner_groups.setdefault((oc_cell, oc_code), []).append(f)
        nc_cell, nc_code = int(face_conn.neighbor_cell[f]), int(face_conn.neighbor_cube_face[f])
        if nc_cell < n_prism and nc_code in _PRISM_QUAD_CODES:
            neighbor_groups.setdefault((nc_cell, nc_code), []).append(f)

    owner_primary = np.ones(n_faces, dtype=bool)
    neighbor_primary = np.ones(n_faces, dtype=bool)
    for flist in owner_groups.values():
        for f in sorted(flist)[1:]:
            owner_primary[f] = False
    for flist in neighbor_groups.values():
        for f in sorted(flist)[1:]:
            neighbor_primary[f] = False
    for flist in boundary_owner_groups.values():
        for f in sorted(flist)[1:]:
            owner_primary[f] = False

    # ---- 混合分组检测（B-8 修复，2026-08-25）----
    # BL 挤出在几何尖角棱处产生拓扑缝隙的固有产物：某个棱柱四边形侧面
    # 的 2 条三角化子面记录中，一条在域边界分组（单侧暴露、无真实邻居）、
    # 另一条是内部界面（与真实邻居配对）——内部分组只有 1 条记录，旧版
    # multi-source 代码按组内 2 条记录取 group[1] 直接越界崩溃（真实复现：
    # cube_demo 尖角网格，solve steady 在 FP 几何构建阶段 IndexError）。
    # 语义：内部子面记录担任整张四边形面的 primary——子面覆盖的对角线半区
    # 内照常构建跨单元插值，另半区逐 FP 取边界子面记录的幽灵态（两条
    # 记录共享同一 owner 棱柱与立方体面，FP 网格逐点重合）。边界子面记录
    # 不再参与残差累加（否则与内部记录在整张面上重复计数），但幽灵态仍需
    # 计算（见 mixed_bnd_face 与 inviscid_kernel.py 幽灵态预计算的分支）。
    mixed_nb_keys = sorted(set(owner_groups) & set(boundary_owner_groups))
    mixed_ow_keys = sorted(set(neighbor_groups) & set(boundary_owner_groups))
    mixed_nb_partner = np.full(n_faces, -1, dtype=np.int64)
    mixed_ow_partner = np.full(n_faces, -1, dtype=np.int64)
    mixed_nb_mask = np.zeros((n_faces, n_fp), dtype=np.bool_)
    mixed_ow_mask = np.zeros((n_faces, n_fp), dtype=np.bool_)
    mixed_bnd_face = np.zeros(n_faces, dtype=np.bool_)
    mixed_p0_bnd_frac = np.zeros(n_faces, dtype=np.float64)

    def _register_mixed(key, int_faces, bnd_faces, partner_arr, mask_arr):
        """登记一个混合分组：f_int 为整张面 primary，bf 供幽灵态取用。"""
        f_int, bf = int_faces[0], bnd_faces[0]
        owner_primary[bf] = False
        mixed_bnd_face[bf] = True
        partner_arr[f_int] = bf
        cell_node_ids = mesh._fixed_prism_conn[key[0]]
        quad_local_idx = prism_quad_local_idx(key[1])
        half, is_std = _classify_half(
            cell_node_ids, quad_local_idx, face_conn.face_node_ids[f_int]
        )
        is_lower = is_lower_fp_standard if is_std else is_lower_fp_flipped
        interior_mask = is_lower if half == "lower" else ~is_lower
        mask_arr[f_int] = ~interior_mask  # True = 边界半区（取幽灵态）
        # P0 单态/面粒度：边界半区通量单独用幽灵态算，按面积占比混合，
        # 见 inviscid_p0_kernel.py 的 mixed_p0_bnd_frac 分支。
        total_area = float(face_conn.area[f_int]) + float(face_conn.area[bf])
        mixed_p0_bnd_frac[f_int] = float(face_conn.area[bf]) / max(total_area, 1e-300)

    for key in mixed_nb_keys:
        int_faces = sorted(owner_groups[key])
        bnd_faces = sorted(boundary_owner_groups[key])
        if len(int_faces) != 1 or len(bnd_faces) != 1:
            raise RuntimeError(
                f"混合边界/内部棱柱四边形分组 (cell={key[0]}, face={key[1]}) 含 "
                f"{len(int_faces)} 条内部子面 + {len(bnd_faces)} 条边界子面记录，"
                f"仅支持 1+1（非协调网格拓扑，需检查网格生成）。"
            )
        _register_mixed(key, int_faces, bnd_faces, mixed_nb_partner, mixed_nb_mask)
    for key in mixed_ow_keys:
        int_faces = sorted(neighbor_groups[key])
        bnd_faces = sorted(boundary_owner_groups[key])
        if len(int_faces) != 1 or len(bnd_faces) != 1:
            raise RuntimeError(
                f"混合边界/内部棱柱四边形分组（neighbor 角色）(cell={key[0]}, "
                f"face={key[1]}) 含 {len(int_faces)} 条内部子面 + {len(bnd_faces)} 条边界"
                f"子面记录，仅支持 1+1（非协调网格拓扑，需检查网格生成）。"
            )
        _register_mixed(key, int_faces, bnd_faces, mixed_ow_partner, mixed_ow_mask)
    if mixed_nb_keys or mixed_ow_keys:
        logger.info(
            f"检测到混合边界/内部四边形分组（尖角缝隙面）：nb={len(mixed_nb_keys)}, "
            f"ow={len(mixed_ow_keys)}，内部子面记录担任整张面 primary"
        )

    return FaceGroups(
        owner_groups=owner_groups,
        neighbor_groups=neighbor_groups,
        boundary_owner_groups=boundary_owner_groups,
        owner_primary=owner_primary,
        neighbor_primary=neighbor_primary,
        mixed_nb_partner=mixed_nb_partner,
        mixed_ow_partner=mixed_ow_partner,
        mixed_nb_mask=mixed_nb_mask,
        mixed_ow_mask=mixed_ow_mask,
        mixed_bnd_face=mixed_bnd_face,
        mixed_p0_bnd_frac=mixed_p0_bnd_frac,
    )
