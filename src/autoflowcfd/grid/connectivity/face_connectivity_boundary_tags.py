"""
AutoFlowCFD - 边界面的边界组标签：体网格逐面边界组（`BoundaryMap.faces`，见该类文档）按节点三元组精确对号。

供边界幽灵态构建（`core/fr_solver/boundary/ghost.py`）、周期配对与边界导出共用。2026-10-22（I-130）以前边界组只存
单元下标：按单元打标签在角 / 棱单元上无从判定每个面属于哪个组，求解期只能回到原始面网格对每个边界面做几何最近邻
匹配；现在边界组本身就是面，打标签是精确的整数对号，不涉及任何几何容差。
"""

from typing import Dict, Tuple

import numpy as np


def triangle_keys(tri: np.ndarray) -> np.ndarray:
    """三角形 `(n, 3)` 节点编号 -> 与顶点顺序无关的键 `(n,)`（排序后的三个 int64 按字节拼成定长 void；不会像
    `(a * n + b) * n + c` 那样在大网格上溢出）。"""
    t = np.ascontiguousarray(np.sort(np.asarray(tri, dtype=np.int64).reshape(-1, 3), axis=1))
    return t.view(np.dtype((np.void, t.dtype.itemsize * 3))).ravel()


def face_group_codes(face_nodes: np.ndarray, faces_by_group: Dict[str, np.ndarray]) -> Tuple[np.ndarray, list]:
    """`face_nodes (m, 3)` 中每个面所属的组：`(codes (m,) int32，-1 为不属于任何组；names)`，`codes` 是 `names`
    （`faces_by_group` 的键顺序）里的下标。

    Raises:
        ValueError: `face_nodes` 里有重复的面；某个组的面在 `face_nodes` 里找不到（组与这些面不对应：网格与边界组
            不是同一份，或调用方漏了面）；同一个面属于两个组
    """
    names = list(faces_by_group)
    keys_f = triangle_keys(face_nodes)
    m = keys_f.size
    parts = [triangle_keys(faces_by_group[name]) for name in names]
    keys_g = np.concatenate(parts) if parts else keys_f[:0]
    code_g = np.concatenate([np.full(p.size, gi, dtype=np.int32) for gi, p in enumerate(parts)]) if parts \
        else np.empty(0, dtype=np.int32)
    uniq, inv = np.unique(np.concatenate([keys_f, keys_g]), return_inverse=True)
    inv_f, inv_g = inv[:m], inv[m:]
    if np.unique(inv_f).size != m:
        raise ValueError(f"{m - np.unique(inv_f).size} 个面在面表里重复出现")
    position = np.full(uniq.size, -1, dtype=np.int64)
    position[inv_f] = np.arange(m)
    target = position[inv_g]
    missing = target < 0
    if missing.any():
        lost = {names[g]: int(n) for g, n in enumerate(np.bincount(code_g[missing], minlength=len(names))) if n}
        raise ValueError(f"边界组的面在网格的边界面里找不到：{lost}（边界组与这份网格不对应）")
    if np.unique(target).size != target.size:
        raise ValueError(f"{target.size - np.unique(target).size} 个边界面同时属于多个边界组")
    codes = np.full(m, -1, dtype=np.int32)
    codes[target] = code_g
    return codes, names


def tag_boundary_groups_for_mesh(mesh, face_conn=None) -> Tuple[np.ndarray, Dict[str, int]]:
    """面连接关系（缺省 `mesh.face_connectivity`；单独传入用于尚未挂回 mesh 的构造中途）的边界面打组标签：
    `(group_code (n_faces,) int32，内部面 / 未匹配为 -1；name_to_code)`。

    组取 `mesh.boundary_faces`（`HighOrderMesh.load_from_volume_mesh` 从 `VolumeMeshData.boundaries.faces` 取得）。
    周期组不参与：配对之后它们的面是内部面（`face_connectivity_periodic.py`）。

    Raises:
        ValueError: 网格没有逐面边界组；或见 `face_group_codes`
    """
    fc = face_conn if face_conn is not None else mesh.face_connectivity
    faces = getattr(mesh, "boundary_faces", None)
    if faces is None:
        raise ValueError("网格没有逐面的边界组（mesh.boundary_faces，来自 VolumeMeshData.boundaries.faces）")
    bc_types = mesh.boundary_bc_types or {}
    groups = {name: tri for name, tri in faces.items() if bc_types.get(name) != "PERIODIC"}
    boundary_idx = fc.get_boundary_face_indices()
    codes, names = face_group_codes(fc.face_node_ids[boundary_idx], groups)
    group_code = np.full(fc.n_faces, -1, dtype=np.int32)
    group_code[boundary_idx] = codes
    return group_code, {name: code for code, name in enumerate(names)}
