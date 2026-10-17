"""面通量点的物理坐标（唯一来源）。

P>=1：单元几何映射（直边四面体为仿射、直边棱柱在侧面上为双线性）属于解点插值多项式空间，解点坐标经 owner 面的
外插算子即得通量点的精确物理坐标（`flux_point_positions`）。

P0：每个面记录只有一个通量点，位于参考面中心的像；解点只有一个（单元中心），外插只能得到单元中心。所以 P0 的
通量点位置在构造面几何时由网格节点直接算出（`p0_flux_points`）：三角形面为三个顶点的平均；棱柱四边形侧面在网格
生成阶段被拆成两条三角形子面记录，其中"主"记录代表整个四边形（面积权重是两个三角形之和），参考面中心的像是四个
顶点的平均（侧面映射在边参数与高度方向上双线性）——两条子面记录都取这一点。

此前壁面力的力臂、SEM 合成湍流入口的采样坐标、WMLES 的匹配点高度都用外插求通量点坐标，P0 下全部落在单元中心。
"""

import numpy as np


def p0_flux_points(owner_cell, owner_cube_face, face_node_ids, node_coords) -> np.ndarray:
    """每个面记录 P0 通量点的物理坐标 `(n_faces, 3)`。

    同一 `(owner_cell, owner_cube_face)` 的记录（拆分四边形的两条子面）合并顶点后取平均；单条记录即三角形面，取
    三个顶点的平均。

    Args:
        owner_cell / owner_cube_face: (n_faces,) 面记录的 owner 单元与 owner 参考面编号
        face_node_ids: (n_faces, 3) 面记录的三个顶点（全局节点号）
        node_coords: (n_nodes, 3) 节点坐标
    """
    owner_cell = np.asarray(owner_cell, dtype=np.int64)
    owner_cube_face = np.asarray(owner_cube_face, dtype=np.int64)
    nodes = np.asarray(face_node_ids, dtype=np.int64)
    X = np.asarray(node_coords, dtype=np.float64)
    out = X[nodes].mean(axis=1)

    key = owner_cell * (int(owner_cube_face.max()) + 1) + owner_cube_face
    order = np.argsort(key, kind="stable")
    k_sorted = key[order]
    starts = np.flatnonzero(np.r_[True, k_sorted[1:] != k_sorted[:-1]])
    sizes = np.diff(np.r_[starts, key.size])
    if np.any(sizes > 2):
        raise ValueError(f"{int(np.count_nonzero(sizes > 2))} 个 (owner 单元, 参考面) 对应两条以上面记录")
    pairs = starts[sizes == 2]
    if pairs.size:
        a, b = order[pairs], order[pairs + 1]
        both = np.sort(np.concatenate([nodes[a], nodes[b]], axis=1), axis=1)   # (n_pairs, 6)
        repeated = np.zeros(both.shape, dtype=bool)
        repeated[:, 1:] = both[:, 1:] == both[:, :-1]
        n_unique = both.shape[1] - repeated.sum(axis=1)
        if np.any(n_unique != 4):
            raise ValueError(f"{int(np.count_nonzero(n_unique != 4))} 个拆分四边形的两条子面合起来不是 4 个顶点")
        quad_center = np.einsum('pk,pkd->pd', ~repeated, X[both]) / 4.0
        out[a] = quad_center
        out[b] = quad_center
    return out


def flux_point_positions(mesh, flat, faces, fp) -> np.ndarray:
    """面记录 `faces[k]` 第 `fp[k]` 个通量点的物理坐标 `(m, 3)`（owner 侧）。

    Args:
        mesh: 提供 `sps_coords (n_cells, n_sps, 3)`（单机网格或分布式紧凑空间适配器）
        flat: 面几何（与 `faces` 同一个索引空间）
    """
    faces = np.asarray(faces, dtype=np.int64)
    fp = np.asarray(fp, dtype=np.int64)
    X = np.asarray(mesh.sps_coords, dtype=np.float64)
    if X.shape[1] == 1:
        return np.asarray(flat.face_point)[faces]
    E = np.asarray(flat.boundary_extrap_native)[np.asarray(flat.owner_face_op)[faces], fp]   # (m, n_sps)
    return np.einsum('ks,ksd->kd', E, X[np.asarray(flat.owner_cell)[faces]])
