"""mesh_background_merge._build_merged_mesh 用到的两个独立小工具。

从 mesh_background_merge.py 拆分出来：`_refine_large_boundary_faces`
（tetgen 之前按最大边长迭代二分边界面的过大三角形，避免生成过大的边界四面体）和
`_export_partial_mesh_and_exit`（`--bl-only`/`--core-only` 等调试导出
路径共用的导出后直接退出进程的辅助函数）。两者都只被 _build_merged_mesh
自己调用，没有独立复用需求，纯粹为了控制文件行数而拆开。
"""

import sys
from typing import Tuple

import numpy as np
from loguru import logger


# 每轮把超长边对半，最长边 L 需要约 log2(L/h) 轮；64 轮对应 2^64 的尺度比，只是防死循环的护栏
_MAX_REFINE_ITERATIONS = 64
_MAX_REFINED_BOUNDARY_VERTICES = 20_000_000


def _split_face(v, m):
    """一个三角形按被标记的边拆分（共形：被拆边的中点由相邻面共享）。

    v: 三个顶点 (v0, v1, v2)；m: 三条边 (v0v1, v1v2, v2v0) 的中点索引（未标记为 -1）。返回保持原绕向的子三角形。
    """
    marked = [k for k in range(3) if m[k] >= 0]
    if len(marked) == 3:
        m01, m12, m20 = m
        return [(v[0], m01, m20), (m01, v[1], m12), (m20, m12, v[2]), (m01, m12, m20)]
    if len(marked) == 1:
        k = marked[0]                                   # 旋转到被标记边是 v0v1
        a, b, c = v[k], v[(k + 1) % 3], v[(k + 2) % 3]
        mid = m[k]
        return [(a, mid, c), (mid, b, c)]
    k = [j for j in range(3) if m[j] < 0][0]            # 两条被标记：旋转到未标记边是 v2v0
    a, b, c = v[(k + 1) % 3], v[(k + 2) % 3], v[k]
    m0, m1 = m[(k + 1) % 3], m[(k + 2) % 3]             # m0 在 ab 上、m1 在 bc 上
    return [(m0, b, m1), (a, m0, m1), (a, m1, c)]


def _refine_large_boundary_faces(
    vertices: np.ndarray,
    faces: np.ndarray,
    max_edge_length: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """共形地细分边界表面，直到没有边超过 max_edge_length（防止 TetGen 生成违反目标尺寸的巨大边界四面体）。

    每轮把超长边各插一个中点（相邻面共享同一个中点），每个面按自己被标记的边数拆成 2/3/4 个子三角形——
    不产生悬挂节点。2026-10-09 以前每个面只二分自己的最长边、各自新建中点：相邻面不在同一点剖分（悬挂节点），
    两个面都拆同一条边时还会出现两个重合的中点，交给 tetgen 的表面不封闭。
    """
    verts = np.asarray(vertices, dtype=np.float64)
    cur = np.asarray(faces, dtype=np.int64)
    if max_edge_length <= 0:
        return verts, cur
    # 细分后的规模事先按表面积估计（边长 h 的正三角形面积 sqrt(3)/4 h^2，顶点数约为面数的一半）；
    # 规模不可接受时明确报错，而不是细分到一半悄悄停下、把超长边留给 tetgen
    a, b, c = verts[cur[:, 0]], verts[cur[:, 1]], verts[cur[:, 2]]
    area = 0.5 * float(np.linalg.norm(np.cross(b - a, c - a), axis=1).sum())
    estimated_vertices = int(area / (np.sqrt(3.0) / 4.0 * max_edge_length ** 2) / 2)
    if estimated_vertices > _MAX_REFINED_BOUNDARY_VERTICES:
        raise ValueError(
            f"边界面按 max_cell_size={max_edge_length:g} m 细分后约 {estimated_vertices:,} 个顶点，超过上限 "
            f"{_MAX_REFINED_BOUNDARY_VERTICES:,}：max_cell_size 相对这张表面太小")

    for iteration in range(_MAX_REFINE_ITERATIONS):
        edges = np.stack([cur[:, [0, 1]], cur[:, [1, 2]], cur[:, [2, 0]]], axis=1)      # (n, 3, 2)
        key = np.sort(edges, axis=2).reshape(-1, 2)
        uniq, inverse = np.unique(key, axis=0, return_inverse=True)
        length = np.linalg.norm(verts[uniq[:, 0]] - verts[uniq[:, 1]], axis=1)
        long_edge = np.flatnonzero(length > max_edge_length)
        if len(long_edge) == 0:
            break
        mid_of_edge = np.full(len(uniq), -1, dtype=np.int64)
        mid_of_edge[long_edge] = len(verts) + np.arange(len(long_edge))
        verts = np.vstack([verts, 0.5 * (verts[uniq[long_edge, 0]] + verts[uniq[long_edge, 1]])])
        mids = mid_of_edge[inverse.ravel()].reshape(-1, 3)
        split = (mids >= 0).any(axis=1)
        new_faces = [tri for f, m in zip(cur[split].tolist(), mids[split].tolist()) for tri in _split_face(f, m)]
        cur = np.vstack([cur[~split], np.array(new_faces, dtype=np.int64)])
        logger.info(f"  Refinement iteration {iteration + 1}: split {len(long_edge)} edges, "
                    f"{int(split.sum())} faces -> {len(new_faces)}")

    else:
        raise RuntimeError(f"边界面细分 {_MAX_REFINE_ITERATIONS} 轮后仍有超过 {max_edge_length:g} m 的边")
    logger.info(f"Boundary refinement: faces {len(faces)} -> {len(cur)}, vertices {len(vertices)} -> {len(verts)}")
    return verts, cur.astype(np.int32)


def _export_partial_mesh_and_exit(
    nodes: np.ndarray,
    prism_cells: np.ndarray,
    tet_cells: np.ndarray,
    output_path: str,
    label: str,
    surface_nodes: np.ndarray,
    surface_faces: np.ndarray,
    surface_boundaries,
) -> None:
    """导出局部（仅 BL / 仅过渡层 / 仅核心）调试网格并退出进程。

    所有 `--*-only` CLI 标志的早期停止路径共用此函数
    （参见 cli/grid/commands.py 的 `--bl-only`/`--trans-only`/`--core-only`）。
    这些标志用于在网格查看器（ANSA 等）中直接检查管线各阶段的生成结果——
    在调查 BL/过渡层到核心填充界面问题时反复需要此功能，但此前没有可复用的方式，
    每次都要写临时脚本。

    边界标注见 `mesh_boundary.label_partial_mesh_boundaries`：落在输入面网格上的外部面取所在面网格三角形
    的组，其余外部面（与未导出部分之间的界面）的所属单元归入 'INTERFACE'。

    Args:
        nodes: (n_nodes, 3) 节点坐标，单位：米
        prism_cells: (n_prism, 6) 棱柱连接关系，或空数组 (0, 6)
        tet_cells: (n_tet, 4) 四面体连接关系，或空数组 (0, 4)
        output_path: .nas 文件输出路径
        label: 当前阶段的易读名称，仅用于日志
        surface_nodes, surface_faces, surface_boundaries: 输入面网格（米制）及其边界分组
    """
    from ...nas_io.nas_export import export_volume_mesh_to_nas
    from ...structures import NodeArray, PrismCells, TetrahedralCells, GridMetadata, VolumeMeshData

    logger.success(f"Exporting {label} mesh to: {output_path}")
    try:
        nodes_obj = NodeArray.from_array(nodes)

        n_prism = len(prism_cells)
        prism_cells_obj = None
        if n_prism:
            prism_volumes = PrismCells.compute_volumes(nodes_obj, prism_cells.astype(np.int32))
            prism_cells_obj = PrismCells(connectivity=prism_cells.astype(np.int32), volumes=prism_volumes)

        tet_cells32 = tet_cells.astype(np.int32)
        tet_volumes = (
            TetrahedralCells.compute_volumes(nodes_obj, tet_cells32)
            if len(tet_cells32) else np.empty(0, dtype=np.float64)
        )
        cells_obj = TetrahedralCells(connectivity=tet_cells32, volumes=tet_volumes)

        from ..utils.mesh_boundary import label_partial_mesh_boundaries
        boundaries_obj = label_partial_mesh_boundaries(
            nodes, prism_cells, tet_cells, surface_nodes, surface_faces, surface_boundaries)
        groups = boundaries_obj.groups
        metadata = GridMetadata(
            node_count=len(nodes), cell_count=n_prism + len(tet_cells32),
            boundary_groups=list(groups.keys()), file_format="partial",
        )
        vol_mesh = VolumeMeshData(
            nodes=nodes_obj, cells=cells_obj, boundaries=boundaries_obj,
            metadata=metadata, prism_cells=prism_cells_obj,
            surface_mesh={'nodes': surface_nodes, 'faces': surface_faces, 'boundaries': surface_boundaries},
        )
        # export_volume_mesh_to_nas 期望输入为米，内部会转换为毫米
        export_volume_mesh_to_nas(vol_mesh, output_path, scale_factor=1000.0)
        logger.success(f"{label} mesh exported successfully.")
    except Exception as e:
        logger.error(f"Failed to export {label} mesh: {e}")
        import traceback
        traceback.print_exc()
        # 导出失败必须以非零退出码报告——此前这里无条件 sys.exit(0)，
        # 靠退出码判断成败的调用方会把失败误判为成功（第四次评审发现1）。
        sys.exit(1)

    sys.exit(0)
