"""Unit tests for cli/solve_helpers.compute_wall_distance_for_solver's
use_eikonal handling - specifically that it now actually builds a node
adjacency graph (grid.node_connectivity.build_node_adjacency) and forwards
it to solver.compute_wall_distance_field, instead of the previous dead
branch that printed a "falling back to KD-Tree" warning and did exactly
that regardless of the flag.
"""

from unittest.mock import MagicMock, patch

import numpy as np

from autoflowcfd.cli.solve.helpers import compute_wall_distance_for_solver
from autoflowcfd.grid.structures import (
    BoundaryMap, GridMetadata, NodeArray, TetrahedralCells, VolumeMeshData,
)


def _volume_mesh_with_wall():
    """两个共享一个面的四面体，WALL 组 = **单元** 0。

    2026-09-15 更正：原 fixture 写的是 `groups={'wall': [0, 1]}` 配一个
    只有 1 个单元、4 个节点的网格——那两个数当时被当作**节点**索引，正是
    `BoundaryMap.groups` 契约（存**单元**索引，见该类 `groups` 字段文档）
    被误读的体现，而这个误读在生产代码里造成了一处一阶物理错误（壁面
    距离场算到一堆按编号散布在全域的任意节点上，见
    `solve_wall_distance.wall_nodes_from_boundary_faces` 的完整推导）。
    现在 fixture 按契约给单元索引，并且单元数 > 1 以便 WALL 组有真实的
    边界面可取。
    """
    nodes = NodeArray(
        x=np.array([0.0, 1.0, 0.0, 0.0, 1.0]),
        y=np.array([0.0, 0.0, 1.0, 0.0, 1.0]),
        z=np.array([0.0, 0.0, 0.0, 1.0, 1.0]),
    )
    cells = TetrahedralCells(
        connectivity=np.array([[0, 1, 2, 3], [1, 2, 3, 4]], dtype=np.int32),
        volumes=np.array([1.0 / 6.0, 1.0 / 6.0]),
    )
    boundaries = BoundaryMap(groups={'wall': np.array([0], dtype=np.int32)},
                             bc_types={'wall': 'WALL'})
    metadata = GridMetadata(node_count=5, cell_count=2,
                            boundary_groups=['wall'], file_format='test')
    return VolumeMeshData(nodes=nodes, cells=cells, boundaries=boundaries,
                          metadata=metadata)


class TestComputeWallDistanceForSolverEikonalWiring:
    def test_non_turbulent_model_skips_everything(self):
        solver = MagicMock()
        solver.turb_model_name = 'NONE'
        compute_wall_distance_for_solver(solver, _volume_mesh_with_wall(), use_eikonal=True)
        solver.compute_wall_distance_field.assert_not_called()

    def test_use_eikonal_false_does_not_build_connectivity(self):
        """The adjacency graph has a real construction cost - it must only
        be built when actually needed."""
        solver = MagicMock()
        solver.turb_model_name = 'SST'
        with patch("autoflowcfd.grid.connectivity.node_connectivity.build_node_adjacency") as mock_build:
            compute_wall_distance_for_solver(solver, _volume_mesh_with_wall(), use_eikonal=False)
        mock_build.assert_not_called()
        solver.compute_wall_distance_field.assert_called_once()
        _, kwargs = solver.compute_wall_distance_field.call_args
        assert kwargs == {}  # positional-only call, no connectivity/use_eikonal kwargs on this path

    def test_use_eikonal_true_builds_and_forwards_connectivity(self):
        solver = MagicMock()
        solver.turb_model_name = 'DDES'
        compute_wall_distance_for_solver(solver, _volume_mesh_with_wall(), use_eikonal=True)

        solver.compute_wall_distance_field.assert_called_once()
        args, kwargs = solver.compute_wall_distance_field.call_args
        assert kwargs["use_eikonal"] is True
        connectivity = kwargs["connectivity"]
        assert connectivity.shape[0] == 5  # one row per node
        # node 0 and node 1 share the single tet - must be neighbors.
        assert 1 in connectivity[0]
        assert 0 in connectivity[1]


class TestWallNodesComeFromBoundaryFacesNotIndexGuessing:
    """壁面节点必须取自 WALL 组的**边界面**（2026-09-15 一阶物理错误修复）。

    原实现用 `max(indices) >= n_nodes` 猜 `BoundaryMap.groups` 存的是单元
    还是节点索引。它按契约恒为**单元**索引，所以那是在猜一件已经确定的
    事——而且**恰好只在 WALL 组上猜错**：壁面的边界单元就是边界层棱柱、
    占单元索引低位区间 `[0, n_prism)`，而两张真实 ANSA 网格都满足
    `n_prism < n_nodes`：

        cube_demo : body  range=[0,136974]  n_nodes=187702  -> 猜错
        plate_demo: body  range=[0, 65235]  n_nodes= 88496  -> 猜错

    实测后果（plate_demo）：壁距 max 0.976 m，而到平板的真实最远距离是
    4.359 m；修复后逐位吻合独立 KD-Tree 核算的 4.359498 m。SST 的 F1/F2
    混合、omega 壁面目标值、nu_t 限幅、DES 长度尺度与 WMLES 全部由壁距
    驱动，所以这不是精度问题。
    """

    def test_wall_nodes_are_boundary_face_nodes(self):
        from autoflowcfd.cli.solve.wall_distance import wall_nodes_from_boundary_faces
        vd = _volume_mesh_with_wall()
        wn, n_faces = wall_nodes_from_boundary_faces(vd, vd.boundaries)
        # 单元 0 = [0,1,2,3]，与单元 1 共享面 [1,2,3]，故有 3 个边界面，
        # 其节点并集恰是单元 0 的四个节点。
        assert n_faces == 3
        assert sorted(wn.tolist()) == [0, 1, 2, 3]

    def test_old_heuristic_would_have_picked_the_wrong_set(self):
        """fail 半边：复现原判据在这个拓扑上的错误结果。

        WALL 组是 `[0]`，`max(0) < n_nodes=5`，所以原实现会走 else 分支、
        把 `[0]` 当**节点**索引——壁面节点集只剩 {0}，而正确答案是
        {0,1,2,3}。
        """
        vd = _volume_mesh_with_wall()
        idx = vd.boundaries.get_cell_indices('wall')
        n_nodes = vd.node_count
        assert idx.max() < n_nodes, "本用例必须命中原判据的错误分支"
        old_result = sorted(idx[idx < n_nodes].tolist())
        assert old_result == [0], "原判据会把单元索引当节点索引"

        from autoflowcfd.cli.solve.wall_distance import wall_nodes_from_boundary_faces
        new_result = sorted(wall_nodes_from_boundary_faces(vd, vd.boundaries)[0].tolist())
        assert new_result != old_result
        assert new_result == [0, 1, 2, 3]

    def test_cell_index_out_of_range_is_rejected_not_truncated(self):
        """BoundaryMap 与体网格不是一对时必须显式报错。

        原实现对越界索引的处理是静默过滤（`indices[indices < n_nodes]`
        / `if cell_idx < len(all_connectivity)`），于是"两份文件对不上"
        会表现为一个悄悄少了一部分壁面的距离场。
        """
        import click
        import pytest

        from autoflowcfd.cli.solve.wall_distance import wall_nodes_from_boundary_faces
        vd = _volume_mesh_with_wall()
        vd.boundaries.groups['wall'] = np.array([0, 999], dtype=np.int32)
        with pytest.raises(click.ClickException, match='超出体网格单元数'):
            wall_nodes_from_boundary_faces(vd, vd.boundaries)

    def test_non_wall_groups_are_excluded(self):
        """只有 bc_type == 'WALL' 的组参与壁距；SLIP_WALL（外场/风洞壁）
        必须排除——否则外场壁会把全域的壁距压到域半高量级。"""
        from autoflowcfd.cli.solve.wall_distance import wall_nodes_from_boundary_faces
        vd = _volume_mesh_with_wall()
        vd.boundaries.groups['tunnel'] = np.array([1], dtype=np.int32)
        vd.boundaries.bc_types['tunnel'] = 'SLIP_WALL'
        wn, n_faces = wall_nodes_from_boundary_faces(vd, vd.boundaries)
        assert n_faces == 3
        assert sorted(wn.tolist()) == [0, 1, 2, 3]
        assert 4 not in wn.tolist()   # 节点 4 只属于 tunnel 那个单元

    def test_no_wall_group_returns_empty(self):
        """没有 WALL 组时返回空集，由调用方决定报错——不在这里兜底。"""
        from autoflowcfd.cli.solve.wall_distance import wall_nodes_from_boundary_faces
        vd = _volume_mesh_with_wall()
        vd.boundaries.bc_types['wall'] = 'SLIP_WALL'
        wn, n_faces = wall_nodes_from_boundary_faces(vd, vd.boundaries)
        assert len(wn) == 0 and n_faces == 0

    def test_misnamed_accessor_is_gone(self):
        """`get_node_indices` 这个错名别名必须不再存在。

        它返回的恒是单元索引，名字却让调用方以为是节点索引——这正是本次
        缺陷的直接成因。需要单元索引用 `get_cell_indices`，需要壁面节点用
        `wall_nodes_from_boundary_faces`。
        """
        from autoflowcfd.grid.structures import BoundaryMap as BM
        assert not hasattr(BM, 'get_node_indices')
        assert hasattr(BM, 'get_cell_indices')

    def test_solver_receives_the_face_derived_set(self):
        """端到端：`compute_wall_distance_for_solver` 传给求解器的必须是
        面导出的节点集。"""
        solver = MagicMock()
        solver.turb_model_name = 'SST'
        vd = _volume_mesh_with_wall()
        compute_wall_distance_for_solver(solver, vd, use_eikonal=False)
        solver.compute_wall_distance_field.assert_called_once()
        args, _ = solver.compute_wall_distance_field.call_args
        mesh_nodes, wall_indices = args[0], args[1]
        assert mesh_nodes.shape == (5, 3)
        assert sorted(np.asarray(wall_indices).tolist()) == [0, 1, 2, 3]
