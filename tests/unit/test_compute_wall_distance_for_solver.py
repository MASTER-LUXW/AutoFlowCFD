"""CLI 壁距入口（`cli/solve/wall_distance.py::compute_wall_distance_for_solver`）与壁面面片的取法
（`core/utils/wall_distance/surface.py`）。"""

from unittest.mock import MagicMock, patch

import click
import numpy as np
import pytest

from autoflowcfd.cli.solve.helpers import compute_wall_distance_for_solver
from autoflowcfd.core.utils.wall_distance import wall_face_nodes
from autoflowcfd.grid.structures import (
    BoundaryMap, GridMetadata, NodeArray, TetrahedralCells, VolumeMeshData,
)

_APPLY = "autoflowcfd.core.fr_solver.turbulence.apply_wall_distance_source"


def _volume_mesh_with_wall():
    """两个共享一个面的四面体，WALL 组 = **单元** 0（`BoundaryMap.groups` 存单元索引）。

    单元 0 = [0,1,2,3] 与单元 1 共享面 [1,2,3]，故单元 0 有 3 个边界面，节点并集恰是
    {0,1,2,3}；节点 4 只属于单元 1。
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


class TestCliEntry:
    def test_non_turbulent_model_skips_everything(self):
        solver = MagicMock()
        solver.turb_model_name = 'NONE'
        with patch(_APPLY) as mock_apply:
            compute_wall_distance_for_solver(solver, _volume_mesh_with_wall())
        mock_apply.assert_not_called()

    @pytest.mark.parametrize("model", ["SST", "SA", "DDES", "LES"])
    def test_every_wall_distance_model_gets_the_source(self, model):
        """判据取唯一的模型分类：此前入口写死 sst/ddes/iddes/wmles/les 一份列表，SA 会被
        静默跳过（随后在求解时因缺壁距报错）。"""
        solver = MagicMock()
        solver.turb_model_name = model
        with patch(_APPLY) as mock_apply:
            compute_wall_distance_for_solver(solver, _volume_mesh_with_wall())
        mock_apply.assert_called_once()

    def test_solver_receives_the_face_derived_surface(self):
        """施加到求解器上的来源由 WALL 组的边界面构造：壁面节点处距离为 0，节点 4 不在壁面上。"""
        solver = MagicMock()
        solver.turb_model_name = 'SST'
        vd = _volume_mesh_with_wall()
        with patch(_APPLY) as mock_apply:
            compute_wall_distance_for_solver(solver, vd)
        (_, source), _ = mock_apply.call_args
        assert source.n_wall_faces == 3
        nodes = vd.nodes.get_coordinates()
        assert np.all(source.query(nodes[:4]) == 0.0)
        # 节点 4 = (1,1,1)：三个壁面面片是 x=0、y=0、z=0 坐标面上的直角三角形，投影 (1,1,0) 等
        # 都落在三角形外，最近点是斜边中点（如 (0.5,0.5,0)），距离 sqrt(1.5)
        assert source.query(nodes[4:5])[0] == pytest.approx(np.sqrt(1.5), rel=1e-14)

    def test_out_of_range_cell_index_is_a_cli_error(self):
        vd = _volume_mesh_with_wall()
        vd.boundaries.groups['wall'] = np.array([0, 999], dtype=np.int32)
        solver = MagicMock()
        solver.turb_model_name = 'SST'
        with pytest.raises(click.ClickException, match='超出体网格单元数'):
            compute_wall_distance_for_solver(solver, vd)


class TestWallFacesComeFromBoundaryFaces:
    """壁面取自 WALL 组的**边界面**（2026-09-15 一阶物理错误修复：此前用 `max(indices) >=
    n_nodes` 猜 `BoundaryMap.groups` 存的是单元还是节点索引，恰好只在 WALL 组上猜错——
    plate_demo 壁距 max 0.976 m，真实 4.359 m）。"""

    def test_wall_faces_are_the_wall_cells_boundary_faces(self):
        fn = wall_face_nodes(_volume_mesh_with_wall())
        assert fn.shape[0] == 3
        assert sorted(np.unique(fn[fn >= 0]).tolist()) == [0, 1, 2, 3]

    def test_cell_index_out_of_range_is_rejected_not_truncated(self):
        vd = _volume_mesh_with_wall()
        vd.boundaries.groups['wall'] = np.array([0, 999], dtype=np.int32)
        with pytest.raises(ValueError, match='超出体网格单元数'):
            wall_face_nodes(vd)

    def test_non_wall_groups_are_excluded(self):
        """只有 bc_type == 'WALL' 的组参与壁距；SLIP_WALL（外场/风洞壁）必须排除。"""
        vd = _volume_mesh_with_wall()
        vd.boundaries.groups['tunnel'] = np.array([1], dtype=np.int32)
        vd.boundaries.bc_types['tunnel'] = 'SLIP_WALL'
        fn = wall_face_nodes(vd)
        assert fn.shape[0] == 3 and 4 not in fn.tolist()

    def test_no_wall_group_returns_empty(self):
        vd = _volume_mesh_with_wall()
        vd.boundaries.bc_types['wall'] = 'SLIP_WALL'
        assert wall_face_nodes(vd).shape[0] == 0

    def test_misnamed_accessor_is_gone(self):
        from autoflowcfd.grid.structures import BoundaryMap as BM
        assert not hasattr(BM, 'get_node_indices')
        assert hasattr(BM, 'get_cell_indices')
