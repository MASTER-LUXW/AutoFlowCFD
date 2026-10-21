"""网格导入中的退化几何与无法使用的卡片报错，不再静默删除（I-129 / I-131 / I-133）。

此前：面积 <= 0 的面被 FaceData 静默删除；NAS 解析最多容忍 5% 的元素 / 节点行被丢弃；导入体网格时体积
< 1e-20 m^3（绝对阈值）或重定向后仍为负的 CTETRA / CPENTA 被删除——网格因此带洞、拓扑不封闭，下游在缺面 / 缺
单元的网格上继续。
"""

import numpy as np
import pytest

from autoflowcfd.grid.nas_io.nas_parser_exceptions import NASParseError
from autoflowcfd.grid.schema.grid_faces import FaceData


def test_non_positive_face_area_is_an_error():
    with pytest.raises(ValueError, match="面积不为正"):
        FaceData(connectivity=np.array([[0, -1], [0, 1]], dtype=np.int32), area=np.array([1.0, 0.0]),
                 normal=np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]), center=np.zeros((2, 3)))


def test_element_referencing_a_missing_node_is_an_error(tmp_path):
    from autoflowcfd.grid.nas_io.parser_core import NASParser

    path = tmp_path / "dangling.nas"
    path.write_text("GRID,1,,0.0,0.0,0.0\nGRID,2,,1.0,0.0,0.0\nGRID,3,,0.0,1.0,0.0\n"
                    "CTRIA3,1,1,1,2,3\nCTRIA3,2,1,1,2,9\n")
    with pytest.raises(NASParseError, match="引用不存在的节点 1.*第 5 行"):
        NASParser(str(path)).parse()


def _volume_file(tmp_path, tet_line):
    lines = ["GRID    {:>8}       0{:>8}{:>8}{:>8}\n".format(i + 1, *map(str, p))
             for i, p in enumerate([(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0),
                                    (0.5, 0.5, 0.0)])]
    path = tmp_path / "vol.nas"
    path.write_text("".join(lines) + "CTETRA         1       1       1       2       3       4\n" + tet_line)
    return path


def test_degenerate_tetrahedron_is_an_error_with_its_card(tmp_path):
    from autoflowcfd.grid.nas_io.nas_parser_volume import parse_volume_mesh_nas

    # 第二个四面体四点共面（节点 5 在 1-2-3 平面上）
    path = _volume_file(tmp_path, "CTETRA         2       1       1       2       3       5\n")
    with pytest.raises(ValueError, match="1 个 CTETRA 单元退化.*第 2 张 CTETRA"):
        parse_volume_mesh_nas(str(path), units="m")


def test_relative_criterion_accepts_a_tiny_but_well_shaped_cell(tmp_path):
    """判据相对单元自身尺度：微米量级的正常单元不被当成退化（此前的绝对阈值 1e-20 m^3 会误判）。"""
    from autoflowcfd.grid.nas_io.nas_parser_volume import _require_valid_cells

    h = 1e-7
    xyz = np.array([[0, 0, 0], [h, 0, 0], [0, h, 0], [0, 0, h]], dtype=float)
    _require_valid_cells("x", "CTETRA", xyz, np.array([[0, 1, 2, 3]]), np.array([[1, 2, 3, 4]]))
