"""原生四面体/棱柱几何构造必须拒绝非正 Jacobian。

此前原生分支（生产路径）没有任何 det>0 检查：退化四面体得到 det=0、`inv = adj/0` 的无穷大度量，
扭曲棱柱在部分解点上 det<0，网格照常构造完成，日志还写着"all Jacobians verified positive"。
"""

from types import SimpleNamespace

import numpy as np
import pytest

from autoflowcfd.grid.curved_mapping.curved_mapping import MeshDistortionError
from autoflowcfd.grid.high_order.high_order_mesh import HighOrderMesh
from tests.unit.test_fr_residual_inviscid import _MockCells, _MockNodes


def _load(nodes, tet_conn, prism_conn, order):
    mock = SimpleNamespace(
        cell_count=len(tet_conn) + len(prism_conn),
        nodes=_MockNodes(np.asarray(nodes, dtype=float)),
        cells=_MockCells(np.asarray(tet_conn, dtype=np.int32).reshape(-1, 4)),
        prism_cells=_MockCells(np.asarray(prism_conn, dtype=np.int32).reshape(-1, 6)),
    )
    mesh = HighOrderMesh(order=order)
    mesh.load_from_volume_mesh(mock)
    return mesh


GOOD_PRISM_NODES = [[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 0, 1], [0, 1, 1]]


@pytest.mark.parametrize("order", [1, 2])
def test_valid_cells_load(order):
    nodes = GOOD_PRISM_NODES + [[5, 0, 0], [6, 0, 0], [5, 1, 0], [5, 0, 1]]
    mesh = _load(nodes, [[6, 7, 8, 9]], [[0, 1, 2, 3, 4, 5]], order)
    assert (mesh.jacobians["det_jacs"] > 0).all()


@pytest.mark.parametrize("order", [1, 2])
def test_degenerate_tet_is_rejected(order):
    # 第 4 个顶点落在底面所在平面上：体积为零
    nodes = GOOD_PRISM_NODES + [[5, 0, 0], [6, 0, 0], [5, 1, 0], [5.3, 0.3, 0.0]]
    with pytest.raises(MeshDistortionError, match="四面体"):
        _load(nodes, [[6, 7, 8, 9]], [[0, 1, 2, 3, 4, 5]], order)


@pytest.mark.parametrize("order", [1, 2])
def test_twisted_prism_is_rejected(order):
    # 顶面相对底面绕竖轴转过约 180 度：映射在单元内部翻转（整体朝向修正救不回来）
    nodes = [[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0.2], [0, 1, 0.2], [1, 0, 0.2],
             [5, 0, 0], [6, 0, 0], [5, 1, 0], [5, 0, 1]]
    with pytest.raises(MeshDistortionError, match="棱柱"):
        _load(nodes, [[6, 7, 8, 9]], [[0, 1, 2, 3, 4, 5]], order)
