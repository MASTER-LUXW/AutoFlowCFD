"""`build_face_connectivity` 直接给出原生面编码，局部面反解向量化（I-124）。

此前先产出坍缩坐标编码（0~5）再由 `with_native_face_codes(prism_native=...)` 静态翻译，反解对每个面做 Python 循环
并逐个候选局部面构造集合比较。判据：与逐面集合比较的参照实现逐位相同。
"""

import numpy as np

from autoflowcfd.grid.connectivity.face_connectivity import (
    CUBE_FACE_CODES,
    NATIVE_PRISM_FACE_CODE_RANGE,
    _native_face_tables,
    build_face_connectivity,
)
from tests.unit.test_fr_residual_inviscid import _build_synthetic_mixed_mesh


def _reference_code(cell_nodes, face_nodes, table):
    face = set(int(n) for n in face_nodes)
    hits = [code for idx, code in table
            if (set(int(cell_nodes[i]) for i in idx) == face if len(idx) == 3
                else face.issubset(set(int(cell_nodes[i]) for i in idx)))]
    assert len(hits) == 1
    return hits[0]


def test_codes_match_the_per_face_set_reference_on_a_mixed_mesh():
    mesh = _build_synthetic_mixed_mesh(1)
    fc = mesh.face_connectivity
    n_prism = mesh.n_prism_cells
    tables = _native_face_tables()
    prism_conn, tet_conn = mesh._fixed_prism_conn, mesh._fixed_tet_conn

    def cell(c):
        return (prism_conn[c], tables["prism"]) if c < n_prism else (tet_conn[c - n_prism], tables["tet"])

    for f in range(fc.n_faces):
        nodes, table = cell(int(fc.owner_cell[f]))
        assert fc.owner_cube_face[f] == _reference_code(nodes, fc.face_node_ids[f], table)
        if fc.neighbor_cell[f] >= 0:
            nodes, table = cell(int(fc.neighbor_cell[f]))
            assert fc.neighbor_cube_face[f] == _reference_code(nodes, fc.face_node_ids[f], table)
        else:
            assert fc.neighbor_cube_face[f] == -1
    lo, hi = NATIVE_PRISM_FACE_CODE_RANGE
    assert np.all((fc.owner_cube_face[fc.owner_cell < n_prism] >= lo) & (fc.owner_cube_face[fc.owner_cell < n_prism] < hi))
    tet_codes = {CUBE_FACE_CODES[f"tet_native_v{k}"] for k in range(4)}
    assert set(fc.owner_cube_face[fc.owner_cell >= n_prism].tolist()) <= tet_codes


def test_shared_tet_pair_gets_native_codes_directly():
    nodes = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 1]], dtype=float)
    fc = build_face_connectivity(None, np.array([[0, 1, 2, 3], [1, 2, 3, 4]], dtype=np.int32), nodes)
    shared = np.flatnonzero(fc.neighbor_cell >= 0)
    assert shared.size == 1
    # 共享面 (1,2,3)：单元 0 排除顶点 0、单元 1 排除顶点 3
    assert {int(fc.owner_cube_face[shared[0]]), int(fc.neighbor_cube_face[shared[0]])} == \
        {CUBE_FACE_CODES["tet_native_v0"], CUBE_FACE_CODES["tet_native_v3"]}

