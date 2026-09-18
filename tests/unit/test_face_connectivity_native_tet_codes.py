"""AutoFlowCFD V2.0 - `FRFaceConnectivity.with_native_face_codes` 单元测试
（Part7 阶段2生产接入架构设计，第一节：面标识符推广）。
"""

import numpy as np

from autoflowcfd.grid.connectivity.face_connectivity import (
    CUBE_FACE_CODES,
    build_face_connectivity,
)


def _shared_tet_pair():
    """2 个共享面的四面体（与 test_fr_residual_inviscid.py 前 2 个单元
    同一个构造），零棱柱。"""
    nodes = np.array(
        [[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 1]],
        dtype=float,
    )
    tet_conn = np.array([[0, 1, 2, 3], [1, 2, 3, 4]], dtype=np.int32)
    return build_face_connectivity(prism_connectivity=None, tet_connectivity=tet_conn, nodes=nodes)


def test_with_native_face_codes_translates_only_valid_tet_codes():
    fc = _shared_tet_pair()
    n_prism = 0
    native_fc = fc.with_native_face_codes(n_prism)

    native_codes = {CUBE_FACE_CODES[f"tet_native_v{k}"] for k in range(4)}
    collapsed_tet_codes = {CUBE_FACE_CODES[k] for k in ("a=-1", "a=+1", "b=-1", "c=-1")}

    for arr_name in ("owner_cube_face", "neighbor_cube_face"):
        orig = getattr(fc, arr_name)
        translated = getattr(native_fc, arr_name)
        for i in range(len(orig)):
            if orig[i] == -1:
                assert translated[i] == -1
            elif orig[i] in collapsed_tet_codes:
                assert translated[i] in native_codes, f"{arr_name}[{i}]={orig[i]} 应该被翻译成 native 编码"
            else:
                assert translated[i] == orig[i]


def test_with_native_face_codes_preserves_geometry_fields_unchanged():
    fc = _shared_tet_pair()
    native_fc = fc.with_native_face_codes(0)
    np.testing.assert_array_equal(native_fc.owner_cell, fc.owner_cell)
    np.testing.assert_array_equal(native_fc.neighbor_cell, fc.neighbor_cell)
    np.testing.assert_array_equal(native_fc.normal, fc.normal)
    np.testing.assert_array_equal(native_fc.area, fc.area)
    np.testing.assert_array_equal(native_fc.center, fc.center)
    np.testing.assert_array_equal(native_fc.is_boundary, fc.is_boundary)


def test_with_native_face_codes_does_not_mutate_original():
    fc = _shared_tet_pair()
    original_owner = fc.owner_cube_face.copy()
    _ = fc.with_native_face_codes(0)
    np.testing.assert_array_equal(fc.owner_cube_face, original_owner)
