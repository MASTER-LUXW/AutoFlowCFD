"""P0 面几何只算一次（I-8，2026-10-19，`inviscid_p0.py::p0_face_geometry`）。

此前 CPU 的 P0 残差每次求值都重新提取面几何（含逐组 Python 去重循环），GPU 那边另按网格缓存一份；现在缓存在面通量点
数据上，CPU / GPU / 测试共用，返回只读数组（缓存不会被调用方改写）。
"""

from unittest.mock import patch

import numpy as np
import pytest

from autoflowcfd.boundary.fr_ghost_state import DefaultGhostProvider
from autoflowcfd.core.fr_operators.face_kernels import get_flat_face_geometry
from autoflowcfd.core.fr_residual import inviscid_p0
from autoflowcfd.core.fr_residual.inviscid import primitive_to_conserved
from tests.unit.test_fr_residual_inviscid import _build_synthetic_mixed_mesh


def test_geometry_is_built_once_and_read_only():
    mesh = _build_synthetic_mixed_mesh(0)
    ffp = mesh.face_flux_points
    first = inviscid_p0.p0_face_geometry(ffp)
    assert inviscid_p0.p0_face_geometry(ffp) is first
    for arr in first:
        assert not arr.flags.writeable
        with pytest.raises(ValueError):
            arr[0] = 0.0


def test_residual_evaluations_reuse_the_cached_geometry():
    mesh = _build_synthetic_mixed_mesh(0)
    Q = np.tile([1.2, 30.0, 1.0, -2.0, 1.0e5], (mesh.n_cells, 1, 1))
    U = primitive_to_conserved(Q)
    flat = get_flat_face_geometry(mesh, mesh.operators)
    with patch.object(inviscid_p0, "_build_p0_face_geometry", wraps=inviscid_p0._build_p0_face_geometry) as build:
        r1 = inviscid_p0.compute_inviscid_residual_fv_p0(U, mesh, flat, DefaultGhostProvider())
        r2 = inviscid_p0.compute_inviscid_residual_fv_p0(U, mesh, flat, DefaultGhostProvider())
    assert build.call_count <= 1                          # 网格构造期可能已经算过（=0）
    np.testing.assert_array_equal(r1, r2)


def test_non_kernel_face_data_is_refused():
    with pytest.raises(TypeError, match="_KernelFaceData"):
        inviscid_p0.p0_face_geometry([object()])
