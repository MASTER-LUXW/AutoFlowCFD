"""展平面几何缓存挂在面通量点数据上（I-21，2026-10-19，`face_kernels.py::get_flat_face_geometry`）。

此前是模块级单槽全局缓存：网格释放后仍被强引用（展平几何连同面通量点数据滞留内存），两张网格交替求值时每次重建。
"""

import gc
import weakref
from unittest.mock import patch

from autoflowcfd.core.fr_operators import face_kernels
from tests.unit.test_fr_residual_inviscid import _build_synthetic_mixed_mesh


def test_geometry_is_released_with_its_mesh():
    mesh = _build_synthetic_mixed_mesh(1)
    flat_ref = weakref.ref(face_kernels.get_flat_face_geometry(mesh, mesh.operators))
    del mesh
    gc.collect()
    assert flat_ref() is None


def test_alternating_meshes_do_not_rebuild():
    a, b = _build_synthetic_mixed_mesh(1), _build_synthetic_mixed_mesh(1)
    fa = face_kernels.get_flat_face_geometry(a, a.operators)       # 网格构造期可能已经建好
    fb = face_kernels.get_flat_face_geometry(b, b.operators)
    assert fa is not fb
    with patch.object(face_kernels, "build_flat_face_geometry", wraps=face_kernels.build_flat_face_geometry) as build:
        for _ in range(3):
            assert face_kernels.get_flat_face_geometry(a, a.operators) is fa
            assert face_kernels.get_flat_face_geometry(b, b.operators) is fb
    assert build.call_count == 0                                    # 交替求值不再重建


def test_new_face_flux_point_data_gets_its_own_geometry():
    """换阶时 `mesh.face_flux_points` 被整体替换：新对象没有缓存、重建（不复用上一阶的几何）。"""
    mesh = _build_synthetic_mixed_mesh(1)
    f1 = face_kernels.get_flat_face_geometry(mesh, mesh.operators)
    mesh.set_order(2)
    f2 = face_kernels.get_flat_face_geometry(mesh, mesh.operators)
    assert f2 is not f1 and f2.n_fp != f1.n_fp
