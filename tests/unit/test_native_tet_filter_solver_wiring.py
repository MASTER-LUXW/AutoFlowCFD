"""AutoFlowCFD V2.0 - native 四面体（路径C）模态滤波器接入求解器主循环
（`core/fr_solver/filter.py::build_filter_func`）决定性验证。
"""

from types import SimpleNamespace

import numpy as np
import pytest

from autoflowcfd.core.fr_solver.filter import build_filter_func
from autoflowcfd.fr.native_simplex_basis import build_native_tet_operators

from .test_native_tet_mesh_geometry_wiring import _build_synthetic_mixed_mesh


@pytest.mark.parametrize("order", [1, 2, 3])
def test_native_filter_freezes_padding_rows_and_preserves_constant_field(order):
    mesh = _build_synthetic_mixed_mesh(order, "native")
    n_prisms = mesh.n_prism_cells
    ref_native, _ = build_native_tet_operators(order)

    # **显式在 `sensor` 档下**构造算子与回调：本测试钉的是"滤波器真的
    # 施加时填充行不被改写"，需要一个非恒等矩阵。默认档已于 2026-09-19
    # 改成 `off`（恒等 -> `build_filter_func` 返回 None，见
    # `fr/modal_filter.py` 里"默认值 2026-09-19 改为 off"那一节），在默认
    # 档下这条契约没有可测对象 —— 显式指定档位比 skip 强：契约仍然被执行。
    from ._filter_mode import (
        reload_filter_modules,
        restore_default_filter_modules,
    )

    _, ops_mod = reload_filter_modules(AFCFD_FILTER_MODE="sensor")
    try:
        ops = ops_mod.generate_fr_operators(order)
        solver = SimpleNamespace(mesh=mesh, ops=ops)
        filter_func = build_filter_func(solver)
        assert filter_func is not None, (
            "sensor 档下滤波矩阵应当非恒等、build_filter_func 不该返回 None")
        _run_padding_freeze_checks(mesh, order, filter_func,
                                   ref_native, n_prisms)
    finally:
        restore_default_filter_modules()
    return


def _run_padding_freeze_checks(mesh, order, filter_func, ref_native,
                               n_prisms):
    n_native = ref_native.shape[0]

    n_cells, n_sps = mesh.n_cells, mesh.n_sps_per_cell
    n_vars = 7  # 5 欧拉变量 + k, omega（滤波器只作用于前5个，见 filter.py 文档）
    rng = np.random.default_rng(11)

    # 常数场（含真实行与填充行）：滤波后必须保持不变（自由流场保持性）
    U_flat = np.zeros((n_cells * n_sps, n_vars))
    U = U_flat.reshape(n_cells, n_sps, n_vars)
    U[:, :, 0] = 1.225
    U[:, :, 1] = 36.75
    U[:, :, 4] = 2.5e5
    U_filtered = filter_func(U_flat.copy()).reshape(n_cells, n_sps, n_vars)
    np.testing.assert_allclose(U_filtered, U, atol=1e-8)

    # 非常数场：填充行必须原样通过（不被滤波器改写成别的值），真实行
    # 可以变化（滤波器本来就会改变高阶模态内容）。
    U2 = np.zeros((n_cells, n_sps, n_vars))
    U2[:, :, 0] = 1.2 + 0.05 * rng.standard_normal((n_cells, n_sps))
    U2[:, :, 1] = 30.0 + rng.standard_normal((n_cells, n_sps))
    U2[:, :, 4] = 1.0e5 + 1e3 * rng.standard_normal((n_cells, n_sps))
    U2_flat = U2.reshape(n_cells * n_sps, n_vars).copy()
    U2_filtered = filter_func(U2_flat).reshape(n_cells, n_sps, n_vars)

    n_tets = n_cells - n_prisms
    for i in range(n_tets):
        cell_id = n_prisms + i
        np.testing.assert_allclose(
            U2_filtered[cell_id, n_native:, :5], U2[cell_id, n_native:, :5], atol=1e-12,
            err_msg=f"order={order}, cell {cell_id}: 填充行应在滤波后原样不变",
        )
    # 湍流量（索引5,6）完全不参与滤波，任何单元任何行都应逐位不变
    np.testing.assert_array_equal(U2_filtered[:, :, 5:], U2[:, :, 5:])
