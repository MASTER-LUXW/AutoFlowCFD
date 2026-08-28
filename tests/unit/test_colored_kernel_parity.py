"""无粘/粘性界面 kernel 的着色版 vs 非着色版一致性回归测试。

V2.0 专家组盲审发现：`inviscid_kernel.py`/`inviscid_kernel_colored.py`、
`viscous_flux_kernel.py`/`viscous_flux_kernel_colored.py` 各自的核心循环体
几乎逐字重复（着色版用图着色 scatter-add 消除并行写冲突，非着色版用
per-thread 私有缓冲区+归约），代码注释里也明确写着"两处必须同步修改"——
这是真实的维护风险，一旦某次只改了其中一侧就会静默产生分叉，且没有任何
自动化机制会发现。

本测试不合并/重构两套实现（有炸出新 bug 的风险，评审本身也建议暂不重构），
而是用一个非均匀（非平凡）状态场，通过 `AFCFD_USE_COLORING` 环境变量
分别驱动同一个 `compute_inviscid_residual_fr`/`compute_viscous_residual`
入口走着色/非着色两条路径，断言残差数值一致——把"人工同步维护"的风险
转成自动化检测：未来任何一侧被单独修改导致分叉，这个测试会立刻失败。
"""

import os
from types import SimpleNamespace

import numpy as np
import pytest

from autoflowcfd.core.fr_residual.inviscid import (
    compute_inviscid_residual_fr,
    primitive_to_conserved,
)
from autoflowcfd.core.fr_residual.viscous import compute_viscous_residual
from autoflowcfd.grid.high_order.high_order_mesh import HighOrderMesh


class _MockNodes:
    def __init__(self, coords):
        self._coords = coords

    def get_coordinates(self):
        return self._coords


class _MockCells:
    def __init__(self, connectivity):
        self.connectivity = connectivity


def _build_synthetic_mixed_mesh(order: int) -> HighOrderMesh:
    """2 个共享面的四面体 + 2 个共享侧面的棱柱，覆盖内部面 + 边界面两种
    情形——与 test_fr_residual_inviscid.py::_build_synthetic_mixed_mesh
    完全相同的几何构造（本文件独立保留一份，避免跨测试文件的私有函数
    依赖）。"""
    nodes = np.array(
        [
            [0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 1],
            [10, 0, 0], [11, 0, 0], [10, 1, 0], [10, 0, 1], [11, 0, 1], [10, 1, 1],
        ],
        dtype=float,
    )
    tet_conn = np.array([[0, 1, 2, 3], [1, 2, 3, 4]], dtype=np.int32)
    nodes = np.vstack([nodes, [[9, -1, 0], [9, -1, 1]]])
    prism_conn = np.array(
        [
            [5, 6, 7, 8, 9, 10],
            [5, 7, 11, 8, 10, 12],
        ],
        dtype=np.int32,
    )

    mock_volume = SimpleNamespace(
        cell_count=len(tet_conn) + len(prism_conn),
        nodes=_MockNodes(nodes),
        cells=_MockCells(tet_conn),
        prism_cells=_MockCells(prism_conn),
    )

    mesh = HighOrderMesh(order=order)
    mesh.load_from_volume_mesh(mock_volume)
    return mesh


def _random_nonuniform_state(mesh: HighOrderMesh, seed: int) -> np.ndarray:
    """构造一个逐 cell/逐 SP 各不相同的合法物理状态场（非均匀流场）——
    均匀流场会被自由流场保持性直接吸收界面跳跃项，测不出着色/非着色两条
    路径在真实非零界面通量下是否一致，必须用非平凡状态。"""
    rng = np.random.default_rng(seed)
    n_cells, n_sps = mesh.n_cells, mesh.n_sps_per_cell
    rho = rng.uniform(0.8, 1.6, (n_cells, n_sps))
    u = rng.uniform(-40.0, 40.0, (n_cells, n_sps))
    v = rng.uniform(-40.0, 40.0, (n_cells, n_sps))
    w = rng.uniform(-40.0, 40.0, (n_cells, n_sps))
    p = rng.uniform(6e4, 1.6e5, (n_cells, n_sps))
    Q = np.stack([rho, u, v, w, p], axis=-1)
    return primitive_to_conserved(Q)


class TestInviscidColoredParity:
    @pytest.mark.parametrize("order", [1, 2])
    def test_colored_matches_uncolored(self, order, monkeypatch):
        mesh = _build_synthetic_mixed_mesh(order)
        U = _random_nonuniform_state(mesh, seed=1234)

        monkeypatch.setenv("AFCFD_USE_COLORING", "1")
        residual_colored = compute_inviscid_residual_fr(U, mesh, mesh.operators)

        monkeypatch.setenv("AFCFD_USE_COLORING", "0")
        residual_uncolored = compute_inviscid_residual_fr(U, mesh, mesh.operators)

        np.testing.assert_allclose(
            residual_colored, residual_uncolored, rtol=1e-10, atol=1e-8,
            err_msg=(
                "着色版 (compute_inviscid_interface_correction_kernel_colored) 与"
                "非着色版 (compute_inviscid_interface_correction_kernel) 的无粘"
                "残差不一致——两侧实现已发生分叉，见 inviscid_kernel.py/"
                "inviscid_kernel_colored.py 模块文档。"
            ),
        )


class TestViscousColoredParity:
    @pytest.mark.parametrize("order", [1, 2])
    def test_colored_matches_uncolored(self, order, monkeypatch):
        mesh = _build_synthetic_mixed_mesh(order)
        U = _random_nonuniform_state(mesh, seed=5678)

        monkeypatch.setenv("AFCFD_USE_COLORING", "1")
        residual_colored = compute_viscous_residual(U, U, mesh.operators, mesh)

        monkeypatch.setenv("AFCFD_USE_COLORING", "0")
        residual_uncolored = compute_viscous_residual(U, U, mesh.operators, mesh)

        np.testing.assert_allclose(
            residual_colored, residual_uncolored, rtol=1e-10, atol=1e-8,
            err_msg=(
                "着色版 (compute_viscous_interface_correction_kernel_colored) 与"
                "非着色版 (compute_viscous_interface_correction_kernel) 的粘性"
                "残差不一致——两侧实现已发生分叉，见 viscous_flux_kernel.py/"
                "viscous_flux_kernel_colored.py 模块文档。"
            ),
        )
