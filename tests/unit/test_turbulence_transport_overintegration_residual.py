"""湍流输运过积分：完整对流残差对解析解、自由流保持（从 test_turbulence_transport_overintegration.py 拆出，背景见该文件模块文档）。"""

import numpy as np
import pytest

from autoflowcfd.core.turbulence import transport as tp
from autoflowcfd.core.fr_operators.volume_contract import get_overintegration_context
from autoflowcfd.core.turbulence.transport.convection import _scalar_convection_volume_overintegrated
from autoflowcfd.core.turbulence.transport.diffusion import _scalar_diffusion_volume_overintegrated
from autoflowcfd.fr.operators import generate_fr_operators

from tests.unit.test_fr_residual_inviscid import _build_synthetic_mixed_mesh
from tests.unit._turbulence_transport_overintegration_common import (
    _per_type_slices,
    _real_dof_mask,
)


class TestFullConvectionResidualVsAnalytic:
    """**整条对流残差**（体积项 + 界面上风校正）与解析值比较。

    这是本轮系统遍历核心计算环节时发现的最重要一条：前面的用例只单独
    考察体积项，这一条走公开接口 `compute_scalar_convection_residual`。

    算例设计：`rho`/`u` 取**常数**（于是通过每个单元各面的质量通量精确
    守恒，避开 `filter_scalar_field`/"缺失 dilution"那段文档记录过的
    "对 sum_faces(mass_flux) 敏感"这个已被证伪的方向），`phi = 1 + G·x`
    取线性。于是

        d(rho*phi)/dt = -div(rho*u*phi) = -rho*(u·G)   （处处同一个常数）

    有闭式解，且 `rho*u*phi` 本身只有一次、完全落在 P1 解空间内——也就是
    说**任何非零误差都只能来自离散算子本身**，不是表示能力不足。

    实测（2026-09-15，当时的坍缩棱柱基、不去混叠）：

        order=1 棱柱 相对误差 1.1294  （113%！残差范围 [-12.26, +1.11]，
                                      解析值 -8.5750）
        order=1 四面体 6.84e-15       （机器零）
        order=2 棱柱 2.26e-09
        order=2 四面体 1.97e-14

    打开去混叠后 order=1 棱柱降到 **4.61e-10**（改善约 2.4e9 倍），
    order=2 逐位不变（2.2597e-09 -> 2.2576e-09）。

    根因：非仿射棱柱上 `adj(J)` 是非平凡多项式，`adj(J)*rho*u*phi` 的
    真实次数高于 1，在 P1 空间里对它求导就是"先混叠再求导"；四面体在
    这张网格上是仿射的（实测同一单元内 det(J) 跨度恰好 0），adj(J) 是
    常数，所以没有这个问题——这也解释了为什么两类单元差了 14 个数量级。

    **边界条件不是原因**：order=1 与 order=2 的边界面数完全相同
    （cell0 8 面其中 6 边界、cell1 6 面其中 6 边界、…），而 order=2 给出
    的是**精确的** -8.5750。
    """

    RHO = 1.225
    UVEC = np.array([30.0, 7.0, -4.0])
    GRAD = np.array([0.3, -0.2, 0.15])

    def _run(self, order):
        mesh = _build_synthetic_mixed_mesh(order)
        ops = generate_fr_operators(order)
        nc, ns = mesh.n_cells, mesh.n_sps_per_cell
        X = mesh.sps_coords.reshape(-1, 3)
        phi = (1.0 + X @ self.GRAD).reshape(nc, ns)
        rho = np.full((nc, ns), self.RHO)
        vel = np.tile(self.UVEC, (nc, ns, 1))
        res = tp.compute_scalar_convection_residual(phi, rho, vel, mesh, ops)
        exact = -self.RHO * float(self.UVEC @ self.GRAD)
        errs = {}
        for name, sl in _per_type_slices(mesh, order):
            errs[name] = np.abs(res[sl] - exact).max() / abs(exact)
        return errs

    def test_dealiasing_gives_an_accurate_order1_prism(self):
        """去混叠后 P1 棱柱必须准（原生实测 2.3e-10；已删除的坍缩档是 4.6e-10、不去混叠时 1.13）。"""
        errs = self._run(1)
        assert errs["prism"] < 1e-8, (
            f"打开去混叠后 order=1 棱柱相对误差仍有 "
            f"{errs['prism']:.3e}（实测应为 ~2e-10）")
        assert errs["tet"] < 1e-12

    def test_order2_is_accurate(self):
        """order=2 同样精确。"""
        errs = self._run(2)
        for k in ("prism", "tet"):
            assert errs[k] < 1e-7, f"order=2 {k} 相对误差 {errs[k]:.3e}"


class TestFreestreamPreservation:
    """安全性：常数场的散度必须为零——去混叠不能破坏自由流场保持性。"""

    @pytest.mark.parametrize("order", [1, 2])
    def test_constant_field_gives_zero_divergence(self, order):
        mesh = _build_synthetic_mixed_mesh(order)
        ops = generate_fr_operators(order)
        n_cells, n_sps = mesh.n_cells, mesh.n_sps_per_cell
        phi = np.full((n_cells, n_sps), 0.17)
        rho = np.full((n_cells, n_sps), 1.225)
        vel = np.zeros((n_cells, n_sps, 3))
        vel[..., 0] = 30.0
        oi = get_overintegration_context(mesh, ops)
        det = mesh.jacobians["det_jacs"].reshape(n_cells, n_sps)
        div = _scalar_convection_volume_overintegrated(
            phi, rho, vel, oi, n_sps) / det
        m = _real_dof_mask(mesh, order)
        ref = 1.225 * 30.0 * 0.17 / max(
            float(np.abs(mesh.sps_coords).max()), 1.0)
        assert np.abs(div[m]).max() < 1e-9 * max(ref, 1.0), (
            f"order={order}: 常数场散度 {np.abs(div[m]).max():.3e} 不是零——"
            f"去混叠破坏了自由流场保持性")

    @pytest.mark.parametrize("order", [1, 2])
    def test_constant_gradient_zero_gamma_gives_zero(self, order):
        """Gamma 恒为零 -> 扩散通量恒为零 -> 散度必须精确为零。"""
        mesh = _build_synthetic_mixed_mesh(order)
        ops = generate_fr_operators(order)
        n_cells, n_sps = mesh.n_cells, mesh.n_sps_per_cell
        oi = get_overintegration_context(mesh, ops)
        grad = np.ones((n_cells, n_sps, 3))
        div = _scalar_diffusion_volume_overintegrated(
            np.zeros((n_cells, n_sps)), grad, oi, n_sps, 1.8e-5)
        np.testing.assert_allclose(div, 0.0, rtol=0, atol=0)
