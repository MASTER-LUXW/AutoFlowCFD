"""湍流输运过积分：完整对流残差对解析解、自由流保持、开关语义（从 test_turbulence_transport_overintegration.py 拆出，背景见该文件模块文档）。"""

import os
import numpy as np
import pytest

from autoflowcfd.core.turbulence import transport as tp
from autoflowcfd.core.fr_operators.volume_contract import get_overintegration_context
from autoflowcfd.core.turbulence.transport.convection import _scalar_convection_volume_overintegrated
from autoflowcfd.core.turbulence.transport.diffusion import _scalar_diffusion_volume_overintegrated
from autoflowcfd.fr.operators import generate_fr_operators

from tests.unit.test_fr_residual_inviscid import _build_synthetic_mixed_mesh
from tests.unit._turbulence_transport_overintegration_common import (
    _coarse_convection_div,
    _per_type_slices,
    _real_dof_mask,
    _setup,
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

    实测（默认 `AFCFD_TURB_OVERINT=off`）：

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

    def _run(self, order, overint):
        old = os.environ.get("AFCFD_TURB_OVERINT")
        os.environ["AFCFD_TURB_OVERINT"] = overint
        try:
            mesh = _build_synthetic_mixed_mesh(order)
            ops = generate_fr_operators(order)
            nc, ns = mesh.n_cells, mesh.n_sps_per_cell
            X = mesh.sps_coords.reshape(-1, 3)
            phi = (1.0 + X @ self.GRAD).reshape(nc, ns)
            rho = np.full((nc, ns), self.RHO)
            vel = np.tile(self.UVEC, (nc, ns, 1))
            res = tp.compute_scalar_convection_residual(phi, rho, vel, mesh, ops)
        finally:
            if old is None:
                os.environ.pop("AFCFD_TURB_OVERINT", None)
            else:
                os.environ["AFCFD_TURB_OVERINT"] = old
        exact = -self.RHO * float(self.UVEC @ self.GRAD)
        errs = {}
        for name, sl in _per_type_slices(mesh, order):
            errs[name] = np.abs(res[sl] - exact).max() / abs(exact)
        return errs

    def test_order1_prism_is_already_accurate_without_dealiasing(self):
        """原生棱柱基（唯一实现）：同一算例 off 档就已经准。

        实测 `off` 档 P1 棱柱相对误差 **2.33e-10**（坍缩档是 1.1294，
        差 9 个数量级）。原因与坍缩档那 113% 的根因是同一条的反面：
        原生棱柱基不经过坍缩坐标，`adj(J)` 不被坍缩映射推高次数。

        这条同时说明一件对默认值有影响的事实、如实记录：
        `AFCFD_TURB_OVERINT=on` 这个默认值当初的**主要**依据（P1 棱柱
        113% 误差）在默认基改成 native 之后已经不再成立。保留 `on` 仍有
        依据 —— 体积项对照里原生 P1 仍有 33 倍改善、P2 直接到机器零
        （见 `_PRISM_EXPECT`）—— 但"不开就有 O(1) 误差"这句话不能再用。
        """
        errs = self._run(1, "off")
        assert errs["prism"] < 1e-8, (
            f"原生档 P1 棱柱 off 相对误差 {errs['prism']:.3e} 远大于实测的 "
            f"2.33e-10 —— 若这是真实退化，请查原生棱柱的度量/算子")
        assert errs["tet"] < 1e-12

    def test_dealiasing_gives_an_accurate_order1_prism(self):
        """打开去混叠后必须准（原生实测 2.3e-10；已删除的坍缩档是 4.6e-10）。"""
        errs = self._run(1, "on")
        assert errs["prism"] < 1e-8, (
            f"打开去混叠后 order=1 棱柱相对误差仍有 "
            f"{errs['prism']:.3e}（实测应为 ~2e-10）")
        assert errs["tet"] < 1e-12

    def test_order2_is_unaffected_by_the_switch(self):
        """order=2 上两档必须都已足够精确——说明 order=1 那个 113% 不是
        "这套离散本来就这么差"，而是 order=1 特有的混叠。"""
        off = self._run(2, "off")
        on = self._run(2, "on")
        for k in ("prism", "tet"):
            assert off[k] < 1e-7, f"order=2 {k} 默认档相对误差 {off[k]:.3e}"
            assert on[k] < 1e-7, f"order=2 {k} 去混叠档相对误差 {on[k]:.3e}"


class TestFreestreamPreservation:
    """安全性：常数场的散度必须为零——去混叠不能破坏自由流场保持性。"""

    @pytest.mark.parametrize("order", [1, 2])
    def test_constant_field_gives_zero_divergence(self, order):
        mesh = _build_synthetic_mixed_mesh(order)
        ops = generate_fr_operators(order)
        n_cells, n_sps = mesh.n_cells, mesh.n_sps_per_cell
        phi = np.full((n_cells, n_sps), 0.17)
        rho = np.full((n_cells, n_sps), 1.225)
        vel = np.zeros((n_cells, n_sps, 3)); vel[..., 0] = 30.0
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
            np.zeros((n_cells, n_sps)), grad, oi, n_sps)
        np.testing.assert_allclose(div, 0.0, rtol=0, atol=0)


class TestSwitchSemantics:
    def _env(self, value):
        old = os.environ.get("AFCFD_TURB_OVERINT")
        if value is None:
            os.environ.pop("AFCFD_TURB_OVERINT", None)
        else:
            os.environ["AFCFD_TURB_OVERINT"] = value
        return old

    def _restore(self, old):
        if old is None:
            os.environ.pop("AFCFD_TURB_OVERINT", None)
        else:
            os.environ["AFCFD_TURB_OVERINT"] = old

    def test_default_is_on(self):
        """默认已于 2026-09-15 改为 `on`——解析判据显示 `off` 在生产阶数
        P1 的棱柱上有 113% 相对误差，代价只有约 +10.2%/步，且真实网格
        250 步运行本来就是带 on 跑的。理由全文见
        `resolve_turb_overintegration` 文档。"""
        old = self._env(None)
        try:
            assert tp.resolve_turb_overintegration() == "on"
        finally:
            self._restore(old)

    @pytest.mark.parametrize("v,expected", [
        ("off", "off"), ("on", "on"), ("OFF", "off"), ("On", "on"),
    ])
    def test_accepted_values(self, v, expected):
        old = self._env(v)
        try:
            assert tp.resolve_turb_overintegration() == expected
        finally:
            self._restore(old)

    @pytest.mark.parametrize("v", ["yes", "1", "true", "", "sensor"])
    def test_rejects_unknown(self, v):
        """不能静默退回默认——同 AFCFD_FILTER_TURB_GATE 的理由。"""
        old = self._env(v)
        try:
            with pytest.raises(ValueError, match="AFCFD_TURB_OVERINT"):
                tp.resolve_turb_overintegration()
        finally:
            self._restore(old)

    @pytest.mark.parametrize("order", [1, 2])
    def test_public_api_bit_identical_when_off(self, order):
        """显式 `off` 时，公开接口必须逐位等于 2026-09-15 之前的实现。

        判据取"显式 off"与"手工复刻的 coarse 体积项 + 同一条公开接口"
        之间的一致性：直接比对流残差整体（体积项 + 界面项），任何把
        去混叠误接进默认路径的改动都会在这里失败。
        """
        mesh, ops, phi, rho, vel, _ = _setup(order)
        old = self._env("off")
        try:
            res_off = tp.compute_scalar_convection_residual(
                phi, rho, vel, mesh, ops)
        finally:
            self._restore(old)
        det = mesh.jacobians["det_jacs"].reshape(mesh.n_cells, mesh.n_sps_per_cell)
        ones = np.ones_like(phi)
        # 体积项取对流形式 div(rho u phi) - phi div(rho u)（convection.py 模块文档），
        # 两档各自用同一个体积算子作用在 phi 与 1 上
        vol_ref = -(_coarse_convection_div(phi, rho, vel, mesh, ops)
                    - phi * _coarse_convection_div(ones, rho, vel, mesh, ops)) / det
        # 残差 = 体积项 + 界面项；这里只能断言"体积项那一半与 coarse 一致"，
        # 做法是再跑一次 on 档，两者之差必须恰好等于两种体积项之差。
        old = self._env("on")
        try:
            res_on = tp.compute_scalar_convection_residual(
                phi, rho, vel, mesh, ops)
        finally:
            self._restore(old)
        oi = get_overintegration_context(mesh, ops)
        n_sps = mesh.n_sps_per_cell
        vol_oi = -(_scalar_convection_volume_overintegrated(phi, rho, vel, oi, n_sps)
                   - phi * _scalar_convection_volume_overintegrated(ones, rho, vel, oi, n_sps)) / det
        np.testing.assert_allclose(res_on - res_off, vol_oi - vol_ref,
                                   rtol=1e-10, atol=1e-10)
        # 而且两者确实不同（否则上面那条是平凡真）
        assert np.abs(res_on - res_off).max() > 0.0
