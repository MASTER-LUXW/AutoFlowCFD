"""真实功能新增测试（2026-09-12）：`core/fr_solver/residual_diagnostics.py`
——参照 Fluent scaled residuals / STAR-CCM+ Max 监视器补齐的按方程
归一化残差 + 最大残差定位诊断。见该模块文档"背景"一节：cube_demo
791,492 单元真实网格排查中，合并 RMS 残差被少数方块前驻点单元的能量
方程残差主导（比全局 RMS 还大 1 个数量级），只用一个数字完全看不出来。
"""
import numpy as np
import pytest

from autoflowcfd.core.fr_solver.residual_diagnostics import (
    compute_scaled_residuals, format_scaled_residual_line,
)


class TestComputeScaledResiduals:
    def test_uniform_residual_gives_expected_scaled_values(self):
        """构造一个已知的均匀残差场，手算验证归一化公式。"""
        n_cells, n_sps, n_vars = 4, 2, 5
        dU_dt = np.zeros((n_cells, n_sps, n_vars))
        dU_dt[..., 0] = 1.0    # rho 残差
        dU_dt[..., 1] = 10.0   # rho_u 残差
        dU_dt[..., 4] = 1000.0  # rho_E 残差

        freestream = {"rho_inf": 1.225, "vel_inf": 33.33, "p_inf": 101325.0}
        diag = compute_scaled_residuals(dU_dt, freestream)

        np.testing.assert_allclose(diag.rms_per_var[0], 1.0)
        np.testing.assert_allclose(diag.rms_per_var[1], 10.0)
        np.testing.assert_allclose(diag.rms_per_var[4], 1000.0)

        expected_scaled_rho = 1.0 / 1.225
        expected_scaled_rho_u = 10.0 / (1.225 * 33.33)
        expected_scaled_rho_E = 1000.0 / 101325.0
        np.testing.assert_allclose(diag.scaled_rms_per_var[0], expected_scaled_rho, rtol=1e-10)
        np.testing.assert_allclose(diag.scaled_rms_per_var[1], expected_scaled_rho_u, rtol=1e-10)
        np.testing.assert_allclose(diag.scaled_rms_per_var[4], expected_scaled_rho_E, rtol=1e-10)

    def test_max_residual_location_identifies_outlier_cell(self):
        """真实排查场景的最小复现：绝大多数单元残差很小，个别单元的
        能量方程残差极大——必须能精确定位到那个单元/变量。"""
        n_cells, n_sps, n_vars = 100, 1, 5
        rng = np.random.default_rng(0)
        dU_dt = rng.uniform(-10, 10, size=(n_cells, n_sps, n_vars))

        outlier_cell = 42
        dU_dt[outlier_cell, 0, 4] = 5.4e7  # 模拟真实排查里驻点单元的能量残差

        freestream = {"rho_inf": 1.225, "vel_inf": 33.33, "p_inf": 101325.0}
        diag = compute_scaled_residuals(dU_dt, freestream)

        assert diag.max_abs_cell == outlier_cell
        assert diag.max_abs_var == 4  # rho_E
        assert diag.max_abs == pytest.approx(5.4e7)

    def test_energy_dominance_is_visible_in_scaled_but_not_hidden(self):
        """真实病理的核心断言：即使 rho_E 的绝对残差比其他变量大 5 个
        数量级，归一化之后仍然能通过 scaled_rms_per_var 逐变量看出
        "只有能量方程在这个位置异常"，而不是被合并进一个数字里看不出来。"""
        n_cells, n_sps, n_vars = 10, 1, 5
        dU_dt = np.zeros((n_cells, n_sps, n_vars))
        dU_dt[..., 0] = 5.0     # rho: 正常量级
        dU_dt[..., 4] = 5.4e7   # rho_E: 真实排查里驻点单元的量级

        freestream = {"rho_inf": 1.225, "vel_inf": 33.33, "p_inf": 101325.0}
        diag = compute_scaled_residuals(dU_dt, freestream)

        # 归一化后 energy 的 scaled residual 仍然远大于 density 的——
        # 说明"能量方程在这里真的有问题"这个信息在归一化后依然保留，
        # 不是被強行拉到同一量级、看不出差异。
        assert diag.scaled_rms_per_var[4] > diag.scaled_rms_per_var[0] * 100


class TestFormatScaledResidualLine:
    def test_format_includes_all_variables_and_max_location(self):
        n_cells, n_sps, n_vars = 5, 1, 5
        dU_dt = np.ones((n_cells, n_sps, n_vars))
        dU_dt[3, 0, 2] = 999.0  # 让 cell 3, var 2 (rho_v) 成为最大值

        freestream = {"rho_inf": 1.225, "vel_inf": 33.33, "p_inf": 101325.0}
        diag = compute_scaled_residuals(dU_dt, freestream)
        line = format_scaled_residual_line(diag)

        assert "rho=" in line
        assert "rho_u=" in line
        assert "rho_E=" in line
        assert "rho_v@cell3" in line
