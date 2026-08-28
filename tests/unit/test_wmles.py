"""WMLES 壁面应力模型单元测试 (T-05)。

覆盖 V2.0 专家组盲审发现并修复的问题：`solve_friction_velocity_iterative`
此前只用对数律（仅在 y+ >~ 30 准确），缓冲层（y+ < 30）会被外推、给出
偏差明显的 u_tau。现在缓冲层改用 Spalding 全 y+ 律重新求解。
"""

import numpy as np
import pytest

from autoflowcfd.core.turbulence.wmles import WMLESModel


class TestSpaldingLawFormula:
    """Spalding 律公式本身的正确性（与对数律的渐近一致性）。"""

    def test_spalding_matches_log_law_at_high_yplus(self):
        """y+ 很大时，Spalding 律应渐近收敛到对数律（两者共享同一个
        对数律渐近行为，是 Spalding 公式本身设计上的要求）。"""
        model = WMLESModel(kappa=0.41, B=5.2, nu=1.5e-5)
        y_plus = np.array([500.0, 1000.0])
        u_plus_spalding = model.compute_spalding_law(y_plus)
        u_plus_log = model.compute_log_law_velocity(y_plus)
        np.testing.assert_allclose(u_plus_spalding, u_plus_log, rtol=1e-3)


class TestFrictionVelocityBufferLayer:
    """`solve_friction_velocity_iterative` 的缓冲层（y+ < 30）修正。"""

    def test_buffer_layer_recovers_true_u_tau_via_spalding(self):
        """真实 bug 回归测试：构造一个自洽的缓冲层场景（用 Spalding 律
        本身生成 u_mag，保证"真值"就是 Spalding 律给出的解），断言
        求解器能正确收敛回这个真值，而不是发散到下限（此前实现里的
        符号笔误会让 u_tau 单调收缩到 1e-6 的下限，见修复记录）。"""
        nu = 1.5e-5
        model = WMLESModel(kappa=0.41, B=5.2, nu=nu)

        y_plus_true = np.array([15.0])
        u_plus_true = model.compute_spalding_law(y_plus_true)
        u_tau_true = np.array([0.5])
        y_dist = y_plus_true * nu / u_tau_true
        u_mag = u_tau_true * u_plus_true
        u_tangent = np.stack([u_mag, np.zeros_like(u_mag), np.zeros_like(u_mag)], axis=-1)

        u_tau_recovered = model.solve_friction_velocity_iterative(u_tangent, y_dist)

        rel_err = abs(u_tau_recovered[0] - u_tau_true[0]) / u_tau_true[0]
        assert rel_err < 1e-6, f"buffer-layer u_tau recovery failed: rel_err={rel_err:.3e}"
        assert model.y_plus[0] < 30.0

    def test_log_law_region_unaffected(self):
        """y+ 明显在对数律区（>30）的点不应被缓冲层修正逻辑触碰，
        仍应精确收敛（回归防护：确保新增分支不影响主流程）。"""
        nu = 1.5e-5
        model = WMLESModel(kappa=0.41, B=5.2, nu=nu)

        y_plus_hi = np.array([200.0])
        u_plus_hi = model.compute_log_law_velocity(y_plus_hi)
        u_tau_true = np.array([0.3])
        y_dist = y_plus_hi * nu / u_tau_true
        u_mag = u_tau_true * u_plus_hi
        u_tangent = np.stack([u_mag, np.zeros_like(u_mag), np.zeros_like(u_mag)], axis=-1)

        u_tau_recovered = model.solve_friction_velocity_iterative(u_tangent, y_dist)

        rel_err = abs(u_tau_recovered[0] - u_tau_true[0]) / u_tau_true[0]
        assert rel_err < 1e-6
        assert model.y_plus[0] > 30.0

    def test_mixed_batch_buffer_and_log_law_points(self):
        """同一批点里混有缓冲层和对数律区两类点，两者应各自独立正确
        求解（验证掩码索引逻辑不会互相污染）。"""
        nu = 1.5e-5
        model = WMLESModel(kappa=0.41, B=5.2, nu=nu)

        y_plus_targets = np.array([10.0, 200.0, 20.0])
        u_tau_true = np.array([0.5, 0.3, 0.4])
        y_dist = y_plus_targets * nu / u_tau_true

        u_plus_true = np.where(
            y_plus_targets < 30.0,
            model.compute_spalding_law(y_plus_targets),
            model.compute_log_law_velocity(y_plus_targets),
        )
        u_mag = u_tau_true * u_plus_true
        u_tangent = np.stack([u_mag, np.zeros_like(u_mag), np.zeros_like(u_mag)], axis=-1)

        u_tau_recovered = model.solve_friction_velocity_iterative(u_tangent, y_dist)

        np.testing.assert_allclose(u_tau_recovered, u_tau_true, rtol=1e-5)
