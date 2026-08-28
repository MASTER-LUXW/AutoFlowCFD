"""Unit tests for core/turbulence/des.py's DDES shielding function fix.

第四次评审第三轮修复：`DDESModel.compute_shielding_function` 的 `r_d`
公式此前只用应变率模 `|S|`，遗漏了 Gritskevich et al. (2012) SST-DDES
公式要求的涡量模 `|Ω|`（正确公式是联合尺度
`sqrt(0.5*(|S|^2+|Ω|^2))`），且 `c_w1`（对应文献 `C_d1`）默认值用的是
SA-DES97 的 8.0，而不是该论文为 SST 框架重新标定的 20.0。

这些测试钉住修复后的行为，防止日后被误改回旧值/旧公式而不被任何测试
捕捉到（此前 des.py 完全没有专属单元测试）。
"""

import numpy as np
import pytest

from autoflowcfd.core.turbulence.des import DDESModel, IDDESModel, compute_h_max_and_h_wn


class TestDDESConstants:
    """钉住 Gritskevich et al. (2012) 为 SST 框架重新标定的常数，
    不被误改回 SA-DES97 的原始值 8.0。"""

    def test_ddes_default_c_w1_is_20(self):
        assert DDESModel().c_w1 == 20.0

    def test_iddes_default_c_w1_is_20(self):
        assert IDDESModel().c_w1 == 20.0


class TestVorticityMagnitude:
    """`compute_vorticity_magnitude` 与 `compute_strain_rate_magnitude`
    是完全平行的实现（同一套 |X|=sqrt(2*X_ij*X_ij) 约定），用纯旋转/
    纯应变两个正交的合成算例分别验证。"""

    def test_pure_rotation_has_zero_strain_nonzero_vorticity(self):
        """grad_u 只有反对称分量：应变率必须恰好为零，涡量必须非零。"""
        n_cells, n_sps = 3, 2
        grad_u = np.zeros((n_cells, n_sps, 3, 3))
        grad_u[..., 0, 1] = 1.0
        grad_u[..., 1, 0] = -1.0  # 纯刚体旋转（xy 平面）

        ddes = DDESModel()
        S_mag = ddes.compute_strain_rate_magnitude(grad_u)
        Omega_mag = ddes.compute_vorticity_magnitude(grad_u)

        np.testing.assert_allclose(S_mag, 0.0, atol=1e-12)
        assert np.all(Omega_mag > 1.0)

    def test_pure_strain_has_zero_vorticity_nonzero_strain(self):
        """grad_u 只有对称分量：涡量必须恰好为零，应变率必须非零。"""
        n_cells, n_sps = 3, 2
        grad_u = np.zeros((n_cells, n_sps, 3, 3))
        grad_u[..., 0, 1] = 1.0
        grad_u[..., 1, 0] = 1.0  # 纯剪切应变（对称）

        ddes = DDESModel()
        S_mag = ddes.compute_strain_rate_magnitude(grad_u)
        Omega_mag = ddes.compute_vorticity_magnitude(grad_u)

        assert np.all(S_mag > 1.0)
        np.testing.assert_allclose(Omega_mag, 0.0, atol=1e-12)


class TestShieldingFunctionAnalytic:
    """用简单剪切流（simple shear，∂u/∂y=γ 为常数）的解析性质校验
    `r_d`/`f_d`：简单剪切流的应变率模与涡量模严格相等（|S|=|Ω|=γ），
    这是一个可以手算验证、不依赖数值巧合的解析结果。"""

    def _simple_shear_grad_u(self, n_cells, n_sps, gamma):
        grad_u = np.zeros((n_cells, n_sps, 3, 3))
        grad_u[..., 0, 1] = gamma  # ∂u/∂y = γ，其余分量为零
        return grad_u

    def test_simple_shear_strain_equals_vorticity_magnitude(self):
        """解析恒等式：简单剪切流 |S| = |Ω| = |γ|（与 grad_u 是否反对称
        无关，只由"仅一个非对角分量非零"这一几何特征决定）。"""
        gamma = 3.7
        grad_u = self._simple_shear_grad_u(2, 1, gamma)
        ddes = DDESModel()
        S_mag = ddes.compute_strain_rate_magnitude(grad_u)
        Omega_mag = ddes.compute_vorticity_magnitude(grad_u)
        np.testing.assert_allclose(S_mag, abs(gamma), rtol=1e-10)
        np.testing.assert_allclose(Omega_mag, abs(gamma), rtol=1e-10)

    def test_r_d_matches_hand_calculation(self):
        """在简单剪切流下手算 r_d/f_d，与函数输出逐位对比。

        简单剪切流下 S_Omega_mag = sqrt(0.5*(γ²+γ²)) = |γ|，代入
        r_d = (nu_t+nu)/(kappa²*d_w²*|γ|)，f_d = 1-tanh[(c_w1*r_d)^3]。
        """
        gamma = 50.0  # 典型近壁高剪切量级 (1/s)
        n_cells, n_sps = 2, 1
        grad_u = self._simple_shear_grad_u(n_cells, n_sps, gamma)

        d_w = np.full((n_cells, n_sps), 0.005)
        nu_t = np.full((n_cells, n_sps), 2e-4)
        omega = np.full((n_cells, n_sps), 500.0)
        nu = np.full((n_cells, n_sps), 1.5e-5)
        kappa = 0.41

        ddes = DDESModel()
        f_d = ddes.compute_shielding_function(d_w, nu_t, omega, nu, kappa=kappa, grad_u=grad_u)

        S_omega_mag_expected = abs(gamma)
        r_d_expected = (nu_t + nu) / (kappa**2 * d_w**2 * S_omega_mag_expected)
        r_d_expected = np.minimum(r_d_expected, 10.0)
        f_d_expected = 1.0 - np.tanh((ddes.c_w1 * r_d_expected) ** 3)

        np.testing.assert_allclose(f_d, f_d_expected, rtol=1e-10)

    def test_vorticity_term_prevents_denominator_floor_in_pure_rotation(self):
        """修复前的回归防护：纯旋转（|S|=0）区域此前会让分母被裁剪到
        1e-6 下限、人为地把 r_d 推到远大于真实值——现在 |Ω| 项接管，
        r_d 应该反映真实的旋转速率量级，而不是被下限裁剪主导。"""
        omega_rate = 200.0  # 涡量模（1/s），量级远大于 1e-6 下限
        n_cells, n_sps = 2, 1
        grad_u = np.zeros((n_cells, n_sps, 3, 3))
        grad_u[..., 0, 1] = omega_rate
        grad_u[..., 1, 0] = -omega_rate  # 纯旋转

        d_w = np.full((n_cells, n_sps), 0.01)
        nu_t = np.full((n_cells, n_sps), 1e-4)
        omega = np.full((n_cells, n_sps), 100.0)
        nu = np.full((n_cells, n_sps), 1.5e-5)
        kappa = 0.41

        ddes = DDESModel()
        f_d = ddes.compute_shielding_function(d_w, nu_t, omega, nu, kappa=kappa, grad_u=grad_u)

        # 用裁剪到 1e-6 的分母重新算一遍"修复前会得到的"错误 r_d/f_d，
        # 确认修复后的结果与之明显不同（真正用到了 Ω 项，不是退化到下限）。
        r_d_if_floored = (nu_t + nu) / (kappa**2 * d_w**2 * 1e-6)
        r_d_if_floored = np.minimum(r_d_if_floored, 10.0)
        f_d_if_floored = 1.0 - np.tanh((ddes.c_w1 * r_d_if_floored) ** 3)

        assert not np.allclose(f_d, f_d_if_floored, atol=1e-6)


class TestIDDESGridScaleGeometry:
    """compute_h_max_and_h_wn 用一个几何已知的合成"网格"（1 棱柱 + 1 四面体）
    钉住 h_max/h_wn 的精确解析值，防止边长->h_max/h_wn 的映射逻辑被误改
    （例如棱柱竖直边的切片索引写错、单元顺序假设弄反）。"""

    class _MockMesh:
        """只提供 compute_h_max_and_h_wn 需要读取的最小属性集
        （n_cells/n_prism_cells/_node_coords/_fixed_prism_conn/
        _fixed_tet_conn），不是真正的 HighOrderMesh。"""

        def __init__(self):
            self.n_cells = 2
            self.n_prism_cells = 1
            # 棱柱：底三角形 v0,v1,v2 边长 1/1/sqrt(2)；顶三角形 w0,w1,w2
            # 由底三角形沿 z 平移 0.01 得到（水平边长不变）；竖直边
            # （近似壁面法向）长度恰好 0.01——典型薄 BL 棱柱形状。
            # 四面体：三条正交棱边长度均为 3，用于验证各向同性单元下
            # h_wn 退化为 h_max。
            self._node_coords = np.array([
                [0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0],
                [0.0, 0.0, 0.01], [1.0, 0.0, 0.01], [0.0, 1.0, 0.01],
                [0.0, 0.0, 0.0], [3.0, 0.0, 0.0], [0.0, 3.0, 0.0], [0.0, 0.0, 3.0],
            ])
            self._fixed_prism_conn = np.array([[0, 1, 2, 3, 4, 5]])
            self._fixed_tet_conn = np.array([[6, 7, 8, 9]])

    def test_prism_h_max_and_h_wn(self):
        """薄棱柱：h_max 取三角形对角边 sqrt(2)，h_wn 取竖直边 0.01——
        验证"最薄方向"被正确识别为法向间距，而不是与 h_max 混淆。"""
        mesh = self._MockMesh()
        h_max, h_wn = compute_h_max_and_h_wn(mesh)
        np.testing.assert_allclose(h_max[0], np.sqrt(2.0), rtol=1e-10)
        np.testing.assert_allclose(h_wn[0], 0.01, rtol=1e-10)

    def test_tet_h_wn_equals_h_max(self):
        """四面体（核心/LES 区）没有明确的"法向"概念，h_wn 应退化为
        h_max（见 compute_h_max_and_h_wn 文档的各向同性简化说明）。"""
        mesh = self._MockMesh()
        h_max, h_wn = compute_h_max_and_h_wn(mesh)
        expected_tet_h_max = 3.0 * np.sqrt(2.0)
        np.testing.assert_allclose(h_max[1], expected_tet_h_max, rtol=1e-10)
        np.testing.assert_allclose(h_wn[1], expected_tet_h_max, rtol=1e-10)


class TestIDDESFormulaComponents:
    """逐个手算钉住 IDDES 各公式分量（alpha/f_B/f_e1/f_e2/Δ/l_IDDES），
    与本次实现（IDDESModel 类文档字符串所列公式/常数置信度说明）一一
    对应，防止日后被静默改回旧的、非标准的启发式实现。"""

    def test_alpha_formula(self):
        iddes = IDDESModel()
        d_w = np.array([0.1, 0.5])
        h_max = np.array([1.0, 1.0])
        alpha = iddes.compute_alpha(d_w, h_max)
        np.testing.assert_allclose(alpha, [0.25 - 0.1, 0.25 - 0.5], rtol=1e-12)

    def test_f_b_at_alpha_zero_is_one(self):
        """alpha=0（d_w 恰好等于 0.25*h_max）：2*exp(0)=2，min(2,1)=1，
        f_B 应精确等于 1（RANS 区）。"""
        iddes = IDDESModel()
        f_b = iddes.compute_f_b(np.array([0.0]))
        np.testing.assert_allclose(f_b, [1.0], atol=1e-12)

    def test_f_b_vanishes_far_from_wall(self):
        """|alpha| 很大（d_w 远大于 h_max）时 exp(-9*alpha^2)->0，
        f_B 应趋近 0（LES 区）。"""
        iddes = IDDESModel()
        f_b_far = iddes.compute_f_b(np.array([-5.0]))
        assert f_b_far[0] < 1e-6

    def test_f_b_never_exceeds_one(self):
        """alpha 接近 0 附近 2*exp(-9*alpha^2) 会超过 1，min(...,1) 必须
        生效裁剪，不能让 f_B 突破其定义域上限。"""
        iddes = IDDESModel()
        alpha = np.linspace(-0.05, 0.05, 21)
        f_b = iddes.compute_f_b(alpha)
        assert np.all(f_b <= 1.0)

    def test_f_e1_piecewise_matches_hand_calculation(self):
        """alpha>=0 与 alpha<0 两支必须各自对应不同的指数系数
        （11.09 vs 9），不能被误合并成同一个分支。"""
        iddes = IDDESModel()
        alpha = np.array([-0.3, 0.3])
        f_e1 = iddes.compute_f_e1(alpha)
        expected = np.array([
            2.0 * np.exp(-9.0 * 0.3**2),
            2.0 * np.exp(-11.09 * 0.3**2),
        ])
        np.testing.assert_allclose(f_e1, expected, rtol=1e-12)

    def test_f_e2_matches_hand_calculation(self):
        """r_dt 用纯 nu_t（不含分子粘度），r_dl 用纯 nu（不含涡粘）——
        与 DDESModel.compute_shielding_function 的 r_d=(nu_t+nu)/(...)
        联合尺度是两个不同的量，此测试钉住两者不能被混用。"""
        iddes = IDDESModel()
        nu_t = np.array([1e-4])
        nu = np.array([1.5e-5])
        d_w = np.array([0.01])
        S_Omega_mag = np.array([50.0])
        kappa = 0.41
        f_e2 = iddes.compute_f_e2(nu_t, nu, d_w, S_Omega_mag, kappa=kappa)

        denom = kappa**2 * d_w**2 * S_Omega_mag
        r_dt = nu_t / denom
        r_dl = nu / denom
        f_t = np.tanh((iddes.c_t**2 * r_dt) ** 3)
        f_l = np.tanh((iddes.c_l**2 * r_dl) ** 10)
        expected = 1.0 - np.maximum(f_t, f_l)
        np.testing.assert_allclose(f_e2, expected, rtol=1e-12)

    def test_grid_scale_iddes_matches_hand_calculation_and_bounded_by_h_max(self):
        iddes = IDDESModel(c_w=0.15)
        d_w = np.array([0.02, 5.0])
        h_max = np.array([1.0, 1.0])
        h_wn = np.array([0.05, 0.05])
        delta = iddes.compute_grid_scale_iddes(d_w, h_max, h_wn)

        inner = np.maximum(np.maximum(0.15 * d_w, 0.15 * h_max), h_wn)
        expected = np.minimum(inner, h_max)
        np.testing.assert_allclose(delta, expected, rtol=1e-12)
        # Δ 不能超过 h_max（公式最外层 min 的直接后果，防止远场大 d_w
        # 把 Δ 推到远超单元自身尺寸的物理无意义值）
        assert np.all(delta <= h_max + 1e-12)

    def test_effective_length_scale_reduces_to_l_rans_when_f_b_is_one(self):
        """f_B=1, f_e=0 时（极近壁）l_IDDES 应精确退化为纯 l_RANS，
        与 DDES 的近壁极限行为一致（凸组合权重全部压在 RANS 侧）。"""
        iddes = IDDESModel(c_des=0.78)
        k = np.array([1e-3])
        omega = np.array([100.0])
        beta_star = 0.09
        delta = np.array([1.0])
        f_b = np.array([1.0])
        f_e = np.array([0.0])
        l_iddes = iddes.compute_effective_length_scale_iddes(k, omega, beta_star, delta, f_b, f_e)
        l_rans_expected = np.sqrt(k) / (beta_star * omega)
        np.testing.assert_allclose(l_iddes, l_rans_expected, rtol=1e-10)

    def test_effective_length_scale_reduces_to_l_les_when_f_b_is_zero(self):
        """f_B=0 时（远离壁面/分离区）l_IDDES 应精确退化为纯
        l_LES=C_DES*Δ，且与 f_e 取值无关——(1-f_B) 权重为零时 f_e 项
        必须完全不参与，这是凸组合公式结构本身的直接推论。"""
        iddes = IDDESModel(c_des=0.78)
        k = np.array([1e-3])
        omega = np.array([100.0])
        beta_star = 0.09
        delta = np.array([2.0])
        f_b = np.array([0.0])
        f_e = np.array([0.5])
        l_iddes = iddes.compute_effective_length_scale_iddes(k, omega, beta_star, delta, f_b, f_e)
        np.testing.assert_allclose(l_iddes, [0.78 * 2.0], rtol=1e-10)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
