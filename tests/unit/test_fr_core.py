"""
AutoFlowCFD V2.0 - FR 模块单元测试

本模块测试 Flux Reconstruction 方法的核心组件。
"""

import numpy as np
import pytest
from autoflowcfd.fr import (
    FROperators,
    generate_fr_operators,
    gauss_legendre,
    gauss_lobatto,
    compute_diff_matrix_1d,
    compute_interpolation_matrix,
)
from autoflowcfd.core.fr_solver.state import FRState  # 导入 S-01 核心类


class TestQuadraturePoints:
    """测试求积点集生成"""

    def test_gauss_legendre(self):
        """测试Gauss-Legendre点集生成"""
        n = 3
        points, weights = gauss_legendre(n)
        
        # 检查点数
        assert len(points) == n
        assert len(weights) == n
        
        # 检查范围在(-1, 1)内
        assert np.all(points > -1.0) and np.all(points < 1.0)
        
        # 检查权重为正
        assert np.all(weights > 0)
        
        # 检查权重之和约为2（积分区间长度）
        assert abs(np.sum(weights) - 2.0) < 1e-10

    def test_gauss_lobatto(self):
        """测试Gauss-Lobatto点集生成"""
        n = 4
        points, weights = gauss_lobatto(n)
        
        # 检查点数
        assert len(points) == n
        
        # 检查端点精确为-1和1
        assert abs(points[0] + 1.0) < 1e-14
        assert abs(points[-1] - 1.0) < 1e-14
        
        # 检查点集有序
        assert np.all(np.diff(points) > 0)


class TestDiffMatrix:
    """测试微分矩阵"""

    def test_diff_matrix_constant_function(self):
        """测试常数函数的导数应为零"""
        n = 4
        points, _ = gauss_legendre(n)
        D = compute_diff_matrix_1d(points)
        
        # 常数函数
        f = np.ones(n)
        df = D @ f
        
        # 导数应接近零
        assert np.allclose(df, 0.0, atol=1e-10)

    def test_diff_matrix_linear_function(self):
        """测试线性函数 f(x) = x 的导数应为1"""
        n = 4
        points, _ = gauss_legendre(n)
        D = compute_diff_matrix_1d(points)
        
        # 线性函数 f(x) = x
        f = points.copy()
        df = D @ f
        
        # 导数应接近1
        assert np.allclose(df, 1.0, atol=1e-10)

    def test_diff_matrix_quadratic_function(self):
        """测试二次函数 f(x) = x^2 的导数应为2x"""
        n = 4
        points, _ = gauss_legendre(n)
        D = compute_diff_matrix_1d(points)
        
        # 二次函数 f(x) = x^2
        f = points ** 2
        df = D @ f
        
        # 导数应为 2x
        expected = 2 * points
        assert np.allclose(df, expected, atol=1e-10)


class TestInterpolation:
    """测试插值矩阵"""

    def test_interpolation_identity(self):
        """测试在SPs处的插值应保持原值"""
        n_sps = 3
        sps, _ = gauss_legendre(n_sps)
        
        # 构造单位插值矩阵（SPs到SPs）
        L = compute_interpolation_matrix(sps, sps)
        
        # 应为单位矩阵
        assert np.allclose(L, np.eye(n_sps), atol=1e-10)

    def test_interpolation_accuracy(self):
        """测试插值精度"""
        n_sps = 4
        sps, _ = gauss_legendre(n_sps)
        
        # 在SPs上定义一个多项式函数
        f_sps = sps ** 2
        
        # 在更多点上评估
        n_fps = 5
        fps = np.linspace(-1, 1, n_fps)
        
        # 计算插值矩阵
        L = compute_interpolation_matrix(sps, fps)
        
        # 插值
        f_fps = L @ f_sps
        
        # 期望值
        expected = fps ** 2
        
        # 对于二次多项式，3个点应该能精确插值
        assert np.allclose(f_fps, expected, atol=1e-10)


class TestFROperators:
    """测试FR算子生成器"""

    def test_generate_operators_p1(self):
        """测试P=1阶算子生成"""
        order = 1
        ops = generate_fr_operators(order)
        
        n = order + 1  # SPs数量
        n_fps = n + 1  # FPs数量（Lobatto）
        
        # 检查形状
        assert ops.D_1d.shape == (n, n)
        assert ops.D_3d.shape == (n**3, n**3, 3)
        assert ops.L_interp.shape == (n_fps, n)
        assert ops.g_left.shape == (n,)
        assert ops.g_right.shape == (n,)

    def test_generate_operators_p2(self):
        """测试P=2阶算子生成"""
        order = 2
        ops = generate_fr_operators(order)
        
        n = order + 1
        n_fps = n + 1
        
        # 检查形状
        assert ops.D_1d.shape == (n, n)
        assert ops.D_3d.shape == (n**3, n**3, 3)
        assert ops.L_interp.shape == (n_fps, n)
        assert ops.g_left.shape == (n,)
        assert ops.g_right.shape == (n,)

    def test_correction_weights_properties(self):
        """测试 Radau/VCJH 校正函数导数 g_L'/g_R' 的定义性质。

        g_left/g_right 存的是校正多项式在各 SP 处的**导数**值（不是
        校正多项式本身的值——那样的量在所有 SP 处恒为零，见
        matrix_operators.compute_correction_weights 的文档说明）。
        用重新独立求解校正多项式（单项式基 + 边界条件 + 正交性约束）
        再解析求导，交叉验证 compute_correction_weights 的输出，
        并验证 g_R(x) = g_L(-x) 对称性蕴含的 g_R'(x_i) = -g_L'(-x_i)
        （SPs 是关于原点对称的 Gauss-Legendre 点，故 x_i -> -x_i
        对应索引反转）。
        """
        for order in [0, 1, 2, 3]:
            n = order + 1
            ops = generate_fr_operators(order)
            sps, _ = gauss_legendre(n)

            assert np.all(np.isfinite(ops.g_left))
            assert np.all(np.isfinite(ops.g_right))
            assert ops.g_left.shape == (n,)
            assert ops.g_right.shape == (n,)

            # 独立重新求解校正多项式系数，验证边界条件与正交性
            def moment(j):
                return 0.0 if j % 2 == 1 else 2.0 / (j + 1)

            A = [[(-1.0) ** k for k in range(n + 1)]] + [[1.0] * (n + 1)]
            b = [1.0, 0.0]
            for m in range(0, n - 1):
                A.append([moment(m + k) for k in range(n + 1)])
                b.append(0.0)
            c_left = np.linalg.solve(np.array(A), np.array(b))

            def polyval(c, x):
                return sum(c[k] * x**k for k in range(len(c)))

            assert abs(polyval(c_left, -1.0) - 1.0) < 1e-10
            assert abs(polyval(c_left, 1.0)) < 1e-10
            xs_hi, ws_hi = np.polynomial.legendre.leggauss(20)
            for m in range(0, n - 1):
                integral = np.sum(ws_hi * polyval(c_left, xs_hi) * xs_hi**m)
                assert abs(integral) < 1e-10, f"g_L not orthogonal to degree-{m} at P={order}"

            dc_left = np.array([c_left[k] * k for k in range(1, n + 1)])
            gL_prime_ref = np.array([polyval(dc_left, x) for x in sps])
            assert np.allclose(ops.g_left, gL_prime_ref, atol=1e-10), (
                f"g_left mismatch vs independently-derived correction polynomial at P={order}"
            )

            # g_R(x) = g_L(-x) => g_R'(x_i) = -g_L'(-x_i); SPs 关于原点对称，
            # 故 -x_i 对应反转索引 x_{n-1-i}
            gR_prime_expected = -gL_prime_ref[::-1]
            assert np.allclose(ops.g_right, gR_prime_expected, atol=1e-10), (
                f"g_right does not match g_L(-x) symmetry at P={order}"
            )


class TestFRState:
    """测试FR状态数据结构"""

    def test_fr_state_initialization(self):
        """Test S-01: FRState initialization with turbulence variables."""
        n_cells, n_sps = 10, 8
        state = FRState(n_cells, n_sps, n_vars=7) # V2.0 默认支持 SST 模型
        
        assert state.U.shape == (n_cells, n_sps, 7)
        assert state.Q.shape == (n_cells, n_sps, 7)
        assert np.all(state.U == 0)

    def test_fr_state_uniform_flow(self):
        """测试均匀流场初始化"""
        from autoflowcfd.core.fr_solver.state import FRState
        
        n_cells = 5
        n_sps = 8
        state = FRState(n_cells, n_sps)
        
        # 初始化均匀流场
        state.initialize_uniform(rho=1.0, u=10.0, v=0.0, w=0.0, p=101325.0)
        
        # 检查密度
        assert np.allclose(state.U[:, :, 0], 1.0)
        
        # 检查动量
        assert np.allclose(state.U[:, :, 1], 10.0)
        assert np.allclose(state.U[:, :, 2], 0.0)
        assert np.allclose(state.U[:, :, 3], 0.0)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
