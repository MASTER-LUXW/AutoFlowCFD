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

        # 检查形状
        assert ops.D_1d.shape == (n, n)
        assert ops.D_3d.shape == (n**3, n**3, 3)

    def test_generate_operators_p2(self):
        """测试P=2阶算子生成"""
        order = 2
        ops = generate_fr_operators(order)
        
        n = order + 1

        # 检查形状
        assert ops.D_1d.shape == (n, n)
        assert ops.D_3d.shape == (n**3, n**3, 3)


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
