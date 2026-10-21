"""
AutoFlowCFD V2.0 - FR 模块单元测试

本模块测试 Flux Reconstruction 方法的核心组件。
"""

import numpy as np
import pytest
from autoflowcfd.fr import (
    generate_fr_operators,
    gauss_legendre,
    gauss_lobatto,
)
from autoflowcfd.fr.native_padding import sps_width
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


class TestFROperators:
    """测试FR算子生成器"""

    def test_generate_operators_p1(self):
        """测试P=1阶算子生成"""
        order = 1
        ops = generate_fr_operators(order)

        # 检查形状
        assert ops.n_sps == sps_width(order)
        assert ops.D_native_prism_padded.shape == (sps_width(order), sps_width(order), 3)

    def test_generate_operators_p2(self):
        """测试P=2阶算子生成"""
        order = 2
        ops = generate_fr_operators(order)

        # 检查形状
        assert ops.n_sps == sps_width(order)
        assert ops.D_native_prism_padded.shape == (sps_width(order), sps_width(order), 3)


class TestFRState:
    """测试FR状态数据结构"""

    def test_fr_state_initialization(self):
        """S-01：状态只含平均流 5 个守恒变量（湍流场在模型对象上；SST 的两个历史槽位 2026-10-05 删除）。"""
        n_cells, n_sps = 10, 8
        state = FRState(n_cells, n_sps)

        assert state.U.shape == (n_cells, n_sps, 5)
        assert state.Q.shape == (n_cells, n_sps, 5)
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
