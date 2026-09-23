"""AutoFlowCFD V2.0 - Persson-Peraire 模态传感器 + 局部人工粘性单元测试。

见 core/fr_operators/artificial_viscosity.py 模块文档：这是一个默认关闭
（opt-in）的新增稳定性能力，用解本身的模态谱衰减速率（而非残差量级）
判断单元是否欠分辨率——2026-08-29 cube_demo 真实网格调查发现现有的
模态滤波器/机制3残差异常检测都存在结构性局限后，作为行业调研（PyFR
稳定性工具箱第三层）指向的新方向引入。公式已对照 mirgecom（现役生产级
DG 代码）文档核实，不是凭记忆重新推导。
"""

import numpy as np

# 私有构造函数从它**真正的**所在模块导入，而不是靠包 `__init__`
# 的 re-export（那里只 re-export 公开名；人工粘性模块 2026-09-20
# 拆成子包，见该包文档）。
from autoflowcfd.core.fr_operators.artificial_viscosity.sensor_operators import (
    _build_native_prism_sensor_operators,
    _build_native_tet_sensor_operators,
)
from autoflowcfd.core.fr_operators.artificial_viscosity import (
    compute_artificial_viscosity_ramp,
    compute_persson_peraire_artificial_viscosity,
    compute_persson_peraire_sensor_native_prism,
    compute_persson_peraire_sensor_native_tet,
)
from autoflowcfd.fr.quadrature_points import gauss_legendre


def _native_tet_synth(order: int):
    """返回 `(V, top_idx, n_native)`：原生四面体 PKD 基的"模态系数 -> 节点值"
    矩阵与一个顶模态下标。

    传感器算子只给 `V_inv`（它只需要"节点值 -> 模态"这个方向），而这里要
    **反过来**造一个"顶模态被污染了多少"完全可控的解场，所以取它的逆。
    """
    V_inv, top_mask, n_native = _build_native_tet_sensor_operators(order)
    return np.linalg.inv(V_inv), int(np.nonzero(top_mask)[0][0]), n_native


def _native_prism_synth(order: int):
    """原生棱柱版，返回 `(V, top_idx, n_native)`。理由同上。"""
    V_inv, _mass, top_mask, n_native = (
        _build_native_prism_sensor_operators(order))
    return np.linalg.inv(V_inv), int(np.nonzero(top_mask)[0][0]), n_native


class TestSensorOrderZeroShortCircuit:
    """order==0 没有"上一阶"可截断，传感器必须恒定输出 -inf（永不触发
    人工粘性），不应该尝试构造任何模态算子。
    """

    def test_order_0_returns_negative_infinity(self):
        field = np.full((10, 1), 42.0)
        s_e = compute_persson_peraire_sensor_native_tet(field, 0)
        assert np.all(np.isneginf(s_e))
        ramp = compute_artificial_viscosity_ramp(s_e, 0)
        assert np.all(ramp == 0.0)


class TestSensorConstantFieldIsSmooth:
    """常数场没有任何高阶模态内容，传感器必须判定为极度光滑（s_e 远
    低于任意合理阈值），ramp 恒为 0。
    """

    def test_tet_constant_field_ramp_zero(self):
        order = 2
        _V, _ti, n_native = _native_tet_synth(order)
        field = np.full((5, n_native), 42.0)
        s_e = compute_persson_peraire_sensor_native_tet(field, order)
        ramp = compute_artificial_viscosity_ramp(s_e, order)
        assert np.all(ramp == 0.0)
        assert np.all(s_e < -10.0)

    def test_prism_constant_field_ramp_zero(self):
        order = 2
        _V, _ti, n_native = _native_prism_synth(order)
        field = np.full((5, n_native), -13.5)
        s_e = compute_persson_peraire_sensor_native_prism(field, order)
        ramp = compute_artificial_viscosity_ramp(s_e, order)
        assert np.all(ramp == 0.0)


class TestSensorMonotonicInTopModeAmplitude:
    """核心判据：传感器对"污染进最高阶模态的能量占比"必须单调响应——
    幅度越大，s_e 越大，最终越过阈值后 ramp 从 0 平滑爬升到 1。这是
    传感器与残差量级检测机制（机制3）本质不同的地方：这里测的是解
    本身的模态谱结构，不是残差绝对/相对大小。
    """

    def test_tet_sensor_monotonic_and_thresholds_correctly(self):
        order = 2
        V, top_idx, n_sps = _native_tet_synth(order)

        amplitudes = [0.0, 0.01, 0.1, 1.0, 3.0, 10.0, 30.0, 100.0]
        s_e_values = []
        for amp in amplitudes:
            modal = np.zeros(n_sps)
            modal[0] = 100.0
            modal[top_idx] = amp
            field = (V @ modal)[np.newaxis, :]
            s_e = compute_persson_peraire_sensor_native_tet(field, order)
            s_e_values.append(float(s_e[0]))

        # strictly increasing in amplitude
        assert all(a < b for a, b in zip(s_e_values, s_e_values[1:])), s_e_values

        ramps = compute_artificial_viscosity_ramp(np.array(s_e_values), order)
        # low amplitude -> untriggered, high amplitude -> (near-)fully triggered
        assert ramps[0] == 0.0
        assert ramps[-1] > 0.99
        # ramp itself must be monotonically non-decreasing
        assert np.all(np.diff(ramps) >= -1e-12)

    def test_tiny_contamination_does_not_trigger_but_large_does(self):
        order = 2
        V, top_idx, n_sps = _native_tet_synth(order)

        modal_tiny = np.zeros(n_sps)
        modal_tiny[0] = 100.0
        modal_tiny[top_idx] = 1e-6
        field_tiny = (V @ modal_tiny)[np.newaxis, :]
        s_e_tiny = compute_persson_peraire_sensor_native_tet(field_tiny, order)
        assert compute_artificial_viscosity_ramp(s_e_tiny, order)[0] == 0.0

        modal_big = np.zeros(n_sps)
        modal_big[0] = 100.0
        modal_big[top_idx] = 30.0
        field_big = (V @ modal_big)[np.newaxis, :]
        s_e_big = compute_persson_peraire_sensor_native_tet(field_big, order)
        assert compute_artificial_viscosity_ramp(s_e_big, order)[0] > 0.5


    def test_native_prism_sensor_monotonic_with_diagonal_mass(self):
        """原生棱柱版同一判据。

        **单独测它的理由**：原生棱柱基在参考棱柱上正交但**不归一**
        （P1 实测对角质量是 8/2.667/4/1.333/5.333/1.778 这样一组各不相同
        的数），所以 L2 能量必须写成 `sum_m M_mm chat_m^2`。四面体那条能
        直接 `sum(chat^2)` 是因为 PKD 正交归一；漏掉对角质量不会报错、
        只会给出错的能量占比，所以要有一条只覆盖这条路径的判据。
        """
        order = 2
        V, top_idx, n_sps = _native_prism_synth(order)

        # 幅度上界比四面体那条大：同一幅度下两条基给出的能量占比不同
        # （模态数与对角质量都不同），原生棱柱在 amp=100 时 ramp 只到
        # 0.810，要到 amp~1000 才接近完全触发。判据本身（单调 + 起点为
        # 0 + 足够大的污染必须完全触发）不变。
        amplitudes = [0.0, 0.01, 0.1, 1.0, 3.0, 10.0, 30.0, 100.0,
                      300.0, 1000.0]
        s_e_values = []
        for amp in amplitudes:
            modal = np.zeros(n_sps)
            modal[0] = 100.0
            modal[top_idx] = amp
            field = (V @ modal)[np.newaxis, :]
            s_e = compute_persson_peraire_sensor_native_prism(field, order)
            s_e_values.append(float(s_e[0]))

        assert all(a < b for a, b in zip(s_e_values, s_e_values[1:])), s_e_values
        ramps = compute_artificial_viscosity_ramp(np.array(s_e_values), order)
        assert ramps[0] == 0.0
        assert ramps[-1] > 0.99
        assert np.all(np.diff(ramps) >= -1e-12)


class TestRampPiecewiseFormula:
    """直接核对 mirgecom 文档给出的分段公式数值（不依赖传感器，纯测试
    compute_artificial_viscosity_ramp 本身的数学实现）。
    """

    def test_below_threshold_is_zero(self):
        order = 2
        s0 = -4.0 * np.log10(order)
        s_e = np.array([s0 - 2.0])  # kappa=1.0 default, well below s0-kappa
        ramp = compute_artificial_viscosity_ramp(s_e, order)
        assert ramp[0] == 0.0

    def test_above_threshold_is_one(self):
        order = 2
        s0 = -4.0 * np.log10(order)
        s_e = np.array([s0 + 2.0])
        ramp = compute_artificial_viscosity_ramp(s_e, order)
        assert ramp[0] == 1.0

    def test_at_s0_is_half(self):
        order = 2
        s0 = -4.0 * np.log10(order)
        s_e = np.array([s0])
        ramp = compute_artificial_viscosity_ramp(s_e, order)
        assert abs(ramp[0] - 0.5) < 1e-12


class TestFullPipelineWithSyntheticSolver:
    """`compute_persson_peraire_artificial_viscosity` 端到端测试，用一个
    最小的假 solver（namespace 模拟真实 FRSolver 接口）验证：健康
    （常数）流场输出全零人工粘性，output 形状/量纲正确。
    """

    def test_uniform_flow_produces_zero_artificial_viscosity(self):
        from types import SimpleNamespace

        order = 2
        # 端到端流水线拿到的是**带零填充的全局宽度** `(order+1)^3`
        # （原生基每单元只有 `(p+1)^2(p+2)/2` 个真实自由度，其余槽位冻结
        # 在初值），所以这里不能用"真实自由度数"。
        n_sps = (order + 1) ** 3
        n_cells = 6
        Q = np.zeros((n_cells, n_sps, 5))
        Q[:, :, 0] = 1.225  # rho
        Q[:, :, 1] = 30.0  # u
        Q[:, :, 4] = 101325.0  # p

        mesh = SimpleNamespace(
            n_prism_cells=3,
            cell_volumes=np.full(n_cells, 1e-4),
        )
        solver = SimpleNamespace(
            mesh=mesh,
            state=SimpleNamespace(Q=Q),
            current_order=order,
            order=order,
        )

        eps = compute_persson_peraire_artificial_viscosity(solver)
        assert eps.shape == (n_cells, n_sps)
        assert np.all(eps == 0.0)

    def test_order_0_solver_returns_zero(self):
        from types import SimpleNamespace

        n_cells, n_sps = 4, 1
        Q = np.zeros((n_cells, n_sps, 5))
        Q[:, :, 0] = 1.225
        mesh = SimpleNamespace(n_prism_cells=2, cell_volumes=np.full(n_cells, 1e-4))
        solver = SimpleNamespace(mesh=mesh, state=SimpleNamespace(Q=Q), current_order=0, order=0)

        eps = compute_persson_peraire_artificial_viscosity(solver)
        assert eps.shape == (n_cells, n_sps)
        assert np.all(eps == 0.0)
