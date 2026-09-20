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
    _build_sensor_operators,
)
from autoflowcfd.core.fr_operators.artificial_viscosity import (
    compute_artificial_viscosity_ramp,
    compute_persson_peraire_artificial_viscosity,
    compute_persson_peraire_sensor,
)
from autoflowcfd.fr.quadrature_points import gauss_legendre


def _ref_cube_sps(order: int) -> np.ndarray:
    n1d = order + 1
    sps_1d, _ = gauss_legendre(n1d)
    xx, yy, zz = np.meshgrid(sps_1d, sps_1d, sps_1d, indexing="ij")
    return np.column_stack([xx.ravel(), yy.ravel(), zz.ravel()])


class TestSensorOrderZeroShortCircuit:
    """order==0 没有"上一阶"可截断，传感器必须恒定输出 -inf（永不触发
    人工粘性），不应该尝试构造任何模态算子。
    """

    def test_order_0_returns_negative_infinity(self):
        ref = _ref_cube_sps(0)
        field = np.full((10, 1), 42.0)
        s_e = compute_persson_peraire_sensor(field, "tet", 0, ref)
        assert np.all(np.isneginf(s_e))
        ramp = compute_artificial_viscosity_ramp(s_e, 0)
        assert np.all(ramp == 0.0)


class TestSensorConstantFieldIsSmooth:
    """常数场没有任何高阶模态内容，传感器必须判定为极度光滑（s_e 远
    低于任意合理阈值），ramp 恒为 0。
    """

    def test_tet_constant_field_ramp_zero(self):
        order = 2
        ref = _ref_cube_sps(order)
        field = np.full((5, ref.shape[0]), 42.0)
        s_e = compute_persson_peraire_sensor(field, "tet", order, ref)
        ramp = compute_artificial_viscosity_ramp(s_e, order)
        assert np.all(ramp == 0.0)
        assert np.all(s_e < -10.0)

    def test_prism_constant_field_ramp_zero(self):
        order = 2
        ref = _ref_cube_sps(order)
        field = np.full((5, ref.shape[0]), -13.5)
        s_e = compute_persson_peraire_sensor(field, "prism", order, ref)
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
        ref = _ref_cube_sps(order)
        V, V_inv, qw, top_mask = _build_sensor_operators("tet", order, ref)
        n_sps = V.shape[0]
        top_idx = np.nonzero(top_mask)[0][0]

        amplitudes = [0.0, 0.01, 0.1, 1.0, 3.0, 10.0, 30.0, 100.0]
        s_e_values = []
        for amp in amplitudes:
            modal = np.zeros(n_sps)
            modal[0] = 100.0
            modal[top_idx] = amp
            field = (V @ modal)[np.newaxis, :]
            s_e = compute_persson_peraire_sensor(field, "tet", order, ref)
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
        ref = _ref_cube_sps(order)
        V, V_inv, qw, top_mask = _build_sensor_operators("tet", order, ref)
        n_sps = V.shape[0]
        top_idx = np.nonzero(top_mask)[0][0]

        modal_tiny = np.zeros(n_sps)
        modal_tiny[0] = 100.0
        modal_tiny[top_idx] = 1e-6
        field_tiny = (V @ modal_tiny)[np.newaxis, :]
        s_e_tiny = compute_persson_peraire_sensor(field_tiny, "tet", order, ref)
        assert compute_artificial_viscosity_ramp(s_e_tiny, order)[0] == 0.0

        modal_big = np.zeros(n_sps)
        modal_big[0] = 100.0
        modal_big[top_idx] = 30.0
        field_big = (V @ modal_big)[np.newaxis, :]
        s_e_big = compute_persson_peraire_sensor(field_big, "tet", order, ref)
        assert compute_artificial_viscosity_ramp(s_e_big, order)[0] > 0.5


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
        ref = _ref_cube_sps(order)
        n_sps = ref.shape[0]
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
