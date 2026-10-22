"""后处理模块的单元测试。"""

import unittest
import numpy as np
from pathlib import Path
import tempfile
import json
import csv

from autoflowcfd.grid.schema import (
    GridData, NodeArray, CellArray, BoundaryMap, GridMetadata,
)
from autoflowcfd.postprocess.solution_vector import SolutionVector
from autoflowcfd.postprocess import (
    AerodynamicCoefficients,
    AerodynamicForces,
    VTKExporter,
    ConvergenceAnalyzer,
    SimulationReport,
    TransientStatistics,
    PressurePSD,
)


class TestAerodynamicCoefficients(unittest.TestCase):
    """气动力系数数据类的测试"""

    def test_default_values(self):
        """系数默认值为零"""
        coeffs = AerodynamicCoefficients()
        self.assertEqual(coeffs.Cd, 0.0)
        self.assertEqual(coeffs.Cl, 0.0)
        self.assertEqual(coeffs.Cm, 0.0)

    def test_to_dict(self):
        """转换为字典"""
        coeffs = AerodynamicCoefficients(Cd=0.3, Cl=0.1, Cm=-0.05)
        d = coeffs.to_dict()
        self.assertAlmostEqual(d['Cd'], 0.3)
        self.assertAlmostEqual(d['Cl'], 0.1)
        self.assertAlmostEqual(d['Cm'], -0.05)

    def test_string_representation(self):
        """字符串表示包含全部系数"""
        coeffs = AerodynamicCoefficients(Cd=0.28)
        s = str(coeffs)
        self.assertIn("Cd", s)
        self.assertIn("0.28", s)


class TestAerodynamicForces(unittest.TestCase):
    """气动力数据类的测试"""

    def test_default_values(self):
        """力的默认值为零"""
        forces = AerodynamicForces()
        self.assertEqual(forces.drag_force, 0.0)
        self.assertEqual(forces.lift_force, 0.0)

    def test_to_dict(self):
        """转换为字典"""
        forces = AerodynamicForces(drag_force=150.0, lift_force=-20.0)
        d = forces.to_dict()
        self.assertAlmostEqual(d['drag_force'], 150.0)
        self.assertAlmostEqual(d['lift_force'], -20.0)


class TestCoefficientCalculatorRemoved(unittest.TestCase):
    """第三轮评审整改：V1 CoefficientCalculator 已移除（依赖从未存在的
    get_face_data()、全场 ρRT 均值压力伪积分，系数恒为 0）。生产路径是
    postprocess/fr_coefficients.py 的 FR 原生积分（其单元测试见
    tests/unit/test_fr_coefficients.py 等）。这里只验证失效入口确实不存在。"""

    def test_removed_from_module(self):
        import autoflowcfd.postprocess.coefficients as coef_mod
        self.assertFalse(hasattr(coef_mod, "CoefficientCalculator"))

    def test_removed_from_package_exports(self):
        import autoflowcfd.postprocess as postprocess_pkg
        self.assertNotIn("CoefficientCalculator", postprocess_pkg.__all__)


class TestVTKExporter(unittest.TestCase):
    """Test VTK exporter"""

    def setUp(self):
        """准备测试夹具"""
        nodes = NodeArray(
            x=np.array([0.0, 1.0, 2.0]),
            y=np.array([0.0, 0.0, 0.0]),
            z=np.array([0.0, 0.0, 0.0])
        )
        cells = CellArray(
            connectivity=np.array([[0, 1, 2]]),
            cell_type=np.array([0])
        )
        boundaries = BoundaryMap(
            groups={},
            bc_types={}
        )
        metadata = GridMetadata(
            node_count=3,
            cell_count=1,
            boundary_groups=[],
            file_format="v24"
        )
        self.grid_data = GridData(
            nodes=nodes,
            cells=cells,
            boundaries=boundaries,
            metadata=metadata
        )
        U = np.array([[1.2, 1.2 * 30.0, 0.0, 0.0, 101325.0 / 0.4 + 0.5 * 1.2 * 30.0 ** 2]])
        self.solution = SolutionVector(U)

    def test_export_legacy_format(self):
        """导出 legacy VTK 格式"""
        exporter = VTKExporter(self.grid_data, self.solution)

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "test.vtk"
            result = exporter.export(str(output_path), fields=['velocity', 'pressure'])

            self.assertTrue(result.exists())
            self.assertEqual(result.suffix, '.vtk')

            # Check file content
            with open(result, 'r') as f:
                content = f.read()
                self.assertIn("vtk DataFile Version 3.0", content)
                self.assertIn("DATASET UNSTRUCTURED_GRID", content)

    def test_export_with_custom_fields(self):
        """导出指定的场"""
        exporter = VTKExporter(self.grid_data, self.solution)

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "test.vtk"
            exporter.export(str(output_path), fields=['velocity'])

            with open(output_path, 'r') as f:
                content = f.read()
                self.assertIn("VECTORS Velocity", content)

    def test_export_invalid_field(self):
        """拒绝非法的场名"""
        exporter = VTKExporter(self.grid_data, self.solution)

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "test.vtk"
            with self.assertRaises(ValueError):
                exporter.export(str(output_path), fields=['invalid_field'])

    def test_export_invalid_format(self):
        """拒绝非法的格式"""
        exporter = VTKExporter(self.grid_data, self.solution)

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "test.vtu"
            with self.assertRaises(ValueError):
                exporter.export(str(output_path), format='invalid')


class TestConvergenceAnalyzer(unittest.TestCase):
    """Test convergence analyzer"""

    def setUp(self):
        """准备测试夹具"""
        self.analyzer = ConvergenceAnalyzer()

    def test_add_iteration(self):
        """添加迭代数据"""
        self.analyzer.add_iteration(
            iteration=1,
            residuals={'continuity': 1e-2, 'momentum': 1e-3},
            cfl=5.0
        )
        self.assertEqual(len(self.analyzer.history), 1)
        self.assertEqual(self.analyzer.history[0].iteration, 1)

    def test_export_csv(self):
        """把收敛历史导出为 CSV"""
        # Add some iterations
        for i in range(5):
            self.analyzer.add_iteration(
                iteration=i+1,
                residuals={'continuity': 10**(-i-2)},
                cfl=5.0 + i
            )

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "convergence.csv"
            result = self.analyzer.export_csv(str(output_path))

            self.assertTrue(result.exists())

            # Verify CSV content
            with open(result, 'r') as f:
                reader = csv.reader(f)
                rows = list(reader)
                self.assertGreater(len(rows), 1)  # Header + data
                self.assertIn('iteration', rows[0])

    def test_get_summary(self):
        """取仿真摘要"""
        # Add iterations
        for i in range(10):
            self.analyzer.add_iteration(
                iteration=i+1,
                residuals={'continuity': 10**(-i-2)},
                cfl=5.0
            )

        summary = self.analyzer.get_summary(computation_time=100.0)
        self.assertEqual(summary.total_iterations, 10)
        self.assertEqual(summary.computation_time, 100.0)


class TestSimulationReport(unittest.TestCase):
    """仿真报告生成器的测试"""

    def setUp(self):
        """准备测试夹具"""
        self.config = {'backend': 'cpu', 'order': 2}
        self.analyzer = ConvergenceAnalyzer()

        # Add some iterations
        for i in range(5):
            self.analyzer.add_iteration(
                iteration=i+1,
                residuals={'continuity': 10**(-i-2)},
                cfl=5.0
            )

        self.report = SimulationReport(self.config, self.analyzer)

    def test_generate_report(self):
        """生成 JSON 报告"""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "report.json"
            result = self.report.generate(str(output_path), computation_time=60.0)

            self.assertTrue(result.exists())

            # Verify JSON content
            with open(result, 'r') as f:
                report_data = json.load(f)
                self.assertIn('metadata', report_data)
                self.assertIn('configuration', report_data)
                self.assertIn('summary', report_data)
                self.assertEqual(report_data['metadata']['software'], 'AutoFlowCFD')


class TestTransientStatistics(unittest.TestCase):
    """瞬态统计：节点时均/RMS 与 numpy 直接计算一致（Welford，ddof=1）。"""

    def setUp(self):
        nodes = NodeArray(x=np.array([0.0, 1.0, 2.0]), y=np.array([0.0, 0.0, 0.0]), z=np.array([0.0, 0.0, 0.0]))
        cells = CellArray(connectivity=np.array([[0, 1, 2]]), cell_type=np.array([0]))
        self.grid_data = GridData(nodes=nodes, cells=cells, boundaries=BoundaryMap(groups={}, bc_types={}),
                                  metadata=GridMetadata(node_count=3, cell_count=1, boundary_groups=[],
                                                        file_format="v24"))

    def _sample(self, u, p):
        rho = 1.2
        U = np.array([[rho, rho * u, 0.0, 0.0, p / 0.4 + 0.5 * rho * u * u]])
        return SolutionVector(U)

    def test_mean_and_rms_match_direct_computation(self):
        stats = TransientStatistics(self.grid_data)
        us = [30.0, 31.5, 29.0, 33.0, 28.5]
        ps = [101300.0, 101350.0, 101280.0, 101400.0, 101250.0]
        for i, (u, p) in enumerate(zip(us, ps)):
            stats.accumulate(self._sample(u, p), time=0.01 * i)
        result = stats.compute_statistics()
        self.assertEqual(result.num_samples, 5)
        self.assertAlmostEqual(result.sampling_time, 0.04)
        np.testing.assert_allclose(result.mean_fields['velocity_u'], np.mean(us), rtol=1e-12)
        np.testing.assert_allclose(result.rms_fields['velocity_u_rms'], np.std(us, ddof=1), rtol=1e-12)
        np.testing.assert_allclose(result.mean_fields['pressure'], np.mean(ps), rtol=1e-12)
        np.testing.assert_allclose(result.rms_fields['pressure_rms'], np.std(ps, ddof=1), rtol=1e-9)

    def test_first_sample_counts(self):
        """此前 Welford 在计数自增前更新（第 2 个样本直接覆盖均值），首样本被丢弃。"""
        stats = TransientStatistics(self.grid_data)
        stats.accumulate(self._sample(10.0, 101325.0), time=0.0)
        stats.accumulate(self._sample(20.0, 101325.0), time=1.0)
        np.testing.assert_allclose(stats.compute_statistics().mean_fields['velocity_u'], 15.0, rtol=1e-12)

    def test_sample_with_wrong_cell_count_is_rejected(self):
        stats = TransientStatistics(self.grid_data)
        with self.assertRaises(ValueError):
            stats.accumulate(SolutionVector(np.ones((2, 5))), time=0.0)

    def test_compute_statistics_no_samples(self):
        with self.assertRaises(RuntimeError):
            TransientStatistics(self.grid_data).compute_statistics()

    def test_coefficient_statistics(self):
        stats = TransientStatistics(self.grid_data)
        cds = [0.30, 0.32, 0.29, 0.31]
        for cd in cds:
            stats.accumulate_coefficients(cd, 0.1, 0.0)
        out = stats.get_coefficient_statistics()
        self.assertAlmostEqual(out['Cd']['mean'], np.mean(cds), places=14)
        self.assertAlmostEqual(out['Cd']['std'], np.std(cds, ddof=1), places=14)


class TestPressurePSD(unittest.TestCase):
    """压力 PSD 分析器（Welch 法，I-158）"""

    def setUp(self):
        self.monitor_points = [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0)]
        self.dt = 1e-4
        self.psd = PressurePSD(self.monitor_points)

    def _fill(self, values, dt=None):
        dt = self.dt if dt is None else dt
        for k, v in enumerate(values):
            self.psd.add_sample(time=k * dt, pressures=[v, v])

    def test_empty_monitor_points(self):
        with self.assertRaises(ValueError):
            PressurePSD([])

    def test_add_sample(self):
        self.psd.add_sample(time=0.0, pressures=[101325.0, 101326.0])
        self.assertEqual(len(self.psd.times), 1)
        self.assertEqual(len(self.psd.pressure_history[0]), 1)

    def test_add_sample_length_mismatch(self):
        with self.assertRaises(ValueError):
            self.psd.add_sample(time=0.0, pressures=[101325.0])

    def test_times_must_increase(self):
        self.psd.add_sample(time=1.0, pressures=[0.0, 0.0])
        with self.assertRaises(ValueError):
            self.psd.add_sample(time=1.0, pressures=[0.0, 0.0])

    def test_compute_psd_insufficient_samples(self):
        with self.assertRaises(RuntimeError):
            self.psd.compute_psd(0)

    def test_non_uniform_sampling_is_an_error(self):
        """自适应时间步 / 跳过的样本：频率轴会错，必须报错（此前只用构造时的 dt，从不检查）。"""
        for t in np.r_[np.arange(20) * self.dt, 20 * self.dt + 0.5 * self.dt]:
            self.psd.add_sample(time=t, pressures=[0.0, 0.0])
        with self.assertRaisesRegex(ValueError, "等间隔"):
            self.psd.compute_psd(0)

    def test_parseval_white_noise(self):
        """单边 PSD 的积分等于脉动方差（窗能量修正正确）。"""
        rng = np.random.default_rng(0)
        self._fill(101325.0 + rng.normal(0.0, 3.0, 4096))
        freqs, psd = self.psd.compute_psd(0)
        self.assertAlmostEqual(np.sum(psd) * (freqs[1] - freqs[0]) / 9.0, 1.0, delta=0.05)

    def test_nyquist_bin_is_not_doubled(self):
        """全部能量在 Nyquist 频率的序列（+1, -1 交替，方差 1）：积分必须为 1（此前 Nyquist 频点也乘 2，得 2）。"""
        self._fill(np.where(np.arange(64) % 2 == 0, 1.0, -1.0))
        freqs, psd = self.psd.compute_psd(0)
        self.assertAlmostEqual(np.sum(psd) * (freqs[1] - freqs[0]), 1.0, places=6)

    def test_welch_averaging_reduces_variance(self):
        """白噪声谱的相对离散度随分段平均下降（此前单段周期图约为 1，不随样本数下降）。"""
        rng = np.random.default_rng(1)
        self._fill(rng.normal(0.0, 1.0, 8192))
        _, psd = self.psd.compute_psd(0)
        self.assertLess(np.std(psd[1:-1]) / np.mean(psd[1:-1]), 0.5)

    def test_find_dominant_frequency(self):
        self._fill(101325.0 + 10.0 * np.sin(2 * np.pi * 100 * np.arange(1000) * self.dt))
        freq, _ = self.psd.find_dominant_frequency(0, min_freq=50, max_freq=150)
        self.assertGreater(freq, 90)
        self.assertLess(freq, 110)

    def test_invalid_point_index(self):
        """拒绝非法的点索引"""
        with self.assertRaises(IndexError):
            self.psd.compute_psd(10)


if __name__ == '__main__':
    unittest.main()
