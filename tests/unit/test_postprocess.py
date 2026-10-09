"""后处理模块的单元测试。"""

import unittest
import numpy as np
from pathlib import Path
import tempfile
import json
import csv

from autoflowcfd.grid.structures import (
    GridData, NodeArray, CellArray, BoundaryMap, GridMetadata,
)
from autoflowcfd.core.backend.base import SolutionVector
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
        self.solution = SolutionVector()
    
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
    """瞬态统计计算器的测试"""
    
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
        boundaries = BoundaryMap(groups={}, bc_types={})
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
        self.solution = SolutionVector()
    
    def test_initialization(self):
        """统计计算器初始化"""
        stats = TransientStatistics(self.grid_data, window_size=50)
        self.assertEqual(stats.window_size, 50)
        self.assertEqual(stats.n_samples, 0)
    
    def test_invalid_window_size(self):
        """拒绝非法的窗口大小"""
        with self.assertRaises(ValueError):
            TransientStatistics(self.grid_data, window_size=0)
    
    def test_accumulate_samples(self):
        """累积解的样本"""
        stats = TransientStatistics(self.grid_data, window_size=10)
        
        for i in range(5):
            stats.accumulate(self.solution, time=i*0.01)
        
        self.assertEqual(stats.n_samples, 5)
        self.assertEqual(len(stats.samples), 5)
    
    def test_sliding_window(self):
        """滑动窗口生效"""
        stats = TransientStatistics(self.grid_data, window_size=3)
        
        for i in range(10):
            stats.accumulate(self.solution, time=i*0.01)
        
        # Should only keep last 3 samples
        self.assertEqual(len(stats.samples), 3)
        self.assertEqual(stats.n_samples, 10)
    
    def test_compute_statistics_no_samples(self):
        """没有样本时计算统计量报错"""
        stats = TransientStatistics(self.grid_data)
        
        with self.assertRaises(RuntimeError):
            stats.compute_statistics()
    
    def test_compute_statistics_with_samples(self):
        """用累积的样本计算统计量"""
        stats = TransientStatistics(self.grid_data, window_size=10)
        
        for i in range(5):
            stats.accumulate(self.solution, time=i*0.01)
        
        result = stats.compute_statistics()
        self.assertIsInstance(result, type(stats).compute_statistics.__annotations__.get('return', object))
        self.assertEqual(result.num_samples, 5)


class TestPressurePSD(unittest.TestCase):
    """压力 PSD 分析器的测试"""
    
    def setUp(self):
        """准备测试夹具"""
        self.monitor_points = [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0)]
        self.dt = 1e-4
        self.psd = PressurePSD(self.monitor_points, self.dt)
    
    def test_initialization(self):
        """PSD 分析器初始化"""
        self.assertEqual(len(self.psd.monitor_points), 2)
        self.assertEqual(self.psd.dt, 1e-4)
    
    def test_invalid_dt(self):
        """拒绝非法的时间步长"""
        with self.assertRaises(ValueError):
            PressurePSD(self.monitor_points, dt=0.0)
    
    def test_empty_monitor_points(self):
        """拒绝空的监测点"""
        with self.assertRaises(ValueError):
            PressurePSD([], dt=1e-4)
    
    def test_add_sample(self):
        """添加压力样本"""
        self.psd.add_sample(time=0.0, pressures=[101325.0, 101326.0])
        self.assertEqual(len(self.psd.times), 1)
        self.assertEqual(len(self.psd.pressure_history[0]), 1)
    
    def test_add_sample_length_mismatch(self):
        """拒绝长度不符的压力数组"""
        with self.assertRaises(ValueError):
            self.psd.add_sample(time=0.0, pressures=[101325.0])
    
    def test_compute_psd_insufficient_samples(self):
        """样本不足时计算 PSD 报错"""
        with self.assertRaises(RuntimeError):
            self.psd.compute_psd(0)
    
    def test_compute_psd_valid(self):
        """样本充足时计算 PSD"""
        # Add enough samples
        for i in range(20):
            pressure = 101325.0 + 10.0 * np.sin(2 * np.pi * 100 * i * self.dt)
            self.psd.add_sample(time=i*self.dt, pressures=[pressure, pressure])
        
        freqs, psd_vals = self.psd.compute_psd(0)
        
        self.assertGreater(len(freqs), 0)
        self.assertEqual(len(freqs), len(psd_vals))
        self.assertGreater(freqs[-1], 0)  # Max frequency > 0
    
    def test_find_dominant_frequency(self):
        """找主频"""
        # Add sinusoidal signal at 100 Hz
        for i in range(100):
            pressure = 101325.0 + 10.0 * np.sin(2 * np.pi * 100 * i * self.dt)
            self.psd.add_sample(time=i*self.dt, pressures=[pressure, pressure])
        
        freq, psd_val = self.psd.find_dominant_frequency(0, min_freq=50, max_freq=150)
        
        # 应找到接近 100 Hz 的频率
        self.assertGreater(freq, 90)
        self.assertLess(freq, 110)
    
    def test_invalid_point_index(self):
        """拒绝非法的点索引"""
        with self.assertRaises(IndexError):
            self.psd.compute_psd(10)


if __name__ == '__main__':
    unittest.main()
