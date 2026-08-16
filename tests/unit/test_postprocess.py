"""Unit tests for postprocessing module."""

import unittest
import numpy as np
from pathlib import Path
import tempfile
import json
import csv

from autoflowcfd.grid.structures import (
    GridData, NodeArray, CellArray, BoundaryMap, GridMetadata,
    TetrahedralCells, VolumeMeshData,
)
from autoflowcfd.core.backend.base import SolutionVector
from autoflowcfd.postprocess import (
    CoefficientCalculator,
    AerodynamicCoefficients,
    AerodynamicForces,
    VTKExporter,
    ConvergenceAnalyzer,
    SimulationReport,
    TransientStatistics,
    PressurePSD,
)


class TestAerodynamicCoefficients(unittest.TestCase):
    """Test aerodynamic coefficients data class"""
    
    def test_default_values(self):
        """Test default coefficient values are zero"""
        coeffs = AerodynamicCoefficients()
        self.assertEqual(coeffs.Cd, 0.0)
        self.assertEqual(coeffs.Cl, 0.0)
        self.assertEqual(coeffs.Cm, 0.0)
    
    def test_to_dict(self):
        """Test conversion to dictionary"""
        coeffs = AerodynamicCoefficients(Cd=0.3, Cl=0.1, Cm=-0.05)
        d = coeffs.to_dict()
        self.assertAlmostEqual(d['Cd'], 0.3)
        self.assertAlmostEqual(d['Cl'], 0.1)
        self.assertAlmostEqual(d['Cm'], -0.05)
    
    def test_string_representation(self):
        """Test string representation contains all coefficients"""
        coeffs = AerodynamicCoefficients(Cd=0.28)
        s = str(coeffs)
        self.assertIn("Cd", s)
        self.assertIn("0.28", s)


class TestAerodynamicForces(unittest.TestCase):
    """Test aerodynamic forces data class"""
    
    def test_default_values(self):
        """Test default force values are zero"""
        forces = AerodynamicForces()
        self.assertEqual(forces.drag_force, 0.0)
        self.assertEqual(forces.lift_force, 0.0)
    
    def test_to_dict(self):
        """Test conversion to dictionary"""
        forces = AerodynamicForces(drag_force=150.0, lift_force=-20.0)
        d = forces.to_dict()
        self.assertAlmostEqual(d['drag_force'], 150.0)
        self.assertAlmostEqual(d['lift_force'], -20.0)


class TestCoefficientCalculator(unittest.TestCase):
    """Test coefficient calculator"""
    
    def setUp(self):
        """Set up test fixtures.

        calculate_forces() delegates to core.aero_coeffs.AeroCoefficientCalculator
        for a real pressure/skin-friction surface integration (see
        coefficients.py), which needs actual volume-mesh face connectivity -
        a bare surface GridData has no such thing. Two tets sharing a face
        (rather than a single closed cell) so a non-uniform per-cell
        pressure gives a genuinely non-trivial net force, matching the
        fixture used to verify the real-integration fix itself.
        """
        nodes = NodeArray(
            x=np.array([0.0, 1.0, 0.0, 0.0, 1.0]),
            y=np.array([0.0, 0.0, 1.0, 0.0, 1.0]),
            z=np.array([0.0, 0.0, 0.0, 1.0, 1.0]),
        )
        cells = TetrahedralCells(
            connectivity=np.array([[0, 1, 2, 3], [1, 2, 3, 4]], dtype=np.int32),
            volumes=np.array([1.0 / 6.0, 1.0 / 6.0]),
        )
        boundaries = BoundaryMap(
            groups={"body": np.array([0, 1], dtype=np.int32)},
            bc_types={"body": "WALL"}
        )
        metadata = GridMetadata(
            node_count=5,
            cell_count=2,
            boundary_groups=["body"],
            file_format="volume"
        )
        self.grid_data = VolumeMeshData(
            nodes=nodes,
            cells=cells,
            boundaries=boundaries,
            metadata=metadata
        )

        # Solution with a non-uniform per-cell pressure (zero velocity) so
        # the surface integration has a genuine, non-zero force to find.
        gamma = 1.4
        data = np.zeros((2, 7))
        data[:, 0] = 1.225  # rho
        data[0, 4] = (101325.0 + 2000.0) / (gamma - 1.0)
        data[1, 4] = (101325.0 - 2000.0) / (gamma - 1.0)
        self.solution = SolutionVector(data=data, n_cells=2, n_variables=7)
    
    def test_initialization(self):
        """Test calculator initialization"""
        calc = CoefficientCalculator(
            self.grid_data,
            self.solution,
            reference_area=2.2,
            velocity=30.0
        )
        self.assertEqual(calc.reference_area, 2.2)
        self.assertEqual(calc.velocity, 30.0)
        self.assertAlmostEqual(calc.dynamic_pressure, 0.5 * 1.225 * 30.0**2)
    
    def test_invalid_reference_area(self):
        """Test rejection of invalid reference area"""
        with self.assertRaises(ValueError):
            CoefficientCalculator(
                self.grid_data,
                self.solution,
                reference_area=-1.0
            )
    
    def test_invalid_velocity(self):
        """Test rejection of invalid velocity"""
        with self.assertRaises(ValueError):
            CoefficientCalculator(
                self.grid_data,
                self.solution,
                velocity=0.0
            )
    
    def test_calculate_returns_coefficients(self):
        """Test calculate method returns AerodynamicCoefficients"""
        calc = CoefficientCalculator(self.grid_data, self.solution)
        coeffs = calc.calculate()
        self.assertIsInstance(coeffs, AerodynamicCoefficients)
    
    def test_calculate_forces_returns_forces(self):
        """Test calculate_forces method returns AerodynamicForces"""
        calc = CoefficientCalculator(self.grid_data, self.solution)
        forces = calc.calculate_forces()
        self.assertIsInstance(forces, AerodynamicForces)
    
    def test_calculate_by_boundary_invalid_name(self):
        """Test rejection of invalid boundary name"""
        calc = CoefficientCalculator(self.grid_data, self.solution)
        with self.assertRaises(KeyError):
            calc.calculate_by_boundary("INVALID")


class TestVTKExporter(unittest.TestCase):
    """Test VTK exporter"""
    
    def setUp(self):
        """Set up test fixtures"""
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
        """Test export to legacy VTK format"""
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
        """Test export with specific fields"""
        exporter = VTKExporter(self.grid_data, self.solution)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "test.vtk"
            exporter.export(str(output_path), fields=['velocity'])
            
            with open(output_path, 'r') as f:
                content = f.read()
                self.assertIn("VECTORS Velocity", content)
    
    def test_export_invalid_field(self):
        """Test rejection of invalid field name"""
        exporter = VTKExporter(self.grid_data, self.solution)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "test.vtk"
            with self.assertRaises(ValueError):
                exporter.export(str(output_path), fields=['invalid_field'])
    
    def test_export_invalid_format(self):
        """Test rejection of invalid format"""
        exporter = VTKExporter(self.grid_data, self.solution)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "test.vtu"
            with self.assertRaises(ValueError):
                exporter.export(str(output_path), format='invalid')


class TestConvergenceAnalyzer(unittest.TestCase):
    """Test convergence analyzer"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.analyzer = ConvergenceAnalyzer()
    
    def test_add_iteration(self):
        """Test adding iteration data"""
        self.analyzer.add_iteration(
            iteration=1,
            residuals={'continuity': 1e-2, 'momentum': 1e-3},
            cfl=5.0
        )
        self.assertEqual(len(self.analyzer.history), 1)
        self.assertEqual(self.analyzer.history[0].iteration, 1)
    
    def test_export_csv(self):
        """Test exporting convergence history to CSV"""
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
        """Test getting simulation summary"""
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
    """Test simulation report generator"""
    
    def setUp(self):
        """Set up test fixtures"""
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
        """Test generating JSON report"""
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
    """Test transient statistics calculator"""
    
    def setUp(self):
        """Set up test fixtures"""
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
        """Test statistics calculator initialization"""
        stats = TransientStatistics(self.grid_data, window_size=50)
        self.assertEqual(stats.window_size, 50)
        self.assertEqual(stats.n_samples, 0)
    
    def test_invalid_window_size(self):
        """Test rejection of invalid window size"""
        with self.assertRaises(ValueError):
            TransientStatistics(self.grid_data, window_size=0)
    
    def test_accumulate_samples(self):
        """Test accumulating solution samples"""
        stats = TransientStatistics(self.grid_data, window_size=10)
        
        for i in range(5):
            stats.accumulate(self.solution, time=i*0.01)
        
        self.assertEqual(stats.n_samples, 5)
        self.assertEqual(len(stats.samples), 5)
    
    def test_sliding_window(self):
        """Test sliding window enforcement"""
        stats = TransientStatistics(self.grid_data, window_size=3)
        
        for i in range(10):
            stats.accumulate(self.solution, time=i*0.01)
        
        # Should only keep last 3 samples
        self.assertEqual(len(stats.samples), 3)
        self.assertEqual(stats.n_samples, 10)
    
    def test_compute_statistics_no_samples(self):
        """Test error when computing statistics without samples"""
        stats = TransientStatistics(self.grid_data)
        
        with self.assertRaises(RuntimeError):
            stats.compute_statistics()
    
    def test_compute_statistics_with_samples(self):
        """Test computing statistics with accumulated samples"""
        stats = TransientStatistics(self.grid_data, window_size=10)
        
        for i in range(5):
            stats.accumulate(self.solution, time=i*0.01)
        
        result = stats.compute_statistics()
        self.assertIsInstance(result, type(stats).compute_statistics.__annotations__.get('return', object))
        self.assertEqual(result.num_samples, 5)


class TestPressurePSD(unittest.TestCase):
    """Test pressure PSD analyzer"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.monitor_points = [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0)]
        self.dt = 1e-4
        self.psd = PressurePSD(self.monitor_points, self.dt)
    
    def test_initialization(self):
        """Test PSD analyzer initialization"""
        self.assertEqual(len(self.psd.monitor_points), 2)
        self.assertEqual(self.psd.dt, 1e-4)
    
    def test_invalid_dt(self):
        """Test rejection of invalid time step"""
        with self.assertRaises(ValueError):
            PressurePSD(self.monitor_points, dt=0.0)
    
    def test_empty_monitor_points(self):
        """Test rejection of empty monitor points"""
        with self.assertRaises(ValueError):
            PressurePSD([], dt=1e-4)
    
    def test_add_sample(self):
        """Test adding pressure samples"""
        self.psd.add_sample(time=0.0, pressures=[101325.0, 101326.0])
        self.assertEqual(len(self.psd.times), 1)
        self.assertEqual(len(self.psd.pressure_history[0]), 1)
    
    def test_add_sample_length_mismatch(self):
        """Test rejection of mismatched pressure array length"""
        with self.assertRaises(ValueError):
            self.psd.add_sample(time=0.0, pressures=[101325.0])
    
    def test_compute_psd_insufficient_samples(self):
        """Test error when computing PSD with insufficient samples"""
        with self.assertRaises(RuntimeError):
            self.psd.compute_psd(0)
    
    def test_compute_psd_valid(self):
        """Test computing PSD with sufficient samples"""
        # Add enough samples
        for i in range(20):
            pressure = 101325.0 + 10.0 * np.sin(2 * np.pi * 100 * i * self.dt)
            self.psd.add_sample(time=i*self.dt, pressures=[pressure, pressure])
        
        freqs, psd_vals = self.psd.compute_psd(0)
        
        self.assertGreater(len(freqs), 0)
        self.assertEqual(len(freqs), len(psd_vals))
        self.assertGreater(freqs[-1], 0)  # Max frequency > 0
    
    def test_find_dominant_frequency(self):
        """Test finding dominant frequency"""
        # Add sinusoidal signal at 100 Hz
        for i in range(100):
            pressure = 101325.0 + 10.0 * np.sin(2 * np.pi * 100 * i * self.dt)
            self.psd.add_sample(time=i*self.dt, pressures=[pressure, pressure])
        
        freq, psd_val = self.psd.find_dominant_frequency(0, min_freq=50, max_freq=150)
        
        # Should find frequency close to 100 Hz
        self.assertGreater(freq, 90)
        self.assertLess(freq, 110)
    
    def test_invalid_point_index(self):
        """Test rejection of invalid point index"""
        with self.assertRaises(IndexError):
            self.psd.compute_psd(10)


if __name__ == '__main__':
    unittest.main()
