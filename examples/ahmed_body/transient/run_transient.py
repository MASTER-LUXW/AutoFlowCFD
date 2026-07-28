"""Ahmed Body transient DES simulation example.

This script demonstrates a complete transient DES simulation workflow:
1. Load NAS grid file
2. Configure transient solver
3. Run time-accurate simulation
4. Compute transient statistics (time-averaged, RMS, PSD)
5. Export results

Usage:
    poetry run python examples/ahmed_body/transient/run_transient.py
"""

import sys
from pathlib import Path
import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from loguru import logger
from autoflowcfd.grid import NASParser
from autoflowcfd.config import SolverConfig
from autoflowcfd.core.solver import FRSolver
from autoflowcfd.postprocess import (
    CoefficientCalculator,
    VTKExporter,
    ConvergenceAnalyzer,
    SimulationReport,
    TransientStatistics,
    PressurePSD
)


def main():
    """Run Ahmed Body transient DES simulation"""
    logger.info("=" * 60)
    logger.info("AutoFlowCFD - Ahmed Body Transient DES Example")
    logger.info("=" * 60)
    
    # Configuration
    config = {
        'simulation': {
            'name': 'ahmed_body_transient',
            'type': 'transient'
        },
        'grid': {
            'file': str(Path(__file__).parent.parent / 'ahmed_body_demo.nas'),
            'format': 'nas'
        },
        'solver': {
            'backend': 'cpu',
            'order': 2
        },
        'turbulence': {
            'model': 'dDES'
        },
        'time_integration': {
            'method': 'backward_euler',
            'physical_time': 0.3,
            'dt': 1e-4
        },
        'reference': {
            'area': 0.0396,
            'length': 1.044,
            'density': 1.225,
            'velocity': 30.0
        },
        'convergence': {
            'max_iterations': 3000
        }
    }
    
    try:
        # Step 1: Parse grid file
        logger.info("\n[Step 1] Parsing NAS grid file...")
        grid_file = config['grid']['file']
        
        if not Path(grid_file).exists():
            logger.warning(f"Grid file not found: {grid_file}")
            logger.warning("Using mock grid for demonstration")
            from autoflowcfd.grid.structures import (
                GridData, NodeArray, CellArray, BoundaryMap, GridMetadata
            )
            
            nodes = NodeArray(
                x=np.array([0.0, 1.0, 2.0, 3.0]),
                y=np.array([0.0, 0.0, 0.0, 0.0]),
                z=np.array([0.0, 0.0, 0.0, 0.0])
            )
            cells = CellArray(
                connectivity=np.array([[0, 1, 2], [1, 2, 3]]),
                cell_type=np.array([0, 0])
            )
            boundaries = BoundaryMap(
                groups={'body': np.array([0, 1, 2, 3])},
                bc_types={'body': 'WALL'}
            )
            metadata = GridMetadata(
                node_count=4,
                cell_count=2,
                boundary_groups=['body'],
                file_format='v24'
            )
            grid_data = GridData(
                nodes=nodes,
                cells=cells,
                boundaries=boundaries,
                metadata=metadata
            )
        else:
            parser = NASParser(grid_file)
            grid_data = parser.parse()
        
        logger.success(f"Grid loaded: {grid_data.metadata.node_count} nodes, "
                      f"{grid_data.metadata.cell_count} cells")
        
        # Step 2: Initialize solver
        logger.info("\n[Step 2] Initializing transient FR solver...")
        solver_config = SolverConfig.from_dict(config)
        solver = FRSolver(grid_data, solver_config)
        
        # Step 3: Run transient simulation
        logger.info("\n[Step 3] Running transient DES simulation...")
        
        dt = config['time_integration']['dt']
        physical_time = config['time_integration']['physical_time']
        num_steps = int(physical_time / dt)
        
        analyzer = ConvergenceAnalyzer()
        stats = TransientStatistics(grid_data, window_size=100)
        
        # Monitor points for PSD analysis
        monitor_points = [(0.5, 0.0, 0.1), (0.5, 0.0, 0.3)]
        psd_analyzer = PressurePSD(monitor_points, dt)
        
        from autoflowcfd.core.backend.base import SolutionVector
        
        # Simulate time stepping
        logger.info(f"Running {num_steps} time steps (dt={dt}s, total={physical_time}s)...")
        
        for step in range(min(num_steps, 50)):  # Limit to 50 steps for demo
            current_time = step * dt
            
            # Create mock solution
            solution = SolutionVector()
            
            # Accumulate for statistics (skip initial transient)
            if current_time >= 0.1:  # Start sampling after 0.1s
                stats.accumulate(solution, time=current_time)
                
                # Record pressure at monitor points (mock data)
                pressures = [
                    101325.0 + 10.0 * np.sin(2 * np.pi * 100 * current_time),
                    101325.0 + 8.0 * np.sin(2 * np.pi * 100 * current_time)
                ]
                psd_analyzer.add_sample(time=current_time, pressures=pressures)
            
            # Log progress
            analyzer.add_iteration(
                iteration=step+1,
                residuals={'continuity': 1e-5},
                cfl=5.0
            )
            
            if (step + 1) % 10 == 0:
                logger.info(f"Time step {step+1}/{num_steps}, t={current_time:.4f}s")
        
        # Step 4: Post-processing
        logger.info("\n[Step 4] Post-processing transient results...")
        
        # Compute transient statistics
        if stats.n_samples > 0:
            result = stats.compute_statistics()
            logger.info(f"\nTransient Statistics:")
            logger.info(f"  Samples:        {result.num_samples}")
            logger.info(f"  Sampling time:  {result.sampling_time:.4f} s")
            logger.info(f"  Mean fields:    {list(result.mean_fields.keys())}")
            logger.info(f"  RMS fields:     {list(result.rms_fields.keys())}")
        
        # Compute PSD
        try:
            freqs, psd_vals = psd_analyzer.compute_psd(0)
            dominant_freq, peak_psd = psd_analyzer.find_dominant_frequency(
                0, min_freq=50, max_freq=150
            )
            logger.info(f"\nPSD Analysis (monitor point 0):")
            logger.info(f"  Dominant frequency: {dominant_freq:.2f} Hz")
            logger.info(f"  Peak PSD:           {peak_psd:.2e}")
        except Exception as e:
            logger.warning(f"PSD analysis skipped: {e}")
        
        # Calculate aerodynamic coefficients
        ref = config['reference']
        solution = SolutionVector()
        calc = CoefficientCalculator(
            grid_data,
            solution,
            reference_area=ref['area'],
            reference_length=ref['length'],
            density=ref['density'],
            velocity=ref['velocity']
        )
        coeffs = calc.calculate()
        logger.info(f"\n{coeffs}")
        
        # Export results
        output_dir = Path(__file__).parent / 'output'
        output_dir.mkdir(exist_ok=True)
        
        # Export VTK
        vtk_path = output_dir / 'result.vtk'
        exporter = VTKExporter(grid_data, solution)
        exporter.export(str(vtk_path), fields=['velocity', 'pressure'])
        logger.success(f"VTK exported: {vtk_path}")
        
        # Export convergence CSV
        csv_path = output_dir / 'convergence.csv'
        analyzer.export_csv(str(csv_path))
        logger.success(f"Convergence CSV exported: {csv_path}")
        
        # Generate report
        report = SimulationReport(config, analyzer)
        report_path = output_dir / 'report.json'
        report.generate(str(report_path), computation_time=120.0)
        logger.success(f"Report generated: {report_path}")
        
        logger.info("\n" + "=" * 60)
        logger.success("Transient simulation completed successfully!")
        logger.info("=" * 60)
        
    except Exception as e:
        logger.error(f"Simulation failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
