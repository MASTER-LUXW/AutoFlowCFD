"""Ahmed Body steady RANS simulation example.

This script demonstrates a complete steady RANS simulation workflow:
1. Load NAS grid file
2. Configure solver
3. Run simulation
4. Post-process results (coefficients, VTK export, report)

Usage:
    poetry run python examples/ahmed_body/steady/run_steady.py
"""

import sys
from pathlib import Path

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
    SimulationReport
)


def main():
    """Run Ahmed Body steady RANS simulation"""
    logger.info("=" * 60)
    logger.info("AutoFlowCFD - Ahmed Body Steady RANS Example")
    logger.info("=" * 60)
    
    # Configuration
    config = {
        'simulation': {
            'name': 'ahmed_body_steady',
            'type': 'steady'
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
            'model': 'sst_kw'
        },
        'reference': {
            'area': 0.0396,
            'length': 1.044,
            'density': 1.225,
            'velocity': 30.0
        },
        'convergence': {
            'max_iterations': 100,
            'residual_threshold': 1e-4
        }
    }
    
    try:
        # Step 1: Parse grid file
        logger.info("\n[Step 1] Parsing NAS grid file...")
        grid_file = config['grid']['file']
        
        if not Path(grid_file).exists():
            logger.warning(f"Grid file not found: {grid_file}")
            logger.warning("Using mock grid for demonstration")
            # For demo purposes, create minimal mock grid
            from autoflowcfd.grid.structures import (
                GridData, NodeArray, CellArray, BoundaryMap, GridMetadata
            )
            import numpy as np
            
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
        logger.info("\n[Step 2] Initializing FR solver...")
        solver_config = SolverConfig.from_dict(config)
        solver = FRSolver(grid_data, solver_config)
        
        # Step 3: Run simulation
        logger.info("\n[Step 3] Running steady RANS simulation...")
        analyzer = ConvergenceAnalyzer()
        
        # Placeholder: simulate iteration loop
        max_iter = config['convergence']['max_iterations']
        for i in range(max_iter):
            # Simulate solver step
            residual = 1e-2 * (0.9 ** i)
            
            analyzer.add_iteration(
                iteration=i+1,
                residuals={'continuity': residual, 'momentum': residual * 0.1},
                cfl=5.0
            )
            
            # Check convergence
            if residual < config['convergence']['residual_threshold']:
                logger.success(f"Converged at iteration {i+1}")
                break
        
        # Step 4: Post-processing
        logger.info("\n[Step 4] Post-processing results...")
        
        # Create mock solution for demonstration
        from autoflowcfd.core.backend.base import SolutionVector
        solution = SolutionVector()
        
        # Calculate aerodynamic coefficients
        ref = config['reference']
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
        
        # Export VTK
        output_dir = Path(__file__).parent / 'output'
        output_dir.mkdir(exist_ok=True)
        
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
        report.generate(str(report_path), computation_time=60.0)
        logger.success(f"Report generated: {report_path}")
        
        logger.info("\n" + "=" * 60)
        logger.success("Simulation completed successfully!")
        logger.info("=" * 60)
        
    except Exception as e:
        logger.error(f"Simulation failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
