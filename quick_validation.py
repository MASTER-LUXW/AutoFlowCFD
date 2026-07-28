"""Quick validation test for optimized MUSCL reconstruction."""

import sys
import time
from pathlib import Path
from loguru import logger

logger.remove()
logger.add(sys.stderr, level="INFO", format="<level>{message}</level>")

def quick_test():
    """Quick end-to-end test with minimal iterations."""
    logger.info("="*60)
    logger.info("AutoFlowCFD Quick Validation Test")
    logger.info("="*60)
    
    from autoflowcfd.grid import NASParser
    from autoflowcfd.core import FRSolver
    from autoflowcfd.config import SteadyConfig, BackendType, TurbulenceModel
    
    grid_file = Path("C:/Users/luxw_/Desktop/AutoFlowCFD/ahmed_body_demo.nas")
    
    if not grid_file.exists():
        logger.error(f"Grid file not found: {grid_file}")
        return False
    
    logger.info("Step 1: Parsing grid (minimal volume mesh)...")
    start = time.time()
    parser = NASParser(str(grid_file))
    grid_data = parser.parse(
        generate_volume_mesh=True,
        volume_mesh_params={
            'growth_rate': 1.2,
            'max_layers': 3,  # Very small
            'min_cell_size': 0.01,
            'target_cells': 5000
        }
    )
    logger.info(f"✓ Grid parsed in {time.time()-start:.2f}s: {grid_data.node_count} nodes, {grid_data.cell_count} cells")
    
    logger.info("\nStep 2: Initializing solver...")
    config = SteadyConfig(
        backend=BackendType.CPU,
        order=1,
        turbulence=TurbulenceModel.SST_KW,
        max_iter=3,  # Minimal iterations
        cfl_init=0.1,
        cfl_max=1.0,
        convergence_tol=1e-2,
        output_dir="quick_test_results",
        checkpoint_interval=10,
        n_threads=2,
        gpu_device=0,
    )
    
    start = time.time()
    solver = FRSolver(grid_data, config)
    logger.info(f"✓ Solver initialized in {time.time()-start:.2f}s")
    
    logger.info("\nStep 3: Running simulation (3 iterations)...")
    start = time.time()
    result = solver.solve()
    elapsed = time.time() - start
    
    logger.info(f"\n{'='*60}")
    logger.info("Simulation Results:")
    logger.info(f"  Iterations: {result.iterations}")
    logger.info(f"  Time: {elapsed:.2f}s")
    logger.info(f"  Final residual: {result.final_residual:.6e}")
    if result.cd_history:
        logger.info(f"  Final Cd: {result.cd_history[-1]:.4f}")
        logger.info(f"  Final Cl: {result.cl_history[-1]:.4f}")
    logger.info(f"{'='*60}")
    
    # Cleanup
    import shutil
    if Path("quick_test_results").exists():
        shutil.rmtree("quick_test_results")
        logger.info("✓ Cleaned up test results")
    
    return True

if __name__ == "__main__":
    try:
        quick_test()
        logger.success("\n✅ Quick validation PASSED!")
    except Exception as e:
        logger.error(f"\n❌ Quick validation FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
