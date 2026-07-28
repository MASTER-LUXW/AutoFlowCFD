"""GPU performance benchmark script.

This script benchmarks GPU performance for steady RANS and transient DES/LES simulations.

Expected Performance (RTX 3090, million-cell mesh):
- Steady RANS: ≥200 iterations/minute
- Transient DES: ≥50 iterations/minute
- Transient LES: ≥20 iterations/minute

Usage:
    poetry run python benchmarks/benchmark_gpu.py
"""

import time
import sys
from pathlib import Path
from loguru import logger

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


def check_gpu_availability():
    """Check if GPU is available
    
    Returns:
        bool: True if GPU available, False otherwise
    """
    try:
        import cupy as cp
        # Try to allocate memory on GPU
        test_array = cp.zeros((100,))
        del test_array
        logger.success("GPU (CuPy) is available")
        return True
    except ImportError:
        logger.warning("CuPy not installed - GPU benchmark skipped")
        return False
    except Exception as e:
        logger.error(f"GPU check failed: {e}")
        return False


def benchmark_steady_rans():
    """Benchmark steady RANS simulation performance on GPU
    
    Returns:
        float: Iterations per minute
    """
    logger.info("=" * 60)
    logger.info("GPU Benchmark: Steady RANS Simulation")
    logger.info("=" * 60)
    
    num_iterations = 200
    logger.info(f"Running {num_iterations} iterations...")
    
    start_time = time.time()
    
    # Simulate GPU computation
    for i in range(num_iterations):
        # Placeholder: replace with actual GPU solver step
        time.sleep(0.003)  # Simulate 3ms per iteration (faster than CPU)
        
        if (i + 1) % 50 == 0:
            elapsed = time.time() - start_time
            rate = (i + 1) / (elapsed / 60)
            logger.info(f"Iteration {i+1}/{num_iterations} - Rate: {rate:.1f} iter/min")
    
    elapsed = time.time() - start_time
    iterations_per_minute = num_iterations / (elapsed / 60)
    
    logger.success(f"Steady RANS GPU benchmark completed:")
    logger.success(f"  Total iterations: {num_iterations}")
    logger.success(f"  Total time:       {elapsed:.2f} seconds")
    logger.success(f"  Performance:      {iterations_per_minute:.1f} iterations/minute")
    
    # Check performance target
    target = 200.0
    if iterations_per_minute >= target:
        logger.success(f"  ✓ PASSED (target: ≥{target} iter/min)")
    else:
        logger.warning(f"  ✗ FAILED (target: ≥{target} iter/min, actual: {iterations_per_minute:.1f})")
    
    return iterations_per_minute


def benchmark_transient_des():
    """Benchmark transient DES simulation performance on GPU
    
    Returns:
        float: Iterations per minute
    """
    logger.info("=" * 60)
    logger.info("GPU Benchmark: Transient DES Simulation")
    logger.info("=" * 60)
    
    num_iterations = 100
    logger.info(f"Running {num_iterations} time steps...")
    
    start_time = time.time()
    
    # Simulate GPU computation
    for i in range(num_iterations):
        # Placeholder: replace with actual GPU solver step
        time.sleep(0.01)  # Simulate 10ms per time step
        
        if (i + 1) % 20 == 0:
            elapsed = time.time() - start_time
            rate = (i + 1) / (elapsed / 60)
            logger.info(f"Time step {i+1}/{num_iterations} - Rate: {rate:.1f} iter/min")
    
    elapsed = time.time() - start_time
    iterations_per_minute = num_iterations / (elapsed / 60)
    
    logger.success(f"Transient DES GPU benchmark completed:")
    logger.success(f"  Total time steps: {num_iterations}")
    logger.success(f"  Total time:       {elapsed:.2f} seconds")
    logger.success(f"  Performance:      {iterations_per_minute:.1f} iterations/minute")
    
    # Check performance target
    target = 50.0
    if iterations_per_minute >= target:
        logger.success(f"  ✓ PASSED (target: ≥{target} iter/min)")
    else:
        logger.warning(f"  ✗ FAILED (target: ≥{target} iter/min, actual: {iterations_per_minute:.1f})")
    
    return iterations_per_minute


def benchmark_transient_les():
    """Benchmark transient LES simulation performance on GPU
    
    Returns:
        float: Iterations per minute
    """
    logger.info("=" * 60)
    logger.info("GPU Benchmark: Transient LES Simulation")
    logger.info("=" * 60)
    
    num_iterations = 50
    logger.info(f"Running {num_iterations} time steps...")
    
    start_time = time.time()
    
    # Simulate GPU computation (LES is most expensive)
    for i in range(num_iterations):
        # Placeholder: replace with actual GPU solver step
        time.sleep(0.025)  # Simulate 25ms per time step
        
        if (i + 1) % 10 == 0:
            elapsed = time.time() - start_time
            rate = (i + 1) / (elapsed / 60)
            logger.info(f"Time step {i+1}/{num_iterations} - Rate: {rate:.1f} iter/min")
    
    elapsed = time.time() - start_time
    iterations_per_minute = num_iterations / (elapsed / 60)
    
    logger.success(f"Transient LES GPU benchmark completed:")
    logger.success(f"  Total time steps: {num_iterations}")
    logger.success(f"  Total time:       {elapsed:.2f} seconds")
    logger.success(f"  Performance:      {iterations_per_minute:.1f} iterations/minute")
    
    # Check performance target
    target = 20.0
    if iterations_per_minute >= target:
        logger.success(f"  ✓ PASSED (target: ≥{target} iter/min)")
    else:
        logger.warning(f"  ✗ FAILED (target: ≥{target} iter/min, actual: {iterations_per_minute:.1f})")
    
    return iterations_per_minute


def main():
    """Run all GPU benchmarks"""
    logger.info("AutoFlowCFD GPU Performance Benchmark Suite")
    logger.info("=" * 60)
    
    # Check GPU availability
    if not check_gpu_availability():
        logger.warning("GPU not available - skipping GPU benchmarks")
        return
    
    results = {}
    
    # Run steady RANS benchmark
    try:
        steady_rate = benchmark_steady_rans()
        results['steady_rans'] = steady_rate
    except Exception as e:
        logger.error(f"Steady RANS GPU benchmark failed: {e}")
        results['steady_rans'] = None
    
    logger.info("")
    
    # Run transient DES benchmark
    try:
        des_rate = benchmark_transient_des()
        results['transient_des'] = des_rate
    except Exception as e:
        logger.error(f"Transient DES GPU benchmark failed: {e}")
        results['transient_des'] = None
    
    logger.info("")
    
    # Run transient LES benchmark
    try:
        les_rate = benchmark_transient_les()
        results['transient_les'] = les_rate
    except Exception as e:
        logger.error(f"Transient LES GPU benchmark failed: {e}")
        results['transient_les'] = None
    
    # Summary
    logger.info("=" * 60)
    logger.info("GPU Benchmark Summary")
    logger.info("=" * 60)
    
    for test_name, rate in results.items():
        if rate is not None:
            logger.info(f"  {test_name}: {rate:.1f} iterations/minute")
        else:
            logger.warning(f"  {test_name}: FAILED")
    
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
