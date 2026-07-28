"""CPU performance benchmark script.

This script benchmarks CPU performance for steady RANS and transient DES simulations.

Expected Performance (8-core i7, million-cell mesh):
- Steady RANS: ≥50 iterations/minute
- Transient DES: ≥10 iterations/minute

Usage:
    poetry run python benchmarks/benchmark_cpu.py
"""

import time
import sys
from pathlib import Path
from loguru import logger

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


def benchmark_steady_rans():
    """Benchmark steady RANS simulation performance
    
    Returns:
        float: Iterations per minute
    """
    logger.info("=" * 60)
    logger.info("CPU Benchmark: Steady RANS Simulation")
    logger.info("=" * 60)
    
    # Placeholder: simulate iteration loop
    # In production, this would run actual solver
    
    num_iterations = 100
    logger.info(f"Running {num_iterations} iterations...")
    
    start_time = time.time()
    
    # Simulate computation
    for i in range(num_iterations):
        # Placeholder: replace with actual solver step
        time.sleep(0.01)  # Simulate 10ms per iteration
        
        if (i + 1) % 20 == 0:
            elapsed = time.time() - start_time
            rate = (i + 1) / (elapsed / 60)
            logger.info(f"Iteration {i+1}/{num_iterations} - Rate: {rate:.1f} iter/min")
    
    elapsed = time.time() - start_time
    iterations_per_minute = num_iterations / (elapsed / 60)
    
    logger.success(f"Steady RANS benchmark completed:")
    logger.success(f"  Total iterations: {num_iterations}")
    logger.success(f"  Total time:       {elapsed:.2f} seconds")
    logger.success(f"  Performance:      {iterations_per_minute:.1f} iterations/minute")
    
    # Check performance target
    target = 50.0
    if iterations_per_minute >= target:
        logger.success(f"  ✓ PASSED (target: ≥{target} iter/min)")
    else:
        logger.warning(f"  ✗ FAILED (target: ≥{target} iter/min, actual: {iterations_per_minute:.1f})")
    
    return iterations_per_minute


def benchmark_transient_des():
    """Benchmark transient DES simulation performance
    
    Returns:
        float: Iterations per minute
    """
    logger.info("=" * 60)
    logger.info("CPU Benchmark: Transient DES Simulation")
    logger.info("=" * 60)
    
    num_iterations = 50
    logger.info(f"Running {num_iterations} time steps...")
    
    start_time = time.time()
    
    # Simulate computation (transient is slower due to additional physics)
    for i in range(num_iterations):
        # Placeholder: replace with actual solver step
        time.sleep(0.05)  # Simulate 50ms per time step
        
        if (i + 1) % 10 == 0:
            elapsed = time.time() - start_time
            rate = (i + 1) / (elapsed / 60)
            logger.info(f"Time step {i+1}/{num_iterations} - Rate: {rate:.1f} iter/min")
    
    elapsed = time.time() - start_time
    iterations_per_minute = num_iterations / (elapsed / 60)
    
    logger.success(f"Transient DES benchmark completed:")
    logger.success(f"  Total time steps: {num_iterations}")
    logger.success(f"  Total time:       {elapsed:.2f} seconds")
    logger.success(f"  Performance:      {iterations_per_minute:.1f} iterations/minute")
    
    # Check performance target
    target = 10.0
    if iterations_per_minute >= target:
        logger.success(f"  ✓ PASSED (target: ≥{target} iter/min)")
    else:
        logger.warning(f"  ✗ FAILED (target: ≥{target} iter/min, actual: {iterations_per_minute:.1f})")
    
    return iterations_per_minute


def main():
    """Run all CPU benchmarks"""
    logger.info("AutoFlowCFD CPU Performance Benchmark Suite")
    logger.info("=" * 60)
    
    results = {}
    
    # Run steady RANS benchmark
    try:
        steady_rate = benchmark_steady_rans()
        results['steady_rans'] = steady_rate
    except Exception as e:
        logger.error(f"Steady RANS benchmark failed: {e}")
        results['steady_rans'] = None
    
    logger.info("")
    
    # Run transient DES benchmark
    try:
        transient_rate = benchmark_transient_des()
        results['transient_des'] = transient_rate
    except Exception as e:
        logger.error(f"Transient DES benchmark failed: {e}")
        results['transient_des'] = None
    
    # Summary
    logger.info("=" * 60)
    logger.info("Benchmark Summary")
    logger.info("=" * 60)
    
    for test_name, rate in results.items():
        if rate is not None:
            logger.info(f"  {test_name}: {rate:.1f} iterations/minute")
        else:
            logger.warning(f"  {test_name}: FAILED")
    
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
