"""Diagnostic script for multi-core utilization analysis.

This script analyzes the current multi-threading configuration and provides
recommendations for optimizing CPU parallel performance in AutoFlowCFD.
"""

import os
import sys
import multiprocessing
from loguru import logger


def diagnose_threading():
    """Diagnose current threading configuration."""
    logger.info("=" * 70)
    logger.info("AutoFlowCFD Multi-Core Utilization Diagnostic")
    logger.info("=" * 70)
    
    # System information
    cpu_count = multiprocessing.cpu_count()
    logger.info(f"\n📊 System Information:")
    logger.info(f"  CPU cores: {cpu_count}")
    logger.info(f"  Platform: {sys.platform}")
    
    # Environment variables
    logger.info(f"\n🔧 Threading Environment Variables:")
    thread_vars = [
        'MKL_NUM_THREADS',
        'OPENBLAS_NUM_THREADS', 
        'NUMEXPR_NUM_THREADS',
        'VECLIB_MAXIMUM_THREADS',
        'OMP_NUM_THREADS',
        'NUMBA_NUM_THREADS',
        'NUMBA_THREADING_LAYER'
    ]
    
    for var in thread_vars:
        value = os.environ.get(var, 'not set')
        status = "✅" if value != 'not set' else "⚠️"
        logger.info(f"  {status} {var}: {value}")
    
    # Check NumPy BLAS configuration
    try:
        import numpy as np
        logger.info(f"\n📦 NumPy Configuration:")
        logger.info(f"  Version: {np.__version__}")
        
        # Try to detect BLAS library
        try:
            import threadpoolctl
            controllers = threadpoolctl.threadpool_info()
            logger.info(f"  BLAS/LAPACK libraries detected:")
            for ctrl in controllers:
                logger.info(f"    - {ctrl['user_api']}: {ctrl['internal_api']} "
                          f"({ctrl['num_threads']} threads)")
        except ImportError:
            logger.warning("  ⚠️ threadpoolctl not installed. Install with:")
            logger.warning("     pip install threadpoolctl")
            
    except ImportError:
        logger.error("❌ NumPy not available")
        return False
    
    # Check Numba configuration
    try:
        import numba
        from numba import config
        
        logger.info(f"\n🚀 Numba Configuration:")
        logger.info(f"  Version: {numba.__version__}")
        logger.info(f"  Threading layer: {config.THREADING_LAYER}")
        logger.info(f"  NUMBA_NUM_THREADS: {numba.config.NUMBA_NUM_THREADS}")
        
    except ImportError:
        logger.warning("⚠️ Numba not available (CPU backend will be limited)")
    
    # Performance test
    logger.info(f"\n⚡ Performance Test:")
    try:
        from time import time
        import numpy as np
        from numba import njit, prange
        
        # Test NumPy vectorized operation
        n = 10_000_000
        a = np.random.rand(n)
        b = np.random.rand(n)
        
        start = time()
        c = a * b + np.sin(a)  # Vectorized operations
        numpy_time = time() - start
        logger.info(f"  NumPy vectorized ({n:,} elements): {numpy_time:.3f}s")
        
        # Test Numba parallel loop
        @njit(parallel=True)
        def parallel_test(n):
            result = 0.0
            for i in prange(n):
                result += np.sqrt(i)
            return result
        
        start = time()
        parallel_test(10_000_000)
        numba_time = time() - start
        logger.info(f"  Numba parallel loop: {numba_time:.3f}s")
        
        logger.info(f"\n💡 Recommendations:")
        if numpy_time > 0.5:
            logger.warning(f"  ⚠️ NumPy operations seem slow. Consider:")
            logger.warning(f"     - Setting MKL_NUM_THREADS={cpu_count}")
            logger.warning(f"     - Installing optimized BLAS (Intel MKL)")
        else:
            logger.success(f"  ✅ NumPy performance looks good")
            
        if numba_time > 1.0:
            logger.warning(f"  ⚠️ Numba parallel efficiency may be low")
            logger.warning(f"     - Check NUMBA_THREADING_LAYER setting")
            logger.warning(f"     - Consider installing TBB: pip install tbb")
        else:
            logger.success(f"  ✅ Numba parallel performance is acceptable")
            
    except Exception as e:
        logger.error(f"❌ Performance test failed: {e}")
    
    logger.info("\n" + "=" * 70)
    logger.info("Diagnostic complete")
    logger.info("=" * 70)
    
    return True


if __name__ == "__main__":
    success = diagnose_threading()
    sys.exit(0 if success else 1)
