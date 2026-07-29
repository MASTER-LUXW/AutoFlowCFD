"""Verification script for gradient computation equivalence.

This script verifies that batch gradient computation produces numerically
equivalent results to separate computation, ensuring the optimization
does not affect solution accuracy.
"""

import numpy as np
from loguru import logger


def verify_gradient_equivalence(geom, n_samples: int = 1000):
    """Verify that batch gradient computation equals separate computation.
    
    Args:
        geom: FaceGeometry object with mesh topology
        n_samples: Number of random test cases to run
        
    Returns:
        bool: True if all tests pass within tolerance
    """
    from autoflowcfd.core.fvm_gradients import green_gauss_gradient
    
    n_cells = geom.n_cells
    logger.info(f"Starting gradient equivalence verification ({n_samples} tests)")
    logger.info(f"Grid size: {n_cells} cells")
    
    max_rel_errors_k = []
    max_rel_errors_w = []
    
    for test_idx in range(n_samples):
        # Generate random test data with realistic ranges
        np.random.seed(test_idx)
        k = np.random.rand(n_cells) * 100 + 0.1  # k > 0
        omega = np.random.rand(n_cells) * 1000 + 1.0  # omega > 0
        
        # Method A: Separate computation
        gk_sep = green_gauss_gradient(k[:, None], geom)[:, 0, :]
        gw_sep = green_gauss_gradient(omega[:, None], geom)[:, 0, :]
        
        # Method B: Batch computation
        kw = np.column_stack([k, omega])
        gturb = green_gauss_gradient(kw, geom)
        gk_batch = gturb[:, 0, :]
        gw_batch = gturb[:, 1, :]
        
        # Compute relative errors
        rel_err_k = np.abs(gk_sep - gk_batch) / (np.abs(gk_sep) + 1e-30)
        rel_err_w = np.abs(gw_sep - gw_batch) / (np.abs(gw_sep) + 1e-30)
        
        max_rel_errors_k.append(np.max(rel_err_k))
        max_rel_errors_w.append(np.max(rel_err_w))
    
    # Statistical analysis
    max_err_k = np.max(max_rel_errors_k)
    max_err_w = np.max(max_rel_errors_w)
    mean_err_k = np.mean(max_rel_errors_k)
    mean_err_w = np.mean(max_rel_errors_w)
    
    logger.info("=" * 70)
    logger.info("Gradient Computation Equivalence Verification Results")
    logger.info("=" * 70)
    logger.info(f"\nk (turbulent kinetic energy):")
    logger.info(f"  Max relative error:    {max_err_k:.3e}")
    logger.info(f"  Mean relative error:   {mean_err_k:.3e}")
    logger.info(f"  Std deviation:         {np.std(max_rel_errors_k):.3e}")
    
    logger.info(f"\nomega (specific dissipation rate):")
    logger.info(f"  Max relative error:    {max_err_w:.3e}")
    logger.info(f"  Mean relative error:   {mean_err_w:.3e}")
    logger.info(f"  Std deviation:         {np.std(max_rel_errors_w):.3e}")
    
    # Tolerance check (machine precision for double)
    tolerance = 1e-12
    passed = (max_err_k < tolerance and max_err_w < tolerance)
    
    logger.info(f"\n{'='*70}")
    if passed:
        logger.success(f"✅ VERIFICATION PASSED")
        logger.success(f"   All {n_samples} tests within tolerance {tolerance:.1e}")
        logger.success(f"   Batch gradient computation is numerically equivalent")
    else:
        logger.error(f"❌ VERIFICATION FAILED")
        logger.error(f"   Some tests exceeded tolerance {tolerance:.1e}")
        logger.error(f"   Max errors: k={max_err_k:.3e}, omega={max_err_w:.3e}")
    logger.info(f"{'='*70}")
    
    return passed


if __name__ == "__main__":
    # This would be called with actual mesh data in practice
    logger.info("Gradient equivalence verification module loaded")
    logger.info("Use verify_gradient_equivalence(geom) to run tests")
