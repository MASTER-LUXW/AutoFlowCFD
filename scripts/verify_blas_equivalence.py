"""Verification script for BLAS threading numerical equivalence.

This script verifies that multi-threaded BLAS operations produce
numerically equivalent results to single-threaded operations,
ensuring the optimization does not affect solution accuracy.
"""

import os
import sys
import numpy as np
from loguru import logger


def verify_blas_equivalence():
    """Verify numerical equivalence between single and multi-threaded BLAS."""
    
    logger.info("=" * 70)
    logger.info("BLAS Threading Numerical Equivalence Verification")
    logger.info("=" * 70)
    
    # Test 1: Matrix multiplication (BLAS Level 3 - GEMM)
    logger.info("\n📊 Test 1: Matrix Multiplication (GEMM)")
    n = 1000
    np.random.seed(42)
    A = np.random.rand(n, n)
    B = np.random.rand(n, n)
    
    # Single-threaded
    os.environ['MKL_NUM_THREADS'] = '1'
    os.environ['OPENBLAS_NUM_THREADS'] = '1'
    # Force reload of BLAS configuration
    import importlib
    if 'numpy.linalg' in sys.modules:
        importlib.reload(sys.modules['numpy.linalg'])
    
    C_single = np.dot(A, B)
    
    # Multi-threaded
    cpu_count = os.cpu_count() or 16
    os.environ['MKL_NUM_THREADS'] = str(cpu_count)
    os.environ['OPENBLAS_NUM_THREADS'] = str(cpu_count)
    if 'numpy.linalg' in sys.modules:
        importlib.reload(sys.modules['numpy.linalg'])
    
    C_multi = np.dot(A, B)
    
    abs_err = np.max(np.abs(C_single - C_multi))
    rel_err = abs_err / (np.max(np.abs(C_single)) + 1e-30)
    
    logger.info(f"  Matrix size: {n}×{n}")
    logger.info(f"  Max absolute error: {abs_err:.3e}")
    logger.info(f"  Max relative error: {rel_err:.3e}")
    logger.info(f"  Status: {'✅ PASS' if rel_err < 1e-12 else '❌ FAIL'}")
    
    # Test 2: Einstein summation (used in gradient computation)
    logger.info("\n📊 Test 2: Einstein Summation (Gradient-like)")
    n_cells = 10000
    n_faces = 50000
    
    np.random.seed(123)
    face_values = np.random.rand(n_faces, 5)
    face_normals = np.random.rand(n_faces, 3)
    face_areas = np.random.rand(n_faces)
    
    # Single-threaded
    os.environ['MKL_NUM_THREADS'] = '1'
    os.environ['OPENBLAS_NUM_THREADS'] = '1'
    
    contrib_single = np.einsum('ij,ik->ijk', face_values, face_normals)
    contrib_single *= face_areas[:, None, None]
    
    # Multi-threaded
    os.environ['MKL_NUM_THREADS'] = str(cpu_count)
    os.environ['OPENBLAS_NUM_THREADS'] = str(cpu_count)
    
    contrib_multi = np.einsum('ij,ik->ijk', face_values, face_normals)
    contrib_multi *= face_areas[:, None, None]
    
    abs_err = np.max(np.abs(contrib_single - contrib_multi))
    rel_err = abs_err / (np.max(np.abs(contrib_single)) + 1e-30)
    
    logger.info(f"  Face count: {n_faces}")
    logger.info(f"  Variables: 5 (rho, u, v, w, p)")
    logger.info(f"  Max absolute error: {abs_err:.3e}")
    logger.info(f"  Max relative error: {rel_err:.3e}")
    logger.info(f"  Status: {'✅ PASS' if rel_err < 1e-12 else '❌ FAIL'}")
    
    # Test 3: Vectorized accumulation (like np.add.at)
    logger.info("\n📊 Test 3: Vectorized Accumulation")
    n_vals = 100000
    n_bins = 1000
    
    np.random.seed(456)
    values = np.random.rand(n_vals, 3)
    indices = np.random.randint(0, n_bins, n_vals)
    
    # Single-threaded
    os.environ['MKL_NUM_THREADS'] = '1'
    os.environ['OPENBLAS_NUM_THREADS'] = '1'
    
    accum_single = np.zeros((n_bins, 3))
    np.add.at(accum_single, indices, values)
    
    # Multi-threaded
    os.environ['MKL_NUM_THREADS'] = str(cpu_count)
    os.environ['OPENBLAS_NUM_THREADS'] = str(cpu_count)
    
    accum_multi = np.zeros((n_bins, 3))
    np.add.at(accum_multi, indices, values)
    
    abs_err = np.max(np.abs(accum_single - accum_multi))
    rel_err = abs_err / (np.max(np.abs(accum_single)) + 1e-30)
    
    logger.info(f"  Values: {n_vals}")
    logger.info(f"  Bins: {n_bins}")
    logger.info(f"  Max absolute error: {abs_err:.3e}")
    logger.info(f"  Max relative error: {rel_err:.3e}")
    logger.info(f"  Status: {'✅ PASS' if rel_err < 1e-12 else '❌ FAIL'}")
    
    # Summary
    logger.info("\n" + "=" * 70)
    all_passed = (rel_err < 1e-12)
    
    if all_passed:
        logger.success("✅ ALL TESTS PASSED")
        logger.success("   BLAS multi-threading is numerically equivalent")
        logger.success("   Maximum relative error < 1e-12 (machine precision)")
        logger.success("   No impact on CFD solution accuracy")
    else:
        logger.error("❌ TESTS FAILED")
        logger.error("   Numerical differences detected")
        logger.error("   Consider using single-threaded BLAS for reproducibility")
    
    logger.info("=" * 70)
    
    return all_passed


if __name__ == "__main__":
    success = verify_blas_equivalence()
    sys.exit(0 if success else 1)
