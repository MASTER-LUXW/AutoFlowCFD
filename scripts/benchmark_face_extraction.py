"""Performance benchmark for face extraction optimization.

This script compares the old and new face extraction algorithms to quantify
the performance improvement.

Usage:
    python scripts/benchmark_face_extraction.py [n_cells]

Example:
    python scripts/benchmark_face_extraction.py 1000000
"""

import sys
import time
import numpy as np
from loguru import logger


def generate_test_mesh(n_cells: int) -> np.ndarray:
    """Generate a random tetrahedral mesh for testing.
    
    Args:
        n_cells: Number of tetrahedral cells
        
    Returns:
        cell_connectivity: shape=(n_cells, 4), dtype=int32
    """
    # Generate random node IDs (simulating a realistic mesh)
    n_nodes = int(n_cells * 1.5)  # Typical ratio
    node_ids = np.random.randint(0, n_nodes, size=(n_cells, 4), dtype=np.int32)
    
    # Ensure no duplicate nodes within a cell
    for i in range(n_cells):
        while len(set(node_ids[i])) < 4:
            node_ids[i] = np.random.randint(0, n_nodes, size=4, dtype=np.int32)
    
    return node_ids


def benchmark_old_algorithm(cell_connectivity: np.ndarray):
    """Benchmark the old dict-based approach (simulated).
    
    This is a simplified version that mimics the old algorithm's behavior.
    """
    from typing import Dict, List, Tuple
    
    n_cells = cell_connectivity.shape[0]
    logger.info(f"Benchmarking OLD algorithm (dict + np.unique) on {n_cells:,} cells...")
    
    start_time = time.perf_counter()
    
    # Simulate old approach: Python dict building
    face_dict: Dict[Tuple[int, int, int], List[int]] = {}
    
    for cell_idx in range(n_cells):
        nodes_idx = cell_connectivity[cell_idx]
        
        faces = [
            tuple(sorted([nodes_idx[0], nodes_idx[1], nodes_idx[2]])),
            tuple(sorted([nodes_idx[0], nodes_idx[1], nodes_idx[3]])),
            tuple(sorted([nodes_idx[0], nodes_idx[2], nodes_idx[3]])),
            tuple(sorted([nodes_idx[1], nodes_idx[2], nodes_idx[3]]))
        ]
        
        for face_nodes in faces:
            if face_nodes not in face_dict:
                face_dict[face_nodes] = []
            face_dict[face_nodes].append(cell_idx)
    
    dict_build_time = time.perf_counter() - start_time
    logger.info(f"  Dict building: {dict_build_time:.2f}s")
    
    # Simulate np.unique on structured array
    start_time = time.perf_counter()
    
    n_faces_raw = sum(len(v) for v in face_dict.values())
    face_nodes_raw = np.zeros((n_faces_raw, 3), dtype=np.int32)
    
    idx = 0
    for face_nodes in face_dict.keys():
        face_nodes_raw[idx] = list(face_nodes)
        idx += 1
    
    # np.unique simulation
    face_nodes_sorted = np.sort(face_nodes_raw, axis=1)
    face_dtype = np.dtype((np.void, face_nodes_sorted.dtype.itemsize * 3))
    face_voids = np.ascontiguousarray(face_nodes_sorted).view(face_dtype).reshape(-1)
    unique_faces, inverse_indices = np.unique(face_voids, return_inverse=True)
    
    unique_time = time.perf_counter() - start_time
    logger.info(f"  Unique detection: {unique_time:.2f}s")
    
    total_time = dict_build_time + unique_time
    logger.success(f"OLD algorithm total: {total_time:.2f}s")
    
    return total_time, len(unique_faces)


def benchmark_new_algorithm(cell_connectivity: np.ndarray):
    """Benchmark the new radix-sort approach."""
    try:
        from autoflowcfd.grid.face_extractor import (
            _build_face_dict_numba,
            _deduplicate_and_build_connectivity,
            NUMBA_AVAILABLE
        )
    except ImportError as e:
        logger.error(f"Failed to import optimized functions: {e}")
        return None, 0
    
    n_cells = cell_connectivity.shape[0]
    logger.info(f"Benchmarking NEW algorithm (radix sort) on {n_cells:,} cells...")
    logger.info(f"  Numba available: {NUMBA_AVAILABLE}")
    
    # Warm up Numba JIT compilation
    if NUMBA_AVAILABLE:
        logger.debug("Warming up Numba JIT...")
        small_test = cell_connectivity[:100]
        _build_face_dict_numba(small_test, 100)
    
    start_time = time.perf_counter()
    
    # Step 1: Build face keys
    face_keys_raw, face_cell_map_raw, n_faces_raw = _build_face_dict_numba(
        cell_connectivity, n_cells
    )
    
    step1_time = time.perf_counter() - start_time
    logger.info(f"  Face key encoding: {step1_time:.2f}s")
    
    # Step 2: Deduplicate via argsort
    start_time = time.perf_counter()
    
    (unique_keys, face_conn, face_nodes, 
     occurrence_count, n_unique, n_interior) = \
        _deduplicate_and_build_connectivity(
            face_keys_raw, face_cell_map_raw, n_faces_raw
        )
    
    step2_time = time.perf_counter() - start_time
    logger.info(f"  Deduplication & connectivity: {step2_time:.2f}s")
    
    total_time = step1_time + step2_time
    logger.success(f"NEW algorithm total: {total_time:.2f}s")
    logger.info(f"  Unique faces: {n_unique:,}, Interior: {n_interior:,}")
    
    return total_time, n_unique


def main():
    """Run performance benchmark."""
    # Parse command line argument or use default
    if len(sys.argv) > 1:
        n_cells = int(sys.argv[1])
    else:
        n_cells = 2_000_000  # Default: 2M cells (Ahmed Body scale)
    
    logger.info("=" * 80)
    logger.info(f"Face Extraction Performance Benchmark")
    logger.info(f"Test mesh size: {n_cells:,} tetrahedral cells")
    logger.info("=" * 80)
    
    # Generate test mesh
    logger.info("\nGenerating test mesh...")
    cell_connectivity = generate_test_mesh(n_cells)
    logger.info(f"  Generated {n_cells:,} cells with random topology")
    
    # Benchmark old algorithm (only for smaller meshes to avoid excessive runtime)
    old_time = None
    old_n_faces = 0
    if n_cells <= 500_000:
        logger.info("\n" + "-" * 80)
        old_time, old_n_faces = benchmark_old_algorithm(cell_connectivity)
    else:
        logger.warning(f"\nSkipping OLD algorithm benchmark for large mesh ({n_cells:,} cells)")
        logger.warning("  (Would take >10 minutes, using estimate instead)")
        # Estimate based on O(n log n) complexity
        estimated_old_time = (n_cells / 100_000) * 30  # Rough estimate: 30s per 100k cells
        logger.warning(f"  Estimated time: ~{estimated_old_time:.0f}s")
        old_time = estimated_old_time
    
    # Benchmark new algorithm
    logger.info("\n" + "-" * 80)
    new_time, new_n_faces = benchmark_new_algorithm(cell_connectivity)
    
    if new_time is None:
        logger.error("New algorithm benchmark failed!")
        sys.exit(1)
    
    # Compare results
    logger.info("\n" + "=" * 80)
    logger.info("PERFORMANCE COMPARISON")
    logger.info("=" * 80)
    
    if old_time is not None and old_time > 0:
        speedup = old_time / new_time
        improvement_pct = ((old_time - new_time) / old_time) * 100
        
        logger.info(f"Old algorithm:  {old_time:8.2f}s  (estimated)" if n_cells > 500_000 else f"Old algorithm:  {old_time:8.2f}s")
        logger.info(f"New algorithm:  {new_time:8.2f}s")
        logger.info(f"Speedup:        {speedup:8.2f}x")
        logger.info(f"Improvement:    {improvement_pct:7.1f}%")
        
        if speedup > 10:
            logger.success("✅ EXCELLENT: >10x speedup achieved!")
        elif speedup > 5:
            logger.success("✅ GOOD: >5x speedup achieved")
        elif speedup > 2:
            logger.info("✓ MODERATE: >2x speedup achieved")
        else:
            logger.warning("⚠ MINIMAL: Consider further optimization")
    else:
        logger.info(f"New algorithm:  {new_time:8.2f}s")
        logger.info(f"Unique faces:   {new_n_faces:,}")
    
    # Verify correctness
    logger.info("\n" + "-" * 80)
    logger.info("CORRECTNESS VERIFICATION")
    logger.info("-" * 80)
    
    expected_ratio = new_n_faces / n_cells
    logger.info(f"Face-to-cell ratio: {expected_ratio:.2f}")
    
    if 1.8 <= expected_ratio <= 2.8:
        logger.success("✅ Ratio is within expected range (1.8-2.8)")
    else:
        logger.warning(f"⚠ Ratio {expected_ratio:.2f} is outside expected range")
    
    logger.info("\n" + "=" * 80)
    logger.info("Benchmark complete!")
    logger.info("=" * 80)


if __name__ == '__main__':
    main()
