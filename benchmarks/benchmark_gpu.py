"""
AutoFlowCFD V2.0 - GPU 性能基准测试

测试 GPU 各计算模块的实际性能（需要 CuPy + CUDA GPU）。
覆盖：张量收缩、物理通量、AUSM+up、完整残差评估。

使用:
    poetry run python benchmarks/benchmark_gpu.py

预期性能（RTX 3090, 百万级网格）:
- 张量收缩 (cuBLAS): >10x vs CPU
- 物理通量 (CuPy 向量化): >5x vs CPU
- 完整无粘残差 (P2): >5x vs CPU
"""

import time
import sys
import numpy as np
from pathlib import Path
from loguru import logger

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


def check_gpu_availability():
    """检查 GPU 是否可用。"""
    try:
        import cupy as cp
        test_array = cp.zeros((100,))
        del test_array
        # 获取设备信息
        props = cp.cuda.runtime.getDeviceProperties(0)
        name = props.get('name', b'unknown')
        if isinstance(name, bytes):
            name = name.decode()
        mem_mb = props.get('totalGlobalMem', 0) / (1024**2)
        logger.success(f"GPU (CuPy) available: {name}, {mem_mb:.0f} MB")
        return True
    except ImportError:
        logger.warning("CuPy not installed - GPU benchmark skipped")
        return False
    except Exception as e:
        logger.error(f"GPU check failed: {e}")
        return False


def benchmark_tensor_contraction(n_cells=100000, n_sps=27, n_runs=10):
    """基准测试：GPU 张量收缩（体积项核心操作）。

    Args:
        n_cells: 单元数
        n_sps: 每单元 SP 数
        n_runs: 重复次数

    Returns:
        dict: 性能结果
    """
    import cupy as cp
    logger.info("=" * 60)
    logger.info(f"GPU Benchmark: Tensor Contraction ({n_cells} cells, {n_sps} SPs)")
    logger.info("=" * 60)

    D = cp.random.rand(n_sps, n_sps, dtype=cp.float64)
    X = cp.random.rand(n_cells, n_sps, 5, dtype=cp.float64)

    # 预热
    cp.tensordot(D, X, axes=([1], [1]))
    cp.cuda.Stream.null.synchronize()

    start = time.time()
    for _ in range(n_runs):
        result = cp.tensordot(D, X, axes=([1], [1]))
    cp.cuda.Stream.null.synchronize()
    elapsed = (time.time() - start) / n_runs

    gflops = 2.0 * n_cells * n_sps * n_sps * 5 / elapsed / 1e9
    logger.success(f"  Time: {elapsed*1000:.2f} ms per call")
    logger.success(f"  Throughput: {gflops:.1f} GFLOPS")

    # CPU 对比
    D_np = cp.asnumpy(D)
    X_np = cp.asnumpy(X)
    start_cpu = time.time()
    for _ in range(n_runs):
        np.tensordot(D_np, X_np, axes=([1], [1]))
    elapsed_cpu = (time.time() - start_cpu) / n_runs

    speedup = elapsed_cpu / elapsed
    logger.success(f"  CPU time: {elapsed_cpu*1000:.2f} ms")
    logger.success(f"  Speedup: {speedup:.1f}x")

    return {'gpu_ms': elapsed*1000, 'cpu_ms': elapsed_cpu*1000, 'speedup': speedup}


def benchmark_physical_flux(n_points=500000, n_runs=20):
    """基准测试：GPU 欧拉物理通量。"""
    import cupy as cp
    from autoflowcfd.core.gpu.gpu_flux import euler_physical_flux_gpu

    logger.info("=" * 60)
    logger.info(f"GPU Benchmark: Euler Physical Flux ({n_points} points)")
    logger.info("=" * 60)

    Q = cp.zeros((n_points, 5), dtype=cp.float64)
    Q[:, 0] = 1.225
    Q[:, 1] = 30.0
    Q[:, 4] = 101325.0

    # 预热
    euler_physical_flux_gpu(Q)
    cp.cuda.Stream.null.synchronize()

    start = time.time()
    for _ in range(n_runs):
        euler_physical_flux_gpu(Q)
    cp.cuda.Stream.null.synchronize()
    elapsed = (time.time() - start) / n_runs

    logger.success(f"  Time: {elapsed*1000:.2f} ms per call")
    logger.success(f"  Throughput: {n_points/elapsed/1e6:.1f} M points/s")

    return {'gpu_ms': elapsed*1000}


def benchmark_ausm_up(n_faces=200000, n_runs=20):
    """基准测试：GPU AUSM+up 通量计算。"""
    import cupy as cp
    from autoflowcfd.core.gpu.gpu_inviscid import _ausm_up_flux_batch_gpu

    logger.info("=" * 60)
    logger.info(f"GPU Benchmark: AUSM+up Flux ({n_faces} faces)")
    logger.info("=" * 60)

    Q_L = cp.zeros((n_faces, 1, 5), dtype=cp.float64)
    Q_R = cp.zeros((n_faces, 1, 5), dtype=cp.float64)
    Q_L[:, 0, 0] = 1.225
    Q_L[:, 0, 1] = 30.0
    Q_L[:, 0, 4] = 101325.0
    Q_R[:] = Q_L[:]
    normal = cp.zeros((n_faces, 3), dtype=cp.float64)
    normal[:, 0] = 1.0

    # 预热
    _ausm_up_flux_batch_gpu(Q_L, Q_R, normal)
    cp.cuda.Stream.null.synchronize()

    start = time.time()
    for _ in range(n_runs):
        _ausm_up_flux_batch_gpu(Q_L, Q_R, normal)
    cp.cuda.Stream.null.synchronize()
    elapsed = (time.time() - start) / n_runs

    logger.success(f"  Time: {elapsed*1000:.2f} ms per call")
    logger.success(f"  Throughput: {n_faces/elapsed/1e6:.1f} M faces/s")

    return {'gpu_ms': elapsed*1000}


def benchmark_memory_bandwidth(n_cells=1000000, n_runs=20):
    """基准测试：GPU 内存带宽（CPU↔GPU 传输）。"""
    import cupy as cp

    logger.info("=" * 60)
    logger.info(f"GPU Benchmark: Memory Bandwidth ({n_cells} cells)")
    logger.info("=" * 60)

    # 上传带宽
    arr_np = np.random.rand(n_cells, 27, 5).astype(np.float64)
    size_mb = arr_np.nbytes / (1024**2)

    start = time.time()
    for _ in range(n_runs):
        arr_gpu = cp.asarray(arr_np)
    cp.cuda.Stream.null.synchronize()
    elapsed_up = (time.time() - start) / n_runs

    # 下载带宽
    arr_gpu = cp.asarray(arr_np)
    start = time.time()
    for _ in range(n_runs):
        arr_back = cp.asnumpy(arr_gpu)
    elapsed_down = (time.time() - start) / n_runs

    bw_up = size_mb / elapsed_up
    bw_down = size_mb / elapsed_down

    logger.success(f"  Upload: {bw_up:.1f} MB/s ({elapsed_up*1000:.2f} ms)")
    logger.success(f"  Download: {bw_down:.1f} MB/s ({elapsed_down*1000:.2f} ms)")

    return {'upload_mb_s': bw_up, 'download_mb_s': bw_down}


def main():
    """运行所有 GPU 基准测试。"""
    logger.info("AutoFlowCFD GPU Performance Benchmark Suite")
    logger.info("=" * 60)

    if not check_gpu_availability():
        logger.warning("GPU not available - skipping GPU benchmarks")
        return

    results = {}

    benchmarks = [
        ("tensor_contraction", benchmark_tensor_contraction),
        ("physical_flux", benchmark_physical_flux),
        ("ausm_up", benchmark_ausm_up),
        ("memory_bandwidth", benchmark_memory_bandwidth),
    ]

    for name, func in benchmarks:
        try:
            result = func()
            results[name] = result
        except Exception as e:
            logger.error(f"{name} benchmark failed: {e}")
            results[name] = None
        logger.info("")

    # 汇总
    logger.info("=" * 60)
    logger.info("GPU Benchmark Summary")
    logger.info("=" * 60)
    for name, result in results.items():
        if result is not None:
            logger.info(f"  {name}: {result}")
        else:
            logger.warning(f"  {name}: FAILED")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
