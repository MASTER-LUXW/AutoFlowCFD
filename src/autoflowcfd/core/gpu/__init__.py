"""
AutoFlowCFD V2.0 - GPU 加速计算模块（CuPy 统一框架）

基于 CuPy 实现完整的 GPU 加速管线，覆盖 FR 求解器的所有核心计算。

子目录布局（V2.0 第四次评审第四轮重构，2026-08-28：原先 20 个文件全部
平铺在 core/gpu/ 下，与 CPU 侧按关注点分文件夹——fr_solver/、
fr_residual/、turbulence/、mpi/——不对称，随文件数增长愈发难以导航，
现按同一分类方式重新组织，一一对应）：
- solver/: 对应 CPU fr_solver/ —— gpu_solver.py（GPUFRSolver 主体）、
  gpu_solver_init.py、gpu_solver_io.py
- residual/: 对应 CPU fr_residual/ + fr_operators/ 数值核 —— gpu_inviscid.py、
  gpu_viscous.py、gpu_inviscid_volume.py、gpu_p0_inviscid.py（CUDA
  RawKernel）、gpu_volume_contract.py（张量收缩）、gpu_flux.py（欧拉/
  粘性物理通量）、gpu_gradients.py（物理梯度）
- turbulence/: 对应 CPU turbulence/ —— gpu_turbulence_sst.py（SST k-ω
  源项）及后续新增的 GPU 湍流模型
- distributed/: 对应 CPU mpi/ 的 GPU 端 —— gpu_distributed.py（多 GPU +
  MPI 分布式求解器）、gpu_distributed_init.py、gpu_halo_exchange.py
  （CUDA-aware MPI / staging buffer 直接 Halo 交换）

顶层保留（CPU 侧无直接对应的基础设施）：
- array_manager.py: GPU 数组管理与设备管理
- gpu_face_geometry.py: GPU 版面几何缓存
- gpu_modal_filter.py: GPU 模态滤波
- gpu_time_integration.py / gpu_time_integration_imex.py /
  gpu_time_integration_dual.py: GPU 时间积分（含 IMEX/双时间步）

设计原则:
1. 统一 CuPy 框架：所有 GPU 计算走 CuPy（RawKernel/ElementwiseKernel + 向量化 API）
2. GPU 数据驻留：网格数据和求解状态常驻 GPU 显存，只在 I/O 时传输
3. 图着色直接写入：界面校正复用面图着色，同色面无冲突，无需 atomic
4. CPU/GPU 双路径共存：FRSolver (CPU) 与 GPUFRSolver (GPU) 独立，通过 --backend 切换
5. GPU 直接通信：Halo 交换支持 CUDA-aware MPI 零拷贝和 staging buffer 两种模式
6. 完整物理模型：湍流模型源项全程在 GPU 执行

使用:
    autoflowcfd solve steady <grid_file> --backend gpu --gpu-device 0
    mpirun -np 4 autoflowcfd solve steady <grid_file> --backend gpu --multi-gpu
"""

# CuPy 是可选依赖——未安装时整个 GPU 模块不导入，不影响 CPU 路径
try:
    import cupy as cp
    gpu_available = True
except ImportError:
    cp = None
    gpu_available = False


def get_cupy():
    """返回 CuPy 模块，不可用时返回 None。"""
    return cp


def gpu_device_count() -> int:
    """可用 GPU 设备数量。"""
    if not gpu_available:
        return 0
    try:
        return cp.cuda.runtime.getDeviceCount()
    except Exception:
        return 0


def get_device_info(device_id: int = 0) -> dict:
    """获取指定 GPU 设备的详细信息。

    Args:
        device_id: GPU 设备 ID

    Returns:
        设备信息字典，不可用时返回空字典
    """
    if not gpu_available:
        return {'available': False}
    try:
        with cp.cuda.Device(device_id):
            props = cp.cuda.runtime.getDeviceProperties(device_id)
            return {
                'available': True,
                'device_id': device_id,
                'name': props.get('name', b'unknown').decode() if isinstance(props.get('name'), bytes) else str(props.get('name', 'unknown')),
                'compute_capability': f"{props.get('major', 0)}.{props.get('minor', 0)}",
                'total_memory_mb': props.get('totalGlobalMem', 0) / (1024 ** 2),
                'multi_processor_count': props.get('multiProcessorCount', 0),
            }
    except Exception as e:
        return {'available': False, 'error': str(e)}
