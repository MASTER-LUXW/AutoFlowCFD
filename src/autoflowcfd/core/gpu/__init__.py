"""
AutoFlowCFD V2.0 - GPU 加速计算模块（CuPy 统一框架）

基于 CuPy 实现完整的 GPU 加速管线，覆盖 FR 求解器的所有核心计算：
- array_manager.py: GPU 数组管理与设备管理
- gpu_p0_inviscid.py: P0 无粘残差 CUDA kernel（CuPy RawKernel）
- gpu_volume_contract.py: GPU 张量收缩（体积项）
- gpu_flux.py: GPU 欧拉/粘性物理通量
- gpu_face_geometry.py: GPU 版面几何缓存
- gpu_kernels.py: GPU AUSM+up + 界面校正 kernel
- gpu_gradients.py: GPU 物理梯度
- gpu_viscous.py: GPU 粘性残差
- gpu_time_integration.py: GPU 时间积分
- gpu_solver.py: GPU FRSolver
- gpu_distributed.py: 多 GPU + MPI 分布式求解器
- gpu_halo_exchange.py: GPU 直接 Halo 交换（CUDA-aware MPI / staging buffer）
- gpu_turbulence_sst.py: GPU SST k-ω 湍流模型源项

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
