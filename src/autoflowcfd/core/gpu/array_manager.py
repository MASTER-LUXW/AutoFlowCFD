"""
AutoFlowCFD V2.0 - GPU 设备与网格数据上传

网格度量与 FR 算子一次性上传、常驻显存（`upload_mesh_data`）；求解状态由各求解器自己在设备上创建，只在 checkpoint /
输出时传回主机。

2026-10-22（I-113）整理：删除零调用的 gpu_zeros / gpu_empty / gpu_copy / upload_face_geometry / upload_state /
download_state / check_memory / to_gpu / to_cpu 与一条创建了却没有任何 kernel 使用的非阻塞流（`synchronize()` 只同步
这条空流，依赖它等待计算完成是错的）；度量只存 det(J) 与 adj(J)（梯度按 adj/det 现算，不再另存一份 inv(J)，P2 30 万
单元省约 0.6 GB），与单 GPU 路径同一个上传函数；显存余量与 `core/utils/system_memory.py` 同一口径。
"""

from typing import Any, Dict

from loguru import logger

from autoflowcfd.core.gpu import get_cupy, gpu_available


class GPUArrayManager:
    """一块 GPU 设备上的网格数据上传与显存监控。

    Attributes:
        device_id: GPU 设备 ID
    """

    def __init__(self, device_id: int = 0):
        """
        Raises:
            RuntimeError: CuPy 不可用或指定设备不存在
        """
        if not gpu_available:
            raise RuntimeError("CuPy is not available. Install with: pip install cupy-cuda12x")
        self._cp = get_cupy()
        self.device_id = device_id
        n_devices = self._cp.cuda.runtime.getDeviceCount()
        if device_id >= n_devices:
            raise RuntimeError(f"GPU device {device_id} requested but only {n_devices} devices available")
        with self._cp.cuda.Device(device_id):
            # CuPy 的默认显存池是顶层函数（`cupy.cuda` 下没有 get_default_memory_pool）
            self._mem_pool = self._cp.get_default_memory_pool()

        props = self._cp.cuda.runtime.getDeviceProperties(device_id)
        name = props.get('name', b'unknown')
        self._device_name = name.decode() if isinstance(name, bytes) else str(name)
        self._total_memory_mb = props.get('totalGlobalMem', 0) / (1024 ** 2)
        logger.info(f"GPUArrayManager initialized: device {device_id} ({self._device_name}), "
                    f"{self._total_memory_mb:.0f} MB total memory")

    def upload_mesh_data(self, mesh, ops) -> Dict[str, Any]:
        """上传网格的常驻设备数据，返回新字典（换阶重传时不残留上一阶的键，例如降到 P0 后的过积分细点度量）。

        内容：单元 / 解点计数；度量 det(J) 与 adj(J)（解点）及过积分细点度量（`prepare_mesh_data`，与单 GPU 路径
        同一个函数）；FR 算子（`prepare_ops_data`）；单元体积（P0 路径与局部步长）。

        Args:
            mesh: HighOrderMesh，或压缩索引空间的同接口视图（多 GPU）
            ops: FROperators
        """
        from autoflowcfd.core.gpu.residual.gpu_inviscid_volume import prepare_mesh_data, prepare_ops_data

        cp = self._cp
        n_cells, n_sps, n_prism = mesh.n_cells, mesh.n_sps_per_cell, mesh.n_prism_cells
        data: Dict[str, Any] = {'n_cells': n_cells, 'n_sps': n_sps, 'n1d': mesh.n_points_1d}
        data.update(prepare_mesh_data(cp, mesh, self.device_id))
        data.update(prepare_ops_data(cp, ops, self.device_id))
        with cp.cuda.Device(self.device_id):
            data['cell_volumes'] = cp.asarray(mesh.cell_volumes, dtype=cp.float64)
        logger.info(f"Mesh data uploaded to GPU: {n_cells} cells, {n_sps} SPs/cell, "
                    f"P{mesh.n_points_1d - 1}, {n_prism} prism + {n_cells - n_prism} tet")
        return data

    def get_memory_usage(self) -> Dict[str, float]:
        """本设备显存（MB）：`used_mb` 为本进程显存池已占用；`free_mb` 为设备空闲加显存池缓存的空闲块（与
        `system_memory.available_bytes` 同一口径，2026-10-22 以前取"总显存 - 本进程已用"，不计其他进程与驱动占用）。"""
        from autoflowcfd.core.utils.system_memory import device_available_bytes

        with self._cp.cuda.Device(self.device_id):
            used = self._mem_pool.used_bytes() / (1024 ** 2)
            free = device_available_bytes(self._cp) / (1024 ** 2)
        total = self._total_memory_mb
        return {'used_mb': used, 'total_mb': total, 'free_mb': free,
                'usage_percent': used / total * 100 if total > 0 else 0.0}

    def cleanup(self) -> None:
        """等待设备上全部 kernel 完成，释放显存池里的空闲块。"""
        with self._cp.cuda.Device(self.device_id) as device:
            device.synchronize()
            self._mem_pool.free_all_blocks()
        logger.info("GPU resources cleaned up")
