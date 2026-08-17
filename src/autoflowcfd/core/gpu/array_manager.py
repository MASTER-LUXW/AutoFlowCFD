"""
AutoFlowCFD V2.0 - GPU 数组管理与设备管理

管理 CPU↔GPU 数据传输与 GPU 内存生命周期。核心设计：

1. 数据驻留策略：网格数据（jacobians、face geometry）和求解状态（U、Q）
   常驻 GPU 显存，只在 checkpoint/output 时传回 CPU
2. 内存池管理：使用 CuPy 内置 MemoryPool，监控显存使用
3. 批量传输：upload_mesh_data() 一次性上传所有网格数据，减少传输开销
4. 设备选择：支持多 GPU 环境下指定设备 ID

使用:
    mgr = GPUArrayManager(device_id=0)
    mgr.upload_mesh_data(mesh, ops)
    U_gpu = mgr.to_gpu(U_numpy)
    # ... GPU 计算 ...
    U_result = mgr.to_cpu(U_gpu)
"""

import numpy as np
from typing import Dict, Optional, Any
from loguru import logger

from autoflowcfd.core.gpu import gpu_available, get_cupy


class GPUArrayManager:
    """GPU 数组管理器。

    管理 CuPy GPU 数组的生命周期和数据传输。所有网格数据上传后常驻 GPU，
    求解状态也在 GPU 上创建和更新，只在 I/O 时传回 CPU。

    Attributes:
        device_id: GPU 设备 ID
        mesh_data: 已上传的网格数据字典（CuPy 数组）
        state_data: 求解状态数据字典（CuPy 数组）
    """

    def __init__(self, device_id: int = 0):
        """初始化 GPU 数组管理器。

        Args:
            device_id: GPU 设备 ID（默认 0）

        Raises:
            RuntimeError: CuPy 不可用或指定设备不存在
        """
        if not gpu_available:
            raise RuntimeError(
                "CuPy is not available. Install with: pip install cupy-cuda12x"
            )

        self._cp = get_cupy()
        self.device_id = device_id

        # 切换到指定设备
        n_devices = self._cp.cuda.runtime.getDeviceCount()
        if device_id >= n_devices:
            raise RuntimeError(
                f"GPU device {device_id} requested but only {n_devices} devices available"
            )

        with self._cp.cuda.Device(device_id):
            self._stream = self._cp.cuda.Stream(non_blocking=True)
            self._mem_pool = self._cp.cuda.get_default_memory_pool()

        self.mesh_data: Dict[str, Any] = {}
        self.state_data: Dict[str, Any] = {}

        # 设备信息
        props = self._cp.cuda.runtime.getDeviceProperties(device_id)
        name = props.get('name', b'unknown')
        if isinstance(name, bytes):
            name = name.decode()
        self._device_name = str(name)
        self._total_memory_mb = props.get('totalGlobalMem', 0) / (1024 ** 2)

        logger.info(
            f"GPUArrayManager initialized: device {device_id} ({self._device_name}), "
            f"{self._total_memory_mb:.0f} MB total memory"
        )

    @property
    def cp(self):
        """返回 CuPy 模块引用。"""
        return self._cp

    @property
    def stream(self):
        """返回当前 CUDA stream。"""
        return self._stream

    @property
    def device(self):
        """返回当前 CuPy Device 上下文。"""
        return self._cp.cuda.Device(self.device_id)

    # ─── 基础传输 ───────────────────────────────────────────

    def to_gpu(self, np_array: np.ndarray) -> 'cp.ndarray':
        """CPU numpy 数组 → GPU CuPy 数组。

        Args:
            np_array: CPU 上的 numpy 数组

        Returns:
            GPU 上的 CuPy 数组
        """
        with self.device:
            return self._cp.asarray(np_array)

    def to_cpu(self, cp_array: 'cp.ndarray') -> np.ndarray:
        """GPU CuPy 数组 → CPU numpy 数组。

        Args:
            cp_array: GPU 上的 CuPy 数组

        Returns:
            CPU 上的 numpy 数组
        """
        return self._cp.asnumpy(cp_array)

    def gpu_zeros(self, shape, dtype=np.float64) -> 'cp.ndarray':
        """在 GPU 上创建零数组。

        Args:
            shape: 数组形状
            dtype: 数据类型

        Returns:
            GPU 上的零 CuPy 数组
        """
        with self.device:
            return self._cp.zeros(shape, dtype=dtype)

    def gpu_empty(self, shape, dtype=np.float64) -> 'cp.ndarray':
        """在 GPU 上创建未初始化数组。

        Args:
            shape: 数组形状
            dtype: 数据类型

        Returns:
            GPU 上的 CuPy 数组（未初始化）
        """
        with self.device:
            return self._cp.empty(shape, dtype=dtype)

    def gpu_copy(self, cp_array: 'cp.ndarray') -> 'cp.ndarray':
        """在 GPU 上复制数组。

        Args:
            cp_array: 源 CuPy 数组

        Returns:
            新的 CuPy 数组（深拷贝）
        """
        return cp_array.copy()

    # ─── 网格数据上传 ────────────────────────────────────────

    def upload_mesh_data(self, mesh, ops) -> Dict[str, 'cp.ndarray']:
        """一次性上传网格的所有常驻 GPU 数据。

        包括 Jacobian 度量项、面连接关系、面几何信息等。上传后缓存
        在 self.mesh_data 中，后续计算直接引用。

        Args:
            mesh: HighOrderMesh 实例
            ops: FROperators 实例

        Returns:
            mesh_data 字典（key → CuPy 数组）
        """
        with self.device:
            n_cells = mesh.n_cells
            n_sps = mesh.n_sps_per_cell
            n1d = mesh.n_points_1d
            n_prism = mesh.n_prism_cells

            self.mesh_data['n_cells'] = n_cells
            self.mesh_data['n_sps'] = n_sps
            self.mesh_data['n1d'] = n1d
            self.mesh_data['n_prism'] = n_prism

            # ── Jacobian 度量项 ──
            if mesh.jacobians is not None:
                det_jacs = mesh.jacobians['det_jacs'].reshape(n_cells, n_sps)
                inv_jacs = mesh.jacobians['inv_jacs'].reshape(n_cells, n_sps, 3, 3)
                self.mesh_data['det_jacs'] = self._cp.asarray(
                    np.ascontiguousarray(det_jacs, dtype=np.float64)
                )
                self.mesh_data['inv_jacs'] = self._cp.asarray(
                    np.ascontiguousarray(inv_jacs, dtype=np.float64)
                )
                # adj(J) = det(J) * inv(J)，用于逆变通量计算
                adj_j = det_jacs[..., None, None] * inv_jacs
                self.mesh_data['adj_j'] = self._cp.asarray(
                    np.ascontiguousarray(adj_j, dtype=np.float64)
                )

            # ── Fine Jacobian（over-integration 去混叠）──
            if mesh.jacobians_fine is not None:
                n_fine = mesh.n_sps_per_cell_fine
                det_jacs_fine = mesh.jacobians_fine['det_jacs'].reshape(n_cells, n_fine)
                inv_jacs_fine = mesh.jacobians_fine['inv_jacs'].reshape(n_cells, n_fine, 3, 3)
                adj_j_fine = det_jacs_fine[..., None, None] * inv_jacs_fine
                self.mesh_data['det_jacs_fine'] = self._cp.asarray(
                    np.ascontiguousarray(det_jacs_fine, dtype=np.float64)
                )
                self.mesh_data['inv_jacs_fine'] = self._cp.asarray(
                    np.ascontiguousarray(inv_jacs_fine, dtype=np.float64)
                )
                self.mesh_data['adj_j_fine'] = self._cp.asarray(
                    np.ascontiguousarray(adj_j_fine, dtype=np.float64)
                )
                self.mesh_data['n_fine'] = n_fine

            # ── FR 算子（不依赖 cell，上传一次）──
            if ops is not None:
                # 微分矩阵
                for attr_name in ['D_3d_tet', 'D_3d_prism']:
                    D = getattr(ops, attr_name, None)
                    if D is not None:
                        self.mesh_data[attr_name] = self._cp.asarray(
                            np.ascontiguousarray(D, dtype=np.float64)
                        )

                # Over-integration 算子
                for attr_name in [
                    'overint_interp_c2f_tet', 'overint_interp_c2f_prism',
                    'overint_D_fine_tet', 'overint_D_fine_prism',
                    'overint_restrict_f2c_tet', 'overint_restrict_f2c_prism',
                ]:
                    op = getattr(ops, attr_name, None)
                    if op is not None:
                        self.mesh_data[attr_name] = self._cp.asarray(
                            np.ascontiguousarray(op, dtype=np.float64)
                        )

            # ── Cell volumes（P0 路径需要）──
            if hasattr(mesh, 'cell_volumes') and mesh.cell_volumes is not None:
                self.mesh_data['cell_volumes'] = self._cp.asarray(
                    np.ascontiguousarray(mesh.cell_volumes, dtype=np.float64)
                )

            logger.info(
                f"Mesh data uploaded to GPU: {n_cells} cells, {n_sps} SPs/cell, "
                f"P{n1d - 1}, {n_prism} prism + {n_cells - n_prism} tet"
            )

        return self.mesh_data

    def upload_face_geometry(self, flat_face_data: Dict[str, np.ndarray]):
        """上传面几何展平数据到 GPU。

        Args:
            flat_face_data: FlatFaceGeometry 展平后的 numpy 数组字典
        """
        with self.device:
            for key, arr in flat_face_data.items():
                if isinstance(arr, np.ndarray):
                    self.mesh_data[f'face_{key}'] = self._cp.asarray(
                        np.ascontiguousarray(arr)
                    )
                elif isinstance(arr, (list, tuple)):
                    # 面图着色索引列表
                    self.mesh_data[f'face_{key}'] = [
                        self._cp.asarray(np.ascontiguousarray(a)) if isinstance(a, np.ndarray) else a
                        for a in arr
                    ]
                else:
                    self.mesh_data[f'face_{key}'] = arr

    # ─── 状态管理 ────────────────────────────────────────────

    def upload_state(self, U: np.ndarray, Q: Optional[np.ndarray] = None):
        """上传求解器状态到 GPU。

        Args:
            U: 守恒变量 (n_cells, n_sps, n_vars)
            Q: 原始变量（可选，None 时从 U 计算）
        """
        with self.device:
            self.state_data['U'] = self._cp.asarray(
                np.ascontiguousarray(U, dtype=np.float64)
            )
            if Q is not None:
                self.state_data['Q'] = self._cp.asarray(
                    np.ascontiguousarray(Q, dtype=np.float64)
                )

    def download_state(self) -> Dict[str, np.ndarray]:
        """从 GPU 下载求解器状态到 CPU。

        Returns:
            状态字典（key → numpy 数组）
        """
        result = {}
        for key, arr in self.state_data.items():
            if isinstance(arr, self._cp.ndarray):
                result[key] = self._cp.asnumpy(arr)
        return result

    # ─── 显存监控 ────────────────────────────────────────────

    def get_memory_usage(self) -> Dict[str, float]:
        """获取当前 GPU 显存使用信息。

        Returns:
            显存信息字典（MB 单位）
        """
        with self.device:
            used = self._mem_pool.used_bytes() / (1024 ** 2)
            total = self._total_memory_mb
            free = total - used
            return {
                'used_mb': used,
                'total_mb': total,
                'free_mb': free,
                'usage_percent': used / total * 100 if total > 0 else 0,
            }

    def check_memory(self, estimated_bytes: int) -> bool:
        """检查是否有足够的 GPU 显存。

        Args:
            estimated_bytes: 预估需要的额外显存（字节）

        Returns:
            True 如果有足够显存
        """
        usage = self.get_memory_usage()
        needed_mb = estimated_bytes / (1024 ** 2)
        if needed_mb > usage['free_mb'] * 0.9:  # 留 10% 安全余量
            logger.warning(
                f"GPU memory warning: need ~{needed_mb:.0f} MB but only "
                f"{usage['free_mb']:.0f} MB free ({usage['usage_percent']:.1f}% used)"
            )
            return False
        return True

    # ─── 同步与清理 ──────────────────────────────────────────

    def synchronize(self):
        """同步 GPU 操作（等待所有 CUDA kernel 完成）。"""
        self._stream.synchronize()

    def cleanup(self):
        """释放所有 GPU 资源。"""
        with self.device:
            self.mesh_data.clear()
            self.state_data.clear()
            self._stream.synchronize()
            # 释放内存池中的空闲块
            self._mem_pool.free_all_blocks()
            logger.info("GPU resources cleaned up")

    def __del__(self):
        """析构时清理。"""
        try:
            self.cleanup()
        except Exception:
            pass
