"""GPUFRSolver 初始化方法混入类。

从 gpu_solver.py 拆出，控制单文件行数。包含面几何、模态滤波和壁面距离
的 GPU 初始化方法。
"""

import numpy as np
from loguru import logger

from autoflowcfd.core.gpu import get_cupy


class _GPUSolverInitMixin:
    """GPUFRSolver 初始化方法混入。

    子类需要提供：mesh, ops, device_id, order 等属性。
    """

    def _init_face_geometry(self):
        """初始化 GPU 面几何缓存。"""
        try:
            from autoflowcfd.core.fr_operators.face_kernels import get_flat_face_geometry
            flat_face = get_flat_face_geometry(self.mesh, self.ops)
            from autoflowcfd.core.gpu.gpu_face_geometry import build_gpu_flat_face
            self.flat_face_gpu = build_gpu_flat_face(flat_face, self.device_id)
        except Exception as e:
            logger.warning(f"Face geometry init failed: {e}")
            self.flat_face_gpu = None

    def _init_modal_filter_gpu(self):
        """初始化 GPU 模态滤波回调函数。"""
        cp = get_cupy()
        n_cells = self.mesh.n_cells
        n_sps = self.mesh.n_sps_per_cell
        n_prism = self.mesh.n_prism_cells

        try:
            filter_prism = self.ops.filter_prism
            filter_tet = self.ops.filter_tet

            if filter_prism is not None or filter_tet is not None:
                from autoflowcfd.core.gpu.gpu_modal_filter import build_gpu_filter_func
                self.filter_func_gpu = build_gpu_filter_func(
                    n_cells, n_sps, n_prism,
                    filter_prism, filter_tet,
                    device_id=self.device_id,
                )
                logger.debug("GPU modal filter initialized")
        except Exception as e:
            logger.warning(f"Modal filter init failed: {e}, running without filter")
            self.filter_func_gpu = None

    def _init_wall_distance_gpu(self):
        """预计算壁面距离场并上传到 GPU。

        使用 KD-Tree 欧氏距离（与 CPU 版一致），在初始化时一次性计算。
        """
        cp = get_cupy()
        n_cells = self.mesh.n_cells
        n_sps = self.mesh.n_sps_per_cell

        try:
            if hasattr(self.mesh, 'sps_coords') and self.mesh.sps_coords is not None:
                sps_coords = self.mesh.sps_coords.reshape(-1, 3)
            elif hasattr(self.mesh, 'cell_centers') and self.mesh.cell_centers is not None:
                sps_coords = np.tile(self.mesh.cell_centers, (1, n_sps)).reshape(-1, 3)
            else:
                logger.warning("No SP/cell-center coordinates available for wall distance")
                self.wall_distance_gpu = cp.ones((n_cells, n_sps), dtype=cp.float64) * 0.01
                return

            wall_indices = None
            if hasattr(self.mesh, 'boundary_groups'):
                for bg_name, bg in self.mesh.boundary_groups.items():
                    if 'WALL' in bg_name.upper() or bg.get('type', '').upper() == 'WALL':
                        wall_indices = bg.get('node_indices')
                        break
            if wall_indices is None and hasattr(self.mesh, 'nodes'):
                wall_indices = np.array([], dtype=np.int64)

            if wall_indices is not None and len(wall_indices) > 0:
                from scipy.spatial import cKDTree
                wall_coords = self.mesh.nodes[wall_indices]
                tree = cKDTree(wall_coords)
                dist_flat, _ = tree.query(sps_coords, k=1)
                self.wall_distance_gpu = cp.asarray(
                    dist_flat.reshape(n_cells, n_sps)
                )
                logger.info(f"Wall distance computed: min={dist_flat.min():.6e}, max={dist_flat.max():.6e}")
            else:
                volumes = self.mesh_data.get('cell_volumes')
                if volumes is None:
                    volumes = cp.asarray(self.mesh.get_all_cell_volumes())
                h_char = volumes ** (1.0 / 3.0)
                self.wall_distance_gpu = cp.broadcast_to(
                    h_char[:, None], (n_cells, n_sps)
                ).copy()
                logger.warning("Wall distance: using characteristic length as estimate")
        except Exception as e:
            logger.warning(f"Wall distance computation failed: {e}, using fallback")
            self.wall_distance_gpu = cp.ones((n_cells, n_sps), dtype=cp.float64) * 0.01
