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
        """初始化 GPU 面几何缓存。

        此前这里用宽 `except Exception` 吞掉一切异常、把
        `self.flat_face_gpu` 静默设为 None——这正是分布式版本
        `gpu_distributed_init.py::_init_distributed_face_geometry` 在上
        一轮评审中被认定为"必须移除的假通过错误处理"的同一类问题（该
        文件文档明确记录了这个反面教材），第四次评审发现单 GPU 版本没
        有同步修复（发现9）：`flat_face_gpu=None` 时，`gpu_inviscid.py`/
        `gpu_viscous.py` 会在**每次**残差求值时重新构建+重新上传整套
        GPU 面几何（约15个数组的 cp.asarray 上传），既隐藏了初始化失败
        的真实原因，又严重拖慢求解、违背"GPU 数据常驻，只在 I/O 时传输"
        的既定设计原则。修复：去掉吞掉一切异常的 try/except——面几何
        构建失败时必须让求解器初始化真正失败，而不是静默退化成"每步
        重建"这种隐蔽的性能/正确性陷阱。
        """
        from autoflowcfd.core.fr_operators.face_kernels import get_flat_face_geometry
        flat_face = get_flat_face_geometry(self.mesh, self.ops)
        from autoflowcfd.core.gpu.gpu_face_geometry import build_gpu_flat_face
        self.flat_face_gpu = build_gpu_flat_face(flat_face, self.device_id)

    def _init_modal_filter_gpu(self):
        """初始化 GPU 模态滤波回调函数。"""
        cp = get_cupy()
        n_cells = self.mesh.n_cells
        n_sps = self.mesh.n_sps_per_cell
        n_prism = self.mesh.n_prism_cells

        # sensor 档尚未在本后端接线，直接报错而不是静默按 legacy 跑
        # （见 fr_solver/filter.py::resolve_filter_mode）。放在 try 之外，
        # 否则会被下面那个 `except Exception -> warning` 吞掉。
        from autoflowcfd.core.fr_solver.filter import resolve_filter_mode
        resolve_filter_mode("gpu-single")

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

        # 真实 bug 修复（V2.0 专家组盲审发现，2026-08-27）：此前坐标缺失
        # /计算异常时都静默回退到硬编码常量 0.01m——与 CPU 版
        # fr_solver/turbulence.py 的既定原则矛盾（该文件对同样情形显式
        # raise，理由"Industrial-grade calculation requires accurate
        # wall distance, not simplified estimates"）。本方法只在
        # `self.turb_model_gpu is not None`（真的需要壁面距离的湍流
        # 模型，#7 起还包括 sgs_model_gpu 即 WMLES/LES）时才被调用（见
        # gpu_solver.py 调用点），任何几何尺度不是
        # 恰好在 0.01m 量级的真实网格上，这个假常量会系统性带偏 F1/F2
        # 混合函数与 DDES 长度尺度——直接失败，不静默凑一个和网格无关
        # 的常数。
        if hasattr(self.mesh, 'sps_coords') and self.mesh.sps_coords is not None:
            sps_coords = self.mesh.sps_coords.reshape(-1, 3)
        elif hasattr(self.mesh, 'cell_centers') and self.mesh.cell_centers is not None:
            sps_coords = np.tile(self.mesh.cell_centers, (1, n_sps)).reshape(-1, 3)
        else:
            raise RuntimeError(
                f"Wall distance field not computed for turbulence model "
                f"'{self.turb_model_name}': mesh has neither sps_coords nor "
                f"cell_centers. Industrial-grade calculation requires accurate "
                f"wall distance, not simplified estimates."
            )

        wall_indices = None
        # 真实 bug 修复（2026-09-02，排查多GPU分布式SST时发现，与分布式
        # 本身无关，单机路径同样中招）：`hasattr(mesh, 'boundary_groups')`
        # 对"属性存在但值是 None"（没有边界组元数据的网格）恒为 True，
        # `.items()` 会真实 AttributeError——用 `getattr(...) is not None`
        # 才是正确的存在性判据。
        boundary_groups = getattr(self.mesh, 'boundary_groups', None)
        if boundary_groups is not None:
            for bg_name, bg in boundary_groups.items():
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
            # 找不到 WALL 边界组时退回特征长度估计——这是物理上合理的
            # 近似（不是任意常数），仍打印 WARNING 提示精度下降，不属于
            # 本次修复目标（"假常量" 0.01m）范畴，保留原行为。
            volumes = self.mesh_data.get('cell_volumes')
            if volumes is None:
                volumes = cp.asarray(self.mesh.get_all_cell_volumes())
            h_char = volumes ** (1.0 / 3.0)
            self.wall_distance_gpu = cp.broadcast_to(
                h_char[:, None], (n_cells, n_sps)
            ).copy()
            logger.warning("Wall distance: using characteristic length as estimate")
