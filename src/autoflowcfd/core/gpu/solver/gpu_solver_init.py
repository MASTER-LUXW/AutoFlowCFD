"""GPUFRSolver 初始化方法混入类。

从 gpu_solver.py 拆出，控制单文件行数。包含面几何和壁面距离
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
        `gpu_distributed_init/geometry.py::_init_distributed_face_geometry` 在上
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

    def _init_wall_distance_gpu(self):
        """在当前阶数的解点上查询壁面距离并上传 GPU。

        来源与单机 CPU 同一个（`core/utils/wall_distance`，CLI 由体网格
        WALL 边界面构造后以构造参数 `wall_distance_source` 传入）；换阶时
        `gpu_solver_order_continuation` 再调一次，用同一个来源重查。

        **2026-09-25 修复**：此前这里自己遍历 `mesh.boundary_groups` 并把
        `BoundaryMap.groups` 的数组当字典读（真实网格上构造即崩溃），找不到壁面
        时静默退回"单元特征长度"。现在没有来源或没有解点坐标都直接报错
        （此前"坐标缺失时回退 0.01 m 常量"那次修复的同一原则）。
        """
        cp = get_cupy()
        source = getattr(self, "_wall_distance_source", None)
        if source is None:
            raise RuntimeError(
                f"湍流模型 '{self.turb_model_name}' 需要壁面距离，但 GPUFRSolver 没有收到"
                f"壁面距离来源（wall_distance_source）——不退化为特征长度估计。")
        sps = getattr(self.mesh, "sps_coords", None)
        if sps is None:
            raise RuntimeError("网格没有 sps_coords，无法在解点上查询壁面距离")
        dist = source.query(np.asarray(sps).reshape(self.mesh.n_cells, self.mesh.n_sps_per_cell, 3))
        self.wall_distance_gpu = cp.asarray(dist)
        logger.info(f"Wall distance ({source.n_wall_faces} wall triangles, exact) computed: min={dist.min():.6e}, max={dist.max():.6e}")
        model = getattr(self, "turb_model_gpu", None)
        if hasattr(model, "apply_wall_distance"):
            # SST 设 omega 上界、SA-neg 识别壁面解点（`TransportedTurbulence.apply_wall_distance`）
            model.apply_wall_distance(self.wall_distance_gpu, self.mu_molecular / self.freestream["rho_inf"])
