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

        # 滤波档解析放在 try 之外，否则会被下面那个
        # `except Exception -> warning` 吞掉（显式请求得不到满足必须
        # 报错，见 fr_solver/filter.py::resolve_filter_mode）。
        from autoflowcfd.core.fr_solver.filter import resolve_filter_mode
        mode = resolve_filter_mode("gpu-single")

        # **2026-09-18：这里原本套着 `except Exception -> warning +
        # filter_func_gpu = None`，已删除。** 同一个文件的
        # `_init_face_geometry` 早就因为完全同一条理由删掉了宽 except
        # （见那边文档："必须让求解器初始化真正失败，而不是静默退化"）。
        #
        # 为什么必须一起删：本轮给门控接线新加的硬护栏**全部**落在这个
        # try 内部 —— `mesh.face_connectivity is None`、`n_prism` 越界、
        # `cell_is_prism` 形状不符、`cupyx` 缺 scatter_max（见
        # `bounds_sensor._scatter_minmax`）。每一条的注释都写着"不静默
        # 继续/不静默钳"，而实际执行路径是 except -> warning -> **完全
        # 无滤波**继续跑。那比它替换掉的 `project` 退档离默认档更远，
        # 而且只留一条 warning。`resolve_filter_mode` 承诺"四条后端全部
        # 接线、不接线就报错"，靠这个出口整条承诺就落空了。
        #
        # 滤波档是数值方案的一部分，构建失败必须让构造失败。
        filter_prism = self.ops.filter_prism
        filter_tet = self.ops.filter_tet
        if filter_prism is not None or filter_tet is not None:
            if mode == "sensor":
                self.filter_func_gpu = self._build_sensor_gated_filter_gpu(
                    n_cells, n_sps, n_prism, filter_prism, filter_tet)
            else:
                from autoflowcfd.core.gpu.gpu_modal_filter import (
                    build_gpu_filter_func,
                )
                self.filter_func_gpu = build_gpu_filter_func(
                    n_cells, n_sps, n_prism,
                    filter_prism, filter_tet,
                    device_id=self.device_id,
                )
            logger.debug(f"GPU modal filter initialized (mode={mode})")

    def _build_sensor_gated_filter_gpu(self, n_cells, n_sps, n_prism,
                                       filter_prism, filter_tet):
        """单 GPU 的**传感器门控**滤波回调（2026-09-18 接线）。

        `AFCFD_FILTER_MODE=sensor` 是 2026-09-17 定下的默认档，此前只有
        单机 CPU 接线，本后端默认退回 `project`（全局逐 RK stage 施加
        精确投影 = 功能上等于 legacy，P1 退化成 P0、壁面剪应力恒为零）。

        **不新写一份门控实现**：直接复用 CPU 那一个
        `build_sensor_gated_filter_func_arrays`。它连同两个判据内核
        （Persson-Peraire、BJ 越界）都已改成**数组模块无关**——传 CuPy
        矩阵进去，整条回调就走 CuPy 的同名函数（BJ 的邻域散射归约经
        `bounds_sensor._scatter_minmax` 分派到
        `cupyx.scatter_max/scatter_min`；施加步骤在设备上用
        「两个矩阵全场各算一遍再按掩码选」，理由见
        `gpu_modal_filter.py::filter_scalar_field_gated_gpu`）。
        一份实现服务四条后端，而不是各抄一份。

        单 GPU 的全局单元编号满足"棱柱在前"约定，所以用 `n_prism` 即可
        （分布式 local 排列不满足，走 `cell_is_prism`，见
        `gpu_distributed_init.py`）；也不需要 halo 扩展——没有分区边界。

        **没有 CuPy 时整条门控退回 numpy**（本机与 CI 的 numpy 替身
        端到端测试就走这条）：`build_sensor_gated_filter_func_arrays` 从
        滤波矩阵推断数组模块，传 numpy 进去它就是 numpy 路径，数值与
        CPU 单机一致。**不静默跳过门控** —— 那会让"替身测试通过"与
        "真实 GPU 上门控真的生效"脱钩，而这正是本文件此前那个
        `except Exception -> warning` 兜底造成的问题（它把所有硬护栏
        一起吞了，2026-09-18 删除）。完整论证见
        `core/gpu/device_context.py` 模块文档。
        """
        from autoflowcfd.core.fr_solver.filter import (
            build_sensor_gated_filter_func_arrays,
        )
        from autoflowcfd.core.fr_operators.bounds_sensor import (
            resolve_troubled_sensor,
        )
        from autoflowcfd.core.gpu.device_context import device_transfer

        _dev, _to_dev = device_transfer(self.device_id)

        sensor = resolve_troubled_sensor()
        conn = {}
        if sensor in ("bounds", "both"):
            fc = self.mesh.face_connectivity
            if fc is None:
                raise RuntimeError(
                    "AFCFD_TROUBLED_SENSOR=bounds/both 需要 "
                    "mesh.face_connectivity（BJ 判据要面邻居均值），"
                    "当前网格没有构建面连接")
            _n_faces = int(np.asarray(fc.owner_cell).size)
            # BJ 判据的两张边界表。不给它们，贴壁单元会被结构性误判、
            # 壁面剪应力被压掉 14 倍（见 `fr_solver/boundary.py::
            # make_bj_boundary_tables`，惰性求值的理由也在那里）。
            from autoflowcfd.core.fr_solver.boundary import (
                make_bj_boundary_tables,
            )
            _nrm = getattr(fc, "normal", None)

            with _dev:
                conn = dict(
                    owner_cell=_to_dev(fc.owner_cell),
                    neighbor_cell=_to_dev(fc.neighbor_cell),
                    is_boundary=_to_dev(
                        np.asarray(fc.is_boundary, dtype=bool)),
                    freestream=self.freestream,
                    bnd_tables=make_bj_boundary_tables(
                        lambda: getattr(self, "boundary_ghost_provider", None),
                        _n_faces,
                        None if _nrm is None else np.asarray(_nrm),
                        to_device=_to_dev),
                )
        order = int(getattr(self, "current_order", self.order))
        with _dev:
            fp = (filter_prism if hasattr(filter_prism, "device")
                  else _to_dev(filter_prism))
            ft = (filter_tet if hasattr(filter_tet, "device")
                  else _to_dev(filter_tet))
            # 顶点邻域模板（BJ 判据用）——与单机 CPU 路径同一个构造
            # 函数、同一个理由（面邻居在三维四面体上不能把本单元夹住，
            # 见 `fr_operators/vertex_stencil.py`）。单 GPU 是全局网格、
            # 无 halo，所以直接用全局模板即可，不需要重映射。
            vstencil = None
            if sensor in ("bounds", "both"):
                from autoflowcfd.core.fr_operators.vertex_stencil import (
                    build_vertex_stencil,
                )

                vstencil = build_vertex_stencil(self.mesh)
            return build_sensor_gated_filter_func_arrays(
                n_cells, n_sps, order, fp, ft, n_prism=n_prism,
                sensor=sensor, vertex_stencil=vstencil, **conn)

    def _init_wall_distance_gpu(self):
        """在当前阶数的解点上查询壁面距离并上传 GPU。

        来源与单机 CPU 同一个（`core/utils/wall_distance_source.py`，CLI 由体网格
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
        logger.info(f"Wall distance ({source.kind}) computed: min={dist.min():.6e}, max={dist.max():.6e}")
