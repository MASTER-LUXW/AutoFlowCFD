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
