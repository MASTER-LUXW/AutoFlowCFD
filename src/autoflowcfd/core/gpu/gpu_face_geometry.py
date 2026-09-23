"""
AutoFlowCFD V2.0 - GPU 版面几何展平缓存

将 core/fr_face_kernels_flat.py 的 FlatFaceGeometry 所有 numpy 数组
上传为 CuPy 数组，供 GPU 界面校正 kernel 使用。

设计：
- 一次性上传所有面几何数据到 GPU，后续残差评估直接引用
- 面图着色信息直接复用（同色面无 owner_cell 冲突）
- 缓存与 mesh/order 绑定，Order Continuation 阶数切换时重新构建
"""

import numpy as np
from typing import Dict, Any, Optional
from loguru import logger

from autoflowcfd.core.gpu import get_cupy


class GPUFlatFaceGeometry:
    """GPU 版面几何缓存。

    将 FlatFaceGeometry 的所有数组上传到 GPU，供界面校正 kernel 使用。

    Attributes:
        n_faces: 总面数
        n_colors: 图着色颜色数
        owner_cell, neighbor_cell, is_boundary: 面连接关系（CuPy 数组）
        true_normal, area_w: 面法向和面积权重
        color_face_indices: 每个颜色的面子集索引（CuPy 数组列表）
        以及所有其他界面 kernel 需要的几何数据
    """

    def __init__(self, flat_face, device_id: int = 0):
        """从 CPU 版 FlatFaceGeometry 构建 GPU 版本。

        Args:
            flat_face: core/fr_face_kernels_flat.py 的 FlatFaceGeometry 实例
            device_id: GPU 设备 ID
        """
        cp = get_cupy()
        if cp is None:
            raise RuntimeError("CuPy is not available")

        with cp.cuda.Device(device_id):
            self.n_faces = flat_face.n_faces
            self.n_colors = flat_face.n_colors

            # ── 面连接关系 ──
            self.owner_cell = cp.asarray(flat_face.owner_cell)
            self.neighbor_cell = cp.asarray(flat_face.neighbor_cell)
            self.is_boundary = cp.asarray(flat_face.is_boundary)

            # ── 面角色信息 ──
            self.owner_axis = cp.asarray(flat_face.owner_axis)
            self.owner_side = cp.asarray(flat_face.owner_side)
            self.neighbor_axis = cp.asarray(flat_face.neighbor_axis)
            self.neighbor_side = cp.asarray(flat_face.neighbor_side)
            self.owner_is_primary = cp.asarray(flat_face.owner_is_primary)
            self.neighbor_is_primary = cp.asarray(flat_face.neighbor_is_primary)

            # ── 面法向 ──
            self.true_normal = cp.asarray(flat_face.true_normal)

            # ── 自洽方向 adj(J) 行（未归一化、未按 side 定向的原始值，
            # 语义与 CPU 端 owner_adj_row_exact/neighbor_adj_row_exact 一致，
            # 见 fr/face_flux_points/exact_normal.py 模块文档）——2026-08-23
            # 新增，供 GPU 粘性界面校正使用：CPU 端粘性 kernel
            # （viscous_flux_kernel.py）用这两行的原始 (a0,a1,a2) 分量直接
            # 把物理通量投影成逆变（tilde）形式，不是先归一化成单位法向
            # 再单独乘面积——GPU 无粘界面校正（_compute_interface_correction_
            # gpu）目前只用 true_normal（单位法向），这里为粘性项额外上传
            # 这两个字段，不改动无粘路径的既有行为。

            self.owner_adj_row_exact = cp.asarray(flat_face.owner_adj_row_exact)
            self.neighbor_adj_row_exact = cp.asarray(flat_face.neighbor_adj_row_exact)

            # ── 邻居源数据（src0 = 主要来源矩阵，src1 = 稀疏第二来源）──
            self.neighbor_src0_cell = cp.asarray(flat_face.neighbor_src0_cell)
            self.neighbor_src0_mat = cp.asarray(flat_face.neighbor_src0_mat)
            self.neighbor_src1_idx = cp.asarray(flat_face.neighbor_src1_idx)
            self.neighbor_src1_cell = cp.asarray(flat_face.neighbor_src1_cell)
            self.neighbor_src1_mat = cp.asarray(flat_face.neighbor_src1_mat)

            # ── Owner 源数据 ──
            self.owner_src0_cell = cp.asarray(flat_face.owner_src0_cell)
            self.owner_src0_mat = cp.asarray(flat_face.owner_src0_mat)
            self.owner_src1_idx = cp.asarray(flat_face.owner_src1_idx)
            self.owner_src1_cell = cp.asarray(flat_face.owner_src1_cell)
            self.owner_src1_mat = cp.asarray(flat_face.owner_src1_mat)

            # ── 混合拆分面（B-8，语义见 face_kernels.py::FlatFaceGeometry 同名字段文档）──
            self.mixed_nb_partner = cp.asarray(flat_face.mixed_nb_partner)
            self.mixed_nb_mask = cp.asarray(flat_face.mixed_nb_mask)
            self.mixed_ow_partner = cp.asarray(flat_face.mixed_ow_partner)
            self.mixed_ow_mask = cp.asarray(flat_face.mixed_ow_mask)
            self.mixed_bnd_face = cp.asarray(flat_face.mixed_bnd_face)
            self.mixed_p0_bnd_frac = cp.asarray(flat_face.mixed_p0_bnd_frac)

            # ── 原生基面算子 ──
            # `owner_cube_face`/`neighbor_cube_face` 是原始 cube face
            # 编码（四面体真实面 [6,10)、棱柱真实面 [10,15)），GPU 界面
            # kernel 按 `code - 6` 索引 `boundary_extrap_native`/
            # `lift_native`（含义与 CPU 端 inviscid_kernel.py 同名字段
            # 完全一致，见该文件模块文档）。
            self.owner_cube_face = cp.asarray(flat_face.owner_cube_face)
            self.neighbor_cube_face = cp.asarray(flat_face.neighbor_cube_face)
            self.true_area_weight = cp.asarray(flat_face.true_area_weight)
            # `face_area`/`cell_volume`：IP 罚项的长度尺度
            # `h_f = cell_volume[cell] / face_area[face]` 需要（见
            # `face_kernels.FlatFaceGeometry.cell_volume` 文档）。
            self.face_area = cp.asarray(flat_face.face_area)
            self.cell_volume = cp.asarray(flat_face.cell_volume)
            # 参考面求积权重（DG 提升算子用的**正确**权重，见
            # `core/fr_operators/face_kernels.py::FlatFaceGeometry.
            # ref_area_weight` 字段文档）。逐面相同，只有 (n_fp,)。
            self.ref_area_weight = cp.asarray(flat_face.ref_area_weight)
            self.boundary_extrap_native = cp.asarray(flat_face.boundary_extrap_native)
            self.lift_native = cp.asarray(flat_face.lift_native)

            # ── 分布式 local+halo 扩展索引空间逐位置棱柱/四面体类型
            # （#1，2026-08-28 新增）：只有 DistributedFlatFaceGeometry
            # （多 GPU 分布式路径）才有这个字段，单机 FlatFaceGeometry
            # 没有——getattr 默认 None，单机路径完全不受影响。见
            # distributed_flat_face.py::DistributedFlatFaceGeometry
            # 字段文档"棱柱/四面体判据"一节。
            compact_cell_type = getattr(flat_face, 'compact_cell_type', None)
            self.compact_cell_type = (
                cp.asarray(compact_cell_type) if compact_cell_type is not None else None
            )

            # ── 面图着色索引 ──
            self.color_face_indices = []
            for c in range(self.n_colors):
                indices = flat_face.color_face_indices[c]
                if len(indices) > 0:
                    self.color_face_indices.append(cp.asarray(indices))
                else:
                    self.color_face_indices.append(cp.array([], dtype=np.int32))

        logger.info(
            f"GPU face geometry uploaded: {self.n_faces} faces, "
            f"{self.n_colors} colors"
        )


def build_gpu_flat_face(flat_face, device_id: int = 0) -> GPUFlatFaceGeometry:
    """从 CPU FlatFaceGeometry 构建 GPU 版本（工厂函数）。

    Args:
        flat_face: CPU 版 FlatFaceGeometry
        device_id: GPU 设备 ID

    Returns:
        GPUFlatFaceGeometry 实例
    """
    return GPUFlatFaceGeometry(flat_face, device_id)
