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

            # ── 边界外插矩阵 ──
            self.boundary_extrap = cp.asarray(flat_face.boundary_extrap)

            # ── 校正函数导数（g_left, g_right）──
            self.g_left = cp.asarray(flat_face.g_left)
            self.g_right = cp.asarray(flat_face.g_right)

            # ── SP↔FP 映射 ──
            self.dist_fp_of_sp = cp.asarray(flat_face.dist_fp_of_sp)
            self.dist_axis_coord_of_sp = cp.asarray(flat_face.dist_axis_coord_of_sp)

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
