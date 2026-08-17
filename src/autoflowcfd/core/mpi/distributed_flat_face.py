"""
AutoFlowCFD V2.0 - 分布式面几何

将 FlatFaceGeometry 改造为分布式版本：每个 rank 只持有 owner 为 local cell 的面。
面分类为 interior / partition_boundary / physical_boundary。

关键设计:
- 不重排全局面序（保持与原始面序一致，满足退化 Jacobian 敏感性约束）
- 使用 mask 索引分类面（interior_mask, partition_boundary_mask 等）
- partition_boundary 面的 neighbor 数据来自 halo cell（通过 halo 交换获取）
- 面几何数组的索引空间从全局 cell 转为 local+halo 扩展索引

扩展索引约定:
- [0, n_local_cells): local cells
- [n_local_cells, n_total_cells): halo cells
- FlatFaceGeometry 中的 owner_cell/neighbor_cell 使用扩展索引
"""

import numpy as np
from dataclasses import dataclass
from typing import Optional

from autoflowcfd.core.fr_operators.face_kernels import FlatFaceGeometry, get_flat_face_geometry
from autoflowcfd.core.mpi.partition import DistributedPartition


@dataclass
class DistributedFlatFaceGeometry:
    """分布式面几何。

    在 FlatFaceGeometry 基础上增加：
    - 面分类 mask
    - cell 索引从全局转为 local+halo 扩展索引
    - neighbor cell 对 halo cell 的索引映射

    Attributes:
        base_flat: 原始 FlatFaceGeometry（只包含本 rank 负责的面）
        partition: 分区信息
        owner_cell_local: (n_local_faces,) owner cell 的扩展局部索引
        neighbor_cell_local: (n_local_faces,) neighbor cell 的扩展局部索引
            （边界面为 -1，halo cell 映射到 [n_local, n_total) 范围）
        interior_mask: (n_local_faces,) bool
        partition_boundary_mask: (n_local_faces,) bool
        physical_boundary_mask: (n_local_faces,) bool
    """
    base_flat: FlatFaceGeometry
    partition: DistributedPartition
    owner_cell_local: np.ndarray
    neighbor_cell_local: np.ndarray
    interior_mask: np.ndarray
    partition_boundary_mask: np.ndarray
    physical_boundary_mask: np.ndarray

    @property
    def n_faces(self) -> int:
        return self.base_flat.n_faces

    @property
    def n_fp(self) -> int:
        return self.base_flat.n_fp

    @property
    def n_sps(self) -> int:
        return self.base_flat.n_sps

    # 代理属性：直接转发到 base_flat
    @property
    def n_prism(self) -> int:
        return self.base_flat.n_prism

    @property
    def owner_axis(self) -> np.ndarray:
        return self.base_flat.owner_axis

    @property
    def owner_side(self) -> np.ndarray:
        return self.base_flat.owner_side

    @property
    def neighbor_axis(self) -> np.ndarray:
        return self.base_flat.neighbor_axis

    @property
    def neighbor_side(self) -> np.ndarray:
        return self.base_flat.neighbor_side

    @property
    def owner_is_primary(self) -> np.ndarray:
        return self.base_flat.owner_is_primary

    @property
    def neighbor_is_primary(self) -> np.ndarray:
        return self.base_flat.neighbor_is_primary

    @property
    def true_normal(self) -> np.ndarray:
        return self.base_flat.true_normal

    @property
    def is_boundary(self) -> np.ndarray:
        return self.base_flat.is_boundary

    @property
    def neighbor_src0_cell(self) -> np.ndarray:
        return self.base_flat.neighbor_src0_cell

    @property
    def neighbor_src0_mat(self) -> np.ndarray:
        return self.base_flat.neighbor_src0_mat

    @property
    def neighbor_src1_idx(self) -> np.ndarray:
        return self.base_flat.neighbor_src1_idx

    @property
    def neighbor_src1_cell(self) -> np.ndarray:
        return self.base_flat.neighbor_src1_cell

    @property
    def neighbor_src1_mat(self) -> np.ndarray:
        return self.base_flat.neighbor_src1_mat

    @property
    def owner_src0_cell(self) -> np.ndarray:
        return self.base_flat.owner_src0_cell

    @property
    def owner_src0_mat(self) -> np.ndarray:
        return self.base_flat.owner_src0_mat

    @property
    def owner_src1_idx(self) -> np.ndarray:
        return self.base_flat.owner_src1_idx

    @property
    def owner_src1_cell(self) -> np.ndarray:
        return self.base_flat.owner_src1_cell

    @property
    def owner_src1_mat(self) -> np.ndarray:
        return self.base_flat.owner_src1_mat

    @property
    def boundary_extrap(self) -> np.ndarray:
        return self.base_flat.boundary_extrap

    @property
    def g_left(self) -> np.ndarray:
        return self.base_flat.g_left

    @property
    def g_right(self) -> np.ndarray:
        return self.base_flat.g_right

    @property
    def dist_fp_of_sp(self) -> np.ndarray:
        return self.base_flat.dist_fp_of_sp

    @property
    def dist_axis_coord_of_sp(self) -> np.ndarray:
        return self.base_flat.dist_axis_coord_of_sp


def build_distributed_flat_face(
    mesh, ops, partition: DistributedPartition
) -> DistributedFlatFaceGeometry:
    """构建分布式面几何。

    从全局 FlatFaceGeometry 中提取本 rank 负责的面，将 cell 索引
    从全局转为 local+halo 扩展索引。

    Args:
        mesh: HighOrderMesh
        ops: FROperators
        partition: 分区信息

    Returns:
        DistributedFlatFaceGeometry
    """
    # 获取全局面几何
    global_flat = get_flat_face_geometry(mesh, ops)

    # 提取本 rank 负责的面
    local_face_indices = partition.local_faces
    n_local_faces = len(local_face_indices)

    # 构建全局→扩展局部映射
    # local cells: [0, n_local_cells)
    # halo cells: [n_local_cells, n_total_cells)
    n_local_cells = partition.n_local_cells
    g2l = partition.global_to_local.copy()

    # 为 halo cells 分配扩展索引
    halo_g2l = {}
    for i, hc in enumerate(partition.halo_cells):
        halo_g2l[int(hc)] = n_local_cells + i

    # 转换 owner/neighbor cell 索引
    owner_cell_local = np.full(n_local_faces, -1, dtype=np.int64)
    neighbor_cell_local = np.full(n_local_faces, -1, dtype=np.int64)

    for i, f in enumerate(local_face_indices):
        oc = int(global_flat.owner_cell[f])
        loc = g2l[oc]
        if loc >= 0:
            owner_cell_local[i] = loc
        elif oc in halo_g2l:
            owner_cell_local[i] = halo_g2l[oc]

        nc = int(global_flat.neighbor_cell[f])
        if nc < 0:
            neighbor_cell_local[i] = -1  # 边界面
        else:
            loc_n = g2l[nc]
            if loc_n >= 0:
                neighbor_cell_local[i] = loc_n
            elif nc in halo_g2l:
                neighbor_cell_local[i] = halo_g2l[nc]

    # 面分类 mask（从 partition 的 face_classification 获取）
    fc = partition.face_classification
    interior_mask = fc.interior_mask
    partition_boundary_mask = fc.partition_boundary_mask
    physical_boundary_mask = fc.physical_boundary_mask

    # 创建新的 FlatFaceGeometry（只包含本 rank 的面）
    # 从全局 flat 中提取子集
    sub_flat = FlatFaceGeometry(
        n_faces=n_local_faces,
        n_fp=global_flat.n_fp,
        n_sps=global_flat.n_sps,
        n_prism=global_flat.n_prism,
        owner_cell=owner_cell_local,
        neighbor_cell=neighbor_cell_local,
        is_boundary=global_flat.is_boundary[local_face_indices],
        owner_axis=global_flat.owner_axis[local_face_indices],
        owner_side=global_flat.owner_side[local_face_indices],
        neighbor_axis=global_flat.neighbor_axis[local_face_indices],
        neighbor_side=global_flat.neighbor_side[local_face_indices],
        owner_is_primary=global_flat.owner_is_primary[local_face_indices],
        neighbor_is_primary=global_flat.neighbor_is_primary[local_face_indices],
        true_normal=global_flat.true_normal[local_face_indices],
        neighbor_src0_cell=global_flat.neighbor_src0_cell[local_face_indices],
        neighbor_src0_mat=global_flat.neighbor_src0_mat[local_face_indices],
        neighbor_src1_idx=global_flat.neighbor_src1_idx[local_face_indices],
        neighbor_src1_cell=global_flat.neighbor_src1_cell,  # 紧凑数组，不变
        neighbor_src1_mat=global_flat.neighbor_src1_mat,
        owner_src0_cell=global_flat.owner_src0_cell[local_face_indices],
        owner_src0_mat=global_flat.owner_src0_mat[local_face_indices],
        owner_src1_idx=global_flat.owner_src1_idx[local_face_indices],
        owner_src1_cell=global_flat.owner_src1_cell,
        owner_src1_mat=global_flat.owner_src1_mat,
        boundary_extrap=global_flat.boundary_extrap,
        g_left=global_flat.g_left,
        g_right=global_flat.g_right,
        n1d=global_flat.n1d,
        dist_fp_of_sp=global_flat.dist_fp_of_sp,
        dist_axis_coord_of_sp=global_flat.dist_axis_coord_of_sp,
    )

    return DistributedFlatFaceGeometry(
        base_flat=sub_flat,
        partition=partition,
        owner_cell_local=owner_cell_local,
        neighbor_cell_local=neighbor_cell_local,
        interior_mask=interior_mask,
        partition_boundary_mask=partition_boundary_mask,
        physical_boundary_mask=physical_boundary_mask,
    )
