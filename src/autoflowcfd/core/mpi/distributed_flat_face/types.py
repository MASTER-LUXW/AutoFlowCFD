"""AutoFlowCFD V2.0 - 分布式扁平面几何的数据类与紧凑 src1 展开

从 `src/autoflowcfd/core/mpi/distributed_flat_face.py`(原 552 行)拆出(2026-09-24, 项目"单文件不超 500 行"规范)。**纯搬家, 逻辑未改**。
"""

import numpy as np

from dataclasses import dataclass


from autoflowcfd.core.fr_operators.face_kernels import FlatFaceGeometry

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
        compact_cell_type: (n_local+n_halo,) int8，0=棱柱/1=四面体，按
            本类采用的"棱柱在前"扩展索引排列（见下）——冗余于
            `oc < base_flat.n_prism` 这个判据，保留作为独立于排列假设
            的真值来源，见 build_distributed_flat_face 文档"compact_
            cell_type"一节。
        perm: (n_local+n_halo,) int64。halo 交换协议原生排列
            （[0,n_local)=partition.local_cells 自身顺序，
            [n_local,n_local+n_halo)=partition.halo_cells 自身顺序，
            与 GPUHaloExchange/HaloExchange 的 send/recv 协议一致）下标
            到本类实际采用的"棱柱在前、四面体在后"扩展索引排列的置换：
            `array_native[perm]` 得到按本类排列重排后的数组。owner_
            cell_local/neighbor_cell_local 等本类字段全部已经是"棱柱在前"
            排列，**不是**原生排列——消费方如果拿到的场数据（例如
            halo 交换直接产出的 U_extended）还是原生排列，必须先用
            `perm` 重排，残差算完后再用 `inv_perm` 换回原生排列，才能
            正确对齐 owner_cell_local 等索引，见
            core/gpu/gpu_distributed.py 的消费方式。
        inv_perm: (n_local+n_halo,) int64，perm 的逆置换：
            `array_native = array_permuted[inv_perm]`。
        compact_global_ids: (n_local+n_halo,) int64，"棱柱在前"扩展索引
            空间每个位置对应的全局单元编号——供调用方从全局网格几何
            （jacobians/cell_volumes 等）里按同一顺序抽取出 local+halo
            子集，见 core/gpu/gpu_distributed.py 里 mesh_data 的构造。
    """
    base_flat: FlatFaceGeometry
    partition: DistributedPartition
    owner_cell_local: np.ndarray
    neighbor_cell_local: np.ndarray
    interior_mask: np.ndarray
    partition_boundary_mask: np.ndarray
    physical_boundary_mask: np.ndarray
    compact_cell_type: np.ndarray = None
    perm: np.ndarray = None
    inv_perm: np.ndarray = None
    compact_global_ids: np.ndarray = None

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
    def ref_area_weight(self) -> np.ndarray:
        return self.base_flat.ref_area_weight

    @property
    def true_area_weight(self) -> np.ndarray:
        return self.base_flat.true_area_weight

    @property
    def face_area(self) -> np.ndarray:
        return self.base_flat.face_area

    @property
    def cell_volume(self) -> np.ndarray:
        return self.base_flat.cell_volume

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


def _expand_compact_src1(src1_idx: np.ndarray, src1_cell_compact: np.ndarray) -> np.ndarray:
    """把 src1_idx（per-face，-1=无第二来源）+ 紧凑 src1_cell 数组，展开成
    per-face 的单元全局索引数组（-1=无）,供 halo 扩展的依赖扫描使用。"""
    out = np.full(src1_idx.shape[0], -1, dtype=np.int64)
    valid = src1_idx >= 0
    if np.any(valid):
        out[valid] = src1_cell_compact[src1_idx[valid]]
    return out
