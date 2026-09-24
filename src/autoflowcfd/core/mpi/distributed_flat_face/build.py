"""AutoFlowCFD V2.0 - 从局部网格 + 分区构造分布式扁平面几何

从 `src/autoflowcfd/core/mpi/distributed_flat_face.py`(原 552 行)拆出(2026-09-24, 项目"单文件不超 500 行"规范)。**纯搬家, 逻辑未改**。
"""

import numpy as np


from typing import Optional

from autoflowcfd.core.fr_operators.face_kernels import FlatFaceGeometry, get_flat_face_geometry

from autoflowcfd.core.mpi.partition import DistributedPartition
from .types import DistributedFlatFaceGeometry, _expand_compact_src1


def build_distributed_flat_face(
    mesh, ops, partition: DistributedPartition, cell_partition: Optional[np.ndarray] = None
) -> DistributedFlatFaceGeometry:
    """构建分布式面几何。

    从全局 FlatFaceGeometry 中提取本 rank 负责的面，将 cell 索引
    从全局转为 local+halo 扩展索引。

    Args:
        mesh: HighOrderMesh
        ops: FROperators
        partition: 分区信息（若提供 cell_partition，本函数可能原地扩展
            其 halo_cells/send_lists/recv_lists，见下）
        cell_partition: (n_global_cells,) 每个单元的分区编号，可选。
            提供时会在提取本 rank 面之前，先检测 FR Flux Point 多源
            交叉插值（src0/src1）依赖的单元是否已被基础 1-ring halo
            覆盖，缺失的自动扩展进 halo（见
            partition.py::extend_halo_for_flux_point_cross_references
            的完整文档）。不提供时（例如调用方没有 cell_partition 可用）
            仍会在缺失时于下方重映射阶段报错，而不是静默产生 P>=1
            阶数下读错邻居数据这类比崩溃更隐蔽的错误——第四次评审
            用真实 cube_demo 网格验证过：4-rank 简单 block 分区下，
            16%~19% 的单元存在这类未被基础 halo 覆盖的额外依赖，必须
            扩展才能让 --n-ranks/--multi-gpu 真正跑通。

    Returns:
        DistributedFlatFaceGeometry
    """
    # 获取全局面几何
    global_flat = get_flat_face_geometry(mesh, ops)

    if cell_partition is not None:
        from autoflowcfd.core.mpi.partition import extend_halo_for_flux_point_cross_references

        extra_dep_arrays = [
            global_flat.neighbor_src0_cell,
            _expand_compact_src1(global_flat.neighbor_src1_idx, global_flat.neighbor_src1_cell),
            global_flat.owner_src0_cell,
            _expand_compact_src1(global_flat.owner_src1_idx, global_flat.owner_src1_cell),
        ]
        extend_halo_for_flux_point_cross_references(
            partition, cell_partition, mesh.face_connectivity, extra_dep_arrays,
        )

    # 提取本 rank 负责的面
    local_face_indices = partition.local_faces
    n_local_faces = len(local_face_indices)

    # 构建全局→扩展局部映射。
    #
    # 真实 bug 修复（#1，V2.0 专家组盲审第4轮，2026-08-28）：此前这里
    # 沿用 partition.global_to_local 的排列——local cells 在前
    # [0,n_local_cells)（按 partition.local_cells 自身顺序），halo cells
    # 在后 [n_local_cells,n_total)（按 partition.halo_cells 自身顺序）。
    # 但本项目全局单元编号有一条贯穿全代码库的硬约定：棱柱恒占据
    # [0,n_prism)，四面体恒占据 [n_prism,n_cells)——GPU/CPU 残差 kernel
    # 的体积项（gpu_inviscid_volume.py::compute_volume_term_gpu、
    # gpu_viscous.py 的 div_G、gpu_gradients.py 的 compute_physical_
    # gradient_gpu 等）全都靠对这个扩展索引数组做`array[:n_prism]`/
    # `array[n_prism:]`**切片**（不是逐元素查表）来分别喂给棱柱/四面体
    # 专用算子——"先本地后halo"这个排列不满足"棱柱都在前、四面体都在后"，
    # 通常会把本应各自连续的棱柱/四面体拆成四段（本地棱柱、本地四面体、
    # halo棱柱、halo四面体），任何单一阈值切片都无法正确复原。
    #
    # 修复：扩展索引空间改按"棱柱在前、四面体在后"重新排列（与全局编号
    # 约定一致），而不是"local在前、halo在后"。halo 交换（gpu_halo_
    # exchange.py/halo.py）自身的 send/recv 协议仍然使用 partition 对象
    # 自己的 local_cells/halo_cells 原生顺序（那是更基础、被多处依赖的
    # 协议，本次不改动）——两个排列不一致是有意为之：本文件构造的
    # local+halo 扩展索引只在"调用残差计算函数之前/之后各做一次显式
    # 重排"这个局部范围内使用（见 gpu_distributed.py::compute_
    # inviscid_residual_gpu 对 perm/inv_perm 的消费），不影响 halo
    # 交换协议本身。
    n_local_cells = partition.n_local_cells
    halo_cells = partition.halo_cells
    n_halo = len(halo_cells)

    # native_global_ids：halo 交换协议原生排列下，扩展索引空间每个位置
    # 对应的全局单元编号（[0,n_local)=local_cells 自身顺序，
    # [n_local,n_local+n_halo)=halo_cells 自身顺序）。
    native_global_ids = (
        np.concatenate([partition.local_cells, halo_cells]) if n_halo > 0 else partition.local_cells.copy()
    )
    n_prism_global = getattr(mesh, 'n_prism_cells', 0)
    is_prism_native = native_global_ids < n_prism_global

    # perm[k] = k 号"棱柱在前"位置对应的 native_global_ids 下标；
    # inv_perm 是其逆置换（native 位置 -> 棱柱在前位置），供
    # gpu_distributed.py 在调用残差函数前后对 U_extended_gpu/残差数组
    # 做重排/逆重排使用。
    perm = np.concatenate([np.flatnonzero(is_prism_native), np.flatnonzero(~is_prism_native)])
    inv_perm = np.empty_like(perm)
    inv_perm[perm] = np.arange(len(perm), dtype=perm.dtype)

    compact_global_ids = native_global_ids[perm]  # "棱柱在前"顺序下每个扩展索引对应的全局单元编号
    n_prism_compact = int(np.sum(is_prism_native))

    # 全局单元索引 -> "棱柱在前"扩展索引空间的完整向量化映射表（覆盖
    # [0, n_global_cells)，真正既非 local 也非 halo 的单元位置保持 -1）。
    # 第四次评审修复（发现3）：owner_src0_cell/neighbor_src0_cell/
    # owner_src1_cell/neighbor_src1_cell 此前只按 local_face_indices 对
    # *面轴* 做了切片，cell 值本身仍是未重映射的全局单元索引——CPU
    # inviscid_kernel.py 的 `Q[c0, s, v]`/GPU gpu_inviscid.py 都直接把它
    # 当索引用，而此时 Q/Q_gpu 只有 n_local+n_halo 大小，会 IndexError
    # 崩溃，或更危险地"恰好"落在数组范围内、悄悄读到不相关 cell 的数据。
    cell_g2l_extended = np.full(len(partition.global_to_local), -1, dtype=np.int64)
    cell_g2l_extended[compact_global_ids] = np.arange(len(compact_global_ids), dtype=np.int64)

    def _remap_cell_indices(
        global_cells: np.ndarray, field_name: str, used_mask: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """把一批全局单元索引（-1 表示"无此来源"，原样保留）重映射到
        local+halo 扩展索引空间。真正映射失败（既非 local 也非 halo）
        说明 halo 层构建没有把这个face-flux-point 插值实际依赖的单元
        纳入——这是更上游（partition/halo 层构建）的缺陷，必须报错，
        不能静默截断/用错误索引兜底（那样会在 P>=1 阶数下产生"读到
        毫不相关 cell 数据"这类比崩溃更隐蔽的错误残差）。

        used_mask: 仅对 neighbor_src1_cell/owner_src1_cell 这类*紧凑*
            数组需要——这些数组是跨所有 rank 共享的全局紧凑表，本 rank
            的 kernel 只会通过本 rank local 面的 src1_idx 去访问其中一
            部分槽位，其余槽位属于其它 rank 的面，本来就不该、也不会
            在本 rank 的 local+halo 单元空间里能解析——不能因为这些
            "本来就不归本 rank 管"的槽位报错（那不是发现3描述的真实
            bug，而是没有排除无关槽位导致的误报）。为 None 时（本函数
            用于非紧凑、按面轴切片过的数组）视为全部位置都需要校验。
        """
        out = np.full_like(global_cells, -1)
        valid = global_cells >= 0
        if used_mask is not None:
            valid = valid & used_mask
        if np.any(valid):
            mapped = cell_g2l_extended[global_cells[valid]]
            bad = mapped < 0
            if np.any(bad):
                bad_global = np.unique(global_cells[valid][bad])
                raise RuntimeError(
                    f"build_distributed_flat_face: {field_name} 引用了 "
                    f"{len(bad_global)} 个既非 local 也非 halo 的单元（全局索引"
                    f"示例 {bad_global[:5].tolist()}）——halo 层构建没有把这些 "
                    f"face-flux-point 交叉插值实际依赖的单元纳入，必须先扩展 "
                    f"halo 层（core/mpi/partition.py），不能在这里静默丢弃或"
                    f"用错误索引兜底。"
                )
            out[valid] = mapped
        return out

    # 转换 owner/neighbor cell 索引（向量化，与上面 cell_g2l_extended 的
    # 语义完全一致：-1 表示边界面/无来源，不参与映射）
    owner_cell_local = _remap_cell_indices(global_flat.owner_cell[local_face_indices], "owner_cell")
    neighbor_cell_local = _remap_cell_indices(global_flat.neighbor_cell[local_face_indices], "neighbor_cell")

    # 面分类 mask（从 partition 的 face_classification 获取）
    fc = partition.face_classification
    interior_mask = fc.interior_mask
    partition_boundary_mask = fc.partition_boundary_mask
    physical_boundary_mask = fc.physical_boundary_mask

    # 全局面索引 -> 本 rank 局部面索引的映射（-1 表示该全局面不属于本 rank）。
    # 第四次评审修复：此前 FlatFaceGeometry 构造遗漏了 8 个必需字段
    # （owner_adj_row_exact/neighbor_adj_row_exact/mixed_nb_partner/
    # mixed_nb_mask/mixed_ow_partner/mixed_ow_mask/mixed_bnd_face/
    # mixed_p0_bnd_frac）+ 2 个面图着色字段（color_face_indices/
    # n_colors）——FlatFaceGeometry 是 @dataclass 且这 10 个字段都无
    # 默认值，任何 `mpirun --n-ranks`/`--multi-gpu` 调用第一步就会抛
    # TypeError。owner_adj_row_exact/neighbor_adj_row_exact/
    # mixed_nb_mask/mixed_ow_mask/mixed_bnd_face/mixed_p0_bnd_frac 是
    # 逐面（可能再加 n_fp 维）的物理量，直接按 local_face_indices 切片
    # 即可；mixed_nb_partner/mixed_ow_partner/color_face_indices 里存的
    # 是*另一个面的全局索引*，必须重映射到本 rank 的局部面编号。
    n_faces_global = global_flat.n_faces
    face_g2l = np.full(n_faces_global, -1, dtype=np.int64)
    face_g2l[local_face_indices] = np.arange(n_local_faces, dtype=np.int64)

    def _remap_mixed_partner(global_partner_for_local_faces: np.ndarray, field_name: str) -> np.ndarray:
        """把 mixed_{nb,ow}_partner（配对面的*全局*索引，-1 表示非混合面）
        重映射为本 rank 的局部面索引。

        混合配对（B-8）的两个半区（内部界面半区 + 边界半区）是同一个
        棱柱四边形侧面三角化拆分出的两条子面记录，共享同一个 owner
        单元——而 partition.local_faces 正是按 owner 单元是否为本 rank
        local cell 来筛选面的（见 core/mpi/partition.py），所以配对面
        必然与当前面同属一个 rank，不存在跨 rank 拆分的可能。据此，
        任何 partner>=0 但重映射失败（说明 partner 落在本 rank
        local_face_indices 之外）都必然是分区逻辑本身出了问题，必须
        报错，不能静默产生 -1（那会让下游把一个真实存在的混合配对
        误判为"非混合面"，悄悄丢弃边界罚项/幽灵态耦合）。
        """
        local_partner = np.full_like(global_partner_for_local_faces, -1)
        valid = global_partner_for_local_faces >= 0
        if np.any(valid):
            mapped = face_g2l[global_partner_for_local_faces[valid]]
            bad = mapped < 0
            if np.any(bad):
                bad_global = global_partner_for_local_faces[valid][bad]
                raise RuntimeError(
                    f"build_distributed_flat_face: {field_name} 引用的 "
                    f"{int(np.sum(bad))} 个配对面（全局索引示例 "
                    f"{bad_global[:5].tolist()}）不在本 rank 的 "
                    f"local_face_indices 中——混合配对(B-8)的两个半区应共享"
                    f"同一 owner 单元、必然同属一个 rank，这说明分区/面分类"
                    f"逻辑存在不一致，必须先修复，不能静默按非混合面处理。"
                )
            local_partner[valid] = mapped
        return local_partner

    global_flat_mixed_nb_partner = global_flat.mixed_nb_partner[local_face_indices]
    global_flat_mixed_ow_partner = global_flat.mixed_ow_partner[local_face_indices]
    mixed_nb_partner_local = _remap_mixed_partner(global_flat_mixed_nb_partner, "mixed_nb_partner")
    mixed_ow_partner_local = _remap_mixed_partner(global_flat_mixed_ow_partner, "mixed_ow_partner")

    # 面图着色（HPC-02）：color_face_indices 是按*全局*面索引分组的颜色
    # 列表；本 rank 只需要每个颜色里落在自己 local_face_indices 内的那
    # 部分，重映射为局部索引。不属于本 rank 的面被安静过滤掉——这是
    # 预期行为（那些面归其它 rank 所有），不同于上面 mixed partner 的
    # "必须同 rank"不变量。
    local_color_face_indices = []
    for global_color_faces in global_flat.color_face_indices:
        local_idx = face_g2l[global_color_faces]
        local_color_face_indices.append(local_idx[local_idx >= 0])

    # neighbor_src1_cell/owner_src1_cell 是跨所有 rank 共享的紧凑数组，
    # 本 rank 的 kernel 只会通过本 rank local 面的 src1_idx 访问其中一部分
    # 槽位——计算这部分"本 rank 实际会用到"的槽位掩码，传给
    # _remap_cell_indices 的 used_mask，避免把"压根不归本 rank 管"的槽位
    # 也当成本 rank 的依赖去校验（那样会对不属于本 rank 的引用产生误报，
    # 见该函数文档）。
    def _used_slot_mask(idx_local_faces: np.ndarray, compact_len: int) -> np.ndarray:
        mask = np.zeros(compact_len, dtype=bool)
        used = idx_local_faces[idx_local_faces >= 0]
        if used.size > 0:
            mask[used] = True
        return mask

    neighbor_src1_used_mask = _used_slot_mask(
        global_flat.neighbor_src1_idx[local_face_indices], len(global_flat.neighbor_src1_cell)
    )
    owner_src1_used_mask = _used_slot_mask(
        global_flat.owner_src1_idx[local_face_indices], len(global_flat.owner_src1_cell)
    )

    # 创建新的 FlatFaceGeometry（只包含本 rank 的面）
    # 从全局 flat 中提取子集
    sub_flat = FlatFaceGeometry(
        n_faces=n_local_faces,
        n_fp=global_flat.n_fp,
        n_sps=global_flat.n_sps,
        # n_prism 改用"棱柱在前"扩展索引空间下的棱柱计数（不是
        # global_flat.n_prism 那个全局计数），见上方 cell_g2l_extended
        # 构造处的完整说明——下游 gpu_inviscid_volume.py/gpu_viscous.py/
        # gpu_gradients.py 的体积项切片正是靠这个值分开棱柱/四面体。
        n_prism=n_prism_compact,
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
        owner_adj_row_exact=global_flat.owner_adj_row_exact[local_face_indices],
        neighbor_adj_row_exact=global_flat.neighbor_adj_row_exact[local_face_indices],
        # native 四面体（路径C）字段（2026-09-02 更新：此前这里的注释写
        # "分布式/MPI 路径明确不支持 native 模式"，已用端到端残差比对
        # 决定性验证证伪这个保守说法——见 tests/unit/
        # test_distributed_compute_residual.py::
        # TestDistributedResidualMatchesSingleMachineNative，分布式与
        # 单机路径残差逐位一致（rel_diff<6e-11）。原样透传本来就是
        # 正确、完整的处理方式，不需要额外分派：`owner_cube_face`/
        # `neighbor_cube_face` 按面索引正确切片（每个面各自的原始
        # cube face 编码，与是否 MPI 分区无关）；`boundary_extrap_native`/
        # `lift_native` 是只依赖 `(order, excluded_vertex)` 的全局共享
        # 常量算子（与其余面算子同一个"可预计算一次、全网格
        # 同阶数单元共享"的性质，见 native_tet.basis.py 模块文档），
        # 不是逐面数据，原样透传（不切片）本来就是唯一正确的做法。CPU
        # 端残差 kernel（`inviscid_kernel.py`/`viscous_flux_kernel.py`）
        # 本身的 native/collapsed 分派逻辑与调用方是否处于 MPI 分区
        # 场景无关，两者天然独立。
        owner_cube_face=global_flat.owner_cube_face[local_face_indices],
        neighbor_cube_face=global_flat.neighbor_cube_face[local_face_indices],
        true_area_weight=global_flat.true_area_weight[local_face_indices],
        # `ref_area_weight` 是**逐面相同**的参考求积权重（(n_fp,)），
        # 所以不按 local_face_indices 切、整块复用。
        ref_area_weight=global_flat.ref_area_weight,
        # `face_area` 是**逐面**物理面积 -> 按面轴切片。
        face_area=global_flat.face_area[local_face_indices],
        # `cell_volume` 存的是*单元*量，必须按 `compact_global_ids`
        # （扩展索引 -> 全局单元编号）重排到"棱柱在前"的 local+halo 索引
        # 空间，与 owner/neighbor cell 索引所在的空间一致 —— 与
        # `_remap_cell_indices` 处理 src0/src1 cell 字段同一个理由（那处
        # 漏做重映射曾是"P>=1 阶数分区边界读错邻居数据"的直接原因）。
        cell_volume=global_flat.cell_volume[compact_global_ids],
        boundary_extrap_native=global_flat.boundary_extrap_native,
        lift_native=global_flat.lift_native,
        # src0/src1 cell 字段存的是*单元*索引（不是面索引），必须重映射到
        # local+halo 扩展索引空间——此前只对 src0 做了面轴切片、完全没做
        # cell 值重映射，src1 的紧凑数组更是原样透传，是发现3描述的
        # P>=1 阶数分区边界读错邻居数据 bug 的直接原因。
        neighbor_src0_cell=_remap_cell_indices(
            global_flat.neighbor_src0_cell[local_face_indices], "neighbor_src0_cell"
        ),
        neighbor_src0_mat=global_flat.neighbor_src0_mat[local_face_indices],
        neighbor_src1_idx=global_flat.neighbor_src1_idx[local_face_indices],
        # 紧凑数组本身不按 local_face_indices 切片（由 neighbor_src1_idx
        # 索引，语义不变），但其中本 rank 实际会用到的 cell 值必须重映射
        # （用 used_mask 排除不属于本 rank 的槽位，见上方文档）。
        neighbor_src1_cell=_remap_cell_indices(
            global_flat.neighbor_src1_cell, "neighbor_src1_cell", used_mask=neighbor_src1_used_mask
        ),
        neighbor_src1_mat=global_flat.neighbor_src1_mat,
        owner_src0_cell=_remap_cell_indices(
            global_flat.owner_src0_cell[local_face_indices], "owner_src0_cell"
        ),
        owner_src0_mat=global_flat.owner_src0_mat[local_face_indices],
        owner_src1_idx=global_flat.owner_src1_idx[local_face_indices],
        owner_src1_cell=_remap_cell_indices(
            global_flat.owner_src1_cell, "owner_src1_cell", used_mask=owner_src1_used_mask
        ),
        owner_src1_mat=global_flat.owner_src1_mat,
        mixed_nb_partner=mixed_nb_partner_local,
        mixed_nb_mask=global_flat.mixed_nb_mask[local_face_indices],
        mixed_ow_partner=mixed_ow_partner_local,
        mixed_ow_mask=global_flat.mixed_ow_mask[local_face_indices],
        mixed_bnd_face=global_flat.mixed_bnd_face[local_face_indices],
        mixed_p0_bnd_frac=global_flat.mixed_p0_bnd_frac[local_face_indices],
        n1d=global_flat.n1d,
        color_face_indices=local_color_face_indices,
        n_colors=global_flat.n_colors,
    )

    # compact_cell_type：扩展索引空间（现已是"棱柱在前"排列，见上方
    # cell_g2l_extended 构造处）逐位置的真实棱柱/四面体类型，冗余于
    # `oc < n_prism_compact` 这个现在已经正确的单一阈值判据——两者
    # 应恒一致，保留这个查表数组是为了让 gpu_inviscid.py 等消费方在
    # "逐元素查表"和"切片"两种消费方式下都有一个显式、独立于位置排列
    # 假设的真值来源（一旦未来这里的排列约定又被改动，这个数组仍能
    # 从全局单元编号直接算出正确答案，不依赖调用方对排列约定的假设）。
    compact_cell_type = np.where(compact_global_ids < n_prism_global, 0, 1).astype(np.int8)

    return DistributedFlatFaceGeometry(
        base_flat=sub_flat,
        partition=partition,
        owner_cell_local=owner_cell_local,
        neighbor_cell_local=neighbor_cell_local,
        interior_mask=interior_mask,
        partition_boundary_mask=partition_boundary_mask,
        physical_boundary_mask=physical_boundary_mask,
        compact_cell_type=compact_cell_type,
        perm=perm,
        inv_perm=inv_perm,
        compact_global_ids=compact_global_ids,
    )
