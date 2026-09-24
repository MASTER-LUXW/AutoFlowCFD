"""AutoFlowCFD V2.0 - 标量场的面外插与面校正分配。

从 `core/turbulence/transport.py`（原 1476 行）拆出（2026-09-24，项目
"单文件不超 500 行"规范）。**纯搬家，逻辑未改** —— 判据是
`ProjectFiles/V2.0/29` 记录的 SST 黄金轨迹逐位相同。

这一层是对流/扩散两条残差共用的基础设施：把 SPs 上的标量场外插到面
通量点、把面上算好的校正量按 DG 提升算子分配回体积节点。
"""

import os
import numpy as np
from typing import NamedTuple

import numba

from autoflowcfd.core.fr_operators.volume_contract import (
    contravariant_flux_from_metric,
)
from autoflowcfd.core.turbulence.transport_kernel import (
    extrapolate_scalar_to_faces_kernel,
    distribute_corrections_to_cells_kernel,
    distribute_corrections_to_cells_kernel_colored,
    _extrap_owner_scalar_to_faces,
)


def _extrapolate_scalar_to_faces(
    scalar_sps, flat, ops, mesh, wall_dirichlet_zero_face=None,
    wall_dirichlet_value_face=None, has_wall_dirichlet_value=None,
):
    """将 SPs 上的标量场外插到所有面的通量点（numba kernel 版本）。

    使用 turbulence_transport_kernel.py 的 numba 编译函数替代纯 Python 循环。
    owner 侧用 `boundary_extrap_native[code-6]` 矩阵，neighbor 侧用
    neighbor_sources 矩阵。

    Args:
        wall_dirichlet_zero_face: (n_faces,) bool，可选。对标记为 True 的
            WALL 边界面，ghost 值用 Dirichlet-zero 镜像（ghost=-owner）
            而不是默认的 Neumann（ghost=owner）——只在外插 k 场时由
            调用方传入，其余场（omega/rho/velocity/gamma_field 等）不传，
            退回原有 Neumann 默认，见 extrapolate_scalar_to_faces_kernel
            文档。
        wall_dirichlet_value_face: (n_faces, n_fp) float，可选。非零
            Dirichlet 目标值（例如 omega 壁面解析式），与
            `has_wall_dirichlet_value` 配对使用，见 kernel 文档。只在
            外插 omega 场时由调用方传入。
        has_wall_dirichlet_value: (n_faces,) bool，可选，标记哪些面要
            使用 `wall_dirichlet_value_face` 而不是 Neumann 默认——与
            `wall_dirichlet_zero_face` 互斥。

    Returns:
        phi_owner_fp: (n_faces, n_fp)
        phi_neighbor_fp: (n_faces, n_fp)
    """
    if wall_dirichlet_zero_face is None:
        wall_dirichlet_zero_face = np.zeros(flat.n_faces, dtype=np.bool_)
    if has_wall_dirichlet_value is None:
        has_wall_dirichlet_value = np.zeros(flat.n_faces, dtype=np.bool_)
    if wall_dirichlet_value_face is None:
        wall_dirichlet_value_face = np.zeros((flat.n_faces, flat.n_fp), dtype=np.float64)
    return extrapolate_scalar_to_faces_kernel(
        scalar_sps,
        flat.neighbor_src0_cell, flat.neighbor_src0_mat,
        flat.neighbor_src1_idx, flat.neighbor_src1_cell, flat.neighbor_src1_mat,
        flat.owner_cell, flat.n_faces, flat.n_fp, flat.n_sps,
        wall_dirichlet_zero_face,
        flat.mixed_nb_partner, flat.mixed_nb_mask,
        has_wall_dirichlet_value,
        wall_dirichlet_value_face,
        flat.owner_cube_face, flat.boundary_extrap_native,
    )


def _extrapolate_owner_only_to_faces(scalar_sps, flat):
    """将 SPs 标量场只外插到面的 owner 侧（不算 neighbor 侧）。

    真实内存修复（V2.0 专家组盲审第四次评审，2026-08-28，cube_demo
    79万单元/187万面生产网格 P2+DDES 首次真实 CLI 冒烟测试触发 OOM
    崩溃）：`compute_scalar_convection_residual` 此前对 rho/velocity 都调用
    `_extrapolate_scalar_to_faces`（同时算 owner+neighbor 两侧），但下面
    `mass_flux`/`rho_u_owner` 的计算只用了 owner 侧的 rho/velocity——
    neighbor 侧的 rho_neighbor_fp、以及每个速度分量的 vel_neighbor_fp
    （(n_faces,n_fp,3)，本项目 cube_demo 规模下单个约 400MB）算出来后
    从未被读取，纯粹的死计算+死内存占用。真正需要两侧的只有标量场
    本身（phi_owner_fp/phi_neighbor_fp，用于上风选择）。

    直接调用 `_extrap_owner_scalar_to_faces`（transport_kernel.py 已有的
    独立 owner-only kernel，`extrapolate_scalar_to_faces_kernel` 内部
    本来就是分别调用 owner/neighbor 两个子 kernel 再打包返回，这里只
    调用其中一半），physically 精确等价于 `_extrapolate_scalar_to_faces`
    返回值的第一个分量，不是近似。

    Returns:
        phi_owner_fp: (n_faces, n_fp)
    """
    return _extrap_owner_scalar_to_faces(
        scalar_sps,
        flat.owner_cell, flat.n_faces, flat.n_fp, flat.n_sps,
        flat.owner_cube_face, flat.boundary_extrap_native,
    )


class ScalarConvectionGeometry(NamedTuple):
    """k/omega 两次标量对流调用共享的、**与标量本身无关**的几何/流场量。

    性能优化（2026-09-13，用户反馈"每步耗时过长"后的真实剖析结论）：
    `compute_turbulence_transport_residual` 对 k 和 omega 各调一次
    `compute_scalar_convection_residual`，两次传入的 `rho`/`velocity`/
    `mesh`/`ops` 完全相同，于是下面这两个量此前被**逐字重复计算了两遍**：

    - `rho_u_tilde`：逆变质量通量 adj(J) @ (rho*u)，体积项用。原实现是
      `np.matmul(adj_j_chunk, (rho*vel*phi))`，其中 phi 在该点是标量、
      可以从度量乘法里提出来（`adj_j@(rho*u*phi) == phi*(adj_j@(rho*u))`），
      因此这份与 phi 无关的部分只需算一次。
    - `mass_flux`：面通量点上的 (rho*u)·n̂（owner 侧），界面上风项用。
      算它需要把 rho 和 3 个速度分量各外插一次到全部 187 万面×n_fp 个
      通量点（4 次 numba 外插 kernel + 一次归约），同样与 phi 无关。

    共享后这两块工作从每步 2 次降为 1 次。数值上：两次调用此前拿到的
    就是同一个数学量（同一段代码、同一份输入），共享只是不再重算，
    k/omega 各自的上风选择/跳变量计算完全不受影响。
    """
    rho_u_tilde: np.ndarray   # (n_cells, n_sps, 3)
    mass_flux: np.ndarray     # (n_faces, n_fp)


def precompute_scalar_convection_geometry(rho, velocity, mesh, ops, flat):
    """算出 k/omega 共享的 `ScalarConvectionGeometry`（见该类文档）。"""
    n_cells = mesh.n_cells
    n_sps = mesh.n_sps_per_cell
    det_jacs = mesh.jacobians["det_jacs"].reshape(n_cells, n_sps)
    inv_jacs = mesh.jacobians["inv_jacs"].reshape(n_cells, n_sps, 3, 3)

    rho_u = rho[:, :, None] * velocity                      # (n_cells,n_sps,3)
    rho_u_tilde = contravariant_flux_from_metric(
        det_jacs, inv_jacs, rho_u[..., None]
    )[..., 0]                                               # (n_cells,n_sps,3)
    del rho_u

    n_fp = flat.n_fp
    rho_owner_fp = _extrapolate_owner_only_to_faces(rho, flat)
    vel_owner_fp = np.empty((flat.n_faces, n_fp, 3))
    for d in range(3):
        vel_owner_fp[:, :, d] = _extrapolate_owner_only_to_faces(velocity[:, :, d], flat)
    mass_flux = np.sum(rho_owner_fp[..., None] * vel_owner_fp * flat.true_normal, axis=-1)
    del rho_owner_fp, vel_owner_fp

    return ScalarConvectionGeometry(rho_u_tilde=rho_u_tilde, mass_flux=mass_flux)


def _distribute_correction_to_cells(raw_jump_fp, flat, ops, mesh, raw_jump_fp_neighbor=None):
    """将面通量点上的**未加权**跳变量分配回 SPs 残差（numba kernel 版本）。

    使用 turbulence_transport_kernel.py 的 kernel 替代纯 Python for f in range(n_faces) 循环。
    默认使用图着色方案（同色面无冲突，直接写入共享 buffer），
    通过环境变量 AFCFD_USE_COLORING 可回退到 per-thread buffer 方案。

    native 四面体（路径C）真实 bug 修复（2026-08-30，见
    transport_kernel.py::distribute_corrections_to_cells_kernel 文档）：
    参数从"已经预乘 |adj_row| 面元幅值因子的 correction_fp"改为
    **未加权**的 `raw_jump_fp`（调用方 `compute_scalar_convection_
    residual`/`compute_scalar_diffusion_residual` 不再自己乘 `adj_mag`），
    加权方式（原生用 `true_area_weight`，即真实物理面积权重）与分配方式
    （DG 提升算子）都在 kernel 内部完成：跳变量在 Python 层保持"物理通量
    密度差"的原始形态传进去，由 kernel 按 `owner_cube_face`/
    `neighbor_cube_face` 索引 `lift_native` 完成加权与提升。已删除的坍缩
    分支当年用的是另一套（`|adj_row|` 加权 + 1D `_distribute_point`
    分配），两者的最终乘积其实是同一个量，只是 `|adj_row|` 由谁提供不同
    —— 完整对照见 `transport_kernel.py::_weighted_jump_native` 文档。

    Args:
        raw_jump_fp_neighbor: (n_faces, n_fp) 可选，neighbor 侧独立的
            未加权跳变量（真实 bug 修复，2026-09-12，见
            `compute_scalar_convection_residual` 模块文档"owner/neighbor
            跳变量不对称"一节完整推导）。默认为 None 时回退为
            `raw_jump_fp`（与此前行为一致）——`compute_scalar_diffusion_
            residual` 的跳变量本身按 BR1 公共梯度定义、对 owner/neighbor
            天然对称，不需要独立的 neighbor 侧跳变量，继续用这个默认值
            保持数值结果不变。

    Returns:
        correction_sps: (n_cells, n_sps)
    """
    n_cells = mesh.n_cells
    n_sps = flat.n_sps
    det_jacs = mesh.jacobians["det_jacs"].reshape(n_cells, n_sps)
    if raw_jump_fp_neighbor is None:
        raw_jump_fp_neighbor = raw_jump_fp

    use_coloring = os.environ.get("AFCFD_USE_COLORING", "1") == "1"

    if use_coloring:
        correction_sps = np.zeros((n_cells, n_sps))
        for c in range(flat.n_colors):
            face_indices = flat.color_face_indices[c]
            if len(face_indices) == 0:
                continue
            distribute_corrections_to_cells_kernel_colored(
                raw_jump_fp,
                flat.owner_cell, flat.neighbor_cell,
                det_jacs,
                n_cells, n_sps,
                face_indices,
                correction_sps,
                flat.owner_cube_face, flat.neighbor_cube_face,
                flat.owner_adj_row_exact, flat.neighbor_adj_row_exact,
                flat.true_area_weight,
                flat.lift_native,
                flat.owner_is_primary, flat.neighbor_is_primary,
                raw_jump_fp_neighbor,
            )
        return correction_sps
    else:
        n_threads = numba.get_num_threads()
        return distribute_corrections_to_cells_kernel(
            raw_jump_fp,
            flat.owner_cell, flat.neighbor_cell,
            det_jacs,
            n_cells, n_sps, flat.n_faces,
            n_threads,
            flat.owner_cube_face, flat.neighbor_cube_face,
            flat.owner_adj_row_exact, flat.neighbor_adj_row_exact,
            flat.true_area_weight,
            flat.lift_native,
            flat.owner_is_primary, flat.neighbor_is_primary,
            raw_jump_fp_neighbor,
        )
