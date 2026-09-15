"""
AutoFlowCFD V2.0 - 湍流标量输运方程 FR 残差（完整 SST k-omega 输运）

为 SST k-omega 湍流模型补全对流和扩散输运项，使 k/omega 不再仅是逐点
ODE 源项弛豫，而是通过 FR 高阶离散真正参与空间输运。

输运方程:
    d(rho*k)/dt + div(rho*U*k) = S_k + div(Gamma_k * grad(k))
    d(rho*omega)/dt + div(rho*U*omega) = S_omega + div(Gamma_omega * grad(omega))

其中 Gamma_k = mu + sigma_k * rho * nu_t, Gamma_omega = mu + sigma_omega * rho * nu_t。

离散方法:
    - 对流项：FR 体积项（逆变标量通量散度）+ 界面上风通量校正
    - 扩散项：FR 体积项（逆变扩散通量散度）+ BR1 界面平均通量校正
    - 界面校正分配与 fr_residual_inviscid.py 使用相同的 g'/dist 映射

符号约定:
    残差 = 对流残差 + 扩散残差，返回值直接作为 dphi/dt 被调用方相加
    （fr_solver/turbulence.py::update_fields: k += dt*(Sk + transport_k)），
    与平均流残差的 RHS 约定一致（step.py: dU/dt = inv_res + visc_res）:
    - 对流残差 = -div(rho*U*phi)/det(J)（含界面上风校正）
    - 扩散残差 = +div(Gamma*grad(phi))/det(J)（含 BR1 界面校正）——
      与 viscous_flux.py 的"粘性项是 +div(G)"完全同一约定。此前这里误写为
      -div(Gamma*grad(phi))（反扩散），指数放大 2Δx 棋盘模态，把 k/omega 场
      两极分化到正性限制器的上下界（真实复现：cube_demo 全新计算 100 步内
      54% 单元贴下界 1e-12、38% 贴上界，之后冻结、残差停滞），2026-08-25 修复。
    更新: phi += dt * (transport_residual + source/rho)
"""

import os
import numpy as np
from typing import NamedTuple, Optional, Tuple

import numba

from autoflowcfd.core.fr_operators.gradients import compute_physical_scalar_gradient, compute_physical_gradient
from autoflowcfd.core.fr_operators.volume_contract import (
    OVERINT_CHUNK_CELLS, contract_shared_operator_1axis,
    contract_shared_operator_2axis, contravariant_flux_from_metric,
    get_overintegration_context,
)
from autoflowcfd.core.fr_operators.face_kernels import get_flat_face_geometry
from autoflowcfd.core.fr_operators.troubled_cell import suppress_residual_outliers
from autoflowcfd.core.turbulence.transport_kernel import (
    extrapolate_scalar_to_faces_kernel,
    distribute_corrections_to_cells_kernel,
    distribute_corrections_to_cells_kernel_colored,
    _extrap_owner_scalar_to_faces,
    scalar_convection_volume_kernel,
    scalar_volume_divergence_kernel,
)
from autoflowcfd.core.fr_operators.volume_contract import contravariant_flux_from_metric


def _extrapolate_scalar_to_faces(
    scalar_sps, flat, ops, mesh, wall_dirichlet_zero_face=None,
    wall_dirichlet_value_face=None, has_wall_dirichlet_value=None,
):
    """将 SPs 上的标量场外插到所有面的通量点（numba kernel 版本）。

    使用 turbulence_transport_kernel.py 的 numba 编译函数替代纯 Python 循环。
    owner 侧用 boundary_extrap 矩阵，neighbor 侧用 neighbor_sources 矩阵。

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
        scalar_sps, flat.boundary_extrap,
        flat.neighbor_src0_cell, flat.neighbor_src0_mat,
        flat.neighbor_src1_idx, flat.neighbor_src1_cell, flat.neighbor_src1_mat,
        flat.owner_cell, flat.owner_axis, flat.owner_side,
        flat.n_prism, flat.n_faces, flat.n_fp, flat.n_sps,
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
        scalar_sps, flat.boundary_extrap,
        flat.owner_cell, flat.owner_axis, flat.owner_side,
        flat.n_prism, flat.n_faces, flat.n_fp, flat.n_sps,
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
    加权方式（collapsed 用 `|adj_row|`、native 用 `true_area_weight`）
    与分配方式（collapsed 用 1D `_distribute_point`、native 用 DG 提升
    算子）都下沉到 kernel 内部按 `owner_cube_face`/`neighbor_cube_face`
    分派——原因：native 面需要的加权量（真实物理面积权重）与 collapsed
    面（度量张量 adj 行模长）不是同一个量，不能在 Python 层统一预乘
    后再传给一个"只认 collapsed 分配方式"的 kernel。

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
                flat.owner_axis, flat.owner_side,
                flat.neighbor_axis, flat.neighbor_side,
                det_jacs,
                flat.g_left, flat.g_right,
                flat.dist_fp_of_sp, flat.dist_axis_coord_of_sp,
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
            flat.owner_axis, flat.owner_side,
            flat.neighbor_axis, flat.neighbor_side,
            det_jacs,
            flat.g_left, flat.g_right,
            flat.dist_fp_of_sp, flat.dist_axis_coord_of_sp,
            n_cells, n_sps, flat.n_faces,
            n_threads,
            flat.owner_cube_face, flat.neighbor_cube_face,
            flat.owner_adj_row_exact, flat.neighbor_adj_row_exact,
            flat.true_area_weight,
            flat.lift_native,
            flat.owner_is_primary, flat.neighbor_is_primary,
            raw_jump_fp_neighbor,
        )


#: k/omega 输运体积项是否走过积分（去混叠），由 `AFCFD_TURB_OVERINT` 选择：
#:   "off"（默认，与此前行为逐位一致）
#:   "on" —— 对流/扩散体积项都在 FINE 点上算完再限制回 coarse
#:
#: **为什么需要它（2026-09-15）**：`fr/collapsed_basis.py::
#: build_overintegration_operators` 文档记录了去混叠的动机，并给出实测
#: 数字——对解析残差恒为 0 的线性剪切场，不去混叠的 P2 体积项算出的残差
#: 是真值的 43~62 倍。那套机制一直**只接在平均流的无粘体积项**上
#: （`fr_residual/inviscid.py` 的 `mesh.jacobians_fine is not None` 分支），
#: 粘性项与本模块的 k/omega 输运项**完全没有**。
#:
#: 而 k/omega 对流体积项算的是 `div(adj(J)·rho·u·phi)`——一个**三重**
#: 非线性乘积，真实多项式次数约 `3*order + 度量次数`，远高于 order。
#: 直接在 coarse SPs 上对它做微分就是"先混叠再求导"。
#:
#: 这与 `filter_scalar_field` 文档记录的 2026-09-12 真实 P1 发散症状
#: 直接对应：那次的诊断原文是"P1 多项式在单元内部相邻解点间出现数量级
#: 跳变 -> 外插到面通量点后被上风格式放大成巨大的虚假对流残差"，正是
#: 对流项混叠的形态。当时的处置是给 k/omega 加模态滤波器，而那个滤波器
#: 每个 RK stage 清掉一整阶（见 `fr/modal_filter.py`）——用牺牲阶数换
#: 稳定。去混叠是同一个问题的**不牺牲阶数**的正解。
#:
#: **默认已于 2026-09-15 改为 "on"**，三条证据齐备后才改的：
#:
#: 1. **解析判据**：`rho`/`u` 取常数、`phi` 取线性（`rho*u*phi` 只有一次、
#:    完全落在 P1 解空间内，任何非零误差都只能来自离散算子本身），
#:    `d(rho*phi)/dt = -rho*(u·G)` 有闭式解。走公开接口
#:    `compute_scalar_convection_residual` 实测：
#:
#:        默认 off: order=1 棱柱相对误差 **1.1294（113%）**
#:                  （残差范围 [-12.26, +1.11]，解析值 -8.5750）
#:                  order=1 四面体 6.84e-15（机器零）
#:        打开 on : order=1 棱柱 **4.61e-10**（改善约 2.4e9 倍）
#:                  order=2 两档逐位相同（2.2597e-09 / 2.2576e-09）
#:
#:    棱柱正是边界层单元、k/omega 最要紧的地方；而 order=2 在两档下都
#:    已足够精确，说明那个 113% 不是"这套离散本来就这么差"，是 order=1
#:    特有的混叠（非仿射棱柱上 adj(J) 是非平凡多项式，`adj(J)*rho*u*phi`
#:    真实次数高于 1；四面体在该网格上仿射、adj(J) 常数，所以差 14 个
#:    数量级）。边界条件不是原因：order=1 与 order=2 的边界面数完全相同。
#: 2. **代价已量化**：微基准（5 万单元、3 线程、V=1）单次标量链路
#:    85.5ms，按 79 万单元线性外推 1.354s；每步 4 次（k/omega 对流 +
#:    k/omega 扩散）约 5.42s，相对实测 ~53s/步约 **+10.2%**。
#: 3. **真实网格已验证**：79 万单元 cube_demo 的两个 250 步运行
#:    （`true_cfl03`/`true_adaptive`，零阶数损失 + 逐点 omega 下限）本来
#:    就是带 `TURB_OVERINT=on` 跑的，已单调收敛 137+ 步、om_max 全程不动。
#:
#: +10% 的单步代价换掉生产阶数上 113% 的残差误差——这个权衡不需要再等。
#: `off` 保留为受控 A/B 与回归复现的入口。
def resolve_turb_overintegration() -> str:
    """返回 k/omega 输运体积项的去混叠开关，并校验取值。

    每次调用都重读环境变量（不缓存模块级常量）：这一维不影响算子构造
    （过积分三件套与 `jacobians_fine` 在 `order>=1` 时本来就无条件构造
    好了，见 `fr/operators.py` 与 `grid/high_order/high_order_mesh_order.py`），
    运行期读取是安全的，也让测试可以直接 monkeypatch 环境变量。
    """
    v = os.environ.get("AFCFD_TURB_OVERINT", "on").lower()
    if v not in ("off", "on"):
        raise ValueError(
            f"AFCFD_TURB_OVERINT={v!r} 不是合法取值（off | on）。"
            f"'on'（默认）把对流/扩散体积项改走 FINE 点去混叠，"
            f"'off' 是 2026-09-15 之前的行为（直接在 coarse SPs 上微分，"
            f"order=1 棱柱有 113% 相对误差），保留作受控 A/B 入口。")
    return v


#: `get_overintegration_context` 的旧名别名：三个消费方（无粘体积项、
#: 本模块的 k/omega 输运、粘性体积项）共用同一份实现，见
#: `fr_operators/volume_contract.py::get_overintegration_context`。
_turb_overint_ops = get_overintegration_context

_TURB_OVERINT_CHUNK_CELLS = OVERINT_CHUNK_CELLS


def _scalar_convection_volume_overintegrated(
    scalar_field, rho, velocity, oi, n_sps,
):
    """对流体积项 `div(adj(J)*rho*u*phi)` 的去混叠版，返回 (n_cells,n_sps)。

    链路与 `fr_residual/inviscid.py` 的过积分分支逐项对应：
      ① phi/rho/u 各自精确插值到 FINE 点（它们各自次数 <= order，
         插值不引入误差——**乘积必须在 FINE 点上做**，这正是去混叠的
         全部内容：先插值再相乘，而不是先相乘再插值）；
      ② 用解析精确的 FINE 点度量算逆变通量；
      ③ 用 FINE 网格自己的微分矩阵求散度；
      ④ 精确插值限制回 coarse SPs。
    """
    n_cells = scalar_field.shape[0]
    n_fine = oi["n_fine"]
    det_fine, inv_fine = oi["det_fine"], oi["inv_fine"]
    div_F = np.empty((n_cells, n_sps))
    for seg_lo, seg_hi, op_c2f, op_D_fine, op_f2c in oi["segs"]:
        for c0 in range(seg_lo, seg_hi, _TURB_OVERINT_CHUNK_CELLS):
            c1 = min(c0 + _TURB_OVERINT_CHUNK_CELLS, seg_hi)
            phi_f = contract_shared_operator_1axis(
                op_c2f, np.ascontiguousarray(scalar_field[c0:c1, :, None]))
            rho_f = contract_shared_operator_1axis(
                op_c2f, np.ascontiguousarray(rho[c0:c1, :, None]))
            u_f = contract_shared_operator_1axis(
                op_c2f, np.ascontiguousarray(velocity[c0:c1]))       # (n,n_fine,3)
            # rho*u*phi 在 FINE 点上相乘（去混叠的核心）
            F_phys_f = (rho_f * phi_f * u_f)[..., None]              # (n,n_fine,3,1)
            del phi_f, rho_f, u_f
            F_tilde_f = contravariant_flux_from_metric(
                det_fine[c0:c1], inv_fine[c0:c1], F_phys_f)
            del F_phys_f
            div_f = contract_shared_operator_2axis(op_D_fine, F_tilde_f)  # (n,n_fine,1)
            del F_tilde_f
            div_F[c0:c1] = contract_shared_operator_1axis(op_f2c, div_f)[..., 0]
            del div_f
    return div_F


def compute_scalar_convection_residual(
    scalar_field: np.ndarray,
    rho: np.ndarray,
    velocity: np.ndarray,
    mesh,
    ops,
    wall_dirichlet_zero_face: np.ndarray = None,
    wall_dirichlet_value_face: np.ndarray = None,
    has_wall_dirichlet_value: np.ndarray = None,
    flat_face_override=None,
    conv_geom: "ScalarConvectionGeometry" = None,
) -> np.ndarray:
    """计算标量对流 FR 残差（体积项 + 界面上风校正）。

    对流方程: d(rho*phi)/dt + div(rho*U*phi) = 0
    残差 = -div(rho*U*phi)/det(J) + interface_correction/det(J)

    Args:
        scalar_field: (n_cells, n_sps) 标量场（k 或 omega）
        rho: (n_cells, n_sps) 密度
        velocity: (n_cells, n_sps, 3) 速度
        mesh: HighOrderMesh
        ops: FROperators
        wall_dirichlet_zero_face: (n_faces,) bool，可选，见
            `_extrapolate_scalar_to_faces` 文档——只影响 scalar_field
            自身外插到面的 ghost 值（决定上风通量的物理量），不影响
            rho/velocity 的外插（无滑移壁面上 u=0 已经由平均流的
            WALL 幽灵态保证，这里不需要重复处理）。
        wall_dirichlet_value_face, has_wall_dirichlet_value: 见
            `_extrapolate_scalar_to_faces` 文档，omega 壁面解析式用，
            与 wall_dirichlet_zero_face 互斥。
        flat_face_override: 显式传入时优先使用，不再调用
            `get_flat_face_geometry(mesh, ops)`（2026-09-02 分布式湍流
            移植新增，与 `core/fr_residual/inviscid.py::compute_
            inviscid_residual_fr` 同名参数同一个道理）——分布式路径下
            `mesh` 是 `DistributedMeshAdapter`，其 `face_connectivity`
            是 local+halo 压缩索引空间下的 `DistributedFlatFaceGeometry`，
            不是 `get_flat_face_geometry` 内部期望的完整全局
            `FRFaceConnectivity`，用它重新构建一遍会得到错误的面几何。
            调用方需要传入已经按同一套压缩索引空间构造好的
            `dist_fc.base_flat`。单机路径不传，行为完全不变。

    Returns:
        residual: (n_cells, n_sps) 对流残差（dphi/dt 量纲，已除以 rho 前的
                  原始残差，调用方需自行除以 rho）
    """
    n_cells = mesh.n_cells
    n_sps = mesh.n_sps_per_cell
    n_prism = mesh.n_prism_cells

    det_jacs = mesh.jacobians["det_jacs"].reshape(n_cells, n_sps)
    inv_jacs = mesh.jacobians["inv_jacs"].reshape(n_cells, n_sps, 3, 3)

    # === 体积项：按单元分块执行 adj_j 构造→物理通量→逆变通量→散度 全链路
    # （真实内存修复，2026-09-01，见项目 memory：cube_demo 79万单元 P2+SST
    # 组合内存峰值实测约需 37GB，超过常见 32GB 工作站配置，会在 P2 阶段
    # 第一步内触发 numpy ArrayMemoryError）。此前这里一次性对全场分配
    # `adj_j`（(n_cells,n_sps,3,3)~1.7GB）+ `rho_u_phi`/`F_tilde`
    # （各~500MB），SST 每步要独立调用本函数 4 次（k 对流/k 扩散/omega
    # 对流/omega 扩散，见 compute_turbulence_transport_residual），即使
    # 每次都及时 del，瞬时峰值仍会与同一步里平均流 inviscid.py/
    # viscous_flux.py 自己的大数组同时驻留，合计推高总峰值。这条链上
    # 每一步都是 cell 局部的（`np.matmul`/`np.tensordot` 按 cell 批量，
    # cell 之间零数据依赖），把 cell 轴切块、块内走完链路再进下一块，
    # 与 `fr_residual/inviscid.py`/`viscous_flux.py` 的过积分/体积项
    # 分块修复同一个原理、同一个分块大小（`_VISC_CHUNK_CELLS`），块内
    # 计算形状/求和顺序与全场版逐位一致，数值结果不变。
    # native 四面体（路径C）：D_3d_tet 是坍缩坐标专属微分矩阵，对 native
    # 单纯形基节点没有意义，需改用已零填充到全局宽度的 D_native_tet_padded
    # （理由与 gradients.py::compute_physical_gradient/viscous_flux.py
    # 体积项同一处文档，此前这里从未适配，见本模块 native 分支引入记录）。
    _tet_op_D = ops.D_native_tet_padded if getattr(ops, "D_native_tet_padded", None) is not None else ops.D_3d_tet
    # 性能优化（2026-09-13）：整条"度量×通量→散度"链压进 `scalar_convection_
    # volume_kernel`（按 cell prange，见该 kernel 文档），并复用调用方预先
    # 算好的、与标量无关的 `rho_u_tilde`（k/omega 两次调用共享，见
    # `ScalarConvectionGeometry` 文档）。原实现是 Python 层分块 +
    # `np.matmul` 逐点 3x3@3x1 微型 gemm + 3 次 `np.tensordot`，既物化多份
    # 块中间数组、又完全不随 CPU 核数并行。
    if conv_geom is not None:
        rho_u_tilde = conv_geom.rho_u_tilde
    else:
        rho_u = rho[:, :, None] * velocity
        rho_u_tilde = contravariant_flux_from_metric(det_jacs, inv_jacs, rho_u[..., None])[..., 0]
        del rho_u
    # 去混叠（AFCFD_TURB_OVERINT=on）：`rho*u*phi` 是三重非线性乘积，
    # 直接在 coarse SPs 上微分等价于"先混叠再求导"，理由与实测数字见
    # `resolve_turb_overintegration` 文档。默认 off，行为逐位不变。
    _oi = _turb_overint_ops(mesh, ops) if resolve_turb_overintegration() == "on" else None
    if _oi is not None:
        div_F = _scalar_convection_volume_overintegrated(
            scalar_field, rho, velocity, _oi, n_sps)
    else:
        div_F = np.empty((n_cells, n_sps))
        if n_prism > 0:
            scalar_convection_volume_kernel(
                np.ascontiguousarray(scalar_field[:n_prism]),
                np.ascontiguousarray(rho_u_tilde[:n_prism]),
                np.ascontiguousarray(ops.D_3d_prism), div_F[:n_prism],
            )
        if n_cells > n_prism:
            scalar_convection_volume_kernel(
                np.ascontiguousarray(scalar_field[n_prism:]),
                np.ascontiguousarray(rho_u_tilde[n_prism:]),
                np.ascontiguousarray(_tet_op_D), div_F[n_prism:],
            )
    if conv_geom is None:
        del rho_u_tilde

    # 真实复现（2026-08-21，79万单元生产网格，Order Continuation P0->P1
    # 切换后）：退化单元（坍缩坐标/BL 挤出导致 det(J) 局部极小，见
    # troubled_cell.py 模块文档）上这里会真的溢出到 inf，`np.errstate` 只是
    #
    # **过时表述已更正（2026-09-15）**：这里原先写的是"平均流残差有
    # mechanism-1/2 两道专门保护，本模块至今没有"。两句都不再成立：
    # (a) 平均流对残差的**实际干预**早已全部由机制 3
    #     （`suppress_residual_outliers`）承担，机制 1/2 的检测判据只
    #     保留用于诊断报告（见 troubled_cell.py 文档"机制3"一节）；
    # (b) 本模块**已经接入**机制 3——见
    #     `compute_turbulence_transport_residual` 末尾对
    #     `dk_dt_transport`/`domega_dt_transport` 的
    #     `suppress_residual_outliers` 调用（那里的注释记录了接入过程）。
    # 也就是说本模块现在与平均流用的是同一套、也是唯一在实际起作用的
    # 那套保护。排查"湍流输运缺 troubled-cell 保护"这条疑点时是靠核实
    # 代码而不是照搬这段注释才发现它过时的——本项目有过多次过时标注被
    # 当成真实缺口的先例。
    # 让这个*已知、已经在下游处理*的溢出不再往 stderr 打印 RuntimeWarning
    # 噪音——不改变任何数值结果：`compute_turbulence_transport_residual`
    # 末尾的 `np.where(np.isfinite(...), ..., 0.0)` 本来就会把这类 inf/nan
    # 结果清零，交给 SST.update_fields 的正性限制器接管（见该函数文档
    # "NaN/Inf 隔离"一节），此前只是没有抑制这个警告，容易被误读成新
    # 出现的异常。真正的解析修复（把 troubled-cell 的 mechanism-1/2
    # 保护接入湍流标量输运）是比这更大的独立工作，见
    # transport_kernel.py::extrapolate_scalar_to_faces_kernel 文档同类
    # 说明。
    with np.errstate(over='ignore', invalid='ignore'):
        residual = -div_F / det_jacs  # 体积项对流残差
    del div_F  # 体积项已经收尾，界面项不再需要它（F_tilde_chunk 已在分块循环内逐块释放）

    # === 界面项（上风校正）===
    flat = flat_face_override if flat_face_override is not None else get_flat_face_geometry(mesh, ops)
    n_fp = flat.n_fp

    # 外插 rho, velocity, scalar 到面通量点——rho/velocity 只需要 owner 侧
    # （下面 mass_flux 只用 owner 侧状态，见 _extrapolate_owner_only_to_faces
    # 文档"真实内存修复"一节：neighbor 侧此前算出来从未被读取，纯浪费）；
    # 只有标量场本身需要两侧（上风选择要比较 owner/neighbor）。
    if conv_geom is None:
        rho_owner_fp = _extrapolate_owner_only_to_faces(rho, flat)
        vel_owner_fp = np.zeros((flat.n_faces, n_fp, 3))
        for d in range(3):
            vel_owner_fp[:, :, d] = _extrapolate_owner_only_to_faces(velocity[:, :, d], flat)
    phi_owner_fp, phi_neighbor_fp = _extrapolate_scalar_to_faces(
        scalar_field, flat, ops, mesh, wall_dirichlet_zero_face,
        wall_dirichlet_value_face, has_wall_dirichlet_value,
    )

    # 计算每个面通量点的物理质量通量（使用 true_normal）
    # mass_flux_phys[f,fp] = (rho * U) . n̂_true
    # `mass_flux` 与标量无关，k/omega 共享（见 `ScalarConvectionGeometry`
    # 文档）——传入 conv_geom 时直接复用，省掉 rho 与 3 个速度分量各一次
    # 全场面外插（187 万面×n_fp 个通量点）。
    if conv_geom is not None:
        mass_flux = conv_geom.mass_flux
    else:
        rho_u_owner = rho_owner_fp[..., None] * vel_owner_fp  # (n_faces, n_fp, 3)
        mass_flux = np.sum(rho_u_owner * flat.true_normal, axis=-1)  # (n_faces, n_fp)
        del rho_owner_fp, vel_owner_fp, rho_u_owner  # 同上"真实内存修复"一节，及时释放

    # 迎风选择
    phi_upwind = np.where(mass_flux >= 0, phi_owner_fp, phi_neighbor_fp)

    # 通量差（用于校正分配）——真实 bug 修复（2026-09-12，cube_demo
    # 791,492 单元真实网格 P0 阶段长程续算 k_max/omega_max 复合增长排查
    # 发现，见下方"owner/neighbor 跳变量不对称"完整推导）：owner 侧和
    # neighbor 侧的校正必须分别相对各自的面值计算，不能共用同一个相对
    # phi_owner_fp 的跳变量。
    #
    # 完整推导（DG/FR 标准做法，见 core/fr_residual/inviscid_kernel.py
    # 里 AUSM+up 通量分两次、分别用 owner/neighbor 各自状态当"内部通量"
    # 计算 jump_owner/jump_neighbor 的既有正确实现——本函数此前没有跟
    # 那个模式对齐，是本模块独立实现、独立踩坑的同一类问题）：
    #   owner 侧："体积项"（P1+ 时由 D_3d 算出，P0 时恒为 0）隐含假设
    #     该单元通过每个面的"自身内部通量"是 mass_flux*phi_owner_fp
    #     （用 owner 自己的面值）；界面校正 = 真实通量(common,用上风值)
    #     减去这个假设值 = mass_flux*(phi_upwind-phi_owner_fp)，即现有
    #     的 raw_jump_fp，continue 用于 owner 侧不变。
    #   neighbor 侧：同一个面，站在 neighbor 自己的体积项视角，它的
    #     "自身内部通量"假设是（用 neighbor 自己的面值，法向对调一次，
    #     两次取负抵消）mass_flux*phi_neighbor_fp——不是 phi_owner_fp！
    #     neighbor 侧真正的界面校正 = mass_flux*(phi_upwind-phi_neighbor_fp)，
    #     是一个独立于 raw_jump_fp 的量，只在 phi_owner_fp==phi_neighbor_fp
    #     （面上没有跳变）时才恰好相等。
    #
    # 此前代码把 owner 侧算出的 raw_jump_fp 原样又给 neighbor 侧用（只是
    # 累加符号相反），在 owner 恰好处于该面上风侧时（mass_flux>=0）
    # phi_upwind==phi_owner_fp，raw_jump_fp 恒为 0——这种情形下 neighbor
    # （下风侧，真实网格中占全部内部面里的一半）完全没有从这个面收到
    # 任何对流稀释/浓缩效果，等价于凭空丢弃了这部分物理输运。这是 P0
    # 阶段（体积项恒零、全部输运只能靠界面项）唯一的输运来源，这个缺口
    # 因此完全暴露：局部单元一旦（哪怕因为其他机制）值略高于周围，会因为
    # 系统性地收不到来自上风侧的正确稀释而持续偏高，若同时有其他面
    # 贡献哪怕很小的净流入，缺乏正确抵消的稀释就会造成真实、持续、复合
    # 的数值增长（真实观测：cube_demo 单元 87035，k 从 iter 2600 的 0.99
    # 复合增长到 iter 3100 的 2.05，约 1.2x/100步；最小 2-四面体合成网格
    # 决定性复现：owner 上风时 neighbor 完全收不到本该有的稀释，conv_k
    # 恒为 0，与真实物理应有的非零稀释矛盾）。P1+ 阶段体积项非零、能
    # 部分补偿这个缺口，这也是该问题只在长期停留 P0 的场景下才充分暴露
    # 的原因。
    # 注（2026-09-12，尝试并撤销的一次修复，记录下来避免后续重蹈覆辙）：
    # 这里曾经尝试在 P0（n_sps==1）时改用"直接有限体积"公式（跳变量直接
    # 取 phi_upwind，不减任何一侧自身面值），模仿 core/fr_residual/
    # inviscid_p0.py 的做法，理由是 P0 架构上 volume_term 恒为 0（D_3d
    # 是 1x1 零矩阵），下面这套"跳变量=上风值-自身面值"的 DG 差额公式
    # 因此永远缺失一部分理应由 volume_term 提供的贡献。用合成网格做
    # 独立正确性检验（均匀标量场、无跳变，纯对流残差必须处处精确为
    # 零——这是不依赖具体公式、只依赖"无源无跳变"这个前提的基本不变量）
    # 时**决定性证伪**：改成"不减自身面值"后，均匀场在某些单元上给出
    # 高达 451 的伪残差（应为 0），根源是这个简化后的公式又变得对
    # `sum_faces(mass_flux)`（局部质量守恒残差，真实网格计算中后期该量
    # 远非零，尤其是本次真实排查最早定位到的"驻点残差"单元）敏感——
    # inviscid_p0.py 的直接公式之所以安全，是因为它求解的是完整
    # 5 变量 Euler 方程组本身（该公式与连续性方程自洽是同一个解），而
    # k/omega 是附着在已经独立演化、瞬时并不精确满足连续性的密度场上的
    # 被动标量方程，直接照搬会重新引入当天早些时候已经用真实数据证伪过
    # 的"质量不守恒交叉项"敏感性——绕了一圈证实那次证伪的判断没错，只是
    # 用错了地方（旧公式的"减自身面值"设计正是为了规避这个敏感性，代价
    # 是缺一部分 dilution；不能两者都要）。已撤销，保留下面的原始形式，
    # 只保留 owner/neighbor 跳变量分别独立计算这一个改动（见上方修复
    # 说明），未继续尝试消除"缺失 dilution"这个更深的架构缺口。
    delta_phi_owner = phi_upwind - phi_owner_fp  # (n_faces, n_fp)
    delta_phi_neighbor = phi_upwind - phi_neighbor_fp  # (n_faces, n_fp)
    del phi_upwind, phi_owner_fp, phi_neighbor_fp
    # 面元幅值因子（真实修复，2026-08-25 代码审查）：上面用单位法向算出的是物理
    # 通量密度差，而平均流无粘/粘性界面项送进同一套分配链路的跳越量都是协变
    # 通量（物理通量 × |adj_row|，含面元幅值）：inviscid_kernel.py L197
    # `F_common_n * adj_mag`（adj_mag 归一化只用于方向对齐检查，幅值随后乘回）、
    # viscous_flux_kernel.py L182 `adjrow_o · G`。缺这个 ~O(h²) 因子会把校正放大
    # ~1/h²（细网格 10²~10³ 倍）。true_normal 是单位向量（见
    # face_flux_points_exact_normal.py），必须补回 |adj_row|。
    #
    # native 四面体（路径C）真实 bug 修复（2026-08-30，见
    # transport_kernel.py::distribute_corrections_to_cells_kernel 文档）：
    # 此前这里统一用 |owner_adj_row_exact| 当面元幅值因子——但 native 面
    # 需要的是真实物理面积权重 `true_area_weight`（DG 提升算子弱形式积分
    # 用的量），与 collapsed 面的度量张量 adj 行模长不是同一个量，不能
    # 在这里统一预乘后再传给下游。改为只传"未加权"的 `mass_flux*delta_phi`，
    # 加权方式按面类型分派下沉到 `_distribute_correction_to_cells`/
    # kernel 内部。
    raw_jump_fp = mass_flux * delta_phi_owner  # (n_faces, n_fp)，owner 侧未加权物理通量密度差
    raw_jump_fp_neighbor = mass_flux * delta_phi_neighbor  # neighbor 侧独立的未加权跳变量
    del mass_flux, delta_phi_owner, delta_phi_neighbor

    # 分配回 SPs
    interface_correction = _distribute_correction_to_cells(
        raw_jump_fp, flat, ops, mesh, raw_jump_fp_neighbor=raw_jump_fp_neighbor,
    )
    with np.errstate(over='ignore', invalid='ignore'):
        residual = residual + interface_correction

    return residual


def _scalar_diffusion_volume_overintegrated(gamma_field, grad_phi, oi, n_sps):
    """扩散体积项 `div(adj(J)*Gamma*grad_phi)` 的去混叠版，返回
    (n_cells,n_sps)。

    与对流版同一条链路：Gamma 与 grad_phi 各自精确插值到 FINE 点后**在
    FINE 点相乘**，用 FINE 点度量算逆变通量、FINE 微分矩阵求散度，再
    精确限制回 coarse。

    **`Gamma` 自身的混叠：已量化、刻意不实施（2026-09-15 结论）**。
    `Gamma = mu + sigma*rho*nu_t` 里 `nu_t` 是**商**、根本不是多项式，
    它在 coarse 点上的节点值本身已经是一个投影结果——本函数只能把这份
    节点表示精确插值到 FINE 点，无法像平均流那样"在 FINE 点重新求值
    非线性通量函数"（`fr_residual/inviscid.py` 能那样做是因为它手里有 Q）。
    所以这里去掉的是 **Gamma×grad_phi 乘积以及与度量项乘积**的混叠。

    要不要把 Gamma 自身那层也去掉，做过受控测量（`Gamma = mu +
    s*rho*k/omega`，rho/k/omega 全取一次场，与解析散度比较，相对 L-inf）：

        order=1 prism: 插值 Gamma 1.0019e-02 -> 细点精确 3.2491e-05  (308x 更好)
        order=1 tet  : 插值 Gamma 9.3413e-03 -> 细点精确 1.4530e-04  ( 64x 更好)
        order=2 prism: 插值 Gamma 1.7408e-02 -> 细点精确 3.4978e-02  (0.50x **更差**)
        order=2 tet  : 插值 Gamma 1.8643e-03 -> 细点精确 1.6869e-03  (1.11x)

    **order=2 上"细点精确求值"反而更差**，不是测量噪声：在细点精确求值
    一个有理函数、再对它的 degree-over_order 插值求导，会把更多高频内容
    带进微分算子；而插值过的 Gamma 本身更平滑。也就是说这条改动**不是
    单调有益**的。

    代价侧同样不小：生产里 `sigma` 来自 SST 混合函数 `F1`，而 `F1` 依赖
    `wall_distance`——那是纯几何量，**必须重新 KD-Tree 查询、不能插值**
    （2026-09-05 真实 bug 修复：阶数切换时把 wall_distance 当解多项式场
    插值导致 d1 系统性偏大），细点是 791492x64 ≈ 5070 万个查询点；此外
    每步还要在 8 倍点数上重算 `F1`/`nu_t`/`CD_kw`。

    综合判断：在 order=2 净负、order=1 的收益又落在一个相对误差已经只有
    1e-2 的项上（对比本轮去掉的 200%~498%），不值这个代价。**这是一条
    有数据支撑的结论，不是待办项**——若将来工作阶数或精度诉求变化需要
    重新评估，上面的数字与代价分析可以直接复用。
    """
    n_cells = gamma_field.shape[0]
    det_fine, inv_fine = oi["det_fine"], oi["inv_fine"]
    div_G = np.empty((n_cells, n_sps))
    for seg_lo, seg_hi, op_c2f, op_D_fine, op_f2c in oi["segs"]:
        for c0 in range(seg_lo, seg_hi, _TURB_OVERINT_CHUNK_CELLS):
            c1 = min(c0 + _TURB_OVERINT_CHUNK_CELLS, seg_hi)
            gam_f = contract_shared_operator_1axis(
                op_c2f, np.ascontiguousarray(gamma_field[c0:c1, :, None]))
            grad_f = contract_shared_operator_1axis(
                op_c2f, np.ascontiguousarray(grad_phi[c0:c1]))        # (n,n_fine,3)
            G_phys_f = (gam_f * grad_f)[..., None]                    # (n,n_fine,3,1)
            del gam_f, grad_f
            G_tilde_f = contravariant_flux_from_metric(
                det_fine[c0:c1], inv_fine[c0:c1], G_phys_f)
            del G_phys_f
            div_f = contract_shared_operator_2axis(op_D_fine, G_tilde_f)
            del G_tilde_f
            div_G[c0:c1] = contract_shared_operator_1axis(op_f2c, div_f)[..., 0]
            del div_f
    return div_G


def compute_scalar_diffusion_residual(
    scalar_field: np.ndarray,
    gamma_field: np.ndarray,
    mesh,
    ops,
    wall_dirichlet_zero_face: np.ndarray = None,
    wall_dirichlet_value_face: np.ndarray = None,
    has_wall_dirichlet_value: np.ndarray = None,
    flat_face_override=None,
) -> np.ndarray:
    """计算标量扩散 FR 残差（体积项 + BR1 界面校正），返回值为 dphi/dt。

    扩散方程: d(rho*phi)/dt = div(Gamma * grad(phi))

    符号约定（2026-08-25 修复）：本函数返回值被调用方直接作为 dphi/dt
    相加（见模块文档"符号约定"），与平均流粘性残差 viscous_flux.py 的
    "粘性项是 +div(G)"同一约定——物理扩散使峰值摊平、谷值抬升，
    dphi/dt = +div(Gamma*grad(phi))/det(J)。此前误写成 -div(...)（反扩散），
    指数放大棋盘模态导致 k/omega 场双峰触限、求解停滞（见模块文档）。
    界面校正对应地用 `residual - interface_correction`：分配 kernel 对 owner
    侧是 -=（见 transport_kernel.py::distribute_corrections_to_cells_kernel），
    因此这里减去它等于对 dphi/dt 施加 +lift(G_common - G_internal)——
    与对流 `residual + interface_correction` 的表面差异全部来自体积项符号，
    不是扩散物理要求相反符号（此前注释"扩散是反梯度通量，校正应减小残差"
    的物理表述有误，一并更正）。

    Args:
        scalar_field: (n_cells, n_sps) 标量场
        gamma_field: (n_cells, n_sps) 有效扩散系数 Gamma
        mesh: HighOrderMesh
        ops: FROperators
        wall_dirichlet_zero_face: (n_faces,) bool，可选。保留参数以兼容调用方：
            2026-08-25 校正改用梯度差形式后，WALL Dirichlet-zero 的奇镜像
            ghost（ghost=-owner，见 extrapolate_scalar_to_faces_kernel 文档）
            对梯度场是对称的（奇函数的导数是偶函数），边界面上梯度跳跃自然为
            零，不需要单独处理；此前用状态跳跃校正时这个掩码决定 phi 的 ghost
            取值，校正改梯度差后不再有数值作用，留作后续补壁面扩散通量的接口。
        wall_dirichlet_value_face, has_wall_dirichlet_value: 同理保留以兼容
            调用方（omega 壁面解析式），出于与上面 wall_dirichlet_zero_face
            完全相同的理由（本函数不再外插 scalar_field 自身，只外插
            gamma_field/grad_phi），当前同样对本函数的数值结果没有影响——
            壁面 Dirichlet 值目前只通过 `compute_scalar_convection_residual`
            的上风 ghost 生效，diffusion 侧的解析壁面通量是更大的独立工作
            （与 k=0 情形是同一个已有的架构限制，不是本次新引入的差异）。

            2026-09-05 曾尝试补上这里的 SIPG（对称内罚 Galerkin）风格
            Dirichlet 罚通量（`penalty = gamma_face*(C_pen/d1)*(target-
            phi_owner)`，显式加进 flux_jump_phys），真实网格验证（从
            cube_demo 791,492 单元真实 checkpoint 续算 20 步）**决定性
            证伪**：C_pen/d1 这个有效"弹簧系数"在细网格近壁单元上
            （y+~1 设计意味着 d1 可以小到 1e-5~1e-4 量级）过大，用和
            其余残差项相同的显式时间积分（没有做成 point-implicit，
            对比 sst.py::update_fields 里 destruction 项那样的处理）
            必然刚性超调——20 步内把全域 omega_mean 从 2.8e4 打到
            5.4e11（远超 1e6 的安全上限，且是全域均值，不是局部
            异常），k_mean 被连带压垮到 0.16，是真实的数值失稳而不是
            "改善"。已完整撤销（Git 历史可查这次尝试+撤销的完整过程），
            扩散侧解析壁面通量这个架构缺口依然存在，如需真正补上，
            必须先把罚项做成 point-implicit（或大幅限制 dt/加严格的
            CFL 缩放），不能像这次一样直接显式代入——留给后续需要
            专门处理数值刚性的独立工作，不要重复这次已经证伪的显式
            实现方式。当前生产代码依赖 `enforce_omega_wall_relaxation`
            （事后松弛，已用真实 900+ 步续算验证是稳定的）作为这个
            架构缺口的安全缓解措施。

        flat_face_override: 见 `compute_scalar_convection_residual` 同名
            参数文档，分布式路径复用同一个约定。

    Returns:
        residual: (n_cells, n_sps) 扩散残差
    """
    n_cells = mesh.n_cells
    n_sps = mesh.n_sps_per_cell
    n_prism = mesh.n_prism_cells

    det_jacs = mesh.jacobians["det_jacs"].reshape(n_cells, n_sps)
    inv_jacs = mesh.jacobians["inv_jacs"].reshape(n_cells, n_sps, 3, 3)

    # === 体积项 ===
    # 计算标量梯度（度量项一致）——grad_phi 下面界面项的逐分量外插还要
    # 复用，不能纳入分块（只能分块处理它之后、只在体积项内部一次性
    # 使用的 adj_j/G_phys/G_tilde）。
    grad_phi = compute_physical_scalar_gradient(scalar_field, mesh, ops)  # (n_cells, n_sps, 3)

    # 按单元分块执行 adj_j 构造→扩散通量→逆变通量→散度 全链路（真实
    # 内存修复，2026-09-01，理由与 compute_scalar_convection_residual
    # 体积项分块同一处文档：cube_demo 79万单元 P2+SST 组合内存峰值实测
    # 约需 37GB，超过常见 32GB 工作站配置）。
    _tet_op_D = ops.D_native_tet_padded if getattr(ops, "D_native_tet_padded", None) is not None else ops.D_3d_tet
    # 性能优化（2026-09-13，与对流项体积项同一次剖析、同一手法）：整条
    # "度量×扩散通量→散度"链换成 `contravariant_flux_from_metric` +
    # `scalar_volume_divergence_kernel` 两个 numba prange kernel，取代
    # 原来 Python 层分块 + 逐点 3x3@3x1 微型 gemm + 3 次 tensordot 的
    # numpy 链路（见两个 kernel 各自的文档）。
    # 去混叠（AFCFD_TURB_OVERINT=on），理由见 `resolve_turb_overintegration`
    # 与 `_scalar_diffusion_volume_overintegrated`（含那里明确写出的、
    # Gamma 自身混叠仍在的局限）。默认 off，行为逐位不变。
    _oi = _turb_overint_ops(mesh, ops) if resolve_turb_overintegration() == "on" else None
    if _oi is not None:
        div_G = _scalar_diffusion_volume_overintegrated(
            gamma_field, grad_phi, _oi, n_sps)
    else:
        G_phys = gamma_field[:, :, None] * grad_phi                      # (n_cells,n_sps,3)
        G_tilde = contravariant_flux_from_metric(det_jacs, inv_jacs, G_phys[..., None])[..., 0]
        del G_phys
        div_G = np.empty((n_cells, n_sps))
        if n_prism > 0:
            scalar_volume_divergence_kernel(
                np.ascontiguousarray(G_tilde[:n_prism]),
                np.ascontiguousarray(ops.D_3d_prism), div_G[:n_prism],
            )
        if n_cells > n_prism:
            scalar_volume_divergence_kernel(
                np.ascontiguousarray(G_tilde[n_prism:]),
                np.ascontiguousarray(_tet_op_D), div_G[n_prism:],
            )
        del G_tilde

    # 扩散对 dphi/dt 的贡献是 +div(G)/det(J)（见本函数文档符号约定，
    # 与 viscous_flux.py::"residual = div_comp / det_jacs"同一约定）。
    # 退化单元溢出保护：理由/验证方式同 compute_scalar_convection_
    # residual 里对应的 errstate（见该函数文档），同一类已知、已在
    # compute_turbulence_transport_residual 末尾被下游清零处理的溢出。
    with np.errstate(over='ignore', invalid='ignore'):
        residual = div_G / det_jacs
    del div_G

    # === 界面项（BR1 平均通量校正）===
    flat = flat_face_override if flat_face_override is not None else get_flat_face_geometry(mesh, ops)
    n_fp = flat.n_fp

    # 外插 Gamma 和标量梯度到面通量点。标量场本身不再外插：2026-08-25 校正改
    # 用梯度差形式后，phi 的界面取值（原 BR1 phi_avg）不再进入校正项；
    # gamma_field 用 Neumann 默认（扩散系数与是否 Dirichlet 无关），
    # grad_phi 逐分量双侧外插——BR1 公共通量需要平均梯度，两侧缺一不可。
    gamma_owner_fp, gamma_neighbor_fp = _extrapolate_scalar_to_faces(gamma_field, flat, ops, mesh)

    grad_owner_fp = np.zeros((flat.n_faces, n_fp, 3))
    grad_neighbor_fp = np.zeros((flat.n_faces, n_fp, 3))
    for d in range(3):
        go, gn = _extrapolate_scalar_to_faces(grad_phi[:, :, d], flat, ops, mesh)
        grad_owner_fp[:, :, d] = go
        grad_neighbor_fp[:, :, d] = gn
    del grad_phi  # 体积项+这里的逐分量外插都用完了，终于可以释放

    # BR1 平均：gamma_face = 0.5*(gamma_o + gamma_n)
    gamma_face = 0.5 * (gamma_owner_fp + gamma_neighbor_fp)
    del gamma_owner_fp, gamma_neighbor_fp

    # 通量差（真实修复，2026-08-25）：G_common - G_internal =
    # gamma_face*(grad_avg - grad_owner) = gamma_face*0.5*(grad_n - grad_o)
    # （phi_avg 的梯度即两侧外插梯度的平均）。此前实现因"grad_phi_neighbor
    # 不能直接外推到面 FPs"而放弃梯度差、改用状态跳跃 gamma_face*0.5*(phi_n
    # - phi_o) 冒充通量差——外插算子对任何标量场（包括梯度分量）本来就同样
    # 适用，这个前提不成立；且状态跳跃缺一个 1/长度因子，量纲与通量密度差
    # ~h 倍，与反扩散体积项叠加后成为 k/omega 场双峰触限失稳的放大器。
    delta_grad = 0.5 * (grad_neighbor_fp - grad_owner_fp)  # (n_faces, n_fp, 3)
    with np.errstate(over='ignore', invalid='ignore'):
        # 取法向分量：扩散通量差是矢量差的法向投影，坐标不变；简单三分量
        # 求和会随坐标系旋转变号/变幅值，不是标量不变量。
        flux_jump_phys = gamma_face * np.sum(delta_grad * flat.true_normal, axis=-1)
    del grad_owner_fp, grad_neighbor_fp, delta_grad, gamma_face

    # 面元幅值因子（真实修复，2026-08-25 代码审查）：上面用单位法向点积算出的是物理
    # 通量密度差，而平均流无粘/粘性界面项送进同一套分配链路的跳越量都是协变
    # 通量（物理通量 × |adj_row|，含面元幅值）：inviscid_kernel.py L197
    # `F_common_n * adj_mag`（归一化只用于方向对齐检查，幅值随后乘回）、
    # viscous_flux_kernel.py L182 `adjrow_o · G`。缺这个 ~O(h²) 因子会把校正放大
    # ~1/h²（细网格 10²~10³ 倍），破坏体积项与界面项的量级平衡——实测：
    # 修复前抛物线场内部残差均值被界面项主导（+143 界面 / -21 体积），
    # 补 |adj_row| 后界面项回到与体积项同量级。true_normal 是单位向量，必须补回。
    #
    # native 四面体（路径C）真实 bug 修复（2026-08-30，理由同
    # compute_scalar_convection_residual 同名注释）：不在这里统一预乘
    # |adj_row|，改为传未加权的 flux_jump_phys，加权方式按面类型分派
    # 下沉到 _distribute_correction_to_cells/kernel 内部。
    raw_jump_fp = flux_jump_phys

    # 分配回 SPs（kernel 对 owner 侧 -=、neighbor 侧 +=）
    interface_correction = _distribute_correction_to_cells(raw_jump_fp, flat, ops, mesh)
    # 扩散校正合成符号（见本函数文档符号约定）：kernel 返回的 owner 侧贡献是
    # -lift(correction_fp)，这里 residual - interface_correction =
    # +lift(G_common - G_internal)，即对 dphi/dt 施加标准 +lift 校正：
    # 邻居值/梯度更高时 owner 获得正的扩散增量，方向与物理一致。
    with np.errstate(over='ignore', invalid='ignore'):
        residual = residual - interface_correction

    return residual


def _compute_wall_dirichlet_face_mask(solver) -> np.ndarray:
    """算出哪些面是**真实无滑移**WALL 边界面，供 k 场的 Dirichlet-zero
    ghost 及 omega 壁面解析式（Wilcox omega wall function）使用（见
    extrapolate_scalar_to_faces_kernel 文档）。

    数据来源：`solver.boundary_ghost_provider`——真实求解路径下是
    `boundary.fr_ghost_state.BoundaryGhostStateProvider`，持有
    `group_code`（每个面所属边界组的整数编码，-1 表示内部面/未匹配）和
    `code_to_config`（编码 -> {'type': 'WALL', 'is_no_slip': bool,...}）。用
    `code_to_config` 里显式标记为 WALL 的编码集合对 `group_code` 做一次
    向量化匹配（`np.isin`），成本是对 187 万面级别网格的一次数组比较，
    不是逐面 Python 循环。

    真实 bug 修复（2026-09-12，cube_demo 791,492 单元真实网格 P1 阶段
    omega 独立于历史、确定性地在特定单元收敛到同一个数值~984312 的排查
    发现，完整推导见 `enforce_omega_wall_relaxation` 文档）：此前这里把
    `cfg.get("type")=="WALL"` 的编码**不加区分**全部当作需要 Wilcox 近壁
    omega 解析式（`omega_wall=60*nu/(beta1*d1^2)`，专为*真实粘性无滑移*
    边界层设计）处理的壁面——但本项目的 WALL 类型边界组同时覆盖两种物理
    上完全不同的情形（见 `boundary/fr_ghost_state.py::build_wall_ghost_
    state` 的 `is_no_slip` 参数）：`is_no_slip=True`（真实固壁，如
    cube_demo 的 "body"）与 `is_no_slip=False`（滑移壁，如 cube_demo 的
    风洞外壁 "tunnel"，用于近似远场/对称边界，物理上零剪切、不产生真实
    边界层）。滑移壁没有真实的近壁粘性子层，Wilcox 公式在这里没有物理
    意义；真实复现：cube_demo 的 "tunnel" 边界组配置为
    `is_no_slip=False`，其 owner 单元因为不属于任何 BL 棱柱加密区、`d1`
    （该单元到最近 body 表面的距离）经常很小，代入公式得到远超合理量级
    的值，被 `_compute_omega_wall_target` 内部的 `omega_max` 安全上限
    钳成 1,000,000，随后 `enforce_omega_wall_relaxation` 每步把这些
    "滑移壁"owner 单元的 omega 强行按固定 relax=0.5 拉向这个物理上荒谬
    的目标（0.5*2267.82+0.5*1e6=501133.91，与真实观测的单步跳变值精确
    吻合，决定性验证：手术式重置 omega 场后单步复现同一批单元同一数值），
    与流场是否真的发展出边界层完全无关——这些单元的平均流速度全程精确
    等于来流值（无滑移损耗），却因为这个 bug 被反复拉向 1e6 附近，最终
    稳定在耗散项与松弛项相互竞争的一个不动点 984312.6254，与该单元是否
    真的靠近任何真实固壁毫无关系。修复：只把 `is_no_slip` 非 False
    （默认 True，与 `fr_ghost_state.py` 的默认值一致）的 WALL 编码计入——
    真实无滑移壁（body）行为完全不变，滑移壁（tunnel）的 k/omega 现在
    正确地退回默认 Neumann（零梯度）处理，与它"物理上应表现得像对称面/
    远场"这个建模意图一致。

    防御性回退：如果 `boundary_ghost_provider` 不是这个类型（例如某些
    测试用的自定义 ghost provider 只是一个普通 callable，没有
    group_code/code_to_config），拿不到分组信息时返回全 False——退回
    调用方原有的 Neumann 默认，不是新的静默 bug（这是修复前唯一的行为，
    对这些没有分组信息的场景数值结果不变）。
    """
    mesh = solver.mesh
    n_faces = mesh.face_connectivity.n_faces
    provider = getattr(solver, "boundary_ghost_provider", None)
    group_code = getattr(provider, "group_code", None)
    code_to_config = getattr(provider, "code_to_config", None)
    if group_code is None or code_to_config is None:
        return np.zeros(n_faces, dtype=np.bool_)

    wall_codes = [
        code for code, cfg in code_to_config.items()
        if cfg.get("type") == "WALL" and cfg.get("is_no_slip", True)
    ]
    if not wall_codes:
        return np.zeros(n_faces, dtype=np.bool_)
    return np.isin(group_code, wall_codes)


def _compute_omega_wall_target(
    solver, wall_mask: np.ndarray, mu: float, rho: np.ndarray, flat_face_override=None,
) -> Tuple[np.ndarray, np.ndarray]:
    """按 Wilcox 解析式计算 WALL 面各自的 omega 目标值（真实修复，
    V2.0 专家组盲审发现，2026-08-28）：

        omega_wall = 60*nu / (beta1 * d1^2)

    （Wilcox《Turbulence Modeling for CFD》标准公式，beta1=0.075 是
    SST 内层 beta 系数，omega/sst.py::SSTModelFR.beta1）。

    `d1` 的取法（与直接对 `solver.wall_distance` 做面外插的方案不同，
    是本次实现有意的选择，不是疏漏）：不能用
    `_extrapolate_scalar_to_faces(solver.wall_distance, ...)` 把
    wall_distance 外插到 WALL 面本身——wall_distance 是"到最近壁面的
    距离"，在几何上就位于壁面的这个面自身，外插值会趋于 0，代入公式
    会让 omega_wall 发散到无穷大，不是"精度稍差"而是量纲上完全错误。
    Wilcox 公式里的 d1 本来就是"近壁第一层网格点到壁面的距离"，不是
    "壁面到自身的距离"——这里直接取该面 owner 单元自身 SPs 上
    wall_distance 场的最小值，作为该单元的近壁特征距离（该单元里离墙
    最近的 SP 到墙的真实距离），比在面上外插整个场更贴近公式原意，
    也从根本上避免了除以零。

    Args:
        solver: FRSolver 实例（需要 solver.wall_distance 已计算）
        wall_mask: (n_faces,) bool，WALL 边界面掩码（
            _compute_wall_dirichlet_face_mask 的返回值）
        mu: 分子动力粘度
        rho: (n_cells, n_sps) 密度场，用于取 owner 单元的代表密度算 nu
        flat_face_override: 显式传入时优先使用，不再调用
            `get_flat_face_geometry(solver.mesh, solver.ops)`（真实修复，
            2026-09-02，见 `compute_turbulence_transport_residual` 同名参数
            文档——本函数此前是该文件里唯一没有跟随 flat_face_override 传参
            约定的函数，分布式场景下 `solver.mesh.face_connectivity` 是
            `DistributedFlatFaceGeometry`，不具备 `owner_cube_face` 等字段，
            "完全分布式加载"模式下会直接崩溃；"传统模式"不崩溃是因为
            `get_flat_face_geometry` 按 `mesh.face_flux_points` 对象身份
            缓存，而"传统模式"下这个身份与构造 `dist_fc` 时已经缓存过的
            全局 mesh 是同一个对象，命中缓存返回的是**全局**（非 compact）
            FlatFaceGeometry——`wall_face_idx`（来自 compact 空间的
            wall_mask）被当成全局面索引使用，语义上是错的，只是在
            现有测试里因为从未真正匹配到 WALL 边界组（wall_mask 恒为全
            False，见 `_compute_wall_dirichlet_face_mask` 提前返回分支）
            而没有被触发）

    Returns:
        (omega_wall_value_face, has_value_face)：
        - omega_wall_value_face: (n_faces, n_fp) float，WALL 面上恒为
          该面 owner 单元算出的标量（对该面所有 FP 广播同一个值，不是
          外插得到的逐 FP 不同值——d1 本身就是单元级别的代表量，不需要
          逐 FP 精细区分），非 WALL 面为 0（不会被使用，has_value_face
          对应位置为 False）
        - has_value_face: (n_faces,) bool，与 wall_mask 相同

    真实 bug 修复（2026-09-05，真实网格验证决定性发现）：`d1 = np.maximum(
    d1, 1e-8)` 只防止除零，不防止结果本身失控——cube_demo 791,492 单元
    真实网格上至少有一个 WALL 面 owner 单元的 wall_distance 恰好卡在
    这个 1e-8 下限（真实反推：观测到的 omega_wall 异常值 1.176e14 精确
    对应 d1=1e-8 代入公式的结果），算出 `omega_wall=60*nu/(beta1*d1^2)
    ~1e14`——比"远超任何工程壁面 omega 值"的安全上限 `omega_max`
    (=1e6，sst.py::SSTModelFR.k_max/omega_max 文档) 还要大 8 个数量级。
    此前唯一的消费者（`compute_scalar_convection_residual` 的上风
    ghost）碰巧没有暴露这个问题——无滑移壁面上对流通量本身趋于零，
    ghost 值再大也乘的是接近零的质量通量，天然被掩盖；2026-09-04/05
    新增的两个消费者（`enforce_omega_wall_relaxation` 直接把这个值
    混合进 omega_field 本身、以及当天当场被证伪撤销的 SIPG 罚项）
    都没有这层"乘以近零对流通量"的天然保护，完全暴露了这个此前从未
    触发过的缺口——真实复现：`enforce_omega_wall_relaxation` 点隐式
    公式本身完全正确（bounded in [0,1) 已有专门单元测试钉住），但
    "正确地"把 omega 松弛向一个物理上荒谬的 1e14 目标值，5 步内就把
    全域 omega_mean 打到 1.08e12。修复：在这里、也就是唯一的真值来源，
    把 `omega_wall` 夹到 `solver.turb_model.omega_max`（没有该属性时
    退回 1e6 保守默认），让所有消费者（现在的和未来任何新增的）都
    自动受益，不需要各自重复防御。
    """
    flat = flat_face_override if flat_face_override is not None else get_flat_face_geometry(solver.mesh, solver.ops)
    n_faces = flat.n_faces
    n_fp = flat.n_fp
    beta1 = getattr(solver.turb_model, "beta1", 0.075)
    omega_max = getattr(solver.turb_model, "omega_max", 1e6)

    omega_wall_value_face = np.zeros((n_faces, n_fp), dtype=np.float64)
    wall_face_idx = np.nonzero(wall_mask)[0]
    if len(wall_face_idx) > 0:
        owner_cells = flat.owner_cell[wall_face_idx]
        # **长度尺度口径（2026-09-15 发现的系统性偏差，可切换）**
        #
        # Menter 的 omega 壁面处理 `omega_wall = 10*6*nu/(beta1*Δy1^2)`
        # （这里的 60 就是 10x6）是按**第一层单元中心**的壁距标定的经验
        # 公式。而这里原先取的是 `min`——单元内**全部解点**壁距的最小值。
        # 那既不是单元中心也不是单元高度，不对应任何标准口径，而且因为
        # Gauss-Legendre 解点在高阶时向单元边界聚集，它让目标值产生
        # **随阶数变化**的系统性高估（解点相对壁面的归一化位置实测）：
        #
        #   order=1: 最近解点 0.2113*h -> (0.5/0.2113)^2 =  5.60x 高估
        #   order=2: 最近解点 0.1127*h -> (0.5/0.1127)^2 = 19.68x 高估
        #   order=3: 最近解点 0.0694*h -> (0.5/0.0694)^2 = 51.86x 高估
        #
        # 一个经过标定的壁面函数绝不该有这种阶数依赖。这也解释了为什么
        # 本函数的目标值会顶到 `omega_max`、被下游文档称作"1e6 量级的
        # 应急上限"而不是"日常合理松弛目标"（见
        # `enforce_omega_wall_relaxation` 里两次被真实数据证伪的尝试
        # 记录）——它被喂了一个小 2.4~7.2 倍的长度尺度。
        #
        # `mean`（单元内解点壁距的均值，≈ 形心壁距）与 Menter 的口径
        # 一致，且对阶数是一阶无关的。**默认仍为 `min`**：这是湍流模型
        # 的物理改动，降低近壁 omega 会抬高 nu_t，必须用真实长程数据
        # 验证过才能改默认值——本项目在 omega 壁面处理上已经有两次
        # "数学上更对但真实数据证伪"的先例。
        _d1_mode = os.environ.get("AFCFD_OMEGA_WALL_D1", "min").lower()
        if _d1_mode not in ("min", "mean"):
            raise ValueError(
                f"AFCFD_OMEGA_WALL_D1={_d1_mode!r} 不是合法取值（min | mean）。"
                f"'min' 是既有行为（单元内解点壁距最小值），'mean' 是与 "
                f"Menter 标定口径一致的形心壁距。")
        wd_owner = solver.wall_distance[owner_cells]
        d1 = wd_owner.min(axis=1) if _d1_mode == "min" else wd_owner.mean(axis=1)
        d1 = np.maximum(d1, 1e-8)
        # 只统计真实自由度（2026-09-15 审计）：`rho[owner_cells]` 的行
        # 单元类型任意混合，所以用逐行掩码版。native 四面体的零填充槽位
        # 冻结在初值、会变馊，混进 nu = mu/rho 会带进几个百分点的偏差。
        # 阶数从**数组自身**的 SP 轴反解，不读 solver.current_order/order：
        # 填充划分由被归约数组的 n_sps 决定，从数组反解恒与它自洽（理由见
        # `order_from_n_sps` 文档）。
        from autoflowcfd.fr.native_tet_padding import (
            order_from_n_sps, reduce_rows_over_real_sps,
        )
        _rows = rho[owner_cells]
        rho_owner = reduce_rows_over_real_sps(
            _rows, owner_cells < solver.mesh.n_prism_cells,
            order_from_n_sps(_rows.shape[1]), 'mean')
        nu_owner = mu / np.maximum(rho_owner, 1e-10)
        omega_wall = 60.0 * nu_owner / (beta1 * d1**2)
        omega_wall = np.minimum(omega_wall, omega_max)
        omega_wall_value_face[wall_face_idx, :] = omega_wall[:, None]

    return omega_wall_value_face, wall_mask


def enforce_omega_wall_relaxation(solver, dt, relax: float = None,
                                   flat_face_override=None) -> None:
    """真实 bug 修复（2026-09-04，cube_demo 791,492 单元真实网格 Order
    Continuation P0->P1 跨阶后长程发散排查发现，grad_vel 修复之后仍持续
    发散的第二个独立根因）：`_compute_omega_wall_target` 按 Wilcox 解析式
    算出的壁面 omega 目标值（`omega_wall=60*nu/(beta1*d1^2)`，量级可达
    1e5~1e6）**只通过 `compute_scalar_convection_residual` 的上风 ghost
    生效**——`compute_scalar_diffusion_residual`（近壁 omega 动力学的
    主导机制，因为壁面无滑移使对流通量本身趋于零）文档明确写明这个
    解析值"当前对本函数的数值结果没有影响"，是已知、有意搁置的架构
    缺口（"diffusion 侧的解析壁面通量是更大的独立工作"）。

    真实后果（决定性验证，见 verify_gradfix_500steps.py 长程复现）：
    没有扩散侧的强约束，纯靠耗散项 D_omega=rho*beta*omega^2 的显式
    积分，边界层棱柱单元的 omega 会在数十~上百步内被压向下界
    （真实测得：166,980个边界层单元里 90,416 个、66%在150步内至少有
    一个解点 omega<1e-6，且这个比例逐步增长而非趋于稳定）——omega
    塌陷经 nu_t=a1*k/max(a1*omega,...) 的近零分母奇点反过来把湍流
    粘性比推到安全上限（真实测得 nu_t/nu_molecular~1e5，触及
    TURBULENT_VISCOSITY_RATIO_MAX），持续向平均流注入过量粘性应力，
    是 grad_vel 修复后仍能观测到的中长期（~100步后）持续增长的直接
    驱动源（而不是 grad_vel bug 本身遗留的影响——那个 bug 修复后已
    验证首个~90步完全无发散迹象，本机制独立起效于其后）。

    本函数用最低数值风险的方式补上这个缺口：不改动扩散残差/DG通量
    的稳定性特征，而是在 update_fields+positivity limiter 之后，
    直接对 WALL 面 owner 单元的 omega_field 做一次向解析壁面目标值的
    松弛（标准壁面函数做法，等价于 OpenFOAM omegaWallFunction 对
    近壁单元值的直接赋值/松弛处理，不是发明新方案）。`relax` 是
    固定松弛系数（每步只走向目标值的这个比例，不是硬性 hard-set，
    避免单步冲击过大引入新的震荡）。

    2026-09-05 曾尝试把这里改成"点隐式"推导的动态松弛系数
    （`relax_eff = dt*c_wall/(1+dt*c_wall)`，c_wall 正比于 1/d1^2）
    ——数学上确实排除了显式罚项的刚性超调（另一次已撤销的 SIPG
    尝试），但真实网格验证**再次证伪**：动态 relax_eff 对细网格近壁
    单元（d1 小）天然趋近 1（几乎每步都把 omega 直接怼到 target），
    而 target 本身（哪怕已经被下面 `_compute_omega_wall_target` 的
    `omega_max` 上限保护，不再是失控的 1e14）仍然是 1e6 这个量级的
    "应急上限"，不是"日常合理松弛目标"——把大量边界层单元在几步内
    强行拉到这个量级，会让 D_k=rho*beta_star*k*omega 这个耗散项跟着
    暴涨，2 步内就把全域 k_mean 从 38 打到 0.17（真实数值，不是
    NaN/Inf，但同样是不可接受的物理扰动）。而固定的 `relax=0.5`
    对*所有*单元一视同仁地只走一半路程，天然更温和、给耦合系统留出
    调整时间——这版已用真实生产续算验证 900+ 步保持平均流场零漂移
    （见项目记忆），比"数学上更精确"但经验证更具破坏性的点隐式版本
    更适合作为当前的工程选择。教训：这类近壁松弛的"正确性"不能只看
    单个 ODE 是否无条件稳定，还要看它对耦合场（k 反过来依赖 omega）
    造成的扰动幅度是否温和——本函数改回固定 relax，`dt` 参数保留
    只是为了不破坏调用方签名，不再参与计算。

    Args:
        solver: FRSolver 实例
        dt: 未使用（保留参数位置以兼容调用方签名，见上面"教训"一节）。
        relax: 松弛系数，每步 omega_field[wall_owner] 更新为
            `(1-relax)*old + relax*omega_wall_target`
        flat_face_override: 分布式路径复用同一约定，见
            `compute_turbulence_transport_residual` 同名参数文档
    """
    if relax is None:
        relax = 0.5
    wall_mask = _compute_wall_dirichlet_face_mask(solver)
    if not np.any(wall_mask):
        return

    Q = solver.state.Q
    rho = Q[:, :, 0]
    omega_wall_value_face, has_wall = _compute_omega_wall_target(
        solver, wall_mask, solver.mu_molecular, rho, flat_face_override=flat_face_override,
    )

    flat = flat_face_override if flat_face_override is not None else get_flat_face_geometry(solver.mesh, solver.ops)
    wall_face_idx = np.nonzero(has_wall)[0]
    if len(wall_face_idx) == 0:
        return
    owner_cells = flat.owner_cell[wall_face_idx]
    target = omega_wall_value_face[wall_face_idx, 0]  # 同一面上恒为同一常数，见函数文档

    turb = solver.turb_model
    # 同一个 owner 单元可能是多个 WALL 面的 owner（角部单元）——用
    # np.add.at 累加再除以命中次数取平均目标值，不能直接花式索引赋值
    # 覆盖（后写的面会覆盖先写的面，不是真正的平均）。
    sum_target = np.zeros(solver.state.n_cells)
    count = np.zeros(solver.state.n_cells)
    np.add.at(sum_target, owner_cells, target)
    np.add.at(count, owner_cells, 1.0)
    hit_cells = np.nonzero(count > 0)[0]
    avg_target = sum_target[hit_cells] / count[hit_cells]

    turb.omega_field[hit_cells, :] = (
        (1.0 - relax) * turb.omega_field[hit_cells, :] + relax * avg_target[:, None]
    )


def compute_turbulence_transport_residual(
    solver,
    grad_vel: np.ndarray = None,
    grad_k: np.ndarray = None,
    grad_omega: np.ndarray = None,
    flat_face_override=None,
) -> Tuple[np.ndarray, np.ndarray]:
    """计算 k/omega 的完整输运残差（对流 + 扩散）。

    入口函数：从 solver 获取流场和湍流场信息，分别计算 k 和 omega 的
    对流+扩散残差，返回 dk/dt 和 domega/dt 的输运贡献（已除以密度）。

    Args:
        solver: FRSolver 实例（需要已初始化 SST/DDES 湍流模型）
        grad_vel, grad_k, grad_omega: 可选，调用方（`fr_solver_
            turbulence.compute_turbulence_source`）如果已经算过这三个量，
            直接传进来复用，跳过内部重新计算——性能优化：唯一真实调用方
            `compute_turbulence_source` 在调用本函数*之前*就已经为
            `compute_source_terms` 算过完全相同的 grad_vel/grad_k/
            grad_omega（同一个 solver.state.U/turb_model.k_field/
            omega_field，同一套 mesh/ops，数学上是同一个量），此前这里
            总是无条件重新算一遍——`compute_physical_gradient` 是本项目
            profile 过的真实热点（79万单元 P1 阶段单步 7.5s 累计），这里
            的重复调用是三次里的一次，真实测得省下约 1.6s/步。三者任一
            为 None 时退回原来的内部计算（保持本函数可独立调用的公开
            API 行为不变，不依赖调用方一定会传）。
        flat_face_override: 显式传入时优先使用，透传给内部四次
            `compute_scalar_convection_residual`/`compute_scalar_
            diffusion_residual` 调用（2026-09-02 分布式湍流移植新增，
            见这两个函数同名参数文档）——分布式路径下 `solver` 是
            `DistributedTurbulenceSolverAdapter`（`solver.mesh` 是
            `DistributedMeshAdapter`），必须传入 `dist_fc.base_flat`，
            否则会尝试从压缩索引空间的适配器重新构建全局面几何。

    Returns:
        (dk_dt_transport, domega_dt_transport): 各自 (n_cells, n_sps)，
        输运项对 dk/dt 和 domega/dt 的贡献
    """
    if solver.turb_model is None or not hasattr(solver.turb_model, 'k_field'):
        n_cells, n_sps = solver.state.U.shape[:2]
        return np.zeros((n_cells, n_sps)), np.zeros((n_cells, n_sps))

    turb = solver.turb_model
    Q = solver.state.Q
    rho = Q[:, :, 0]  # (n_cells, n_sps)
    vel = Q[:, :, 1:4]  # (n_cells, n_sps, 3)

    mu = solver.mu_molecular
    rho_nu_t = rho * turb.nu_t  # 动力涡粘度 mu_t = rho * nu_t

    # 计算有效扩散系数 Gamma_k, Gamma_omega
    # 需要 F1 blending 来确定 sigma_k, sigma_omega
    if grad_vel is None:
        # 真实 bug 修复（2026-09-03）：同 fr_solver/turbulence.py::
        # compute_turbulence_source 里的 grad_vel 修复——不能对*守恒*
        # 变量 U 求梯度再切片动量分量冒充速度梯度，见该处文档。这里
        # `Q`/`vel`（上面已经从 solver.state.Q 取出的原始变量）本来就是
        # 正确的速度，直接对它求梯度。
        grad_vel = compute_physical_gradient(vel, solver.mesh, solver.ops)
    S_mag = turb.compute_strain_rate_magnitude(grad_vel)
    nu = mu / np.maximum(rho, 1e-10)

    # 交叉扩散项（F1 计算需要）
    if grad_k is None:
        grad_k = compute_physical_scalar_gradient(turb.k_field, solver.mesh, solver.ops)
    if grad_omega is None:
        grad_omega = compute_physical_scalar_gradient(turb.omega_field, solver.mesh, solver.ops)

    # 梯度幅值裁剪（真实 bug，已修复，2026-08-21）：这里的 grad_k/grad_omega
    # 此前完全没有上限保护——`fr_solver/turbulence.py::compute_turbulence_
    # source` 里给 compute_source_terms 用的那一份 grad_k/grad_omega 早就有
    # 同样的 max_grad_mag=1e6 裁剪（见该文件"正性保持检查"注释），但本函数
    # 参数文档明确说明这里*刻意*不复用那份裁剪后的值、自己独立重新计算，
    # 于是这份独立计算的副本一直没有对应的裁剪。真实复现（cube_demo 生产
    # 网格，P1 阶数，DDES）：mesh 在坍缩坐标+度量退化单元（troubled_cell.py
    # 诊断此网格 P1 阶段 95.15% 单元面法向失配>1度）上，对*理论上处处为
    # 常数*的初始 k/omega 场求梯度，参考空间导数本应恰好为 0，但浮点舍入
    # 误差量级的非零值被 adj(J)/det(J) 这个在退化单元上可以任意大的度量
    # 比值放大到 >1e150（np.linalg.norm 内部计算 x*x 时溢出到 inf，py-spy
    # 采样证实的真实复现）——多数为普通浮点噪声，但间或有值落入次正规数
    # （denormal/subnormal）区间，x86 硬件处理这类数值要走慢得多的微码
    # 路径：单次 `np.sum(grad_k*grad_omega, axis=-1)`（下面这一行）在
    # ~19M 元素规模上因此实测卡住数分钟，而不是正常的毫秒级——是一次
    # "看起来像死锁、实际是每个浮点算子被拖慢几十~上百倍"的真实性能故障，
    # py-spy 对卡住进程的调用栈采样直接定位到本行。与 fr_solver/
    # turbulence.py 用完全相同的裁剪公式（不是发明新阈值，是把已经在
    # 别处验证过、这里唯一遗漏的同一道安全网补齐）。
    # np.linalg.norm 内部对每个分量求平方——在同一类退化单元上分量本身
    # 就已经是溢出级别的量，平方会先于这里的裁剪逻辑触发一次 inf；
    # errstate 只是抑制这一步的警告噪音，紧接着的 np.maximum(...,1e-10)/
    # np.clip(...,0,1) 已经能正确处理 inf 输入（inf>max_grad_mag 恒真，
    # scale=max_grad_mag/inf=0，裁剪结果趋于 0，不是 nan），不依赖这个
    # errstate 才能得到正确结果。
    with np.errstate(over='ignore', invalid='ignore'):
        max_grad_mag = 1e6
        grad_k_mag = np.linalg.norm(grad_k, axis=-1)
        grad_omega_mag = np.linalg.norm(grad_omega, axis=-1)
        if np.any(grad_k_mag > max_grad_mag):
            scale_k = max_grad_mag / np.maximum(grad_k_mag, 1e-10)
            grad_k = grad_k * np.clip(scale_k, 0, 1)[..., None]
        if np.any(grad_omega_mag > max_grad_mag):
            scale_omega = max_grad_mag / np.maximum(grad_omega_mag, 1e-10)
            grad_omega = grad_omega * np.clip(scale_omega, 0, 1)[..., None]

        grad_dot = np.sum(grad_k * grad_omega, axis=-1)
        omega_safe = np.maximum(turb.omega_field, 1e-10)
        CD_kw = np.maximum(2.0 * rho * turb.sigma_w2 / omega_safe * grad_dot, 1e-10)

    F1 = turb.compute_blending_function_F1(
        turb.k_field, turb.omega_field, solver.wall_distance, nu, S_mag, rho, CD_kw
    )

    sigma_k = F1 * turb.sigma_k1 + (1.0 - F1) * turb.sigma_k2
    sigma_w = F1 * turb.sigma_w1 + (1.0 - F1) * turb.sigma_w2

    gamma_k = mu + sigma_k * rho_nu_t    # (n_cells, n_sps)
    gamma_w = mu + sigma_w * rho_nu_t    # (n_cells, n_sps)

    # WALL 上 k=0 的 Dirichlet 掩码（真实修复，2026-08-21，见
    # transport_kernel.py::extrapolate_scalar_to_faces_kernel 文档）。
    # 数值作用点：对流项的上风 phi ghost（镜像成 -owner 强制壁面 k=0）；
    # 扩散项自 2026-08-25 校正改梯度差形式后该掩码不再有数值影响（奇镜像
    # 对梯度对称，见 compute_scalar_diffusion_residual 参数文档），仍传入
    # 以保持接口一致。
    wall_mask_k = _compute_wall_dirichlet_face_mask(solver)

    # 共享几何量（性能优化 2026-09-13，见 `ScalarConvectionGeometry` 文档）：
    # k 与 omega 的对流调用此前各自重复算了一遍与标量无关的逆变质量通量
    # 和面上 mass_flux，这里统一算一次传给两者。
    _flat_conv = (flat_face_override if flat_face_override is not None
                  else get_flat_face_geometry(solver.mesh, solver.ops))
    conv_geom = precompute_scalar_convection_geometry(
        rho, vel, solver.mesh, solver.ops, _flat_conv,
    )

    # 计算 k 的对流 + 扩散残差
    conv_k = compute_scalar_convection_residual(
        turb.k_field, rho, vel, solver.mesh, solver.ops, wall_dirichlet_zero_face=wall_mask_k,
        flat_face_override=flat_face_override, conv_geom=conv_geom,
    )
    diff_k = compute_scalar_diffusion_residual(
        turb.k_field, gamma_k, solver.mesh, solver.ops, wall_dirichlet_zero_face=wall_mask_k,
        flat_face_override=flat_face_override,
    )
    with np.errstate(over='ignore', invalid='ignore'):
        dk_dt_transport = (conv_k + diff_k) / np.maximum(rho, 1e-10)

    # WALL 上 omega 解析壁面值的 Dirichlet 目标（真实修复，V2.0 专家组
    # 盲审发现，2026-08-28，见 _compute_omega_wall_target 文档）：此前
    # omega 恒用 Neumann（零梯度）默认，是明确记录过的已知限制——现在
    # 用 Wilcox 解析式 60*nu/(beta1*d1^2) 代替。d1 需要 solver.wall_
    # distance 已经计算好（SST/DDES/WMLES 初始化时必然如此，见
    # fr_solver/turbulence.py），否则 _compute_omega_wall_target 里的
    # np.min(solver.wall_distance[...]) 会直接因 wall_distance 为 None
    # 报错——这是有意的（没有壁面距离场，压根不该假装能算出解析壁面值）。
    omega_wall_value_face, has_omega_wall = _compute_omega_wall_target(
        solver, wall_mask_k, mu, rho, flat_face_override=flat_face_override,
    )

    # 计算 omega 的对流 + 扩散残差
    conv_w = compute_scalar_convection_residual(
        turb.omega_field, rho, vel, solver.mesh, solver.ops,
        wall_dirichlet_value_face=omega_wall_value_face, has_wall_dirichlet_value=has_omega_wall,
        flat_face_override=flat_face_override, conv_geom=conv_geom,
    )
    diff_w = compute_scalar_diffusion_residual(
        turb.omega_field, gamma_w, solver.mesh, solver.ops,
        wall_dirichlet_value_face=omega_wall_value_face, has_wall_dirichlet_value=has_omega_wall,
        flat_face_override=flat_face_override,
    )
    with np.errstate(over='ignore', invalid='ignore'):
        domega_dt_transport = (conv_w + diff_w) / np.maximum(rho, 1e-10)

    # 机制3（症状检测，2026-08-22）：退化单元（坍缩坐标/BL 挤出，见
    # fr_operators/troubled_cell.py 模块文档）上本函数算出的残差可能
    # 出现量级异常（真实复现：cube_demo 生产网格 P0->P1 切换后，
    # transport.py 内部多处除以 det(J) 的地方溢出到 inf，见本文件
    # 上方的 errstate 注释）——平均流残差（inviscid.py/viscous_flux.py）
    # 早就用 suppress_residual_outliers 处理同一类问题（"取代此前先
    # 用 det(J)/法向失配几何量预判、按整个单元降阶的机制1/2"，见
    # troubled_cell.py 文档"机制3"一节），本函数此前一直没有接入这套
    # 机制，只在最后做一次朴素的 isfinite 归零——两者不冲突：
    # suppress_residual_outliers 用同单元其余 SP 的残差中位数做参照，
    # 能捕捉"明显偏大但还是有限值"的异常（isfinite 捕捉不到这类），
    # 按 (cell,SP) 粒度清零，不牵连同一单元里其余健康 SP；下面的
    # isfinite 归零保留作最后一道防线（例如整个单元所有 SP 都异常、
    # 中位数参照本身也失真的极端情形）。
    dk_dt_transport = suppress_residual_outliers(
        dk_dt_transport[:, :, None], turb.k_field[:, :, None]
    )[:, :, 0]
    domega_dt_transport = suppress_residual_outliers(
        domega_dt_transport[:, :, None], turb.omega_field[:, :, None]
    )[:, :, 0]

    # NaN/Inf 隔离（最后一道防线）：退化网格上梯度/Jacobian 可能产生非
    # 有限值，归零后由 SST.update_fields 的二次防护和 positivity
    # limiter 接管
    dk_dt_transport = np.where(np.isfinite(dk_dt_transport), dk_dt_transport, 0.0)
    domega_dt_transport = np.where(np.isfinite(domega_dt_transport), domega_dt_transport, 0.0)

    return dk_dt_transport, domega_dt_transport
