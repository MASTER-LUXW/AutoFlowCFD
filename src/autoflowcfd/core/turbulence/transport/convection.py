"""AutoFlowCFD V2.0 - 湍流标量的**对流**残差（含去混叠体积项）。

从 `core/turbulence/transport.py` 拆出（2026-09-24）；界面项 2026-09-26 改为两侧各自坐标系（见 `face_frames.py`）。

残差约定：`-rho U . grad(phi)` 的 FR 离散（体积项 + 界面上风校正），返回
`rho dphi/dt` 的对流部分，见包 `__init__.py` 的"符号约定"一节。

## 体积项取对流形式（2026-10-02）

被求解的方程是 `rho dphi/dt + rho U . grad(phi) = ...`（未知量是 k 与 `w = ln omega`
本身），界面项是对流形式的跳变量 `m (phi_up - phi_side)`（均匀场恒为零）。体积项
此前是守恒形式 `-div_vol(rho U phi)`，与对流形式只差 `-phi div_vol(rho U)`——离散体积
散度逐点为零时才消失，而高阶 FR 的平均流只在"体积散度 + 界面修正"整体上满足连续性。
所以体积项取

    div_vol(rho U phi) - phi div_vol(rho U)

（同一个体积算子分别作用在 `phi` 与 `1` 上，后者只依赖冻结平均流，见
`ScalarConvectionGeometry.mass_divergence`）：均匀场与常数平移逐位保持。

实测（plate_demo P1，b283352 第 120 步检查点）：守恒形式下均匀 `phi = c` 的对流残差
`/rho` 在平板前缘锐边棱柱上是 `+8020 c` 1/s（严格正比于 c）；`w = ln omega` 约 13.8，伪源
+1.1e5 1/s 压过 omega 耗散（-8.3e4），123 个解点被推到 `omega_max` 钳位，钳位处残差
不可微，JFNK 湍流 GMRES 每步跑满 200 次。全场 99 分位也有 1.7e4 1/s。
"""

import os
import numpy as np


from autoflowcfd.core.fr_operators.volume_contract import (
    OVERINT_CHUNK_CELLS, contract_shared_operator_1axis,
    contract_shared_operator_2axis, contravariant_flux_from_metric,
    get_overintegration_context,
)
from autoflowcfd.core.fr_operators.face_kernels import get_flat_face_geometry
from autoflowcfd.core.turbulence.transport_kernel import (
    scalar_convection_volume_kernel,
)

from .faces import (
    ScalarConvectionGeometry,
    _extrapolate_scalar_to_faces,
    _extrapolate_scalar_to_faces_neighbor_frame,
    _lift_side_jumps,
    precompute_scalar_convection_geometry,
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
    div_F = np.empty((n_cells, n_sps))
    # 每段自带自己的 n_fine 与已切好的细点度量（2026-09-17）：native
    # 四面体的过积分细网格轴不再填充到棱柱的 (oo+1)^3 宽度，两段的
    # n_fine 不同了。度量按**段内局部**索引切（`i0 = c0 - seg_lo`）——
    # 用全局 c0 去切段内数组会静默取到错误的单元。
    for (seg_lo, seg_hi, n_fine, det_seg, inv_seg,
         op_c2f, op_D_fine, op_f2c) in oi["segs"]:
        for c0 in range(seg_lo, seg_hi, _TURB_OVERINT_CHUNK_CELLS):
            c1 = min(c0 + _TURB_OVERINT_CHUNK_CELLS, seg_hi)
            i0, i1 = c0 - seg_lo, c1 - seg_lo
            phi_f = contract_shared_operator_1axis(
                op_c2f, np.ascontiguousarray(scalar_field[c0:c1, :, None]))
            rho_f = contract_shared_operator_1axis(
                op_c2f, np.ascontiguousarray(rho[c0:c1, :, None]))
            u_f = contract_shared_operator_1axis(
                op_c2f, np.ascontiguousarray(velocity[c0:c1]))       # (n,n_fine,3)
            # rho*u*phi 在 FINE 点上相乘（去混叠的核心）
            F_phys_f = (rho_f * phi_f * u_f)[..., None]              # (n,n_fine,3,1)
            del phi_f, rho_f, u_f
            # ascontiguousarray：段内度量是带偏移的非连续视图
            # （`det_all[n_prism:, :n_fine_tet]`），numba kernel 要连续输入
            F_tilde_f = contravariant_flux_from_metric(
                det_seg[i0:i1], inv_seg[i0:i1], F_phys_f)  # 度量视图直接传（见 contravariant_flux_from_metric）
            del F_phys_f
            div_f = contract_shared_operator_2axis(op_D_fine, F_tilde_f)  # (n,n_fine,1)
            del F_tilde_f
            div_F[c0:c1] = contract_shared_operator_1axis(op_f2c, div_f)[..., 0]
            del div_f
    return div_F


def scalar_convection_volume_divergence(scalar_field, rho, velocity, rho_u_tilde, mesh, ops):
    """对流体积算子 `div_vol(adj(J) rho u phi)`（参考空间、除 det 之前），`(n_cells, n_sps)`。

    残差（`phi`）与 `ScalarConvectionGeometry.mass_divergence`（`phi = 1`）共用这一个函数，
    保证对流形式里相减的两项是同一个离散算子。native 四面体用零填充到全局宽度的
    `D_native_tet_padded`；去混叠见 `resolve_turb_overintegration`。
    """
    n_cells = mesh.n_cells
    n_sps = mesh.n_sps_per_cell
    n_prism = mesh.n_prism_cells
    oi = _turb_overint_ops(mesh, ops) if resolve_turb_overintegration() == "on" else None
    if oi is not None:
        return _scalar_convection_volume_overintegrated(scalar_field, rho, velocity, oi, n_sps)
    tet_op_D = ops.D_native_tet_padded if getattr(ops, "D_native_tet_padded", None) is not None else ops.D_3d_tet
    div_F = np.empty((n_cells, n_sps))
    if n_prism > 0:
        scalar_convection_volume_kernel(
            np.ascontiguousarray(scalar_field[:n_prism]), np.ascontiguousarray(rho_u_tilde[:n_prism]),
            np.ascontiguousarray(ops.D_3d_prism), div_F[:n_prism])
    if n_cells > n_prism:
        scalar_convection_volume_kernel(
            np.ascontiguousarray(scalar_field[n_prism:]), np.ascontiguousarray(rho_u_tilde[n_prism:]),
            np.ascontiguousarray(tet_op_D), div_F[n_prism:])
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
    open_boundary_face: np.ndarray = None,
    freestream_value: float = None,
) -> np.ndarray:
    """标量对流 FR 残差 `-rho U . grad(phi)`（对流形式体积项 + 界面上风校正，见模块
    文档），返回 `rho dphi/dt` 的对流部分（调用方再除以 rho）。

    ## 界面项

    两侧各在自己的通量点顺序里构造跳变量（`faces.py` / `face_frames.py`）：

        J_side = m_side * (phi_upwind - phi_side)，   phi_upwind = phi_side 若 m_side >= 0
                                                                  phi_other 否则

    `m_side` 是该侧外法向上的质量通量，两侧都取 owner 的迹（同一物理点上
    `m_neighbor = -m_owner`，公共通量单值、上风选择一致），提升符号 -1。

    "自身通量"用 `m * phi_side` 而不是 `(rho*u)_side . n * phi_side`（2026-09-12 的
    设计，记录在此避免重蹈）：k/omega 是附着在瞬时并不精确满足连续性的密度场
    上的被动标量，后者对局部质量守恒残差敏感——均匀标量场在真实网格上会给出
    高达 451 的伪残差（应为 0）。前者对均匀场恒为零。

    Args:
        scalar_field: (n_cells, n_sps) 标量（k 或 omega）
        rho: (n_cells, n_sps)；velocity: (n_cells, n_sps, 3)
        wall_dirichlet_zero_face: (n_faces,) bool，WALL 上 k=0 的镜像 ghost。
        wall_dirichlet_value_face, has_wall_dirichlet_value: omega 壁面解析值
            ghost（与上一个互斥），见 `faces._extrapolate_scalar_to_faces`。
        flat_face_override: 分布式路径的 `DistributedFlatFaceGeometry`（其
            `mesh` 是 `DistributedMeshAdapter`，不能再调 `get_flat_face_geometry`）。
        conv_geom: 与标量无关的共享几何（k/omega 两次调用共用，隐式路径在 Newton
            步起点算一次），缺省时当场算。
        open_boundary_face, freestream_value: 来流条件（2026-09-25）：开放边界面上
            质量通量指向域内的通量点，外部态取来流值；指向域外保持外插（零梯度出流）。
    """
    n_cells = mesh.n_cells
    n_sps = mesh.n_sps_per_cell
    det_jacs = mesh.jacobians["det_jacs"].reshape(n_cells, n_sps)
    flat = flat_face_override if flat_face_override is not None else get_flat_face_geometry(mesh, ops)
    if conv_geom is None:
        conv_geom = precompute_scalar_convection_geometry(rho, velocity, mesh, ops, flat)

    # === 体积项（对流形式，见模块文档）===
    div_F = scalar_convection_volume_divergence(scalar_field, rho, velocity, conv_geom.rho_u_tilde, mesh, ops)
    div_F -= scalar_field * conv_geom.mass_divergence
    # 退化单元上 1/det(J) 可以溢出；非有限值由 `compute_turbulence_transport_residual`
    # 末尾统一清零，这里只抑制警告噪音。
    with np.errstate(over='ignore', invalid='ignore'):
        residual = -div_F / det_jacs
    del div_F

    # === 界面项（两侧各自坐标系）===
    masks = (wall_dirichlet_zero_face, wall_dirichlet_value_face, has_wall_dirichlet_value)
    phi_o, phi_o_other = _extrapolate_scalar_to_faces(scalar_field, flat, ops, mesh, *masks)
    m_o = conv_geom.mass_flux
    if open_boundary_face is not None:
        inflow = open_boundary_face[:, None] & (m_o < 0)
        phi_o_other = np.where(inflow, freestream_value, phi_o_other)
    # 与 face_frames.convection_jump_point 同一规则（湍流解析 Jacobian 用那个点函数）
    jump_o = m_o * (np.where(m_o >= 0, phi_o, phi_o_other) - phi_o)
    del phi_o, phi_o_other

    phi_n, phi_n_other = _extrapolate_scalar_to_faces_neighbor_frame(scalar_field, flat, *masks)
    m_n = conv_geom.mass_flux_neighbor
    jump_n = m_n * (np.where(m_n >= 0, phi_n, phi_n_other) - phi_n)
    del phi_n, phi_n_other

    with np.errstate(over='ignore', invalid='ignore'):
        return residual + _lift_side_jumps(jump_o, jump_n, -1.0, flat, mesh)
