"""AutoFlowCFD V2.0 - 湍流标量的**对流**残差（含去混叠体积项）。

从 `core/turbulence/transport/` 拆出（2026-09-24）；界面项 2026-09-26 改为两侧各自坐标系（见 `face_frames.py`）。

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

from typing import Optional
import numpy as np

from numba import njit, prange

from autoflowcfd.core.fr_operators.volume_contract import (
    cell_blocks, contract_local, contravariant_point, fine_segments, get_overintegration_context,
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
    inflow_points,
    precompute_scalar_convection_geometry,
)
from .face_frames import convection_face_jumps_kernel

#: 没有开放边界时传给 `convection_face_jumps_kernel` 的来流点掩码。
_NO_INFLOW = np.zeros((0, 0), dtype=np.bool_)


#: k/omega 输运体积项恒走过积分（去混叠）：对流通量 `adj(J) rho u phi` 是非线性乘积，真实多项式次数约
#: `3*order + 度量次数`，直接在 coarse SPs 上微分就是"先混叠再求导"（实测 P1 体积项 33 倍改善、P2 到机器零，
#: 单步代价约 +10%）。P0 没有过积分几何（`get_overintegration_context` 返回 None），走单点路径。


@njit(cache=True, parallel=True)
def _convection_volume_kernel(phi, rho, velocity, lo, c2f, det_f, inv_f, D2, f2c, out, n_blocks):
    """段内单元 `lo + i`：`out[c, :, v] = f2c D_fine (adj(J) rho u phi_v)`（`phi` 的各列独立）。"""
    n = det_f.shape[0]
    n_fine, n_sps = c2f.shape
    V = phi.shape[2]
    for b in prange(n_blocks):
        r_c = np.empty((n_sps, 1))
        phi_f = np.empty((n_fine, V))
        rho_f = np.empty((n_fine, 1))
        u_f = np.empty((n_fine, 3))
        f_p = np.empty((3, V))
        f_t = np.empty((3 * n_fine, V))
        div_f = np.empty((n_fine, V))
        for i in range(b * n // n_blocks, (b + 1) * n // n_blocks):
            c = lo + i
            for s in range(n_sps):
                r_c[s, 0] = rho[c, s]
            contract_local(c2f, phi[c], phi_f)
            contract_local(c2f, r_c, rho_f)
            contract_local(c2f, velocity[c], u_f)
            for p in range(n_fine):
                for j in range(3):
                    for v in range(V):
                        f_p[j, v] = rho_f[p, 0] * phi_f[p, v] * u_f[p, j]
                contravariant_point(det_f[i, p], inv_f[i, p], f_p, f_t[3 * p:3 * p + 3])
            contract_local(D2, f_t, div_f)
            contract_local(f2c, div_f, out[c])


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
    逐单元在块内局部工作数组里走完（理由见 `fr_residual/inviscid.py::inviscid_volume_divergence`
    文档"逐单元融合"），与按块分配临时数组的旧版逐位相同。
    """
    phi = np.ascontiguousarray(scalar_field[:, :, None], dtype=np.float64)
    out = np.empty((scalar_field.shape[0], n_sps, 1))
    for lo, hi, c2f, det_f, inv_f, D2, f2c in fine_segments(oi):
        _convection_volume_kernel(phi, rho, velocity, lo, c2f, det_f, inv_f, D2, f2c, out, cell_blocks(hi - lo))
    return out[:, :, 0]


def scalar_convection_volume_divergence(scalar_field, rho, velocity, rho_u_tilde, mesh, ops):
    """对流体积算子 `div_vol(adj(J) rho u phi)`（参考空间、除 det 之前），`(n_cells, n_sps)`。

    残差（`phi`）与 `ScalarConvectionGeometry.mass_divergence`（`phi = 1`）共用这一个函数，
    保证对流形式里相减的两项是同一个离散算子。native 四面体用零填充到全局宽度的
    `D_native_tet_padded`；去混叠见模块级说明。
    """
    n_cells = mesh.n_cells
    n_sps = mesh.n_sps_per_cell
    n_prism = mesh.n_prism_cells
    oi = get_overintegration_context(mesh, ops)
    if oi is not None:
        return _scalar_convection_volume_overintegrated(scalar_field, rho, velocity, oi, n_sps)
    tet_op_D = ops.D_native_tet_padded
    div_F = np.empty((n_cells, n_sps))
    if n_prism > 0:
        scalar_convection_volume_kernel(
            np.ascontiguousarray(scalar_field[:n_prism]), np.ascontiguousarray(rho_u_tilde[:n_prism]),
            np.ascontiguousarray(ops.D_native_prism_padded), div_F[:n_prism])
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
    wall_dirichlet_zero_face: Optional[np.ndarray] = None,
    wall_dirichlet_value_face: Optional[np.ndarray] = None,
    has_wall_dirichlet_value: Optional[np.ndarray] = None,
    flat_face_override=None,
    conv_geom: "Optional[ScalarConvectionGeometry]" = None,
    open_boundary_face: Optional[np.ndarray] = None,
    freestream_value: Optional[float] = None,
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
        open_boundary_face, freestream_value: 来流条件（2026-09-25）：开放边界组的逐面掩码（按组类型，
            `omega_wall.py::open_boundary_code_mask`）与来流值。两侧坐标系的边界点（含混合拆分面的边界半区）上质量
            通量指向域内的通量点，外部态取来流值；指向域外保持外插（零梯度出流），见 `faces.py::inflow_points`。
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
    # 逐点规则即 face_frames.convection_jump_point（湍流解析 Jacobian 用同一个点函数）
    if open_boundary_face is not None and freestream_value is None:
        raise ValueError("给出开放边界面（open_boundary_face）时必须同时给出来流值 freestream_value")
    masks = (wall_dirichlet_zero_face, wall_dirichlet_value_face, has_wall_dirichlet_value)
    fs = 0.0 if freestream_value is None else float(freestream_value)
    phi_o, phi_o_other = _extrapolate_scalar_to_faces(scalar_field, flat, ops, mesh, *masks)
    m_o = conv_geom.mass_flux
    inflow_o = _NO_INFLOW if open_boundary_face is None else inflow_points(np, flat, "owner", open_boundary_face, m_o)
    jump_o = convection_face_jumps_kernel(m_o, phi_o, phi_o_other, inflow_o, fs)
    del phi_o, phi_o_other, inflow_o

    phi_n, phi_n_other = _extrapolate_scalar_to_faces_neighbor_frame(scalar_field, flat, *masks)
    m_n = conv_geom.mass_flux_neighbor
    inflow_n = (_NO_INFLOW if open_boundary_face is None
                else inflow_points(np, flat, "neighbor", open_boundary_face, m_n))
    jump_n = convection_face_jumps_kernel(m_n, phi_n, phi_n_other, inflow_n, fs)
    del phi_n, phi_n_other, inflow_n

    with np.errstate(over='ignore', invalid='ignore'):
        return residual + _lift_side_jumps(jump_o, jump_n, -1.0, flat, mesh)
