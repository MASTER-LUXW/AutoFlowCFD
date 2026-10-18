"""平衡壁面模型的壁面剪应力：匹配点 LES 解 -> 壁面律 -> 壁面牵引 -> 动量残差（与壁面力积分共用同一个牵引）。

壁面模型激活时（WMLES，或 RANS 自动壁面处理 `treatment.py`）物理无滑移壁面的幽灵态取滑移镜像（`core/fr_solver/boundary/ghost.py`：WALL 组 is_no_slip=False），
粘性公共通量在壁面上的切向牵引为零，壁面剪应力只来自这里——壁面模型的 tau_w 取代而不是叠加在解析梯度剪应力上
（Kawai & Larsson 2012）。平均流解析单元块（`fr_residual/jacobian/wall_stress.py`）用 `wall_traction_jacobian`。

数组模块无关（`xp` 为 numpy 或 cupy）；匹配点几何（`matching.py`）在构造时算好并放到该后端的设备上。
"""

from autoflowcfd.core.utils.array_module import scatter_add

from .law import wall_shear_stress, wall_shear_stress_linearization


def wall_traction(xp, matching, U, mu: float, kappa: float, B: float):
    """壁面通量点上壁面对流体的牵引 `(m, 3)` 与 y+ `(m,)`。

    匹配点速度取 DG 解在该点的值：守恒变量多项式插值到匹配点后 `u = (rho u) / rho`；壁面律用流体相对壁面的
    切向速度 `u_t = P (u - u_wall)`（移动壁面，静止壁 u_wall = 0）。

    Args:
        matching: `WallMatching`（数组已在 xp 所在的设备上）
        U: 守恒变量 `(n_cells, n_sps, >=5)`
        mu: 分子动力粘度
    """
    rho, vel, u_t = _matching_velocity(xp, matching, U)
    return wall_shear_stress(xp, u_t, matching.h_wm, rho, mu / rho, kappa, B)


def _matching_velocity(xp, matching, U):
    """匹配点的 (rho (m,), 速度 (m, 3), 相对壁面的切向速度 (m, 3))。"""
    U_m = xp.einsum('ks,ksv->kv', matching.E_m, U[matching.owner][..., :5])
    rho = U_m[:, 0]
    vel = U_m[:, 1:4] / rho[:, None]
    rel = vel - matching.wall_velocity
    n = matching.normal
    u_t = rel - xp.sum(rel * n, axis=-1, keepdims=True) * n
    return rho, vel, u_t


def wall_traction_jacobian(xp, matching, U, mu: float, kappa: float, B: float):
    """壁面牵引对匹配点守恒变量的导数 `dtau/dU_m (m, 3, 5)`（解析，`law.wall_shear_stress_linearization`）。

    u_t = P (u - u_wall)（P = I - n n^T），u = (rho u)/rho：dtau/d(rho u) = (dtau/du_t) P / rho，
    dtau/drho = dtau/drho|_u - (dtau/du_t) P u / rho；牵引与能量无关。对 owner 单元解点的导数为
    `dtau/dU_m (x) E_m`（`E_m` 为解点 -> 匹配点插值行），只进 owner 单元的对角块。
    """
    rho, vel, u_t = _matching_velocity(xp, matching, U)
    _, dtau_du, dtau_drho = wall_shear_stress_linearization(xp, u_t, matching.h_wm, rho, mu, kappa, B)
    n = matching.normal
    P = xp.eye(3) - n[:, :, None] * n[:, None, :]
    dtau_dvel = xp.einsum('kij,kjl->kil', dtau_du, P)
    J = xp.zeros((rho.shape[0], 3, 5))
    J[:, :, 0] = dtau_drho - xp.einsum('kij,kj->ki', dtau_dvel, vel) / rho[:, None]
    J[:, :, 1:4] = dtau_dvel / rho[:, None, None]
    return J


def wall_flux(xp, matching, traction):
    """壁面通过面积权重传给流体的动量与能量通量 `(m, 4)`：`area * [tau, tau . u_wall]`（移动壁面上剪应力对流体
    做功；静止壁能量分量为 0，与绝热无滑移壁一致）。"""
    power = xp.sum(traction * matching.wall_velocity, axis=-1, keepdims=True)
    return matching.area[:, None] * xp.concatenate([traction, power], axis=-1)


def wall_stress_residual(xp, matching, traction, n_cells: int, n_sps: int):
    """壁面牵引对平均流残差的贡献 `(n_cells, n_sps, 5)`（动量与能量分量，`wall_flux`）。

    与粘性公共通量同一个约定：通量经 owner 面提升算子的该通量点列分布到解点、除 det(J)。
    """
    contrib = (matching.lift * matching.inv_det)[:, :, None] * wall_flux(xp, matching, traction)[:, None, :]
    out = xp.zeros((n_cells, n_sps, 5))
    scatter_add(xp, out, (matching.owner, slice(None), slice(1, 5)), contrib)
    return out
