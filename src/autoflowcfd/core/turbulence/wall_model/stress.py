"""WMLES 壁面剪应力：匹配点 LES 解 -> 壁面律 -> 壁面牵引 -> 动量残差（与壁面力积分共用同一个牵引）。

WMLES 激活时物理无滑移壁面的幽灵态取滑移镜像（`core/fr_solver/boundary/ghost.py`：WALL 组 is_no_slip=False），
粘性公共通量在壁面上的切向牵引为零，壁面剪应力只来自这里——壁面模型的 tau_w 取代而不是叠加在解析梯度剪应力上
（Kawai & Larsson 2012）。

数组模块无关（`xp` 为 numpy 或 cupy）；匹配点几何（`matching.py`）在构造时算好并放到该后端的设备上。
"""

from autoflowcfd.core.utils.array_module import scatter_add

from .law import wall_shear_stress


def wall_traction(xp, matching, U, mu: float, kappa: float, B: float):
    """壁面通量点上壁面对流体的牵引 `(m, 3)` 与 y+ `(m,)`。

    匹配点速度取 DG 解在该点的值：守恒变量多项式插值到匹配点后 `u = (rho u) / rho`。

    Args:
        matching: `WallMatching`（数组已在 xp 所在的设备上）
        U: 守恒变量 `(n_cells, n_sps, >=5)`
        mu: 分子动力粘度
    """
    U_m = xp.einsum('ks,ksv->kv', matching.E_m, U[matching.owner][..., :5])
    rho = U_m[:, 0]
    vel = U_m[:, 1:4] / rho[:, None]
    n = matching.normal
    u_t = vel - xp.sum(vel * n, axis=-1, keepdims=True) * n
    return wall_shear_stress(xp, u_t, matching.h_wm, rho, mu / rho, kappa, B)


def wall_stress_residual(xp, matching, traction, n_cells: int, n_sps: int):
    """壁面牵引对平均流动量残差的贡献 `(n_cells, n_sps, 5)`（只有动量分量非零）。

    与粘性公共通量同一个约定：动量通量 = 物理面积权重 x 牵引，经 owner 面提升算子的该通量点列分布到解点、
    除 det(J)。
    """
    flux = matching.area[:, None] * traction                                      # (m, 3)
    contrib = (matching.lift * matching.inv_det)[:, :, None] * flux[:, None, :]    # (m, n_sps, 3)
    out = xp.zeros((n_cells, n_sps, 5))
    scatter_add(xp, out, (matching.owner, slice(None), slice(1, 4)), contrib)
    return out
