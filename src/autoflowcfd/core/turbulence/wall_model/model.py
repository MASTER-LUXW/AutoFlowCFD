"""平衡壁面模型对象（WMLES 与 RANS 自动壁面处理共用）：选壁面通量点、缓存匹配点几何、给出壁面牵引与动量残差修正。

只作用于**物理无滑移**壁面（WALL 组且 `physical_no_slip`）；SLIP_WALL（隧道壁、远场滑移面，幽灵态同样是
滑移镜像）没有壁面剪应力。移动壁面（组配置的 `wall_velocity`，例如移动地面）按流体相对壁面的切向速度取壁面律，
剪应力对流体做的功进能量方程（`stress.py`）。

这些壁面的幽灵态必须是滑移镜像（`is_no_slip=False`，`core/fr_solver/boundary/ghost.py` 在有壁面模型时如此构造）：
无滑移幽灵态下粘性公共通量本身已带壁面剪应力，再加壁面模型的牵引就是重复计入。用户 `bc_overrides` 显式覆盖成
无滑移幽灵态时直接报错。
"""

from typing import Optional

import numpy as np

from .law import B_LOG, KAPPA
from .matching import WallMatching, build_wall_matching, cell_integration_weights
from .stress import wall_stress_residual, wall_traction, wall_traction_jacobian


def wall_model_codes(provider) -> list:
    """施加壁面模型的边界组编码：物理无滑移的 WALL 组（幽灵态必须是滑移镜像，见模块文档）。"""
    from autoflowcfd.boundary.wall_kind import is_physical_no_slip_wall

    codes = [code for code, cfg in provider.code_to_config.items() if is_physical_no_slip_wall(cfg)]
    names = dict(getattr(provider, "code_to_name", {}) or {})
    no_slip_ghost = [names.get(c, str(c)) for c in codes if provider.code_to_config[c].get("is_no_slip", True)]
    if no_slip_ghost:
        raise ValueError(
            f"边界组 {no_slip_ghost} 配置为无滑移幽灵态（is_no_slip=True），同时启用了平衡壁面模型（WMLES 或 "
            f"--wall-treatment auto）：壁面剪应力会被计入两次。壁面模型下壁面幽灵态应为滑移镜像（is_no_slip=False、"
            f"physical_no_slip=True），或改用 --wall-treatment resolved")
    return codes


def _wall_velocities(provider, code) -> np.ndarray:
    """逐壁面通量点的壁面速度 `(m, 3)`（所属组配置的 `wall_velocity`，没有该项为静止壁）。"""
    out = np.zeros((len(code), 3))
    for c in np.unique(code):
        wv = provider.code_to_config[int(c)].get("wall_velocity")
        if wv is not None:
            out[code == c] = np.asarray(wv, dtype=np.float64)
    return out


class EquilibriumWallModel:
    """Spalding 壁面律的平衡壁面模型（kappa = 0.41、B = 5.2）。"""

    def __init__(self, kappa: float = KAPPA, B: float = B_LOG):
        self.kappa = kappa
        self.B = B
        self.y_plus = None
        # 缓存键持有面几何对象本身（不是 id()：旧对象被回收后 id 可能被新对象复用）。面几何随网格/阶数
        # 重建，换阶数后自动重算。主机版本构造一次，各数组模块的版本由它搬运（GPU 求解器的残差用设备版本、
        # 壁面力积分用主机版本，两者不互相驱逐）
        self._cache_flat = None
        self._cache = {}

    def matching(self, xp, mesh, flat, provider) -> Optional[WallMatching]:
        """匹配点几何（只依赖网格几何）；没有物理无滑移壁面时为 None。"""
        if self._cache_flat is not flat:
            self._cache_flat, self._cache = flat, {"numpy": self._build(mesh, flat, provider)}
        host = self._cache["numpy"]
        if host is None:
            return None
        if xp.__name__ not in self._cache:
            self._cache[xp.__name__] = host.to(xp)
        return self._cache[xp.__name__]

    @staticmethod
    def _build(mesh, flat, provider) -> Optional[WallMatching]:
        from autoflowcfd.core.fr_residual.wall_points import wall_flux_points

        points = wall_flux_points(flat, np.asarray(provider.group_code), wall_model_codes(provider))
        if points[0].size == 0:
            return None
        W, det, cell_is_prism, order = cell_integration_weights(mesh)
        return build_wall_matching(mesh, flat, points, cell_is_prism, W, det, order,
                                   wall_velocity=_wall_velocities(provider, points[3]))

    def traction(self, xp, mesh, flat, provider, U, mu: float):
        """(匹配点几何, 壁面对流体的牵引 (m, 3))；没有物理无滑移壁面时 (None, None)。"""
        m = self.matching(xp, mesh, flat, provider)
        if m is None:
            return None, None
        tau, self.y_plus = wall_traction(xp, m, U, mu, self.kappa, self.B)
        return m, tau

    def traction_jacobian(self, xp, mesh, flat, provider, U, mu: float):
        """(匹配点几何, 牵引对匹配点守恒变量的导数 (m, 3, 5))；没有物理无滑移壁面时 (None, None)。"""
        m = self.matching(xp, mesh, flat, provider)
        if m is None:
            return None, None
        return m, wall_traction_jacobian(xp, m, U, mu, self.kappa, self.B)

    def residual_correction(self, xp, mesh, flat, provider, U, mu: float):
        """壁面剪应力对平均流残差的贡献 `(n_cells, n_sps, 5)`（动量；移动壁面另有能量）；没有物理无滑移壁面时 None。"""
        m, tau = self.traction(xp, mesh, flat, provider, U, mu)
        if m is None:
            return None
        n_cells, n_sps = U.shape[:2]
        return wall_stress_residual(xp, m, tau, n_cells, n_sps)
