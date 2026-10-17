"""WMLES 壁面模型对象：选壁面通量点、缓存匹配点几何、给出壁面牵引与动量残差修正。

只作用于**物理无滑移**壁面（WALL 组且 `physical_no_slip`）；SLIP_WALL（隧道壁、远场滑移面，幽灵态同样是
滑移镜像）没有壁面剪应力。此前的实现对两者一视同仁，WMLES 算例在滑移远场上也施加了壁面剪应力。
"""

from typing import Optional

import numpy as np

from .law import B_LOG, KAPPA
from .matching import WallMatching, build_wall_matching
from .stress import wall_stress_residual, wall_traction


def wall_model_codes(provider) -> list:
    """施加壁面模型的边界组编码：物理无滑移的 WALL 组。"""
    return [code for code, cfg in provider.code_to_config.items()
            if cfg.get("type") == "WALL" and cfg.get("physical_no_slip", True)]


class WMLESModel:
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
        from autoflowcfd.fr.native_padding import order_from_n_sps, sp_integration_weights

        points = wall_flux_points(flat, np.asarray(provider.group_code), wall_model_codes(provider))
        if points[0].size == 0:
            return None
        n_cells = int(mesh.n_cells)
        n_sps = int(np.asarray(mesh.sps_coords).shape[1])
        order = order_from_n_sps(n_sps)
        cell_is_prism = np.arange(n_cells) < int(mesh.n_prism_cells)
        det = np.asarray(mesh.jacobians["det_jacs"]).reshape(n_cells, n_sps)
        W = sp_integration_weights(det, cell_is_prism, order)
        return build_wall_matching(mesh, flat, points, cell_is_prism, W, det, order)

    def traction(self, xp, mesh, flat, provider, U, mu: float):
        """(匹配点几何, 壁面对流体的牵引 (m, 3))；没有物理无滑移壁面时 (None, None)。"""
        m = self.matching(xp, mesh, flat, provider)
        if m is None:
            return None, None
        tau, self.y_plus = wall_traction(xp, m, U, mu, self.kappa, self.B)
        return m, tau

    def residual_correction(self, xp, mesh, flat, provider, U, mu: float):
        """壁面剪应力对平均流残差的贡献 `(n_cells, n_sps, 5)`；没有物理无滑移壁面时 None。"""
        m, tau = self.traction(xp, mesh, flat, provider, U, mu)
        if m is None:
            return None
        n_cells, n_sps = U.shape[:2]
        return wall_stress_residual(xp, m, tau, n_cells, n_sps)
