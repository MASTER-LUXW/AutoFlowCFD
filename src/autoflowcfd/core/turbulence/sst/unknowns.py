"""AutoFlowCFD V2.0 - SST 的输运场与 Newton 未知量（`turbulence/transported.py` 接口）。

被输运的是 `k` 与 `w = ln omega`（`log_omega.py`）：模型存物理 omega，Newton 未知量的第二列是
它的对数，写回时经 `omega_from_log` 施加上界。CPU（`SSTModelFR`）与 GPU（`GPUTurbulenceSST`）
共用本混入类。
"""

import numpy as np

from autoflowcfd.core.turbulence.transported import TransportedTurbulence

from .bounds import ABS_FLOOR, K_FLOOR_FRACTION
from .bounds import apply_omega_upper_bound
from .log_omega import admissible_omega, log_omega, omega_from_log


class _SSTTransportedMixin(TransportedTurbulence):
    """SST 的 `TransportedTurbulence` 实现。"""

    TRANSPORTED_FIELDS = ("k_field", "omega_field")
    NEWTON_LOG_COLUMNS = (1,)
    CACHED_ATTRS = ("nu_t", "_last_F1", "_last_beta_blend", "_omega_realizability_min")
    OUTPUT_FIELD_KEYS = ("k", "omega")

    def _unknown_from_field(self, j, field, xp):
        return field if j == 0 else log_omega(field, xp)

    def _field_from_unknown(self, j, column, xp):
        return column if j == 0 else omega_from_log(column, self.omega_max, xp)

    def unknown_scales(self):
        """`(k, ln omega)` 的逐列尺度：k 为尺度下限（物理性限幅与差分步长），`ln omega` 是
        O(1) 的对数量、取 1（它的限幅是绝对的，见 `ScaledFieldRowLimits` 的 `log_columns`）。"""
        return max(ABS_FLOOR, K_FLOOR_FRACTION * float(self.k_inf)), 1.0

    def freestream_values(self):
        return float(self.k_inf), float(self.omega_inf)

    def upper_bounds(self):
        return float(self.k_max), float(self.omega_max)

    #: 视图（`like`）从父模型复制的常数与全局量（`k_inf/omega_inf` 由构造参数给定：开边界来流
    #: ghost 取它们作为来流值，必须与真正的模型一致）
    _LIKE_ATTRS = ("sigma_k1", "sigma_k2", "sigma_w1", "sigma_w2", "beta1", "beta2",
                   "a1", "kappa", "beta_star", "k_max", "omega_max", "production_factor")

    def _new_instance(self, n_cells: int, n_sps: int):
        """同类的空实例（CPU 与 GPU 两个类的构造参数不同，各自给出）。"""
        raise NotImplementedError

    def like(self, n_cells, n_sps, wall_distance=None):
        """视图：常数、上界与斜坡取自本实例（omega 上界是全局最小壁距给定的，见 `bounds.py`）；
        SST 不需要逐点壁面信息，`wall_distance` 不用。"""
        other = self._new_instance(n_cells, n_sps)
        for attr in self._LIKE_ATTRS:
            if hasattr(self, attr):
                setattr(other, attr, getattr(self, attr))
        return other

    def apply_wall_distance(self, wall_distance, nu_ref, global_min=None):
        """omega 上界随最近壁面解点给定（`bounds.py` 模块文档）。"""
        apply_omega_upper_bound(self, wall_distance, nu_ref, global_min=global_min)

    def restore_transported(self, fields, source: str = "checkpoint"):
        """恢复 `(k, omega)`；omega 经可容许性投影（旧格式直接输运 omega、允许越过零，见
        `log_omega.py::admissible_omega`）。"""
        k, omega = fields
        xp = np
        if hasattr(omega, "device"):
            from autoflowcfd.core.gpu import get_cupy
            xp = get_cupy()
        self.k_field = k
        self.omega_field = admissible_omega(omega, self.omega_inf, xp=xp, source=source)


def sst_dirichlet_spec(wall_zero_face, omega_wall_face, has_omega_wall):
    """SST 两个 Newton 未知量的壁面 Dirichlet 规格 `(faces, values)`（主机数组，湍流解析块
    Jacobian 的 `TurbulenceLinearization` 用）：k 在壁面为 0；`w = ln omega` 取壁面解析值的对数
    （与残差 `transport/residual.py` 同一换算；没有目标的面该值不被读取）。"""
    return ((np.asarray(wall_zero_face, dtype=bool), np.asarray(has_omega_wall, dtype=bool)),
            (None, log_omega(np.asarray(omega_wall_face, dtype=np.float64), np)))
