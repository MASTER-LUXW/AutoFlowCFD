"""AutoFlowCFD V2.0 - SA-neg 模型对象（numpy / cupy 共用一个类）。

模型只持有场与常数，并实现 `turbulence/transported.py` 的接口；逐点物理全部在 `pointwise.py`，
速率求值（源项 + 输运）在 `rates.py`。数组模块 `xp` 在构造时给定（CPU 求解器传 numpy、GPU
求解器传 cupy），全部方法都在该模块上运算。

## 壁面上的解点：强 Dirichlet `nu_tilde = 0`

原生四面体的解点含顶点、棱点与面点，会正好落在壁面上（壁距来源保证 `d == 0` 当且仅当
如此，`core/utils/wall_distance`）。SA 的耗散项 `cw1 fw (nu_tilde / d)^2` 与 `S_tilde` 里的
`nu_tilde / (kappa^2 d^2)` 在 `d = 0` 奇异：连续解 `nu_tilde ~ kappa u_tau d` 使各乘积有限，
但弱施加壁面条件的离散解在壁面解点上 `nu_tilde != 0`，耗散项随之爆炸（层流槽道四面体实测
这些解点的残差 3e15，比其余解点大 12 个数量级，差分 Jacobian 被舍入淹没）。方程在这些点上
退化成 `c nu_tilde^2 = 0`（`c ~ 1/d_floor^2`）：离散解本来就被它压到 0，只是经由一个病态的
二重根。这里直接取其极限——在壁面解点上施加精确边界值 `nu_tilde = 0`（连续有限元 / 格点型
有限体积对壁面结点的同一做法）：

* 场值：`apply_wall_distance` 把这些解点置 0，限制器与恢复路径保持它；
* 速率：`rates.py` 在这些解点上给出零速率（源项与输运都不作用），显式推进与 Newton 都不再
  改动它们；
* Jacobian：这些行置零（`turbulence/jacobian/assemble.py` 的 `strong_rows`），与零填充槽位
  同一处理——Newton 线性系统里只剩伪时间项，更新量为零。

棱柱的解点在单元内部，从不落在壁面上，这一条对它们不起作用。
"""

import numpy as np

from autoflowcfd.core.turbulence.limits import TURBULENT_VISCOSITY_RATIO_MAX, report_viscosity_cap
from autoflowcfd.core.turbulence.nonfinite import replace_nonfinite
from autoflowcfd.core.turbulence.transported import TransportedTurbulence
from autoflowcfd.core.utils.pointwise_threads import map_cell_chunks

from .constants import chi_for_viscosity_ratio
from .pointwise import sa_eddy_viscosity, sa_source_terms, vorticity_magnitude


def _pointwise_source(nu_tilde, nu, d_wall, grad_vel, production_factor, xp):
    """源项、点隐式阻尼系数与涡粘（`SAModel.compute_source_terms` 的逐点部分）。"""
    source, damping = sa_source_terms(nu_tilde, nu, d_wall, vorticity_magnitude(grad_vel, xp), production_factor, xp)
    return source, damping, sa_eddy_viscosity(nu_tilde, nu, xp)


class SAModel(TransportedTurbulence):
    """SA-neg 湍流模型（被输运的是 `nu_tilde`）。

    Attributes:
        nu_tilde_field: `(n_cells, n_sps)` 被输运的 SA 变量
        nu_t: `(n_cells, n_sps)` 涡粘（最近一次源项求值刷新）
        nu_tilde_inf: 来流值（开放边界的来流条件与初场）
        nu_ref: 来流运动粘度 `mu / rho_inf`（未知量尺度与上界的参考量）
        nu_tilde_max: 上界（`TURBULENT_VISCOSITY_RATIO_MAX * nu_ref`，`nu_t <= nu_tilde`）
        production_factor: 产生项斜坡因子（求解器维护，与 SST 同一机制）
    """

    TRANSPORTED_FIELDS = ("nu_tilde_field",)
    #: 壁面解点（`d == 0`）上的强 Dirichlet 值（模块文档）
    WALL_POINT_VALUE = 0.0
    NEWTON_LOG_COLUMNS = ()
    #: `nu_tilde` 的 Newton 更新不做相对变化限幅：SA-neg 的负支让任意实数值都良态（上界由步后的
    #: `apply_positivity_limiter` 钳到 `nu_tilde_max`），限幅保护的"非物理态"在这里不存在。SU2 对 SA-neg 同样
    #: 不做欠松弛（只对不带负支的 SA 做）。限幅的代价在湍流/非湍流界面上：那里 `nu_tilde` 只有来流量级
    #: （1e-5），界面推进需要它一步涨几十上百倍，而限幅基准是 max(|u|, 单元均值, 分子粘度)——cube_demo P1
    #: （2026-10-10）每步约 3000 行（1500 个单元，离物面 4~6 cm）被压到 2%~7%，随后局部伪时间步长被逐步削到
    #: 千分之几，界面冻住。
    NEWTON_UNLIMITED_COLUMNS = (0,)
    CACHED_ATTRS = ("nu_t", "_last_damping")
    OUTPUT_FIELD_KEYS = ("nu_tilde",)

    def __init__(self, n_cells: int, n_sps: int, nu_ref: float, viscosity_ratio: float, xp=np):
        """
        Args:
            n_cells, n_sps: 场的形状
            nu_ref: 来流运动粘度 `mu / rho_inf`
            viscosity_ratio: 来流涡粘比 `mu_t / mu`（与 SST 定 omega_inf 用的同一个参数）
            xp: 数组模块（numpy 或 cupy）
        """
        if not (np.isfinite(nu_ref) and nu_ref > 0.0):
            raise ValueError(f"SA 来流运动粘度必须为正有限值，收到 {nu_ref!r}")
        self.xp = xp
        self.n_cells, self.n_sps = int(n_cells), int(n_sps)
        self.nu_ref = float(nu_ref)
        self.viscosity_ratio = float(viscosity_ratio)
        self.nu_tilde_inf = chi_for_viscosity_ratio(self.viscosity_ratio) * self.nu_ref
        self.nu_tilde_max = TURBULENT_VISCOSITY_RATIO_MAX * self.nu_ref
        self.nu_tilde_field = xp.full((self.n_cells, self.n_sps), self.nu_tilde_inf)
        self.nu_t = xp.zeros((self.n_cells, self.n_sps))
        self._last_damping = None
        self.production_factor = 1.0
        self.wall_points = None
        self.viscosity_cap_activations = 0      # 涡粘比上限激活的累计解点数（limits.py::report_viscosity_cap）
        self.nonfinite_count = 0                # 非有限值替换的累计解点数（nonfinite.py::replace_nonfinite）
        self._nonfinite_reported = 0
        self._viscosity_cap_reported = 0

    # ---- TransportedTurbulence ----
    def _unknown_from_field(self, j, field, xp):
        return field

    def _field_from_unknown(self, j, column, xp):
        return column

    def unknown_scales(self):
        """`nu_tilde` 的尺度：来流运动粘度（壁面 `nu_tilde` 为 0，物理性限幅在那里退化成绝对
        限幅，尺度取分子粘度量级；见 `ScaledFieldRowLimits`）。"""
        return (self.nu_ref,)

    def freestream_values(self):
        return (self.nu_tilde_inf,)

    def upper_bounds(self):
        return (self.nu_tilde_max,)

    def like(self, n_cells: int, n_sps: int, wall_distance=None) -> "SAModel":
        other = SAModel(n_cells, n_sps, self.nu_ref, self.viscosity_ratio, xp=self.xp)
        other.production_factor = self.production_factor
        if wall_distance is not None:
            other.wall_points = self.xp.asarray(wall_distance) == 0.0
        return other

    def apply_wall_distance(self, wall_distance, nu_ref, global_min=None):
        """记下壁面解点（`d == 0`）并把场值置为强 Dirichlet 值（模块文档）。SA 没有随壁距变化的
        全局量，`global_min` 不用。"""
        wall = self.xp.asarray(wall_distance) == 0.0
        if wall.shape != self.nu_tilde_field.shape:
            raise ValueError(f"壁距形状 {wall.shape} 与 nu_tilde 场 {self.nu_tilde_field.shape} 不符")
        self.wall_points = wall
        self._pin_wall_points()

    def restore_transported(self, fields, source: str = "checkpoint"):
        super().restore_transported(fields, source)
        self._pin_wall_points()

    def _pin_wall_points(self):
        if self.wall_points is not None and self.wall_points.shape == self.nu_tilde_field.shape:
            self.nu_tilde_field = self.xp.where(self.wall_points, self.WALL_POINT_VALUE, self.nu_tilde_field)

    # ---- 求值 ----
    def compute_source_terms(self, Q, grad_vel, d_wall, mu):
        """`rho (P - D)`（带 rho），并刷新 `nu_t` 与点隐式阻尼系数。"""
        xp = self.xp
        rho = Q[:, :, 0]
        nu = mu / xp.maximum(rho, 1e-10)
        # 逐点函数按单元分块并行（numpy 下；同一份定义、结果逐位相同，`core/utils/pointwise_threads.py`）
        source, damping, self.nu_t = map_cell_chunks(
            _pointwise_source, (self.nu_tilde_field, nu, d_wall, grad_vel), self.production_factor, xp, xp=xp)
        self._last_damping = damping
        return replace_nonfinite(self, rho * source, "SA 源项", xp)

    def update_fields(self, dt, sources, transports):
        """一步显式更新：源项点隐式阻尼（系数取最近一次源项求值），输运显式。"""
        xp = self.xp
        if self._last_damping is None:
            raise RuntimeError("SA 显式更新需要源项求值刷新的阻尼系数：须先在同一个场上求源项")
        d_nt = sources[0] / (1.0 + dt * self._last_damping)
        if transports[0] is not None:
            d_nt = d_nt + transports[0]
        d_nt = replace_nonfinite(self, d_nt, "SA nu_tilde 增量", xp)
        self.nu_tilde_field = self.nu_tilde_field + dt * d_nt
        self.apply_positivity_limiter()

    def apply_positivity_limiter(self):
        """非有限值恢复为来流值、上界 `nu_tilde_max`。**不裁剪下界**：SA-neg 的负支是良态的、
        把负值推回零（`constants.py` 模块文档），裁剪会让离散稳态在那些解点上无解。"""
        xp = self.xp
        nt = self.nu_tilde_field
        nt = replace_nonfinite(self, nt, "SA nu_tilde 场", xp, fill=self.nu_tilde_inf)
        report_viscosity_cap(self, int(xp.count_nonzero(nt > self.nu_tilde_max)), "SA nu_tilde")
        self.nu_tilde_field = xp.minimum(nt, self.nu_tilde_max)
        self._pin_wall_points()
