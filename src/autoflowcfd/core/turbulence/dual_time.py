"""湍流输运方程的双时间步（DUAL_TIME）推进，四个后端共用的唯一实现。

## 为什么需要（2026-10-09）

此前 DUAL_TIME 下湍流场每个物理步只按**物理**时间步长 dt 做一次显式更新（源项点隐式阻尼 + 显式输运），
平均流则在伪时间内迭代里按局部伪时间步推进。物理 dt 是按时间精度选的，不受近壁小单元上湍流方程显式稳定
极限的约束：通道算例 dt=1e-3 时 10 步内 k 到 -1e9、omega 到 1e-141；cube_demo DDES P2（dt=1e-4）3 步后
k 有 16 万个解点为负（最小 -1.2e8）、omega 有 2.8 万个解点为 0，而续算时这些值又被可容许性投影静默修正。

## 做法：与平均流同一套双时间步

湍流未知量 u（模型的未知量空间：SST 为 k 与 ln omega，SA 为 nu_tilde）的物理时间离散与平均流相同——
第一个物理步 BDF1、之后 BDF2（平均流见 `time_integration/dual.py`）：

    (a0 u^{n+1} - b) / dt = S(u^{n+1}) + T(u^{n+1})
    BDF1：a0 = 1，  b = u^n
    BDF2：a0 = 3/2，b = 2 u^n - u^{n-1} / 2

每个物理步内做与平均流相同次数的伪时间内迭代（`time_integrator.dual_time_steps`），伪时间步取平均流同一份
按物理波速的局部步长 dtau。一次内迭代 = 模型自身的显式更新（源项点隐式阻尼 + 输运，与稳态显式路径同一份）
得到 u*，再把物理时间项按点隐式并入：

    u <- (u* + dtau b / dt) / (1 + dtau a0 / dt)

内迭代收敛（u 不再变化）时恰好满足上面的 BDF 方程。物理时间项点隐式处理，dtau 大于 dt 时也稳定。
产生项渐变每个物理步只推进一次（第一次内迭代）。

## 后端

各后端的湍流更新函数（单机 CPU `fr_solver/turbulence/source.py::compute_turbulence_source`、单 GPU
`compute_turbulence_source_gpu`、CPU 分布式 `distributed_compute_turbulence_source_and_viscosity`、多 GPU
`_compute_turbulence_source_distributed`）接受 `physical_time`（本模块的 `PhysicalTimeTerm`）与
`advance_ramp`，在 `update_fields` 之后、后处理之前调用 `physical_time.apply`。分布式的求值在"本地 + halo"
的紧凑视图上进行，`to_view` 把本地排列的 b 换到视图排列（与输运场写进视图同一套 halo 交换与重排）。
"""

from typing import Callable, Optional


def reset_dual_time_history(owner) -> None:
    """清空上一物理时间层（平均流与湍流一起）：构造、换阶、重分发、开始一次新的 `solve()` 时调用。"""
    owner._dual_time_U_prev = None
    owner._dual_time_turb_prev = None


def _unknowns(model, xp) -> tuple:
    return tuple(model._unknown_from_field(j, f, xp) for j, f in enumerate(model.transported_fields()))


class PhysicalTimeTerm:
    """一个物理步内的 BDF 物理时间项（见模块文档）。

    Args:
        model: 本地排列的湍流模型（物理步开始时的状态即 u^n）
        xp: 模型数组所在的模块（numpy / cupy）
        previous: 上一物理时间层的未知量 u^{n-1}（本地排列）；None 时用 BDF1
        dt: 物理时间步长
        to_view: 本地排列 -> 求值视图排列的映射（分布式的紧凑视图）；None 表示求值就在本地模型上
    """

    def __init__(self, model, xp, previous, dt: float, to_view: Optional[Callable] = None):
        self.current = tuple(u.copy() for u in _unknowns(model, xp))
        if previous is None:
            self.a0, b = 1.0, self.current
        else:
            self.a0, b = 1.5, tuple(2.0 * c - 0.5 * p for c, p in zip(self.current, previous))
        self._b_local = b
        self._b_view = None
        self._to_view = to_view
        self.dt = float(dt)

    def apply(self, view_model, xp, dtau) -> None:
        """在刚做完显式更新的模型（本地模型或分布式视图）上并入物理时间项，再过正性/上界限幅。"""
        if self._to_view is None:
            b = self._b_local
        else:
            if self._b_view is None:
                self._b_view = tuple(self._to_view(x) for x in self._b_local)
            b = self._b_view
        ratio = dtau / self.dt
        fields = []
        for j, field in enumerate(view_model.transported_fields()):
            u = view_model._unknown_from_field(j, field, xp)
            fields.append(view_model._field_from_unknown(j, (u + ratio * b[j]) / (1.0 + ratio * self.a0), xp))
        view_model.set_transported_fields(fields)
        view_model.apply_positivity_limiter()


def advance_turbulence_dual_time(owner, model, xp, update: Callable, dtau, dt: float, n_inner: int,
                                 to_view: Optional[Callable] = None):
    """推进湍流场一个物理步（`n_inner` 次伪时间内迭代），返回最后一次 `update` 的返回值。

    Args:
        owner: 持有 `_dual_time_turb_prev` 的求解器
        model: 本地排列的湍流模型
        xp: 模型数组所在的模块
        update: `update(dtau, physical_time, advance_ramp)`——后端的一次湍流更新
        dtau: 伪时间步长（本地排列；分布式由后端换到视图排列）
        dt: 物理时间步长
        n_inner: 内迭代次数
        to_view: 见 `PhysicalTimeTerm`
    """
    if n_inner < 1:
        raise ValueError(f"双时间步内迭代次数必须 >= 1，收到 {n_inner}")
    term = PhysicalTimeTerm(model, xp, getattr(owner, "_dual_time_turb_prev", None), dt, to_view)
    result = None
    for it in range(n_inner):
        result = update(dtau, term, it == 0)
    owner._dual_time_turb_prev = term.current
    return result
