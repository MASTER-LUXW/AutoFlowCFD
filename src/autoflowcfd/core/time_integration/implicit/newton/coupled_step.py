"""AutoFlowCFD V2.0 - 平均流 + 湍流输运方程紧耦合的一个 PTC-Newton-Krylov 步（全部后端共用）。

## 为什么必须紧耦合（2026-10-02）

此前 NK + SST 是分离式：每步先在冻结平均流上对 `(k, w)` 做一个 Newton 步，再在冻结涡粘上
对平均流做一个。CFL 到上限后两个子问题都近乎精确求解，外层就是对耦合稳态方程的块
Gauss-Seidel 迭代——它在不动点处的谱半径一旦大于 1 就周期振荡，与线性求解精度无关。

实测（远场槽道 SST P3，`tests/unit/test_implicit_sst_nk.py`）：湍流对流体积项改成一致的
对流形式之后（`turbulence/transport/convection.py` 模块文档），分离式 140 步只降 74 倍、
进入 4 步一周期的极限环（每步都是完整接受的 Newton 步、无任何限幅，GMRES 3~4 次）；
从同一状态出发冻结任一子系统则各自二次收敛（湍流 9 步到 8e-12、平均流 8 步到 1e-7）；
同一状态切到本文件的紧耦合步，8 步平均流残差 27.6 -> 2.3e-7、湍流 4.6 -> 1.1e-8。
09-30 删除 omega 壁面整单元强约束之前同一算例的"任何 CFL 下发散、冻结任一子系统则各自
收敛"也是这一机理。

## 做法

未知量 `x = (rho, rho u, rho v, rho w, rho E, t_1 .. t_n)`，每行一个解点；湍流未知量 `t` 由模型
声明（`turbulence/transported.py`：SST 为 `(k, ln omega)`，SA-neg 为 `nu_tilde`）：

    ( I/dtau + dR/dx ) dx = -R(x),     R = [ Gamma R_mean(U; mu_t(t, U)),  R_t(t; U) ]

* 残差每次求值都按当前平均流重算湍流的冻结输入（速度梯度、质量通量、壁面目标值）、按
  当前 `(k, omega)` 刷新涡粘——差分 matvec 含两个子系统之间的全部耦合，外层不再有块
  Gauss-Seidel；
* 预处理：两套已有的块预处理（平均流 / 湍流，各自解析装配、各自多层）组成块对角；
* 物理性限幅：平均流列按正性限制器点集逐单元（`PositivityLimiter.density_pressure_limits`），
  湍流列逐解点（`ScaledFieldRowLimits`），逐行取小；
* 伪时间步长：平均流那一份（与此前分离式湍流步同一取法，见 `fr_solver/turbulence/implicit.py`）。

## 后端适配器（`CoupledBackend` 协议）

    turb                    隐式湍流适配器（`fr_solver/turbulence/implicit.py` 模块文档"后端"）
    red / cell_is_prism / order / n_sps / scales_mean
    state() -> x            当前 (N, 5 + n)（`red.xp` 上，本 rank 行）
    snapshot() / restore(s) 平均流状态 + 湍流输运场 + 模型缓存（试探求值后还原）
    set_trial(x)            写入试探状态（平均流 5 列 + 湍流未知量经模型写回）
    trial_mu_t()            最近一次 `turb.rates` 之后平均流粘性残差用的涡粘（后端排列）
    mean_residual(mu_t)     平均流 `Gamma R`，(N, 5)
    mean_assembler(mu_t)    平均流解析单元块装配器（不覆盖时 None）
    positivity              该后端的 `PositivityLimiter`
    cell_colors() / coupling_graph / coarse_context()
    install(x)              写入新状态并做湍流步后收尾（正性限幅、刷新 nu_t 与 DES 尺度）

## 跨步状态（挂在求解器上，换阶时由 `mean_flow_step.reset_newton_state` 清空）

    _newton_forcing / _newton_dtau_scale / _newton_local_dtau   同平均流 Newton 步
    _newton_block_precond     平均流块缓存
    _newton_turb_state        {"block": 湍流块缓存}
    _newton_last_info         本步诊断，另含两个子系统各自的残差范数
"""

from typing import Optional

import numpy as np
from loguru import logger

from autoflowcfd.core.utils.array_module import array_module
from autoflowcfd.fr.native_padding import real_row_mask, real_sps_per_cell

from ..blocks.block_budget import plan_block_memory
from ..blocks.block_jacobi import BlockJacobiCache
from ..coarse.cell_coarse import mean_coarse_available, mean_coarse_weights
from .forcing import EisenstatWalkerForcing
from .globalization import ResidualNorm
from .jfnk import step_newton_krylov
from .mean_flow_step import N_MEAN_FLOW_VARS, note_linear_growth
from .physicality import ScaledFieldRowLimits

#: 两个子系统在耦合未知量里的列（湍流列数由模型声明，`TransportedTurbulence.n_transported`）。
MEAN_COLUMNS = slice(0, N_MEAN_FLOW_VARS)
TURB_COLUMNS = slice(N_MEAN_FLOW_VARS, None)


class CoupledResidual:
    """`R(x)`（模块文档"做法"）。纯函数：求值后还原求解器状态。基态 `x0` 上的结果记下来
    （Newton 步与子系统范数共用，不重复求值）。做成类而不是闭包（在整个 Krylov 求解期间
    存活，项目规范）。"""

    __slots__ = ("be", "real_rows", "x0", "r0")

    def __init__(self, backend, real_rows, x0):
        self.be = backend
        self.real_rows = real_rows
        self.x0 = x0
        self.r0 = None

    def __call__(self, x):
        if x is self.x0 and self.r0 is not None:
            return self.r0
        be = self.be
        xp = be.red.xp
        snap = be.snapshot()
        try:
            be.set_trial(x)
            be.turb.prepare_inputs()
            rates = be.turb.rates()
            out = xp.empty((x.shape[0], N_MEAN_FLOW_VARS + len(rates)), dtype=xp.float64)
            out[:, :N_MEAN_FLOW_VARS] = be.mean_residual(be.trial_mu_t())
            for j, rate in enumerate(rates):
                out[:, N_MEAN_FLOW_VARS + j] = -rate.ravel()
            out[~self.real_rows, N_MEAN_FLOW_VARS:] = 0.0
        finally:
            be.restore(snap)
        if x is self.x0:
            self.r0 = out
        return out


class SubsystemResidual:
    """耦合残差限制到湍流列上（平均流冻结在基态 `x0`）：湍流块缓存的差分装配兜底用它，湍流
    子系统的解析块 Jacobian 也以它为参照验证。"""

    __slots__ = ("full", "x0", "cols")

    def __init__(self, full, x0, cols: slice):
        self.full, self.x0, self.cols = full, x0, cols

    def __call__(self, u_sub):
        x = self.x0.copy()
        x[:, self.cols] = u_sub
        return self.full(x)[:, self.cols]


class FrozenViscosityMeanResidual:
    """平均流列的残差，湍流状态与涡粘都冻结在基态（`mu_t0`）：平均流块缓存的差分装配兜底
    （P0 等解析装配不覆盖的离散）用它——与 P>=1 解析装配器同一个冻结涡粘约定，且不必每次
    求值都重算湍流。"""

    __slots__ = ("be", "x0", "mu_t0")

    def __init__(self, backend, x0, mu_t0):
        self.be, self.x0, self.mu_t0 = backend, x0, mu_t0

    def __call__(self, u_mean):
        be = self.be
        x = self.x0.copy()
        x[:, MEAN_COLUMNS] = u_mean
        snap = be.snapshot()
        try:
            be.set_trial(x)
            return be.mean_residual(self.mu_t0)
        finally:
            be.restore(snap)


class _BlockDiagonal:
    """两个子系统预处理子的块对角组合（`PseudoTransientDiagonal` 接口）。

    部分灵活（`gmres.py` 模块文档）：整块非线性的列（块 ILU 档的多层 K 循环）标在 `nonlinear_columns`；平均流
    块 Jacobi 叠加单元级粗校正时（`coarse/cell_coarse.py`），它的非线性部分只是粗层向量，经 `coarse_rows`
    交给 GMRES，平均流列本身不标为非线性。"""

    __slots__ = ("mean", "turb", "dtau", "flexible", "nonlinear_columns", "coarse_rows", "n_coarse_cols",
                 "_mean_cols_nonlinear")

    def __init__(self, mean, turb, dtau, n_turb: int):
        self.mean, self.turb, self.dtau = mean, turb, dtau
        self.flexible = bool(mean.flexible or turb.flexible)
        self.coarse_rows = getattr(mean, "coarse_rows", None)
        self.n_coarse_cols = mean.n_coarse_cols if self.coarse_rows is not None else 0
        self._mean_cols_nonlinear = bool(mean.flexible) and self.coarse_rows is None
        turb_nl = bool(turb.flexible)
        # 只有一部分列整列非线性时 GMRES 只为那几列存预处理向量；两块整列同为非线性时整份存
        self.nonlinear_columns = (
            np.array([self._mean_cols_nonlinear] * N_MEAN_FLOW_VARS + [turb_nl] * int(n_turb), dtype=bool)
            if self._mean_cols_nonlinear != turb_nl else None)

    # 两块的作用都只读输入、返回新数组（接受非连续的列切片），所以这里不再先复制输入
    def apply(self, v):
        out = array_module(v).empty_like(v)
        out[:, :N_MEAN_FLOW_VARS] = self.mean.apply(v[:, :N_MEAN_FLOW_VARS])
        out[:, N_MEAN_FLOW_VARS:] = self.turb.apply(v[:, N_MEAN_FLOW_VARS:])
        return out

    def apply_split(self, v):
        """平均流粗校正之外的作用与粗层向量（`linear_solve.py::_SolveSpaceSplit`）。"""
        out = array_module(v).empty_like(v)
        out[:, :N_MEAN_FLOW_VARS], c = self.mean.apply_split(v[:, :N_MEAN_FLOW_VARS])
        out[:, N_MEAN_FLOW_VARS:] = self.turb.apply(v[:, N_MEAN_FLOW_VARS:])
        return out, c

    def apply_linear(self, v):
        """线性部分的作用（整列非线性的列置零、平均流不含粗校正，由 GMRES 用存下来的量补上）。"""
        out = array_module(v).empty_like(v)
        vm = v[:, :N_MEAN_FLOW_VARS]
        if self.coarse_rows is not None:
            out[:, :N_MEAN_FLOW_VARS] = self.mean.apply_linear(vm)
        else:
            out[:, :N_MEAN_FLOW_VARS] = 0.0 if self.mean.flexible else self.mean.apply(vm)
        out[:, N_MEAN_FLOW_VARS:] = 0.0 if self.turb.flexible else self.turb.apply(v[:, N_MEAN_FLOW_VARS:])
        return out


class CoupledBlockPreconditioner:
    """平均流 / 湍流两套块缓存（`BlockJacobiCache`）的块对角组合，接口是 `jfnk` 用到的
    `BlockJacobiCache` 子集。两套块都在基态上解析装配（另一子系统冻结在基态），与此前
    分离式两步各自的预处理同一份块。"""

    __slots__ = ("be", "mean", "turb", "residual", "n_turb")

    def __init__(self, backend, mean_cache: BlockJacobiCache, turb_cache: BlockJacobiCache, residual):
        self.be, self.mean, self.turb, self.residual = backend, mean_cache, turb_cache, residual
        self.n_turb = int(backend.turb.model.n_transported)

    @property
    def flexible(self) -> bool:
        return self.mean.flexible or self.turb.flexible

    @property
    def restart_cap(self):
        """两个缓存共用的一次规划给出的重启长度上限（`block_budget.py::plan_block_memory`）。"""
        return self.mean.restart_cap

    @property
    def flexible_fraction(self) -> float:
        """需要存预处理向量的分量占比（重启长度按它分内存，`linear_solve.py::krylov_restart`）：两个缓存各自的
        占比按列数加权。"""
        return ((N_MEAN_FLOW_VARS * self.mean.flexible_fraction + self.n_turb * self.turb.flexible_fraction)
                / (N_MEAN_FLOW_VARS + self.n_turb))

    def _with_assemblers(self, u0, action: str, r0, scales, dtau):
        be = self.be
        snap = be.snapshot()
        try:
            be.set_trial(u0)
            be.turb.prepare_inputs()
            be.turb.rates()
            mu_t0 = be.trial_mu_t()
            self.mean.assembler = be.mean_assembler(mu_t0)
            self.turb.assembler = be.turb.block_assembler()
            subsystems = ((self.mean, MEAN_COLUMNS, FrozenViscosityMeanResidual(be, u0, mu_t0)),
                          (self.turb, TURB_COLUMNS, SubsystemResidual(self.residual, u0, TURB_COLUMNS)))
            return [getattr(cache, action)(sub_residual, u0[:, cols].copy(), r0[:, cols].copy(), scales[cols], dtau)
                    for cache, cols, sub_residual in subsystems]
        finally:
            be.restore(snap)

    def begin_step(self, residual, u0, r0, scales, dtau) -> None:
        self._with_assemblers(u0, "begin_step", r0, scales, dtau)

    @property
    def recoverable(self) -> bool:
        return self.mean.recoverable or self.turb.recoverable

    def recover(self, residual, u0, r0, scales, dtau) -> bool:
        """本步线性求解没到容差后的补救（`BlockJacobiCache.recover`），两套块各做各的；任一有动作即重新求解。"""
        if not self.recoverable:
            return False
        return any(self._with_assemblers(u0, "recover", r0, scales, dtau))

    def preconditioner(self, dtau, n_var: int):
        if n_var != N_MEAN_FLOW_VARS + self.n_turb:
            raise ValueError(f"耦合预处理只接受 {N_MEAN_FLOW_VARS + self.n_turb} 列，收到 {n_var}")
        return _BlockDiagonal(self.mean.preconditioner(dtau, N_MEAN_FLOW_VARS),
                              self.turb.preconditioner(dtau, self.n_turb), dtau, self.n_turb)

    def stale_budget(self) -> Optional[int]:
        """本步 GMRES 的过时预算：复用中的块的基线迭代数，加上 `recover` 重装配实际要付的代价——它把**两套**块
        一起重装配，所以松弛是两者之和。

        2026-10-10 以前取两套块各自预算的较小者：湍流块的松弛只有 3 次迭代，耦合系统的 GMRES 次数比基线
        多 4 次就触发一次平均流 + 湍流的整套重装配（折合约 11 次迭代）。cube_demo SA P1 双时间步的第一个
        物理步 6 次内迭代里因此重装配 4 次（每次 35~75 s，外加被丢弃的那次过时求解），而重装配之后 GMRES
        次数并没有下降（15 -> 19）：次数上升来自右端项，不是块过时。
        """
        stale = [c for c in (self.mean, self.turb) if c.stale_budget() is not None]
        if not stale:
            return None
        # 没装配过的块（预处理被禁用）没有松弛，不参与
        # 有松弛预算的块都装配过，基线迭代数已记下
        return int(min(c.baseline_iters for c in stale if c.baseline_iters is not None)
                   + (self.mean.refresh_slack or 0) + (self.turb.refresh_slack or 0))

    def record(self, gmres_iters: int, accepted: bool) -> None:
        self.mean.record(gmres_iters, accepted)
        self.turb.record(gmres_iters, accepted)


class CoupledPhysicality:
    """逐行物理性限值：平均流列在正性限制器点集上逐单元、湍流列逐解点，逐行取小。"""

    __slots__ = ("positivity", "turb_limits")

    def __init__(self, positivity, turb_limits: ScaledFieldRowLimits):
        self.positivity, self.turb_limits = positivity, turb_limits

    def __call__(self, u0, du, red):
        a_mean = self.positivity.density_pressure_limits(u0[:, :N_MEAN_FLOW_VARS], du[:, :N_MEAN_FLOW_VARS], red)
        a_turb = self.turb_limits(u0[:, N_MEAN_FLOW_VARS:], du[:, N_MEAN_FLOW_VARS:], red)
        return red.xp.minimum(a_mean, a_turb)


def _caches(solver, backend):
    """两套块缓存（首次构造；换阶后由 `reset_newton_state` 清空重建）。两者共享预处理内存预算。"""
    need_turb = getattr(solver, "_newton_turb_state", None) is None
    need_mean = solver._newton_block_precond is None
    if need_turb or need_mean:
        n_real_prism, n_real_tet = real_sps_per_cell(int(backend.order))
        cell_is_prism = np.asarray(backend.cell_is_prism, dtype=bool)
        n_turb = int(backend.turb.model.n_transported)
        mean_coarse = mean_coarse_available(n_real_prism, n_real_tet)
        plan = plan_block_memory(backend.red, cell_is_prism, backend.n_sps, n_real_prism, n_real_tet,
                                 N_MEAN_FLOW_VARS, n_turb, mean_coarse=mean_coarse)
        common = dict(cell_is_prism=cell_is_prism, n_sps=backend.n_sps, plan=plan,
                      n_real_prism=n_real_prism, n_real_tet=n_real_tet, red=backend.red,
                      colors=backend.cell_colors(), global_coarse=backend.coarse_context())
        if need_turb:
            solver._newton_turb_state = {
                "block": BlockJacobiCache(n_var=n_turb, **common)}
        if need_mean:
            solver._newton_block_precond = BlockJacobiCache(
                n_var=N_MEAN_FLOW_VARS, coupling_graph=backend.coupling_graph, with_turbulence=True,
                coarse_weights=mean_coarse_weights(backend.positivity) if mean_coarse else None, **common)
    return solver._newton_block_precond, solver._newton_turb_state["block"]


def step_coupled_newton(solver, backend, dtau_flat, *, physical_time=None,
                        advance_ramp: bool = True) -> dict:
    """平均流 + 湍流紧耦合的一个 PTC-Newton-Krylov 步：求解、写回新状态、记录诊断，返回 info。

    Args:
        solver: 持有跨步状态的求解器对象（模块文档"跨步状态"）。
        backend: 该后端的耦合适配器（模块文档"后端适配器"）。
        dtau_flat: `(N,)` 平均流逐解点伪时间步长（天花板，见 `jfnk.py`）。
        physical_time: 双时间步的 BDF 物理时间项（`physical_time.py`，平均流与湍流列同一个），稳态为 None。
        advance_ramp: 是否推进产生项渐变计数（双时间步的内迭代里只有每个物理步的第一次推进）。
    """
    red = backend.red
    xp = red.xp
    if solver._newton_forcing is None:
        solver._newton_forcing = EisenstatWalkerForcing()
    if advance_ramp:
        backend.turb.advance_ramp()
    mean_cache, turb_cache = _caches(solver, backend)
    real = xp.asarray(real_row_mask(np.asarray(backend.cell_is_prism, dtype=bool), backend.n_sps,
                                    int(backend.order)))
    model = backend.turb.model
    t_scales = model.unknown_scales()
    scales = np.concatenate([np.asarray(backend.scales_mean, dtype=np.float64)[:N_MEAN_FLOW_VARS],
                             np.asarray(t_scales, dtype=np.float64)])
    physicality = CoupledPhysicality(
        backend.positivity,
        ScaledFieldRowLimits(t_scales, log_columns=model.NEWTON_LOG_COLUMNS, rows_per_cell=backend.n_sps,
                             real_rows=real, unlimited_columns=model.NEWTON_UNLIMITED_COLUMNS))
    x0 = xp.ascontiguousarray(backend.state(), dtype=xp.float64)
    residual = CoupledResidual(backend, real, x0)
    norm_scales = residual_reference_scales(x0, scales, backend.positivity.norm_weights, red,
                                            model.NEWTON_LOG_COLUMNS)
    dtau_flat = xp.ascontiguousarray(dtau_flat, dtype=xp.float64).ravel()
    x_new, info = step_newton_krylov(
        residual, x0, dtau_flat, scales,
        forcing=solver._newton_forcing, dtau_scale=solver._newton_dtau_scale,
        block_precond=CoupledBlockPreconditioner(backend, mean_cache, turb_cache, residual),
        physicality=physicality, rows_per_cell=1, red=red,
        local_dtau_scale=solver._newton_local_dtau, norm_weights=backend.positivity.norm_weights, real_rows=real,
        physical_time=physical_time, norm_scales=norm_scales)
    solver._newton_dtau_scale = info["dtau_scale"]
    solver._newton_local_dtau = info["local_dtau_scale"]
    backend.install(x_new)
    info.update(_subsystem_norms(residual.r0, backend.positivity.norm_weights, red, norm_scales))
    solver._newton_last_info = info
    note_linear_growth(solver, info)
    if info["theta"] <= 0.0:
        logger.warning(
            "耦合 Newton 步未能前进（theta=0, gmres_info=%s, gmres_iters=%d, dtau_scale=%.3e, 本步已缩 "
            "%d 档）——dtau 缩到下限仍拿不到被接受的步，检查残差求值在当前状态上是否已非物理"
            % (info["gmres_info"], info["gmres_iters"], info["dtau_scale"], info["n_dtau_cuts"]))
    elif info["n_dtau_cuts"] > 0:
        logger.info("耦合 Newton 步缩 %d 档 dtau 后被接受（dtau_scale=%.3e, theta=%.3f）"
                    % (info["n_dtau_cuts"], info["dtau_scale"], info["theta"]))
    return info


def residual_reference_scales(x0, scales, weights, red, log_columns=()) -> np.ndarray:
    """残差范数的逐列参考量级：平均流列原样（来流量级，`scales`），湍流列取该未知量在场里的实际量级——
    体积加权 RMS（与 `ResidualNorm` 同一组权重，分布式下是全局量），下限为 `scales`（模型的
    `unknown_scales`：差分步长与物理性限幅的尺度下限）。

    残差是 `dU/dt`，除以参考量级后是"该未知量的相对变化率"，各方程才可比。平均流的来流量级就是场的
    量级；湍流的 `unknown_scales` 不是——SA 的是分子粘度，而湍流区里 `nu_tilde` 是它的几百上千倍
    （cube_demo SA P1：体积加权前的逐点 RMS 769 nu，99 分位 3352 nu）。2026-10-11 用它当参考量级
    （上一提交）把湍流残差在范数里放大了约三个数量级：合成范数 2e4，平均流只有 1~2。GMRES 的容差
    是合成范数的 0.1 倍，平均流各行的线性残差是自身的上千倍也算达标；接受判据同理。cube_demo 从
    稳态 P1 第 44 步起 CFL 300 续算：线性求解达标、整步接受的一步里平均流残差 1.28 -> 8.42，下一步
    91.5，Cd 0.734 -> 0.676，不限幅时三步后最大速度 394 m/s。

    对数未知量（`log_columns`，`ln omega`）的残差本身就是相对变化率，保持 `scales`。逐步按当前场
    求值：场的量级随收敛缓慢变化，同一步内步前、步后的范数用同一组值。
    """
    out = np.array(scales, dtype=np.float64)
    xp = red.xp
    w = xp.asarray(weights, dtype=xp.float64).ravel()
    total = red.sum(w)
    if total <= 0.0:
        return out
    skip = {N_MEAN_FLOW_VARS + int(c) for c in log_columns}
    for j in range(N_MEAN_FLOW_VARS, out.size):
        if j not in skip:
            out[j] = max(out[j], float(np.sqrt(red.sum(w * x0[:, j] * x0[:, j]) / total)))
    return out


def _subsystem_norms(r0, weights, red, scales) -> dict:
    """步前两个子系统各自的残差范数（监控与升阶判据用，与接受判据同一个体积加权、按参考量级无量纲化的
    RMS）。`res_norm` 本身是两者合在一起的耦合范数。"""
    norm = ResidualNorm(weights, red, scales)
    return {"res_norm_mean": norm.restricted(slice(0, N_MEAN_FLOW_VARS))(r0[:, :N_MEAN_FLOW_VARS]),
            "res_norm_turbulence": norm.restricted(slice(N_MEAN_FLOW_VARS, None))(r0[:, N_MEAN_FLOW_VARS:])}
