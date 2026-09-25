"""伪时间稳态求解器与瞬态计算的时间积分格式。

稳态求解器在伪时间上推进解，直到残差趋于 0。为此我们采用显式的
强稳定性保持 Runge-Kutta 格式（SSP-RK2 / SSP-RK3）——这是有限体积
CFD 里标准、可证明正确的显式积分器，配合按对流+声速+粘性 CFL 条件
确定的**局部（逐单元）时间步长**。

这套实现取代了之前的版本，那个版本：(a) 自称"backward Euler"，实际
做的却是显式前向欧拉步；(b) RK2/AB3 用的是占位的残差历史；(c) 通过对
密度/速度/通量做硬幅值限幅来掩盖发散。现在物理正定性只在数学上确实
需要的地方（rho>0，p>0）才强制施加，用一个保持速度不变的*压力下限*
实现，发散会被报告出来，而不是被掩盖。
"""

from __future__ import annotations

import numpy as np
from enum import Enum
from typing import Callable, Optional

from autoflowcfd.core.time_integration.positivity import assert_admissible




def _finish_stage(U, filter_func, positivity_func):
    """每个 RK stage 的收尾：先滤波、再正性保持。

    顺序是刻意的：正性保持必须**最后**动手，离开 stage 的状态才一定可容许
    （滤波是线性的模态衰减，可能把刚被限制好的单元重新推出可容许集）。

    `positivity_func` 由求解器按网格构建（守恒的 Zhang–Shu 限制器，见
    `time_integration/positivity/`）；积分器被脱离网格单独使用时为 None，
    此时只检查、不修改 —— 此前的逐点硬钳不守恒，且正是真实网格上一步爆炸
    的放大器，不再使用。
    """
    if filter_func is not None:
        U = filter_func(U)
    if positivity_func is not None:
        positivity_func(U)
    else:
        assert_admissible(U)
    return U

class TimeIntegrationScheme(Enum):
    """时间积分格式枚举。"""
    FORWARD_EULER = "forward_euler"
    SSP_RK2 = "ssp_rk2"
    SSP_RK3 = "ssp_rk3"
    IMEX_EULER = "imex_euler"  # 一阶 IMEX（阻尼 Picard 子迭代，见 imex.py）
    DUAL_TIME = "dual_time"    # 双时间步长
    #: 矩阵自由 Newton-Krylov + 伪瞬态延拓（**唯一的真隐式格式**，
    #: 2026-09-19 新增，见 `time_integration/implicit/jfnk.py`）。
    #: 只用于**稳态**求解：它把"收敛步数由 h_min 决定"这条显式格式的
    #: 固有约束换成"由非线性程度决定"。
    NEWTON_KRYLOV = "newton_krylov"

    # **2026-09-18 删除三个旧别名**：
    #     BACKWARD_EULER = "forward_euler"
    #     RUNGE_KUTTA_2 = "ssp_rk2"
    #     ADAMS_BASHFORTH_3 = "ssp_rk3"
    # 它们的值与已有成员重复，在 Python 里因此是**同一个成员的别名**
    # （实测 `TimeIntegrationScheme.BACKWARD_EULER is
    # TimeIntegrationScheme.FORWARD_EULER` 为 True）。后果是选"后向 Euler"
    # （**隐式**格式）静默拿到前向 Euler（**显式**格式）——这正是本项目
    # 一贯不接受的假选项：一个看起来能选、实际给别的数值方案、且毫无提示
    # 的枚举值。全仓库 grep 确认这三个名字除定义处外零引用，所以删除不
    # 破坏任何调用方；真要复现历史配置请显式写 FORWARD_EULER/SSP_RK2/
    # SSP_RK3。
    #
    # **2026-09-19 更新**：真正的隐式稳态求解器已实现，见
    # `NEWTON_KRYLOV` 与 `time_integration/implicit/`（矩阵自由
    # Newton-Krylov + 伪瞬态延拓 + inexact-Newton forcing term）。
    # 预处理是单元块 Jacobi（`implicit/block_jacobi.py`，着色有限差分
    # 装配、跨 Newton 步复用；2026-09-25 替代了此前的逐 SP 对角形式——
    # 旧注释"每单元 n_sps*n_var 次残差求值、成本不可接受"漏掉了着色，
    # 真实次数与单元数无关）。CFL 律是 SER（`adaptive_cfl/ser.py`）。
    #
    # 其余两个仍然**不是**隐式格式：IMEX_EULER 只对粘性/源项做阻尼
    # Picard 子迭代（不是 Newton），DUAL_TIME 的内层仍是显式推进。
    # 不要再用一个别名把这些区别盖住。



#: **用户词汇 -> 枚举的唯一事实来源**（2026-09-18）。
#:
#: 此前同一个语义有**三套**词汇表、各自一份映射：
#:   * CLI `--time-method`：`rk3` / `imex` / `dual-time`
#:     （`cli/solve/transient.py` 一份 3 项的表）
#:   * `api.run_transient`：上面那些 + 枚举自身的取值
#:     （`api.py` 一份 8 项的表）
#:   * YAML/配置层：`backward_euler` / `rk2` / `rk3` / `ab3`
#:     （`config/solver_config.py` 里**另一个同名枚举**，取值范围与核心层
#:     不兼容，其中 `backward_euler`/`ab3` 在核心层根本没有实现）
#:
#: 三份之中 `api_config.py` 那一份是错的：它把 `dual-time` 映到
#: `BACKWARD_EULER`、把 `imex` 映到 `RK3`（都是静默给错值），未知取值还
#: 静默退到 RK3。本项目已多次因"同一件事有多份实现、只改了一份"出真实
#: 缺陷，所以这里合并成一份。
_SCHEME_ALIASES = {
    "rk3": TimeIntegrationScheme.SSP_RK3,
    "ssp_rk3": TimeIntegrationScheme.SSP_RK3,
    "rk2": TimeIntegrationScheme.SSP_RK2,
    "ssp_rk2": TimeIntegrationScheme.SSP_RK2,
    "imex": TimeIntegrationScheme.IMEX_EULER,
    "imex_euler": TimeIntegrationScheme.IMEX_EULER,
    "newton-krylov": TimeIntegrationScheme.NEWTON_KRYLOV,
    "newton_krylov": TimeIntegrationScheme.NEWTON_KRYLOV,
    "jfnk": TimeIntegrationScheme.NEWTON_KRYLOV,
    "implicit": TimeIntegrationScheme.NEWTON_KRYLOV,
    "dual-time": TimeIntegrationScheme.DUAL_TIME,
    "dual_time": TimeIntegrationScheme.DUAL_TIME,
    "forward_euler": TimeIntegrationScheme.FORWARD_EULER,
    "euler": TimeIntegrationScheme.FORWARD_EULER,
}


def scheme_names():
    """全部合法的用户侧取值（已排序），供错误信息与 CLI 帮助文本使用。

    **不要**在别处硬编码这张表——那正是被合并掉的那三份重复。
    """
    return sorted(_SCHEME_ALIASES)


def scheme_from_name(name):
    """把用户侧字符串（CLI/YAML/API）解析成 `TimeIntegrationScheme`。

    Args:
        name: 用户给的取值，大小写不敏感；也接受已经是枚举的对象（直接
            返回，便于调用方不必先判类型）。

    Returns:
        `TimeIntegrationScheme` 成员。

    Raises:
        ValueError: 取值不合法。**不静默退回默认值**——那会让一次拼写
            错误静默地把整个算例换成别的时间积分方案（`api_config.py`
            此前正是 `.get(name, RK3)`）。报错信息里列出全部合法取值。
    """
    if isinstance(name, TimeIntegrationScheme):
        return name
    key = str(name).strip().lower()
    scheme = _SCHEME_ALIASES.get(key)
    if scheme is None:
        raise ValueError(
            f"未知的时间积分方案 {name!r}；合法取值：{scheme_names()}。"
            f"唯一的隐式格式是稳态 Newton-Krylov（`newton-krylov`）；"
            f"`backward_euler`/`ab3` 这类名字曾经出现在配置层枚举里，但核心"
            f"层从未实现，已删除而不是留成静默映射到显式格式的假选项。"
        )
    return scheme

#: 分布式后端（CPU-MPI、多 GPU）实现了的时间积分方案。
#:
#: NEWTON_KRYLOV 的 GMRES 内积、Eisenstat–Walker 判据、线搜索与物理性限幅
#: 都经跨 rank 的归约对象（`core/mpi/reductions.py`），块 Jacobi 用全局一致
#: 着色（`core/mpi/distributed_implicit.py`）。表外的方案由
#: `require_distributed_scheme` 在构造时拒绝，而不是让 `step()` 走到某个回退
#: 分支里静默换成别的格式 —— IMEX 此前正是那样在分布式上跑成了前向 Euler。
DISTRIBUTED_SCHEMES = (
    TimeIntegrationScheme.FORWARD_EULER,
    TimeIntegrationScheme.SSP_RK2,
    TimeIntegrationScheme.SSP_RK3,
    TimeIntegrationScheme.IMEX_EULER,
    TimeIntegrationScheme.DUAL_TIME,
    TimeIntegrationScheme.NEWTON_KRYLOV,
)


def require_distributed_scheme(name):
    """解析时间积分方案并确认分布式后端支持它；返回枚举成员。"""
    scheme = scheme_from_name(name)
    if scheme not in DISTRIBUTED_SCHEMES:
        raise ValueError(
            f"分布式后端不支持时间积分方案 {scheme.value!r}；支持："
            f"{[s.value for s in DISTRIBUTED_SCHEMES]}（原因见 "
            f"`time_integration/base.py::DISTRIBUTED_SCHEMES` 的说明）。")
    return scheme


# SSP-RK Shu-Osher 系数：各阶段形如
#   u^(i) = sum_k alpha[i,k] u^(k) + beta[i] dt L(u^(i-1))
# 其中 L(u) = -R(u)。这里按格式分别存储各自的阶段系数表。
_SSP_RK2 = {
    "stages": 2,
    # u1 = u0 + dt L0 ;  u2 = 1/2 u0 + 1/2 (u1 + dt L1)
    "alpha": [[1.0], [0.5, 0.5]],
    "beta": [1.0, 0.5],
}
_SSP_RK3 = {
    "stages": 3,
    "alpha": [[1.0],
              [0.75, 0.25],
              [1.0/3.0, 0.0, 2.0/3.0]],
    "beta": [1.0, 0.25, 2.0/3.0],
}
_EULER = {"stages": 1, "alpha": [[1.0]], "beta": [1.0]}

_SCHEME_TABLE = {
    TimeIntegrationScheme.FORWARD_EULER: _EULER,
    TimeIntegrationScheme.SSP_RK2: _SSP_RK2,
    TimeIntegrationScheme.SSP_RK3: _SSP_RK3,
}



class TimeIntegrator:
    """带局部时间步长的显式 SSP Runge-Kutta 积分器。"""

    def __init__(
        self,
        scheme: TimeIntegrationScheme = TimeIntegrationScheme.SSP_RK3,
        dt: float = 1e-5,
        cfl_target: float = 1.0,
        dual_time_steps: int = 20,  # 每个物理步内的伪时间迭代次数
        # 默认值从 3 提高到 20：真实测得 BDF1 单物理步（受控衰减算例，
        # dt_physical=0.05）在默认 cfl 起点=下限=0.1 的保守步长策略下，
        # 3 次内迭代只完成 28% 的目标收敛量，需要约 60 次才能收敛到
        # BDF1 精度（见 time_integration.py::step_dual_time 文档 cfl
        # 起点选择的说明）。20 不是"证明足够"的精确值，是在"默认值必须
        # 明显好于 3"与"不无谓拖慢每个物理步"之间的工程折衷，通过
        # --dual-time-inner-iter（CLI）/FRSolver(dual_time_inner_iter=...)
        # 暴露给需要更严格收敛的场景调整，而不是像此前那样完全没有
        # 途径设置。
    ):
        # 把任何旧别名映射到规范的枚举成员。
        self.scheme = TimeIntegrationScheme(scheme.value) if isinstance(scheme, TimeIntegrationScheme) \
            else TimeIntegrationScheme(scheme)
        self.dt = dt
        self.cfl_target = cfl_target
        self.dual_time_steps = dual_time_steps
        self.n_steps = 0
        self.current_time = 0.0
        self._table = _SCHEME_TABLE.get(self.scheme, _EULER)

    # ------------------------------------------------------------------
    def step(
        self,
        solution: np.ndarray,
        residual_func: Callable[[np.ndarray], np.ndarray],
        dt_local: np.ndarray,
        residual0: Optional[np.ndarray] = None,
        filter_func: Optional[Callable[[np.ndarray], np.ndarray]] = None,
        positivity_func: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    ) -> np.ndarray:
        """根据配置的方案，推进一个时间步。
        
        对于显式SSP-RK格式，严格按照Shu-Osher形式实现三阶段计算，
        每个阶段都重新计算残差以确保时间精度。
        
        Args:
            solution: 当前解 U^n
            residual_func: 残差计算函数 R(U)
            dt_local: 局部时间步长数组
            positivity_func: 可选的正性保持回调（求解器按网格构建的守恒
                Zhang–Shu 限制器，见 time_integration/positivity/）；None 时
                每个 stage 只检查可容许性、不修改（见 `_finish_stage`）
            residual0: 预计算的初始残差（可选优化）
            filter_func: 可选的模态滤波回调（见 core/fr_solver_filter.py），
                每个 RK stage 的正定性投影之后立即施加一次——不能只在最终
                组合结果上滤波一次：真实复现，坍缩坐标节点配置法的混叠
                噪声会在*中间* stage（Stage1/Stage2 各自重新求值残差时）
                就已经放大到 NaN，等不到最终组合完成，见
                _ssp_rk_stage_step 里各 stage 后的调用点

        Returns:
            U_new: 更新后的解 U^{n+1}
        """
        if self.scheme == TimeIntegrationScheme.IMEX_EULER:
            # IMEX 需要把残差拆成显式对流项/隐式粘性+源项两部分分别求值
            # （见 step_imex 文档），这个拆分只有调用方（fr_solver.py）知道
            # 怎么做——通用的单一 residual_func 接口表达不了。此前这里直接
            # 把同一个组合残差函数传了两遍，代数上退化成 total_res=2*R(U)，
            # 不是真正的 IMEX。做法与下面 DUAL_TIME 分支一致：拒绝走这条
            # 通用入口，调用方须直接调用 step_imex(...)。
            raise ValueError(
                "IMEX_EULER scheme 需要拆分的显式(对流)/隐式(粘性+源项)残差函数，"
                "请直接调用 step_imex(solution, residual_explicit, residual_implicit, ...)，"
                "不要通过通用的 step(...) 入口（该入口只有一个组合残差函数，无法拆分）"
            )

        elif self.scheme == TimeIntegrationScheme.DUAL_TIME:
            # DUAL_TIME 需要真正的物理时间步长 dt_physical 与上一物理时间层
            # 的解 solution_prev（BDF2 时间导数项必需，见 step_dual_time
            # 文档），这两个概念在这个通用 dt_local 数组接口里表达不了，
            # 调用方（fr_solver.py::step）须直接调用 step_dual_time，不能
            # 经过这个通用分发入口。
            raise ValueError(
                "DUAL_TIME scheme 需要 dt_physical/solution_prev，请直接调用 "
                "step_dual_time(...)，不要通过通用的 step(...) 入口"
            )

        elif self.scheme == TimeIntegrationScheme.NEWTON_KRYLOV:
            # 隐式稳态步需要逐 SP 的伪时间步长（PTC 对角项）与守恒变量的
            # 参考量级（Fréchet 差分的无量纲化），后者这个通用接口里没有。
            # 与上面两条同一个原则：拒绝走通用入口，不静默退化成显式 RK。
            # 调用方（`fr_solver/step.py`）直接调
            # `time_integration.implicit.step_newton_krylov(...)`。
            raise ValueError(
                "NEWTON_KRYLOV scheme 需要守恒变量参考量级（Fréchet 差分的"
                "无量纲化，见 implicit/jacobian_vector.py），这个通用 "
                "step(...) 入口里没有；请直接调用 "
                "time_integration.implicit.step_newton_krylov(...)"
            )

        else:
            U_new = self._ssp_rk_stage_step(
                solution, residual_func, dt_local, residual0, filter_func=filter_func,
                positivity_func=positivity_func,
            )
            self.n_steps += 1
            return U_new

    def _ssp_rk_stage_step(
        self,
        solution: np.ndarray,
        residual_func: Callable[[np.ndarray], np.ndarray],
        dt_local: np.ndarray,
        residual0: Optional[np.ndarray] = None,
        table: Optional[dict] = None,
        filter_func: Optional[Callable[[np.ndarray], np.ndarray]] = None,
        positivity_func: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    ) -> np.ndarray:
        """SSP-RK2/RK3 的 Shu-Osher stage 推进本体，不含 scheme 分发/计步——
        从 step() 拆出来，供 step_dual_time 的内层伪时间迭代复用（见该方法
        文档：内层迭代此前用的是纯前向欧拉，但 dt_local/pseudo_dt 是按这套
        SSP-RK 格式的稳定性域标定的 CFL 步长，前向欧拉的稳定性域明显更小，
        直接复用同一个 CFL 数会失稳——真实复现：Couette 层流验证算例第一次
        内层迭代残差就从 5.5e6 暴涨到 2e27，两步内 NaN）。

        Args:
            table: 显式指定要用的 Shu-Osher 系数表；None 时用 self._table
                （即 self.scheme 对应的表）。step_dual_time 调用时必须显式
                传入 _SSP_RK3——self.scheme 此时是 DUAL_TIME 本身，
                self._table 会回退成 1-stage 的 _EULER 表（_SCHEME_TABLE
                里没有 DUAL_TIME 这个键），伪时间迭代想用的是 RK3，不是
                self.scheme 这个外层枚举。
        """
        tbl = table if table is not None else self._table
        alpha = tbl["alpha"]
        beta = tbl["beta"]
        n_stages = tbl["stages"]
        dt = dt_local[:, None]

        # Stage 0: 初始状态
        U0 = solution.copy()

        # 如果提供了预计算的残差，直接使用；否则计算
        if residual0 is not None:
            L0 = -residual0  # dU/dt = -R(U)
        else:
            L0 = -residual_func(U0)

        # === Stage 1 ===
        # U^(1) = U^0 + dt * L(U^0)
        U_stage1 = U0 + dt * L0
        del L0  # B-12 P2 OOM 修复第⑤级（2026-08-26）：L0 已完成使命（只参与
        # Stage 1 组合），但其引用会让 1.2GB 数组（79万单元×27SP×7变量）在后续
        # stage 残差求值期间白白驻留——retest7 实测（见下方 Stage 2 处同类注释）：
        # 体积项分块后峰值已大降，但 Stage 2 粘性残差求值时 U0/L0/U_stage1/L1/
        # U_stage2 五份共存（~6GB）仍把剩余 commit 压到临界，连 815MiB 的
        # correction 缓冲都分配失败。纯引用管理，不改变任何数值。
        U_stage1 = _finish_stage(U_stage1, filter_func, positivity_func)

        # FORWARD_EULER 只有 1 个 stage：_EULER 表里 alpha 只有 alpha[0]，
        # 下面 Stage 2/3 无条件访问 alpha[1] 会越界 IndexError（已实测复现）。
        # 单级前向欧拉的结果就是 Stage 1 本身，直接返回。
        if n_stages == 1:
            return U_stage1

        # 重新计算Stage 1的残差（关键：不能省略）
        L1 = -residual_func(U_stage1)

        # === Stage 2 ===
        # U^(2) = alpha[1,0]*U^0 + alpha[1,1]*U^(1) + beta[1]*dt*L(U^(1))
        U_stage2 = (alpha[1][0] * U0 +
                   alpha[1][1] * U_stage1 +
                   beta[1] * dt * L1)
        del L1  # 同 Stage 1 处 del L0 的 B-12 注释：L1 只参与 Stage 2 组合，
        # 组合完成立即释放，避免与 Stage 2 残差求值的瞬态数组共存。
        U_stage2 = _finish_stage(U_stage2, filter_func, positivity_func)

        # 重新计算Stage 2的残差（关键：不能省略）
        L2 = -residual_func(U_stage2)

        # === Stage 3 (如果是RK3) ===
        if n_stages >= 3:
            # U^(3) = alpha[2,0]*U^0 + alpha[2,1]*U^(1) + alpha[2,2]*U^(2) + beta[2]*dt*L(U^(2))
            U_stage3 = (alpha[2][0] * U0 +
                       alpha[2][1] * U_stage1 +
                       alpha[2][2] * U_stage2 +
                       beta[2] * dt * L2)
            U_stage3 = _finish_stage(U_stage3, filter_func, positivity_func)

            # 对于RK3，最终解就是U^(3)
            U_new = U_stage3
        else:
            # 对于RK2，最终解是U^(2)
            U_new = U_stage2

        return U_new
    
    def step_imex(
        self,
        solution: np.ndarray,
        residual_explicit: Callable[[np.ndarray], np.ndarray],
        residual_implicit: Callable[[np.ndarray], np.ndarray],
        dt_local: np.ndarray,
        positivity_func: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    ) -> np.ndarray:
        """执行一步 IMEX Euler 推进 (S-05)。实现见
        time_integration/imex.py::step_imex（从本文件拆出，控制单文件
        行数），文档字符串也在那里。"""
        from .imex import step_imex as _step_imex

        return _step_imex(self, solution, residual_explicit, residual_implicit, dt_local,
                          positivity_func=positivity_func)

    def step_dual_time(
        self,
        solution: np.ndarray,
        spatial_residual: Callable[[np.ndarray], np.ndarray],
        pseudo_dt: np.ndarray,
        dt_physical: float,
        solution_prev: Optional[np.ndarray] = None,
        max_inner_iter: int = 5,
        tol: float = 1e-4,
        filter_func: Optional[Callable[[np.ndarray], np.ndarray]] = None,
        positivity_func: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    ) -> np.ndarray:
        """执行一步 Dual-Time Stepping (S-05)。实现见
        time_integration/dual.py::step_dual_time（从本文件拆出，控制
        单文件行数），文档字符串也在那里。"""
        from .dual import step_dual_time as _step_dual_time

        return _step_dual_time(
            self, solution, spatial_residual, pseudo_dt, dt_physical,
            solution_prev=solution_prev, max_inner_iter=max_inner_iter, tol=tol, filter_func=filter_func,
            positivity_func=positivity_func,
        )

    def reset(self) -> None:
        self.n_steps = 0
        self.current_time = 0.0
