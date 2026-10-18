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

import math

import numpy as np
from loguru import logger
from enum import Enum
from typing import Callable, Optional

from autoflowcfd.core.time_integration.positivity import assert_admissible




def _finish_stage(U, positivity_func):
    """每个 RK stage 的收尾：正性保持。

    `positivity_func` 由求解器按网格构建（守恒的 Zhang–Shu 限制器，见
    `time_integration/positivity/`）；积分器被脱离网格单独使用时为 None，
    此时只检查、不修改 —— 此前的逐点硬钳不守恒，且正是真实网格上一步爆炸
    的放大器，不再使用。
    """
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
    # DUAL_TIME 的内迭代同样是隐式 Newton-Krylov（`implicit/physical_time.py`）。
    #
    # **2026-10-17 删除 IMEX_EULER**：它的"隐式"子步是以 U_new 为基点的阻尼 Picard 迭代——每次子迭代
    # 再走一步显式步，收敛到的是稳态而不是隐式 Euler 解，收敛时还返回了未经最后一次更新的值。时间
    # 精确的隐式推进由双时间步承担。



#: **用户词汇 -> 枚举的唯一事实来源**（2026-09-18）。
#:
#: 此前同一个语义有**三套**词汇表、各自一份映射：
#:   * CLI `--time-method`：`rk3` / `dual-time`
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
#: 分支里静默换成别的格式。
DISTRIBUTED_SCHEMES = (
    TimeIntegrationScheme.FORWARD_EULER,
    TimeIntegrationScheme.SSP_RK2,
    TimeIntegrationScheme.SSP_RK3,
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



#: 每一步解 Newton-Krylov 线性系统的格式：隐式稳态，以及双时间步（内迭代是带 BDF 物理时间项的同一个
#: Newton 步，`implicit/physical_time.py`）。CFL 律、全局一致的块着色等按它判断
IMPLICIT_SCHEMES = (TimeIntegrationScheme.NEWTON_KRYLOV, TimeIntegrationScheme.DUAL_TIME)

#: 双时间步每个物理步的内迭代（带 BDF 物理时间项的 Newton 步，`implicit/physical_time.py`）次数上限的
#: 默认值，全部后端与 CLI 的唯一来源。内迭代按收敛判据提前结束（下一个常量），通常 2~7 次
DEFAULT_DUAL_TIME_STEPS = 20

#: 双时间步每个物理步内迭代的收敛判据：BDF 方程的残差 `||G||` 相对该物理步第一次内迭代之前下降到这个
#: 比例即停（`implicit/physical_time.py`）
DEFAULT_DUAL_TIME_TOLERANCE = 1.0e-3

#: 双时间步内迭代的另一个收敛判据：一个被完整接受的 Newton 步的更新量（按参考量归一的均方根，与残差
#: 同一个加权范数）小于这个值即停。物理步起点已经满足 BDF 方程到舍入水平时（均匀流、已收敛的定常解上
#: 继续推进）`||G||` 的初值就是舍入噪声，相对判据永远不成立；此时解每次迭代只在第 8 位以后变化，
#: 远小于任何时间离散误差
DUAL_TIME_UPDATE_FLOOR = 1.0e-8

#: 稳态相对收敛容差的默认值（残差降到初值的这个倍数即收敛；唯一来源：CLI `--tol`、配置 `convergence_tol`、
#: API 与 `solve()` 都取它。2026-10-09 以前 CLI 六处写死 1e-6、配置层默认 1e-3，两者互不相干）
DEFAULT_STEADY_TOL = 1e-6

#: 稳态命令传给 `solve()` 的 dt：按局部伪时间步推进的格式（显式、隐式稳态）不用它推进平均流（局部步长由 CFL
#: 决定），只有合成湍流入口（SEM）按它输运涡结构
STEADY_DT = 1e-3



def physical_time_steps(total_time: float, dt: float) -> int:
    """推进到物理时刻 `total_time` 所需的时间步数（瞬态命令、配置层、API 共用的唯一换算）。

    `total_time / dt` 在相对 1e-9 内是整数时取该整数（`int(0.3 / 0.1)` 截断成 2 这类浮点误差不再丢步）；
    否则取能越过 `total_time` 的最小步数，并报告实际结束时刻。
    """
    if not (dt > 0.0) or not (total_time > 0.0):
        raise ValueError(f"物理时长与时间步必须为正，收到 total_time={total_time}, dt={dt}")
    ratio = total_time / dt
    nearest = round(ratio)
    if nearest >= 1 and abs(ratio - nearest) <= 1e-9 * ratio:
        return int(nearest)
    n = int(math.ceil(ratio))
    logger.warning(f"物理时长 {total_time:g} s 不是时间步 {dt:g} s 的整数倍：推进 {n} 步，"
                   f"结束于 t = {n * dt:.9g} s")
    return n

class TimeIntegrator:
    """带局部时间步长的显式 SSP Runge-Kutta 积分器。"""

    def __init__(
        self,
        scheme: TimeIntegrationScheme = TimeIntegrationScheme.SSP_RK3,
        dt: float = 1e-5,
        cfl_target: float = 1.0,
        dual_time_steps: int = DEFAULT_DUAL_TIME_STEPS,  # 每个物理步的内迭代次数上限
    ):
        # 把任何旧别名映射到规范的枚举成员。
        self.scheme = TimeIntegrationScheme(scheme.value) if isinstance(scheme, TimeIntegrationScheme) \
            else TimeIntegrationScheme(scheme)
        self.dt = dt
        self.cfl_target = cfl_target
        self.dual_time_steps = dual_time_steps
        self.dual_time_tolerance = DEFAULT_DUAL_TIME_TOLERANCE
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

        Returns:
            U_new: 更新后的解 U^{n+1}
        """
        if self.scheme == TimeIntegrationScheme.DUAL_TIME:
            # DUAL_TIME 每个物理步是若干个带 BDF 物理时间项的 Newton 步，由各后端的 step() 经
            # `implicit/physical_time.py::advance_physical_step` 推进，不经过这个显式积分器入口。
            raise ValueError(
                "DUAL_TIME scheme 由隐式内迭代推进（implicit/physical_time.py），不经过通用的 step(...) 入口"
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
                solution, residual_func, dt_local, residual0, positivity_func=positivity_func,
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
        positivity_func: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    ) -> np.ndarray:
        """SSP-RK2/RK3 的 Shu-Osher stage 推进本体，不含 scheme 分发/计步。

        Args:
            table: 显式指定要用的 Shu-Osher 系数表；None 时用 self._table（即 self.scheme 对应的表）。
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
        U_stage1 = _finish_stage(U_stage1, positivity_func)

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
        U_stage2 = _finish_stage(U_stage2, positivity_func)

        # 重新计算Stage 2的残差（关键：不能省略）
        L2 = -residual_func(U_stage2)

        # === Stage 3 (如果是RK3) ===
        if n_stages >= 3:
            # U^(3) = alpha[2,0]*U^0 + alpha[2,1]*U^(1) + alpha[2,2]*U^(2) + beta[2]*dt*L(U^(2))
            U_stage3 = (alpha[2][0] * U0 +
                       alpha[2][1] * U_stage1 +
                       alpha[2][2] * U_stage2 +
                       beta[2] * dt * L2)
            U_stage3 = _finish_stage(U_stage3, positivity_func)

            # 对于RK3，最终解就是U^(3)
            U_new = U_stage3
        else:
            # 对于RK2，最终解是U^(2)
            U_new = U_stage2

        return U_new
    
    def reset(self) -> None:
        self.n_steps = 0
        self.current_time = 0.0
