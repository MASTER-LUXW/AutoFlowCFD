"""AutoFlowCFD V2.0 - 逐步收敛历史（四个后端共用，写进 checkpoint、续算时恢复）。

每个物理步（稳态为一个伪时间步）一条：监控残差（收敛判据用的那个 RMS，`time_integration/residual_norm.py`）、
5 个平均流方程各自的 RMS、该步所用的 CFL，单机设置了参考面积时另有逐步压力积分力系数（与每步日志同一组数）。
两条求解循环（定阶 `solve_loop.py` 与 Order Continuation `order_continuation/run.py`）每步调用一次 `record_step`，
与是否打印日志无关。

## 为什么要有它（I-172，2026-10-19）

此前求解器只在各后端的单步里追加总残差（`residual_history`），checkpoint 写出端写的收敛历史只有当前迭代号
`{"iterations": [n]}`：`post convergence` 画出空图、`post report` 的残差与收敛结论全空，续算之后连总残差也从零
开始。显式时间精确推进一个物理步切成若干子步时，单步里追加还会每个子步记一条，历史长度与步数对不上。

## 迭代编号

第一条为 1；续算时由 checkpoint 恢复（`from_checkpoint`），编号接着 checkpoint 的迭代数往下走，与中间 checkpoint
的文件名一致。2026-10-19 之前写的 checkpoint 只有迭代号，恢复成空历史、编号从该迭代数之后开始。
"""

import math
from typing import Dict, List, Optional

import numpy as np

from autoflowcfd.core.fr_solver.residual_diagnostics import MEAN_FLOW_EQUATIONS

#: 收敛历史里监控残差的名字（逐方程残差用 `MEAN_FLOW_EQUATIONS`）
MONITOR_RESIDUAL = "rms"

#: 逐步压力积分力系数（只含无粘数值通量的壁面压力部分，`postprocess/fr_coefficients.py::pressure_force_coefficients`）
COEFFICIENT_KEYS = ("Cd_p", "Cl_p", "Cs_p")


class ConvergenceHistory:
    """逐步收敛历史，各列表等长、与 `iterations` 一一对应（没有算的量记 nan）。"""

    def __init__(self, start_iteration: int = 0):
        """
        Args:
            start_iteration: 第一条之前已经走过的迭代数（续算起点）
        """
        self._start = int(start_iteration)
        self.iterations: List[int] = []
        self.residual: List[float] = []
        self.per_equation: Dict[str, List[float]] = {eq: [] for eq in MEAN_FLOW_EQUATIONS}
        self.cfl: List[float] = []
        self.coefficients: Dict[str, List[float]] = {key: [] for key in COEFFICIENT_KEYS}

    def append(self, residual: float, per_equation=None, cfl: float = math.nan,
               coefficients: Optional[Dict[str, float]] = None) -> None:
        """记一步。`per_equation` 为 5 个平均流方程的 RMS（None 记 nan），`coefficients` 键为 `COEFFICIENT_KEYS`。"""
        self.iterations.append((self.iterations[-1] if self.iterations else self._start) + 1)
        self.residual.append(float(residual))
        values = np.full(len(MEAN_FLOW_EQUATIONS), np.nan) if per_equation is None else np.asarray(per_equation)
        for eq, value in zip(MEAN_FLOW_EQUATIONS, values):
            self.per_equation[eq].append(float(value))
        self.cfl.append(float(cfl))
        for key in COEFFICIENT_KEYS:
            self.coefficients[key].append(math.nan if coefficients is None else float(coefficients[key]))

    @property
    def last_coefficients(self) -> Optional[Dict[str, float]]:
        """最近一步的力系数；那一步没有算时 None。"""
        if not self.iterations or math.isnan(self.coefficients[COEFFICIENT_KEYS[0]][-1]):
            return None
        return {key: self.coefficients[key][-1] for key in COEFFICIENT_KEYS}

    def to_checkpoint(self) -> dict:
        """`CheckpointManager.save` 的 history 字典（全部步都没有算力系数时不写系数组）。"""
        out = {
            "iterations": list(self.iterations),
            "residuals": {MONITOR_RESIDUAL: list(self.residual), **{eq: list(v) for eq, v in self.per_equation.items()}},
            "cfl_history": list(self.cfl),
        }
        if any(not math.isnan(v) for v in self.coefficients[COEFFICIENT_KEYS[0]]):
            out["coefficients"] = {key: list(v) for key, v in self.coefficients.items()}
        return out

    @classmethod
    def from_checkpoint(cls, history: dict, iteration: int) -> "ConvergenceHistory":
        """由 checkpoint 的 history 与迭代数恢复（续算接着记）。没有逐步历史的旧 checkpoint 恢复成空历史。"""
        residuals = history.get("residuals", {})
        if MONITOR_RESIDUAL not in residuals:
            return cls(start_iteration=iteration)
        iterations = [int(i) for i in history["iterations"]]
        n = len(iterations)
        out = cls(start_iteration=iterations[0] - 1 if n else iteration)

        def column(values):
            values = [float(v) for v in values]
            if len(values) != n:
                raise ValueError(f"checkpoint 收敛历史各列长度不一致（{len(values)} vs 迭代数 {n}）")
            return values

        out.iterations = iterations
        out.residual = column(residuals[MONITOR_RESIDUAL])
        out.per_equation = {eq: column(residuals.get(eq, [math.nan] * n)) for eq in MEAN_FLOW_EQUATIONS}
        out.cfl = column(history.get("cfl_history", [math.nan] * n))
        coefficients = history.get("coefficients", {})
        out.coefficients = {key: column(coefficients.get(key, [math.nan] * n)) for key in COEFFICIENT_KEYS}
        if n and iterations[-1] != iteration:
            raise ValueError(f"checkpoint 收敛历史的最后一步 {iterations[-1]} 与 checkpoint 迭代数 {iteration} 不符")
        return out


def checkpoint_history(solver, iteration: int) -> dict:
    """checkpoint 写出端（单机与分布式共用）的收敛历史：`solver.convergence_history`，最后一步必须就是 `iteration`。

    两者不一致说明调用方传的迭代数与求解循环实际走过的步数不是一回事，续算时编号会错位（`from_checkpoint` 同样
    校验）——在写出时就报错，而不是留到续算才发现。
    """
    history = solver.convergence_history
    if history.iterations and history.iterations[-1] != int(iteration):
        raise ValueError(f"写 checkpoint 的迭代数 {iteration} 与收敛历史的最后一步 {history.iterations[-1]} 不符")
    return history.to_checkpoint()


def record_step(solver, residual: float) -> None:
    """求解循环每步调用一次：把这一步记进 `solver.convergence_history`。

    逐方程 RMS 取单步里算好的 `solver._residual_per_equation`（`monitor_residuals`）；CFL 取当前值
    （`adaptive_cfl/policy.py::current_cfl_number`）；设置了 `_reference_area` 时算一次压力积分力系数（每步日志
    读的也是这一份，`format_step_line`）。
    """
    from autoflowcfd.core.time_integration.adaptive_cfl.policy import current_cfl_number

    coefficients = None
    ref_area = getattr(solver, "_reference_area", None)
    if ref_area is not None and ref_area > 0:
        from autoflowcfd.postprocess.fr_coefficients import pressure_force_coefficients

        aero = pressure_force_coefficients(solver.host_view(), ref_area)
        coefficients = {"Cd_p": aero["Cd"], "Cl_p": aero["Cl"], "Cs_p": aero["Cs"]}
    solver.convergence_history.append(residual, getattr(solver, "_residual_per_equation", None),
                                      current_cfl_number(solver), coefficients)
