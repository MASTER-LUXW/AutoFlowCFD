"""AutoFlowCFD V2.0 - 瞬态逐物理步的气动力历史（`forces.csv`）。

时间精确算例（涡脱落、抖振）的标准监视量是每个时间步的总气动力：Strouhal 数来自升力 / 侧力的周期，平均阻力与
脉动幅值都要含粘性部分（Re = 100 圆柱粘性阻力约占总阻力的 1/4）。每步日志里的 `Cd_p/Cl_p/Cs_p` 只是压力积分
（`convergence_history.record_step`，为了不在每步算速度梯度），求解结束时才报一次总系数。

`solve transient --force-history`（需要 `--reference-area`）每个物理步追加一行
`time, iteration, Cd, Cl, Cs, Cd_p, Cl_p, Cs_p`（风轴系、全部 WALL 组，与结束时的报告同一积分：
`postprocess/fr_coefficients.py::total_and_pressure_force_coefficients`）。代价：每步一次全场速度 / 温度梯度与
壁面通量点积分；分布式（CPU-MPI、多 GPU）每步还要把整场状态汇总到 root（`core/mpi/global_view.py`），所以
只在显式要求时打开。

记录是集体操作（求解循环每步一次，`convergence_history.record_step` 调用）；续算时参考面积随 checkpoint 持久化
（`FORCE_HISTORY_KEY`），截掉晚于 checkpoint 物理时间的行再接着追加（`series_file.py`）。
"""

from pathlib import Path
from typing import Optional

from .series_file import append_series_row, prepare_series_file

FORCE_FILE = "forces.csv"
#: checkpoint 元数据里的气动力历史参考面积（续算时据此重新挂上）
FORCE_HISTORY_KEY = "force_history_reference_area"
COLUMNS = ("Cd", "Cl", "Cs", "Cd_p", "Cl_p", "Cs_p")
HEADER = ("time", "iteration") + COLUMNS


class ForceHistoryRecorder:
    """见模块文档。构造与 `record` 都是集体操作（分布式时全部 rank 调用）。"""

    def __init__(self, solver, reference_area: Optional[float], output_dir, resume_time: Optional[float] = None):
        """`reference_area`：root 上必须为正（分布式的其余 rank 可为 None，广播取 root 的值）。

        Raises:
            ValueError: 参考面积缺失或非正
        """
        from autoflowcfd.core.mpi import get_comm, get_size, is_root

        self._root = is_root()
        self._comm = get_comm() if get_size() > 1 else None
        area = reference_area if self._comm is None else self._comm.bcast(reference_area, root=0)
        if area is None or not float(area) > 0.0:
            raise ValueError("逐步气动力历史（--force-history）需要正的参考面积（--reference-area）")
        self.reference_area = float(area)
        self._distributed = getattr(solver, "partition", None) is not None
        self.path = Path(output_dir) / FORCE_FILE
        if self._root:
            prepare_series_file(self.path, HEADER, resume_time, "气动力历史的列")

    def record(self, solver) -> None:
        """求解循环每步调用一次（集体操作）：root 追加一行。"""
        from autoflowcfd.postprocess.fr_coefficients import total_and_pressure_force_coefficients

        if self._distributed:
            from autoflowcfd.core.mpi.global_view import gather_global_view

            view = gather_global_view(solver)
        else:
            view = solver if self._root else None
        if view is None:                             # 非 root
            return
        c = total_and_pressure_force_coefficients(view, self.reference_area)
        history = solver.convergence_history
        iteration = history.iterations[-1] if history.iterations else 0
        append_series_row(self.path, [repr(float(solver.time_integrator.current_time)), int(iteration)]
                          + [repr(c[k]) for k in COLUMNS])


def attach_force_history(solver, reference_area: Optional[float], output_dir,
                         resume_time: Optional[float] = None) -> ForceHistoryRecorder:
    """给求解器挂上逐步气动力历史（集体操作）。"""
    solver._force_recorder = ForceHistoryRecorder(solver, reference_area, output_dir, resume_time=resume_time)
    return solver._force_recorder


def attach_force_history_on_resume(solver, metadata: dict, output_dir, requested: bool,
                                   reference_area: Optional[float]) -> Optional[ForceHistoryRecorder]:
    """续算：checkpoint 记录了气动力历史时按记录的参考面积接着写（截掉晚于 checkpoint 物理时间的行）；没有记录但本次
    要求（`requested`）时按 `reference_area` 新挂上、从 checkpoint 时刻起写。都不是时不挂。"""
    from autoflowcfd.core.utils.checkpoint_time import PHYSICAL_TIME_KEY

    recorded = metadata.get(FORCE_HISTORY_KEY)
    if recorded is None and not requested:
        return None
    area = float(recorded) if recorded is not None else reference_area
    return attach_force_history(solver, area, output_dir, resume_time=float(metadata[PHYSICAL_TIME_KEY]))
