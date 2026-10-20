"""AutoFlowCFD V2.0 - checkpoint 里"决定物理解本身"的那组参数（唯一的写入端）。

来流三要素、攻角/侧滑角、分子粘度、来流湍流强度与粘性比决定的是**物理算例
本身**，必须随 checkpoint 持久化、在 resume 时恢复 —— 不能像 CFL 那样让用户
每次 resume 重新指定，也不能按默认值猜（猜出来的是另一个物理算例，续算会在
错误的来流上静默跑到底）。

**为什么单独成模块（2026-09-25）**：此前只有单机写入端
（`cli/solve/checkpoint_io/write.py`）写这组键，分布式写入端
（`core/mpi/distributed_checkpoint/save.py`，CPU-MPI 与多 GPU 共用）一个都
不写。分布式 resume 在 2026-09-24 改成"来流缺失即报错"之后，**每一次**分布式
resume 都会因此失败；改之前则是静默按默认来流（33.33 m/s、零攻角、默认粘度与
Tu/VR）重建 —— 另一个物理算例。两个写入端现在共用本函数。

`solve_path_metadata` 是另一类：时间格式与低马赫预处理开关**不改变**定常解，但续算默认应沿用（隐式稳态续算
不应被静默换回显式格式、关掉预处理跑的算例不应续算时被打开），`solve resume` 可用同名选项覆盖。
"""


def physics_metadata(solver) -> dict:
    """求解器上决定物理解的参数，写进 checkpoint 元数据。

    `solver.freestream` 必须带 `rho_inf/vel_inf/p_inf`（缺失即 KeyError，
    不猜）。攻角/侧滑角缺省为 0（早于 2026-09-17 的求解器对象没有这两个键，
    它们产生时的真实行为就是零攻角）。粘度、Tu/VR、人工粘性开关与近壁处理方式直接读属性：全部真实求解器（单机
    CPU/GPU、CPU-MPI 两种加载模式、多 GPU 两种加载模式）都无条件设置它们；2026-10-19（I-87）以前为轻量测试替身
    留了 getattr 缺省，某条路径真的丢了属性时会按缺省值写进 checkpoint、续算成另一个算例。
    """
    fs = solver.freestream
    return {
        "rho_inf": float(fs["rho_inf"]),
        "vel_inf": float(fs["vel_inf"]),
        "p_inf": float(fs["p_inf"]),
        "aoa_deg": float(fs.get("aoa_deg", 0.0) or 0.0),
        "aos_deg": float(fs.get("aos_deg", 0.0) or 0.0),
        "mu_molecular": float(solver.mu_molecular),
        "turbulence_intensity": float(solver._turbulence_intensity),
        "viscosity_ratio": float(solver._viscosity_ratio),
        # 问题单元人工粘性（2026-10-01）：启用时它是离散格式的一部分（定常解里
        # 的 div(nu grad U) 不为零），续算必须沿用，否则会静默换成另一个离散问题。
        "artificial_viscosity_enabled": bool(solver.artificial_viscosity_enabled),
        "artificial_viscosity_alpha": float(solver.artificial_viscosity_alpha),
        "wall_treatment": str(solver.wall_treatment),
    }


def solve_path_metadata(solver) -> dict:
    """续算默认沿用的推进设置：时间格式与低马赫预处理的请求值（读取端见
    `cli/solve/checkpoint_io/rebuild.py::resolve_resume_time_scheme / resolve_resume_low_mach_precond`）。"""
    return {
        "time_scheme": solver.time_integrator.scheme.value,
        "low_mach_precond": bool(solver.low_mach_precond),
    }


def run_status_metadata(solver) -> dict:
    """求解器自己的收敛结论（`core/fr_solver/state.py::solve_status`）；还没有调用过 `solve()` 时为 "running"。
    后处理报告的收敛结论只取它，不按别的阈值重新判定。"""
    return {"solve_status": getattr(solver, "solve_status", "running")}
