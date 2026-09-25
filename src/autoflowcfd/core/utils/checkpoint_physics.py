"""AutoFlowCFD V2.0 - checkpoint 里"决定物理解本身"的那组参数（唯一的写入端）。

来流三要素、攻角/侧滑角、分子粘度、来流湍流强度与粘性比决定的是**物理算例
本身**，必须随 checkpoint 持久化、在 resume 时恢复 —— 不能像 CFL 那样让用户
每次 resume 重新指定，也不能按默认值猜（猜出来的是另一个物理算例，续算会在
错误的来流上静默跑到底）。

**为什么单独成模块（2026-09-25）**：此前只有单机写入端
（`cli/solve_checkpoint_io/write.py`）写这组键，分布式写入端
（`core/mpi/distributed_checkpoint/save.py`，CPU-MPI 与多 GPU 共用）一个都
不写。分布式 resume 在 2026-09-24 改成"来流缺失即报错"之后，**每一次**分布式
resume 都会因此失败；改之前则是静默按默认来流（33.33 m/s、零攻角、默认粘度与
Tu/VR）重建 —— 另一个物理算例。两个写入端现在共用本函数。
"""


def physics_metadata(solver) -> dict:
    """求解器上决定物理解的参数，写进 checkpoint 元数据。

    `solver.freestream` 必须带 `rho_inf/vel_inf/p_inf`（缺失即 KeyError，
    不猜）。攻角/侧滑角缺省为 0（早于 2026-09-17 的求解器对象没有这两个键，
    它们产生时的真实行为就是零攻角）。粘度与 Tu/VR 的缺省只为轻量替身对象
    （单元测试）保留：全部真实求解器（单机 CPU/GPU、CPU-MPI 两种加载模式、
    多 GPU 两种加载模式）都无条件设置这三个属性。
    """
    fs = solver.freestream
    return {
        "rho_inf": float(fs["rho_inf"]),
        "vel_inf": float(fs["vel_inf"]),
        "p_inf": float(fs["p_inf"]),
        "aoa_deg": float(fs.get("aoa_deg", 0.0) or 0.0),
        "aos_deg": float(fs.get("aos_deg", 0.0) or 0.0),
        "mu_molecular": float(getattr(solver, "mu_molecular", 1.8e-5)),
        "turbulence_intensity": float(getattr(solver, "_turbulence_intensity", 0.01)),
        "viscosity_ratio": float(getattr(solver, "_viscosity_ratio", 5.0)),
    }
