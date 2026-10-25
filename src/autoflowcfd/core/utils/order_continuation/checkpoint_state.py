"""AutoFlowCFD V2.0 - Order Continuation 随 checkpoint 持久化的阶段状态（写入/恢复的唯一实现）。

`run_order_continuation` 每步把本阶段的残差下降基准写到 `solver._phase_initial_residual`；resume 恢复出的
第一个阶段用它做升阶/收敛判据的种子。没有它时判据从 resume 的第一步重新起算——resume 时残差已经降过
一截，同样的下降倍数会在比原运行更低的绝对残差上才达到，或者反过来在基准偏低时提前升阶。

2026-10-05 以前单机写入端（`cli/solve/checkpoint_io/write.py`）与恢复端（`restore.py`）各写一份，两个分布式
后端（`core/mpi/distributed_checkpoint/save.py` 与 `cli/solve/distributed_checkpoint_io.py`）一份都没有。
分布式下这个值是全域残差范数（各 rank 一致），root 写出即可。

基准同时写入残差范数的定义版本（`time_integration/residual_norm.py::RESIDUAL_NORM_VERSION`）：定义改过
（2026-10-17：真实自由度 RMS，此前各后端定义不同且含填充槽位）之后，旧 checkpoint 里的基准与新残差不可比，
续算时不用它。

与基准一起写入本阶段已走的步数（`solver._phase_iterations`）：续算出来的第一个阶段从它接着数——非最终阶段的
步数上限（`--phase-max-iter`）、提前升阶前的最少步数与日志步号都按阶段内的真实步数算。2026-10-25 以前不落盘，
续算时阶段从第 1 步重新数：方柱 SA URANS 从 P0 阶段末的 checkpoint 续算，P0 又跑满 150 步才升到 P1。基准与步数
总是同时写出；只有基准没有步数的是此前写出的 checkpoint，按旧格式报错（转换用一次性脚本，不在源码里兼容）。
"""

from loguru import logger

from autoflowcfd.core.time_integration.residual_norm import RESIDUAL_NORM_VERSION

PHASE_INITIAL_RESIDUAL_KEY = "phase_initial_residual"
RESIDUAL_NORM_VERSION_KEY = "residual_norm_version"
PHASE_ITERATIONS_KEY = "phase_iterations"


def phase_state_metadata(solver) -> dict:
    """要写进 checkpoint 元数据的阶段状态；第一次 `step()` 之前尚未产生时为空（h5py attrs 不接受 None）。"""
    value = getattr(solver, "_phase_initial_residual", None)
    if value is None:
        return {}
    return {PHASE_INITIAL_RESIDUAL_KEY: float(value), RESIDUAL_NORM_VERSION_KEY: RESIDUAL_NORM_VERSION,
            PHASE_ITERATIONS_KEY: int(solver._phase_iterations)}


def restore_phase_state(solver, metadata: dict) -> None:
    """从 checkpoint 元数据恢复阶段状态；没有阶段基准的 checkpoint 不设置（`run_order_continuation` 打印警告、
    基准与阶段步数从这次 resume 的第一步重新起算）。

    Raises:
        ValueError: 有阶段基准而没有阶段已走步数（2026-10-25 以前写出的 checkpoint，见模块文档）
    """
    value = metadata.get(PHASE_INITIAL_RESIDUAL_KEY)
    if value is None:
        return
    version = metadata.get(RESIDUAL_NORM_VERSION_KEY)
    if version is None or int(version) != RESIDUAL_NORM_VERSION:
        logger.warning(f"checkpoint 的阶段残差基准按旧的残差范数定义（版本 {version}，当前 {RESIDUAL_NORM_VERSION}）"
                       f"写出，与当前残差不可比，不用它做种子（升阶/收敛判据与阶段步数从这次续算的第一步起算）")
        return
    iterations = metadata.get(PHASE_ITERATIONS_KEY)
    if iterations is None:
        raise ValueError(f"checkpoint 记录了阶段残差基准但没有阶段已走步数（{PHASE_ITERATIONS_KEY}），是 2026-10-25 "
                         f"以前的格式：续算会把阶段步数从 1 重新数（非最终阶段的步数上限被重复使用）。用一次性转换"
                         f"脚本补上该字段（取 checkpoint 在本阶段已走的步数）后再续算")
    solver._phase_initial_residual = float(value)
    solver._phase_iterations = int(iterations)
