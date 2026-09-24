"""AutoFlowCFD V2.0 - resume 后湍流场爆掉时的钳制/重置安全网

从 `src/autoflowcfd/core/mpi/distributed_order_continuation.py`(原 640 行)拆出(2026-09-24, 项目"单文件不超 500 行"规范)。**纯搬家, 逻辑未改**。
"""





from autoflowcfd.core.mpi import is_root


def _reset_turbulence_if_resumed_field_exploded(solver) -> None:
    """真实完整性缺口修复（2026-09-05）：单机 CPU `run_order_continuation`
    早就有的"resume 时检测湍流场是否被上界大面积钳制、若是则重置到
    来流初值"安全网（见 `core/utils/order_continuation.py::
    run_order_continuation` 该处文档完整推导——cube_demo 791,492 单元
    真实网格 `solve resume` 决定性验证发现并修复的判据本身的 bug），
    此前**从未移植**到本模块——CPU MPI"传统模式"/"完全分布式加载"、
    单机 GPU（`GPUFRSolver`）、多 GPU 分布式，三种后端全部统一走本
    模块的 `run_distributed_order_continuation`，但这个函数从头到尾
    都没有这项安全检查，等价于这三条路径的 resume 完全没有保护——
    真正爆炸的 checkpoint 被 resume 之后会静默从垃圾状态继续演化到底，
    没有任何救援机制（用户主动要求代码复审时发现，见项目记忆
    `cube_demo_gradvel_and_omega_wall_fixes_2026_09_05` "发现但未修复
    的真实完整性缺口"一节）。

    正确性关键——为什么不能直接照搬单机版 `np.mean`/`np.sum`：湍流场
    在这三条后端上都是**按 rank/设备本地分片**存储的（`turb_model.
    k_field`/`turb_model_gpu.k_field` 构造时就是 `(n_local_cells,
    n_sps)`，见 `distributed_solver.py`/`gpu_distributed.py` 对应
    `SSTModelFR(n_local, ...)`/`GPUTurbulenceSST(n_local_cells, ...)`
    构造调用——不含 halo，纯本 rank/设备份额）。如果每个 rank/设备只用
    自己的本地数据独立计算钳制比例、独立决定是否重置，不同 rank 可能
    对同一次 resume 做出不一致的判断（比如某个 rank 的本地分片恰好
    健康、另一个 rank 的本地分片真的大面积爆炸）——有的 rank 重置了
    湍流场、有的没有，后续每一步的 halo 交换会把这种"部分 rank 已重置、
    部分没有"的不一致状态混合扩散到全场，是比完全没有这项安全网更
    危险的半吊子实现，不能这样做。

    解决方式：用 `allreduce_sum`（`core/mpi/comm.py`，非 MPI/单 rank
    环境下自动降级为 no-op 直接返回本地值，见该函数文档——单机 GPU
    复用这条路径天然正确，不需要单独分支）分别对本 rank/设备的
    "钳制单元数""总单元数"两个标量求全局和，全局占比 =
    全局钳制数/全局总数，保证所有 rank/设备用同一个全局统计量做出
    同一个决定，不会出现前一段说的分裂状态。

    Args:
        solver: `DistributedFRSolver`（两种模式）/ `GPUFRSolver`（单机
            GPU，`_interpolate_to_new_order` 走 `gpu_solver_order_
            continuation.py`）/ `MultiGPUDistributedSolver`。
    """
    from autoflowcfd.core.mpi.comm import allreduce_sum
    from autoflowcfd.core.fr_solver.turbulence import _set_freestream_turbulence

    turb = getattr(solver, 'turb_model', None) or getattr(solver, 'turb_model_gpu', None)
    if turb is None or not hasattr(turb, 'k_max') or not hasattr(turb, 'k_field'):
        return

    k = turb.k_field
    omega = getattr(turb, 'omega_field', None)
    k_max_limit = turb.k_max
    omega_max_limit = getattr(turb, 'omega_max', None)

    # `.sum()`/`.size`/`float(...)` 对 numpy 和 CuPy 数组语义完全一致
    # （CuPy 0-d 数组 `float()` 会做一次隐式 device->host 拷贝，是
    # 已有的既定用法，见 gpu_solver_io.py 的 `float(dt_mean)`），不需要
    # 区分后端。
    k_hit_local = float((k >= 0.9 * k_max_limit).sum())
    n_local_total = float(k.size)
    omega_hit_local = 0.0
    if omega is not None and omega_max_limit is not None:
        omega_hit_local = float((omega >= 0.9 * omega_max_limit).sum())

    k_hit_global = allreduce_sum(k_hit_local)
    omega_hit_global = allreduce_sum(omega_hit_local)
    n_total_global = allreduce_sum(n_local_total)

    k_near_ceiling_frac = k_hit_global / max(n_total_global, 1.0)
    omega_near_ceiling_frac = omega_hit_global / max(n_total_global, 1.0)
    ceiling_frac_threshold = 0.10

    if k_near_ceiling_frac > ceiling_frac_threshold or omega_near_ceiling_frac > ceiling_frac_threshold:
        k_inf, omega_inf = _set_freestream_turbulence(solver)
        if is_root():
            print(f"[WARN] Resume: {100*k_near_ceiling_frac:.2f}% of k / "
                  f"{100*omega_near_ceiling_frac:.2f}% of omega values (global, "
                  f"summed across all ranks/devices) are clamped near their ceiling "
                  f"(k_max={k_max_limit:.2f}, omega_max={omega_max_limit}) — exceeds "
                  f"{100*ceiling_frac_threshold:.0f}% threshold, turbulence field not "
                  f"recovered from previous explosion. Resetting to freestream values.")
        turb.k_field[:] = k_inf
        turb.omega_field[:] = omega_inf
        if hasattr(turb, 'nu_t'):
            turb.nu_t[:] = 0.0
