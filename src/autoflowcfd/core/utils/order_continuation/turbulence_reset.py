"""AutoFlowCFD V2.0 - resume 后湍流场被上界大面积钳制时的重置安全网（全部后端共用的唯一实现）

从 `src/autoflowcfd/core/utils/order_continuation.py`(原 926 行)拆出(2026-09-24, 项目"单文件不超 500 行"规范)。**纯搬家, 逻辑未改**。
"""






from autoflowcfd.core.fr_solver.turbulence import _set_freestream_turbulence


def _reset_turbulence_if_resumed_field_exploded(solver) -> None:
    """resume 时若湍流场被上界大面积钳制，则重置到来流初值 —— **全部后端
    共用的唯一实现**（单机 CPU / CPU MPI 两种模式 / 单机 GPU / 多 GPU）。

    ## 为什么只能有一份（2026-09-24 合并）

    此前单机 `run_order_continuation` 里内联一份（`np.mean`），
    `core/mpi/distributed_order_continuation/turbulence_reset.py` 再写一份
    （`allreduce_sum`）。后者是前者的严格推广：`allreduce_sum` 在非 MPI /
    单 rank 下原样返回本地值，于是 `count/size` 与 `np.mean(bool数组)`
    是同一个双精度除法，**逐位相同**；另外它兼容 `turb_model_gpu`。两份
    并存正是本项目反复出过真实缺陷的形态（"两份实现只改了一份"），所以
    保留推广的那份、放在后端中立的这里。

    ## 判据的来历（原单机内联处的注释，一并移入）

    检测湍流场是否被上界大面积钳制（旧 checkpoint k/omega 爆炸后
    resume 被上界截断）。如果超过 10% 的单元 k/omega 接近上界，
    说明湍流场从未真正恢复，必须重置到来流初值让 SST 源项重新
    建立平衡。

    真实 bug 修复（2026-09-05，cube_demo 791,492 单元真实网格
    `solve resume` 长程验证决定性发现）：本节注释一直描述的
    判据是"统计有多大比例的单元被钳制在上界附近"，但下面这段
    代码此前从未真正这样算过——只拿 k_mean 和一个
    max(1%*k_max, 10*k_inf) 公式比较，是对注释意图的错误实现。
    真实复现：cube_demo 这类强分离钝体绕流（尾流/剪切层湍流度
    远高于来流），健康、充分发展的 k 场 k_mean=38.11（是
    k_inf=0.167 的 228 倍），但 reset_threshold=max(5.55,1.67)
    =5.55——k_mean 远超这个阈值，被误判成"爆炸残留"，
    resume 第一次调用 solver.solve() 就把整个 k_field/
    omega_field 直接清零重置回自由流初值，销毁了几千步真实
    演化出的湍流场（同一批 solve resume 直接调用
    rebuild_solver_from_checkpoint+手动 solver.step() 不经过
    这段 resumed 分支时完全正常，交叉验证坐实了问题就在这里）。
    用真实数据核实：这份健康 checkpoint 上，真正被钳制在
    k_max/omega_max 90%以上的单元占比分别只有 4.36%/0.07%，
    远低于注释一直声称的 10% 判据——现在改成真正按这个比例
    判断，而不是看均值。

    ## 分布式下的正确性

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
    from autoflowcfd.core.mpi import is_root
    from autoflowcfd.core.mpi.comm import allreduce_sum

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
