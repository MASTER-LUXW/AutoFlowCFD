"""AutoFlowCFD V2.0 - 把 checkpoint 里的场恢复到一个已存在的 solver 上

从 `src/autoflowcfd/cli/solve/checkpoint_io.py`(原 614 行)拆出(2026-09-24, 项目"单文件不超 500 行"规范)。**纯搬家, 逻辑未改**。
"""




import click
import numpy as np

from autoflowcfd.core.utils.order_continuation.checkpoint_state import restore_phase_state


def restore_state_from_checkpoint(
    checkpoint_path: str,
    solver,
    metadata: dict,
) -> int:
    """从 checkpoint 恢复求解器状态，用于 `solve transient --init-from` 以稳态结果
    为初场启动瞬态仿真（典型工作流：先跑稳态 SST 收敛到平衡态，再用 DDES/LES
    从该流场启动瞬态计算——避免从均匀流场直接启动 DES 需要极长的瞬态发展时间）。

    平均流取 checkpoint 的 5 个守恒变量；湍流模型的输运场与涡粘按模型声明的字段名恢复
    （`restore_turbulence_from_fields`），checkpoint 由另一个湍流模型写出或是层流时保留来流初值。

    Args:
        checkpoint_path: checkpoint 文件路径
        solver: 已创建但尚未开始求解的 FRSolver 实例
        metadata: checkpoint 加载后返回的 metadata 字典（含 fields 键）

    Returns:
        checkpoint 记录的迭代数（供调用方打印日志）

    Raises:
        click.ClickException: checkpoint 缺少 U_sps 字段或形状不兼容
    """
    fields = metadata.get("fields", {})
    if "U_sps" not in fields:
        raise click.ClickException(
            f"Checkpoint '{checkpoint_path}' 缺少 'U_sps' 字段（完整的 (n_cells,n_sps,n_vars) "
            f"求解器状态）——不是本版本 solve_helpers.write_checkpoint 写出的 checkpoint，"
            f"无法精确恢复。"
        )

    ckpt_iter = metadata.get("iteration", 0)
    # 2026-10-04 以前这里只拷状态数组、从不恢复模型上的湍流场——"稳态 SST 收敛 -> --init-from 启动
    # DDES"的湍流场其实是来流初值。
    solver.state.U = mean_flow_state_from_checkpoint(fields["U_sps"], solver.state)
    solver.state._update_primitives()
    restored = restore_turbulence_from_fields(solver, fields)
    print("   ✅ 从 checkpoint 恢复平均流场" + ("与湍流场" if restored else ""))
    return ckpt_iter


def mean_flow_state_from_checkpoint(U_ckpt, state) -> np.ndarray:
    """checkpoint 的 `U_sps` -> 求解器状态（平均流守恒变量，`fr_solver/state.py::N_MEAN_FLOW_VARS`）。

    2026-10-05 以前 CPU 单机 SST 族的状态带两个从未更新的 k/omega 槽位，那时写出的 checkpoint 是 7 列；
    只取前 5 列（湍流场按模型声明的字段名另存，见 `restore_turbulence_from_fields`）。

    Raises:
        click.ClickException: 单元数/解点数不符（网格或阶数已变化）或不足 5 列
    """
    from autoflowcfd.core.fr_solver.state import N_MEAN_FLOW_VARS

    U_ckpt = np.asarray(U_ckpt)
    if U_ckpt.ndim != 3 or U_ckpt.shape[:2] != state.U.shape[:2] or U_ckpt.shape[2] < N_MEAN_FLOW_VARS:
        raise click.ClickException(
            f"Checkpoint 状态形状 {U_ckpt.shape} 与重建求解器的状态形状 "
            f"{state.U.shape} 不匹配（网格或阶数可能已变化），拒绝恢复。"
        )
    return np.ascontiguousarray(U_ckpt[:, :, :N_MEAN_FLOW_VARS])


def restore_turbulence_from_fields(solver, fields: dict) -> bool:
    """由 checkpoint 字段恢复湍流模型的输运场与涡粘（`solve resume` 与 `--init-from` 共用）。

    字段名取模型声明的 `TRANSPORTED_FIELDS`（`write_checkpoint` 按同一组名字写）。checkpoint 是
    另一个湍流模型写的（例如稳态 SST -> 瞬态 SA）或旧版本没有这些字段时，保留构造时的来流初值
    并提示。

    Returns:
        是否恢复了输运场

    Raises:
        click.ClickException: 字段形状与求解器当前几何不匹配
    """
    # 形状必须与刚重建的 turb_model 字段一致（否则说明网格/阶数不匹配，同 U_sps 的处理，
    # 拒绝恢复而不是静默截断/广播）。
    turb_model = getattr(solver, "turb_model", None)
    names = getattr(turb_model, "TRANSPORTED_FIELDS", ())
    if turb_model is None or not names:
        return False
    if not all(name in fields for name in names):
        # 涡粘也不恢复：它是另一个模型（或层流）的量，与留在来流初值的输运场不一致
        print(f"   ⚠️  Checkpoint 缺少湍流场 {'/'.join(names)}（旧版本 checkpoint、层流或另一个湍流模型）："
              "湍流场从来流初值重新开始，与已恢复的平均流场不连续，收敛可能需要重新爬升。")
        return False
    for name, current in zip(names, turb_model.transported_fields()):
        if fields[name].shape != current.shape:
            raise click.ClickException(
                f"Checkpoint 湍流场 {name} 形状 {fields[name].shape} 与重建求解器的 turb_model "
                f"形状 {current.shape} 不匹配（网格或阶数可能已变化），拒绝恢复。"
            )
    turb_model.restore_transported([np.asarray(fields[name]) for name in names])
    # 跳过 production ramp（2026-08-25 代码审查）：k/omega 场已精确恢复，
    # 说明湍流已充分发展，再重新压制产生项 50 步会把已收敛的湍流场
    # 往回压。order_continuation.py 的 resume 分支已有同样的跳过逻辑，
    # 但 order_continuation_enabled=False 或 order < 2 时 solver.solve()
    # 走普通循环不经过那里，而 init_turbulence_models 已把重建求解器的
    # _turb_ramp_step 推到 ≈1（production_factor≈0.02）——在这里统一补上。
    if hasattr(turb_model, "production_factor"):
        turb_model.production_factor = 1.0
        solver._turb_ramp_step = solver._turb_production_ramp_steps
        solver._turb_production_ramp_complete = True
        # 同步置位基准重置完成标记（与 order_continuation.py 的 resume
        # 分支一致）：否则 run_order_continuation 循环里的"ramp 完成 →
        # 重置残差基准"检测会在 resume 后第一步把上面刚恢复的
        # _phase_initial_residual 丢掉。
        solver._ramp_baseline_reset_done = True

    # nu_t 恢复（配套 write_checkpoint 的 nu_t 持久化）：
    # checkpoint 里有就精确恢复，没有时保留 FRSolver 构造时的零值。
    # nu_t 直接影响粘性残差（mu_eff = mu + nu_t），缺失会导致
    # resume 后第一步粘性应力突变、残差跳升。
    if hasattr(turb_model, "nu_t"):
        if "nu_t" in fields:
            nu_t_restored = fields["nu_t"]
            if turb_model.nu_t is not None and nu_t_restored.shape != turb_model.nu_t.shape:
                raise click.ClickException(
                    f"Checkpoint nu_t 形状 {nu_t_restored.shape} 与重建求解器的 "
                    f"nu_t 形状 {turb_model.nu_t.shape} 不匹配，拒绝恢复。"
                )
            turb_model.nu_t = nu_t_restored
        elif turb_model.nu_t is not None:
            print("   ⚠️  Checkpoint 缺少 nu_t（旧版本 checkpoint）："
                  "涡粘度从零重新开始，粘性残差可能短暂跳升。")
    return True


def restore_solver_state_from_fields(solver, fields: dict, metadata: dict) -> None:
    """把 checkpoint 的 (U_sps/k_field/omega_field/nu_t/...) 字段原地灌入一个

    **几何已经就绪**的 FRSolver（mesh/ops/turb_model 已构造好），只替换
    状态数组，不重新加载网格/重建面几何。

    从 `rebuild_solver_from_checkpoint` 里提炼出来（原先内联在那里，
    是它跟"重新加载网格+构造 FRSolver"绑在一起的唯一原因是历史实现
    没有拆分）：`post transient-mean/transient-rms` 要在同一个 case 的
    全部 checkpoint 上依次计算气动系数时间平均（P-03），如果每个
    checkpoint 都重新走一遍 `rebuild_solver_from_checkpoint`（重新加载
    网格、重建 Flux Points 几何），在真实网格上单次就要 1~2 分钟——
    多个 checkpoint 线性相乘会让这条命令实际不可用。真正需要按
    checkpoint 变化的只有这里灌入的状态数组，网格/算子/gh ost provider
    在同一个 case 内完全不变，构造一次、状态原地替换即可。

    Args:
        solver: 已经完整构造好的 FRSolver（geometry/turb_model 就绪）
        fields: checkpoint metadata['fields'] 字典
        metadata: checkpoint 的完整 metadata 字典（读 phase_initial_residual）

    Raises:
        click.ClickException: 状态形状与 solver 当前几何不匹配
    """
    solver.state.U = mean_flow_state_from_checkpoint(fields["U_sps"], solver.state)
    solver.state._update_primitives()

    restore_turbulence_from_fields(solver, fields)

    # `tau_accum` 恢复（2026-09-24，配套 write_checkpoint 的持久化，见
    # 那边的完整理由）：`_tau_accum_seeded` 这个标记让 `FRSolver.solve()`
    # 知道"这次的 tau 不是从零开始的"，从而不把它清掉 —— `solve()` 里那句
    # `self.tau_accum = None` 的原注释写的是"tau_accum 的语义是本次求解
    # 调用已推进的伪时间"，那个语义对 resume 接力的长程算例是错的：它要
    # 回答的是"物理场走了多远"，而物理场是跨 resume 延续的。
    #
    # 旧版本 checkpoint 没有这个字段时什么都不做：`solve()` 照旧从零起算，
    # 与改动前行为完全一致（只是那份 checkpoint 的 tau 信息已经丢了，
    # 无法追回，不假装有）。
    if "tau_accum" in fields:
        _tau = fields["tau_accum"]
        _n_cells = int(getattr(solver.mesh, "n_cells", 0) or 0)
        if _n_cells and getattr(_tau, "shape", (0,))[0] != _n_cells:
            print(f"   ⚠️  Checkpoint tau_accum 长度 {_tau.shape} 与网格单元数 "
                  f"{_n_cells} 不符，跳过恢复（伪时间预算将从零起算）。")
        else:
            solver.tau_accum = _tau
            solver._tau_accum_seeded = True

    # Order Continuation 阶段起始残差（与分布式恢复端共用，见 order_continuation/checkpoint_state.py）
    restore_phase_state(solver, metadata)

    # 标记这个 solver 的状态是从 checkpoint 恢复的真实解、不是构造函数
    # 生成的均匀自由流场占位值——order_continuation.run_order_continuation
    # 用这个标记决定要不要把状态重置回 P0 重新爬升，见该函数文档：真实
    # 复现的 bug（2026-08-22），checkpoint 若是在 P1/P2 阶段中途存的，
    # 不加这个标记会被 run_order_continuation 误判成"刚构造、还没跑过
    # Order Continuation"，把刚恢复的真实解丢弃、替换成均匀自由流场从
    # P0 重新开始整个爬升——恢复等于白恢复，且悄悄发生、resume 不会报
    # 任何错误或警告。
    solver._resumed_from_checkpoint = True
