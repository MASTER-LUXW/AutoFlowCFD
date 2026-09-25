"""AutoFlowCFD V2.0 - 落盘：checkpoint 与最终结果

从 `src/autoflowcfd/cli/solve/checkpoint_io.py`(原 614 行)拆出(2026-09-24, 项目"单文件不超 500 行"规范)。**纯搬家, 逻辑未改**。
"""

import os

import pickle

from typing import Optional

from autoflowcfd.core.utils.checkpoint_physics import physics_metadata



def save_results(solver, output_dir: str, quiet: bool = False):
    """
    保存求解结果到指定目录。

    Args:
        solver: FRSolver实例
        output_dir: 输出目录路径
        quiet: 静默模式，不打印详细信息
    """
    os.makedirs(output_dir, exist_ok=True)

    # 保存最终状态
    state_path = os.path.join(output_dir, "final_state.pkl")
    with open(state_path, 'wb') as f:
        pickle.dump({
            'U': solver.state.U,
            'Q': solver.state.Q,
            'n_cells': solver.state.n_cells,
            'n_sps': solver.state.n_sps,
            'n_vars': solver.state.n_vars
        }, f)

    if not quiet:
        print(f"✅ Results saved to: {output_dir}")
        print(f"   - Final state: {state_path}")


def write_checkpoint(
    solver,
    output_dir: str,
    iteration: int,
    input_file: str,
    order: int,
    turbulence_model: str,
    backend: str,
    history: Optional[dict] = None,
    quiet: bool = False,
    surface_mesh: Optional[str] = None,
    target_order: Optional[int] = None,
) -> Optional[str]:
    """把求解器状态写成 HDF5 checkpoint，供 `solve resume` 真正恢复求解
    （V2.0 二次评审 Tier 1 #13/#14：此前 `solve steady/transient` 从不
    写 checkpoint，`solve resume` 因此永远无事可做，`post` 命令组也找
    不到任何 checkpoint 文件）。

    `CheckpointManager.save()` 的 `solution` 参数是 V1 时代遗留的
    `(n_cells, n_vars)` 单元中心格式；FR 的真实解是
    `(n_cells, n_sps, n_vars)` 多点存储，直接拍扁成单元中心值会丢失
    高阶信息，不能作为恢复求解的依据。这里用 `solution` 存一份逐单元
    平均值（供不关心高阶细节、只看大致场分布的场景，如未来的
    `post` 命令），把完整的 `(n_cells,n_sps,n_vars)` 状态通过
    `extra_fields` 整个存下来（HDF5 支持任意形状数组，不需要拍平），
    `resume` 读回时用这份完整状态精确恢复，不是有损重启。

    metadata 里额外存重建 FRSolver 所需的全部构造参数（input_file 用于
    重新加载网格——mesh/face_connectivity 这类对象本身没有放进
    checkpoint，序列化+反序列化整个网格对象比重新跑一遍
    `load_mesh_for_solver` 更脆弱、更没必要）。

    surface_mesh 同样在此存下（当 input_file 是 .nas 体网格、需要靠
    原始面网格反推边界分组时）——此前只存了 input_file，resume 时
    体网格路径能自动带回，但面网格路径每次都得用户手动重新传
    --surface-mesh，两者本该对称：都是"重建这个 checkpoint 需要的
    构造参数"，只有一半被持久化没有道理。传 None（原始求解本就没有
    面网格，例如 input_file 是已经内嵌边界信息的 .pkl）时跳过，不写
    入 metadata——h5py attrs 不接受 None，且 rebuild_solver_from_
    checkpoint 的回退逻辑用 metadata.get() 缺省 None 处理即可，不需要
    显式哨兵值。

    target_order 单独存（默认等于 order，向后兼容不区分两者的旧调用
    方）——真实复现的 bug（2026-08-22）：`order` 参数记录的是这次
    checkpoint 保存时 solver.current_order（用于让 resume 时重建的
    mesh/FRSolver 初始状态形状与保存的 U_sps 对得上，见上面 order 参数
    文档），但 FRSolver 的 order 构造参数同时也决定了 self.order——
    Order Continuation 是否继续爬升的目标阶数判据（solver.py 里
    `self.order >= 2` 的门槛）。Order Continuation 跑到一半（例如 P0
    阶段中途）存的 checkpoint，current_order（0）和真正的目标阶数
    （CLI --order，例如 2）从这一刻起就不再相等——只存一个字段、
    resume 时用它同时驱动"重建形状"和"续跑目标"，两者只要不相等就必然
    有一个是错的：要么形状对不上直接报错拒绝恢复，要么（更隐蔽）形状
    凑巧对上但目标阶数被错当成 current_order，Order Continuation 的
    再触发条件 self.order>=2 悄悄变 False，resume 出来的求解器整个
    跳过阶数爬升逻辑，退化成没有 Drop/P{n} 分阶段日志、也永远不会真正
    提升到目标阶数的普通定阶迭代——真实网格已复现（cube_demo 791k
    单元，P0 checkpoint resume 后日志格式从 "P0 Iter N: ... Drop: ...x"
    变成了普通的 "Iteration N: Residual = ... | Time/step: ...s"，且
    残差在 8.8e6 附近原地打转，不会向 P1 转变）。

    Returns:
        checkpoint 文件路径；h5py 不可用等失败情形返回 None（不中止求解）
    """
    from types import SimpleNamespace
    from autoflowcfd.core.utils.checkpoint import CheckpointManager, H5PY_AVAILABLE

    if not H5PY_AVAILABLE:
        if not quiet:
            print("   ⚠️  h5py not available, skipping checkpoint write (final_state.pkl is still saved)")
        return None

    config = SimpleNamespace(
        mode="steady" if history is None else "transient",
        backend=backend,
        order=order,
        turbulence=turbulence_model,
    )
    manager = CheckpointManager(config, output_dir=output_dir, quiet=quiet)

    # 只对**真实自由度**取平均（2026-09-15 系统性审计）：直接
    # `.mean(axis=1)` 会把 native 四面体的零填充槽位一起算进去，而那些
    # 槽位按约定在初始化时复制真实 SP #0、之后残差行填零/滤波行是单位阵，
    # **永远冻结在初值**（实测推进 10 步后与真实 SP#0 相差 3.4%，order=1
    # 下占一半槽位）。见 fr/native_padding.py::
    # reduce_per_cell_over_real_sps。
    # 注意 `U_sps`（下方 extra_fields）始终是精确的逐 SP 数据，所有内部
    # 消费方（resume、气动力后处理）都强制要求它并在缺失时报错；本字段
    # 只供粗粒度外部消费方使用。
    from autoflowcfd.fr.native_padding import (
        native_tet_n_real_sps, order_from_n_sps, reduce_per_cell_over_real_sps,
    )
    _U_ck = solver.state.U
    _order_ck = order_from_n_sps(_U_ck.shape[1])
    if native_tet_n_real_sps(_order_ck) >= _U_ck.shape[1]:
        # order==0：n_native == n_sps == 1，**根本不存在填充槽位**，掩码
        # 在数学上是恒等操作。走这条分支只是为了不去碰 `solver.mesh`
        # ——P0 的检查点写出路径（含只提供 state/freestream 的调用方）
        # 本来就不需要网格。这不是兜底，是一个可证的无操作。
        solution_cell_avg = _U_ck.mean(axis=1)  # (n_cells, n_vars)
    else:
        solution_cell_avg = reduce_per_cell_over_real_sps(
            _U_ck, solver.mesh.n_prism_cells, _order_ck,
            'mean')  # (n_cells, n_vars)
    extra_fields = {"U_sps": solver.state.U, "Q_sps": solver.state.Q}

    # 湍流场 (k_field/omega_field) 持久化（真实 bug，2026-08-23，用户直接
    # 问"k和omega场在ckpt中没有存储的问题存在吗"发现）：此前只存平均流场
    # U_sps/Q_sps，SSTModelFR.k_field/omega_field 从未写入 checkpoint。
    # resume 时 rebuild_solver_from_checkpoint 走 FRSolver(...) 全新构造，
    # 内部全新 SSTModelFR.__init__ 无条件把湍流场初始化成 k=1e-6/omega=1.0
    # 这个"刚开始求解"的均匀猜测值——resume 出来的求解器因此是"平均流场
    # 精确恢复到收敛态、湍流场却被悄悄打回起点"的不一致状态，物理上不
    # 连续。用 hasattr 而非硬编码 SST，同样覆盖内部复用 SSTModelFR 字段
    # 的 DES 包装；turb_model 为 None（--turbulence none）或不含这两个
    # 属性的湍流模型（如纯 SGS 的 LES/WMLES）时自然跳过，不强行造字段。
    turb_model = getattr(solver, "turb_model", None)
    if turb_model is not None:
        if hasattr(turb_model, "k_field"):
            extra_fields["k_field"] = turb_model.k_field
        if hasattr(turb_model, "omega_field"):
            extra_fields["omega_field"] = turb_model.omega_field
        # nu_t（湍流涡粘度）持久化：此前只存 k/omega，nu_t 在 resume 后
        # 从 FRSolver 构造时的零值重新开始，而粘性残差计算依赖 nu_t
        # （mu_eff = mu_molecular + nu_t）。Checkpoint 时刻 nu_t 已有充分
        # 发展的湍流结构，resume 后 nu_t=0 导致粘性应力突变、残差跳升。
        if hasattr(turb_model, "nu_t") and turb_model.nu_t is not None:
            extra_fields["nu_t"] = turb_model.nu_t

    # `tau_accum`（逐单元累计伪时间）持久化（2026-09-24）：与上面 k/omega/
    # nu_t 同一类 —— resume 会精确恢复物理场、却把这个计数打回 0。
    #
    # 它不影响物理解，但它是**唯一**能回答"物理场到底走了多远"的量，而
    # 残差范数完全不回答这个（`pseudotime_budget.py` 模块文档记录过：
    # plate_demo 上残差单调下降 350 步、物理场只走完绕板特征时间的 1.8%，
    # 那次误判追了好几天）。长程算例正常就是靠 `solve resume` 接力跑的，
    # 计数一重置，这个量恰好在最需要它的场景下失效。
    #
    # 真实踩到过（2026-09-24 本次）：做自适应 CFL 闭环 A/B 时，两臂一个
    # 是原运行、一个是 resume，按"相同迭代步"比较 Cd 得出了"高 CFL 在振荡"
    # 的结论；改按累计伪时间对齐才发现高 CFL 臂只是多走了 5 倍伪时间，
    # 结论完全反了。tau 不跨 resume 延续正是那次差点写错的直接原因。
    _tau = getattr(solver, "tau_accum", None)
    if _tau is not None:
        extra_fields["tau_accum"] = _tau

    metadata = {
        "input_file": input_file,
        "order": order,
        "target_order": target_order if target_order is not None else order,
        "turbulence_model": turbulence_model,
        "backend": backend,
        "n_sps_per_cell": solver.state.n_sps,
        "n_vars": solver.state.n_vars,
        # 时间格式（2026-09-25）：`solve resume` 默认沿用它（隐式稳态续算
        # 不应被静默换回显式格式）。
        "time_scheme": solver.time_integrator.scheme.value,
        # 决定物理解的参数（来流、攻角/侧滑角、粘度、Tu/VR）：与分布式写入端
        # 共用唯一的写入函数，见 core/utils/checkpoint_physics.py。
        **physics_metadata(solver),
    }
    if surface_mesh:
        metadata["surface_mesh"] = surface_mesh

    # Order Continuation 阶段起始残差持久化（2026-08-23，配套
    # order_continuation.py::run_order_continuation 的 resume 状态丢失
    # 修复，见该函数文档）：h5py attrs 不接受 None，未设置时（例如
    # checkpoint_callback 在 run_order_continuation 第一次 solver.step()
    # 之前就被调用——实际不会发生，但防御性地允许缺失）跳过，同
    # surface_mesh 的处理方式一致。
    phase_initial_residual = getattr(solver, "_phase_initial_residual", None)
    if phase_initial_residual is not None:
        metadata["phase_initial_residual"] = float(phase_initial_residual)

    path = manager.save(
        solution_cell_avg,
        history or {"iterations": [iteration]},
        iteration,
        metadata=metadata,
        extra_fields=extra_fields,
    )
    if path and not quiet:
        print(f"   - Checkpoint: {path}")
    return path
