"""求解命令的结果/checkpoint 持久化辅助函数 —— 从 solve_helpers.py 拆出，控制单文件行数。

见 solve_helpers.py 文档说明整体拆分结构。
"""

import os
import pickle
from typing import Optional

import click


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


def restore_state_from_checkpoint(
    checkpoint_path: str,
    solver,
    metadata: dict,
) -> int:
    """从 checkpoint 恢复求解器状态，用于 `solve transient --init-from` 以稳态结果
    为初场启动瞬态仿真（典型工作流：先跑稳态 SST 收敛到平衡态，再用 DDES/LES
    从该流场启动瞬态计算——避免从均匀流场直接启动 DES 需要极长的瞬态发展时间）。

    处理 n_vars 不匹配的情况：稳态 SST 与瞬态 DDES 都是 7 变量（rho/rho_u/rho_v/
    rho_w/rho_e/rho_k/rho_omega），直接拷贝；稳态 `none`（5 变量）→ 瞬态 DDES
    （7 变量）时，前 5 个守恒变量直接拷贝，湍流量（rho_k/rho_omega）用自由来流
    默认值初始化（k=1e-6, omega=1e-2，与 FRState.initialize_uniform 的默认值
    一致）；反之稳态 SST（7 变量）→ 瞬态 LES（5 变量）时只取前 5 个。

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

    U_ckpt = fields["U_sps"]
    n_vars_ckpt = U_ckpt.shape[2] if U_ckpt.ndim == 3 else 0
    n_vars_solver = solver.state.n_vars
    ckpt_iter = metadata.get("iteration", 0)

    # 形状校验：n_cells 和 n_sps 必须一致（网格/阶数不匹配）
    if U_ckpt.shape[0] != solver.state.n_cells or U_ckpt.shape[1] != solver.state.n_sps:
        raise click.ClickException(
            f"Checkpoint 状态形状 {U_ckpt.shape} 与重建求解器的状态形状 "
            f"{solver.state.U.shape} 不匹配（网格或阶数可能已变化），拒绝恢复。"
        )

    if n_vars_ckpt == n_vars_solver:
        # 变量数一致：直接整体拷贝（最常见路径：稳态 SST→瞬态 DDES 都是 7 vars）
        solver.state.U = U_ckpt.copy()
        print(f"   ✅ 从 checkpoint 恢复完整状态 ({n_vars_ckpt} vars)")
    elif n_vars_ckpt < n_vars_solver:
        # checkpoint 变量少于新求解器：拷贝流体部分，湍流量用自由来流默认值初始化
        solver.state.U[:, :, :n_vars_ckpt] = U_ckpt
        if n_vars_solver > 5:
            # 用自由来流条件初始化湍流量（与 FRState.initialize_uniform 默认值一致）
            rho_inf = solver.freestream.get("rho_inf", 1.225)
            solver.state.U[:, :, 5] = rho_inf * 1e-6   # rho*k
            solver.state.U[:, :, 6] = rho_inf * 1e-2   # rho*omega
        print(f"   ✅ 从 checkpoint 恢复流体场 ({n_vars_ckpt} vars)，"
              f"湍流量用自由来流默认值初始化（新求解器需要 {n_vars_solver} vars）")
    else:
        # checkpoint 变量多于新求解器：只取前 n_vars_solver 个（如稳态 SST→瞬态 LES）
        solver.state.U = U_ckpt[:, :, :n_vars_solver].copy()
        print(f"   ✅ 从 checkpoint 恢复前 {n_vars_solver} 个变量（checkpoint 有 "
              f"{n_vars_ckpt} vars，新求解器只需 {n_vars_solver}）")

    solver.state._update_primitives()
    return ckpt_iter


def rebuild_solver_from_checkpoint(
    checkpoint_path: str,
    backend: Optional[str] = None,
    surface_mesh: Optional[str] = None,
    threads: int = -1,
    reference_area: Optional[float] = None,
    skip_quality_check: bool = False,
):
    """从 checkpoint 完整重建一个带解场的 FRSolver（不继续迭代）。

    从 `solve resume` 里提炼出的公共重建逻辑：checkpoint 的 metadata 记录了
    重建 FRSolver 所需的全部构造参数（input_file/order/turbulence_model/
    backend/自由来流条件），据此重新走一遍 load_mesh_for_solver + FRSolver(...)
    构造出求解器，再用 checkpoint 里完整保存的 (n_cells,n_sps,n_vars) 状态
    （metadata['fields']['U_sps']）整体替换初始化生成的均匀流场。

    `solve resume` 用它接着跑更多迭代；`post coefficients` 用它在不继续
    迭代的情况下拿到一个状态完整、几何完整（mesh.face_connectivity/
    face_flux_points）的求解器，喂给
    `postprocess.fr_coefficients.compute_aerodynamic_coefficients_fr`
    —— 这是气动系数计算真正需要的输入（FR 原生多点解 + 面几何）。旧版
    V1 `CoefficientCalculator`（假设单元中心 `GridData`/`SolutionVector`、
    依赖从未存在的 `get_face_data()`、系数恒为 0，见 6_整体专家组二次评审.md
    发现 23）已在第三轮评审整改中移除。

    Args:
        checkpoint_path: checkpoint 文件路径（solve steady/transient 产出）
        backend: 后端覆盖，None 时沿用 checkpoint 记录的原始后端
        surface_mesh: 面网格路径覆盖。None 时回退到 checkpoint metadata
            里存的 surface_mesh（write_checkpoint 若拿到了就会存下，见
            该函数文档）；两者都没有、且 input_file 是 .nas 体网格时，
            下面 load_mesh_for_solver 会因缺边界信息直接报错，不会
            静默用错误网格求解
        threads: CPU 后端 numba 并行 kernel 使用的线程数
        skip_quality_check: 跳过重建时的网格质量门检查（B-11）——原求解靠该选项
            才跑得起来时，resume/后处理重建也必须同样跳过；默认 False 仍强制
        reference_area: 气动系数参考面积 (m^2) 覆盖。None 时尝试从
            volume_data.surface_mesh 自动估算（X 方向正投影面积，见
            solve_aero_coefficients._compute_reference_area_auto），
            与 `solve steady` 的同名逻辑一致——真实 bug（2026-08-22）：
            此前只有 solve_steady_command.py 会设置 solver._reference_area
            （run_order_continuation 的每步日志靠这个属性判断要不要打印
            Cd/Cl/Cs），resume 出来的求解器上这个属性完全没设置过，哪怕
            --reference-area 传了，resume 期间的每步日志也永远不会带
            气动力系数，直到 solve() 整个跑完才会通过 resume() 自己那次
            额外的 _report_aerodynamic_coefficients 调用打印一次

    Returns:
        (solver, iteration, metadata): 重建好的 FRSolver 实例（状态已从
        checkpoint 恢复）、checkpoint 记录的迭代数、以及重建所用的完整
        metadata 字典（含 input_file/order/turbulence_model/backend，
        调用方续写 checkpoint 时需要，不必重新加载一遍 checkpoint 文件）

    Raises:
        click.ClickException: checkpoint 缺少 U_sps 字段、缺少 input_file，
            或状态形状与重建求解器不匹配
    """
    from types import SimpleNamespace
    from autoflowcfd.core import FRSolver
    from autoflowcfd.core.utils.checkpoint import CheckpointManager
    from autoflowcfd.cli.solve_mesh_loader import load_mesh_for_solver
    from autoflowcfd.cli.solve_wall_distance import compute_wall_distance_for_solver

    _solution, _history, iteration, metadata = CheckpointManager(
        config=SimpleNamespace(), output_dir="."
    ).load(checkpoint_path)

    fields = metadata.get("fields", {})
    if "U_sps" not in fields:
        raise click.ClickException(
            f"Checkpoint '{checkpoint_path}' 缺少 'U_sps' 字段（完整的 (n_cells,n_sps,n_vars) "
            f"求解器状态）——不是本版本 write_checkpoint 写出的 checkpoint，无法精确重建。"
        )

    input_file = metadata.get("input_file")
    if not input_file:
        raise click.ClickException("Checkpoint metadata 缺少 'input_file'，无法重新加载网格。")

    order = int(metadata.get("order", 2))
    # target_order（Order Continuation 的最终目标阶数，solver.order）与
    # order（checkpoint 保存那一刻的 solver.current_order，决定重建
    # mesh/FRSolver 初始状态要用哪个 n_sps 才能跟保存的 U_sps 形状对上）
    # 是两个独立的量，checkpoint 若是 Order Continuation 爬升到目标阶数
    # 之前存的（例如 P0 阶段中途），二者不相等——见 write_checkpoint 的
    # target_order 参数文档。缺省回退到 order 本身，兼容旧 checkpoint
    # （没有 target_order 字段，那种情况下当时 order 记的就是静态目标
    # 阶数，二者天然相等，回退安全）。
    target_order = int(metadata.get("target_order", order))
    turbulence_model = metadata.get("turbulence_model", "sst")
    target_backend = backend or metadata.get("backend", "cpu")
    resolved_surface_mesh = surface_mesh or metadata.get("surface_mesh")

    mesh, volume_data = load_mesh_for_solver(
        input_file, order, surface_mesh=resolved_surface_mesh,
        # B-11（2026-08-26）：原求解若靠 --skip-quality-check 才跑得起来，
        # resume/post 重建时这里却无条件重新强制质量门，导致同一个网格上产出的
        # checkpoint 永远无法被 resume/后处理，与 solve 侧语义不一致。默认仍然强制。
        skip_quality_check=skip_quality_check,
    )

    solver = FRSolver(
        mesh=mesh,
        backend=target_backend,
        order=order,
        turb_model_name=turbulence_model,
        rho_inf=metadata.get("rho_inf", 1.225),
        vel_inf=metadata.get("vel_inf", 33.33),
        p_inf=metadata.get("p_inf", 101325.0),
        n_threads=threads,
        # Tu/VR 从 checkpoint metadata 恢复（2026-08-25 添加）：
        # 保证 Resume 时湍流场重置用的参数与原始计算一致。
        # 旧 checkpoint 没有这两个字段，回退到默认值（Tu=0.01, VR=5.0）。
        turbulence_intensity=metadata.get("turbulence_intensity", 0.01),
        viscosity_ratio=metadata.get("viscosity_ratio", 5.0),
        # mu_molecular 从 checkpoint metadata 恢复（2026-08-27 补齐）：与
        # 上面 Tu/VR 同一批需要持久化的物理量，此前遗漏——非标准空气工况
        # （--mu-molecular 显式设置过的算例）resume 后会悄悄换回标准海平面
        # 空气粘度 1.8e-5，粘性残差/壁面剪切力全部用错误粘度重新计算。
        mu_molecular=metadata.get("mu_molecular", 1.8e-5),
    )
    # FRSolver.__init__ 用同一个 order 参数同时设置 self.current_order
    # 和 self.order（ramp 目标）——上面为了让 mesh/初始状态形状匹配
    # checkpoint，传的是 checkpoint 时的 current_order，这里把
    # self.order 单独纠正回真正的目标阶数，否则 solve() 里
    # `self.order_continuation_enabled and self.order >= 2` 这个门槛
    # 会被错误地拿 current_order 去判断，P0 checkpoint resume 出来的
    # 求解器会误判目标阶数已经是 0、直接跳过 Order Continuation 的
    # 继续爬升。
    solver.order = target_order
    compute_wall_distance_for_solver(solver, volume_data)

    # 与 solve_steady_command.py 同一段逻辑保持一致（见上面 reference_area
    # 参数文档）：未显式传参数时尝试自动估算，让 resume 期间的每步日志
    # 也能带 Cd/Cl/Cs，不必等到 solve() 整个跑完才看到一次。
    resolved_reference_area = reference_area
    if resolved_reference_area is None:
        from autoflowcfd.cli.solve_aero_coefficients import _compute_reference_area_auto
        resolved_reference_area = _compute_reference_area_auto(volume_data)
    solver._reference_area = resolved_reference_area

    restore_solver_state_from_fields(solver, fields, metadata)

    metadata["order"] = order
    metadata["target_order"] = target_order
    metadata["turbulence_model"] = turbulence_model
    metadata["backend"] = target_backend
    metadata["surface_mesh"] = resolved_surface_mesh
    return solver, iteration, metadata


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
    U_restored = fields["U_sps"]
    if U_restored.shape != solver.state.U.shape:
        raise click.ClickException(
            f"Checkpoint 状态形状 {U_restored.shape} 与重建求解器的状态形状 "
            f"{solver.state.U.shape} 不匹配（网格或阶数可能已变化），拒绝恢复。"
        )
    solver.state.U = U_restored
    solver.state._update_primitives()

    # 湍流场恢复（配套 write_checkpoint 的 k_field/omega_field 持久化，
    # 见该函数文档）：checkpoint 里有就精确恢复，形状必须与刚重建的
    # turb_model 字段一致（否则说明网格/阶数不匹配，同 U_sps 的处理，
    # 拒绝恢复而不是静默截断/广播）；checkpoint 是旧版本写的、没有这两个
    # 字段时，保留 FRSolver 构造时已经生成的均匀初始猜测值，打印警告——
    # 这是此前一直存在的行为，向后兼容，不因为新加了持久化就让旧
    # checkpoint 无法 resume。
    turb_model = getattr(solver, "turb_model", None)
    if turb_model is not None and (hasattr(turb_model, "k_field") or hasattr(turb_model, "omega_field")):
        if "k_field" in fields and "omega_field" in fields:
            k_restored = fields["k_field"]
            omega_restored = fields["omega_field"]
            if k_restored.shape != turb_model.k_field.shape or omega_restored.shape != turb_model.omega_field.shape:
                raise click.ClickException(
                    f"Checkpoint 湍流场形状 k={k_restored.shape}/omega={omega_restored.shape} 与重建求解器的 "
                    f"turb_model 形状 k={turb_model.k_field.shape}/omega={turb_model.omega_field.shape} "
                    f"不匹配（网格或阶数可能已变化），拒绝恢复。"
                )
            turb_model.k_field = k_restored
            turb_model.omega_field = omega_restored
            # 跳过 production ramp（2026-08-25 代码审查）：k/omega 场已精确恢复，
            # 说明湍流已充分发展，再重新压制产生项 50 步会把已收敛的湍流场
            # 往回压。order_continuation.py 的 resume 分支已有同样的跳过逻辑，
            # 但 order_continuation_enabled=False 或 order < 2 时 solver.solve()
            # 走普通循环不经过那里，而 init_turbulence_models 已把重建求解器的
            # _turb_ramp_step 推到 ≈1（production_factor≈0.02）——在这里统一补上。
            if hasattr(turb_model, "production_factor"):
                turb_model.production_factor = 1.0
                solver._turb_ramp_step = getattr(solver, "_turb_production_ramp_steps", 50)
                solver._turb_production_ramp_complete = True
                # 同步置位基准重置完成标记（与 order_continuation.py 的 resume
                # 分支一致）：否则 run_order_continuation 循环里的"ramp 完成 →
                # 重置残差基准"检测会在 resume 后第一步把上面刚恢复的
                # _phase_initial_residual 丢掉。
                solver._ramp_baseline_reset_done = True
        else:
            print("   ⚠️  Checkpoint 缺少 k_field/omega_field（旧版本 checkpoint）："
                  "湍流场从均匀初始猜测值重新开始，与已恢复的平均流场不连续，"
                  "SST 收敛可能需要重新爬升。")

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

    # Order Continuation 阶段起始残差恢复（配套 write_checkpoint 的
    # phase_initial_residual 持久化，见该函数文档）：checkpoint 里有就
    # 恢复到 solver 属性上，供 run_order_continuation 在 resume 恢复出的
    # 第一个阶段用作残差下降判据的种子；旧版本 checkpoint 没有这个字段
    # 时不设置，run_order_continuation 会走向后兼容分支（打印警告，
    # 从这次 resume 的第一步重新捕获，不崩溃）。
    _phase_initial_residual = metadata.get("phase_initial_residual")
    if _phase_initial_residual is not None:
        solver._phase_initial_residual = float(_phase_initial_residual)

    # 标记这个 solver 的状态是从 checkpoint 恢复的真实解、不是构造函数
    # 生成的均匀自由流场占位值——order_continuation.run_order_continuation
    # 用这个标记决定要不要把状态重置回 P0 重新爬升，见该函数文档：真实
    # 复现的 bug（2026-08-22），checkpoint 若是在 P1/P2 阶段中途存的，
    # 不加这个标记会被 run_order_continuation 误判成"刚构造、还没跑过
    # Order Continuation"，把刚恢复的真实解丢弃、替换成均匀自由流场从
    # P0 重新开始整个爬升——恢复等于白恢复，且悄悄发生、resume 不会报
    # 任何错误或警告。
    solver._resumed_from_checkpoint = True


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

    solution_cell_avg = solver.state.U.mean(axis=1)  # (n_cells, n_vars)，供粗粒度消费方使用
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

    metadata = {
        "input_file": input_file,
        "order": order,
        "target_order": target_order if target_order is not None else order,
        "turbulence_model": turbulence_model,
        "backend": backend,
        "n_sps_per_cell": solver.state.n_sps,
        "n_vars": solver.state.n_vars,
        "rho_inf": solver.freestream["rho_inf"],
        "vel_inf": solver.freestream["vel_inf"],
        "p_inf": solver.freestream["p_inf"],
        # Tu/VR 持久化（2026-08-25 添加）：Resume 时必须用原始 Tu/VR 值，
        # 否则会用默认值（Tu=0.01, VR=5.0）覆盖用户设置的值，导致湍流场
        # 重置时用的参数与原始计算不一致。
        "turbulence_intensity": getattr(solver, '_turbulence_intensity', 0.01),
        "viscosity_ratio": getattr(solver, '_viscosity_ratio', 5.0),
        # mu_molecular 持久化（2026-08-27 补齐，与上面 Tu/VR 同一类遗漏）：
        # 见 rebuild_solver_from_checkpoint 里对应恢复处的说明。getattr
        # 兜底与 Tu/VR 同一个理由：轻量 fake/mock solver（单元测试）不一定
        # 设置这个属性，真实 FRSolver/GPUFRSolver 恒会设置。
        "mu_molecular": getattr(solver, 'mu_molecular', 1.8e-5),
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
