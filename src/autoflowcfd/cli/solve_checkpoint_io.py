"""求解命令的结果/checkpoint 持久化辅助函数 —— 从 solve_helpers.py 拆出，控制单文件行数。

见 solve_helpers.py 文档说明整体拆分结构。
"""

import logging
import os
import pickle
from typing import Optional

import click

logger = logging.getLogger(__name__)


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
    —— 这是气动系数计算真正需要的输入（FR 原生多点解 + 面几何），不是
    `postprocess.coefficients.CoefficientCalculator` 假设的 V1 单元中心
    `GridData`/`SolutionVector`（该实现的 `get_face_data()` 从未存在过，
    气动系数恒为 0，见 6_整体专家组二次评审.md 发现 23）。

    Args:
        checkpoint_path: checkpoint 文件路径（solve steady/transient 产出）
        backend: 后端覆盖，None 时沿用 checkpoint 记录的原始后端
        surface_mesh: checkpoint 记录的 input_file 若是 .nas 体网格，
            需要提供原始面网格来反推边界分组
        threads: CPU 后端 numba 并行 kernel 使用的线程数

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
    turbulence_model = metadata.get("turbulence_model", "sst")
    target_backend = backend or metadata.get("backend", "cpu")

    mesh, volume_data = load_mesh_for_solver(input_file, order, surface_mesh=surface_mesh)

    solver = FRSolver(
        mesh=mesh,
        backend=target_backend,
        order=order,
        turb_model_name=turbulence_model,
        rho_inf=metadata.get("rho_inf", 1.225),
        vel_inf=metadata.get("vel_inf", 33.33),
        p_inf=metadata.get("p_inf", 101325.0),
        n_threads=threads,
    )
    compute_wall_distance_for_solver(solver, volume_data)

    U_restored = fields["U_sps"]
    if U_restored.shape != solver.state.U.shape:
        raise click.ClickException(
            f"Checkpoint 状态形状 {U_restored.shape} 与重建求解器的状态形状 "
            f"{solver.state.U.shape} 不匹配（网格或阶数可能已变化），拒绝恢复。"
        )
    solver.state.U = U_restored
    solver.state._update_primitives()

    metadata["order"] = order
    metadata["turbulence_model"] = turbulence_model
    metadata["backend"] = target_backend
    return solver, iteration, metadata


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

    metadata = {
        "input_file": input_file,
        "order": order,
        "turbulence_model": turbulence_model,
        "backend": backend,
        "n_sps_per_cell": solver.state.n_sps,
        "n_vars": solver.state.n_vars,
        "rho_inf": solver.freestream["rho_inf"],
        "vel_inf": solver.freestream["vel_inf"],
        "p_inf": solver.freestream["p_inf"],
    }

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
