"""AutoFlowCFD V2.0 - 从 checkpoint 重建一个全新 solver（含网格/算子重建）

从 `src/autoflowcfd/cli/solve/checkpoint_io.py`(原 614 行)拆出(2026-09-24, 项目"单文件不超 500 行"规范)。**纯搬家, 逻辑未改**。
"""



from typing import Optional

import click
from .restore import restore_solver_state_from_fields


def physics_from_metadata(metadata: dict) -> dict:
    """checkpoint 元数据里决定物理解的参数（单机与分布式 resume 共用的读取端；
    写入端是 `core/utils/checkpoint_physics.py::physics_metadata`）。

    * 来流三要素 `rho_inf/vel_inf/p_inf`：**缺任何一个就报错，不猜**
      （2026-09-24）。此前两条重建路径都写 `metadata.get("vel_inf", 33.33)`
      —— 那是与 CLI 默认值并存的第二份事实来源，按它重建出来的是**另一个
      物理算例**，续算会在错误的来流上静默跑到底。
    * 攻角/侧滑角、分子粘度、Tu/VR：早于各自持久化日期（2026-09-17 /
      08-27 / 08-25）的单机 checkpoint 没有这些键，它们产生时的真实值就是
      那时的内置值，所以缺省按那时的值恢复。分布式 checkpoint 在
      2026-09-25 之前连来流三要素都没写，因此一律在上一条就报错。
    """
    missing = [k for k in ("rho_inf", "vel_inf", "p_inf") if k not in metadata]
    if missing:
        raise click.ClickException(
            "checkpoint 元数据缺少来流参数 " + ", ".join(missing)
            + " —— 无法忠实重建求解器（按默认值猜会得到另一个物理算例，"
            "续算将在错误的来流上静默进行）。单机 checkpoint 缺它们说明来自"
            "早于 V2.0 的格式；分布式 checkpoint 在 2026-09-25 之前从未写入"
            "这组参数。请从头求解。")
    return {
        "rho_inf": float(metadata["rho_inf"]),
        "vel_inf": float(metadata["vel_inf"]),
        "p_inf": float(metadata["p_inf"]),
        "aoa_deg": float(metadata.get("aoa_deg", 0.0)),
        "aos_deg": float(metadata.get("aos_deg", 0.0)),
        "mu_molecular": float(metadata.get("mu_molecular", 1.8e-5)),
        "turbulence_intensity": float(metadata.get("turbulence_intensity", 0.01)),
        "viscosity_ratio": float(metadata.get("viscosity_ratio", 5.0)),
    }


def rebuild_solver_from_checkpoint(
    checkpoint_path: str,
    backend: Optional[str] = None,
    surface_mesh: Optional[str] = None,
    threads: int = -1,
    reference_area: Optional[float] = None,
    skip_quality_check: bool = False,
    cfl_start: float = 0.1,
    cfl_max: float = 0.5,
    cfl_min: float = 0.05,
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
    from autoflowcfd.cli.solve.mesh_loader import load_mesh_for_solver
    from autoflowcfd.cli.solve.wall_distance import compute_wall_distance_for_solver

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
        # 决定物理解的参数一律从 checkpoint 恢复（来流缺失即报错，见
        # physics_from_metadata）；CFL 等纯数值参数由调用方指定。
        **physics_from_metadata(metadata),
        n_threads=threads,
        # CFL（2026-09-07）：resume 时的自适应 CFL 参数由调用方（CLI
        # `--cfl-start`/`--cfl-max`）显式指定，不从 checkpoint 恢复——
        # CFL 是纯数值加速参数、不影响物理解，用户每次 resume 都可以
        # 根据上一段的收敛表现重新调（比如上一段稳定收敛了就调高、
        # 发散了就调低）。
        cfl_start=cfl_start,
        cfl_max=cfl_max,
        # cfl_min 必须能透传（2026-09-15）：控制器默认下限 0.05 高于真 P1
        # 在 79 万单元 cube_demo 上实测稳定的 0.03，resume 低 CFL 工况时
        # 不传它会被钳回 0.05（见 adaptive_cfl.py 模块文档第 11 条）。
        cfl_min=cfl_min,
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
        from autoflowcfd.cli.solve.aero_coefficients import _compute_reference_area_auto
        from autoflowcfd.core.utils.flow_direction import direction_from_freestream

        # 参考面积沿**来流方向**投影（有攻角时按 X 投影会偏大
        # 1/cos(alpha)，15 度就是 3.5%，直接进 Cd 的分母）
        resolved_reference_area = _compute_reference_area_auto(
            volume_data, direction=direction_from_freestream(solver.freestream))
    solver._reference_area = resolved_reference_area

    restore_solver_state_from_fields(solver, fields, metadata)

    metadata["order"] = order
    metadata["target_order"] = target_order
    metadata["turbulence_model"] = turbulence_model
    metadata["backend"] = target_backend
    metadata["surface_mesh"] = resolved_surface_mesh
    return solver, iteration, metadata
