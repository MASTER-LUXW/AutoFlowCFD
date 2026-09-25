"""AutoFlowCFD V2.0 - `solve steady` 的单机 CPU后端分支。

从 `cli/solve/steady.py` 拆出（2026-09-25，项目「单文件不超 500 行」规范）；参数由
`command.py` 在公共前段（物理常数解析与范围校验）之后逐名传入。
"""

import click

from autoflowcfd.core import FRSolver
from autoflowcfd.core.time_integration.base import TimeIntegrationScheme
from autoflowcfd.cli.solve.helpers import (
    compute_wall_distance_for_solver,
    load_mesh_for_solver,
    save_results,
    write_checkpoint,
)
from autoflowcfd.cli.solve.aero_coefficients import (
    _compute_reference_area_auto,
    _report_aerodynamic_coefficients,
)


def _run_cpu_single(
    *,
    aoa_deg, aos_deg, artificial_viscosity_alpha, artificial_viscosity_enabled,
    backend, cfl_max, cfl_min, cfl_start, checkpoint_interval,
    entropy_stable_volume_enabled, input_file, max_iter, mu_molecular, order,
    output_dir, p_inf, phase_max_iter, reference_area, residual_drop_threshold,
    rho_inf, sem_num_eddies, skip_quality_check, surface_mesh, threads,
    turbulence_intensity, turbulence_model, use_eikonal, vel_inf, viscosity_ratio,
):
    """`solve steady` 的单机 CPU路径。"""
    # 单机求解器路径：所有 rank 加载完整网格
    mesh, volume_data = load_mesh_for_solver(
        input_file, order, surface_mesh=surface_mesh, skip_quality_check=skip_quality_check,
    )
    # 单机求解器路径（默认）
    solver = FRSolver(
        mesh=mesh,
        backend=backend,
        order=order,
        turb_model_name=turbulence_model,
        time_scheme=TimeIntegrationScheme.SSP_RK3,
        n_threads=threads,
        cfl_start=cfl_start,
        cfl_max=cfl_max,
        cfl_min=cfl_min,
        turbulence_intensity=turbulence_intensity,
        viscosity_ratio=viscosity_ratio,
        sem_num_eddies=sem_num_eddies,
        mu_molecular=mu_molecular,
        rho_inf=rho_inf, vel_inf=vel_inf, p_inf=p_inf,
        aoa_deg=aoa_deg, aos_deg=aos_deg,
        artificial_viscosity_enabled=artificial_viscosity_enabled,
        artificial_viscosity_alpha=artificial_viscosity_alpha,
        entropy_stable_volume_enabled=entropy_stable_volume_enabled,
    )

    # 2.5. 计算壁面距离场（如果湍流模型需要）
    compute_wall_distance_for_solver(solver, volume_data, use_eikonal=use_eikonal)

    # 传递参考面积到求解器，供迭代中输出气动力系数
    # 如果未指定 --reference-area，尝试从面网格自动计算投影面积
    if reference_area is None:
        from autoflowcfd.core.utils.flow_direction import (
            direction_from_freestream,
        )

        # 参考面积必须沿**来流方向**投影（有攻角时按 X 投影会
        # 偏大 1/cos(alpha)，15 度就是 3.5%，直接进 Cd 分母）
        auto_ref_area = _compute_reference_area_auto(
            volume_data,
            direction=direction_from_freestream(solver.freestream))
        if auto_ref_area is not None:
            reference_area = auto_ref_area
    solver._reference_area = reference_area

    # 构建中间 checkpoint 保存回调（每 checkpoint_interval 步保存一次）
    def _checkpoint_cb(solver_ref, iteration):
        if iteration % checkpoint_interval != 0:
            return
        try:
            save_results(solver_ref, output_dir, quiet=True)
            # 必须用 solver_ref.current_order（这一步实际求解用的阶数），
            # 不能用外层闭包捕获的 order（CLI --order，Order Continuation
            # 的最终目标阶数）——真实复现：Order Continuation 还没爬升到
            # 目标阶数时（例如 P0 阶段的中间 checkpoint）两者不相等，用
            # 目标阶数重建 mesh/FRSolver 会得到与 checkpoint 里存的
            # U_sps 形状不匹配的 n_sps，resume 直接报错拒绝恢复。
            write_checkpoint(
                solver_ref, output_dir, iteration,
                input_file, solver_ref.current_order, turbulence_model, backend,
                quiet=True, surface_mesh=surface_mesh, target_order=solver_ref.order,
            )
            print(f"   [Checkpoint] iter {iteration} saved")
        except Exception as e:
            print(f"   [Checkpoint] Warning: save failed at iter {iteration}: {e}")

    # 3. 执行求解
    try:
        result = solver.solve(max_iter=max_iter, dt=1e-3, tol=1e-6,
                              checkpoint_callback=_checkpoint_cb,
                              phase_max_iter=phase_max_iter,
                              residual_drop_threshold=residual_drop_threshold)
        print(f"\n✅ Simulation Finished: Iterations={result.iterations}, Residual={result.final_residual:.6e}")

        # 4. 保存结果（.pkl 全量状态 + HDF5 checkpoint，后者供 solve resume 使用）
        save_results(solver, output_dir)
        # solver.current_order 而非 order：理由同上方 _checkpoint_cb 里的
        # 说明。正常跑完的情况下 Order Continuation 应该已经爬升到目标
        # 阶数、二者相等，但读活的值而不是假设闭包变量仍然成立更稳妥。
        write_checkpoint(
            solver, output_dir, result.iterations, input_file, solver.current_order,
            turbulence_model, backend, surface_mesh=surface_mesh, target_order=solver.order,
        )

        # 5. 气动系数（提供 --reference-area 时）
        _report_aerodynamic_coefficients(solver, reference_area)

    except Exception as e:
        print(f"\n❌ Simulation Failed: {str(e)}")
        raise click.Abort()
