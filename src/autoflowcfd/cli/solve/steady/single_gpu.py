"""AutoFlowCFD V2.0 - `solve steady` 的单 GPU后端分支。

从 `cli/solve/steady.py` 拆出（2026-09-25，项目「单文件不超 500 行」规范）；参数由
`command.py` 在公共前段（物理常数解析与范围校验）之后逐名传入。
"""

import click

from autoflowcfd.cli.solve.helpers import load_mesh_for_solver


def _run_single_gpu(
    *,
    aoa_deg, aos_deg, cfl_max, cfl_min, cfl_start, gpu_device, input_file, max_iter,
    mu_molecular, order, output_dir, p_inf, phase_max_iter, residual_drop_threshold,
    rho_inf, skip_quality_check, surface_mesh, time_scheme, turbulence_intensity,
    turbulence_model, vel_inf, viscosity_ratio,
):
    """`solve steady` 的单 GPU路径。"""
    # 单 GPU 路径
    from autoflowcfd.core.gpu import gpu_available
    if not gpu_available:
        print("\n❌ CuPy not available. Install with: pip install cupy-cuda12x")
        raise click.Abort()

    from autoflowcfd.core.gpu.solver.gpu_solver import GPUFRSolver
    from autoflowcfd.fr.operators import generate_fr_operators

    mesh, volume_data = load_mesh_for_solver(
        input_file, order, surface_mesh=surface_mesh, skip_quality_check=skip_quality_check
    )
    ops = generate_fr_operators(order)

    solver = GPUFRSolver(
        mesh=mesh, ops=ops, order=order,
        device_id=gpu_device,
        time_scheme=time_scheme,
        turbulence_intensity=turbulence_intensity,
        viscosity_ratio=viscosity_ratio,
        mu_molecular=mu_molecular,
        rho_inf=rho_inf, vel_inf=vel_inf, p_inf=p_inf,
        aoa_deg=aoa_deg, aos_deg=aos_deg,
        # 真实 bug 修复（V2.0 专家组盲审发现）：此前从不传 turb_model，
        # --turbulence-model 无论填什么都被静默丢弃、恒定跑层流，
        # 终端打印的 Turbulence 行却仍显示用户输入的模型名。
        turb_model=turbulence_model.upper(),
        # 真实缺口修复（2026-09-14）：`--cfl-start/--cfl-max` 此前只
        # 到得了 CPU 的 FRSolver，GPU 路径连自适应 CFL 控制器都没有、
        # 恒用固定 CFL。GPUFRSolver 现在有了控制器（见
        # core/gpu/solver/gpu_solver.py 里 _cfl_controller 的注释），
        # 这两个 CLI 选项必须一并透传，否则又是一个"选项在 GPU 下被
        # 静默丢弃"的陷阱（与上面 turb_model 那处同类）。
        cfl_start=cfl_start, cfl_max=cfl_max, cfl_min=cfl_min,
    )

    try:
        result = solver.solve(
            max_iter=max_iter, dt=1e-3, tol=1e-6,
            phase_max_iter=phase_max_iter, residual_drop_threshold=residual_drop_threshold,
        )
        print(f"\n✅ GPU Simulation Finished: Iterations={result['iterations']}")

        # 保存结果
        state_cpu = solver.get_state_cpu()
        import pickle, os
        os.makedirs(output_dir, exist_ok=True)
        with open(os.path.join(output_dir, 'final_state.pkl'), 'wb') as f:
            pickle.dump(state_cpu, f)
        solver.cleanup()

    except Exception as e:
        print(f"\n❌ GPU Simulation Failed: {e}")
        raise click.Abort()
