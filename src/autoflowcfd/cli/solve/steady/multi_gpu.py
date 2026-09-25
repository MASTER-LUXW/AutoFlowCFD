"""AutoFlowCFD V2.0 - `solve steady` 的多 GPU + MPI 分布式（传统模式 / 完全分布式加载）后端分支。

从 `cli/solve/steady.py` 拆出（2026-09-25，项目「单文件不超 500 行」规范）；参数由
`command.py` 在公共前段（物理常数解析与范围校验）之后逐名传入。
"""

import click

from autoflowcfd.cli.solve.wall_distance import wall_distance_source_if_needed
from autoflowcfd.cli.solve.helpers import load_mesh_for_solver


def _run_multi_gpu(
    *,
    use_eikonal,
    aoa_deg, aos_deg, cfl_max, cfl_min, cfl_start, checkpoint_interval,
    fully_distributed, gpu_device, input_file, max_iter, mu_molecular, n_ranks,
    order, output_dir, p_inf, phase_max_iter, residual_drop_threshold, rho_inf,
    skip_quality_check, surface_mesh, turbulence_intensity, turbulence_model,
    vel_inf, viscosity_ratio,
):
    """`solve steady` 的多 GPU + MPI 分布式（传统模式 / 完全分布式加载）路径。"""
    # 多 GPU + MPI 分布式路径
    from autoflowcfd.core.gpu import gpu_available
    if not gpu_available:
        print("\n❌ CuPy not available. Install with: pip install cupy-cuda12x")
        raise click.Abort()

    from autoflowcfd.core.mpi import mpi_available, is_root
    if not mpi_available:
        print("\n❌ MPI not available. Install mpi4py and run with mpirun.")
        raise click.Abort()

    from autoflowcfd.core.gpu.distributed.gpu_distributed import MultiGPUDistributedSolver
    from autoflowcfd.fr.operators import generate_fr_operators

    if fully_distributed:
        # 多 GPU"完全分布式加载"（#1，2026-09-02 实现——此前只有
        # "传统模式"，见 MultiGPUDistributedSolver.from_fully_
        # distributed_package/gpu_distributed_fully_distributed.py
        # 模块文档）：只有 root rank 加载完整网格，root 预先按每个
        # rank 的 compact 索引空间切好紧凑包再分发，与 CPU
        # `--fully-distributed`（不加 --multi-gpu）同一套
        # `distributed_mesh_load_v2`/`build_fully_distributed_rank_
        # package`，只是构造出的是 GPU 常驻状态的求解器。
        from autoflowcfd.core.mpi.distributed_mesh_loader import distributed_mesh_load_v2
        from autoflowcfd.core.fr_solver.mach_ref import resolve_mach_ref

        freestream = {"rho_inf": rho_inf, "vel_inf": vel_inf,
                      "p_inf": p_inf,
                      "aoa_deg": aoa_deg, "aos_deg": aos_deg}
        mach_ref = resolve_mach_ref(rho_inf, vel_inf, p_inf)
        package, root_context = distributed_mesh_load_v2(
            input_file, order, surface_mesh, n_ranks,
            freestream=freestream, mu_molecular=mu_molecular, mach_ref=mach_ref,
            enable_viscous=True, skip_quality_check=skip_quality_check,
            turb_model_name=turbulence_model.upper(),
            use_eikonal=use_eikonal,
            turbulence_intensity=turbulence_intensity, viscosity_ratio=viscosity_ratio,
            cfl_start=cfl_start, cfl_max=cfl_max, cfl_min=cfl_min,
        )
        solver = MultiGPUDistributedSolver.from_fully_distributed_package(
            package, n_ranks=n_ranks, device_id=gpu_device, root_context=root_context,
        )
    else:
        mesh, volume_data = load_mesh_for_solver(
            input_file, order, surface_mesh=surface_mesh, skip_quality_check=skip_quality_check
        )
        ops = generate_fr_operators(order)

        solver = MultiGPUDistributedSolver(
            mesh=mesh, ops=ops, n_ranks=n_ranks,
            device_id=gpu_device,
            mu_molecular=mu_molecular,
            rho_inf=rho_inf, vel_inf=vel_inf, p_inf=p_inf,
            aoa_deg=aoa_deg, aos_deg=aos_deg,
            # 真实 bug 修复（V2.0 专家组盲审发现）：此前从不传 turb_model，
            # --turbulence-model 无论填什么都被静默丢弃、恒定跑层流，
            # 终端打印的 Turbulence 行却仍显示用户输入的模型名。
            turb_model=turbulence_model.upper(),
            wall_distance_source=wall_distance_source_if_needed(turbulence_model, volume_data, use_eikonal),
            # SST 分布式湍流真正接入后（2026-09-02）才需要这两个值算
            # k_inf/omega_inf——此前 turb_model 恒被拒绝，这两个 CLI
            # 选项从未真正传到这里过，与单机 GPU 路径（上面
            # GPUFRSolver 构造处）保持一致。
            turbulence_intensity=turbulence_intensity,
            viscosity_ratio=viscosity_ratio,
            # 真实缺口修复（2026-09-15）：`--cfl-start/--cfl-max/--cfl-min`
            # 此前在**全部分布式路径**上被静默丢弃（这四处构造点都不传），
            # 与本文件上方注释记录过的 turb_model/turbulence_intensity
            # 同类。多 GPU 传统模式有自适应控制器（见 gpu_distributed.py
            # 里 _cfl_controller 构造处），必须一并透传。
            cfl_start=cfl_start, cfl_max=cfl_max, cfl_min=cfl_min,
        )

    # 中间 checkpoint 保存回调（2026-09-02 补齐——此前 output_interval
    # 只控制进度打印，跑到一半崩溃/被杀会丢失全部进度，与单机路径
    # `_checkpoint_cb` 同一个设计，见 solver.solve/save_checkpoint_
    # distributed 文档"完成度"一节说明）。
    def _multi_gpu_checkpoint_cb(solver_ref, iteration):
        if iteration % checkpoint_interval != 0:
            return
        try:
            # order/target_order 分离（2026-09-02，Order Continuation
            # 接入多GPU分布式路径后补齐——与 CPU 分布式
            # _distributed_checkpoint_cb 同一处修复同一个理由）：
            # 爬坡阶段中途的 checkpoint 必须记录
            # solver_ref.current_order（U_sps 实际形状），不是固定
            # 的目标 order。
            saved_path = solver_ref.save_checkpoint_distributed(
                output_dir, iteration, input_file,
                solver_ref.current_order, turbulence_model, backend="gpu",
                target_order=solver_ref.order, surface_mesh=surface_mesh,
            )
            if saved_path and is_root():
                print(f"   [Checkpoint] iter {iteration} saved: {saved_path}")
        except Exception as e:
            if is_root():
                print(f"   [Checkpoint] Warning: save failed at iter {iteration}: {e}")

    try:
        result = solver.solve(
            max_iter=max_iter, dt=1e-3, tol=1e-6,
            checkpoint_callback=_multi_gpu_checkpoint_cb,
            phase_max_iter=phase_max_iter, residual_drop_threshold=residual_drop_threshold,
        )
        print(f"\n✅ Multi-GPU Simulation Finished")
        # #4（2026-08-28）：此前这里从不保存结果——分布式 checkpoint
        # save/load 依赖的 self.U_gpu 本地尺寸缺陷（#1）修复之前，
        # 保存也没有意义，见 MultiGPUDistributedSolver.
        # save_checkpoint_distributed 文档。
        saved_path = solver.save_checkpoint_distributed(
            output_dir, max_iter, input_file,
            solver.current_order, turbulence_model, backend="gpu",
            target_order=solver.order, surface_mesh=surface_mesh,
        )
        if saved_path and is_root():
            print(f"   Checkpoint saved: {saved_path}")
        solver.cleanup()
    except Exception as e:
        print(f"\n❌ Multi-GPU Simulation Failed: {e}")
        raise click.Abort()
