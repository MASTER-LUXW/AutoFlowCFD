"""AutoFlowCFD V2.0 - 求解入口：稳态、瞬态、从 checkpoint 续算

从 `src/autoflowcfd/api.py` 的 `AutoFlowCFDAPI` 拆出（2026-09-24，项目「单文件不超
500 行」规范）。mixin 是本仓库既有惯例（`_SolverGeometryMixin`、
`_GPUSolverInitMixin` 等），沿用它而不是另发明一套。

**只含方法，没有状态**：全部属性由 `AutoFlowCFDAPI` 的 `__init__` 建立，
这里通过 `self` 访问。
"""

from typing import Optional, Any
from loguru import logger
from autoflowcfd.grid.structures import VolumeMeshData
from autoflowcfd.core import FRSolver, TransientSolver  # 从core模块导入TransientSolver
from autoflowcfd.config.solver_config import SteadyConfig, TransientConfig
from autoflowcfd.api_config import api_resume_simulation
from .helpers import _turbulence_model_str


class _APISolveMixin:
    """求解入口：稳态、瞬态、从 checkpoint 续算"""

    def run_steady(
        self,
        volume_mesh: VolumeMeshData,
        backend: Optional[str] = None,
        order: Optional[int] = None,
        turbulence_model: Optional[str] = None,
        max_iter: Optional[int] = None,
        dt: float = 1e-4,
        tol: float = 1e-6,
        rho_inf: Optional[float] = None,
        vel_inf: Optional[float] = None,
        p_inf: Optional[float] = None,
        threads: int = -1,
        output_dir: str = "./results",
        config: Optional[SteadyConfig] = None,
        phase_max_iter: Optional[int] = None,
        residual_drop_threshold: Optional[float] = None,
        **kwargs
    ) -> Any:
        """Run steady-state FR simulation.

        Args:
            volume_mesh: VolumeMeshData（generate_volume_mesh 的输出，不是
                load_grid 返回的表面 GridData——V2 求解器需要体网格）
            backend, order, turbulence_model, max_iter, rho_inf, vel_inf,
                p_inf: None（默认）时按"`config`（若提供）里的同名字段 >
                内建默认值"解析；显式传值始终优先于 `config`。真实修复
                （V2.0 专家组盲审发现，2026-08-28）：此前这些形参恒为
                字面默认值，`config` 参数根本不存在——`create_steady_
                config()`/`ConfigLoader` 能构建出完整校验过的
                `SteadyConfig`，但没有任何路径把它真正喂给这里，配置
                对象是自洽但对求解行为完全没有影响的孤儿数据结构。
            config: 可选的 `SteadyConfig`（`create_steady_config()`/
                `load_solver_config()` 的返回值）。提供时，上面几个显式
                形参未被调用方传值（即仍是 None）的，改用它的同名字段
                （`turbulence_model` 对应 `config.turbulence`，需要按
                `TurbulenceModel` 枚举做取值映射，见模块内
                `_turbulence_model_str`；其余字段名完全一致）；
                `mu_molecular`/`turbulence_intensity`/`viscosity_ratio`
                这三个 `SolverConfig` 基类字段同理，未显式经由 `**kwargs`
                传入时会从 `config` 补上，再透传给 FRSolver 构造函数。
            phase_max_iter, residual_drop_threshold: None（默认）时按
                "config（若提供）里的同名字段 > 内建默认值"解析，与
                order/max_iter 等参数同一套优先级规则；直接透传给
                `FRSolver.solve()`，见该方法/`order_continuation.
                run_order_continuation` 同名参数文档。
            dt, tol: 时间步长与收敛容差，直接透传给 FRSolver.solve()
            threads: CPU 后端 numba 并行线程数
            output_dir: Output directory
            **kwargs: 其余参数透传给 FRSolver 构造函数
                （例如 mu_molecular/dual_time_inner_iter/bc_overrides；
                config 提供时，这三个物理量的 config 字段仅在 kwargs 里
                没有同名键时才会被补入）

        Returns:
            SolverResult: Simulation result object（同时把 solver 存在
            self.solver，供 calculate_coefficients/export_vtk 使用）

        此前这里直接把 `grid_data`（表面网格）传给 `FRSolver(mesh=grid_data,
        ...)`——但 FRSolver 要求的 `mesh` 是 HighOrderMesh（真正的高阶
        网格对象，带 face_connectivity/sps_coords 等 FR 求解需要的一切），
        不是原始 GridData/VolumeMeshData；且随后 `solver.solve()` 调用
        （在 resume_simulation 路径里还多传了 initial_solution/
        start_iteration 两个 FRSolver.solve() 根本不接受的参数）在当前
        V2 FR 架构下必然出错。改为镜像 CLI `solve steady`
        （cli/solve/steady.py）的真实构造流程：先用
        HighOrderMesh.load_from_volume_mesh 把体网格升格成高阶网格，
        再构造 FRSolver，需要湍流模型时补上壁面距离场
        （V2.0 专家组评审逐行核实：此前的实现从未被真正跑通过）。
        """
        logger.info("Starting steady-state FR simulation")

        from autoflowcfd.grid.high_order.high_order_mesh import HighOrderMesh
        from autoflowcfd.cli.solve.wall_distance import compute_wall_distance_for_solver

        backend = backend if backend is not None else (config.backend.value if config is not None else "cpu")
        order = order if order is not None else (config.order if config is not None else 2)
        if turbulence_model is not None:
            pass
        elif config is not None:
            turbulence_model = _turbulence_model_str(config.turbulence)
        else:
            turbulence_model = "sst"
        max_iter = max_iter if max_iter is not None else (config.max_iter if config is not None else 1000)
        rho_inf = rho_inf if rho_inf is not None else (config.rho_inf if config is not None else 1.225)
        vel_inf = vel_inf if vel_inf is not None else (config.vel_inf if config is not None else 33.33)
        p_inf = p_inf if p_inf is not None else (config.p_inf if config is not None else 101325.0)
        phase_max_iter = phase_max_iter if phase_max_iter is not None else (config.phase_max_iter if config is not None else None)
        residual_drop_threshold = residual_drop_threshold if residual_drop_threshold is not None else (config.residual_drop_threshold if config is not None else 1e2)
        if config is not None:
            for field in ("mu_molecular", "turbulence_intensity", "viscosity_ratio"):
                kwargs.setdefault(field, getattr(config, field))
            # SteadyConfig.cfl_init/cfl_max -> FRSolver.cfl_start/cfl_max
            # （2026-09-07：此前这两个 config 字段从未真正接到求解器上）。
            if getattr(config, "cfl_init", None) is not None:
                kwargs.setdefault("cfl_start", config.cfl_init)
            if getattr(config, "cfl_max", None) is not None:
                kwargs.setdefault("cfl_max", config.cfl_max)
            # cfl_min（2026-09-15）：配置层此前没有这个字段，于是 YAML 用户
            # 到不了低于控制器默认下限 0.05 的工作点——而真 P1 实测稳定的
            # CFL 在 0.03 量级。见 SteadyConfig.cfl_min 文档。
            if getattr(config, "cfl_min", None) is not None:
                kwargs.setdefault("cfl_min", config.cfl_min)

        mesh = HighOrderMesh(order=order)
        mesh.load_from_volume_mesh(volume_mesh)

        solver = FRSolver(
            mesh=mesh,
            backend=backend,
            order=order,
            turb_model_name=turbulence_model,
            rho_inf=rho_inf, vel_inf=vel_inf, p_inf=p_inf,
            n_threads=threads,
            **kwargs,
        )
        compute_wall_distance_for_solver(solver, volume_mesh)

        result = solver.solve(
            max_iter=max_iter, dt=dt, tol=tol,
            phase_max_iter=phase_max_iter,
            residual_drop_threshold=residual_drop_threshold,
        )
        self.solver = solver

        logger.info(
            f"Simulation complete: {result.iterations} iterations, "
            f"converged={result.converged}"
        )

        return result

    def run_transient(
        self,
        volume_mesh: VolumeMeshData,
        backend: Optional[str] = None,
        order: Optional[int] = None,
        time_method: str = "rk3",
        turbulence_model: Optional[str] = None,
        mode: str = None,
        physical_time: Optional[float] = None,
        dt: Optional[float] = None,
        tol: float = 0.0,
        rho_inf: Optional[float] = None,
        vel_inf: Optional[float] = None,
        p_inf: Optional[float] = None,
        threads: int = -1,
        output_dir: str = "./transient_results",
        config: Optional[TransientConfig] = None,
        phase_max_iter: Optional[int] = None,
        residual_drop_threshold: Optional[float] = None,
        **kwargs
    ) -> Any:
        """Run transient FR simulation (DES/LES).

        Args:
            volume_mesh: VolumeMeshData（generate_volume_mesh 的输出）
            backend, order, turbulence_model, physical_time, dt, rho_inf,
                vel_inf, p_inf: None（默认）时按"`config`（若提供）里的
                同名字段 > 内建默认值"解析，显式传值始终优先——与
                run_steady 的 `config` 参数同一套规则（见该方法文档）。
                `physical_time` 对应 `config.total_time`。
            time_method: 时间推进方案，与 core.time_integration.base.
                TimeIntegrationScheme 的取值对齐：
                "rk3"（默认，SSP_RK3）/"imex"（IMEX_EULER）/
                "dual-time"（DUAL_TIME）/"forward_euler"
            turbulence_model: Turbulence model (none/sst/ddes/wmles/les)
            mode: turbulence_model 的别名（向后兼容）
            physical_time: 总物理时间（秒）；未提供时按 dt*1000 估算迭代数
            dt: 时间步长
            tol: 收敛容差（瞬态通常传 0.0，跑满 max_iter）
            rho_inf, vel_inf, p_inf: 自由来流条件
            threads: CPU 后端 numba 并行线程数
            output_dir: Output directory
            config: 可选的 `TransientConfig`，见上方参数说明与
                run_steady 同名参数的文档（`mu_molecular`/
                `turbulence_intensity`/`viscosity_ratio` 同样从
                `config` 补入 `kwargs`）。
            phase_max_iter, residual_drop_threshold: 见 run_steady 同名
                参数文档，瞬态同样共用 `FRSolver.solve()`/
                `run_order_continuation` 这一套机制（目标阶数 >= 1 时
                生效，见 order_continuation/policy.py）。
            **kwargs: 其余参数透传给 FRSolver 构造函数

        Returns:
            SolverResult: Simulation result object（同时把 solver 存在
            self.solver）

        此前的实现构造了一整套本项目从未在其他任何地方使用的
        solver_config.TimeIntegrationScheme/TransientConfig 映射链，
        最终仍然是把表面 GridData 直接传给 FRSolver（同 run_steady 的
        问题）——见 run_steady 文档字符串。改为与 run_steady 相同的
        HighOrderMesh 构造流程，time_method 直接对齐 core 层真正使用
        的 TimeIntegrationScheme 取值，不再引入第二套不兼容的枚举
        （即 C-01/S-05 指出的双枚举不兼容问题的源头之一）。
        """
        logger.info("Starting transient FR simulation")

        from autoflowcfd.grid.high_order.high_order_mesh import HighOrderMesh
        from autoflowcfd.cli.solve.wall_distance import compute_wall_distance_for_solver

        if mode is not None:
            turbulence_model = mode

        backend = backend if backend is not None else (config.backend.value if config is not None else "cpu")
        order = order if order is not None else (config.order if config is not None else 2)
        if turbulence_model is not None:
            pass
        elif config is not None:
            turbulence_model = _turbulence_model_str(config.turbulence)
        else:
            turbulence_model = "sst"
        dt = dt if dt is not None else (config.dt if config is not None else 1e-4)
        if physical_time is None and config is not None:
            physical_time = config.total_time
        rho_inf = rho_inf if rho_inf is not None else (config.rho_inf if config is not None else 1.225)
        vel_inf = vel_inf if vel_inf is not None else (config.vel_inf if config is not None else 33.33)
        p_inf = p_inf if p_inf is not None else (config.p_inf if config is not None else 101325.0)
        phase_max_iter = phase_max_iter if phase_max_iter is not None else (config.phase_max_iter if config is not None else None)
        residual_drop_threshold = residual_drop_threshold if residual_drop_threshold is not None else (config.residual_drop_threshold if config is not None else 1e2)
        if config is not None:
            for field in ("mu_molecular", "turbulence_intensity", "viscosity_ratio"):
                kwargs.setdefault(field, getattr(config, field))
            # 自适应 CFL 三元组（2026-09-17 补齐）。`TransientConfig` 此前
            # 根本没有这三个字段，所以 YAML/config 用户配置不出瞬态的 CFL
            # ——而 `--time-method rk3/imex` 下 `step()` 忽略 dt、按逐单元
            # 局部 CFL 步长推进，那条路径上控制器是**激活**的。字段名映射
            # 与 run_steady 一致（config.cfl_init -> FRSolver.cfl_start）。
            # dual-time 档不构造这个控制器，这三个值对它无效。
            if getattr(config, "cfl_init", None) is not None:
                kwargs.setdefault("cfl_start", config.cfl_init)
            if getattr(config, "cfl_max", None) is not None:
                kwargs.setdefault("cfl_max", config.cfl_max)
            if getattr(config, "cfl_min", None) is not None:
                kwargs.setdefault("cfl_min", config.cfl_min)

        # 词汇->枚举的唯一事实来源在
        # `core/time_integration/base.py::scheme_from_name`（2026-09-18
        # 合并，此前这里、CLI、配置层各有一份，其中配置层那份把
        # dual-time 映到 backward_euler、imex 映到 RK3，都是静默给错值）。
        from autoflowcfd.core.time_integration.base import scheme_from_name
        core_time_scheme = scheme_from_name(time_method)

        max_iter = int(physical_time / dt) if physical_time is not None else 1000

        mesh = HighOrderMesh(order=order)
        mesh.load_from_volume_mesh(volume_mesh)

        solver = TransientSolver(
            mesh=mesh,
            backend=backend,
            order=order,
            turb_model_name=turbulence_model,
            time_scheme=core_time_scheme,
            rho_inf=rho_inf, vel_inf=vel_inf, p_inf=p_inf,
            n_threads=threads,
            **kwargs,
        )
        compute_wall_distance_for_solver(solver, volume_mesh)

        result = solver.solve(
            max_iter=max_iter, dt=dt, tol=tol,
            phase_max_iter=phase_max_iter,
            residual_drop_threshold=residual_drop_threshold,
        )
        self.solver = solver

        logger.info(
            f"Transient simulation complete: {result.iterations} iterations, "
            f"converged={result.converged}"
        )

        return result

    def resume_simulation(self, checkpoint_file: str, **kwargs) -> Any:
        """从检查点恢复仿真。"""
        return api_resume_simulation(self, checkpoint_file, **kwargs)
