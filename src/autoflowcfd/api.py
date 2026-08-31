"""AutoFlowCFD API（V2.0 纯 FR 架构）。

提供 AutoFlowCFD V2.0 的高层接口，支持网格处理、FR求解和后处理。
"""

import os
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Union, Any
from loguru import logger

# V2.0 Core Import
from autoflowcfd.grid.nas_io.parser import NASParser
from autoflowcfd.grid.mesh_gen.tetgen.volume_mesh_generator import VolumeMeshGenerator
from autoflowcfd.grid.structures import GridData, VolumeMeshData
from autoflowcfd.grid.validation.validator import GridValidator
from autoflowcfd.core import FRSolver, TransientSolver  # 从core模块导入TransientSolver

from autoflowcfd.config.solver_config import SteadyConfig, TransientConfig

from autoflowcfd.core.backend import get_available_backends

# 从拆分的模块导入辅助函数（控制单文件行数）
from autoflowcfd.api_grid_ops import (
    api_load_grid, api_get_grid_info, api_validate_grid,
    api_validate_surface_grid, api_generate_volume_mesh,
    api_get_volume_mesh_info, api_validate_volume_mesh,
)
from autoflowcfd.api_config import (
    api_create_steady_config, api_create_transient_config,
    api_load_config, api_resume_simulation,
)


def _turbulence_model_str(turb_config_value) -> str:
    """把 `config.solver_config.TurbulenceModel` 枚举值映射到求解器/CLI
    真正使用的字符串取值（none/sst/ddes/wmles/les，见
    fr_solver/turbulence.py::init_turbulence_models）。

    配套 #6（配置层接入）：`SteadyConfig`/`TransientConfig.turbulence`
    字段用的是这套独立枚举，命名（sst_kw vs sst）和取值范围
    （多了从未实现的 sa/des）都和求解器实际接受的字符串不完全一致，
    真正把 config 对象喂给 FRSolver 构造之前必须先做这层转换，而不是
    直接 `.value` 传下去（那样 "sst_kw" 会被当成未知湍流模型字符串，
    "sa"/"des" 会静默传给一个从不认识这两个值、只会当成"未识别输入"
    处理的构造函数）。
    """
    from autoflowcfd.config.solver_config import TurbulenceModel
    mapping = {
        TurbulenceModel.NONE: "none",
        TurbulenceModel.SST_KW: "sst",
        TurbulenceModel.DDES: "ddes",
        TurbulenceModel.IDDES: "iddes",
        TurbulenceModel.WMLES: "wmles",
        TurbulenceModel.LES: "les",
    }
    if turb_config_value not in mapping:
        raise ValueError(
            f"Turbulence model '{turb_config_value.value}' is representable in "
            f"SteadyConfig/TransientConfig but is not actually implemented by "
            f"FRSolver/GPUFRSolver — only none/sst_kw/ddes/iddes/wmles/les are real "
            f"solver options."
        )
    return mapping[turb_config_value]


class AutoFlowCFDAPI:
    """AutoFlowCFD V2.0 主 API 类（纯 FR 架构）。
    
    提供 AutoFlowCFD V2.0 的高层接口，支持网格处理、FR 求解和后处理。
    """
    
    def __init__(self, verbose: bool = False):
        self.verbose = verbose
        from autoflowcfd.config.loader import ConfigLoader
        self._config_loader = ConfigLoader()  # 初始化config_loader
        self.grid_data: Optional[GridData] = None
        self.volume_mesh: Optional[VolumeMeshData] = None
        self.solver = None
        self.convergence_history = []  # 收敛历史

    # ========================================================================
    # Version and Environment
    # ========================================================================
    
    def get_version(self) -> str:
        """获取软件版本信息。
        
        Returns:
            版本号字符串
        """
        from autoflowcfd import __version__
        return __version__
    
    def check_environment(self) -> Dict[str, Any]:
        """检查运行环境和可用资源。
        
        Returns:
            环境信息字典
        """
        import platform
        from autoflowcfd import __version__
        
        backends = get_available_backends()
        return {
            'platform': platform.platform(),
            'backends': backends,
            'gpu_available': backends.get('gpu', False),
            'cpu_count': os.cpu_count(),
            'python_version': os.sys.version,
            'autoflowcfd_version': __version__,
        }

    # ========================================================================
    # Grid Operations
    # ========================================================================
    
    def load_grid(self, grid_file, encoding="UTF-8", validate=True):
        """Load and parse grid file."""
        return api_load_grid(self, grid_file, encoding, validate)

    def get_grid_info(self, grid_data):
        """Get grid information and statistics."""
        return api_get_grid_info(self, grid_data)

    def validate_grid(self, grid_data):
        """验证网格质量。"""
        return api_validate_grid(self, grid_data)

    def _validate_surface_grid(self, grid_data):
        """验证表面网格质量的内部方法。"""
        return api_validate_surface_grid(self, grid_data)

    # ========================================================================
    # Volume Mesh Operations
    # ========================================================================
    
    def generate_volume_mesh(self, grid_data, method="tetrahedral", **kwargs):
        """Generate volume mesh from grid data."""
        return api_generate_volume_mesh(self, grid_data, method, **kwargs)

    def get_volume_mesh_info(self, volume_mesh):
        """Get volume mesh information and statistics."""
        return api_get_volume_mesh_info(self, volume_mesh)

    def validate_volume_mesh(self, volume_mesh):
        """Validate volume mesh quality."""
        return api_validate_volume_mesh(self, volume_mesh)

    # ========================================================================
    # Solver Operations (V2.0 FR Only)
    # ========================================================================
    
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
        tet_basis_mode: str = "collapsed",
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
            tet_basis_mode: 四面体体积基函数选择——"collapsed"（默认，
                行为与此前完全一致）或 "native"（路径C，见
                `fr/native_simplex_basis.py`/`ProjectFiles/V2.0/
                8_算法重构-微分算子对坍缩坐标退化参考轴的病态条件数-
                Part6~8.md`，修复坍缩坐标 Duffy 变换在退化参考轴附近
                导致的 P1/P2 残差异常；已在合成小网格上做过体积项/
                修正项/模态滤波/过积分去混叠的端到端决定性验证，尚未
                在真实生产规模网格上验证过）。直接决定 `HighOrderMesh`
                的构造参数，不经过 `config`/`SteadyConfig`（那套 YAML
                配置层尚未加这个字段，属于独立的后续工作）。
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
        （cli/solve_steady_command.py）的真实构造流程：先用
        HighOrderMesh.load_from_volume_mesh 把体网格升格成高阶网格，
        再构造 FRSolver，需要湍流模型时补上壁面距离场
        （V2.0 专家组评审逐行核实：此前的实现从未被真正跑通过）。
        """
        logger.info("Starting steady-state FR simulation")

        from autoflowcfd.grid.high_order.high_order_mesh import HighOrderMesh
        from autoflowcfd.cli.solve_wall_distance import compute_wall_distance_for_solver

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
        if config is not None:
            for field in ("mu_molecular", "turbulence_intensity", "viscosity_ratio"):
                kwargs.setdefault(field, getattr(config, field))

        mesh = HighOrderMesh(order=order, tet_basis_mode=tet_basis_mode)
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

        result = solver.solve(max_iter=max_iter, dt=dt, tol=tol)
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
        tet_basis_mode: str = "collapsed",
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
            tet_basis_mode: 四面体体积基函数选择，见 run_steady 同名
                参数文档（"collapsed"/"native"）。
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
        from autoflowcfd.cli.solve_wall_distance import compute_wall_distance_for_solver
        from autoflowcfd.core.time_integration.base import TimeIntegrationScheme as CoreTimeScheme

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
        if config is not None:
            for field in ("mu_molecular", "turbulence_intensity", "viscosity_ratio"):
                kwargs.setdefault(field, getattr(config, field))

        time_scheme_map = {
            'rk3': CoreTimeScheme.SSP_RK3,
            'ssp_rk3': CoreTimeScheme.SSP_RK3,
            'rk2': CoreTimeScheme.SSP_RK2,
            'ssp_rk2': CoreTimeScheme.SSP_RK2,
            'imex': CoreTimeScheme.IMEX_EULER,
            'dual-time': CoreTimeScheme.DUAL_TIME,
            'dual_time': CoreTimeScheme.DUAL_TIME,
            'forward_euler': CoreTimeScheme.FORWARD_EULER,
        }
        if time_method not in time_scheme_map:
            raise ValueError(
                f"Unknown time_method '{time_method}', expected one of "
                f"{sorted(time_scheme_map)}"
            )
        core_time_scheme = time_scheme_map[time_method]

        max_iter = int(physical_time / dt) if physical_time is not None else 1000

        mesh = HighOrderMesh(order=order, tet_basis_mode=tet_basis_mode)
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

        result = solver.solve(max_iter=max_iter, dt=dt, tol=tol)
        self.solver = solver

        logger.info(
            f"Transient simulation complete: {result.iterations} iterations, "
            f"converged={result.converged}"
        )

        return result
    
    def resume_simulation(self, checkpoint_file: str, **kwargs) -> Any:
        """从检查点恢复仿真。"""
        return api_resume_simulation(self, checkpoint_file, **kwargs)
    
    def create_steady_config(self, **kwargs) -> SteadyConfig:
        """创建稳态配置。"""
        return api_create_steady_config(self, **kwargs)

    def create_transient_config(self, **kwargs) -> TransientConfig:
        """创建瞬态配置。"""
        return api_create_transient_config(self, **kwargs)

    def load_solver_config(self, config_file: str) -> Union[SteadyConfig, TransientConfig]:
        """从 YAML 文件加载一个真正校验过的 `SteadyConfig`/`TransientConfig`
        对象（`mode: steady`/`mode: transient` 决定返回哪一个），可直接
        传给 `run_steady(config=...)`/`run_transient(config=...)`。

        真实修复（V2.0 专家组盲审发现，2026-08-28）：`self._config_loader`
        此前从构造函数（`__init__`）里初始化后就再也没被读取过——一个
        纯粹的死属性。这里是它第一个、也是唯一有意义的真实用途，与
        `load_config()`（返回未经校验的裸字典，供需要原始 YAML 内容的
        场景用）是两个不同粒度的入口，不是重复实现。

        Args:
            config_file: YAML 配置文件路径（与 CLI `--config` 选项、
                `config/loader.py::ConfigLoader` 用的是同一套 schema）

        Returns:
            SteadyConfig 或 TransientConfig
        """
        return self._config_loader.load(config_file)

    def load_config(self, config_file: str) -> Dict[str, Any]:
        """加载配置文件。"""
        return api_load_config(self, config_file)

    # ========================================================================
    # Post-processing
    # ========================================================================
    
    def calculate_coefficients(
        self,
        result: Any,
        reference_area: float = 1.0,
        reference_length: float = 1.0,
        density: float = 1.225,
        velocity: float = 33.33
    ) -> Dict[str, float]:
        """计算气动力系数。
        
        Args:
            result: 求解器结果
            reference_area: 参考面积
            reference_length: 参考长度
            density: 流体密度
            velocity: 参考速度
            
        Returns:
            气动力系数字典（使用大写键名Cd, Cl等）
        """
        # FR 原生积分路径（唯一真实积分实现，见 api_postprocess.py 同名委托函数文档）
        if self.solver is not None and hasattr(self.solver, 'mesh'):
            from autoflowcfd.postprocess.fr_coefficients import (
                compute_aerodynamic_coefficients_fr,
            )
            coeffs = compute_aerodynamic_coefficients_fr(
                self.solver,
                reference_area=reference_area,
                reference_length=reference_length,
            )
            return coeffs.to_dict()

        # 无求解器：返回诚实零值并警告，不回退到伪积分实现（旧版 V1
        # CoefficientCalculator 依赖不存在的 get_face_data()，已移除）
        logger.warning(
            "calculate_coefficients: 没有可用的 FR 求解器（需先调用 run_steady/"
            "run_transient 或 resume_simulation），返回零系数。"
        )
        return {
            'Cd': 0.0, 'Cl': 0.0, 'Cm': 0.0,
            'Cs': 0.0, 'Cy': 0.0, 'Cr': 0.0,
        }
    
    def export_vtk(self, result: Any = None, filename: str = None, high_order: bool = False) -> None:
        """导出 VTK 可视化文件。

        使用 VTKExporter 将流场数据导出为 VTK 格式，支持 legacy .vtk
        和 XML .vtu 两种格式（根据文件扩展名自动选择）。

        Args:
            result: 未使用，仅为向后兼容签名保留——真正的解场从
                self.solver.state.U 读取（见下方说明），不是从
                SolverResult 对象（它只有 converged/iterations/
                final_residual 三个字段，从不携带解场，见
                core/fr_solver/state.py）。
            filename: 输出文件名（.vtk 或 .vtu）
            high_order: True 时改走 postprocess/vtk_export_highorder.py
                （#5，2026-08-28 新增）：按 FR 解真实的分段多项式（而不是
                `U.mean(axis=1)` 拍扁的单元中心平均值）导出成 VTK
                VTK_LAGRANGE_TETRAHEDRON/WEDGE 高阶单元，能在 ParaView 里
                看到单元内部真实的多项式分布。只支持 order<=2（见该模块
                文档），且只输出 .vtu（VTK legacy 格式不支持任意阶
                Lagrange 单元）；filename 若没有 .vtu 后缀会被自动改写。

        此前这里用 `result.solution`（SolverResult 根本没有这个字段，
        `hasattr` 检查恒为 False，必然走进"抛异常"分支）和
        `self.grid_data`（run_steady/run_transient 从不写入的表面网格，
        即便写了，单元数也和体网格解场对不上）构造 VTKExporter——两个
        参数都是错的，从未被真正跑通过（V2.0 专家组评审逐行核实）。
        改为镜像 CLI `post export-vtk`（cli/post_export_commands.py）
        真正验证过的用法：VTKExporter 的 `grid_data` 参数只是鸭子类型
        地读取 `.metadata.node_count`/`.cell_count`，`self.volume_mesh`
        （generate_volume_mesh 的输出，run_steady/run_transient 求解的
        就是它）满足这个接口；解场用 `self.solver.state.U.mean(axis=1)`
        拍扁成单元中心平均值（与 CheckpointManager.save 写 checkpoint
        时的约定一致）包装成 SolutionVector。
        """
        from autoflowcfd.postprocess.vtk_export import VTKExporter
        from autoflowcfd.core.backend.base import SolutionVector

        if self.solver is None or self.volume_mesh is None:
            raise ValueError(
                "export_vtk 需要先成功运行 run_steady/run_transient "
                "（需要 self.solver 和 self.volume_mesh 均已设置）。"
            )
        if filename is None:
            raise ValueError("export_vtk 需要提供 filename。")

        if high_order:
            from pathlib import Path

            from autoflowcfd.postprocess.vtk_export_highorder import export_highorder_vtk

            out_path = Path(filename)
            if out_path.suffix != '.vtu':
                out_path = out_path.with_suffix('.vtu')
            export_highorder_vtk(self.solver.mesh, self.solver.state.U, out_path)
            logger.info(f"High-order VTK exported: {out_path}")
            return

        U_cell_avg = self.solver.state.U.mean(axis=1)  # (n_cells, n_vars)
        solution = SolutionVector(
            data=U_cell_avg, n_cells=U_cell_avg.shape[0], n_variables=U_cell_avg.shape[1],
        )

        # 湍流涡粘度（用于精确的 nut 导出），有则给，没有就让 VTKExporter
        # 自己退化成简化估计（它自身文档已说明这个 fallback）。
        mu_t = None
        get_mu_t = getattr(self.solver, '_get_turbulent_viscosity_field', None)
        if callable(get_mu_t):
            mu_t_field = get_mu_t()
            if mu_t_field is not None:
                mu_t = mu_t_field.mean(axis=1)

        exporter = VTKExporter(self.volume_mesh, solution, mu_t=mu_t)

        # 根据扩展名选择格式
        fmt = 'xml' if filename.endswith('.vtu') else 'legacy'
        exporter.export(filename, format=fmt)
        logger.info(f"VTK exported: {filename}")
    
    def get_convergence_history(self, result: Any = None) -> Dict[str, list]:
        """获取收敛历史。

        真实 bug 修复（V2.0 专家组盲审发现，2026-08-27）：此前这里恒为
        硬编码占位符 `{"iterations": [], "residuals": []}`，不管
        run_steady/run_transient 是否已经成功跑完、收敛得多好，调用方
        拿到的永远是两个空列表，且没有任何警告提示这是未实现的占位符。
        现在读取 `self.solver.residual_history`（CPU FRSolver 与
        GPUFRSolver/MultiGPUDistributedSolver 都在各自的 solve 循环里
        逐迭代 append，同一个约定，见 fr_solver/solver.py::solve() 与
        core/utils/order_continuation.py::run_order_continuation）。

        Args:
            result: 未使用，保留以兼容旧调用签名

        Returns:
            包含 iterations（1-based 迭代序号）和 residuals 的字典；
            尚未运行过 run_steady/run_transient（self.solver 为 None）
            或求解器本身未记录历史时返回两个空列表。
        """
        residuals = getattr(self.solver, "residual_history", None) if self.solver is not None else None
        if not residuals:
            return {"iterations": [], "residuals": []}
        return {
            "iterations": list(range(1, len(residuals) + 1)),
            "residuals": list(residuals),
        }

def create_api(verbose: bool = False) -> AutoFlowCFDAPI:
    """Factory function to create API instance."""
    return AutoFlowCFDAPI(verbose=verbose)
