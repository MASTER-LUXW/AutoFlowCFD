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
from autoflowcfd.api_postprocess import api_calculate_coefficients, api_export_vtk
from autoflowcfd.api_config import (
    api_create_steady_config, api_create_transient_config,
    api_load_config, api_resume_simulation,
)


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
        backend: str = "cpu",
        order: int = 2,
        turbulence_model: str = "sst",
        max_iter: int = 1000,
        dt: float = 1e-4,
        tol: float = 1e-6,
        rho_inf: float = 1.225,
        vel_inf: float = 33.33,
        p_inf: float = 101325.0,
        threads: int = -1,
        output_dir: str = "./results",
        **kwargs
    ) -> Any:
        """Run steady-state FR simulation.

        Args:
            volume_mesh: VolumeMeshData（generate_volume_mesh 的输出，不是
                load_grid 返回的表面 GridData——V2 求解器需要体网格）
            backend: Compute backend (cpu/gpu)
            order: FR discretization order (1/2/3)
            turbulence_model: 湍流模型 ("none"/"sst"/"ddes"/"wmles"/"les")
            max_iter: Maximum iterations
            dt, tol: 时间步长与收敛容差，直接透传给 FRSolver.solve()
            rho_inf, vel_inf, p_inf: 自由来流条件
            threads: CPU 后端 numba 并行线程数
            output_dir: Output directory
            **kwargs: 其余参数透传给 FRSolver 构造函数
                （例如 mu_molecular/dual_time_inner_iter/bc_overrides）

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
        backend: str = "cpu",
        order: int = 2,
        time_method: str = "rk3",
        turbulence_model: str = "sst",
        mode: str = None,
        physical_time: float = None,
        dt: float = 1e-4,
        tol: float = 0.0,
        rho_inf: float = 1.225,
        vel_inf: float = 33.33,
        p_inf: float = 101325.0,
        threads: int = -1,
        output_dir: str = "./transient_results",
        **kwargs
    ) -> Any:
        """Run transient FR simulation (DES/LES).

        Args:
            volume_mesh: VolumeMeshData（generate_volume_mesh 的输出）
            backend: Compute backend (cpu/gpu)
            order: FR discretization order
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
        velocity: float = 30.0
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
        # 优先使用 FR 原生积分路径
        if self.solver is not None and hasattr(self.solver, 'mesh'):
            try:
                from autoflowcfd.postprocess.fr_coefficients import (
                    compute_aerodynamic_coefficients_fr,
                )
                coeffs = compute_aerodynamic_coefficients_fr(
                    self.solver,
                    reference_area=reference_area,
                    reference_length=reference_length,
                )
                return coeffs.to_dict()
            except Exception as e:
                logger.warning(f"FR 原生系数计算失败: {e}，回退到 V1 路径")

        # 回退路径：使用 V1 CoefficientCalculator
        from autoflowcfd.postprocess.coefficients import CoefficientCalculator

        if not hasattr(self, 'grid_data') or self.grid_data is None:
            logger.warning("grid_data 不可用，返回零系数")
            return {
                'Cd': 0.0, 'Cl': 0.0, 'Cm': 0.0,
                'Cs': 0.0, 'Cy': 0.0, 'Cr': 0.0,
            }

        solution = result.solution if hasattr(result, 'solution') else None
        calc = CoefficientCalculator(
            self.grid_data,
            solution,
            reference_area=reference_area,
            reference_length=reference_length,
            density=density,
            velocity=velocity
        )

        coeffs = calc.calculate()
        return coeffs.to_dict()
    
    def export_vtk(self, result: Any = None, filename: str = None) -> None:
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
        exporter.export(filename, file_format=fmt)
        logger.info(f"VTK exported: {filename}")
    
    def get_convergence_history(self, result: Any = None) -> Dict[str, list]:
        """获取收敛历史。
        
        Args:
            result: 可选的结果对象
            
        Returns:
            包含iterations和residuals的字典
        """
        # V2.0 FR求解器的收敛历史占位符
        return {
            "iterations": [],
            "residuals": []
        }

def create_api(verbose: bool = False) -> AutoFlowCFDAPI:
    """Factory function to create API instance."""
    return AutoFlowCFDAPI(verbose=verbose)
