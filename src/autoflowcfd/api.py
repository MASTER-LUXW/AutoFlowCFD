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
from autoflowcfd.boundary import BoundaryManager
from autoflowcfd.core import FRSolver, TransientSolver  # 从core模块导入TransientSolver

from autoflowcfd.config.solver_config import SteadyConfig, BackendType, TransientConfig, TurbulenceModel

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
        self.boundary_manager: Optional[BoundaryManager] = None
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
        grid_data: GridData,
        backend: str = "cpu",
        order: int = 2,
        max_iter: int = 1000,
        output_dir: str = "./results",
        **kwargs
    ) -> Any:
        """Run steady-state FR simulation.
        
        Args:
            grid_data: Grid data object
            backend: Compute backend (cpu/gpu)
            order: FR discretization order (1/2/3)
            max_iter: Maximum iterations
            output_dir: Output directory
            **kwargs: Additional configuration parameters
            
        Returns:
            SolverResult: Simulation result object
        """
        logger.info("Starting steady-state FR simulation")
        
        # Create configuration
        config = SteadyConfig(
            backend=BackendType(backend),
            order=order,
            max_iter=max_iter,
            output_dir=output_dir,
            **kwargs
        )
        
        # Create boundary manager if boundaries exist
        if hasattr(grid_data, 'boundaries'):
            bc_manager = BoundaryManager(grid_data.boundaries)
        else:
            bc_manager = None
        
        # Create and run solver
        from autoflowcfd.core.time_integration.base import TimeIntegrationScheme as CoreTimeScheme
        from autoflowcfd.config.solver_config import TimeIntegrationScheme
        
        # Steady默认使用RK3
        core_time_scheme = CoreTimeScheme.SSP_RK3
        
        solver = FRSolver(
            mesh=grid_data,
            order=config.order,
            turb_model_name=config.turbulence.value if hasattr(config.turbulence, 'value') else config.turbulence,
            time_scheme=core_time_scheme
        )
        result = solver.solve()

        logger.info(
            f"Simulation complete: {result.iterations} iterations, "
            f"converged={result.converged}"
        )
        
        return result
    
    def run_transient(
        self,
        grid_data: GridData,
        backend: str = "cpu",
        order: int = 2,
        time_method: str = "dual-time",
        turbulence_model: str = "sst",
        mode: str = None,
        physical_time: float = None,
        dt: float = 1e-4,
        output_dir: str = "./transient_results",
        **kwargs
    ) -> Any:
        """Run transient FR simulation.
        
        Args:
            grid_data: Grid data object
            backend: Compute backend (cpu/gpu)
            order: FR discretization order
            time_method: Time integration method (rk3/imex/dual-time/backward_euler)
            turbulence_model: Turbulence model (sst/ddes/wmles/les)
            mode: Simulation mode ('sst', 'ddes', 'wmles', 'les') - alternative to turbulence_model
            physical_time: Total physical time (seconds)
            dt: Time step size
            output_dir: Output directory
            **kwargs: Additional configuration parameters
            
        Returns:
            SolverResult: Simulation result object
        """
        logger.info("Starting transient FR simulation")
        
        # 创建瞬态配置
        from autoflowcfd.config.solver_config import TransientConfig, TimeIntegrationScheme
        
        # 映射time_method到time_scheme枚举（使用solver_config中的枚举）
        time_scheme_map = {
            'backward_euler': TimeIntegrationScheme.BACKWARD_EULER,
            'rk2': TimeIntegrationScheme.RK2,
            'rk3': TimeIntegrationScheme.RK3,
            'ab3': TimeIntegrationScheme.AB3,
            # 兼容V2.0的新方法名
            'dual-time': TimeIntegrationScheme.BACKWARD_EULER,
            'imex': TimeIntegrationScheme.RK3,
        }
        time_scheme = time_scheme_map.get(time_method, TimeIntegrationScheme.RK3)
        
        # 支持mode参数作为turbulence_model的别名
        if mode is not None and turbulence_model == "sst":
            turbulence_model = mode
        
        # 映射湍流模型
        turb_model_map = {
            'none': TurbulenceModel.NONE,
            'sst': TurbulenceModel.SST_KW,
            'ddes': TurbulenceModel.DDES,
            'wmles': TurbulenceModel.LES,
            'les': TurbulenceModel.LES,
            'des': TurbulenceModel.DDES  # des模式映射到DDES
        }
        turb_model = turb_model_map.get(turbulence_model, TurbulenceModel.SST_KW)
        
        # 计算总时间
        total_time = physical_time if physical_time is not None else dt * 1000
        
        config = TransientConfig(
            backend=BackendType(backend),
            order=order,
            turbulence=turb_model,
            time_scheme=time_scheme,
            dt=dt,
            total_time=total_time,
            output_dir=output_dir,
            **kwargs
        )
        
        # 创建并运行求解器
        from autoflowcfd.core.time_integration.base import TimeIntegrationScheme as CoreTimeScheme
        
        # 将solver_config的枚举转换为core的枚举
        scheme_map = {
            TimeIntegrationScheme.BACKWARD_EULER: CoreTimeScheme.FORWARD_EULER,
            TimeIntegrationScheme.RK2: CoreTimeScheme.SSP_RK2,
            TimeIntegrationScheme.RK3: CoreTimeScheme.SSP_RK3,
            TimeIntegrationScheme.AB3: CoreTimeScheme.ADAMS_BASHFORTH_3,
        }
        core_time_scheme = scheme_map.get(config.time_scheme, CoreTimeScheme.SSP_RK3)
        
        # 使用TransientSolver（它是FRSolver的别名）以便测试可以mock它
        solver = TransientSolver(
            mesh=grid_data,
            order=config.order,
            turb_model_name=config.turbulence.value if hasattr(config.turbulence, 'value') else config.turbulence,
            time_scheme=core_time_scheme
        )
        result = solver.solve()

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
    
    def export_vtk(self, result: Any, filename: str) -> None:
        """导出 VTK 可视化文件。

        使用 VTKExporter 将流场数据导出为 VTK 格式，支持 legacy .vtk
        和 XML .vtu 两种格式（根据文件扩展名自动选择）。

        Args:
            result: 求解器结果
            filename: 输出文件名（.vtk 或 .vtu）
        """
        from autoflowcfd.postprocess.vtk_export import VTKExporter

        grid_data = self.grid_data
        solution = result.solution if hasattr(result, 'solution') else None

        if grid_data is None or solution is None:
            raise ValueError(
                "export_vtk 需要 grid_data 和 solution。"
                "请先运行仿真并确保 grid_data 已加载。"
            )

        # 尝试提取湍流粘度（用于精确的 nut 导出）
        mu_t = None
        if hasattr(result, 'extra_fields') and 'mu_t' in result.extra_fields:
            mu_t = result.extra_fields['mu_t']

        exporter = VTKExporter(grid_data, solution, mu_t=mu_t)

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
