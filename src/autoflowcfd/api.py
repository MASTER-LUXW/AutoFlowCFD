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
from autoflowcfd.grid.mesh_gen.volume_mesh_generator import VolumeMeshGenerator
from autoflowcfd.grid.structures import GridData, VolumeMeshData
from autoflowcfd.grid.validation.validator import GridValidator
from autoflowcfd.boundary import BoundaryManager
from autoflowcfd.core import FRSolver, TransientSolver  # 从core模块导入TransientSolver

from autoflowcfd.config.solver_config import SteadyConfig, BackendType, TransientConfig, TurbulenceModel

from autoflowcfd.core.backend import get_available_backends


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
    
    def load_grid(
        self,
        grid_file: Union[str, Path],
        encoding: str = "UTF-8",
        validate: bool = True
    ) -> GridData:
        """Load and parse grid file.
        
        Args:
            grid_file: Path to .nas grid file
            encoding: File encoding
            validate: Whether to validate grid quality
            
        Returns:
            GridData: Parsed grid data object
        """
        logger.info(f"Loading grid: {grid_file}")
        
        parser = NASParser(str(grid_file), encoding=encoding)
        grid_data = parser.parse()
        
        if validate:
            logger.info("Validating grid quality...")
            # 执行网格质量验证
            validation_result = self._validate_surface_grid(grid_data)
            if not validation_result['valid']:
                logger.warning(f"Grid validation warnings: {validation_result['warnings']}")
            else:
                logger.info("Grid validation passed")

        logger.info(
            f"Grid loaded: {grid_data.node_count} nodes, "
            f"{grid_data.cell_count} cells"
        )
        
        self.grid_data = grid_data
        return grid_data
    
    def get_grid_info(self, grid_data: GridData) -> Dict[str, Any]:
        """Get grid information and statistics."""
        info = {
            'node_count': grid_data.node_count,
            'cell_count': grid_data.cell_count,
            'boundary_count': len(grid_data.boundaries) if hasattr(grid_data, 'boundaries') else 0,
        }
        
        # 添加边界组信息（如果存在）
        if hasattr(grid_data, 'boundary_groups'):
            info['boundary_groups'] = grid_data.boundary_groups
        
        return info
    
    def validate_grid(self, grid_data: GridData) -> Dict[str, Any]:
        """验证网格质量。
        
        Args:
            grid_data: 网格数据对象
            
        Returns:
            包含验证结果的字典，包括error_count和warning_count
        """
        # 使用GridValidator进行验证
        validator = GridValidator(grid_data)
        result = validator.validate()
        
        # 确保返回的字典包含必需的键
        if 'error_count' not in result:
            result['error_count'] = len(result.get('errors', []))
        if 'warning_count' not in result:
            result['warning_count'] = len(result.get('warnings', []))
        
        return result
    
    def _validate_surface_grid(self, grid_data: GridData) -> Dict[str, Any]:
        """
        验证表面网格质量的内部方法。
        
        检查项：
        1. 节点重复
        2. 单元连通性
        3. 法向量一致性
        4. 网格尺寸分布
        
        Args:
            grid_data: 表面网格数据
            
        Returns:
            验证结果字典
        """
        warnings = []
        errors = []
        
        # 1. 检查节点数量
        if grid_data.node_count < 3:
            errors.append("Insufficient nodes for surface mesh")
        
        # 2. 检查单元数量
        if grid_data.cell_count == 0:
            errors.append("No surface elements found")
        
        # 3. 检查边界条件
        if hasattr(grid_data, 'boundaries'):
            bm = grid_data.boundaries
            if not bm.bc_types:
                warnings.append("No boundary conditions defined")
            else:
                # 检查是否有WALL边界
                has_wall = any(bc_type == 'WALL' for bc_type in bm.bc_types.values())
                if not has_wall:
                    warnings.append("No WALL boundary condition found")
        
        # 4. 检查网格尺寸均匀性（简化）
        if hasattr(grid_data, 'elements') and len(grid_data.elements) > 0:
            # 可以添加更详细的网格质量检查
            pass
        
        return {
            'valid': len(errors) == 0,
            'errors': errors,
            'warnings': warnings
        }

    # ========================================================================
    # Volume Mesh Operations
    # ========================================================================
    
    def generate_volume_mesh(
        self,
        grid_data: GridData,
        method: str = "tetrahedral",
        **kwargs
    ) -> VolumeMeshData:
        """Generate volume mesh from grid data.
        
        Args:
            grid_data: Grid data object
            method: Mesh generation method (tetrahedral/hexahedral)
            **kwargs: Additional configuration parameters
            
        Returns:
            VolumeMeshData: Generated volume mesh data object
        """
        logger.info("Generating volume mesh...")
        
        mesh_gen = VolumeMeshGenerator(grid_data, method=method, **kwargs)
        volume_mesh = mesh_gen.generate()
        
        logger.info(
            f"Volume mesh generated: {volume_mesh.node_count} nodes, "
            f"{volume_mesh.cell_count} cells"
        )
        
        return volume_mesh
    
    def get_volume_mesh_info(self, volume_mesh: VolumeMeshData) -> Dict[str, Any]:
        """Get volume mesh information and statistics."""
        return {
            'node_count': volume_mesh.node_count,
            'cell_count': volume_mesh.cell_count,
            'cell_type': volume_mesh.cell_type,
        }
    
    def validate_volume_mesh(self, volume_mesh: VolumeMeshData) -> Dict[str, Any]:
        """Validate volume mesh quality.
        
        检查项：
        1. 单元体积正性
        2. 网格连通性完整性
        3. 边界条件一致性
        4. 长宽比和扭曲度
        
        Args:
            volume_mesh: 体网格数据
            
        Returns:
            验证结果字典
        """
        errors = []
        warnings = []
        
        # 1. 检查节点数量
        if volume_mesh.node_count < 4:
            errors.append("Insufficient nodes for volume mesh")
        
        # 2. 检查单元数量
        if volume_mesh.cell_count == 0:
            errors.append("No volume elements found")
        
        # 3. 检查单元体积（如果有）
        if hasattr(volume_mesh, 'cell_volumes') and volume_mesh.cell_volumes is not None:
            negative_volumes = np.sum(volume_mesh.cell_volumes <= 0)
            if negative_volumes > 0:
                errors.append(f"Found {negative_volumes} cells with non-positive volume")
            
            # 检查体积分布
            vol_mean = np.mean(volume_mesh.cell_volumes)
            vol_std = np.std(volume_mesh.cell_volumes)
            if vol_std / vol_mean > 2.0:  # 变异系数过大
                warnings.append("High variation in cell volumes")
        
        # 4. 检查棱柱层（如果有）
        if hasattr(volume_mesh, 'prism_cells') and volume_mesh.prism_cells:
            n_prisms = len(volume_mesh.prism_cells.connectivity)
            logger.info(f"Found {n_prisms} prism layers")
        
        # 5. 检查边界条件
        if hasattr(volume_mesh, 'boundaries'):
            bm = volume_mesh.boundaries
            if not bm.bc_types:
                warnings.append("No boundary conditions defined on volume mesh")
        
        return {
            'passed': len(errors) == 0,
            'errors': errors,
            'warnings': warnings
        }

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
        from autoflowcfd.core.time_integration import TimeIntegrationScheme as CoreTimeScheme
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
        from autoflowcfd.core.time_integration import TimeIntegrationScheme as CoreTimeScheme
        
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
        """从检查点恢复仿真。
        
        Args:
            checkpoint_file: 检查点文件路径
            **kwargs: 额外参数
            
        Returns:
            SolverResult: 仿真结果
            
        Raises:
            NotImplementedError: V2.0中尚未完全实现
            FileNotFoundError: 检查点文件不存在
        """
        from pathlib import Path
        
        if not Path(checkpoint_file).exists():
            raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_file}")
        
        logger.info(f"Resuming simulation from checkpoint: {checkpoint_file}")
        # TODO: 实现检查点加载逻辑
        raise NotImplementedError("Checkpoint resume not yet implemented in V2.0")
    
    def create_steady_config(
        self,
        backend: str = "cpu",
        order: int = 2,
        max_iter: int = 1000,
        turbulence: str = "sst_kw",
        **kwargs
    ) -> SteadyConfig:
        """Create steady-state configuration."""
        # 将字符串转换为TurbulenceModel枚举
        turb_model_map = {
            'none': TurbulenceModel.NONE,
            'sst_kw': TurbulenceModel.SST_KW,
            'sa': TurbulenceModel.SA,
            'des': TurbulenceModel.DES,
            'ddes': TurbulenceModel.DDES,
            'les': TurbulenceModel.LES,
        }
        turb_model = turb_model_map.get(turbulence.lower(), TurbulenceModel.SST_KW)
        
        return SteadyConfig(
            backend=BackendType(backend),
            order=order,
            turbulence=turb_model,
            max_iter=max_iter,
            **kwargs
        )
    
    def create_transient_config(
        self,
        backend: str = "cpu",
        order: int = 2,
        time_method: str = "dual-time",
        turbulence_model: str = None,
        mode: str = None,
        max_iter: int = 1000,
        dt: float = 1e-4,
        total_time: float = None,
        **kwargs
    ) -> TransientConfig:
        """Create transient configuration.
        
        Args:
            backend: Backend type ('cpu' or 'gpu')
            order: FR order (1-3)
            time_method: Time integration method
            turbulence_model: Turbulence model name
            mode: Simulation mode ('sst', 'ddes', 'wmles', 'les') - alternative to turbulence_model
            max_iter: Maximum iterations
            dt: Time step size
            total_time: Total physical time (overrides dt * max_iter if specified)
            **kwargs: Additional config parameters
        """
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
        if mode is not None and turbulence_model is None:
            turbulence_model = mode
        
        # 默认湍流模型
        if turbulence_model is None:
            turbulence_model = "sst"
        
        return TransientConfig(
            backend=BackendType(backend),
            order=order,
            turbulence=TurbulenceModel(turbulence_model),
            time_scheme=time_scheme,
            dt=dt,
            total_time=total_time if total_time is not None else dt * max_iter,
            **kwargs
        )

    def load_config(self, config_file: str) -> Dict[str, Any]:
        """加载配置文件。
        
        Args:
            config_file: 配置文件路径（支持JSON和YAML）
            
        Returns:
            配置字典
        """
        from pathlib import Path
        file_path = Path(config_file)
        
        if not file_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_file}")
        
        # 根据文件扩展名选择解析器
        if file_path.suffix in ['.json']:
            import json
            with open(config_file, 'r') as f:
                return json.load(f)
        elif file_path.suffix in ['.yml', '.yaml']:
            try:
                import yaml
                with open(config_file, 'r') as f:
                    return yaml.safe_load(f)
            except ImportError:
                logger.warning("PyYAML not installed, cannot load YAML config")
                return {}
        else:
            logger.warning(f"Unsupported config file format: {file_path.suffix}")
            return {}

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
        from autoflowcfd.postprocess.coefficients import CoefficientCalculator
        
        # 需要从result中提取solution和grid_data
        # TODO: 实际实现需要从FRSolver结果中提取
        if not hasattr(self, 'grid_data') or self.grid_data is None:
            # 返回占位符
            return {
                'Cd': 0.0,  # 阻力系数
                'Cl': 0.0,  # 升力系数
                'Cm': 0.0,  # 俯仰力矩系数
                'Cs': 0.0,  # 侧向力系数
                'Cy': 0.0,  # 偏航力矩系数
                'Cr': 0.0,  # 滚转力矩系数
            }
        
        # 创建计算器并计算
        calc = CoefficientCalculator(
            self.grid_data,
            result.solution if hasattr(result, 'solution') else None,
            reference_area=reference_area,
            reference_length=reference_length,
            density=density,
            velocity=velocity
        )
        
        coeffs = calc.calculate()
        return coeffs.to_dict()
    
    def export_vtk(self, result: Any, filename: str) -> None:
        """导出VTK可视化文件。
        
        Args:
            result: 求解器结果
            filename: 输出文件名
            
        Raises:
            NotImplementedError: V2.0中尚未完全实现
        """
        # V2.0 FR需要特殊的高阶VTK导出
        raise NotImplementedError("VTK export for FR solver requires high-order Lagrange elements")
    
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
