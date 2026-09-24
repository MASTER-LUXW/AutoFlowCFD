"""AutoFlowCFD V2.0 - AutoFlowCFDAPI 主类：环境检查、网格与配置入口

从 `src/autoflowcfd/api.py` 拆出（2026-09-24）。方法按职责分到同目录的 mixin 里，
这里只留构造与对外接口。
"""

import os
from typing import Dict, Optional, Union, Any
from autoflowcfd.grid.structures import GridData, VolumeMeshData
from autoflowcfd.config.solver_config import SteadyConfig, TransientConfig
from autoflowcfd.core.backend import get_available_backends
from autoflowcfd.api_grid_ops import (
    api_load_grid, api_get_grid_info, api_validate_grid,
    api_validate_surface_grid, api_generate_volume_mesh,
    api_get_volume_mesh_info, api_validate_volume_mesh,
)
from autoflowcfd.api_config import (
    api_create_steady_config,
    api_create_transient_config,
    api_load_config,
)
from .solve import _APISolveMixin
from .post import _APIPostMixin


class AutoFlowCFDAPI(_APISolveMixin, _APIPostMixin):
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

    def generate_volume_mesh(self, grid_data, method="tetrahedral", **kwargs):
        """Generate volume mesh from grid data."""
        return api_generate_volume_mesh(self, grid_data, method, **kwargs)

    def get_volume_mesh_info(self, volume_mesh):
        """Get volume mesh information and statistics."""
        return api_get_volume_mesh_info(self, volume_mesh)

    def validate_volume_mesh(self, volume_mesh):
        """Validate volume mesh quality."""
        return api_validate_volume_mesh(self, volume_mesh)

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


def create_api(verbose: bool = False) -> AutoFlowCFDAPI:
    """Factory function to create API instance."""
    return AutoFlowCFDAPI(verbose=verbose)
