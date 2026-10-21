"""AutoFlowCFD V2.0 - AutoFlowCFDAPI 主类：构造、版本与环境信息。

方法按职责分在同目录的 mixin 里：`grid.py`（网格）、`config.py`（配置）、`solve.py`（求解与续算）、`post.py`
（后处理）。
"""

import os
from typing import Any, Dict, Optional

from autoflowcfd.grid.structures import GridData, VolumeMeshData

from .config import _APIConfigMixin
from .grid import _APIGridMixin
from .post import _APIPostMixin
from .solve import _APISolveMixin


class AutoFlowCFDAPI(_APIGridMixin, _APIConfigMixin, _APISolveMixin, _APIPostMixin):
    """AutoFlowCFD V2.0 的 Python 接口：网格处理、FR 求解与后处理。

    典型流程：
        >>> api = AutoFlowCFDAPI()
        >>> volume_mesh = api.generate_volume_mesh(api.load_grid("car_surface.nas"))
        >>> result = api.run_steady(volume_mesh, order=2, turbulence="sa", time_scheme="newton-krylov")
        >>> coeffs = api.calculate_coefficients(reference_area=2.2)
    """

    def __init__(self, verbose: bool = False):
        self.verbose = verbose
        self.grid_data: Optional[GridData] = None
        self.volume_mesh: Optional[VolumeMeshData] = None
        self.solver = None

    def get_version(self) -> str:
        """软件版本号。"""
        from autoflowcfd import __version__

        return __version__

    def check_environment(self) -> Dict[str, Any]:
        """运行环境与可用后端。"""
        import platform
        import sys

        from autoflowcfd import __version__
        from autoflowcfd.core.backend import get_available_backends

        backends = get_available_backends()
        return {
            'platform': platform.platform(),
            'backends': backends,
            'gpu_available': backends.get('gpu', False),
            'cpu_count': os.cpu_count(),
            'python_version': sys.version,
            'autoflowcfd_version': __version__,
        }
