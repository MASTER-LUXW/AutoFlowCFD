"""AutoFlowCFDAPI 的后处理辅助方法。

从 api.py 拆出，控制单文件行数。包含气动力系数计算和 VTK 导出。
"""

from typing import Any, Dict
from loguru import logger


def api_calculate_coefficients(self, result: Any = None,
                                reference_area: float = 1.0,
                                reference_length: float = 1.0,
                                density: float = 1.225,
                                velocity: float = 30.0) -> Dict[str, float]:
    """计算气动力系数（委托函数）。

    优先使用 FR 原生积分路径，回退到 V1 CoefficientCalculator。
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
        return {'Cd': 0.0, 'Cl': 0.0, 'Cm': 0.0, 'Cs': 0.0, 'Cy': 0.0, 'Cr': 0.0}

    solution = result.solution if hasattr(result, 'solution') else None
    calc = CoefficientCalculator(
        self.grid_data, solution,
        reference_area=reference_area, reference_length=reference_length,
        density=density, velocity=velocity,
    )
    coeffs = calc.calculate()
    return coeffs.to_dict()


def api_export_vtk(self, result: Any, filename: str) -> None:
    """导出 VTK 可视化文件（委托函数）。"""
    from autoflowcfd.postprocess.vtk_export import VTKExporter

    grid_data = self.grid_data
    solution = result.solution if hasattr(result, 'solution') else None

    if grid_data is None or solution is None:
        raise ValueError(
            "export_vtk 需要 grid_data 和 solution。"
            "请先运行仿真并确保 grid_data 已加载。"
        )

    mu_t = None
    if hasattr(result, 'extra_fields') and 'mu_t' in result.extra_fields:
        mu_t = result.extra_fields['mu_t']

    exporter = VTKExporter(grid_data, solution, mu_t=mu_t)
    fmt = 'xml' if filename.endswith('.vtu') else 'legacy'
    exporter.export(filename, file_format=fmt)
    logger.info(f"VTK exported: {filename}")
