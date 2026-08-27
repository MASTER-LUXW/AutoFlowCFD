"""AutoFlowCFDAPI 的后处理辅助方法。

从 api.py 拆出，控制单文件行数。包含气动力系数计算和 VTK 导出。
"""

from typing import Any, Dict
from loguru import logger


def api_calculate_coefficients(self, result: Any = None,
                                reference_area: float = 1.0,
                                reference_length: float = 1.0,
                                density: float = 1.225,
                                velocity: float = 33.33) -> Dict[str, float]:
    """计算气动力系数（委托函数）。

    唯一生产路径是 FR 原生积分（fr_coefficients.py：面通量点压力积分，含力矩），
    需要先跑过 run_steady/run_transient（self.solver 已就绪）。没有可用的求解器时返回诚实的零系数并警告——
    旧版的 V1 CoefficientCalculator 回退已移除（依赖不存在的
    get_face_data()，系数恒为 0，属失效实现）。
    """
    # FR 原生积分路径（唯一真实积分实现）
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

    # 无求解器：不再回退到伪积分实现，返回诚实零值并警告（第三轮评审整改）
    logger.warning(
        "calculate_coefficients: 没有可用的 FR 求解器（需先调用 run_steady/"
        "run_transient 或 resume_simulation），返回零系数。CLI 场景请用 "
        "`autoflowcfd post coefficients <checkpoint>`。"
    )
    return {'Cd': 0.0, 'Cl': 0.0, 'Cm': 0.0, 'Cs': 0.0, 'Cy': 0.0, 'Cr': 0.0}


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
