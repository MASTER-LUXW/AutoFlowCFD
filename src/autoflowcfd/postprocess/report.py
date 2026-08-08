"""收敛性分析与报告生成模块。

本模块提供分析收敛历史、把收敛曲线导出为 CSV、以及生成 JSON 格式仿真
报告的工具。

Key Components:
    - ConvergenceAnalyzer: 残差与系数历史分析
    - SimulationReport: 完整仿真报告生成

Example:
    >>> from autoflowcfd.postprocess import ConvergenceAnalyzer
    >>> analyzer = ConvergenceAnalyzer()
    >>> analyzer.add_iteration(residuals={'continuity': 1e-3, 'momentum': 1e-4})
    >>> analyzer.export_csv("convergence.csv")
"""

import json
import csv
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime
from loguru import logger
from dataclasses import dataclass, field, asdict

from .. import __version__
from .coefficients import AerodynamicCoefficients


@dataclass
class IterationData:
    """单次迭代的数据。

    Attributes:
        iteration: 迭代数
        residuals: 各方程的残差值
        cfl: CFL 数
        coefficients: 气动系数（可选）
        timestamp: 迭代时间戳
    """
    iteration: int
    residuals: Dict[str, float] = field(default_factory=dict)
    cfl: float = 0.0
    coefficients: Optional[AerodynamicCoefficients] = None
    timestamp: Optional[datetime] = None


@dataclass
class SimulationSummary:
    """仿真汇总统计。

    Attributes:
        total_iterations: 总迭代数
        final_residuals: 最终残差值
        initial_coefficients: 初始气动系数
        final_coefficients: 最终气动系数
        computation_time: 总计算时间（秒）
        converged: 仿真是否收敛
        convergence_criteria: 使用的收敛判据
    """
    total_iterations: int = 0
    final_residuals: Dict[str, float] = field(default_factory=dict)
    initial_coefficients: Optional[AerodynamicCoefficients] = None
    final_coefficients: Optional[AerodynamicCoefficients] = None
    computation_time: float = 0.0
    converged: bool = False
    convergence_criteria: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> Dict:
        """转换成字典。"""
        d = asdict(self)
        if self.initial_coefficients:
            d['initial_coefficients'] = self.initial_coefficients.to_dict()
        else:
            d['initial_coefficients'] = None
        if self.final_coefficients:
            d['final_coefficients'] = self.final_coefficients.to_dict()
        else:
            d['final_coefficients'] = None
        return d


class ConvergenceAnalyzer:
    """收敛历史分析器。

    跟踪并分析仿真过程中的残差历史、CFL 数演化和气动系数变化。

    Attributes:
        history: 迭代数据列表
        start_time: 仿真开始时间

    Example:
        >>> analyzer = ConvergenceAnalyzer()
        >>> analyzer.add_iteration(
        ...     iteration=1,
        ...     residuals={'continuity': 1e-2, 'momentum': 1e-3},
        ...     cfl=5.0
        ... )
        >>> analyzer.export_csv("convergence.csv")
    """

    def __init__(self):
        """初始化收敛分析器。"""
        self.history: List[IterationData] = []
        self.start_time: Optional[datetime] = None
        logger.info("ConvergenceAnalyzer initialized")

    def add_iteration(
        self,
        iteration: int,
        residuals: Optional[Dict[str, float]] = None,
        cfl: float = 0.0,
        coefficients: Optional[AerodynamicCoefficients] = None
    ) -> None:
        """把一次迭代的数据加入历史。

        Args:
            iteration: 迭代数
            residuals: 各方程的残差值
            cfl: CFL 数
            coefficients: 气动系数（可选）

        Example:
            >>> analyzer.add_iteration(
            ...     iteration=100,
            ...     residuals={'continuity': 1e-5},
            ...     cfl=10.0
            ... )
        """
        if residuals is None:
            residuals = {}

        data = IterationData(
            iteration=iteration,
            residuals=residuals.copy(),
            cfl=cfl,
            coefficients=coefficients,
            timestamp=datetime.now()
        )

        self.history.append(data)

        # 每 10 次迭代记录一次进度
        if iteration % 10 == 0:
            res_str = ", ".join([f"{k}={v:.2e}" for k, v in residuals.items()])
            logger.info(f"Iteration {iteration}: residuals=[{res_str}], CFL={cfl:.2f}")

    def export_csv(self, output_path: str) -> Path:
        """把收敛历史导出为 CSV 文件。

        Args:
            output_path: 输出 CSV 文件路径

        Returns:
            Path: 导出文件的路径

        Raises:
            IOError: 文件写入错误

        Example:
            >>> path = analyzer.export_csv("convergence.csv")
        """
        output_path = Path(output_path)
        if not output_path.suffix:
            output_path = output_path.with_suffix('.csv')

        logger.info(f"Exporting convergence history to: {output_path}")

        try:
            with open(output_path, 'w', newline='') as f:
                writer = csv.writer(f)

                # 写表头
                header = ['iteration', 'cfl']
                if self.history:
                    header.extend(sorted(self.history[0].residuals.keys()))
                    if self.history[0].coefficients:
                        header.extend(['Cd', 'Cl', 'Cm', 'Cs', 'Cy', 'Cr'])
                writer.writerow(header)

                # 写数据行
                for data in self.history:
                    row = [data.iteration, data.cfl]
                    res_keys = sorted(data.residuals.keys())
                    row.extend([data.residuals.get(k, 0.0) for k in res_keys])

                    if data.coefficients:
                        coeffs = data.coefficients.to_dict()
                        row.extend([
                            coeffs.get('Cd', 0.0),
                            coeffs.get('Cl', 0.0),
                            coeffs.get('Cm', 0.0),
                            coeffs.get('Cs', 0.0),
                            coeffs.get('Cy', 0.0),
                            coeffs.get('Cr', 0.0)
                        ])

                    writer.writerow(row)

            logger.success(f"Convergence history exported: {output_path}")
            return output_path

        except IOError as e:
            logger.error(f"Failed to export CSV: {e}")
            raise

    def get_summary(self, computation_time: float = 0.0) -> SimulationSummary:
        """获取仿真汇总统计。

        Args:
            computation_time: 总计算时间（秒）

        Returns:
            SimulationSummary: 汇总统计
        """
        if not self.history:
            return SimulationSummary()

        # 取首尾两次迭代
        first_iter = self.history[0]
        last_iter = self.history[-1]

        # 检查收敛（简单判据：残差 < 1e-4）。空的 residuals 字典**不能**
        # 算作已收敛——对空可迭代对象调用 all() 会因为"真空为真"而返回
        # True，没有这个保护，一个从未填充过逐方程残差的调用方（例如只
        # 记录了 CFL/系数）会得到一个假阳性的"已收敛"汇总。
        converged = bool(last_iter.residuals) and all(v < 1e-4 for v in last_iter.residuals.values())

        summary = SimulationSummary(
            total_iterations=len(self.history),
            final_residuals=last_iter.residuals.copy(),
            initial_coefficients=first_iter.coefficients,
            final_coefficients=last_iter.coefficients,
            computation_time=computation_time,
            converged=converged,
            convergence_criteria={'residual_threshold': 1e-4}
        )

        logger.info(f"Simulation summary:\n"
                   f"  Iterations: {summary.total_iterations}\n"
                   f"  Converged:  {summary.converged}\n"
                   f"  Time:       {summary.computation_time:.1f}s")

        return summary


class SimulationReport:
    """仿真报告生成器。

    生成包含仿真参数、收敛历史和最终结果的完整 JSON 报告。

    Attributes:
        config: 仿真配置
        analyzer: 收敛分析器

    Example:
        >>> report = SimulationReport(config, analyzer)
        >>> report.generate("report.json", computation_time=3600.0)
    """

    def __init__(
        self,
        config: dict,
        analyzer: ConvergenceAnalyzer
    ):
        """初始化报告生成器。

        Args:
            config: 仿真配置字典
            analyzer: 带历史数据的收敛分析器
        """
        self.config = config
        self.analyzer = analyzer
        logger.info("SimulationReport initialized")

    def generate(
        self,
        output_path: str,
        computation_time: float = 0.0,
        metadata: Optional[Dict] = None
    ) -> Path:
        """生成 JSON 格式的仿真报告。

        Args:
            output_path: 输出 JSON 文件路径
            computation_time: 总计算时间（秒）
            metadata: 额外元数据（可选）

        Returns:
            Path: 生成的报告路径

        Raises:
            IOError: 文件写入错误

        Example:
            >>> path = report.generate("report.json", computation_time=3600.0)
        """
        output_path = Path(output_path)
        if not output_path.suffix:
            output_path = output_path.with_suffix('.json')

        logger.info(f"Generating simulation report: {output_path}")

        # 构建报告结构
        report = {
            'metadata': {
                'software': 'AutoFlowCFD',
                'version': __version__,
                'generated_at': datetime.now().isoformat(),
                **(metadata or {})
            },
            'configuration': self.config,
            'summary': self.analyzer.get_summary(computation_time).to_dict(),
            'convergence_history': {
                'total_iterations': len(self.analyzer.history),
                'final_residuals': self.analyzer.history[-1].residuals if self.analyzer.history else {},
            }
        }

        # 若有气动系数则加入报告
        if self.analyzer.history and self.analyzer.history[-1].coefficients:
            report['aerodynamic_coefficients'] = \
                self.analyzer.history[-1].coefficients.to_dict()

        try:
            with open(output_path, 'w') as f:
                json.dump(report, f, indent=2, default=str)

            logger.success(f"Simulation report generated: {output_path}")
            return output_path

        except IOError as e:
            logger.error(f"Failed to generate report: {e}")
            raise
