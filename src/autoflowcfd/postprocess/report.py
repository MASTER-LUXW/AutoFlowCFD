"""收敛性分析与报告生成模块。

本模块提供分析收敛历史、把收敛曲线导出为 CSV、以及生成 JSON 格式仿真
报告的工具。

核心组件:
    - ConvergenceAnalyzer: 残差与系数历史分析
    - SimulationReport: 完整仿真报告生成

示例:
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


@dataclass
class IterationData:
    """单次迭代的数据。

    Attributes:
        iteration: 迭代数
        residuals: 各方程的残差值
        cfl: CFL 数
        coefficients: 这一步实际算出的系数 {名称: 值}（例如求解器逐步记录的压力部分 Cd_p / Cl_p / Cs_p）；
            没算的系数不出现，不以 0 代替
        timestamp: 迭代时间戳
    """
    iteration: int
    residuals: Dict[str, float] = field(default_factory=dict)
    cfl: float = 0.0
    coefficients: Dict[str, float] = field(default_factory=dict)
    timestamp: Optional[datetime] = None


@dataclass
class SimulationSummary:
    """仿真汇总统计。

    Attributes:
        total_iterations: 总迭代数
        final_residuals: 最终残差值
        initial_coefficients: 第一步实际算出的系数
        final_coefficients: 最后一步实际算出的系数
        computation_time: 总计算时间（秒）
        solve_status: 求解器自己的结论（checkpoint 元数据 `solve_status`：converged / not_converged / finished /
            running，见 `core/fr_solver/state.py::solve_status`）；不知道时为 None
        converged: solve_status == "converged"；solve_status 未知时为 None——报告不按别的阈值重新判定
    """
    total_iterations: int = 0
    final_residuals: Dict[str, float] = field(default_factory=dict)
    initial_coefficients: Dict[str, float] = field(default_factory=dict)
    final_coefficients: Dict[str, float] = field(default_factory=dict)
    computation_time: float = 0.0
    solve_status: Optional[str] = None
    converged: Optional[bool] = None

    def to_dict(self) -> Dict:
        """转换成字典。"""
        return asdict(self)


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
        coefficients: Optional[Dict[str, float]] = None
    ) -> None:
        """把一次迭代的数据加入历史。

        Args:
            iteration: 迭代数
            residuals: 各方程的残差值
            cfl: CFL 数
            coefficients: 这一步实际算出的系数 {名称: 值}（可选，没算的不要填）

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
            coefficients=dict(coefficients or {}),
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

        # 列取全历史的并集（湍流方程随阶段出现 / 消失、只有部分步算了系数时都不错位）；某步没有的量留空。
        # 2026-10-20 以前表头取第一行的键、每行按自己的键写，键集合变化时列错位；首行无系数而后续有系数时
        # 系数列没有表头；没算的系数写 0.0。
        residual_keys = sorted({k for data in self.history for k in data.residuals})
        coefficient_keys = list(dict.fromkeys(k for data in self.history for k in data.coefficients))
        try:
            with open(output_path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['iteration', 'cfl', *residual_keys, *coefficient_keys])
                for data in self.history:
                    writer.writerow([data.iteration, data.cfl]
                                    + [data.residuals.get(k, '') for k in residual_keys]
                                    + [data.coefficients.get(k, '') for k in coefficient_keys])

            logger.success(f"Convergence history exported: {output_path}")
            return output_path

        except IOError as e:
            logger.error(f"Failed to export CSV: {e}")
            raise

    def get_summary(self, computation_time: float = 0.0, solve_status: Optional[str] = None) -> SimulationSummary:
        """获取仿真汇总统计。

        Args:
            computation_time: 总计算时间（秒）
            solve_status: 求解器自己的结论（checkpoint 元数据 `solve_status`）。2026-10-20 以前这里用写死的
                "全部残差绝对值 < 1e-4" 重新判定 converged，与求解器实际的判据（相对下降、阶段门控）无关，
                同一次运行报告与求解日志可能结论相反。

        Returns:
            SimulationSummary: 汇总统计
        """
        if not self.history:
            return SimulationSummary(solve_status=solve_status,
                                     converged=None if solve_status is None else solve_status == "converged")

        first_iter, last_iter = self.history[0], self.history[-1]
        summary = SimulationSummary(
            total_iterations=len(self.history),
            final_residuals=last_iter.residuals.copy(),
            initial_coefficients=dict(first_iter.coefficients),
            final_coefficients=dict(last_iter.coefficients),
            computation_time=computation_time,
            solve_status=solve_status,
            converged=None if solve_status is None else solve_status == "converged",
        )
        logger.info(f"Simulation summary:\n"
                    f"  Iterations: {summary.total_iterations}\n"
                    f"  Status:     {summary.solve_status}\n"
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
        analyzer: ConvergenceAnalyzer,
        solve_status: Optional[str] = None,
    ):
        """初始化报告生成器。

        Args:
            config: 仿真配置字典
            analyzer: 带历史数据的收敛分析器
            solve_status: 求解器自己的结论（checkpoint 元数据 `solve_status`），见 ConvergenceAnalyzer.get_summary
        """
        self.config = config
        self.analyzer = analyzer
        self.solve_status = solve_status
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
            'summary': self.analyzer.get_summary(computation_time, self.solve_status).to_dict(),
            'convergence_history': {
                'total_iterations': len(self.analyzer.history),
                'final_residuals': self.analyzer.history[-1].residuals if self.analyzer.history else {},
            }
        }

        # 最后一步实际算出的系数（只有算过的键）
        if self.analyzer.history and self.analyzer.history[-1].coefficients:
            report['aerodynamic_coefficients'] = dict(self.analyzer.history[-1].coefficients)

        try:
            with open(output_path, 'w') as f:
                json.dump(report, f, indent=2, default=str)

            logger.success(f"Simulation report generated: {output_path}")
            return output_path

        except IOError as e:
            logger.error(f"Failed to generate report: {e}")
            raise
