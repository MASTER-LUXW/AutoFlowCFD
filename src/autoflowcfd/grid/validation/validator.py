"""网格质量验证器。

本模块实现 CFD 仿真的全面网格质量检查，
验证纵横比、偏斜度和雅可比行列式等几何质量指标，
以确保数值稳定性和精度。

主要功能:
    - 纵横比计算与验证
    - 三角形单元偏斜度测量
    - 雅可比行列式检查
    - 详细统计报告（最大值、平均值、最小值）
    - 可配置的质量阈值
    - 针对大规模网格的性能优化

示例:
    >>> from autoflowcfd.grid.validation.validator import GridValidator
    >>> validator = GridValidator(grid_data)
    >>> results = validator.validate()
    >>> if results['passed']:
    ...     print("网格质量合格")
    ... else:
    ...     print(f"最大纵横比: {results['aspect_ratio']['max']}")
"""

import numpy as np
from typing import Dict, Any, Tuple
from loguru import logger

from ..structures import GridData
from .quality_metrics import compute_triangle_aspect_ratios, compute_triangle_skewness_values


class GridValidator:
    """网格质量校验器
    
    通过计算每个单元的几何指标来验证网格质量。
    确保网格满足稳定 CFD 仿真的质量要求。
    
    Attributes:
        grid_data: 网格数据
        thresholds: 质量阈值配置
    
    Example:
        >>> validator = GridValidator(grid_data)
        >>> results = validator.validate()
        >>> print(f"Aspect ratio max: {results['aspect_ratio']['max']:.2f}")
        >>> print(f"Skewness avg: {results['skewness']['avg']:.3f}")
    """
    
    def __init__(self, grid_data: GridData):
        """初始化校验器
        
        Args:
            grid_data: 网格数据
            
        Raises:
            ValueError: 如果网格数据无效
        """
        self.grid_data = grid_data
        
        # 默认质量阈值基于 CFD 最佳实践
        # 这些值对汽车外部空气动力学是保守的
        self.thresholds = {
            'aspect_ratio_max': 100.0,    # 最大可接受长宽比
            'skewness_max': 0.95,          # 最大可接受偏斜度 (0-1 范围)
            'jacobian_min': 1e-6,          # 最小可接受雅可比行列式
        }
        
        logger.info(
            f"Grid validator initialized for mesh with "
            f"{grid_data.node_count:,} nodes and {grid_data.cell_count:,} cells"
        )
    
    def validate(self) -> Dict[str, Any]:
        """执行完整校验
        
        执行所有质量检查并返回综合结果。
        
        Returns:
            Dict: 校验结果字典,包含以下键:
                - aspect_ratio: Dict with 'max', 'avg', 'min' keys
                - skewness: Dict with 'max', 'avg', 'min' keys
                - jacobian: Dict with 'max', 'avg', 'min' keys
                - passed: Boolean indicating if all checks passed
                - summary: Human-readable summary string
                
        Example:
            >>> results = validator.validate()
            >>> if not results['passed']:
            ...     print("Mesh needs improvement")
        """
        logger.info("Starting grid quality validation...")
        
        results = {
            'aspect_ratio': self._check_aspect_ratio(),
            'skewness': self._check_skewness(),
            'jacobian': self._check_jacobian(),
            'passed': True,
            'summary': ''
        }
        
        # 检查网格是否通过所有质量标准
        failures = []
        
        if results['aspect_ratio']['max'] > self.thresholds['aspect_ratio_max']:
            results['passed'] = False
            failures.append(
                f"Aspect ratio {results['aspect_ratio']['max']:.2f} exceeds "
                f"threshold {self.thresholds['aspect_ratio_max']:.2f}"
            )
            logger.warning(
                f"Aspect ratio exceeds threshold: "
                f"{results['aspect_ratio']['max']:.2f} > {self.thresholds['aspect_ratio_max']:.2f}"
            )
        
        if results['skewness']['max'] > self.thresholds['skewness_max']:
            results['passed'] = False
            failures.append(
                f"Skewness {results['skewness']['max']:.3f} exceeds "
                f"threshold {self.thresholds['skewness_max']:.3f}"
            )
            logger.warning(
                f"Skewness exceeds threshold: "
                f"{results['skewness']['max']:.3f} > {self.thresholds['skewness_max']:.3f}"
            )
        
        if results['jacobian']['min'] < self.thresholds['jacobian_min']:
            results['passed'] = False
            failures.append(
                f"Jacobian {results['jacobian']['min']:.2e} below "
                f"threshold {self.thresholds['jacobian_min']:.2e}"
            )
            logger.warning(
                f"Jacobian below threshold: "
                f"{results['jacobian']['min']:.2e} < {self.thresholds['jacobian_min']:.2e}"
            )

        if results['jacobian']['negative_count'] > 0:
            results['passed'] = False
            failures.append(
                f"{results['jacobian']['negative_count']} cells have inverted "
                f"winding relative to a neighboring cell"
            )
            logger.warning(
                f"{results['jacobian']['negative_count']} cells have inverted "
                f"winding relative to a neighboring cell"
            )

        # 生成摘要
        results['summary'] = self._generate_summary(results, failures)
        
        if results['passed']:
            logger.success("✓ Grid quality validation PASSED")
        else:
            logger.error(f"✗ Grid quality validation FAILED ({len(failures)} issues)")
            for failure in failures:
                logger.error(f"  - {failure}")
        
        return results
    
    def _check_aspect_ratio(self) -> Dict[str, float]:
        """检查长宽比
        
        计算每个三角形单元的长宽比。对于三角形，
        长宽比定义为最长边与最短边的比值。
        
        Returns:
            Dict: 包含 'max'、'avg'、'min' 键的统计字典
            
        注意:
            长宽比 1.0 为完美（等边三角形）。
            值 > 10 表示可能导致数值问题的拉伸单元。
        """
        logger.debug("Computing aspect ratios...")

        connectivity = self.grid_data.cells.connectivity
        node_coords = self.grid_data.nodes.get_coordinates()

        # 复用 quality_metrics.compute_triangle_aspect_ratios（与其他
        # 所有单元类型的长宽比使用相同的相对 epsilon 下限），而不是单独
        # 的临时实现，这样对该公式的未来修复不会在此处静默不同步——
        # 正如本模块的偏斜度检查曾经发生过的那样（见
        # quality_metrics.compute_triangle_skewness_values 的文档字符串）。
        aspect_ratios = compute_triangle_aspect_ratios(node_coords, connectivity)

        # 计算统计量
        stats = {
            'max': float(np.max(aspect_ratios)),
            'avg': float(np.mean(aspect_ratios)),
            'min': float(np.min(aspect_ratios)),
            'std': float(np.std(aspect_ratios))
        }
        
        logger.debug(
            f"Aspect ratio - Max: {stats['max']:.2f}, "
            f"Avg: {stats['avg']:.2f}, Min: {stats['min']:.2f}"
        )
        
        return stats
    
    def _check_skewness(self) -> Dict[str, float]:
        """检查偏斜度
        
        计算三角形单元的偏斜度。偏斜度衡量三角形
        偏离等边三角形的程度。
        
        Returns:
            Dict: 包含 'max'、'avg'、'min' 键的统计字典
            
        注意:
            偏斜度范围从 0（完美等边）到 1（退化）。
            值 > 0.9 表示高度偏斜的单元。
        """
        logger.debug("Computing skewness...")

        connectivity = self.grid_data.cells.connectivity
        node_coords = self.grid_data.nodes.get_coordinates()

        # 复用 quality_metrics.compute_triangle_skewness_values（
        # 标准 Fluent 风格等角偏斜公式），而不是此检查曾经使用的
        # 面积偏差公式。旧公式是一个不同的、非标准的度量，与本项目
        # 在其他地方刻意采用的三角形偏斜度度量不一致——见
        # compute_triangle_skewness_values 自身的文档字符串了解
        # 导致切换的真实假阳性案例（一个有效的 123/29/28 度 BL 顶部三角形）。
        # 此处未同步，这个表面网格检查——网格经过的第一个质量关卡——
        # 会对同一个三角形给出与后续所有检查不同的评分。
        skewness = compute_triangle_skewness_values(node_coords, connectivity)

        # 计算统计量
        stats = {
            'max': float(np.max(skewness)),
            'avg': float(np.mean(skewness)),
            'min': float(np.min(skewness)),
            'std': float(np.std(skewness))
        }
        
        logger.debug(
            f"Skewness - Max: {stats['max']:.3f}, "
            f"Avg: {stats['avg']:.3f}, Min: {stats['min']:.3f}"
        )
        
        return stats
    
    def _check_jacobian(self) -> Dict[str, float]:
        """检查雅可比行列式

        计算每个单元的雅可比行列式（以及通过有向边绕向一致性
        检测相对于邻居方向不一致的单元）。

        实现位于 validator_jacobian.py（按项目 >400 行文件拆分规则
        提取）——见该模块的 check_jacobian 了解完整的 Args/Returns/Notes
        文档。
        """
        from .validator_jacobian import check_jacobian

        return check_jacobian(self.grid_data)

    def _generate_summary(self, results: Dict[str, Any], failures: list) -> str:
        """生成校验结果摘要
        
        创建验证结果的可读摘要。
        
        Args:
            results: Validation results dictionary
            failures: List of failure messages
            
        Returns:
            str: Formatted summary string
        """
        lines = [
            "=" * 60,
            "GRID QUALITY VALIDATION REPORT",
            "=" * 60,
            f"Mesh: {self.grid_data.metadata.node_count:,} nodes, "
            f"{self.grid_data.cell_count:,} cells",
            "-" * 60,
        ]
        
        # 纵横比部分
        ar = results['aspect_ratio']
        lines.append("Aspect Ratio:")
        lines.append(f"  Max: {ar['max']:.2f} (threshold: {self.thresholds['aspect_ratio_max']:.2f})")
        lines.append(f"  Avg: {ar['avg']:.2f}")
        lines.append(f"  Min: {ar['min']:.2f}")
        lines.append("")
        
        # 偏斜度部分
        sk = results['skewness']
        lines.append("Skewness:")
        lines.append(f"  Max: {sk['max']:.3f} (threshold: {self.thresholds['skewness_max']:.3f})")
        lines.append(f"  Avg: {sk['avg']:.3f}")
        lines.append(f"  Min: {sk['min']:.3f}")
        lines.append("")
        
        # 雅可比部分
        jac = results['jacobian']
        lines.append("Jacobian Determinant:")
        lines.append(f"  Max: {jac['max']:.6f}")
        lines.append(f"  Avg: {jac['avg']:.6f}")
        lines.append(f"  Min: {jac['min']:.6f} (threshold: {self.thresholds['jacobian_min']:.2e})")
        if jac['negative_count'] > 0:
            lines.append(f"  WARNING: {jac['negative_count']} cells with negative Jacobian!")
        lines.append("")
        
        # 总体结果
        lines.append("-" * 60)
        if results['passed']:
            lines.append("RESULT: ✓ PASSED - Mesh quality is acceptable")
        else:
            lines.append("RESULT: ✗ FAILED - Mesh quality needs improvement")
            lines.append("")
            lines.append("Issues found:")
            for i, failure in enumerate(failures, 1):
                lines.append(f"  {i}. {failure}")
        
        lines.append("=" * 60)
        
        return "\n".join(lines)
    
    def get_quality_histogram(self, metric: str = 'aspect_ratio', bins: int = 50) -> Tuple[np.ndarray, np.ndarray]:
        """获取质量指标直方图
        
        生成质量指标分布的直方图数据，用于可视化。
        
        Args:
            metric: Quality metric name ('aspect_ratio', 'skewness', or 'jacobian')
            bins: Number of histogram bins
            
        Returns:
            Tuple: (counts, bin_edges) arrays for plotting
            
        Raises:
            ValueError: If metric name is invalid
        """
        if metric == 'aspect_ratio':
            data = self._check_aspect_ratio_values()
        elif metric == 'skewness':
            data = self._compute_skewness_values()
        elif metric == 'jacobian':
            data = self._compute_jacobian_values()
        else:
            raise ValueError(
                f"Invalid metric: {metric}. "
                f"Choose from: 'aspect_ratio', 'skewness', 'jacobian'"
            )
        
        counts, bin_edges = np.histogram(data, bins=bins)
        return counts, bin_edges
    
    def _check_aspect_ratio_values(self) -> np.ndarray:
        """计算所有单元的长宽比值数组

        与 _check_aspect_ratio 复用同一个 quality_metrics 实现（相对
        epsilon 下限），不再维护一份固定 1e-10 epsilon 的独立实现——两者
        曾经用不同公式，导致直方图和 传递/fail 判定对同一批单元给出不一致
        的数值。
        """
        connectivity = self.grid_data.cells.connectivity
        node_coords = self.grid_data.nodes.get_coordinates()
        return compute_triangle_aspect_ratios(node_coords, connectivity)

    def _compute_skewness_values(self) -> np.ndarray:
        """计算所有单元的扭曲度值数组

        与 _check_skewness 复用同一个 quality_metrics 等角偏斜公式，不再
        维护一份旧的面积偏差公式——那套旧公式对形状正常的三角形（例如
        123/29/28 度的 BL 顶部三角形）会产生假阳性，已在 _check_skewness
        里换掉，这里之前一直没跟着换。
        """
        connectivity = self.grid_data.cells.connectivity
        node_coords = self.grid_data.nodes.get_coordinates()
        return compute_triangle_skewness_values(node_coords, connectivity)

    def _compute_jacobian_values(self) -> np.ndarray:
        """计算所有单元的雅可比行列式值数组"""
        connectivity = self.grid_data.cells.connectivity
        nodes = self.grid_data.nodes
        
        cell_coords = np.stack([
            nodes.get_coordinates(connectivity[:, i])
            for i in range(3)
        ], axis=1)
        
        v0 = cell_coords[:, 1] - cell_coords[:, 0]
        v1 = cell_coords[:, 2] - cell_coords[:, 0]
        cross = np.cross(v0, v1)
        
        return np.linalg.norm(cross, axis=1)
