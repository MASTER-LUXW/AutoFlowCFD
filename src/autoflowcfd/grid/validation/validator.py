"""Grid quality validator.

This module implements comprehensive mesh quality checking for CFD simulations.
It validates geometric quality metrics including aspect ratio, skewness,
and Jacobian determinant to ensure numerical stability and accuracy.

Key Features:
    - Aspect ratio calculation and validation
    - Skewness measurement for triangular elements
    - Jacobian determinant checking
    - Detailed statistical reporting (max, avg, min)
    - Configurable quality thresholds
    - Performance-optimized for large meshes

Example:
    >>> from autoflowcfd.grid.validation.validator import GridValidator
    >>> validator = GridValidator(grid_data)
    >>> results = validator.validate()
    >>> if results['passed']:
    ...     print("Mesh quality is acceptable")
    ... else:
    ...     print(f"Max aspect ratio: {results['aspect_ratio']['max']}")
"""

import numpy as np
from typing import Dict, Any, Tuple
from loguru import logger

from ..structures import GridData


class GridValidator:
    """网格质量校验器
    
    Validates mesh quality by computing geometric metrics for each cell.
    Ensures mesh meets quality requirements for stable CFD simulations.
    
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
            ValueError: If grid data is invalid
        """
        self.grid_data = grid_data
        
        # Default quality thresholds based on CFD best practices
        # These values are conservative for automotive external aerodynamics
        self.thresholds = {
            'aspect_ratio_max': 100.0,    # Max acceptable aspect ratio
            'skewness_max': 0.95,          # Max acceptable skewness (0-1 scale)
            'jacobian_min': 1e-6,          # Min acceptable Jacobian determinant
        }
        
        logger.info(
            f"Grid validator initialized for mesh with "
            f"{grid_data.node_count:,} nodes and {grid_data.cell_count:,} cells"
        )
    
    def validate(self) -> Dict[str, Any]:
        """执行完整校验
        
        Performs all quality checks and returns comprehensive results.
        
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
        
        # Check if mesh passes all quality criteria
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

        # Generate summary
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
        
        Computes aspect ratio for each triangular cell. For triangles,
        aspect ratio is defined as the ratio of longest edge to shortest edge.
        
        Returns:
            Dict: Statistics with 'max', 'avg', 'min' keys
            
        Note:
            Aspect ratio of 1.0 is perfect (equilateral triangle).
            Values > 10 indicate stretched cells that may cause numerical issues.
        """
        logger.debug("Computing aspect ratios...")
        
        connectivity = self.grid_data.cells.connectivity
        nodes = self.grid_data.nodes
        
        # Get node coordinates for all cells
        # Shape: (N_cells, 3, 3) where last dim is (x, y, z)
        cell_coords = np.stack([
            nodes.get_coordinates(connectivity[:, i])
            for i in range(3)
        ], axis=1)
        
        # Compute edge lengths for each triangle
        # Edge 0: node 0 to node 1
        edge0 = np.linalg.norm(cell_coords[:, 1] - cell_coords[:, 0], axis=1)
        # Edge 1: node 1 to node 2
        edge1 = np.linalg.norm(cell_coords[:, 2] - cell_coords[:, 1], axis=1)
        # Edge 2: node 2 to node 0
        edge2 = np.linalg.norm(cell_coords[:, 0] - cell_coords[:, 2], axis=1)
        
        # Stack edges
        edges = np.stack([edge0, edge1, edge2], axis=1)
        
        # Aspect ratio = max_edge / min_edge
        max_edges = np.max(edges, axis=1)
        min_edges = np.min(edges, axis=1)
        
        # Avoid division by zero
        min_edges = np.maximum(min_edges, 1e-10)
        
        aspect_ratios = max_edges / min_edges
        
        # Compute statistics
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
        """检查扭曲度
        
        Computes skewness for triangular cells. Skewness measures how much
        a triangle deviates from an equilateral triangle.
        
        Returns:
            Dict: Statistics with 'max', 'avg', 'min' keys
            
        Note:
            Skewness ranges from 0 (perfect equilateral) to 1 (degenerate).
            Values > 0.9 indicate highly skewed cells.
        """
        logger.debug("Computing skewness...")
        
        connectivity = self.grid_data.cells.connectivity
        nodes = self.grid_data.nodes
        
        # Get node coordinates
        cell_coords = np.stack([
            nodes.get_coordinates(connectivity[:, i])
            for i in range(3)
        ], axis=1)
        
        # Compute edge lengths
        edge0 = np.linalg.norm(cell_coords[:, 1] - cell_coords[:, 0], axis=1)
        edge1 = np.linalg.norm(cell_coords[:, 2] - cell_coords[:, 1], axis=1)
        edge2 = np.linalg.norm(cell_coords[:, 0] - cell_coords[:, 2], axis=1)
        
        # Compute area using cross product
        v0 = cell_coords[:, 1] - cell_coords[:, 0]
        v1 = cell_coords[:, 2] - cell_coords[:, 0]
        cross = np.cross(v0, v1)
        areas = 0.5 * np.linalg.norm(cross, axis=1)
        
        # Ideal area for equilateral triangle with same perimeter
        perimeters = edge0 + edge1 + edge2
        ideal_areas = (np.sqrt(3) / 36) * perimeters**2
        
        # Avoid division by zero
        ideal_areas = np.maximum(ideal_areas, 1e-10)
        
        # Skewness = 1 - (actual_area / ideal_area)
        skewness = 1.0 - (areas / ideal_areas)
        
        # Clamp to [0, 1] range
        skewness = np.clip(skewness, 0.0, 1.0)
        
        # Compute statistics
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

        Computes Jacobian determinant for each cell. The Jacobian measures
        the local scaling factor of the coordinate transformation.

        Returns:
            Dict: Statistics with 'max', 'avg', 'min' keys

        Note:
            Positive Jacobian indicates valid cell orientation.
            Negative or near-zero Jacobian indicates inverted or degenerate cells.
        """
        logger.debug("Computing Jacobian determinants...")

        connectivity = self.grid_data.cells.connectivity
        nodes = self.grid_data.nodes

        # Get node coordinates
        cell_coords = np.stack([
            nodes.get_coordinates(connectivity[:, i])
            for i in range(3)
        ], axis=1)

        # Compute vectors from node 0 to nodes 1 and 2
        v0 = cell_coords[:, 1] - cell_coords[:, 0]
        v1 = cell_coords[:, 2] - cell_coords[:, 0]

        # Compute cross product (gives normal vector)
        cross = np.cross(v0, v1)

        # Jacobian determinant is magnitude of cross product
        jacobians = np.linalg.norm(cross, axis=1)

        # A single triangle's winding has no absolute sign without an
        # external reference frame, so "jacobians < 0" here (np.linalg.norm
        # is never negative) could never fire for any mesh - inverted
        # triangles were silently invisible to this check regardless of
        # input. What *is* well-defined without an external reference is
        # whether neighboring triangles agree with each other: on a
        # consistently-oriented manifold surface, two triangles sharing an
        # edge must traverse that shared edge in opposite directions. If
        # they traverse it in the same direction, one of the pair is
        # flipped relative to its neighbor - detect that via directed-edge
        # sign accumulation instead of the magnitude's sign.
        negative_count = self._count_flipped_triangles(connectivity)

        # Compute statistics
        stats = {
            'max': float(np.max(jacobians)),
            'avg': float(np.mean(jacobians)),
            'min': float(np.min(jacobians)),
            'std': float(np.std(jacobians)),
            'negative_count': negative_count
        }

        logger.debug(
            f"Jacobian - Max: {stats['max']:.6f}, "
            f"Avg: {stats['avg']:.6f}, Min: {stats['min']:.6f}, "
            f"Negative: {stats['negative_count']}"
        )

        return stats

    def _count_flipped_triangles(self, connectivity: np.ndarray) -> int:
        """Count triangles whose winding is inconsistent with a neighbor.

        For every shared (manifold-interior) edge - one that borders
        exactly two triangles - a consistently-oriented surface must
        traverse it in opposite directions from each side. Two triangles
        that instead traverse their shared edge in the same direction
        cannot both be correctly oriented; both are flagged. Edges shared
        by a triangle count other than 2 (open boundary or non-manifold)
        aren't orientation-checkable this way and are skipped here - that
        is a distinct mesh-integrity issue, not this metric's concern.
        """
        n1, n2, n3 = connectivity[:, 0], connectivity[:, 1], connectivity[:, 2]
        directed_edges = np.concatenate([
            np.stack([n1, n2], axis=1),
            np.stack([n2, n3], axis=1),
            np.stack([n3, n1], axis=1),
        ], axis=0)
        owner_cells = np.tile(np.arange(len(connectivity)), 3)

        n_nodes = self.grid_data.nodes.count
        lo = np.minimum(directed_edges[:, 0], directed_edges[:, 1]).astype(np.int64)
        hi = np.maximum(directed_edges[:, 0], directed_edges[:, 1]).astype(np.int64)
        edge_key = lo * n_nodes + hi
        sign = np.where(directed_edges[:, 0] < directed_edges[:, 1], 1, -1)

        order = np.argsort(edge_key, kind='stable')
        sorted_keys = edge_key[order]
        sorted_signs = sign[order]
        sorted_owners = owner_cells[order]

        group_end = np.flatnonzero(
            np.concatenate([sorted_keys[1:] != sorted_keys[:-1], [True]])
        )
        group_start = np.concatenate([[0], group_end[:-1] + 1])
        group_size = group_end - group_start + 1

        pair_mask = group_size == 2
        pair_first = group_start[pair_mask]
        same_direction = sorted_signs[pair_first] == sorted_signs[pair_first + 1]
        bad_first = pair_first[same_direction]

        flipped_cells = np.unique(np.concatenate([
            sorted_owners[bad_first], sorted_owners[bad_first + 1]
        ])) if bad_first.size else np.array([], dtype=owner_cells.dtype)

        return int(flipped_cells.size)
    
    def _generate_summary(self, results: Dict[str, Any], failures: list) -> str:
        """生成校验结果摘要
        
        Creates a human-readable summary of validation results.
        
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
        
        # Aspect ratio section
        ar = results['aspect_ratio']
        lines.append("Aspect Ratio:")
        lines.append(f"  Max: {ar['max']:.2f} (threshold: {self.thresholds['aspect_ratio_max']:.2f})")
        lines.append(f"  Avg: {ar['avg']:.2f}")
        lines.append(f"  Min: {ar['min']:.2f}")
        lines.append("")
        
        # Skewness section
        sk = results['skewness']
        lines.append("Skewness:")
        lines.append(f"  Max: {sk['max']:.3f} (threshold: {self.thresholds['skewness_max']:.3f})")
        lines.append(f"  Avg: {sk['avg']:.3f}")
        lines.append(f"  Min: {sk['min']:.3f}")
        lines.append("")
        
        # Jacobian section
        jac = results['jacobian']
        lines.append("Jacobian Determinant:")
        lines.append(f"  Max: {jac['max']:.6f}")
        lines.append(f"  Avg: {jac['avg']:.6f}")
        lines.append(f"  Min: {jac['min']:.6f} (threshold: {self.thresholds['jacobian_min']:.2e})")
        if jac['negative_count'] > 0:
            lines.append(f"  WARNING: {jac['negative_count']} cells with negative Jacobian!")
        lines.append("")
        
        # Overall result
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
        
        Generates histogram data for visualization of quality metric distribution.
        
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
        """计算所有单元的长宽比值数组"""
        connectivity = self.grid_data.cells.connectivity
        nodes = self.grid_data.nodes
        
        cell_coords = np.stack([
            nodes.get_coordinates(connectivity[:, i])
            for i in range(3)
        ], axis=1)
        
        edge0 = np.linalg.norm(cell_coords[:, 1] - cell_coords[:, 0], axis=1)
        edge1 = np.linalg.norm(cell_coords[:, 2] - cell_coords[:, 1], axis=1)
        edge2 = np.linalg.norm(cell_coords[:, 0] - cell_coords[:, 2], axis=1)
        
        edges = np.stack([edge0, edge1, edge2], axis=1)
        max_edges = np.max(edges, axis=1)
        min_edges = np.maximum(np.min(edges, axis=1), 1e-10)
        
        return max_edges / min_edges
    
    def _compute_skewness_values(self) -> np.ndarray:
        """计算所有单元的扭曲度值数组"""
        connectivity = self.grid_data.cells.connectivity
        nodes = self.grid_data.nodes
        
        cell_coords = np.stack([
            nodes.get_coordinates(connectivity[:, i])
            for i in range(3)
        ], axis=1)
        
        edge0 = np.linalg.norm(cell_coords[:, 1] - cell_coords[:, 0], axis=1)
        edge1 = np.linalg.norm(cell_coords[:, 2] - cell_coords[:, 1], axis=1)
        edge2 = np.linalg.norm(cell_coords[:, 0] - cell_coords[:, 2], axis=1)
        
        v0 = cell_coords[:, 1] - cell_coords[:, 0]
        v1 = cell_coords[:, 2] - cell_coords[:, 0]
        cross = np.cross(v0, v1)
        areas = 0.5 * np.linalg.norm(cross, axis=1)
        
        perimeters = edge0 + edge1 + edge2
        ideal_areas = (np.sqrt(3) / 36) * perimeters**2
        ideal_areas = np.maximum(ideal_areas, 1e-10)
        
        skewness = 1.0 - (areas / ideal_areas)
        return np.clip(skewness, 0.0, 1.0)
    
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
