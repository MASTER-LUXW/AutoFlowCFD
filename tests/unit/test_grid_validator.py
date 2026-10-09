"""网格质量校验器的单元测试。"""

import pytest
import numpy as np
from autoflowcfd.grid.structures import (
    GridData,
    NodeArray,
    CellArray,
    BoundaryMap,
    GridMetadata,
)
from autoflowcfd.grid.validation.validator import GridValidator


class TestGridValidator:
    """GridValidator 的测试。"""

    @pytest.fixture
    def simple_grid(self) -> GridData:
        """创建测试用的简单三角形网格。"""
        # 创建等边三角形网格（质量理想）
        nodes = NodeArray(
            x=np.array([0.0, 1.0, 0.5], dtype=np.float64),
            y=np.array([0.0, 0.0, np.sqrt(3)/2], dtype=np.float64),
            z=np.array([0.0, 0.0, 0.0], dtype=np.float64)
        )
        
        cells = CellArray(
            connectivity=np.array([[0, 1, 2]], dtype=np.int32),
            cell_type=np.array([0], dtype=np.int32)
        )
        
        boundaries = BoundaryMap(
            groups={"wall": np.array([0, 1, 2], dtype=np.int32)},
            bc_types={"wall": "WALL"}
        )
        
        metadata = GridMetadata(
            node_count=3,
            cell_count=1,
            boundary_groups=["wall"],
            file_format="v24"
        )
        
        return GridData(
            nodes=nodes,
            cells=cells,
            boundaries=boundaries,
            metadata=metadata
        )
    
    @pytest.fixture
    def stretched_grid(self) -> GridData:
        """创建拉长的三角形网格（质量差）。"""
        # Create a very stretched triangle
        nodes = NodeArray(
            x=np.array([0.0, 10.0, 0.0], dtype=np.float64),
            y=np.array([0.0, 0.0, 0.1], dtype=np.float64),
            z=np.array([0.0, 0.0, 0.0], dtype=np.float64)
        )
        
        cells = CellArray(
            connectivity=np.array([[0, 1, 2]], dtype=np.int32),
            cell_type=np.array([0], dtype=np.int32)
        )
        
        boundaries = BoundaryMap(
            groups={"wall": np.array([0, 1, 2], dtype=np.int32)},
            bc_types={"wall": "WALL"}
        )
        
        metadata = GridMetadata(
            node_count=3,
            cell_count=1,
            boundary_groups=["wall"],
            file_format="v24"
        )
        
        return GridData(
            nodes=nodes,
            cells=cells,
            boundaries=boundaries,
            metadata=metadata
        )
    
    def test_validator_initialization(self, simple_grid: GridData) -> None:
        """Test validator initialization."""
        validator = GridValidator(simple_grid)
        assert validator.grid_data == simple_grid
        assert 'aspect_ratio_max' in validator.thresholds
        assert 'skewness_max' in validator.thresholds
    
    def test_validate_equilateral_triangle(self, simple_grid: GridData) -> None:
        """校验理想的等边三角形。"""
        validator = GridValidator(simple_grid)
        results = validator.validate()
        
        assert results['passed'] is True
        assert results['aspect_ratio']['max'] < 2.0  # Near 1.0 for equilateral
        assert results['skewness']['max'] < 0.1  # Near 0.0 for equilateral
        assert results['jacobian']['min'] > 0.0
    
    def test_validate_stretched_triangle(self, stretched_grid: GridData) -> None:
        """校验拉长的三角形。"""
        validator = GridValidator(stretched_grid)
        results = validator.validate()
        
        # 拉长的三角形长宽比应当很大
        assert results['aspect_ratio']['max'] > 10.0
        # 是否通过取决于阈值
        assert 'aspect_ratio' in results
    
    def test_aspect_ratio_calculation(self, simple_grid: GridData) -> None:
        """长宽比计算。"""
        validator = GridValidator(simple_grid)
        ar_stats = validator._check_aspect_ratio()
        
        assert 'max' in ar_stats
        assert 'avg' in ar_stats
        assert 'min' in ar_stats
        assert ar_stats['min'] >= 1.0  # Aspect ratio cannot be < 1
        assert ar_stats['max'] >= ar_stats['avg']
    
    def test_skewness_calculation(self, simple_grid: GridData) -> None:
        """Test skewness calculation."""
        validator = GridValidator(simple_grid)
        skew_stats = validator._check_skewness()
        
        assert 'max' in skew_stats
        assert 'avg' in skew_stats
        assert 'min' in skew_stats
        assert 0.0 <= skew_stats['min'] <= 1.0
        assert 0.0 <= skew_stats['max'] <= 1.0
    
    def test_jacobian_calculation(self, simple_grid: GridData) -> None:
        """Jacobian 行列式计算。"""
        validator = GridValidator(simple_grid)
        jac_stats = validator._check_jacobian()
        
        assert 'max' in jac_stats
        assert 'avg' in jac_stats
        assert 'min' in jac_stats
        assert jac_stats['min'] > 0.0  # Valid triangle has positive Jacobian
        assert 'negative_count' in jac_stats
    
    def test_validation_summary(self, simple_grid: GridData) -> None:
        """生成校验摘要。"""
        validator = GridValidator(simple_grid)
        results = validator.validate()
        
        summary = results['summary']
        assert "GRID QUALITY VALIDATION REPORT" in summary
        assert "Aspect Ratio:" in summary
        assert "Skewness:" in summary
        assert "Jacobian Determinant:" in summary
        assert "RESULT:" in summary
    
    def test_threshold_violation_detection(self, stretched_grid: GridData) -> None:
        """能检出超过阈值的情形。"""
        # 把阈值设得很严，强制不通过
        validator = GridValidator(stretched_grid)
        validator.thresholds['aspect_ratio_max'] = 5.0
        
        results = validator.validate()
        
        # 应因长宽比过大而不通过
        if results['aspect_ratio']['max'] > 5.0:
            assert results['passed'] is False
    
    def test_quality_histogram(self, simple_grid: GridData) -> None:
        """生成质量指标的直方图。"""
        validator = GridValidator(simple_grid)
        
        # Test aspect ratio histogram
        counts, bins = validator.get_quality_histogram('aspect_ratio', bins=10)
        assert len(counts) == 10
        assert len(bins) == 11  # n bins have n+1 edges
    
    def test_invalid_histogram_metric(self, simple_grid: GridData) -> None:
        """指标名非法抛 ValueError。"""
        validator = GridValidator(simple_grid)
        
        with pytest.raises(ValueError, match="Invalid metric"):
            validator.get_quality_histogram('invalid_metric')
    
    def test_multiple_cells_validation(self) -> None:
        """多个单元的校验。"""
        # Create a mesh with multiple triangles
        nodes = NodeArray(
            x=np.array([0.0, 1.0, 0.0, 1.0], dtype=np.float64),
            y=np.array([0.0, 0.0, 1.0, 1.0], dtype=np.float64),
            z=np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float64)
        )
        
        cells = CellArray(
            connectivity=np.array([[0, 1, 2], [1, 3, 2]], dtype=np.int32),
            cell_type=np.array([0, 0], dtype=np.int32)
        )
        
        boundaries = BoundaryMap(
            groups={"wall": np.array([0, 1, 2, 3], dtype=np.int32)},
            bc_types={"wall": "WALL"}
        )
        
        metadata = GridMetadata(
            node_count=4,
            cell_count=2,
            boundary_groups=["wall"],
            file_format="v24"
        )
        
        grid = GridData(
            nodes=nodes,
            cells=cells,
            boundaries=boundaries,
            metadata=metadata
        )
        
        validator = GridValidator(grid)
        results = validator.validate()
        
        assert results['passed'] is True
        assert 'aspect_ratio' in results
        assert 'skewness' in results
    
    def test_degenerate_triangle_detection(self) -> None:
        """检出退化（坍缩）的三角形。"""
        # 创建退化三角形（三个节点共线）
        nodes = NodeArray(
            x=np.array([0.0, 1.0, 2.0], dtype=np.float64),
            y=np.array([0.0, 0.0, 0.0], dtype=np.float64),
            z=np.array([0.0, 0.0, 0.0], dtype=np.float64)
        )
        
        cells = CellArray(
            connectivity=np.array([[0, 1, 2]], dtype=np.int32),
            cell_type=np.array([0], dtype=np.int32)
        )
        
        boundaries = BoundaryMap(
            groups={"wall": np.array([0, 1, 2], dtype=np.int32)},
            bc_types={"wall": "WALL"}
        )
        
        metadata = GridMetadata(
            node_count=3,
            cell_count=1,
            boundary_groups=["wall"],
            file_format="v24"
        )
        
        grid = GridData(
            nodes=nodes,
            cells=cells,
            boundaries=boundaries,
            metadata=metadata
        )
        
        validator = GridValidator(grid)
        results = validator.validate()
        
        # 退化三角形的 Jacobian 应接近零
        assert results['jacobian']['min'] < 1e-10
    
    def test_custom_thresholds(self, simple_grid: GridData) -> None:
        """使用自定义质量阈值。"""
        validator = GridValidator(simple_grid)
        
        # Customize thresholds
        validator.thresholds['aspect_ratio_max'] = 50.0
        validator.thresholds['skewness_max'] = 0.99
        validator.thresholds['jacobian_min'] = 1e-8
        
        results = validator.validate()
        
        # Should use custom thresholds
        assert validator.thresholds['aspect_ratio_max'] == 50.0
    
    def test_validation_with_negative_jacobian(self) -> None:
        """处理绕向不一致（翻转）的单元。

        单独一个孤立三角形在三维里没有外部参考系就没有绝对的朝向符号——只"交换
        两个顶点"是检测不出来的（面法向叉积的 np.linalg.norm 不论绕向如何都不会
        为负）。*能*检测的是共享一条边的两个三角形彼此是否一致：朝向一致的曲面上，
        它们必须以相反的方向走过共享边。这个夹具构造两个共享一条边的三角形，
        第二个的顶点顺序反过来，让检查有一个真实的不一致可找。
        """
        nodes = NodeArray(
            x=np.array([0.0, 1.0, 1.0, 0.0], dtype=np.float64),
            y=np.array([0.0, 0.0, 1.0, 1.0], dtype=np.float64),
            z=np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float64)
        )

        cells = CellArray(
            connectivity=np.array([
                [0, 1, 2],  # consistently oriented
                [0, 3, 2],  # 共享边 (0,2)，绕向与第一个三角形相同而不是相反——
                            # 两者之一相对另一个是翻转的
            ], dtype=np.int32),
            cell_type=np.array([0, 0], dtype=np.int32)
        )

        boundaries = BoundaryMap(
            groups={"wall": np.array([0, 1], dtype=np.int32)},
            bc_types={"wall": "WALL"}
        )

        metadata = GridMetadata(
            node_count=4,
            cell_count=2,
            boundary_groups=["wall"],
            file_format="v24"
        )

        grid = GridData(
            nodes=nodes,
            cells=cells,
            boundaries=boundaries,
            metadata=metadata
        )

        validator = GridValidator(grid)
        results = validator.validate()

        # 共享这条绕向不一致的边的两个三角形都被标记，
        # 并且这种不一致现在确实会让校验不通过。
        assert results['jacobian']['negative_count'] == 2
        assert results['passed'] is False
