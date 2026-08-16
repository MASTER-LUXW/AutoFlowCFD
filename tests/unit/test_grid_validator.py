"""Unit tests for grid quality validator."""

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
    """Test suite for GridValidator."""

    @pytest.fixture
    def simple_grid(self) -> GridData:
        """Create a simple triangular mesh for testing."""
        # Create an equilateral triangle mesh (ideal quality)
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
        """Create a stretched triangle mesh (poor quality)."""
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
        """Test validation of ideal equilateral triangle."""
        validator = GridValidator(simple_grid)
        results = validator.validate()
        
        assert results['passed'] is True
        assert results['aspect_ratio']['max'] < 2.0  # Near 1.0 for equilateral
        assert results['skewness']['max'] < 0.1  # Near 0.0 for equilateral
        assert results['jacobian']['min'] > 0.0
    
    def test_validate_stretched_triangle(self, stretched_grid: GridData) -> None:
        """Test validation of stretched triangle."""
        validator = GridValidator(stretched_grid)
        results = validator.validate()
        
        # Stretched triangle should have high aspect ratio
        assert results['aspect_ratio']['max'] > 10.0
        # May or may not pass depending on threshold
        assert 'aspect_ratio' in results
    
    def test_aspect_ratio_calculation(self, simple_grid: GridData) -> None:
        """Test aspect ratio calculation."""
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
        """Test Jacobian determinant calculation."""
        validator = GridValidator(simple_grid)
        jac_stats = validator._check_jacobian()
        
        assert 'max' in jac_stats
        assert 'avg' in jac_stats
        assert 'min' in jac_stats
        assert jac_stats['min'] > 0.0  # Valid triangle has positive Jacobian
        assert 'negative_count' in jac_stats
    
    def test_validation_summary(self, simple_grid: GridData) -> None:
        """Test validation summary generation."""
        validator = GridValidator(simple_grid)
        results = validator.validate()
        
        summary = results['summary']
        assert "GRID QUALITY VALIDATION REPORT" in summary
        assert "Aspect Ratio:" in summary
        assert "Skewness:" in summary
        assert "Jacobian Determinant:" in summary
        assert "RESULT:" in summary
    
    def test_threshold_violation_detection(self, stretched_grid: GridData) -> None:
        """Test that threshold violations are detected."""
        # Set very strict thresholds to force failure
        validator = GridValidator(stretched_grid)
        validator.thresholds['aspect_ratio_max'] = 5.0
        
        results = validator.validate()
        
        # Should fail due to high aspect ratio
        if results['aspect_ratio']['max'] > 5.0:
            assert results['passed'] is False
    
    def test_quality_histogram(self, simple_grid: GridData) -> None:
        """Test histogram generation for quality metrics."""
        validator = GridValidator(simple_grid)
        
        # Test aspect ratio histogram
        counts, bins = validator.get_quality_histogram('aspect_ratio', bins=10)
        assert len(counts) == 10
        assert len(bins) == 11  # n bins have n+1 edges
    
    def test_invalid_histogram_metric(self, simple_grid: GridData) -> None:
        """Test that invalid metric name raises ValueError."""
        validator = GridValidator(simple_grid)
        
        with pytest.raises(ValueError, match="Invalid metric"):
            validator.get_quality_histogram('invalid_metric')
    
    def test_multiple_cells_validation(self) -> None:
        """Test validation with multiple cells."""
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
        """Test detection of degenerate (collapsed) triangles."""
        # Create a degenerate triangle (all nodes collinear)
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
        
        # Degenerate triangle should have near-zero Jacobian
        assert results['jacobian']['min'] < 1e-10
    
    def test_custom_thresholds(self, simple_grid: GridData) -> None:
        """Test using custom quality thresholds."""
        validator = GridValidator(simple_grid)
        
        # Customize thresholds
        validator.thresholds['aspect_ratio_max'] = 50.0
        validator.thresholds['skewness_max'] = 0.99
        validator.thresholds['jacobian_min'] = 1e-8
        
        results = validator.validate()
        
        # Should use custom thresholds
        assert validator.thresholds['aspect_ratio_max'] == 50.0
    
    def test_validation_with_negative_jacobian(self) -> None:
        """Test handling of cells with inconsistent (flipped) winding.

        A single isolated triangle has no absolute orientation sign in 3D
        without an external reference frame - "swap two vertices" alone
        isn't detectable that way (np.linalg.norm of the face-normal cross
        product is never negative, regardless of winding). What *is*
        detectable is whether two triangles that share an edge agree with
        each other: on a consistently-oriented surface they must traverse
        their shared edge in opposite directions. This fixture builds two
        triangles sharing an edge, with the second one's vertex order
        reversed, so the check has an actual inconsistency to find.
        """
        nodes = NodeArray(
            x=np.array([0.0, 1.0, 1.0, 0.0], dtype=np.float64),
            y=np.array([0.0, 0.0, 1.0, 1.0], dtype=np.float64),
            z=np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float64)
        )

        cells = CellArray(
            connectivity=np.array([
                [0, 1, 2],  # consistently oriented
                [0, 3, 2],  # shares edge (0,2) with the same winding as
                            # the first triangle instead of the opposite -
                            # one of the pair is flipped relative to the
                            # other
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

        # Both triangles sharing the inconsistently-wound edge are flagged,
        # and the mismatch now actually fails validation.
        assert results['jacobian']['negative_count'] == 2
        assert results['passed'] is False
