"""Unit tests for outlet boundary conditions and mesh quality validation."""

import pytest
import numpy as np
from autoflowcfd.boundary import OutletCharacteristicBC, OutletSpongeBC
from autoflowcfd.grid import MeshQualityValidator, MeshQualityReport


class TestOutletCharacteristicBC:
    """Test suite for characteristic-based outlet BC.

    apply() was removed (dead code, never called by the live solve path
    - see conditions.py's module docstring); only construction and
    validate() are tested now.
    """

    def test_creation_and_validate(self):
        """Test constructing and validating a characteristic outlet BC."""
        bc = OutletCharacteristicBC(
            pressure_ref=101325.0,
            relaxation_factor=0.5,
        )
        assert bc.pressure_ref == 101325.0
        assert bc.validate() is True

    def test_validate_invalid_relaxation_factor(self):
        """Test validation rejects an out-of-range relaxation factor."""
        bc = OutletCharacteristicBC(relaxation_factor=1.5)
        with pytest.raises(ValueError, match="Invalid relaxation_factor"):
            bc.validate()


class TestOutletSpongeBC:
    """Test suite for sponge layer outlet BC."""
    
    def test_damping_factor_calculation(self):
        """Test damping factor varies correctly with position."""
        bc = OutletSpongeBC(
            damping_strength=0.5,
            sponge_fraction=0.1,
            coordinate_axis=0
        )
        
        domain_min = 0.0
        domain_max = 1.0
        
        # Before sponge layer: zero damping
        cell_before = np.array([0.8, 0.5, 0.5])
        damping_before = bc.compute_damping_factor(cell_before, domain_min, domain_max)
        assert damping_before == pytest.approx(0.0)
        
        # At outlet: maximum damping
        cell_at_outlet = np.array([1.0, 0.5, 0.5])
        damping_at_outlet = bc.compute_damping_factor(cell_at_outlet, domain_min, domain_max)
        assert damping_at_outlet == pytest.approx(0.5)
        
        # Mid-sponge: intermediate damping
        cell_mid = np.array([0.95, 0.5, 0.5])
        damping_mid = bc.compute_damping_factor(cell_mid, domain_min, domain_max)
        assert 0.0 < damping_mid < 0.5
    


class TestMeshQualityValidator:
    """Test suite for mesh quality validation."""
    
    def test_validate_good_triangle_mesh(self):
        """Test validation of a good quality triangular mesh."""
        validator = MeshQualityValidator()
        
        # Create equilateral triangle mesh
        nodes = np.array([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.5, np.sqrt(3)/2, 0.0],
        ])
        
        cells = np.array([[0, 1, 2]])
        
        report = validator.validate(nodes, cells, cell_type="triangle")
        
        assert report.n_cells == 1
        assert report.n_nodes == 3
        assert report.negative_volumes == 0
        assert report.min_volume > 0  # Area should be positive
        # Equilateral triangle has perfect aspect ratio (≈1.0)
        assert abs(report.min_aspect_ratio - 1.0) < 0.01
        assert abs(report.max_aspect_ratio - 1.0) < 0.01
    
    def test_validate_tetrahedron_mesh(self):
        """Test validation of tetrahedral mesh."""
        validator = MeshQualityValidator()
        
        # Create regular tetrahedron
        nodes = np.array([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.5, np.sqrt(3)/2, 0.0],
            [0.5, np.sqrt(3)/6, np.sqrt(6)/3],
        ])
        
        cells = np.array([[0, 1, 2, 3]])
        
        report = validator.validate(nodes, cells, cell_type="tetrahedron")
        
        assert report.n_cells == 1
        assert report.negative_volumes == 0
        assert report.min_volume > 0
    
    def test_detect_negative_volume(self):
        """Test detection of negative volume (inverted cell)."""
        validator = MeshQualityValidator()
        
        # Create inverted tetrahedron (swap two vertices)
        nodes = np.array([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.5, np.sqrt(3)/2, 0.0],
            [0.5, np.sqrt(3)/6, np.sqrt(6)/3],
        ])
        
        # Inverted by swapping vertices 2 and 3
        cells = np.array([[0, 1, 3, 2]])
        
        report = validator.validate(nodes, cells, cell_type="tetrahedron")
        
        assert report.negative_volumes == 1
        assert report.passed == False  # Should fail due to negative volume
    
    def test_aspect_ratio_stretched_triangle(self):
        """Test aspect ratio calculation for stretched triangle."""
        validator = MeshQualityValidator()
        
        # Create very stretched triangle
        nodes = np.array([
            [0.0, 0.0, 0.0],
            [10.0, 0.0, 0.0],  # Long edge
            [0.0, 0.1, 0.0],   # Short edge
        ])
        
        cells = np.array([[0, 1, 2]])
        
        report = validator.validate(nodes, cells, cell_type="triangle")
        
        # Aspect ratio should be high (stretched)
        assert report.max_aspect_ratio > 10.0
    
    def test_skewness_calculation(self):
        """Test skewness calculation for non-equilateral triangle."""
        validator = MeshQualityValidator()
        
        # Create right triangle (not equilateral)
        nodes = np.array([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ])
        
        cells = np.array([[0, 1, 2]])
        
        report = validator.validate(nodes, cells, cell_type="triangle")
        
        # Right triangle has angles 90°, 45°, 45°.
        # Equiangular skew = max[(theta_max-60)/(180-60), (60-theta_min)/60]
        #                   = max[(90-60)/120, (60-45)/60] = max(0.25, 0.25) = 0.25
        assert report.max_skewness > 0.2
        assert report.max_skewness < 0.3
    
    def test_quality_report_summary(self):
        """Test that quality report generates readable summary."""
        validator = MeshQualityValidator()
        
        nodes = np.array([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.5, 0.866, 0.0],
        ])
        cells = np.array([[0, 1, 2]])
        
        report = validator.validate(nodes, cells, cell_type="triangle")
        
        summary = report.summary()
        
        # Check that summary contains key information
        assert "MESH QUALITY REPORT" in summary
        assert "PASSED" in summary or "FAILED" in summary
        assert str(report.n_cells) in summary
        assert "Volume Quality:" in summary
    
    def test_multiple_cells_validation(self):
        """Test validation with multiple cells."""
        validator = MeshQualityValidator()
        
        # Create two triangles
        nodes = np.array([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.5, 0.866, 0.0],
            [1.5, 0.866, 0.0],
        ])
        
        cells = np.array([
            [0, 1, 2],
            [1, 3, 2],
        ])
        
        report = validator.validate(nodes, cells, cell_type="triangle")
        
        assert report.n_cells == 2
        assert report.negative_volumes == 0

    def test_radius_ratio_skewness_regular_tet(self):
        """Regular tetrahedron should have skewness ~0 (radius-ratio metric)."""
        validator = MeshQualityValidator()

        nodes = np.array([
            [1.0, 1.0, 1.0],
            [1.0, -1.0, -1.0],
            [-1.0, 1.0, -1.0],
            [-1.0, -1.0, 1.0],
        ])
        cells = np.array([[0, 1, 2, 3]])

        report = validator.validate(nodes, cells, cell_type="tetrahedron")

        assert report.max_skewness < 1e-6

    def test_radius_ratio_skewness_degenerate_tet(self):
        """Near-flat (sliver) tetrahedron should have skewness close to 1."""
        validator = MeshQualityValidator()

        nodes = np.array([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.5, 0.5, 0.001],
        ])
        cells = np.array([[0, 1, 2, 3]])

        report = validator.validate(nodes, cells, cell_type="tetrahedron")

        assert report.max_skewness > 0.99

    def test_orthogonality_perfect_for_mirrored_tets(self):
        """Two tets sharing a face, mirrored straight through it, have a
        centroid-connector exactly along the shared face normal (0 deg)."""
        validator = MeshQualityValidator()

        nodes = np.array([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 1.0, 1.0],
        ])
        cells = np.array([[0, 1, 2, 3], [1, 2, 3, 4]], dtype=np.int32)

        report = validator.validate(nodes, cells, cell_type="tetrahedron")

        assert report.orthogonality_max < 1e-6

    def test_orthogonality_detects_skewed_neighbour(self):
        """Two tets sharing a face, with the second tet's apex skewed off to
        one side (not mirrored) - the centroid connector should deviate
        substantially from the shared face normal."""
        validator = MeshQualityValidator()

        nodes = np.array([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [3.0, 3.0, 0.2],
        ])
        cells = np.array([[0, 1, 2, 3], [0, 1, 2, 4]], dtype=np.int32)

        report = validator.validate(nodes, cells, cell_type="tetrahedron")

        assert report.orthogonality_max > 45.0

    def test_adjacent_volume_ratio_flags_mismatched_neighbours(self):
        """Two tets sharing a face with very different volumes should trip
        the adjacent-cell volume ratio check and fail the mesh, even though
        neither cell individually has a negative or degenerate volume."""
        validator = MeshQualityValidator()

        nodes = np.array([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [10.0, 10.0, 10.0],
        ])
        cells = np.array([[0, 1, 2, 3], [1, 2, 3, 4]], dtype=np.int32)

        report = validator.validate(nodes, cells, cell_type="tetrahedron")

        assert report.adjacent_volume_ratio_max > validator.thresholds['max_adjacent_volume_ratio']
        assert report.passed is False

    def test_bl_core_aspect_ratio_split(self):
        """bl_cell_mask should split aspect ratio stats into separate
        BL-region and core-region figures instead of one pooled figure."""
        validator = MeshQualityValidator()

        # Cell 0: a stretched ("BL-like") tet. Cell 1: a regular ("core-like") tet.
        nodes = np.array([
            [0.0, 0.0, 0.0],
            [10.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [11.0, 1.0, 0.0],
            [11.0, 0.0, 1.0],
            [11.0, 1.0, 1.0],
        ])
        cells = np.array([
            [0, 1, 2, 3],
            [1, 4, 5, 6],
        ], dtype=np.int32)
        bl_cell_mask = np.array([True, False])

        report = validator.validate(nodes, cells, cell_type="tetrahedron", bl_cell_mask=bl_cell_mask)

        assert report.bl_max_aspect_ratio is not None
        assert report.core_max_aspect_ratio is not None
        assert report.bl_max_aspect_ratio > report.core_max_aspect_ratio


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
