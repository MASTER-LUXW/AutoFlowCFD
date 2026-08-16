"""Integration tests for grid parsing module."""

import pytest
import numpy as np
from pathlib import Path
from autoflowcfd.grid import NASParser, GridValidator


class TestGridParsingIntegration:
    """Integration tests for complete grid parsing workflow."""

    @pytest.fixture
    def sample_nas_file(self, tmp_path) -> Path:
        """Create a sample NAS file for testing."""
        nas_content = """$ AutoFlowCFD Test Mesh
$ Generated for integration testing
GRID,1,,0.0,0.0,0.0
GRID,2,,1.0,0.0,0.0
GRID,3,,0.5,0.866,0.0
GRID,4,,1.5,0.866,0.0
GRID,5,,2.0,0.0,0.0
GRID,6,,0.0,1.0,0.0
GRID,7,,1.0,1.0,0.0
GRID,8,,2.0,1.0,0.0
CTRIA3,1,1,1,2,3
CTRIA3,2,1,2,5,4
CTRIA3,3,1,2,4,3
CTRIA3,4,1,6,7,3
CTRIA3,5,1,7,4,3
CTRIA3,6,1,7,8,4
"""
        nas_file = tmp_path / "test_mesh.nas"
        nas_file.write_text(nas_content)
        return nas_file
    
    def test_complete_parsing_workflow(self, sample_nas_file: Path) -> None:
        """Test complete parsing workflow from file to validated grid."""
        # Step 1: Parse NAS file
        parser = NASParser(str(sample_nas_file))
        grid = parser.parse()
        
        # Verify basic structure
        assert grid.node_count == 8
        assert grid.cell_count == 6
        assert grid.metadata.file_format == "v24"
        
        # Verify data types
        assert grid.nodes.x.dtype == np.float64
        assert grid.cells.connectivity.dtype == np.int32
        
        # Verify metadata
        assert grid.metadata.node_count == 8
        assert grid.metadata.cell_count == 6
        assert len(grid.metadata.boundary_groups) > 0
    
    def test_parse_and_validate_workflow(self, sample_nas_file: Path) -> None:
        """Test parsing followed by quality validation."""
        # Parse
        parser = NASParser(str(sample_nas_file))
        grid = parser.parse()
        
        # Validate
        validator = GridValidator(grid)
        results = validator.validate()
        
        # Check validation completed
        assert 'passed' in results
        assert 'aspect_ratio' in results
        assert 'skewness' in results
        assert 'jacobian' in results
        assert 'summary' in results
    
    def test_hdf5_roundtrip(self, sample_nas_file: Path, tmp_path: Path) -> None:
        """Test saving and loading grid via HDF5."""
        pytest.importorskip("h5py")
        
        # Parse original
        parser = NASParser(str(sample_nas_file))
        original_grid = parser.parse()
        
        # Save to HDF5
        h5_file = tmp_path / "grid.h5"
        original_grid.save_hdf5(str(h5_file))
        
        # Load from HDF5
        loaded_grid = type(original_grid).load_hdf5(str(h5_file))
        
        # Verify data integrity
        assert loaded_grid.node_count == original_grid.node_count
        assert loaded_grid.cell_count == original_grid.cell_count
        np.testing.assert_array_equal(loaded_grid.nodes.x, original_grid.nodes.x)
        np.testing.assert_array_equal(loaded_grid.nodes.y, original_grid.nodes.y)
        np.testing.assert_array_equal(loaded_grid.nodes.z, original_grid.nodes.z)
        np.testing.assert_array_equal(
            loaded_grid.cells.connectivity,
            original_grid.cells.connectivity
        )
    
    def test_boundary_mapping(self, sample_nas_file: Path) -> None:
        """Test boundary condition mapping."""
        parser = NASParser(str(sample_nas_file))
        grid = parser.parse()
        
        # Check boundaries exist
        assert len(grid.boundaries.groups) > 0
        
        # Check BC types are valid
        for bc_type in grid.boundaries.bc_types.values():
            assert bc_type in {"INLET", "OUTLET", "WALL", "SYMMETRY", "FARFIELD"}
    
    def test_coordinate_range(self, sample_nas_file: Path) -> None:
        """Test that coordinates are within expected range."""
        parser = NASParser(str(sample_nas_file))
        grid = parser.parse()
        
        # Check bounding box
        bbox = grid.metadata.bounding_box
        assert bbox is not None
        
        # Order matches parser_core.py's _compute_bounding_box producer:
        # (min_x, max_x, min_y, max_y, min_z, max_z).
        min_x, max_x, min_y, max_y, min_z, max_z = bbox
        
        # Verify bounds match actual data
        assert abs(min_x - np.min(grid.nodes.x)) < 1e-10
        assert abs(max_x - np.max(grid.nodes.x)) < 1e-10
        assert abs(min_y - np.min(grid.nodes.y)) < 1e-10
        assert abs(max_y - np.max(grid.nodes.y)) < 1e-10
    
    def test_connectivity_validity(self, sample_nas_file: Path) -> None:
        """Test that cell connectivity references valid nodes."""
        parser = NASParser(str(sample_nas_file))
        grid = parser.parse()
        
        # All node indices should be valid
        max_node_idx = np.max(grid.cells.connectivity)
        assert max_node_idx < grid.node_count
        
        # No negative indices
        assert np.min(grid.cells.connectivity) >= 0
    
    def test_memory_layout_contiguous(self, sample_nas_file: Path) -> None:
        """Test that arrays use contiguous memory layout."""
        parser = NASParser(str(sample_nas_file))
        grid = parser.parse()
        
        # Check SoA arrays are contiguous
        assert grid.nodes.x.flags['C_CONTIGUOUS']
        assert grid.nodes.y.flags['C_CONTIGUOUS']
        assert grid.nodes.z.flags['C_CONTIGUOUS']
        assert grid.cells.connectivity.flags['C_CONTIGUOUS']
        assert grid.cells.cell_type.flags['C_CONTIGUOUS']
    
    def test_large_mesh_performance(self, tmp_path: Path) -> None:
        """Test performance with larger mesh (10k nodes)."""
        import time
        
        # Generate a larger mesh
        lines = []
        nx, ny = 100, 100
        node_id = 1
        
        for j in range(ny):
            for i in range(nx):
                x = i / (nx - 1)
                y = j / (ny - 1)
                z = 0.0
                lines.append(f"GRID,{node_id},,{x:.6f},{y:.6f},{z:.6f}")
                node_id += 1
        
        cell_id = 1
        for j in range(ny - 1):
            for i in range(nx - 1):
                n1 = j * nx + i + 1
                n2 = n1 + 1
                n3 = n1 + nx
                n4 = n3 + 1
                lines.append(f"CTRIA3,{cell_id},1,{n1},{n2},{n3}")
                cell_id += 1
                lines.append(f"CTRIA3,{cell_id},1,{n2},{n4},{n3}")
                cell_id += 1
        
        nas_content = "\n".join(lines)
        nas_file = tmp_path / "large_mesh.nas"
        nas_file.write_text(nas_content)
        
        # Measure parsing time
        start_time = time.time()
        parser = NASParser(str(nas_file))
        grid = parser.parse()
        parse_time = time.time() - start_time
        
        # Verify mesh size
        assert grid.node_count == nx * ny
        assert grid.cell_count == 2 * (nx - 1) * (ny - 1)
        
        # Performance check (should be reasonable for 10k nodes)
        # Allow up to 5 seconds for this size
        assert parse_time < 5.0, f"Parsing took {parse_time:.2f}s, expected < 5s"
        
        print(f"\nParsed {grid.node_count:,} nodes in {parse_time:.3f}s")
        print(f"Throughput: {grid.node_count / parse_time / 1000:.1f}k nodes/sec")
    
    def test_error_handling_corrupt_file(self, tmp_path: Path) -> None:
        """Test error handling with corrupt/incomplete file."""
        # Create a file with only partial data
        nas_content = """GRID,1,,0.0,0.0,0.0
GRID,2,,1.0,0.0,0.0
"""
        # Missing cells
        nas_file = tmp_path / "corrupt.nas"
        nas_file.write_text(nas_content)
        
        parser = NASParser(str(nas_file))
        
        # Should raise appropriate error
        with pytest.raises(Exception):  # NASParseError
            parser.parse()
    
    def test_metadata_consistency(self, sample_nas_file: Path) -> None:
        """Test that metadata is consistent with actual data."""
        parser = NASParser(str(sample_nas_file))
        grid = parser.parse()
        
        # Metadata should match actual counts
        assert grid.metadata.node_count == len(grid.nodes.x)
        assert grid.metadata.cell_count == len(grid.cells.connectivity)
        
        # Boundary groups should match
        assert len(grid.metadata.boundary_groups) == len(grid.boundaries.groups)
    
    def test_quality_metrics_reasonable(self, sample_nas_file: Path) -> None:
        """Test that quality metrics are in reasonable ranges."""
        parser = NASParser(str(sample_nas_file))
        grid = parser.parse()
        
        validator = GridValidator(grid)
        results = validator.validate()
        
        # Aspect ratio should be >= 1
        assert results['aspect_ratio']['min'] >= 1.0
        
        # Skewness should be in [0, 1]
        assert 0.0 <= results['skewness']['min'] <= 1.0
        assert 0.0 <= results['skewness']['max'] <= 1.0
        
        # Jacobian should be positive for valid mesh
        assert results['jacobian']['min'] >= 0.0
