"""Unit tests for VolumeMeshData and tetrahedral volume computation."""
import numpy as np
import pytest
from autoflowcfd.grid.structures import (
    NodeArray, TetrahedralCells, VolumeMeshData, BoundaryMap, GridMetadata
)


class TestTetrahedralCells:
    """Test tetrahedral cell operations."""
    
    def test_compute_single_tet_volume(self):
        """Test volume calculation for a single tetrahedron."""
        # Create a simple tetrahedron with known volume
        # Vertices: (0,0,0), (1,0,0), (0,1,0), (0,0,1)
        # Volume should be 1/6
        nodes = NodeArray(
            x=np.array([0.0, 1.0, 0.0, 0.0]),
            y=np.array([0.0, 0.0, 1.0, 0.0]),
            z=np.array([0.0, 0.0, 0.0, 1.0])
        )
        
        connectivity = np.array([[0, 1, 2, 3]], dtype=np.int32)
        volumes = TetrahedralCells.compute_volumes(nodes, connectivity)
        
        expected_volume = 1.0 / 6.0
        assert len(volumes) == 1
        assert abs(volumes[0] - expected_volume) < 1e-10
    
    def test_compute_multiple_tets(self):
        """Test volume calculation for multiple tetrahedra."""
        # Create two identical tetrahedra
        nodes = NodeArray(
            x=np.array([0.0, 1.0, 0.0, 0.0, 2.0, 3.0, 2.0]),
            y=np.array([0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]),
            z=np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0])
        )
        
        connectivity = np.array([
            [0, 1, 2, 3],  # First tet
            [4, 5, 6, 3]   # Second tet (shares vertex 3)
        ], dtype=np.int32)
        
        volumes = TetrahedralCells.compute_volumes(nodes, connectivity)
        
        assert len(volumes) == 2
        expected_volume = 1.0 / 6.0
        assert abs(volumes[0] - expected_volume) < 1e-10
        assert abs(volumes[1] - expected_volume) < 1e-10
    
    def test_tetrahedral_cells_validation(self):
        """Test TetrahedralCells validation."""
        connectivity = np.array([[0, 1, 2, 3]], dtype=np.int32)
        volumes = np.array([1e-6])
        
        cells = TetrahedralCells(connectivity=connectivity, volumes=volumes)
        
        assert cells.count == 1
        assert cells.volumes[0] == 1e-6
    
    def test_negative_volume_rejection(self):
        """Test that negative volumes are rejected."""
        connectivity = np.array([[0, 1, 2, 3]], dtype=np.int32)
        volumes = np.array([-1e-6])  # Negative volume
        
        with pytest.raises(ValueError, match="non-positive volumes"):
            TetrahedralCells(connectivity=connectivity, volumes=volumes)
    
    def test_shape_mismatch_rejection(self):
        """Test that mismatched shapes are rejected."""
        connectivity = np.array([[0, 1, 2, 3]], dtype=np.int32)
        volumes = np.array([1e-6, 2e-6])  # Wrong size
        
        with pytest.raises(ValueError, match="doesn't match"):
            TetrahedralCells(connectivity=connectivity, volumes=volumes)


class TestVolumeMeshData:
    """Test VolumeMeshData container."""
    
    def test_create_simple_volume_mesh(self):
        """Test creating a simple volume mesh."""
        nodes = NodeArray(
            x=np.array([0.0, 1.0, 0.0, 0.0]),
            y=np.array([0.0, 0.0, 1.0, 0.0]),
            z=np.array([0.0, 0.0, 0.0, 1.0])
        )
        
        connectivity = np.array([[0, 1, 2, 3]], dtype=np.int32)
        volumes = TetrahedralCells.compute_volumes(nodes, connectivity)
        cells = TetrahedralCells(connectivity=connectivity, volumes=volumes)
        
        boundaries = BoundaryMap(groups={}, bc_types={})
        metadata = GridMetadata(
            node_count=4,
            cell_count=1,
            boundary_groups=[],
            file_format="volume"
        )
        
        volume_mesh = VolumeMeshData(
            nodes=nodes,
            cells=cells,
            boundaries=boundaries,
            metadata=metadata
        )
        
        assert volume_mesh.node_count == 4
        assert volume_mesh.cell_count == 1
        assert abs(volume_mesh.total_volume - 1.0/6.0) < 1e-10
    
    def test_get_cell_volumes(self):
        """Test retrieving cell volumes from VolumeMeshData."""
        nodes = NodeArray(
            x=np.array([0.0, 1.0, 0.0, 0.0]),
            y=np.array([0.0, 0.0, 1.0, 0.0]),
            z=np.array([0.0, 0.0, 0.0, 1.0])
        )
        
        connectivity = np.array([[0, 1, 2, 3]], dtype=np.int32)
        volumes = TetrahedralCells.compute_volumes(nodes, connectivity)
        cells = TetrahedralCells(connectivity=connectivity, volumes=volumes)
        
        boundaries = BoundaryMap(groups={}, bc_types={})
        metadata = GridMetadata(
            node_count=4,
            cell_count=1,
            boundary_groups=[],
            file_format="volume"
        )
        
        volume_mesh = VolumeMeshData(
            nodes=nodes,
            cells=cells,
            boundaries=boundaries,
            metadata=metadata
        )
        
        retrieved_volumes = volume_mesh.get_cell_volumes()
        assert len(retrieved_volumes) == 1
        assert abs(retrieved_volumes[0] - 1.0/6.0) < 1e-10
    
    def test_metadata_consistency_check(self):
        """Test that metadata counts are validated."""
        nodes = NodeArray(
            x=np.array([0.0, 1.0, 0.0, 0.0]),
            y=np.array([0.0, 0.0, 1.0, 0.0]),
            z=np.array([0.0, 0.0, 0.0, 1.0])
        )
        
        connectivity = np.array([[0, 1, 2, 3]], dtype=np.int32)
        volumes = TetrahedralCells.compute_volumes(nodes, connectivity)
        cells = TetrahedralCells(connectivity=connectivity, volumes=volumes)
        
        boundaries = BoundaryMap(groups={}, bc_types={})
        
        # Wrong metadata
        metadata = GridMetadata(
            node_count=5,  # Should be 4
            cell_count=1,
            boundary_groups=[],
            file_format="volume"
        )
        
        with pytest.raises(ValueError, match="node count"):
            VolumeMeshData(
                nodes=nodes,
                cells=cells,
                boundaries=boundaries,
                metadata=metadata
            )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
