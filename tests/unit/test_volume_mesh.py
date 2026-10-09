"""VolumeMeshData 与四面体体积计算的单元测试。"""
import numpy as np
import pytest
from autoflowcfd.grid.structures import (
    NodeArray, TetrahedralCells, VolumeMeshData, BoundaryMap, GridMetadata
)


class TestTetrahedralCells:
    """四面体单元操作的测试。"""
    
    def test_compute_single_tet_volume(self):
        """单个四面体的体积计算。"""
        # 创建一个体积已知的简单四面体
        # 顶点：(0,0,0), (1,0,0), (0,1,0), (0,0,1)
        # 体积应为 1/6
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
        """多个四面体的体积计算。"""
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
        """拒绝负体积。"""
        connectivity = np.array([[0, 1, 2, 3]], dtype=np.int32)
        volumes = np.array([-1e-6])  # Negative volume
        
        with pytest.raises(ValueError, match="non-positive volumes"):
            TetrahedralCells(connectivity=connectivity, volumes=volumes)
    
    def test_shape_mismatch_rejection(self):
        """拒绝形状不符。"""
        connectivity = np.array([[0, 1, 2, 3]], dtype=np.int32)
        volumes = np.array([1e-6, 2e-6])  # Wrong size
        
        with pytest.raises(ValueError, match="doesn't match"):
            TetrahedralCells(connectivity=connectivity, volumes=volumes)


class TestVolumeMeshData:
    """Test VolumeMeshData container."""
    
    def test_create_simple_volume_mesh(self):
        """创建简单的体网格。"""
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
        """从 VolumeMeshData 取单元体积。"""
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
        """校验元数据里的数量。"""
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
