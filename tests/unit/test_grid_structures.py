"""网格数据结构的单元测试。"""

import pytest
import numpy as np
from autoflowcfd.grid.structures import (
    NodeArray,
    CellArray,
    BoundaryMap,
    GridMetadata,
    GridData,
)


class TestNodeArray:
    """NodeArray 数据结构的测试。"""

    def test_create_node_array(self) -> None:
        """创建基本的 NodeArray。"""
        nodes = NodeArray(
            x=np.array([0.0, 1.0, 2.0], dtype=np.float64),
            y=np.array([0.0, 0.0, 0.0], dtype=np.float64),
            z=np.array([0.0, 0.0, 0.0], dtype=np.float64)
        )
        
        assert nodes.count == 3
        assert nodes.shape == (3,)
        assert nodes.x.dtype == np.float64
    
    def test_node_array_shape_mismatch(self) -> None:
        """形状不符抛 ValueError。"""
        with pytest.raises(ValueError, match="shape mismatch"):
            NodeArray(
                x=np.array([0.0, 1.0], dtype=np.float64),
                y=np.array([0.0, 0.0, 0.0], dtype=np.float64),
                z=np.array([0.0, 0.0, 0.0], dtype=np.float64)
            )
    
    def test_node_array_wrong_dtype(self) -> None:
        """dtype 不对抛 ValueError。"""
        with pytest.raises(ValueError, match="must be float64"):
            NodeArray(
                x=np.array([0.0, 1.0], dtype=np.float32),
                y=np.array([0.0, 0.0], dtype=np.float32),
                z=np.array([0.0, 0.0], dtype=np.float32)
            )
    
    def test_get_coordinates(self) -> None:
        """以堆叠数组的形式取坐标。"""
        nodes = NodeArray(
            x=np.array([0.0, 1.0, 2.0], dtype=np.float64),
            y=np.array([0.0, 1.0, 2.0], dtype=np.float64),
            z=np.array([0.0, 0.0, 0.0], dtype=np.float64)
        )
        
        coords = nodes.get_coordinates()
        assert coords.shape == (3, 3)
        np.testing.assert_array_equal(coords[0], [0.0, 0.0, 0.0])
        np.testing.assert_array_equal(coords[1], [1.0, 1.0, 0.0])
    
    def test_get_coordinates_with_indices(self) -> None:
        """取指定节点的坐标。"""
        nodes = NodeArray(
            x=np.array([0.0, 1.0, 2.0, 3.0], dtype=np.float64),
            y=np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float64),
            z=np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float64)
        )
        
        indices = np.array([0, 2], dtype=np.int32)
        coords = nodes.get_coordinates(indices)
        assert coords.shape == (2, 3)
        np.testing.assert_array_equal(coords[0], [0.0, 0.0, 0.0])
        np.testing.assert_array_equal(coords[1], [2.0, 0.0, 0.0])


class TestCellArray:
    """CellArray 数据结构的测试。"""

    def test_create_cell_array(self) -> None:
        """创建基本的 CellArray。"""
        cells = CellArray(
            connectivity=np.array([[0, 1, 2], [1, 2, 3]], dtype=np.int32),
            cell_type=np.array([0, 0], dtype=np.int32)
        )
        
        assert cells.count == 2
        assert cells.shape == (2, 3)
        assert cells.connectivity.dtype == np.int32
    
    def test_cell_array_wrong_dimensions(self) -> None:
        """连接关系维数不对抛 ValueError。"""
        with pytest.raises(ValueError, match="must be 2D array"):
            CellArray(
                connectivity=np.array([0, 1, 2], dtype=np.int32),
                cell_type=np.array([0], dtype=np.int32)
            )
    
    def test_cell_array_wrong_columns(self) -> None:
        """非三角形的连接关系抛 ValueError。"""
        with pytest.raises(ValueError, match="must have 3 columns"):
            CellArray(
                connectivity=np.array([[0, 1, 2, 3]], dtype=np.int32),
                cell_type=np.array([0], dtype=np.int32)
            )
    
    def test_cell_array_count_mismatch(self) -> None:
        """数量不符抛 ValueError。"""
        with pytest.raises(ValueError, match="doesn't match"):
            CellArray(
                connectivity=np.array([[0, 1, 2]], dtype=np.int32),
                cell_type=np.array([0, 0], dtype=np.int32)
            )


class TestBoundaryMap:
    """BoundaryMap 数据结构的测试。"""

    def test_create_boundary_map(self) -> None:
        """创建基本的 BoundaryMap。"""
        boundaries = BoundaryMap(
            groups={
                "inlet": np.array([0, 1, 2], dtype=np.int32),
                "outlet": np.array([3, 4, 5], dtype=np.int32),
            },
            bc_types={
                "inlet": "INLET",
                "outlet": "OUTLET",
            }
        )
        
        assert len(boundaries.groups) == 2
        assert boundaries.boundary_names == ["inlet", "outlet"]
        assert boundaries.get_boundary_type("inlet") == "INLET"

    def test_boundary_map_key_mismatch(self) -> None:
        """键不一致抛 ValueError。"""
        with pytest.raises(ValueError, match="keys mismatch"):
            BoundaryMap(
                groups={"inlet": np.array([0, 1], dtype=np.int32)},
                bc_types={"outlet": "OUTLET"}
            )

    def test_get_group_size(self) -> None:
        """取边界组的大小。"""
        boundaries = BoundaryMap(
            groups={"wall": np.array([0, 1, 2, 3], dtype=np.int32)},
            bc_types={"wall": "WALL"}
        )

        assert len(boundaries.get_cell_indices("wall")) == 4

    def test_get_nonexistent_group(self) -> None:
        """访问不存在的组抛 KeyError。"""
        boundaries = BoundaryMap(
            groups={"wall": np.array([0], dtype=np.int32)},
            bc_types={"wall": "WALL"}
        )

        with pytest.raises(KeyError):
            boundaries.get_cell_indices("nonexistent")


class TestGridMetadata:
    """GridMetadata 数据结构的测试。"""

    def test_create_metadata(self) -> None:
        """创建基本的元数据。"""
        metadata = GridMetadata(
            node_count=1000,
            cell_count=2000,
            boundary_groups=["inlet", "outlet", "wall"],
            file_format="v24"
        )
        
        assert metadata.node_count == 1000
        assert metadata.cell_count == 2000
        assert metadata.file_format == "v24"
    
    def test_negative_node_count(self) -> None:
        """数量为负抛 ValueError。"""
        with pytest.raises(ValueError, match="cannot be negative"):
            GridMetadata(
                node_count=-1,
                cell_count=100,
                boundary_groups=[],
                file_format="v24"
            )
    
    def test_summary_string(self) -> None:
        """生成元数据摘要。"""
        metadata = GridMetadata(
            node_count=1000000,
            cell_count=2000000,
            boundary_groups=["wall"],
            file_format="v24",
            bounding_box=(0.0, 0.0, 0.0, 1.0, 1.0, 1.0)
        )
        
        summary = metadata.summary()
        assert "1,000,000" in summary or "1000000" in summary
        assert "v24" in summary


class TestGridData:
    """GridData 数据结构的测试。"""

    def test_create_grid_data(self) -> None:
        """创建完整的 GridData 对象。"""
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
        
        assert grid.node_count == 4
        assert grid.cell_count == 2
        assert grid.metadata.file_format == "v24"
    
    def test_grid_data_count_mismatch(self) -> None:
        """元数据里的数量不符抛 ValueError。"""
        nodes = NodeArray(
            x=np.array([0.0, 1.0], dtype=np.float64),
            y=np.array([0.0, 0.0], dtype=np.float64),
            z=np.array([0.0, 0.0], dtype=np.float64)
        )
        
        cells = CellArray(
            connectivity=np.array([[0, 1, 0]], dtype=np.int32),
            cell_type=np.array([0], dtype=np.int32)
        )
        
        boundaries = BoundaryMap(
            groups={"wall": np.array([0, 1], dtype=np.int32)},
            bc_types={"wall": "WALL"}
        )
        
        # Intentionally wrong metadata
        metadata = GridMetadata(
            node_count=10,  # Wrong!
            cell_count=1,
            boundary_groups=["wall"],
            file_format="v24"
        )
        
        with pytest.raises(ValueError, match="doesn't match"):
            GridData(
                nodes=nodes,
                cells=cells,
                boundaries=boundaries,
                metadata=metadata
            )
    
    def test_hdf5_save_load(self, tmp_path) -> None:
        """网格数据经 HDF5 保存并读回。"""
        pytest.importorskip("h5py")
        
        # Create test grid
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
        
        original_grid = GridData(
            nodes=nodes,
            cells=cells,
            boundaries=boundaries,
            metadata=metadata
        )
        
        # Save to HDF5
        filepath = tmp_path / "test_grid.h5"
        original_grid.save_hdf5(str(filepath))
        
        # Load from HDF5
        loaded_grid = GridData.load_hdf5(str(filepath))
        
        # Verify data integrity
        assert loaded_grid.node_count == original_grid.node_count
        assert loaded_grid.cell_count == original_grid.cell_count
        np.testing.assert_array_equal(loaded_grid.nodes.x, original_grid.nodes.x)
        np.testing.assert_array_equal(loaded_grid.cells.connectivity, original_grid.cells.connectivity)
