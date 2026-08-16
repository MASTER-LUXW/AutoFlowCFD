"""Unit tests for grid data structures."""

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
    """Test suite for NodeArray data structure."""

    def test_create_node_array(self) -> None:
        """Test creating a basic NodeArray."""
        nodes = NodeArray(
            x=np.array([0.0, 1.0, 2.0], dtype=np.float64),
            y=np.array([0.0, 0.0, 0.0], dtype=np.float64),
            z=np.array([0.0, 0.0, 0.0], dtype=np.float64)
        )
        
        assert nodes.count == 3
        assert nodes.shape == (3,)
        assert nodes.x.dtype == np.float64
    
    def test_node_array_shape_mismatch(self) -> None:
        """Test that shape mismatch raises ValueError."""
        with pytest.raises(ValueError, match="shape mismatch"):
            NodeArray(
                x=np.array([0.0, 1.0], dtype=np.float64),
                y=np.array([0.0, 0.0, 0.0], dtype=np.float64),
                z=np.array([0.0, 0.0, 0.0], dtype=np.float64)
            )
    
    def test_node_array_wrong_dtype(self) -> None:
        """Test that wrong dtype raises ValueError."""
        with pytest.raises(ValueError, match="must be float64"):
            NodeArray(
                x=np.array([0.0, 1.0], dtype=np.float32),
                y=np.array([0.0, 0.0], dtype=np.float32),
                z=np.array([0.0, 0.0], dtype=np.float32)
            )
    
    def test_get_coordinates(self) -> None:
        """Test getting coordinates as stacked array."""
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
        """Test getting coordinates for specific nodes."""
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
    """Test suite for CellArray data structure."""

    def test_create_cell_array(self) -> None:
        """Test creating a basic CellArray."""
        cells = CellArray(
            connectivity=np.array([[0, 1, 2], [1, 2, 3]], dtype=np.int32),
            cell_type=np.array([0, 0], dtype=np.int32)
        )
        
        assert cells.count == 2
        assert cells.shape == (2, 3)
        assert cells.connectivity.dtype == np.int32
    
    def test_cell_array_wrong_dimensions(self) -> None:
        """Test that wrong connectivity dimensions raise ValueError."""
        with pytest.raises(ValueError, match="must be 2D array"):
            CellArray(
                connectivity=np.array([0, 1, 2], dtype=np.int32),
                cell_type=np.array([0], dtype=np.int32)
            )
    
    def test_cell_array_wrong_columns(self) -> None:
        """Test that non-triangular connectivity raises ValueError."""
        with pytest.raises(ValueError, match="must have 3 columns"):
            CellArray(
                connectivity=np.array([[0, 1, 2, 3]], dtype=np.int32),
                cell_type=np.array([0], dtype=np.int32)
            )
    
    def test_cell_array_count_mismatch(self) -> None:
        """Test that count mismatch raises ValueError."""
        with pytest.raises(ValueError, match="doesn't match"):
            CellArray(
                connectivity=np.array([[0, 1, 2]], dtype=np.int32),
                cell_type=np.array([0, 0], dtype=np.int32)
            )


class TestBoundaryMap:
    """Test suite for BoundaryMap data structure."""

    def test_create_boundary_map(self) -> None:
        """Test creating a basic BoundaryMap."""
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
        """Test that key mismatch raises ValueError."""
        with pytest.raises(ValueError, match="keys mismatch"):
            BoundaryMap(
                groups={"inlet": np.array([0, 1], dtype=np.int32)},
                bc_types={"outlet": "OUTLET"}
            )

    def test_get_group_size(self) -> None:
        """Test getting boundary group size."""
        boundaries = BoundaryMap(
            groups={"wall": np.array([0, 1, 2, 3], dtype=np.int32)},
            bc_types={"wall": "WALL"}
        )

        assert len(boundaries.get_cell_indices("wall")) == 4

    def test_get_nonexistent_group(self) -> None:
        """Test that accessing nonexistent group raises KeyError."""
        boundaries = BoundaryMap(
            groups={"wall": np.array([0], dtype=np.int32)},
            bc_types={"wall": "WALL"}
        )

        with pytest.raises(KeyError):
            boundaries.get_cell_indices("nonexistent")


class TestGridMetadata:
    """Test suite for GridMetadata data structure."""

    def test_create_metadata(self) -> None:
        """Test creating basic metadata."""
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
        """Test that negative counts raise ValueError."""
        with pytest.raises(ValueError, match="cannot be negative"):
            GridMetadata(
                node_count=-1,
                cell_count=100,
                boundary_groups=[],
                file_format="v24"
            )
    
    def test_summary_string(self) -> None:
        """Test metadata summary generation."""
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
    """Test suite for GridData data structure."""

    def test_create_grid_data(self) -> None:
        """Test creating a complete GridData object."""
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
        """Test that metadata count mismatch raises ValueError."""
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
        """Test saving and loading grid data to/from HDF5."""
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
