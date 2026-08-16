"""Unit tests for NAS file parser."""

import pytest
import numpy as np
from pathlib import Path
from autoflowcfd.grid.nas_io.parser import NASParser, NASFormatError, NASParseError


class TestNASParser:
    """Test suite for NASParser."""

    def test_parser_initialization(self, tmp_path) -> None:
        """Test parser initialization with valid file."""
        # Create a dummy .nas file
        nas_file = tmp_path / "test.nas"
        nas_file.write_text("$ Test NAS file\n")
        
        parser = NASParser(str(nas_file))
        assert parser.file_path == nas_file
        assert parser.encoding == 'UTF-8'
    
    def test_parser_file_not_found(self) -> None:
        """Test that nonexistent file raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            NASParser("nonexistent_file.nas")
    
    def test_parser_invalid_extension_warning(self, tmp_path) -> None:
        """Test warning for non-standard file extension."""
        txt_file = tmp_path / "test.txt"
        txt_file.write_text("$ Test file\n")
        
        # Should not raise, just warn
        parser = NASParser(str(txt_file))
        assert parser is not None
    
    def test_detect_version_v24(self, tmp_path) -> None:
        """Test version detection for v24 format."""
        nas_content = """$ ANSA V24 NASTRAN FILE
GRID,1,,0.0,0.0,0.0
GRID,2,,1.0,0.0,0.0
GRID,3,,0.0,1.0,0.0
CTRIA3,1,1,1,2,3
"""
        nas_file = tmp_path / "test_v24.nas"
        nas_file.write_text(nas_content)
        
        parser = NASParser(str(nas_file))
        version = parser._detect_version()
        assert version == "v24"
    
    def test_parse_simple_mesh(self, tmp_path) -> None:
        """Test parsing a simple triangular mesh."""
        nas_content = """$ Simple triangular mesh
GRID,1,,0.0,0.0,0.0
GRID,2,,1.0,0.0,0.0
GRID,3,,0.0,1.0,0.0
GRID,4,,1.0,1.0,0.0
CTRIA3,1,1,1,2,3
CTRIA3,2,1,2,4,3
"""
        nas_file = tmp_path / "simple_mesh.nas"
        nas_file.write_text(nas_content)
        
        parser = NASParser(str(nas_file))
        grid = parser.parse()
        
        assert grid.node_count == 4
        assert grid.cell_count == 2
        assert grid.metadata.file_format == "v24"
    
    def test_parse_empty_file(self, tmp_path) -> None:
        """Test parsing empty NAS file."""
        nas_file = tmp_path / "empty.nas"
        nas_file.write_text("")
        
        parser = NASParser(str(nas_file))
        
        with pytest.raises(NASParseError, match="No nodes found"):
            parser.parse()
    
    def test_parse_no_cells(self, tmp_path) -> None:
        """Test parsing file with nodes but no cells."""
        nas_content = """GRID,1,,0.0,0.0,0.0
GRID,2,,1.0,0.0,0.0
GRID,3,,0.0,1.0,0.0
"""
        nas_file = tmp_path / "no_cells.nas"
        nas_file.write_text(nas_content)
        
        parser = NASParser(str(nas_file))
        
        with pytest.raises(NASParseError, match="No cells found"):
            parser.parse()
    
    def test_parse_with_comments(self, tmp_path) -> None:
        """Test parsing file with comment lines."""
        nas_content = """$ This is a comment
# Another comment style
GRID,1,,0.0,0.0,0.0
$ Comment in middle
GRID,2,,1.0,0.0,0.0
GRID,3,,0.0,1.0,0.0
$ Cell definitions
CTRIA3,1,1,1,2,3
"""
        nas_file = tmp_path / "with_comments.nas"
        nas_file.write_text(nas_content)
        
        parser = NASParser(str(nas_file))
        grid = parser.parse()
        
        assert grid.node_count == 3
        assert grid.cell_count == 1
    
    def test_parse_negative_coordinates(self, tmp_path) -> None:
        """Test parsing nodes with negative coordinates."""
        nas_content = """GRID,1,,-1.0,-2.0,-3.0
GRID,2,,4.0,5.0,6.0
GRID,3,,0.0,0.0,0.0
CTRIA3,1,1,1,2,3
"""
        nas_file = tmp_path / "negative_coords.nas"
        nas_file.write_text(nas_content)

        # units='m' disables the (now-mandatory-by-default) mm->m
        # conversion, since this test is about negative-coordinate parsing
        # specifically, not unit handling - comparing against raw values
        # keeps that separate.
        parser = NASParser(str(nas_file), units='m')
        grid = parser.parse()

        assert grid.node_count == 3
        np.testing.assert_almost_equal(grid.nodes.x[0], -1.0)
        np.testing.assert_almost_equal(grid.nodes.y[0], -2.0)

    def test_parse_scientific_notation(self, tmp_path) -> None:
        """Test parsing coordinates in scientific notation."""
        nas_content = """GRID,1,,1.0e-3,2.5e+2,3.14e0
GRID,2,,4.0e1,5.0e-1,6.0e2
GRID,3,,0.0,0.0,0.0
CTRIA3,1,1,1,2,3
"""
        nas_file = tmp_path / "scientific.nas"
        nas_file.write_text(nas_content)

        parser = NASParser(str(nas_file), units='m')
        grid = parser.parse()

        assert grid.node_count == 3
        np.testing.assert_almost_equal(grid.nodes.x[0], 1.0e-3)
        np.testing.assert_almost_equal(grid.nodes.y[0], 2.5e+2)

    def test_bounding_box_computation(self, tmp_path) -> None:
        """Test automatic bounding box computation."""
        nas_content = """GRID,1,,0.0,0.0,0.0
GRID,2,,1.0,2.0,3.0
GRID,3,,4.0,5.0,6.0
CTRIA3,1,1,1,2,3
"""
        nas_file = tmp_path / "bbox_test.nas"
        nas_file.write_text(nas_content)

        parser = NASParser(str(nas_file), units='m')
        grid = parser.parse()

        bbox = grid.metadata.bounding_box
        assert bbox is not None
        # Order matches parser_core.py's _compute_bounding_box producer:
        # (min_x, max_x, min_y, max_y, min_z, max_z) - see grid_metadata.py.
        min_x, max_x, min_y, max_y, min_z, max_z = bbox

        assert min_x == 0.0
        assert max_x == 4.0
        assert min_y == 0.0
        assert max_y == 5.0
        assert min_z == 0.0
        assert max_z == 6.0

    # test_get_file_info was removed: it asserted against
    # NASParser.get_file_info(), a method that does not exist anywhere in
    # parser_core.py (confirmed by grep across src/) and has no evidence of
    # ever having existed - not a regression to fix, a stale test for a
    # feature that was never implemented.
    
    def test_large_mesh_parsing(self, tmp_path) -> None:
        """Test parsing a larger mesh (performance test)."""
        # Generate a mesh with 1000 nodes and ~2000 cells
        lines = []
        node_id = 1
        for i in range(100):
            for j in range(10):
                x = i * 0.1
                y = j * 0.1
                z = 0.0
                lines.append(f"GRID,{node_id},,{x:.6f},{y:.6f},{z:.6f}")
                node_id += 1
        
        cell_id = 1
        for i in range(99):
            for j in range(9):
                n1 = i * 10 + j + 1
                n2 = n1 + 1
                n3 = n1 + 10
                n4 = n3 + 1
                lines.append(f"CTRIA3,{cell_id},1,{n1},{n2},{n3}")
                cell_id += 1
                lines.append(f"CTRIA3,{cell_id},1,{n2},{n4},{n3}")
                cell_id += 1
        
        nas_content = "\n".join(lines)
        nas_file = tmp_path / "large_mesh.nas"
        nas_file.write_text(nas_content)
        
        parser = NASParser(str(nas_file))
        grid = parser.parse()
        
        assert grid.node_count == 1000
        assert grid.cell_count > 0
    
    def test_malformed_grid_card(self, tmp_path) -> None:
        """Test handling of malformed GRID cards."""
        nas_content = """GRID,1,,0.0,0.0,0.0
GRID,INVALID_LINE
GRID,2,,1.0,0.0,0.0
GRID,3,,0.0,1.0,0.0
CTRIA3,1,1,1,2,3
"""
        nas_file = tmp_path / "malformed.nas"
        nas_file.write_text(nas_content)
        
        parser = NASParser(str(nas_file))
        grid = parser.parse()
        
        # Should parse valid lines and skip invalid ones
        assert grid.node_count >= 2  # At least the valid nodes
    
    def test_node_indexing_zero_based(self, tmp_path) -> None:
        """Test that Nastran 1-based indices are converted to 0-based."""
        nas_content = """GRID,1,,0.0,0.0,0.0
GRID,2,,1.0,0.0,0.0
GRID,3,,0.0,1.0,0.0
CTRIA3,1,1,1,2,3
"""
        nas_file = tmp_path / "indexing_test.nas"
        nas_file.write_text(nas_content)
        
        parser = NASParser(str(nas_file))
        grid = parser.parse()
        
        # Connectivity should use 0-based indices
        connectivity = grid.cells.connectivity[0]
        assert connectivity[0] == 0  # Node 1 in Nastran -> index 0
        assert connectivity[1] == 1  # Node 2 in Nastran -> index 1
        assert connectivity[2] == 2  # Node 3 in Nastran -> index 2
