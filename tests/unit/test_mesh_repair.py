"""Unit tests for the Stage A mesh quality repair (mesh_gen/mesh_repair.py)."""

import numpy as np

from autoflowcfd.grid.mesh_gen.mesh_repair import (
    smooth_bad_cells,
    compute_movable_node_mask,
    compute_bl_thickness_limit_override,
)
from autoflowcfd.grid.mesh_gen.mesh_prism_to_tet import orient_tetrahedra
from autoflowcfd.grid.validation.quality_validator import MeshQualityValidator
from autoflowcfd.grid.schema.grid_nodes import NodeArray
from autoflowcfd.grid.mesh_gen.face_extractor import FaceExtractor


def _bipyramid():
    """Square-base bipyramid split into 8 tets around its center C - the
    only node with no boundary-face membership (every other node sits on
    the outer hull). A clean, hand-verifiable fixture (no Delaunay
    near-degeneracy) for testing Stage A's node-eligibility and smoothing.

    Returns (nodes, cells, iC) with cells correctly oriented (all volumes
    positive).
    """
    B0, B1, B2, B3 = [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, -1.0, 0.0]
    T, D, C = [0.0, 0.0, 1.0], [0.0, 0.0, -1.0], [0.0, 0.0, 0.0]
    nodes = np.array([B0, B1, B2, B3, T, D, C])
    iB0, iB1, iB2, iB3, iT, iD, iC = range(7)

    base_edges = [(iB0, iB1), (iB1, iB2), (iB2, iB3), (iB3, iB0)]
    cells = []
    for (a, b) in base_edges:
        cells.append([iC, a, b, iT])
        cells.append([iC, a, b, iD])
    cells = np.array(cells, dtype=np.int64)
    cells = orient_tetrahedra(nodes, cells).astype(np.int32)
    return nodes, cells, iC


def _bipyramid_bl_core_split():
    """Same bipyramid geometry as _bipyramid(), but with cells explicitly
    grouped BL-first (the 4 cells touching the "T" pole) then core (the 4
    touching "D") - the layout compute_movable_node_mask's n_bl_cells
    argument assumes. The 4 "equatorial" faces (iC, Bi, Bi+1) are then
    exactly the BL/core interface, and iC is the only node on that
    interface that ISN'T also a physical-boundary node - the case that
    matters for this test.
    """
    B0, B1, B2, B3 = [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, -1.0, 0.0]
    T, D, C = [0.0, 0.0, 1.0], [0.0, 0.0, -1.0], [0.0, 0.0, 0.0]
    nodes = np.array([B0, B1, B2, B3, T, D, C])
    iB0, iB1, iB2, iB3, iT, iD, iC = range(7)

    base_edges = [(iB0, iB1), (iB1, iB2), (iB2, iB3), (iB3, iB0)]
    bl_cells = [[iC, a, b, iT] for (a, b) in base_edges]
    core_cells = [[iC, a, b, iD] for (a, b) in base_edges]
    cells = np.array(bl_cells + core_cells, dtype=np.int64)
    cells = orient_tetrahedra(nodes, cells).astype(np.int32)
    return nodes, cells, iC, len(bl_cells)


class TestMovableNodeMask:
    def test_only_interior_node_is_movable(self):
        nodes, cells, iC = _bipyramid()
        node_arr = NodeArray(x=nodes[:, 0].copy(), y=nodes[:, 1].copy(), z=nodes[:, 2].copy())
        faces = FaceExtractor.extract_faces(cells, node_arr)

        movable = compute_movable_node_mask(len(nodes), faces)

        assert np.where(movable)[0].tolist() == [iC]

    def test_bl_core_interface_node_excluded_when_n_bl_cells_given(self):
        """Regression test for a real defect found on cube_demo: the
        interior node sitting exactly on the BL/core interface (iC here)
        is movable under the plain boundary-only rule (it touches no
        physical boundary face) but must NOT be movable once n_bl_cells
        is given - moving it would leave the core side's tets (already
        fixed by tetgen against the OLD position) overlapping the BL
        side's (now-moved) tets, exactly as observed for real: BL and
        core cells sharing no node at all ending up spatially
        overlapping."""
        nodes, cells, iC, n_bl_cells = _bipyramid_bl_core_split()
        node_arr = NodeArray(x=nodes[:, 0].copy(), y=nodes[:, 1].copy(), z=nodes[:, 2].copy())
        faces = FaceExtractor.extract_faces(cells, node_arr)

        movable_without_split = compute_movable_node_mask(len(nodes), faces)
        movable_with_split = compute_movable_node_mask(len(nodes), faces, n_bl_cells)

        assert movable_without_split[iC], "sanity: iC is not on any physical boundary face"
        assert not movable_with_split[iC]
        # Nothing else should change - the interface exclusion is
        # additive, not a wholesale change to the boundary rule.
        assert np.array_equal(
            np.where(movable_without_split)[0],
            np.union1d(np.where(movable_with_split)[0], [iC]),
        )

    def test_n_bl_cells_none_keeps_prior_behaviour(self):
        """None (the default) must be a strict no-op - existing callers
        with no BL region at all must see unchanged behaviour."""
        nodes, cells, iC, n_bl_cells = _bipyramid_bl_core_split()
        node_arr = NodeArray(x=nodes[:, 0].copy(), y=nodes[:, 1].copy(), z=nodes[:, 2].copy())
        faces = FaceExtractor.extract_faces(cells, node_arr)

        assert np.array_equal(
            compute_movable_node_mask(len(nodes), faces),
            compute_movable_node_mask(len(nodes), faces, None),
        )


class TestSmoothBadCells:
    def test_recovers_perturbed_interior_node(self):
        """A large-but-volume-safe perturbation of the one interior node
        should be smoothed back toward the geometrically correct position,
        and the mesh should pass MeshQualityValidator afterward."""
        nodes, cells, iC = _bipyramid()
        validator = MeshQualityValidator()

        perturbed = nodes.copy()
        perturbed[iC] += np.array([0.55, 0.35, 0.0])

        report_before = validator.validate(perturbed, cells, cell_type="tetrahedron")
        assert report_before.passed is False

        new_nodes, bad_mask, actions = smooth_bad_cells(perturbed, cells, validator, max_passes=5)

        report_after = validator.validate(new_nodes, cells, cell_type="tetrahedron")
        assert report_after.passed is True
        assert np.allclose(new_nodes[iC], [0.0, 0.0, 0.0], atol=1e-9)
        assert any("moved" in a for a in actions)

    def test_never_moves_boundary_nodes(self):
        nodes, cells, iC = _bipyramid()
        validator = MeshQualityValidator()

        perturbed = nodes.copy()
        perturbed[iC] += np.array([0.55, 0.35, 0.0])

        new_nodes, _, _ = smooth_bad_cells(perturbed, cells, validator, max_passes=5)

        non_center = [i for i in range(len(nodes)) if i != iC]
        assert np.allclose(new_nodes[non_center], perturbed[non_center])

    def test_never_moves_bl_core_interface_node_when_n_bl_cells_given(self):
        """End-to-end version of TestMovableNodeMask's interface test,
        through the actual smoothing entry point: with n_bl_cells given,
        a bad cell whose only movable node (under the plain boundary
        rule) sits on the BL/core interface must be left as still-bad
        rather than "fixed" by moving that node."""
        nodes, cells, iC, n_bl_cells = _bipyramid_bl_core_split()
        validator = MeshQualityValidator()

        perturbed = nodes.copy()
        perturbed[iC] += np.array([0.55, 0.35, 0.0])

        new_nodes, bad_mask, actions = smooth_bad_cells(
            perturbed, cells, validator, max_passes=5, n_bl_cells=n_bl_cells,
        )

        assert np.array_equal(new_nodes, perturbed), "iC must not have moved"
        assert np.any(bad_mask), "the cells iC's perturbation broke must still be reported bad"

    def test_never_introduces_negative_volume(self):
        """Even when the perturbation is large enough to leave some cells
        already invalid going in, Stage A must never make a previously-
        valid cell's volume go negative."""
        nodes, cells, iC = _bipyramid()
        validator = MeshQualityValidator()

        perturbed = nodes.copy()
        perturbed[iC] += np.array([0.9, 0.9, 0.0])  # aggressive, may itself invert a cell

        current_volumes = validator._compute_tetrahedron_volumes(perturbed, cells)
        already_bad = current_volumes <= 0

        new_nodes, _, _ = smooth_bad_cells(perturbed, cells, validator, max_passes=5)

        final_volumes = validator._compute_tetrahedron_volumes(new_nodes, cells)
        newly_negative = (final_volumes <= 0) & ~already_bad
        assert not np.any(newly_negative)

    def test_no_op_on_already_good_mesh(self):
        nodes, cells, iC = _bipyramid()
        validator = MeshQualityValidator()

        new_nodes, bad_mask, actions = smooth_bad_cells(nodes, cells, validator, max_passes=5)

        assert np.allclose(new_nodes, nodes)
        assert not np.any(bad_mask)
        assert any("already within thresholds" in a for a in actions)


class TestStageBBlThicknessLimitOverride:
    def test_bl_thickness_limit_override_targets_only_bad_bl_vertices(self):
        # 3 independent BL "columns", one per surface vertex (0, 1, 2),
        # each spanning 4 layers under the layer-stacked convention
        # (global index = layer_idx * n_surface_nodes + local_idx) so
        # every cell's 4 nodes map back to exactly one surface vertex via
        # `% n_surface_nodes`. Cell 1 (surface vertex 1's column) is
        # flagged bad -> only surface vertex 1 should get capped.
        n_surface_nodes = 3
        cells = np.array([
            [0, 3, 6, 9],    # surface vertex 0, layers 0-3
            [1, 4, 7, 10],   # surface vertex 1, layers 0-3 - BAD
            [2, 5, 8, 11],   # surface vertex 2, layers 0-3
        ], dtype=np.int32)
        bad_cell_mask = np.array([False, True, False])
        n_bl_cells = 3

        limit, affected = compute_bl_thickness_limit_override(
            bad_cell_mask, n_bl_cells, cells, n_surface_nodes,
            cap_thickness=0.01,
        )

        assert limit is not None
        assert 1 in affected
        assert limit[1] == 0.01
        assert np.isinf(limit[0]) and np.isinf(limit[2])

    def test_bl_thickness_limit_override_no_op_when_no_bad_bl_cells(self):
        cells = np.array([[0, 1, 2, 3]], dtype=np.int32)
        bad_cell_mask = np.array([False])

        limit, affected = compute_bl_thickness_limit_override(
            bad_cell_mask, n_bl_cells=1, cells=cells, n_surface_nodes=3,
            cap_thickness=0.01,
        )

        assert limit is None
        assert affected == []
