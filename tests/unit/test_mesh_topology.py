"""Topology and validity tests for generated volume meshes.

These tests target the mesh-generation defects found in the grid audit:

* prism -> tetrahedron splitting must be **conformal** (every interior face is
  shared by exactly two cells).  A blindly-applied fixed template produces
  hanging faces that the finite-volume face extractor then mistakes for
  boundary faces, silently turning interior regions into walls.
* generated tetrahedra must have **positive signed volume**, so that inverted
  cells can be detected instead of being hidden behind ``abs()``.
"""

import numpy as np
import pytest

from autoflowcfd.grid.mesh_gen.tetgen.mesh_prism_to_tet import (
    convert_layers_to_tetrahedra, orient_tetrahedra,
)


def _flat_patch(nx=3, ny=3, lx=1.0, ly=1.0):
    """Triangulated flat surface patch in the z=0 plane."""
    xs = np.linspace(0.0, lx, nx + 1)
    ys = np.linspace(0.0, ly, ny + 1)
    nodes, idx = [], {}
    for j in range(ny + 1):
        for i in range(nx + 1):
            idx[(i, j)] = len(nodes)
            nodes.append((xs[i], ys[j], 0.0))
    nodes = np.array(nodes, dtype=np.float64)

    faces = []
    for j in range(ny):
        for i in range(nx):
            a = idx[(i, j)]
            b = idx[(i + 1, j)]
            c = idx[(i + 1, j + 1)]
            d = idx[(i, j + 1)]
            # Two triangles per quad, deliberately with mixed winding so the
            # test does not depend on a tidy input ordering.
            faces.append([a, b, c])
            faces.append([c, d, a])
    return nodes, np.array(faces, dtype=np.int64)


def _stack_layers(base_nodes, n_layers=4, dz=0.1):
    """Stack the patch into n_layers, returning all_nodes."""
    layers = [base_nodes + np.array([0.0, 0.0, k * dz]) for k in range(n_layers)]
    return np.vstack(layers)


def _face_occurrence_counts(tets):
    """Count how many cells each triangular face belongs to."""
    templates = np.array([[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]], dtype=np.int64)
    faces = tets[:, templates].reshape(-1, 3)
    faces = np.sort(faces, axis=1)
    _, counts = np.unique(faces, axis=0, return_counts=True)
    return counts


class TestPrismSplitConformality:
    # NOTE: layer_conn below always has n_layers - 1 entries (one per
    # extrusion STEP), not n_layers (one per node-layer) - see
    # convert_layers_to_tetrahedra's own layer_connectivity docstring.
    # Passing n_layers entries here used to silently miscompute
    # nodes_per_layer internally (off by one), which conformality/closed-
    # surface checks don't happen to notice but the volume checks below do.
    def test_every_face_shared_by_at_most_two_cells(self):
        """The definitive conformality check.

        In a valid volume mesh a face is either on the boundary (1 cell) or
        interior (2 cells).  A count of 3+ means neighbouring prisms disagreed
        on a diagonal, i.e. the mesh is non-conformal.
        """
        base_nodes, base_faces = _flat_patch()
        n_layers = 4
        all_nodes = _stack_layers(base_nodes, n_layers)
        layer_conn = [base_faces.copy() for _ in range(n_layers - 1)]

        tets, _face_of_tet = convert_layers_to_tetrahedra(all_nodes, layer_conn, base_faces)
        counts = _face_occurrence_counts(tets)

        assert counts.max() <= 2, (
            f"non-conformal mesh: {int(np.count_nonzero(counts > 2))} faces are "
            f"shared by more than 2 cells (max={counts.max()})"
        )

    def test_boundary_faces_form_closed_surface(self):
        """Faces owned by exactly one cell must enclose the volume.

        For a closed surface the sum of outward area vectors vanishes.  If the
        split were non-conformal, spurious 'boundary' faces would appear in the
        interior and this sum would not cancel.
        """
        base_nodes, base_faces = _flat_patch()
        n_layers = 4
        all_nodes = _stack_layers(base_nodes, n_layers)
        layer_conn = [base_faces.copy() for _ in range(n_layers - 1)]
        tets, _face_of_tet = convert_layers_to_tetrahedra(all_nodes, layer_conn, base_faces)

        templates = np.array([[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]],
                             dtype=np.int64)
        faces = tets[:, templates].reshape(-1, 3)
        keys = np.sort(faces, axis=1)
        uniq, inverse, counts = np.unique(keys, axis=0, return_inverse=True,
                                          return_counts=True)
        boundary_positions = np.where(counts[inverse] == 1)[0]
        bf = faces[boundary_positions]

        p0, p1, p2 = all_nodes[bf[:, 0]], all_nodes[bf[:, 1]], all_nodes[bf[:, 2]]
        raw = np.cross(p1 - p0, p2 - p0)          # 2 * area * normal
        areas = 0.5 * np.linalg.norm(raw, axis=1)
        unit = raw / np.maximum(2.0 * areas, 1e-30)[:, None]

        # Orient each boundary face outward w.r.t. its owning cell.
        owner = np.repeat(np.arange(len(tets)), 4)[boundary_positions]
        centroids = all_nodes[tets].mean(axis=1)
        fc = (p0 + p1 + p2) / 3.0
        flip = np.einsum('ij,ij->i', unit, fc - centroids[owner]) < 0
        unit[flip] *= -1.0

        total = (unit * areas[:, None]).sum(axis=0)
        scale = areas.sum()
        assert np.linalg.norm(total) / scale < 1e-10, (
            f"boundary faces do not form a closed surface: residual "
            f"{np.linalg.norm(total)/scale:.3e} (mesh is leaking)"
        )

    def test_all_volumes_positive(self):
        """Signed volumes must be positive after orientation repair."""
        base_nodes, base_faces = _flat_patch()
        n_layers = 3
        all_nodes = _stack_layers(base_nodes, n_layers)
        layer_conn = [base_faces.copy() for _ in range(n_layers - 1)]
        tets, _face_of_tet = convert_layers_to_tetrahedra(all_nodes, layer_conn, base_faces)

        p0, p1, p2, p3 = (all_nodes[tets[:, i]] for i in range(4))
        signed = np.einsum('ij,ij->i', p1 - p0,
                           np.cross(p2 - p0, p3 - p0)) / 6.0
        assert np.all(signed > 0), (
            f"{int(np.count_nonzero(signed <= 0))} tets have non-positive "
            f"signed volume"
        )

    def test_volume_sums_to_slab_volume(self):
        """Total mesh volume must equal the analytic extruded slab volume."""
        base_nodes, base_faces = _flat_patch(nx=3, ny=3, lx=1.0, ly=1.0)
        n_layers, dz = 4, 0.1
        all_nodes = _stack_layers(base_nodes, n_layers, dz)
        layer_conn = [base_faces.copy() for _ in range(n_layers - 1)]
        tets, _face_of_tet = convert_layers_to_tetrahedra(all_nodes, layer_conn, base_faces)

        p0, p1, p2, p3 = (all_nodes[tets[:, i]] for i in range(4))
        vol = np.abs(np.einsum('ij,ij->i', p1 - p0,
                               np.cross(p2 - p0, p3 - p0))) / 6.0
        expected = 1.0 * 1.0 * dz * (n_layers - 1)
        assert vol.sum() == pytest.approx(expected, rel=1e-12)


class TestOrientTetrahedra:
    def test_flips_inverted_cell(self):
        nodes = np.array([[0.0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]])
        # Deliberately inverted ordering (negative signed volume).
        tets = np.array([[0, 1, 3, 2]], dtype=np.int64)
        p0, p1, p2, p3 = (nodes[tets[:, i]] for i in range(4))
        before = np.einsum('ij,ij->i', p1 - p0, np.cross(p2 - p0, p3 - p0))
        assert before[0] < 0

        fixed = orient_tetrahedra(nodes, tets.copy())
        q0, q1, q2, q3 = (nodes[fixed[:, i]] for i in range(4))
        after = np.einsum('ij,ij->i', q1 - q0, np.cross(q2 - q0, q3 - q0))
        assert after[0] > 0
        assert set(fixed[0]) == {0, 1, 2, 3}, "vertex set must be preserved"


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
