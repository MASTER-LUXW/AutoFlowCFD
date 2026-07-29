"""Cell-centred gradient reconstruction for the finite-volume solver.

Provides Green-Gauss gradients (with boundary-face contributions) and the
Barth-Jespersen slope limiter.  These are the building blocks for second-order
MUSCL reconstruction and for the viscous stress / turbulent diffusion terms.

All routines are vectorised NumPy and operate on *oriented* face geometry, i.e.
``face_normals`` point from the owner cell (connectivity column 0) to the
neighbour (column 1) for internal faces and outward for boundary faces, as
produced by :class:`autoflowcfd.core.fvm_faces.FVMFaceExtractor`.
"""

from __future__ import annotations

import numpy as np


class FaceGeometry:
    """Immutable bundle of oriented face/cell geometry shared by the solver."""

    __slots__ = (
        "connectivity", "normals", "areas", "centers",
        "boundary_flags", "cell_centroids", "cell_volumes",
        "owner", "neigh", "internal_mask", "boundary_mask",
        "int_owner", "int_neigh", "bnd_owner",
    )

    def __init__(self, connectivity, normals, areas, centers,
                 boundary_flags, cell_centroids, cell_volumes):
        self.connectivity = np.ascontiguousarray(connectivity, dtype=np.int64)
        self.normals = np.ascontiguousarray(normals, dtype=np.float64)
        self.areas = np.ascontiguousarray(areas, dtype=np.float64)
        self.centers = np.ascontiguousarray(centers, dtype=np.float64)
        self.boundary_flags = np.ascontiguousarray(boundary_flags, dtype=np.int32)
        self.cell_centroids = np.ascontiguousarray(cell_centroids, dtype=np.float64)
        self.cell_volumes = np.ascontiguousarray(cell_volumes, dtype=np.float64)

        self.owner = self.connectivity[:, 0]
        self.neigh = self.connectivity[:, 1]
        self.boundary_mask = self.boundary_flags.astype(bool)
        self.internal_mask = ~self.boundary_mask

        self.int_owner = self.owner[self.internal_mask]
        self.int_neigh = self.neigh[self.internal_mask]
        self.bnd_owner = self.owner[self.boundary_mask]

    @property
    def n_cells(self) -> int:
        return len(self.cell_volumes)

    @property
    def n_faces(self) -> int:
        return len(self.areas)


def green_gauss_gradient(cell_values: np.ndarray,
                         geom: FaceGeometry,
                         boundary_face_values: np.ndarray | None = None) -> np.ndarray:
    """Green-Gauss cell gradients: grad(phi)_i = (1/V_i) sum_f phi_f n_f A_f.

    Args:
        cell_values: shape (n_cells, n_vars).
        geom: face geometry (oriented owner->neighbour normals).
        boundary_face_values: optional (n_boundary_faces, n_vars) values to use
            at boundary faces (e.g. ghost/BC states).  If ``None`` the owner
            cell value is used (zero-gradient extrapolation).

    Returns:
        Gradients, shape (n_cells, n_vars, 3).
    """
    cell_values = np.ascontiguousarray(cell_values, dtype=np.float64)
    n_cells, n_vars = cell_values.shape
    grad = np.zeros((n_cells, n_vars, 3), dtype=np.float64)

    # --- internal faces: face value = arithmetic mean of the two cells ---
    io, ineigh = geom.int_owner, geom.int_neigh
    phi_f = 0.5 * (cell_values[io] + cell_values[ineigh])          # (nif, nv)
    aN = geom.areas[geom.internal_mask][:, None] * geom.normals[geom.internal_mask]  # (nif,3)

    # contribution = phi_f (nv) outer aN (3)  -> (nif, nv, 3)
    contrib = phi_f[:, :, None] * aN[:, None, :]
    np.add.at(grad, io, contrib)
    np.add.at(grad, ineigh, -contrib)

    # --- boundary faces: outward normal, use BC value if provided ---
    bo = geom.bnd_owner
    if bo.size:
        if boundary_face_values is None:
            phi_b = cell_values[bo]
        else:
            phi_b = np.ascontiguousarray(boundary_face_values, dtype=np.float64)
        aB = geom.areas[geom.boundary_mask][:, None] * geom.normals[geom.boundary_mask]
        np.add.at(grad, bo, phi_b[:, :, None] * aB[:, None, :])

    grad /= geom.cell_volumes[:, None, None]
    return grad


def barth_jespersen_limiter(cell_values: np.ndarray,
                            grad: np.ndarray,
                            geom: FaceGeometry) -> np.ndarray:
    """Barth-Jespersen slope limiter phi_i in [0, 1] per cell and variable.

    Guarantees the reconstructed face values stay within the min/max of the
    cell and its face-neighbours (monotonicity / no new extrema).

    Args:
        cell_values: (n_cells, n_vars)
        grad: (n_cells, n_vars, 3) unlimited gradients
        geom: face geometry

    Returns:
        Limiter phi, shape (n_cells, n_vars), in [0, 1].
    """
    cell_values = np.ascontiguousarray(cell_values, dtype=np.float64)
    n_cells, n_vars = cell_values.shape

    # Neighbour min/max (self included) over internal-face stencil.
    u_max = cell_values.copy()
    u_min = cell_values.copy()
    io, ineigh = geom.int_owner, geom.int_neigh
    np.maximum.at(u_max, io, cell_values[ineigh])
    np.minimum.at(u_min, io, cell_values[ineigh])
    np.maximum.at(u_max, ineigh, cell_values[io])
    np.minimum.at(u_min, ineigh, cell_values[io])

    phi = np.ones((n_cells, n_vars), dtype=np.float64)
    eps = 1e-12

    def _accumulate(cells, face_sel):
        # Reconstruct at face centroid: delta = grad_i . (x_f - x_i)
        r = geom.centers[face_sel] - geom.cell_centroids[cells]   # (nf, 3)
        delta = np.einsum('nvd,nd->nv', grad[cells], r)           # (nf, nv)
        umax = u_max[cells] - cell_values[cells]
        umin = u_min[cells] - cell_values[cells]

        phi_f = np.ones_like(delta)
        pos = delta > eps
        neg = delta < -eps
        phi_f[pos] = np.minimum(1.0, umax[pos] / delta[pos])
        phi_f[neg] = np.minimum(1.0, umin[neg] / delta[neg])
        phi_f = np.clip(phi_f, 0.0, 1.0)
        np.minimum.at(phi, cells, phi_f)

    # Every face constrains its owner; internal faces also constrain neighbour.
    _accumulate(geom.owner, np.arange(geom.n_faces))
    _accumulate(geom.int_neigh, np.where(geom.internal_mask)[0])
    return phi
