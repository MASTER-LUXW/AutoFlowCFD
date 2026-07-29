"""FVM face extraction from tetrahedral mesh.

Extracts face connectivity and geometry from unstructured tetrahedral mesh
using integer hashing for efficient duplicate detection.

Geometry is *oriented*: every face normal is guaranteed to point from the
``owner`` cell (column 0 of the connectivity) towards the ``neighbour`` cell
(column 1) for internal faces, and out of the domain for boundary faces.  This
orientation is what makes the residual accumulation convention
(``owner -= flux``, ``neighbour += flux``) discretely conservative and gives
the aerodynamic-force integration the correct sign.
"""

import time
import numpy as np
from typing import Dict
from loguru import logger


class FVMFaceExtractor:
    """Extracts oriented face connectivity and geometry from a tet mesh."""

    def __init__(self):
        self.face_connectivity = None
        self.face_areas = None
        self.face_centers = None
        self.face_normals = None
        self.boundary_flags = None
        # Cell centroids are needed both for normal orientation and for
        # gradient reconstruction (Green-Gauss / least-squares).
        self.cell_centroids = None

    def build_from_tetrahedra(self, connectivity: np.ndarray, nodes: np.ndarray) -> Dict[str, np.ndarray]:
        """Build oriented face data structure using integer hashing.

        Args:
            connectivity: Cell connectivity, shape=(n_cells, 4)
            nodes: Node coordinates, shape=(n_nodes, 3)

        Returns:
            Dictionary with face data arrays (see :meth:`get_face_data`).
        """
        connectivity = np.ascontiguousarray(connectivity, dtype=np.int64)
        nodes = np.ascontiguousarray(nodes, dtype=np.float64)

        n_cells = len(connectivity)
        logger.info(f"Building face connectivity from {n_cells} tetrahedra...")
        t_start = time.perf_counter()

        # Cell centroids (mean of the four vertices) -- used for orientation.
        self.cell_centroids = nodes[connectivity].mean(axis=1)

        # Faces of a tetrahedron.  The local node ordering is chosen so that the
        # cross product (p1-p0) x (p2-p0) points *outward* for a positively
        # oriented tet, but we do not rely on that -- orientation is enforced
        # explicitly below so the mesh may have arbitrary winding.
        faces_per_cell = np.array([
            [0, 1, 2],
            [0, 3, 1],
            [0, 2, 3],
            [1, 3, 2],
        ], dtype=np.int64)

        # All faces: shape=(n_cells*4, 3)
        all_faces = connectivity[:, faces_per_cell].reshape(-1, 3)
        n_total_faces = len(all_faces)

        # Sorted node triplets give an orientation-independent key.
        sorted_faces = np.sort(all_faces, axis=1)

        logger.info("Computing face hashes...")
        max_node_id = np.max(connectivity) + 1
        M = np.uint64(max_node_id)
        face_hashes = (
            sorted_faces[:, 0].astype(np.uint64)
            + sorted_faces[:, 1].astype(np.uint64) * M
            + sorted_faces[:, 2].astype(np.uint64) * M * M
        )

        logger.info("Finding unique faces (np.unique on hash array, single-threaded)...")
        t_unique_start = time.perf_counter()
        unique_hashes, inverse_indices, counts = np.unique(
            face_hashes, return_inverse=True, return_counts=True
        )
        n_faces = len(unique_hashes)
        logger.info(
            f"Extracted {n_faces} unique faces from {n_total_faces} total faces "
            f"({time.perf_counter() - t_unique_start:.2f}s)"
        )

        # Output arrays.
        self.face_connectivity = np.full((n_faces, 2), -1, dtype=np.int64)
        self.face_areas = np.zeros(n_faces, dtype=np.float64)
        self.face_centers = np.zeros((n_faces, 3), dtype=np.float64)
        self.face_normals = np.zeros((n_faces, 3), dtype=np.float64)
        self.boundary_flags = np.zeros(n_faces, dtype=np.int32)

        cell_indices = np.repeat(np.arange(n_cells), 4)

        # ------------------------------------------------------------------
        # Boundary faces (appear exactly once).
        # ------------------------------------------------------------------
        t_boundary_start = time.perf_counter()
        boundary_mask = counts[inverse_indices] == 1
        if np.any(boundary_mask):
            logger.info(f"Processing {int(np.sum(boundary_mask))} boundary faces...")
            boundary_positions = np.where(boundary_mask)[0]
            boundary_face_idx = inverse_indices[boundary_positions]

            unique_bf, bf_first_pos = np.unique(boundary_face_idx, return_index=True)
            bf_original_pos = boundary_positions[bf_first_pos]

            owner = cell_indices[bf_original_pos]
            self.face_connectivity[unique_bf, 0] = owner
            self.face_connectivity[unique_bf, 1] = -1
            self.boundary_flags[unique_bf] = 1

            bf_nodes = all_faces[bf_original_pos]
            centers, normals, areas = self._face_geometry(nodes, bf_nodes)

            # Orient outward: normal must point away from the owner centroid.
            outward = centers - self.cell_centroids[owner]
            flip = np.einsum('ij,ij->i', normals, outward) < 0.0
            normals[flip] *= -1.0

            self.face_centers[unique_bf] = centers
            self.face_normals[unique_bf] = normals
            self.face_areas[unique_bf] = areas
            logger.debug(f"Boundary faces processed ({time.perf_counter() - t_boundary_start:.2f}s)")

        # ------------------------------------------------------------------
        # Internal faces (appear exactly twice).
        # ------------------------------------------------------------------
        t_internal_start = time.perf_counter()
        internal_mask = counts[inverse_indices] == 2
        if np.any(internal_mask):
            logger.info(f"Processing {int(np.sum(internal_mask))} internal faces...")
            internal_positions = np.where(internal_mask)[0]
            internal_face_idx = inverse_indices[internal_positions]
            internal_cell_idx = cell_indices[internal_positions]
            internal_face_nodes = all_faces[internal_positions]

            sort_order = np.argsort(internal_face_idx, kind='stable')
            sorted_face_idx = internal_face_idx[sort_order]
            sorted_cell_idx = internal_cell_idx[sort_order]
            sorted_face_nodes = internal_face_nodes[sort_order]

            n_internal_pairs = len(sorted_face_idx) // 2
            if n_internal_pairs > 0:
                face_indices = sorted_face_idx[::2]
                cell0 = sorted_cell_idx[::2]
                cell1 = sorted_cell_idx[1::2]
                face_nodes = sorted_face_nodes[::2]

                # Deterministic owner/neighbour: owner = min(cell), neighbour = max(cell).
                owner = np.minimum(cell0, cell1)
                neigh = np.maximum(cell0, cell1)
                self.face_connectivity[face_indices, 0] = owner
                self.face_connectivity[face_indices, 1] = neigh
                self.boundary_flags[face_indices] = 0

                centers, normals, areas = self._face_geometry(nodes, face_nodes)

                # Orient owner -> neighbour.
                owner_to_neigh = self.cell_centroids[neigh] - self.cell_centroids[owner]
                flip = np.einsum('ij,ij->i', normals, owner_to_neigh) < 0.0
                normals[flip] *= -1.0

                self.face_centers[face_indices] = centers
                self.face_normals[face_indices] = normals
                self.face_areas[face_indices] = areas
            logger.debug(f"Internal faces processed ({time.perf_counter() - t_internal_start:.2f}s)")

        n_boundary = int(np.sum(self.boundary_flags))
        n_internal = n_faces - n_boundary
        logger.info(
            f"Face mapping: {n_faces} total ({n_internal} internal, {n_boundary} boundary) "
            f"[{time.perf_counter() - t_start:.2f}s total]"
        )

        return self.get_face_data()

    @staticmethod
    def _face_geometry(nodes: np.ndarray, face_nodes: np.ndarray):
        """Return (centers, unit_normals, areas) for a batch of triangles.

        Degenerate faces (near-zero area) get a placeholder unit normal of
        ``[0, 0, 1]`` and zero area so downstream code stays finite.
        """
        p0 = nodes[face_nodes[:, 0]]
        p1 = nodes[face_nodes[:, 1]]
        p2 = nodes[face_nodes[:, 2]]

        centers = (p0 + p1 + p2) / 3.0
        raw = np.cross(p1 - p0, p2 - p0)
        areas = 0.5 * np.linalg.norm(raw, axis=1)

        valid = areas > 1e-15
        # 2*area is the magnitude of the raw cross product.
        denom = np.where(valid, 2.0 * areas, 1.0)[:, np.newaxis]
        normals = raw / denom
        normals[~valid] = np.array([0.0, 0.0, 1.0])
        return centers, normals, areas

    def get_face_data(self) -> Dict[str, np.ndarray]:
        """Get all face data."""
        return {
            'connectivity': self.face_connectivity,
            'areas': self.face_areas,
            'centers': self.face_centers,
            'normals': self.face_normals,
            'boundary_flags': self.boundary_flags,
            'cell_centroids': self.cell_centroids,
        }
