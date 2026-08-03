"""Mesh utility functions for volume mesh generation.

Provides validation, quality checking, and boundary detection utilities.
All functions are stateless and can be used independently.
"""

import numpy as np
from typing import Dict, Optional, TYPE_CHECKING
from loguru import logger

if TYPE_CHECKING:
    from ..structures import VolumeMeshData


def validate_surface_mesh(
    nodes: np.ndarray,
    faces: np.ndarray
) -> None:
    """Validate surface mesh inputs.
    
    Args:
        nodes: Node coordinates, shape=(n_nodes, 3)
        faces: Face connectivity, shape=(n_faces, 3)
        
    Raises:
        ValueError: If mesh is invalid
    """
    if nodes.ndim != 2 or nodes.shape[1] != 3:
        raise ValueError(f"Nodes must be (n, 3), got {nodes.shape}")
    
    if faces.ndim != 2 or faces.shape[1] != 3:
        raise ValueError(f"Faces must be (n, 3), got {faces.shape}")
    
    if faces.max() >= len(nodes):
        raise ValueError(
            f"Face indices out of range: max={faces.max()}, "
            f"n_nodes={len(nodes)}"
        )
    
    logger.info(
        f"Surface mesh validated: {len(nodes)} nodes, {len(faces)} faces"
    )


def validate_bounding_box(
    bounding_box: Dict[str, np.ndarray]
) -> None:
    """Validate bounding box definition.
    
    Args:
        bounding_box: {min: [x,y,z], max: [x,y,z]}
        
    Raises:
        ValueError: If bbox is invalid
    """
    if 'min' not in bounding_box or 'max' not in bounding_box:
        raise ValueError("Bounding box must have 'min' and 'max' keys")
    
    if len(bounding_box['min']) != 3 or len(bounding_box['max']) != 3:
        raise ValueError("Bounding box coordinates must be 3D")
    
    if np.any(bounding_box['max'] <= bounding_box['min']):
        raise ValueError("Bounding box max must be > min in all dimensions")


def compute_face_normals(
    nodes: np.ndarray,
    faces: np.ndarray
) -> np.ndarray:
    """Compute unit normal vectors for surface faces.
    
    Uses cross product of edge vectors (right-hand rule).
    Vectorized implementation for better performance.
    
    Args:
        nodes: Node coordinates, shape=(n_nodes, 3)
        faces: Face connectivity, shape=(n_faces, 3)
        
    Returns:
        Unit normals, shape=(n_faces, 3)
    """
    logger.info("Computing face normals (vectorized)...")
    n_faces = len(faces)
    
    # Vectorized computation - much faster than loop
    v0 = nodes[faces[:, 0]]  # shape=(n_faces, 3)
    v1 = nodes[faces[:, 1]]
    v2 = nodes[faces[:, 2]]
    
    # Edge vectors
    e1 = v1 - v0  # shape=(n_faces, 3)
    e2 = v2 - v0
    
    # Cross product for all faces at once
    normals = np.cross(e1, e2)  # shape=(n_faces, 3)
    
    # Compute norms
    norms = np.linalg.norm(normals, axis=1, keepdims=True)  # shape=(n_faces, 1)
    
    # Avoid division by zero
    norms = np.maximum(norms, 1e-10)
    
    # Normalize all normals at once
    normals = normals / norms
    
    # Check for degenerate faces
    degenerate_count = np.sum(norms.flatten() < 1e-9)
    if degenerate_count > 0:
        logger.warning(f"Found {degenerate_count} degenerate faces with near-zero area")
        # Set default normal for degenerate faces
        normals[norms.flatten() < 1e-9] = [0, 0, 1]
    
    logger.info(f"Computed {n_faces} face normals")
    return normals


def check_reached_boundary(
    nodes: np.ndarray,
    bounding_box: Dict[str, np.ndarray]
) -> bool:
    """Check if extrusion has exceeded domain boundary.
    
    Args:
        nodes: Current layer nodes, shape=(n_nodes, 3)
        bounding_box: Domain bounds {min: [x,y,z], max: [x,y,z]}
        
    Returns:
        True if any node exceeds bounding box
    """
    bbox_min = bounding_box['min']
    bbox_max = bounding_box['max']
    
    # Check if any node is outside the bounding box
    if np.any(nodes < bbox_min - 1e-6) or np.any(nodes > bbox_max + 1e-6):
        return True
    
    return False


def check_mesh_quality(volume_mesh: 'VolumeMeshData') -> None:
    """Check mesh quality metrics.
    
    Args:
        volume_mesh: Generated volume mesh
        
    Raises:
        ValueError: If mesh quality is unacceptable
    """
    from ..structures import TetrahedralCells
    
    # Check for negative volumes (should not happen)
    if np.any(volume_mesh.cells.volumes <= 0):
        n_neg = np.sum(volume_mesh.cells.volumes <= 0)
        raise ValueError(f"Found {n_neg} cells with non-positive volume")
    
    # Check aspect ratio (simplified - just check volume range)
    vol_min = volume_mesh.cells.volumes.min()
    vol_max = volume_mesh.cells.volumes.max()
    vol_ratio = vol_max / vol_min if vol_min > 0 else float('inf')
    
    logger.info(
        f"Mesh quality check: "
        f"volume range [{vol_min:.6e}, {vol_max:.6e}], "
        f"ratio={vol_ratio:.2f}"
    )
    
    # Warn if ratio is too high
    if vol_ratio > 1e6:
        logger.warning(
            f"High volume ratio detected ({vol_ratio:.2e}). "
            f"Consider refining mesh or adjusting growth_rate."
        )
