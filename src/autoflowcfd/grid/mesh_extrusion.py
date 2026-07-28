"""Mesh extrusion module for boundary layer generation.

Implements surface extrusion along normals to create layered meshes
suitable for boundary layer resolution in CFD simulations.
"""

import numpy as np
from typing import Dict, List, Tuple
from loguru import logger

from .mesh_utils import compute_face_normals, check_reached_boundary


def extrude_layers(
    surface_nodes: np.ndarray,
    surface_faces: np.ndarray,
    normals: np.ndarray,
    bounding_box: Dict[str, np.ndarray],
    growth_rate: float = 1.2,
    max_layers: int = 30,
    min_cell_size: float = 0.001
) -> Tuple[np.ndarray, List[np.ndarray]]:
    """Extrude surface along normals to create layered mesh with boundary layer resolution.
    
    Strategy (Two-stage extrusion):
    Stage 1 - Boundary Layer (Layers 1-8):
      - Fine resolution for y+ control
      - Growth rate: 1.2
      - Target thickness: ~0.05-0.1m
    
    Stage 2 - Transition/Far-field (Layers 9-20):
      - Coarse resolution for domain filling
      - Growth rate: 1.5-2.0
      - Extend to far-field boundary
    
    Args:
        surface_nodes: Base surface nodes, shape=(n_nodes, 3)
        surface_faces: Surface connectivity, shape=(n_faces, 3)
        normals: Face normals for extrusion direction, shape=(n_faces, 3)
        bounding_box: Domain limits to prevent overshoot
        growth_rate: Geometric growth rate for layer thickness
        max_layers: Maximum number of layers to generate
        min_cell_size: Minimum allowable cell size in meters
        
    Returns:
        all_nodes: Concatenated nodes from all layers, shape=(total_nodes, 3)
        layer_connectivity: List of face indices per layer
    """
    # Calculate characteristic length and initial thickness
    domain_size = np.linalg.norm(
        bounding_box['max'] - bounding_box['min']
    )
    
    # For automotive CFD (Re ~ 1e6 - 1e7), first layer height should target y+ ~ 1-30
    # Using empirical formula: delta_y1 ≈ L * Re^(-0.5) / 100
    # Conservative estimate: 0.002 * L_char for first layer
    base_thickness = domain_size * 0.002  # 0.2% of domain size
    
    # Ensure minimum thickness is reasonable (1mm for automotive scale)
    base_thickness = max(base_thickness, min_cell_size)
    
    # Calculate optimal BL parameters for reduced layers
    bl_layers = min(8, max_layers)  # Use at most 8 layers for BL
    transition_layers = min(4, max_layers - bl_layers)  # Remaining layers for transition
    bl_target_thickness = domain_size * 0.02  # 2% of domain size for BL region
    
    logger.info(
        f"Two-stage extrusion strategy (optimized):\n"
        f"  Stage 1 (BL): {bl_layers} layers, growth_rate=1.2\n"
        f"  Stage 2 (Transition): {transition_layers} layers, growth_rate=1.5\n"
        f"  Initial thickness: {base_thickness:.6f}m\n"
        f"  Expected total cells: ~{len(surface_faces) * 3 * (bl_layers + transition_layers):,}"
    )
    
    all_nodes = [surface_nodes.copy()]
    layer_connectivity = [surface_faces.copy()]
    
    current_nodes = surface_nodes.copy()
    current_thickness = base_thickness
    current_growth_rate = 1.2  # Start with BL growth rate
    
    n_layers_generated = 0
    cumulative_height = 0.0
    
    for layer_idx in range(max_layers):
        # Check if we've reached domain boundary
        if check_reached_boundary(current_nodes, bounding_box):
            logger.info(
                f"Reached domain boundary at layer {layer_idx + 1}, "
                f"stopping extrusion (generated {n_layers_generated} layers)"
            )
            break
        
        # Switch to Stage 2 (Transition) after boundary layer
        if n_layers_generated == bl_layers and current_growth_rate < 1.5:
            current_growth_rate = 1.5
            logger.info(
                f"Switching to Stage 2 (transition) at layer {layer_idx + 1}, "
                f"growth_rate increased to {current_growth_rate}"
            )
        
        # Extrude nodes along averaged normals
        new_nodes = extrude_single_layer(
            current_nodes, surface_faces, normals, current_thickness
        )
        
        # Check minimum cell size
        if current_thickness < min_cell_size:
            logger.warning(
                f"Layer thickness {current_thickness:.6f} below minimum ({min_cell_size}), "
                f"stopping at layer {layer_idx + 1}"
            )
            break
        
        all_nodes.append(new_nodes)
        layer_connectivity.append(surface_faces.copy())
        n_layers_generated += 1
        cumulative_height += current_thickness
        
        # Update for next layer
        current_nodes = new_nodes
        current_thickness *= current_growth_rate
        
        # Log progress
        if (layer_idx + 1) % 5 == 0 or n_layers_generated <= 3:
            logger.info(
                f"  Layer {layer_idx + 1}: thickness={current_thickness:.6f}m, "
                f"cumulative_height={cumulative_height:.6f}m, "
                f"growth_rate={current_growth_rate}"
            )
        
        # Stop if we've filled enough of the domain (40% of domain size)
        if cumulative_height > domain_size * 0.4:
            logger.info(
                f"Cumulative height {cumulative_height:.4f}m exceeds 40% of domain, "
                f"stopping at layer {layer_idx + 1}"
            )
            break
    
    # Concatenate all layers
    all_nodes_array = np.vstack(all_nodes)
    
    logger.info(
        f"Extrusion completed: {n_layers_generated} layers generated, "
        f"total nodes: {len(all_nodes_array)}, "
        f"final cumulative height: {cumulative_height:.4f}m"
    )
    
    return all_nodes_array, layer_connectivity


def extrude_single_layer(
    nodes: np.ndarray,
    faces: np.ndarray,
    normals: np.ndarray,
    thickness: float
) -> np.ndarray:
    """Extrude one layer of nodes.
    
    For each node, average the normals of adjacent faces and move
    along that direction. Vectorized implementation for performance.
    
    Args:
        nodes: Current layer nodes, shape=(n_nodes, 3)
        faces: Face connectivity, shape=(n_faces, 3)
        normals: Face normals, shape=(n_faces, 3)
        thickness: Extrusion distance in meters
        
    Returns:
        New node positions after extrusion, shape=(n_nodes, 3)
    """
    n_nodes = len(nodes)
    new_nodes = nodes.copy()
    
    # Build node-to-face mapping using vectorized operations
    logger.info("Building node-normal mapping (vectorized)...")
    node_normal_sum = np.zeros((n_nodes, 3))
    node_normal_count = np.zeros(n_nodes, dtype=int)
    
    # Vectorized accumulation - much faster than nested loops
    for face_idx in range(len(faces)):
        node_indices = faces[face_idx]
        node_normal_sum[node_indices] += normals[face_idx]
        node_normal_count[node_indices] += 1
    
    # Compute average normals (avoid division by zero)
    logger.info("Computing averaged normals...")
    mask = node_normal_count > 0
    avg_normals = np.zeros_like(node_normal_sum)
    avg_normals[mask] = node_normal_sum[mask] / node_normal_count[mask, np.newaxis]
    
    # Normalize
    norms = np.linalg.norm(avg_normals, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-10)
    avg_normals = avg_normals / norms
    
    # Extrude all nodes at once
    logger.info(f"Extruding layer with thickness={thickness:.6f}...")
    new_nodes[mask] += thickness * avg_normals[mask]
    
    return new_nodes


def convert_layers_to_tetrahedra(
    all_nodes: np.ndarray,
    layer_connectivity: List[np.ndarray],
    base_faces: np.ndarray
) -> np.ndarray:
    """Convert layered prism mesh to tetrahedral mesh.
    
    Each prism (formed between two consecutive layers) is split
    into 3 tetrahedra. Vectorized implementation for performance.
    
    Args:
        all_nodes: All nodes from all layers, shape=(total_nodes, 3)
        layer_connectivity: Face indices per layer
        base_faces: Original surface faces, shape=(n_faces, 3)
        
    Returns:
        Tetrahedral cell connectivity, shape=(n_tets, 4)
    """
    n_layers = len(layer_connectivity)
    n_base_faces = len(base_faces)
    
    if n_layers < 2:
        raise ValueError("Need at least 2 layers to create volume")
    
    # Calculate nodes per layer correctly
    n_total_nodes = len(all_nodes)
    nodes_per_layer = n_total_nodes // n_layers
    
    logger.info(
        f"Converting {n_layers-1} layer pairs to tetrahedra..."
    )
    
    # Pre-allocate array for all tetrahedra
    # Each face × each layer pair = 3 tetrahedra
    n_tets = n_base_faces * (n_layers - 1) * 3
    tetrahedra = np.zeros((n_tets, 4), dtype=np.int64)
    
    # Vectorized generation of all tetrahedra
    tet_idx = 0
    for layer_idx in range(n_layers - 1):
        offset_current = layer_idx * nodes_per_layer
        offset_next = (layer_idx + 1) * nodes_per_layer
        
        # Compute node indices for all faces at once
        n0 = offset_current + base_faces[:, 0]  # shape=(n_faces,)
        n1 = offset_current + base_faces[:, 1]
        n2 = offset_current + base_faces[:, 2]
        n3 = offset_next + base_faces[:, 0]
        n4 = offset_next + base_faces[:, 1]
        n5 = offset_next + base_faces[:, 2]
        
        # Generate 3 tetrahedra per face
        # Tet 1: [n0, n1, n2, n4]
        tetrahedra[tet_idx:tet_idx+n_base_faces, 0] = n0
        tetrahedra[tet_idx:tet_idx+n_base_faces, 1] = n1
        tetrahedra[tet_idx:tet_idx+n_base_faces, 2] = n2
        tetrahedra[tet_idx:tet_idx+n_base_faces, 3] = n4
        tet_idx += n_base_faces
        
        # Tet 2: [n0, n2, n4, n5]
        tetrahedra[tet_idx:tet_idx+n_base_faces, 0] = n0
        tetrahedra[tet_idx:tet_idx+n_base_faces, 1] = n2
        tetrahedra[tet_idx:tet_idx+n_base_faces, 2] = n4
        tetrahedra[tet_idx:tet_idx+n_base_faces, 3] = n5
        tet_idx += n_base_faces
        
        # Tet 3: [n0, n4, n5, n3]
        tetrahedra[tet_idx:tet_idx+n_base_faces, 0] = n0
        tetrahedra[tet_idx:tet_idx+n_base_faces, 1] = n4
        tetrahedra[tet_idx:tet_idx+n_base_faces, 2] = n5
        tetrahedra[tet_idx:tet_idx+n_base_faces, 3] = n3
        tet_idx += n_base_faces
        
        logger.info(f"  Layer {layer_idx+1}/{n_layers-1}: Generated {n_base_faces*3} tets")
    
    logger.info(f"Total tetrahedra generated: {len(tetrahedra)}")
    return tetrahedra
