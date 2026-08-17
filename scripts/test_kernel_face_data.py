"""端到端验证：_KernelFaceData 快速路径正确性和性能。"""
import pickle
import time
import numpy as np
import sys
sys.path.insert(0, r"d:\myWorkspace\AutoFlowCFD\src")

from loguru import logger

# Load test mesh data
with open(r"d:\myWorkspace\AutoFlowCFD\results\cube_volume.pkl", "rb") as f:
    volume_data = pickle.load(f)
print(f"VolumeMeshData: {volume_data.cell_count} cells, {volume_data.node_count} nodes")

# Create HighOrderMesh and load
from autoflowcfd.grid.high_order.high_order_mesh import HighOrderMesh
mesh = HighOrderMesh()
t_load0 = time.perf_counter()
mesh.load_from_volume_mesh(volume_data, build_faces=True)
t_load1 = time.perf_counter()
print(f"load_from_volume_mesh: {t_load1-t_load0:.1f}s")
print(f"Mesh: n_cells={mesh.n_cells}, n_faces={mesh.face_connectivity.n_faces}, n_prism={mesh.n_prism_cells}, n1d={mesh.n_points_1d}")

fc = mesh.face_connectivity

# ---- Phase 1: build_face_flux_points ----
from autoflowcfd.fr.face_flux_points_merge import build_face_flux_points, _KernelFaceData

t0 = time.perf_counter()
result = build_face_flux_points(fc, mesh)
t1 = time.perf_counter()
print(f"\n=== build_face_flux_points: {t1-t0:.1f}s ===")
print(f"Result type: {type(result).__name__}")
assert isinstance(result, _KernelFaceData), "Should return _KernelFaceData"
print(f"n_faces: {result.n_faces}")

# Check flat arrays
n_faces = result.n_faces
nb_single = np.sum(result.nb_src0_cell >= 0)
ow_single = np.sum(result.ow_src0_cell >= 0)
nb_multi = np.sum(result.nb_src1_idx >= 0)
ow_multi = np.sum(result.ow_src1_idx >= 0)
print(f"\n--- Source array stats ---")
print(f"nb_src0_cell: {nb_single} single-source, {n_faces - nb_single} no-source")
print(f"ow_src0_cell: {ow_single} single-source, {n_faces - ow_single} no-source")
print(f"nb_src1_idx: {nb_multi} multi-source (2nd source)")
print(f"ow_src1_idx: {ow_multi} multi-source (2nd source)")
print(f"nb_extra_cell: {len(result.nb_extra_cell)} entries")
print(f"ow_extra_cell: {len(result.ow_extra_cell)} entries")

# ---- Phase 2: on-demand FaceFluxPointGeometry ----
bnd_faces = np.where(fc.is_boundary)[0]
int_faces = np.where(~fc.is_boundary)[0]
print(f"\n--- On-demand FaceFluxPointGeometry ---")
print(f"Boundary faces: {len(bnd_faces)}")

# Boundary face
ffp_bnd = result[int(bnd_faces[0])]
print(f"Boundary ffp: type={type(ffp_bnd).__name__}, owner_axis={ffp_bnd.owner_axis}, neighbor_axis={ffp_bnd.neighbor_axis}")
assert ffp_bnd.neighbor_axis == -1, "Boundary face should have neighbor_axis=-1"

# Interior face
ffp_int = result[int(int_faces[0])]
print(f"Interior ffp: owner_axis={ffp_int.owner_axis}, nb_sources={len(ffp_int.neighbor_sources)}, ow_sources={len(ffp_int.owner_sources)}")

# Check that sources match flat arrays
f_test = int(int_faces[0])
ffp_test = result[f_test]
nb_c0 = int(result.nb_src0_cell[f_test])
if nb_c0 >= 0:
    assert len(ffp_test.neighbor_sources) >= 1
    assert ffp_test.neighbor_sources[0][0] == nb_c0
    assert np.allclose(ffp_test.neighbor_sources[0][1], result.nb_src0_mat[f_test])
    print(f"  nb_src0 match: OK (cell={nb_c0})")
if result.nb_src1_idx[f_test] >= 0:
    idx1 = int(result.nb_src1_idx[f_test])
    assert len(ffp_test.neighbor_sources) >= 2
    assert ffp_test.neighbor_sources[1][0] == int(result.nb_extra_cell[idx1])
    print(f"  nb_src1 match: OK (cell={int(result.nb_extra_cell[idx1])})")

# ---- Phase 3: build_flat_face_geometry fast path ----
print(f"\n--- build_flat_face_geometry (fast path) ---")
mesh.face_flux_points = result  # assign _KernelFaceData

from autoflowcfd.core.fr_operators.face_kernels import build_flat_face_geometry, FlatFaceGeometry
from autoflowcfd.fr.operators import generate_fr_operators

t2 = time.perf_counter()
ops = generate_fr_operators(mesh.n_points_1d - 1)
t3 = time.perf_counter()
print(f"generate_fr_operators: {t3-t2:.1f}s")

t3 = time.perf_counter()
flat = build_flat_face_geometry(mesh, ops)
t4 = time.perf_counter()
print(f"build_flat_face_geometry: {t4-t3:.1f}s")
print(f"Flat type: {type(flat).__name__}")
assert isinstance(flat, FlatFaceGeometry)

# Verify flat arrays match kernel output
print(f"\n--- Verifying flat array consistency ---")
assert flat.n_faces == n_faces
assert np.array_equal(flat.neighbor_src0_cell, result.nb_src0_cell)
assert np.array_equal(flat.neighbor_src0_mat, result.nb_src0_mat)
assert np.array_equal(flat.neighbor_src1_idx, result.nb_src1_idx)
assert np.array_equal(flat.owner_src0_cell, result.ow_src0_cell)
assert np.array_equal(flat.owner_src0_mat, result.ow_src0_mat)
assert np.array_equal(flat.owner_src1_idx, result.ow_src1_idx)
print("All flat array consistency checks PASSED!")

# Verify multi-source faces have correct src1_idx
if nb_multi > 0:
    multi_faces = np.where(result.nb_src1_idx >= 0)[0]
    for f in multi_faces[:5]:
        idx1 = int(result.nb_src1_idx[f])
        assert idx1 >= 0
        assert idx1 < len(result.nb_extra_cell)
        assert flat.neighbor_src1_cell[idx1] == result.nb_extra_cell[idx1]
    print(f"Multi-source src1_idx spot check: PASSED ({nb_multi} faces)")

print(f"\n=== TOTAL: {time.perf_counter()-t0:.1f}s ===")
print("ALL TESTS PASSED!")
