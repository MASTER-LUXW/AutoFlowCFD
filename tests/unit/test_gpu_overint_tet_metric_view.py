"""GPU 过积分的四面体段细点度量是逐单元视图，不再每次求值物化整段（I-114）。

四面体细点度量按单元常数上传（`upload_fine_metrics_gpu`，plate_demo P3 显存 4.7 GB -> 14 MB）；此前
`get_overintegration_segs_gpu` 每次调用都 `ascontiguousarray(broadcast_to(..., (n_tet, n_fine, 3, 3)))`，
每次残差求值临时分配回整段。消费方都是 `matmul(adj_seg, F)`，长度为 1 的细点轴按批维广播即可。
"""

import numpy as np

from autoflowcfd.core.gpu.gpu_overintegration import OVERINT_OPS_KEYS, get_overintegration_segs_gpu
from autoflowcfd.fr.operators import generate_fr_operators
from tests.unit.test_fr_residual_inviscid import _build_synthetic_mixed_mesh


def test_tet_segment_is_a_per_cell_view_and_matmul_matches_the_materialized_form():
    order = 2
    mesh = _build_synthetic_mixed_mesh(order)
    ops = generate_fr_operators(order)
    ops_data = {k: getattr(ops, k) for k in OVERINT_OPS_KEYS}
    n_prism, n_tet = mesh.n_prism_cells, mesh.n_cells - mesh.n_prism_cells
    n_fine_prism = int(ops_data["overint_D_fine_prism"].shape[0])
    rng = np.random.default_rng(1)
    adj_tet_cell = rng.standard_normal((n_tet, 3, 3))
    segs = get_overintegration_segs_gpu(
        {"adj_j_fine_prism": rng.standard_normal((n_prism, n_fine_prism, 3, 3)), "adj_j_fine_tet": adj_tet_cell},
        ops_data, mesh.n_cells, n_prism)
    _lo, _hi, n_fine_tet, adj_seg, *_ = segs[1]
    assert adj_seg.shape == (n_tet, 1, 3, 3) and np.shares_memory(adj_seg, adj_tet_cell)
    F = rng.standard_normal((n_tet, n_fine_tet, 3, 5))
    materialized = np.broadcast_to(adj_tet_cell[:, None], (n_tet, n_fine_tet, 3, 3))
    np.testing.assert_array_equal(np.matmul(adj_seg, F), np.matmul(materialized, F))
