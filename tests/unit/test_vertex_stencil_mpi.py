"""分布式顶点邻域模板（`core/mpi/vertex_stencil_mpi.py`）的多 rank 判据。

本机没有 MPI，集合通信用线程模拟（`tests/unit/_thread_comm.py`，注入
`allgather/allreduce_max/allreduce_min`），每个线程就是一个 rank，同时执行同一段
代码 —— 与真实 MPI 的集合语义相同（全部 rank 进入、全部 rank 得到同一结果）。

核心判据：各 rank 只用自己的 local 单元建模板、在共享顶点上做 MAX/MIN 归约后，
local 单元的 BJ 越界比与单机全局结果**逐位相同**（max/min 与顺序无关），与分区
数无关。反例：不做共享顶点归约时结果不同 —— 证明"跨面跳的顶点邻居"确实存在、
测试有区分力。
"""

import numpy as np
import pytest

from autoflowcfd.core.fr_operators.bounds_sensor import compute_bounds_violation_ratio
from autoflowcfd.core.fr_operators.vertex_stencil import VertexStencil
from autoflowcfd.core.fr_solver.residual_diagnostics import _reference_scales
from autoflowcfd.core.mpi.vertex_stencil_mpi import (
    build_distributed_vertex_stencil, local_vertex_pairs, vertex_pairs_of_cells,
)
from tests.unit._thread_comm import ThreadComm
from tests.unit.test_sensor_gate_distributed import _build_global_case, _rank_view

_REF = _reference_scales({"rho_inf": 1.225, "vel_inf": 30.0, "p_inf": 101325.0}, 5)


def _for_rank(comm, rank):
    """顶点模板构造要的集合通信：按 rank 顺序拼接 / 逐元素 MAX、MIN。"""
    return dict(
        allgather=lambda a: comm.allgather(rank, a),
        allreduce_max=lambda a: comm.collective(rank, np.asarray(a), lambda v: np.maximum.reduce(v)),
        allreduce_min=lambda a: comm.collective(rank, np.asarray(a), lambda v: np.minimum.reduce(v)),
        n_ranks=comm.n)


def _vertex_pairs(n_cells, seed=5, n_hubs=12):
    """每个单元 3 个顶点：链上相邻共享的 i、i+1，外加一个"枢纽"顶点（被相隔很远的
    单元共享 —— 只经顶点相邻、面跳数任意大，正是 1 层 halo 拼不出来的那类邻居）。"""
    rng = np.random.default_rng(seed)
    nodes = np.stack([np.arange(n_cells), np.arange(n_cells) + 1,
                      n_cells + 1 + rng.integers(0, n_hubs, n_cells)], axis=1)
    return nodes.ravel().astype(np.int64), np.repeat(np.arange(n_cells), 3).astype(np.int64)


def _run_ranks(field, owner, neigh, is_bnd, node_g, cell_g, assign, reduce=True):
    """各 rank 的 `(local 单元全局号, 越界比)`，按 rank 排列。"""
    comm = ThreadComm(int(assign.max()) + 1)

    def work(r):
        local_ids = np.flatnonzero(assign == r)
        native_ids, n_local, o_r, n_r, b_r = _rank_view(owner, neigh, is_bnd, local_ids)
        pos = {int(g): i for i, g in enumerate(local_ids)}
        keep = np.isin(cell_g, local_ids)
        cell_nat = np.array([pos[int(c)] for c in cell_g[keep]], dtype=np.int64)
        kw = _for_rank(comm, r) if reduce else dict(n_ranks=1)
        st = build_distributed_vertex_stencil(node_g[keep], cell_nat, **kw)
        ratio = compute_bounds_violation_ratio(
            field[native_ids], o_r, n_r, b_r, ref_scales=_REF, vertex_stencil=st)
        return local_ids, ratio[:n_local]

    return comm.run(work)


@pytest.mark.parametrize("n_ranks", [2, 3, 5])
def test_distributed_ratio_equals_single_machine_bitwise(n_ranks):
    field, owner, neigh, is_bnd, _ = _build_global_case()
    n_cells = field.shape[0]
    node_g, cell_g = _vertex_pairs(n_cells)
    nodes_u, node_local = np.unique(node_g, return_inverse=True)
    global_ratio = compute_bounds_violation_ratio(
        field, owner, neigh, is_bnd, ref_scales=_REF,
        vertex_stencil=VertexStencil(node_local.astype(np.int64), cell_g, int(nodes_u.size)))
    assign = np.random.default_rng(n_ranks).integers(0, n_ranks, n_cells)
    for local_ids, ratio in _run_ranks(field, owner, neigh, is_bnd, node_g, cell_g, assign):
        np.testing.assert_array_equal(ratio, global_ratio[local_ids])


def test_without_shared_node_reduction_the_result_differs():
    field, owner, neigh, is_bnd, _ = _build_global_case()
    n_cells = field.shape[0]
    node_g, cell_g = _vertex_pairs(n_cells)
    nodes_u, node_local = np.unique(node_g, return_inverse=True)
    global_ratio = compute_bounds_violation_ratio(
        field, owner, neigh, is_bnd, ref_scales=_REF,
        vertex_stencil=VertexStencil(node_local.astype(np.int64), cell_g, int(nodes_u.size)))
    assign = np.random.default_rng(3).integers(0, 3, n_cells)
    n_diff = sum(int(np.count_nonzero(ratio != global_ratio[ids]))
                 for ids, ratio in _run_ranks(field, owner, neigh, is_bnd, node_g, cell_g,
                                              assign, reduce=False))
    assert n_diff > 0, "不做共享顶点归约也逐位相同 —— 用例没有跨 rank 的顶点邻居，失去区分力"


def test_vertex_pairs_of_cells_reads_prism_first_connectivity():
    from types import SimpleNamespace
    mesh = SimpleNamespace(
        n_prism_cells=2,
        _fixed_prism_conn=np.array([[0, 1, 2, 3, 4, 5], [3, 4, 5, 6, 7, 8]]),
        _fixed_tet_conn=np.array([[6, 7, 8, 9]]))
    node, pos = vertex_pairs_of_cells(mesh, np.array([2, 0]))
    # 对是集合语义（顺序无关）：位置 0 是全局单元 2（四面体），位置 1 是单元 0（棱柱）
    assert set(zip(node.tolist(), pos.tolist())) == (
        {(n, 0) for n in (6, 7, 8, 9)} | {(n, 1) for n in range(6)})
    # 完全分布式加载随包下发的那份优先
    shipped = (np.array([1]), np.array([0]))
    mesh.local_vertex_pairs = shipped
    assert local_vertex_pairs(mesh, SimpleNamespace(local_cells=np.array([0]))) is shipped
