"""分布式问题单元判据：求解器方法接线、Dirichlet 表与镜像法向到达分布式路径（从 test_sensor_gate_distributed.py 拆出）。"""

import numpy as np
import pytest

from autoflowcfd.core.fr_operators.bounds_sensor import (
    compute_bounds_violation_mask,
)

from tests.unit._patch_pkg import patch_pkg_attr
from tests.unit._sensor_gate_distributed_common import _REF, _build_global_case, _rank_view, _stub_solver


class TestDistributedSolverMethod:
    def test_perm_conversion_gives_the_global_mask(self, monkeypatch):
        """走真实方法体（含 perm 换算），被滤波的单元集合必须等于全局掩码。"""
        monkeypatch.setenv("AFCFD_TROUBLED_SENSOR", "bounds")
        from autoflowcfd.core.mpi.distributed_solver import DistributedFRSolver

        field, owner, neigh, is_bnd, _ = _build_global_case()
        n_cells, n_sps = field.shape[0], field.shape[1]
        local_ids = np.arange(0, n_cells // 2)
        stub, native_ids, n_local = _stub_solver(
            field, owner, neigh, is_bnd, local_ids, n_sps)

        ff = DistributedFRSolver._build_sensor_gated_filter_func_distributed(
            stub, n_local, n_sps, np.zeros(n_local, dtype=bool))
        flat = field[local_ids].reshape(n_local * n_sps, 5).copy()
        out = ff(flat.copy())
        changed = np.any(out.reshape(n_local, n_sps, 5) != field[local_ids],
                         axis=(1, 2))
        global_mask = compute_bounds_violation_mask(
            field, owner, neigh, is_bnd, ref_scales=_REF)
        np.testing.assert_array_equal(
            changed, global_mask[local_ids],
            err_msg="perm 紧凑->原生换算方向错了，或 halo 均值没参与包络")

    def test_reversed_perm_direction_is_detectably_wrong(self, monkeypatch):
        """把 perm 换成它的逆（写错方向的典型形态），结果必须不同。

        用来证明上一条不是恒真——否则「方向搞反」这类 bug 通过不了
        任何检测。
        """
        monkeypatch.setenv("AFCFD_TROUBLED_SENSOR", "bounds")
        from autoflowcfd.core.mpi.distributed_solver import DistributedFRSolver

        field, owner, neigh, is_bnd, _ = _build_global_case()
        n_cells, n_sps = field.shape[0], field.shape[1]
        local_ids = np.arange(0, n_cells // 2)
        stub, native_ids, n_local = _stub_solver(
            field, owner, neigh, is_bnd, local_ids, n_sps)
        stub.dist_flat_face.perm = stub.dist_flat_face.inv_perm   # 方向搞反

        ff = DistributedFRSolver._build_sensor_gated_filter_func_distributed(
            stub, n_local, n_sps, np.zeros(n_local, dtype=bool))
        flat = field[local_ids].reshape(n_local * n_sps, 5).copy()
        out = ff(flat.copy())
        changed = np.any(out.reshape(n_local, n_sps, 5) != field[local_ids],
                         axis=(1, 2))
        global_mask = compute_bounds_violation_mask(
            field, owner, neigh, is_bnd, ref_scales=_REF)
        assert not np.array_equal(changed, global_mask[local_ids]), (
            "perm 方向搞反却得到同样的掩码——这条测试没有区分力")

    def test_inconsistent_boundary_flags_raise(self, monkeypatch):
        """(neighbor_cell_local < 0) 与 is_boundary 不重合必须报错。

        不重合意味着某条内部面没有邻居单元、或某条边界面带着真实邻居，
        两者都会让 BJ 包络读到错误的单元，而那种错误在残差日志里完全
        看不出来。
        """
        monkeypatch.setenv("AFCFD_TROUBLED_SENSOR", "bounds")
        from autoflowcfd.core.mpi.distributed_solver import DistributedFRSolver

        field, owner, neigh, is_bnd, _ = _build_global_case()
        n_cells, n_sps = field.shape[0], field.shape[1]
        local_ids = np.arange(0, n_cells // 2)
        stub, _, n_local = _stub_solver(
            field, owner, neigh, is_bnd, local_ids, n_sps)
        b = np.asarray(stub.dist_flat_face.is_boundary).copy()
        b[0] = not b[0]
        stub.dist_flat_face.is_boundary = b
        with pytest.raises(RuntimeError, match="自洽性"):
            DistributedFRSolver._build_sensor_gated_filter_func_distributed(
                stub, n_local, n_sps, np.zeros(n_local, dtype=bool))

    def test_perm_length_mismatch_raises(self, monkeypatch):
        monkeypatch.setenv("AFCFD_TROUBLED_SENSOR", "bounds")
        from autoflowcfd.core.mpi.distributed_solver import DistributedFRSolver

        field, owner, neigh, is_bnd, _ = _build_global_case()
        n_cells, n_sps = field.shape[0], field.shape[1]
        local_ids = np.arange(0, n_cells // 2)
        stub, _, n_local = _stub_solver(
            field, owner, neigh, is_bnd, local_ids, n_sps)
        stub.partition.n_total_cells += 1
        with pytest.raises(RuntimeError, match="perm"):
            DistributedFRSolver._build_sensor_gated_filter_func_distributed(
                stub, n_local, n_sps, np.zeros(n_local, dtype=bool))

    def test_persson_needs_no_connectivity(self, monkeypatch):
        """persson 档不读面连接，方法必须照样能构造出回调。"""
        monkeypatch.setenv("AFCFD_TROUBLED_SENSOR", "persson")
        from autoflowcfd.core.mpi.distributed_solver import DistributedFRSolver

        field, owner, neigh, is_bnd, _ = _build_global_case()
        n_cells, n_sps = field.shape[0], field.shape[1]
        local_ids = np.arange(0, n_cells // 2)
        stub, _, n_local = _stub_solver(
            field, owner, neigh, is_bnd, local_ids, n_sps)
        ff = DistributedFRSolver._build_sensor_gated_filter_func_distributed(
            stub, n_local, n_sps, np.zeros(n_local, dtype=bool))
        flat = field[local_ids].reshape(n_local * n_sps, 5).copy()
        assert ff(flat.copy()).shape == flat.shape


class TestDirichletTableReachesTheDistributedPath:
    """壁面 Dirichlet 表必须真的进到分布式门控的判据内核里。

    不然这条后端会重犯那个已修的缺陷：贴壁单元被结构性误判、壁面剪应力
    被压掉 14 倍（实测表见 tests/unit/test_bounds_sensor.py::
    TestBoundaryDirichletCompletesTheEnvelope）。

    判据用**给内核装仪表**而不是"看掩码变不变"：BJ 掩码是 5 个变量取
    并集，合成场上别的变量很容易主导，掩码不变并不能说明表没到
    （第一版就是这么写的，它对真实缺陷没有区分力）。
    """

    def test_kernel_receives_a_table_with_finite_wall_entries(self, monkeypatch):
        from types import SimpleNamespace

        monkeypatch.setenv("AFCFD_TROUBLED_SENSOR", "bounds")
        from autoflowcfd.core.mpi.distributed_solver import DistributedFRSolver
        from autoflowcfd.core.fr_operators import bounds_sensor as bs

        field, owner, neigh, is_bnd, _ = _build_global_case()
        n_cells, n_sps = field.shape[0], field.shape[1]
        local_ids = np.arange(0, n_cells // 2)
        _, _, o_r, n_r, b_r = _rank_view(owner, neigh, is_bnd, local_ids)
        n_faces = len(b_r)

        gc = np.full(n_faces, -1, dtype=np.int64)
        gc[np.asarray(b_r, dtype=bool)] = 7
        wall = SimpleNamespace(
            group_code=gc,
            code_to_config={7: {"type": "WALL", "is_no_slip": True,
                                "wall_velocity": [0.0, 0.0, 0.0]}},
            default_config={"type": "FARFIELD"})

        seen = {}
        orig = bs.compute_bounds_violation_ratio

        def spy(*a, **kw):
            seen["bd"] = kw.get("bnd_dirichlet")
            return orig(*a, **kw)

        patch_pkg_attr(monkeypatch, bs, "compute_bounds_violation_ratio", spy)

        stub, _, n_local = _stub_solver(
            field, owner, neigh, is_bnd, local_ids, n_sps)
        stub.local_solver = SimpleNamespace(boundary_ghost_provider=wall)
        ff = DistributedFRSolver._build_sensor_gated_filter_func_distributed(
            stub, n_local, n_sps, np.zeros(n_local, dtype=bool))
        flat = field[local_ids].reshape(n_local * n_sps, 5).copy()
        ff(flat.copy())

        bd = seen.get("bd")
        assert bd is not None, "判据内核根本没收到 bnd_dirichlet"
        bd = np.asarray(bd)
        assert bd.shape == (n_faces, 5), f"表形状 {bd.shape} 不对"
        is_b = np.asarray(b_r, dtype=bool)
        assert np.all(bd[is_b, 1:4] == 0.0), (
            "边界面的动量三列应当是 0（静止无滑移壁）")
        assert np.all(~np.isfinite(bd[~is_b])), "内部面应当全是 NaN"

    def test_no_provider_means_no_table_rather_than_a_crash(self, monkeypatch):
        """provider 还没建好时必须退回"不给表"，不能崩。

        构造顺序在不同后端不同（多 GPU 的滤波初始化就在 provider 之前），
        所以这条路径必须是安全的——退回的行为与修复前逐位一致。
        """
        from types import SimpleNamespace

        monkeypatch.setenv("AFCFD_TROUBLED_SENSOR", "bounds")
        from autoflowcfd.core.mpi.distributed_solver import DistributedFRSolver

        field, owner, neigh, is_bnd, _ = _build_global_case()
        n_cells, n_sps = field.shape[0], field.shape[1]
        local_ids = np.arange(0, n_cells // 2)
        stub, _, n_local = _stub_solver(
            field, owner, neigh, is_bnd, local_ids, n_sps)
        stub.local_solver = SimpleNamespace(boundary_ghost_provider=None)
        ff = DistributedFRSolver._build_sensor_gated_filter_func_distributed(
            stub, n_local, n_sps, np.zeros(n_local, dtype=bool))
        flat = field[local_ids].reshape(n_local * n_sps, 5).copy()
        assert ff(flat.copy()).shape == flat.shape


class TestMirrorNormalsReachTheDistributedPath:
    """对称面/滑移壁的镜像法向必须真的进到分布式门控的判据内核里。

    与上面那条 Dirichlet 表的测试是同一个缺陷的两半：边界面没有邻居单元
    均值 -> 包络单侧收窄 -> "线性场恒不触发"这个设计不变量被破坏。漏掉
    这条后端不会报错，只会让对称面旁的单元被结构性误判，而那在残差日志里
    完全看不出来。
    """

    def test_kernel_receives_reduced_per_face_normals(self, monkeypatch):
        from types import SimpleNamespace

        monkeypatch.setenv("AFCFD_TROUBLED_SENSOR", "bounds")
        from autoflowcfd.core.mpi.distributed_solver import DistributedFRSolver
        from autoflowcfd.core.fr_operators import bounds_sensor as bs

        field, owner, neigh, is_bnd, _ = _build_global_case()
        n_cells, n_sps = field.shape[0], field.shape[1]
        local_ids = np.arange(0, n_cells // 2)
        _, _, _o_r, _n_r, b_r = _rank_view(owner, neigh, is_bnd, local_ids)
        n_faces = len(b_r)

        gc = np.full(n_faces, -1, dtype=np.int64)
        gc[np.asarray(b_r, dtype=bool)] = 3
        sym = SimpleNamespace(
            group_code=gc,
            code_to_config={3: {"type": "SYMMETRY"}},
            default_config={"type": "FARFIELD"})

        seen = {}
        orig = bs.compute_bounds_violation_ratio

        def spy(*a, **kw):
            seen["mir"] = kw.get("bnd_mirror_normal")
            return orig(*a, **kw)

        patch_pkg_attr(monkeypatch, bs, "compute_bounds_violation_ratio", spy)

        stub, _, n_local = _stub_solver(
            field, owner, neigh, is_bnd, local_ids, n_sps)
        stub.local_solver = SimpleNamespace(boundary_ghost_provider=sym)
        ff = DistributedFRSolver._build_sensor_gated_filter_func_distributed(
            stub, n_local, n_sps, np.zeros(n_local, dtype=bool))
        flat = field[local_ids].reshape(n_local * n_sps, 5).copy()
        ff(flat.copy())

        mir = seen.get("mir")
        assert mir is not None, "判据内核根本没收到 bnd_mirror_normal"
        mir = np.asarray(mir)
        assert mir.shape == (n_faces, 3), (
            f"形状 {mir.shape} 不对——逐通量点的 true_normal 应当被归约成"
            f"逐面 (n_faces, 3)")
        is_b = np.asarray(b_r, dtype=bool)
        assert np.allclose(mir[is_b], np.array([0.0, 0.0, 1.0])), (
            "对称面应当拿到单位外法向")
        assert np.all(~np.isfinite(mir[~is_b])), "内部面应当全是 NaN"
