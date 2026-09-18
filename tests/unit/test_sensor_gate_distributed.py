"""CPU MPI 分布式路径上的传感器门控（`AFCFD_FILTER_MODE=sensor`）。

`sensor` 是 2026-09-17 定下的默认滤波档，但当时只有单机 CPU 接线，其余
三条后端默认退回 `project`（全局逐 stage 施加精确投影，功能上等于
legacy：P1 退化成 P0、壁面剪应力恒为零）。2026-09-18 补齐 CPU MPI。

本文件的核心判据是**分区独立性**：BJ 越界判据要面邻居的单元均值，分区
边界上的邻居是 halo 单元。如果实现把分区边界面当边界面排除，掩码就会
随 rank 数变化——同一个算例换分区数得到不同的解，这对求解器不可接受。
`test_mask_is_partition_independent` 直接把这条性质钉成逐位相等。
"""

import numpy as np
import pytest

from autoflowcfd.core.fr_operators.bounds_sensor import (
    compute_bounds_violation_mask,
)
from autoflowcfd.core.fr_solver.filter import (
    build_sensor_gated_filter_func_arrays,
    resolve_filter_mode,
    _SENSOR_MODE_SUPPORTED_BACKENDS,
)


# ---------------------------------------------------------------- 合成算例

def _build_global_case(seed=11, n_cells=240, n_sps=6, n_var=5):
    """一个带面连接的合成全局算例。

    场刻意做成「大部分光滑、少数单元带跳变」，这样掩码既不恒空也不全满
    ——恒空/全满的掩码对「分区是否影响结果」没有区分力。
    """
    rng = np.random.default_rng(seed)
    x = np.linspace(0.0, 1.0, n_cells)
    base = np.stack([1.2 + 0.1 * np.sin(6.0 * x),
                     30.0 * x,
                     0.5 * np.cos(4.0 * x),
                     0.1 * x,
                     101325.0 + 50.0 * x], axis=1)          # (n_cells, n_var)
    field = np.repeat(base[:, None, :], n_sps, axis=1)
    # 单元内非常数内容（光滑部分，量级小）
    field += 1e-3 * rng.normal(size=field.shape) * np.abs(base)[:, None, :]
    # 少数单元注入真正的跳变（会被 BJ 判据标记）
    bad = rng.choice(n_cells, size=n_cells // 12, replace=False)
    field[bad] += 0.35 * np.abs(base)[bad][:, None, :] * rng.normal(
        size=(bad.size, n_sps, n_var))

    # 面连接：一条一维链（每个内部面连相邻单元）+ 若干随机跨接面，
    # 再加两端的物理边界面。随机跨接面是为了让分区切分不平凡。
    o = list(range(n_cells - 1))
    n = list(range(1, n_cells))
    bnd = [False] * len(o)
    for _ in range(n_cells // 4):
        a = int(rng.integers(0, n_cells))
        b = int(rng.integers(0, n_cells))
        if a == b:
            continue
        o.append(a)
        n.append(b)
        bnd.append(False)
    o += [0, n_cells - 1]
    n += [-1, -1]
    bnd += [True, True]
    return (field,
            np.asarray(o, dtype=np.int64),
            np.asarray(n, dtype=np.int64),
            np.asarray(bnd, dtype=bool),
            bad)


_FREESTREAM = {"rho_inf": 1.225, "vel_inf": 30.0, "p_inf": 101325.0}
_REF = np.array([1.225, 1.225 * 30.0, 1.225 * 30.0, 1.225 * 30.0, 101325.0])


def _rank_view(owner, neigh, is_bnd, local_ids):
    """把全局面连接裁成一个 rank 的 local+halo 视图。

    返回的索引空间与 `HaloExchange.exchange` 的**原生**排列一致：
    [0, n_local) 是 local_ids 自身顺序，[n_local, n_total) 是 halo。
    这正是 `DistributedFRSolver` 的 `state.U` / 滤波回调所处的空间。
    """
    local_set = set(int(c) for c in local_ids)
    own_local = np.array([int(c) in local_set for c in owner])
    nb_local = np.array([(not b) and int(c) in local_set
                         for c, b in zip(neigh, is_bnd)])
    keep = np.flatnonzero(own_local | nb_local)      # partition.local_faces

    halo = []
    for f in keep:
        for c, b in ((owner[f], False), (neigh[f], is_bnd[f])):
            if b:
                continue
            if int(c) not in local_set and int(c) not in halo:
                halo.append(int(c))
    halo.sort()

    native_ids = np.concatenate([np.asarray(local_ids, dtype=np.int64),
                                 np.asarray(halo, dtype=np.int64)])
    g2n = {int(g): i for i, g in enumerate(native_ids)}
    o_r = np.array([g2n[int(owner[f])] for f in keep], dtype=np.int64)
    n_r = np.array([g2n[int(neigh[f])] if not is_bnd[f] else 0
                    for f in keep], dtype=np.int64)
    b_r = is_bnd[keep]
    return native_ids, len(local_ids), o_r, n_r, b_r


# ------------------------------------------------------------------- 判据

class TestPartitionIndependence:
    def test_mask_is_partition_independent(self):
        """两个 rank 各自算出的掩码，与单机全局掩码逐位相同。

        这是本次接线的唯一决定性判据：它同时否掉「把分区边界面当边界面
        排除」和「只用本地单元均值」两种偷懒实现——两者都会在分区边界
        附近给出不同的掩码。
        """
        field, owner, neigh, is_bnd, _ = _build_global_case()
        n_cells = field.shape[0]
        global_mask = compute_bounds_violation_mask(
            field, owner, neigh, is_bnd, ref_scales=_REF)
        assert 0 < global_mask.sum() < n_cells, (
            f"掩码必须既不空也不满才有区分力，实际 "
            f"{global_mask.sum()}/{n_cells}")

        cut = n_cells // 2
        rank_cells = [np.arange(0, cut), np.arange(cut, n_cells)]
        n_partition_faces = 0
        for local_ids in rank_cells:
            native_ids, n_local, o_r, n_r, b_r = _rank_view(
                owner, neigh, is_bnd, local_ids)
            assert len(native_ids) > n_local, "该分区应当有 halo 单元"
            n_partition_faces += int(
                ((o_r >= n_local) | ((~b_r) & (n_r >= n_local))).sum())
            field_ext = field[native_ids]
            mask_r = compute_bounds_violation_mask(
                field_ext, o_r, n_r, b_r, ref_scales=_REF)[:n_local]
            np.testing.assert_array_equal(
                mask_r, global_mask[local_ids],
                err_msg="分区掩码与全局掩码不一致：门控结果依赖了分区数")
        assert n_partition_faces > 0, "构造的算例没有分区边界面，测不到东西"

    def test_excluding_partition_faces_would_change_the_mask(self):
        """反证：如果把 halo 面当边界面排除，掩码**确实**会变。

        没有这条，上面那个「逐位相同」可能只是因为算例里分区边界无关紧要。
        """
        field, owner, neigh, is_bnd, _ = _build_global_case()
        n_cells = field.shape[0]
        global_mask = compute_bounds_violation_mask(
            field, owner, neigh, is_bnd, ref_scales=_REF)
        cut = n_cells // 2
        differed = False
        for local_ids in (np.arange(0, cut), np.arange(cut, n_cells)):
            native_ids, n_local, o_r, n_r, b_r = _rank_view(
                owner, neigh, is_bnd, local_ids)
            # 错误实现：把所有涉及 halo 的面标成边界面
            b_wrong = b_r | (o_r >= n_local) | (n_r >= n_local)
            mask_wrong = compute_bounds_violation_mask(
                field[native_ids], o_r, n_r, b_wrong,
                ref_scales=_REF)[:n_local]
            if not np.array_equal(mask_wrong, global_mask[local_ids]):
                differed = True
        assert differed, (
            "构造的算例区分不了正确/错误实现，上面那条逐位相等是空的")


class TestHaloExtendContract:
    def test_halo_extend_none_matches_single_machine(self):
        """n_halo == 0（单 rank）时，扩展路径与单机路径逐位相同。"""
        field, owner, neigh, is_bnd, _ = _build_global_case()
        n_cells, n_sps = field.shape[0], field.shape[1]
        kw = dict(sensor="bounds", owner_cell=owner, neighbor_cell=neigh,
                  is_boundary=is_bnd, freestream=_FREESTREAM)
        # 一个真正非平凡的投影矩阵（把单元内非常数内容抹掉）
        F = np.full((n_sps, n_sps), 1.0 / n_sps)
        cip = np.zeros(n_cells, dtype=bool)
        plain = build_sensor_gated_filter_func_arrays(
            n_cells, n_sps, 1, F, F, cell_is_prism=cip, **kw)
        extended = build_sensor_gated_filter_func_arrays(
            n_cells, n_sps, 1, F, F, cell_is_prism=cip,
            halo_extend=lambda U: U, **kw)
        flat = field.reshape(n_cells * n_sps, 5).copy()
        np.testing.assert_array_equal(plain(flat.copy()), extended(flat.copy()))

    def test_halo_extend_actually_changes_the_result(self):
        """halo 均值真的参与了包络：被施加滤波的单元集合应当恰好等于
        全局掩码在本 rank 上的限制。"""
        field, owner, neigh, is_bnd, _ = _build_global_case()
        n_cells, n_sps = field.shape[0], field.shape[1]
        cut = n_cells // 2
        local_ids = np.arange(0, cut)
        native_ids, n_local, o_r, n_r, b_r = _rank_view(
            owner, neigh, is_bnd, local_ids)
        F = np.full((n_sps, n_sps), 1.0 / n_sps)
        cip = np.zeros(n_local, dtype=bool)
        ff = build_sensor_gated_filter_func_arrays(
            n_local, n_sps, 1, F, F, cell_is_prism=cip, sensor="bounds",
            owner_cell=o_r, neighbor_cell=n_r, is_boundary=b_r,
            freestream=_FREESTREAM,
            halo_extend=lambda U: field[native_ids])
        flat = field[local_ids].reshape(n_local * n_sps, 5).copy()
        out = ff(flat.copy())
        global_mask = compute_bounds_violation_mask(
            field, owner, neigh, is_bnd, ref_scales=_REF)
        changed = np.any(out.reshape(n_local, n_sps, 5) != field[local_ids],
                         axis=(1, 2))
        np.testing.assert_array_equal(
            changed, global_mask[local_ids],
            err_msg="被施加滤波的单元集合应当恰好是全局掩码标记的那些")

    def test_halo_extend_rejected_for_persson(self):
        """persson 是纯单元局部判据，给它 halo_extend 必须报错而不是忽略。"""
        F = np.eye(4) * 0.5
        with pytest.raises(ValueError, match="halo_extend"):
            build_sensor_gated_filter_func_arrays(
                10, 4, 1, F, F, cell_is_prism=np.zeros(10, bool),
                sensor="persson", halo_extend=lambda U: U)


class TestBackendRegistration:
    def test_cpu_mpi_is_registered_as_wired(self):
        assert "cpu-mpi" in _SENSOR_MODE_SUPPORTED_BACKENDS

    def test_cpu_mpi_no_longer_falls_back(self, monkeypatch):
        monkeypatch.setenv("AFCFD_FILTER_MODE", "sensor")
        assert resolve_filter_mode("cpu-mpi") == "sensor"
        assert resolve_filter_mode("cpu-single") == "sensor"

    def test_all_four_backends_resolve_to_sensor(self, monkeypatch):
        """四条后端全部接线（2026-09-18），显式请求都能满足。"""
        monkeypatch.setenv("AFCFD_FILTER_MODE", "sensor")
        for backend in ("cpu-single", "cpu-mpi", "gpu-single", "gpu-mpi"):
            assert resolve_filter_mode(backend) == "sensor", backend

    def test_unknown_backend_still_raises(self, monkeypatch):
        """未知后端名仍必须报错——退档分支删除后这是唯一的行为。"""
        monkeypatch.setenv("AFCFD_FILTER_MODE", "sensor")
        with pytest.raises(NotImplementedError, match="gpu-rocm"):
            resolve_filter_mode("gpu-rocm")


# ------------------------- DistributedFRSolver 方法本体（perm 换算 + 护栏）

def _stub_solver(field, owner, neigh, is_bnd, local_ids, n_sps):
    """构造调用 `_build_sensor_gated_filter_func_distributed` 所需的最小
    替身，并刻意用一个**非恒等**的 perm 置换。

    为什么要非恒等：`state.U` 处在 halo 交换的原生排列，而
    `dist_flat_face.owner_cell_local` 处在「棱柱在前」的紧凑排列，两者靠
    `perm` 换算（紧凑下标 k 对应原生下标 perm[k]）。恒等置换测不出这个
    方向有没有搞反——搞反在真实网格上会静默地对错误的单元取均值。
    """
    from types import SimpleNamespace
    native_ids, n_local, o_nat, n_nat, b_r = _rank_view(
        owner, neigh, is_bnd, local_ids)
    n_total = len(native_ids)
    rng = np.random.default_rng(3)
    perm = rng.permutation(n_total).astype(np.int64)     # native[perm] == permuted
    inv = np.empty_like(perm)
    inv[perm] = np.arange(n_total, dtype=perm.dtype)
    # 把原生索引换成紧凑索引：紧凑下标 = inv[原生下标]
    o_cmp = inv[o_nat]
    n_cmp = np.where(b_r, -1, inv[n_nat])

    dist_fc = SimpleNamespace(
        perm=perm, inv_perm=inv,
        owner_cell_local=o_cmp, neighbor_cell_local=n_cmp,
        is_boundary=b_r,
        compact_cell_type=np.zeros(n_total, dtype=np.int8),
    )
    F = np.full((n_sps, n_sps), 1.0 / n_sps)
    # `local_solver` 是真实类上的惰性属性，门控从它取
    # `boundary_ghost_provider` 来构造无滑移壁面的动量 Dirichlet 表
    # （不给表，贴壁单元会被结构性误判、壁面剪应力被压掉 14 倍，见
    # tests/unit/test_bounds_sensor.py::
    # TestBoundaryDirichletCompletesTheEnvelope）。这里默认给一个
    # 全 FARFIELD 的 provider：表里全是 NaN，等价于不给表，于是下面
    # 那几条关于 perm 换算的判据不受影响。
    prov = SimpleNamespace(
        group_code=np.full(len(b_r), -1, dtype=np.int64),
        code_to_config={},
        default_config={"type": "FARFIELD"},
    )
    stub = SimpleNamespace(
        local_solver=SimpleNamespace(boundary_ghost_provider=prov),
        dist_flat_face=dist_fc,
        partition=SimpleNamespace(n_total_cells=n_total,
                                  n_local_cells=n_local),
        halo_exchange=SimpleNamespace(
            exchange=lambda U_local: np.concatenate(
                [U_local, field[native_ids[n_local:]]], axis=0)),
        freestream=dict(_FREESTREAM),
        ops=SimpleNamespace(filter_prism=F, filter_tet=F),
        order=1, current_order=1,
    )
    return stub, native_ids, n_local


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
        orig = bs.compute_bounds_violation_mask

        def spy(*a, **kw):
            seen["bd"] = kw.get("bnd_dirichlet")
            return orig(*a, **kw)

        monkeypatch.setattr(bs, "compute_bounds_violation_mask", spy)

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
