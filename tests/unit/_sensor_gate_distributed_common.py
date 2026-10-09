"""分布式问题单元判据测试（test_sensor_gate_distributed*.py）共用的算例与求解器替身。"""

import numpy as np

from autoflowcfd.core.fr_solver.residual_diagnostics import (
    _reference_scales,
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


#: 参考量级**读**生产实现，不在测试里再抄一份常量。抄一份的后果是：
#: 哪天 `_reference_scales` 改了口径，测试仍按旧口径通过，于是这套
#: "分布式与单机掩码必须逐位一致"的判据会静默失效。
_REF = _reference_scales(_FREESTREAM, 5)


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

    # `true_normal` 是逐**通量点**的 (n_faces, n_fp, 3)（真实
    # FlatFaceGeometry 就是这个形状），门控要靠它给对称面/滑移壁补镜像
    # 包络贡献。这里给两个通量点，顺带钉住"逐通量点要被归约成逐面"。
    n_faces_stub = len(b_r)
    tn = np.zeros((n_faces_stub, 2, 3))
    tn[:, :, 2] = 1.0
    dist_fc = SimpleNamespace(
        perm=perm, inv_perm=inv,
        owner_cell_local=o_cmp, neighbor_cell_local=n_cmp,
        is_boundary=b_r,
        compact_cell_type=np.zeros(n_total, dtype=np.int8),
        true_normal=tn,
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
    # 合成算例只有面图、没有真实顶点：每个单元给一个私有顶点（编号 = 全局
    # 单元号），顶点包络退化为本单元自身，于是这里的判据仍与面模板全局掩码
    # 逐位可比。真正的跨 rank 顶点归约见 `test_vertex_stencil_mpi.py`。
    stub = SimpleNamespace(
        mesh=SimpleNamespace(local_vertex_pairs=(
            np.asarray(local_ids, dtype=np.int64),
            np.arange(n_local, dtype=np.int64))),
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
