"""AutoFlowCFD V2.0 - 分布式 SST/DDES/IDDES 湍流模型端到端验证
（SST 2026-09-02，DDES/IDDES 同日续接）。

核心判据：与本项目已建立的方法论一致（见 test_distributed_compute_
residual.py 模块文档）——用同一份真实（合成）混合网格，分布式路径算出
的 local cells 结果必须与单机路径逐位一致，而不只是"看起来合理"。

本机没有真实 MPI/CuPy，用假 halo 交换器（返回已知正确的扩展状态）绕开
真正的点对点通信，但测试 `distributed_compute_turbulence_source_and_
viscosity` 自身的全部真实生产逻辑：halo 交换协议+压缩索引空间重排、
mesh/solver 适配器构造、复用单机 `compute_turbulence_source`/
`SSTModelFR.compute_source_terms`/`update_fields` 的真实数值计算。

DDES/IDDES 扩展沿用完全相同的判据，唯一的额外复杂度是 IDDES 需要
`des.py::compute_h_max_and_h_wn` 算出的逐单元几何量按 compact 索引空间
切片（纯逐单元局部量，见该函数文档，不需要新的跨 rank 几何交换）。
"""

import numpy as np
import pytest

from autoflowcfd.core.mpi.partition import build_distributed_partition
from autoflowcfd.core.mpi.distributed_flat_face import build_distributed_flat_face
from autoflowcfd.core.mpi.distributed_turbulence import (
    distributed_compute_turbulence_source_and_viscosity,
    compute_distributed_wall_distance,
)
from autoflowcfd.core.fr_residual.inviscid import primitive_to_conserved
from autoflowcfd.core.turbulence.sst import SSTModelFR
from autoflowcfd.core.turbulence.des import DDESModel, IDDESModel, compute_h_max_and_h_wn
from autoflowcfd.fr.operators import generate_fr_operators
from tests.unit.test_fr_residual_inviscid import _build_synthetic_mixed_mesh


class _FakeHaloExchange:
    def __init__(self, U_extended: np.ndarray):
        self._U_extended = U_extended

    def exchange(self, U_local: np.ndarray) -> np.ndarray:
        return self._U_extended


def _nonuniform_state(mesh, rng):
    """构造一个非均匀但物理合理的流场+k/omega 场（覆盖对流/扩散/源项
    全部非平凡分支，不是解析零残差的平凡情形）。"""
    rho_inf, u_inf, v_inf, w_inf, p_inf = 1.225, 30.0, 5.0, -3.0, 101325.0
    n_cells, n_sps = mesh.n_cells, mesh.n_sps_per_cell
    Q = np.zeros((n_cells, n_sps, 5))
    Q[..., 0] = rho_inf * (1.0 + rng.uniform(-0.02, 0.02, size=(n_cells, n_sps)))
    Q[..., 1] = u_inf + rng.uniform(-3.0, 3.0, size=(n_cells, n_sps))
    Q[..., 2] = v_inf + rng.uniform(-2.0, 2.0, size=(n_cells, n_sps))
    Q[..., 3] = w_inf + rng.uniform(-2.0, 2.0, size=(n_cells, n_sps))
    Q[..., 4] = p_inf * (1.0 + rng.uniform(-0.01, 0.01, size=(n_cells, n_sps)))
    U = np.stack(
        [primitive_to_conserved(Q[c, s]) for c in range(n_cells) for s in range(n_sps)]
    ).reshape(n_cells, n_sps, 5)

    k_field = 1.0 * (1.0 + rng.uniform(-0.3, 0.3, size=(n_cells, n_sps)))
    omega_field = 500.0 * (1.0 + rng.uniform(-0.3, 0.3, size=(n_cells, n_sps)))
    return U, k_field, omega_field


def _make_ddes_model(turb_model_name):
    if turb_model_name == "SST":
        return None
    if turb_model_name == "DDES":
        return DDESModel()
    if turb_model_name == "IDDES":
        return IDDESModel()
    raise ValueError(turb_model_name)


@pytest.mark.parametrize("turb_model_name", ["SST", "DDES", "IDDES"])
@pytest.mark.parametrize("rank", [0, 1])
def test_distributed_turbulence_matches_single_machine(rank, turb_model_name):
    order = 1
    mesh = _build_synthetic_mixed_mesh(order)
    ops = generate_fr_operators(order)
    n_cells, n_sps = mesh.n_cells, mesh.n_sps_per_cell
    mu = 1.8e-5
    rng = np.random.default_rng(4242)

    U, k_field, omega_field = _nonuniform_state(mesh, rng)
    d_wall = np.full((n_cells, n_sps), 0.05)  # 合成网格无真实 WALL 组，用固定值统一两侧
    dt_local = np.full((n_cells, n_sps), 1e-5)

    h_max_global = h_wn_global = None
    if turb_model_name == "IDDES":
        h_max_global, h_wn_global = compute_h_max_and_h_wn(mesh)

    # ---- 单机参考：直接复用 compute_turbulence_source ----
    from types import SimpleNamespace
    from autoflowcfd.core.fr_residual.inviscid import conserved_to_primitive
    from autoflowcfd.core.fr_solver.turbulence import compute_turbulence_source

    class _FakeSingleMachineSolver:
        def __init__(self):
            self.mesh = mesh
            self.ops = ops
            self.turb_model = turb_ref
            self.turb_model_name = turb_model_name
            self.ddes_model = _make_ddes_model(turb_model_name)
            self._iddes_h_max = h_max_global
            self._iddes_h_wn = h_wn_global
            self.wall_distance = d_wall
            self.mu_molecular = mu
            self.state = SimpleNamespace(
                U=U, Q=conserved_to_primitive(U[..., :5]), n_cells=n_cells, n_sps=n_sps,
            )
            self._turbulence_flat_face_override = None
            # 与分布式参照调用的 turb_ramp_steps=0 一致：不做产生项渐变（没有
            # 计数器时 advance_production_ramp 取真实求解器的 50 步默认）
            self._turb_production_ramp_steps = 0

        def _compute_gradients(self):
            from autoflowcfd.core.fr_operators.gradients import compute_physical_gradient
            return compute_physical_gradient(self.state.U[..., :5], self.mesh, self.ops)

        def _get_cell_volumes(self):
            return self.mesh.cell_volumes

    turb_ref = SSTModelFR(n_cells, n_sps)
    turb_ref.k_field = k_field.copy()
    turb_ref.omega_field = omega_field.copy()
    solver_ref = _FakeSingleMachineSolver()
    compute_turbulence_source(solver_ref, dt_local)
    rho_ref = solver_ref.state.Q[..., 0]
    mu_t_ref = rho_ref * turb_ref.nu_t

    # ---- 分布式路径 ----
    cell_partition = np.array([0, 1, 0, 1], dtype=np.int32)
    fc = mesh.face_connectivity
    partition = build_distributed_partition(fc, cell_partition, rank=rank, n_ranks=2)
    dist_fc = build_distributed_flat_face(mesh, ops, partition, cell_partition=cell_partition)

    n_halo = partition.n_halo
    native_ids = (
        np.concatenate([partition.local_cells, partition.halo_cells])
        if n_halo > 0 else partition.local_cells
    )
    U_local = U[partition.local_cells]
    fake_halo_5var = _FakeHaloExchange(U[native_ids][..., :5])

    k_omega_local = np.stack([k_field[partition.local_cells], omega_field[partition.local_cells]], axis=-1)
    k_omega_extended = np.stack([k_field[native_ids], omega_field[native_ids]], axis=-1)
    fake_halo_turb = _FakeHaloExchange(k_omega_extended)

    turb_local = SSTModelFR(partition.n_local_cells, n_sps)
    turb_local.k_field = k_field[partition.local_cells].copy()
    turb_local.omega_field = omega_field[partition.local_cells].copy()

    # wall_distance 必须是 compact 索引空间（local+halo，"棱柱在前"排列，
    # 与 U_compact/mesh_adapter.n_cells 同一个索引空间）——不是 local-only，
    # 这里直接用 dist_fc.compact_global_ids 从全局 d_wall 切片，等价于
    # compute_distributed_wall_distance 对这个合成网格（无真实 WALL 组，
    # 走特征长度回退分支）会产出的同一份值,因为 d_wall 本来就是常数。
    d_wall_compact = d_wall[dist_fc.compact_global_ids]
    dt_local_local = dt_local[partition.local_cells]

    ddes_model_distributed = _make_ddes_model(turb_model_name)
    iddes_h_max_compact = iddes_h_wn_compact = None
    if turb_model_name == "IDDES":
        iddes_h_max_compact = h_max_global[dist_fc.compact_global_ids]
        iddes_h_wn_compact = h_wn_global[dist_fc.compact_global_ids]

    mu_t_compact, _next_ramp = distributed_compute_turbulence_source_and_viscosity(
        U_local, partition, fake_halo_5var, fake_halo_turb, dist_fc, mesh, ops,
        turb_local, mu, d_wall_compact, dt_local_local,
        turb_model_name=turb_model_name, ddes_model=ddes_model_distributed,
        iddes_h_max_compact=iddes_h_max_compact, iddes_h_wn_compact=iddes_h_wn_compact,
    )

    # 决定性判据：分布式算出的 local cells k/omega/nu_t 更新结果，必须
    # 与单机路径对同一批全局单元的结果逐位一致。
    expected_k = turb_ref.k_field[partition.local_cells]
    expected_omega = turb_ref.omega_field[partition.local_cells]
    expected_nu_t = turb_ref.nu_t[partition.local_cells]

    np.testing.assert_allclose(turb_local.k_field, expected_k, rtol=1e-10, atol=1e-14)
    np.testing.assert_allclose(turb_local.omega_field, expected_omega, rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(turb_local.nu_t, expected_nu_t, rtol=1e-10, atol=1e-16)

    # mu_t_field_compact 是 compact 索引空间；换回 local 顺序后与参照比对。
    mu_t_native = mu_t_compact[dist_fc.inv_perm]
    mu_t_local = mu_t_native[:partition.n_local_cells]
    expected_mu_t_local = mu_t_ref[partition.local_cells]
    np.testing.assert_allclose(mu_t_local, expected_mu_t_local, rtol=1e-10, atol=1e-16)


@pytest.mark.parametrize("rank", [0, 1])
def test_distributed_les_viscosity_matches_single_machine(rank):
    """CPU MPI 分布式 LES（WALE）端到端验证（2026-09-02 续接，见
    `distributed_compute_les_viscosity` 文档）——与上面 SST/DDES/IDDES
    同一个判据：分布式路径算出的 local cells `mu_t` 必须与单机路径
    对同一批全局单元的结果逐位一致。WALE 是纯代数模型，不需要跨步
    状态，判据因此比 SST 系列更简单：不需要构造/写回任何持久状态，
    只需要比对一次调用的输出。"""
    from autoflowcfd.core.mpi.distributed_turbulence import distributed_compute_les_viscosity
    from autoflowcfd.core.turbulence.sgs import WALEModel
    from autoflowcfd.core.fr_residual.viscous import compute_gradients

    order = 1
    mesh = _build_synthetic_mixed_mesh(order)
    ops = generate_fr_operators(order)
    n_cells, n_sps = mesh.n_cells, mesh.n_sps_per_cell
    rng = np.random.default_rng(2024)
    U, _k, _omega = _nonuniform_state(mesh, rng)

    # 单机参照：与 apply_turbulence_corrections（fr_solver/turbulence.py）
    # 同一套计算——grad_vel、delta=|V|^(1/3)、WALEModel.compute_eddy_viscosity。
    grad_U_ref = compute_gradients(U, ops, mesh)
    grad_vel_ref = grad_U_ref[:, :, 1:4, :]
    delta_ref = np.power(np.abs(mesh.get_all_cell_volumes()), 1.0 / 3.0)
    delta_ref = np.tile(delta_ref[:, None], (1, n_sps))
    sgs_ref = WALEModel()
    nu_t_ref = sgs_ref.compute_eddy_viscosity(grad_vel_ref, delta_ref)
    from autoflowcfd.core.fr_residual.inviscid import conserved_to_primitive
    rho_ref = conserved_to_primitive(U[..., :5])[..., 0]
    mu_t_ref = rho_ref * nu_t_ref

    cell_partition = np.array([0, 1, 0, 1], dtype=np.int32)
    fc = mesh.face_connectivity
    partition = build_distributed_partition(fc, cell_partition, rank=rank, n_ranks=2)
    dist_fc = build_distributed_flat_face(mesh, ops, partition, cell_partition=cell_partition)

    n_halo = partition.n_halo
    native_ids = (
        np.concatenate([partition.local_cells, partition.halo_cells])
        if n_halo > 0 else partition.local_cells
    )
    fake_halo = _FakeHaloExchange(U[native_ids])
    U_local = U[partition.local_cells]

    sgs_model = WALEModel()
    mu_t_compact = distributed_compute_les_viscosity(
        U_local, partition, fake_halo, dist_fc, mesh, ops, sgs_model,
    )

    mu_t_native = mu_t_compact[dist_fc.inv_perm]
    mu_t_local = mu_t_native[:partition.n_local_cells]
    expected_local = mu_t_ref[partition.local_cells]
    np.testing.assert_allclose(mu_t_local, expected_local, rtol=1e-10, atol=1e-16)


@pytest.mark.parametrize("turb_model_name", ["DDES", "IDDES"])
def test_distributed_ddes_two_consecutive_steps_matches_single_machine(turb_model_name):
    """真实 bug 回归测试（2026-09-02，用两次连续调用才测出来——第一次
    调用之前从未被本模块任何测试覆盖过）：`des_length_scale` 是跨步
    持久状态，此前 `distributed_compute_turbulence_source_and_
    viscosity` 把它（n_local 大小）直接 setattr 到 compact 大小的临时
    视图上，不做任何 halo 交换/重排——第一次调用时 `des_length_scale`
    还是 None，问题被掩盖；第二次调用（真实 DDES/IDDES 生产运行的
    第二个物理步）必现 `ValueError: operands could not be broadcast
    together`（(n_local,n_sps) vs (n_compact,n_sps)）。修复为与
    k_field/omega_field 完全同一套"halo 交换+compact 重排"处理。

    本测试真正模拟两个 rank 之间的 des_length_scale 交换（不是像上面
    单步测试那样从已知全局真值直接切片——第 2 步的"真值"本身就是第
    1 步两个 rank 各自算出来、需要真正交换后才能拼出的结果），对照
    单机路径连续两步的结果。
    """
    order = 1
    mesh = _build_synthetic_mixed_mesh(order)
    ops = generate_fr_operators(order)
    n_cells, n_sps = mesh.n_cells, mesh.n_sps_per_cell
    mu = 1.8e-5
    dt = 1e-3  # 见 TestDdesModelDistinctFromSst 同一处放大注释
    rng = np.random.default_rng(4242)

    U, k_field, omega_field = _nonuniform_state(mesh, rng)
    d_wall = np.full((n_cells, n_sps), 0.05)
    dt_local_global = np.full((n_cells, n_sps), dt)

    h_max_global, h_wn_global = compute_h_max_and_h_wn(mesh)

    # ---- 单机参考：连续两步 ----
    from types import SimpleNamespace
    from autoflowcfd.core.fr_residual.inviscid import conserved_to_primitive
    from autoflowcfd.core.fr_solver.turbulence import compute_turbulence_source

    turb_ref = SSTModelFR(n_cells, n_sps)
    turb_ref.k_field = k_field.copy()
    turb_ref.omega_field = omega_field.copy()

    class _RefSolver:
        def __init__(self_inner):
            self_inner.mesh = mesh
            self_inner.ops = ops
            self_inner.turb_model = turb_ref
            self_inner.turb_model_name = turb_model_name
            self_inner.ddes_model = _make_ddes_model(turb_model_name)
            self_inner._iddes_h_max = h_max_global
            self_inner._iddes_h_wn = h_wn_global
            self_inner.wall_distance = d_wall
            self_inner.mu_molecular = mu
            self_inner.state = SimpleNamespace(
                U=U, Q=conserved_to_primitive(U[..., :5]), n_cells=n_cells, n_sps=n_sps,
            )
            self_inner._turbulence_flat_face_override = None
            # 与分布式参照调用的 turb_ramp_steps=0 一致：不做产生项渐变（没有
            # 计数器时 advance_production_ramp 取真实求解器的 50 步默认）
            self_inner._turb_production_ramp_steps = 0

        def _compute_gradients(self_inner):
            from autoflowcfd.core.fr_operators.gradients import compute_physical_gradient
            return compute_physical_gradient(U[..., :5], mesh, ops)

        def _get_cell_volumes(self_inner):
            return mesh.cell_volumes

    solver_ref = _RefSolver()
    compute_turbulence_source(solver_ref, dt_local_global)
    compute_turbulence_source(solver_ref, dt_local_global)

    # ---- 分布式路径：两个 rank 各自持有一份 turb_local，两步之间真正
    # 交换 U/k/omega/des_length_scale（从两个 rank 各自的最新结果拼出
    # "全局"数组，模拟真实 halo 交换会拿到的邻居数据）----
    cell_partition = np.array([0, 1, 0, 1], dtype=np.int32)
    fc = mesh.face_connectivity
    partitions = {r: build_distributed_partition(fc, cell_partition, rank=r, n_ranks=2) for r in (0, 1)}
    dist_fcs = {r: build_distributed_flat_face(mesh, ops, partitions[r], cell_partition=cell_partition) for r in (0, 1)}

    turb_locals = {}
    for r in (0, 1):
        t = SSTModelFR(partitions[r].n_local_cells, n_sps)
        t.k_field = k_field[partitions[r].local_cells].copy()
        t.omega_field = omega_field[partitions[r].local_cells].copy()
        turb_locals[r] = t

    # "全局"des_length_scale 数组（None 表示还没有任何 rank 算出来过），
    # 每步结束后用两个 rank 各自的 turb_local.des_length_scale 更新。
    des_length_scale_global = None

    for step in range(2):
        results = {}
        # 关键：k_omega_global 必须在本步"两个 rank 都还没更新"这个
        # 快照时刻算一次，不能放进下面的 per-rank 循环里——那样在处理
        # rank=1 时会读到 rank=0 在*本步*已经算完的新值，而不是上一步
        # 结束时的值，制造一个测试自身引入的虚假竞态（不是被测代码的
        # bug）。
        if step == 0:
            k_omega_global = np.stack([k_field, omega_field], axis=-1)
        else:
            k_omega_global = np.stack([
                _scatter_local(mesh, turb_locals, partitions, "k_field"),
                _scatter_local(mesh, turb_locals, partitions, "omega_field"),
            ], axis=-1)

        for r in (0, 1):
            partition = partitions[r]
            dist_fc = dist_fcs[r]
            n_halo = partition.n_halo
            native_ids = (
                np.concatenate([partition.local_cells, partition.halo_cells])
                if n_halo > 0 else partition.local_cells
            )
            U_local = U[partition.local_cells]
            fake_halo_5var = _FakeHaloExchange(U[native_ids][..., :5])

            turb = turb_locals[r]
            k_omega_extended = k_omega_global[native_ids]
            fake_halo_turb = _FakeHaloExchange(k_omega_extended)

            d_wall_compact = d_wall[dist_fc.compact_global_ids]
            dt_local_local = dt_local_global[partition.local_cells]

            des_halo = None
            if des_length_scale_global is not None:
                des_ext = des_length_scale_global[native_ids][:, :, None]
                des_halo = _FakeHaloExchange(des_ext)

            ddes_model = _make_ddes_model(turb_model_name)
            iddes_h_max_compact = iddes_h_wn_compact = None
            if turb_model_name == "IDDES":
                iddes_h_max_compact = h_max_global[dist_fc.compact_global_ids]
                iddes_h_wn_compact = h_wn_global[dist_fc.compact_global_ids]

            distributed_compute_turbulence_source_and_viscosity(
                U_local, partition, fake_halo_5var, fake_halo_turb, dist_fc, mesh, ops,
                turb, mu, d_wall_compact, dt_local_local,
                turb_model_name=turb_model_name, ddes_model=ddes_model,
                iddes_h_max_compact=iddes_h_max_compact, iddes_h_wn_compact=iddes_h_wn_compact,
                des_length_scale_halo_exchange=des_halo,
            )
            results[r] = turb

        # 用本步两个 rank 的结果拼出下一步要用的"全局" des_length_scale。
        des_length_scale_global = np.zeros((n_cells, n_sps))
        for r in (0, 1):
            des_length_scale_global[partitions[r].local_cells] = results[r].des_length_scale

    # 决定性判据：两步之后，两个 rank 各自的 k/omega/nu_t 必须与单机
    # 路径连续两步的结果逐位一致。
    for r in (0, 1):
        turb = turb_locals[r]
        local_cells = partitions[r].local_cells
        np.testing.assert_allclose(turb.k_field, turb_ref.k_field[local_cells], rtol=1e-9, atol=1e-12)
        np.testing.assert_allclose(turb.omega_field, turb_ref.omega_field[local_cells], rtol=1e-9, atol=1e-8)
        np.testing.assert_allclose(turb.nu_t, turb_ref.nu_t[local_cells], rtol=1e-9, atol=1e-14)


def _scatter_local(mesh, turb_locals, partitions, field_name):
    """把两个 rank 各自的 local 结果拼回一个"全局"数组，供下一步构造
    fake halo 用（模拟真实 halo 交换会拿到的、邻居 rank 最新算出的
    数据）。"""
    n_cells, n_sps = mesh.n_cells, mesh.n_sps_per_cell
    out = np.zeros((n_cells, n_sps))
    for r in (0, 1):
        out[partitions[r].local_cells] = getattr(turb_locals[r], field_name)
    return out


class TestDdesModelDistinctFromSst:
    """负向对照：确认 DDES/IDDES 真的改变了结果（不是恰好和纯 SST
    数值上一致，让上面的一致性测试对 DDES/IDDES 分支形同虚设）。"""

    def test_ddes_nu_t_differs_from_plain_sst(self):
        order = 1
        mesh = _build_synthetic_mixed_mesh(order)
        n_cells, n_sps = mesh.n_cells, mesh.n_sps_per_cell
        mu = 1.8e-5
        rng = np.random.default_rng(4242)
        U, k_field, omega_field = _nonuniform_state(mesh, rng)
        d_wall = np.full((n_cells, n_sps), 0.05)
        # 用比其余测试更大的 dt（1e-3 而不是 1e-5）放大两步之间的分叉幅度
        # ——第一步 SST/DDES/IDDES 恰好数值相同（见下方注释），只有第二步
        # 才会分叉，分叉幅度正比于 dt，1e-5 下差异被 np.allclose 默认
        # 容差盖过（真实量级约 1e-7 相对差），1e-3 下能有清晰、不依赖
        # np.allclose 容差选择的可见差异。
        dt_local = np.full((n_cells, n_sps), 1e-3)

        from types import SimpleNamespace
        from autoflowcfd.core.fr_residual.inviscid import conserved_to_primitive
        from autoflowcfd.core.fr_solver.turbulence import compute_turbulence_source
        from autoflowcfd.fr.operators import generate_fr_operators as _gen_ops
        ops = _gen_ops(order)

        def _run(turb_model_name):
            turb = SSTModelFR(n_cells, n_sps)
            turb.k_field = k_field.copy()
            turb.omega_field = omega_field.copy()

            class _Solver:
                def __init__(self_inner):
                    self_inner.mesh = mesh
                    self_inner.ops = ops
                    self_inner.turb_model = turb
                    self_inner.turb_model_name = turb_model_name
                    self_inner.ddes_model = _make_ddes_model(turb_model_name)
                    if turb_model_name == "IDDES":
                        self_inner._iddes_h_max, self_inner._iddes_h_wn = compute_h_max_and_h_wn(mesh)
                    self_inner.wall_distance = d_wall
                    self_inner.mu_molecular = mu
                    self_inner.state = SimpleNamespace(
                        U=U, Q=conserved_to_primitive(U[..., :5]), n_cells=n_cells, n_sps=n_sps,
                    )
                    self_inner._turbulence_flat_face_override = None
                    # 与分布式参照调用的 turb_ramp_steps=0 一致：不做产生项渐变（没有
                    # 计数器时 advance_production_ramp 取真实求解器的 50 步默认）
                    self_inner._turb_production_ramp_steps = 0

                def _compute_gradients(self_inner):
                    from autoflowcfd.core.fr_operators.gradients import compute_physical_gradient
                    return compute_physical_gradient(U[..., :5], mesh, ops)

                def _get_cell_volumes(self_inner):
                    return mesh.cell_volumes

            solver = _Solver()
            # 两步：DDES/IDDES 的 des_length_scale 是"慢一拍"的跨步状态
            # （第一步 apply_to_sst_model 用 nu_t 算出 l_eff，供*下一步*
            # compute_source_terms 读取，见 compute_turbulence_source
            # 里"DDES 的有效长度尺度"一段注释）——第一步 des_length_scale
            # 还是初始的 None，D_k 对三个模型都退化成标准 RANS 公式，
            # Sk/S_omega（进而 nu_t）在第一步结束时对三者恰好相同；第二步
            # compute_source_terms 才会真正读到（DDES/IDDES 各自不同的）
            # des_length_scale 算出不同的 D_k，改变 Sk，进而在第二步
            # update_fields 之后让 k_field 分叉——只跑一步看不出区别，
            # 不是本测试的判据。
            compute_turbulence_source(solver, dt_local)
            compute_turbulence_source(solver, dt_local)
            return turb.k_field

        k_sst = _run("SST")
        k_ddes = _run("DDES")
        k_iddes = _run("IDDES")

        # 用绝对差值直接判定（不用 np.allclose 默认容差——两步之间的
        # 分叉幅度本来就小，容差选择不当会掩盖真实差异，见上方 dt 注释），
        # 阈值 1e-6 远高于浮点噪声、远低于观测到的真实差异量级。
        assert np.max(np.abs(k_sst - k_ddes)) > 1e-6, (
            "DDES 的长度尺度替换应该真正改变 k_field 的演化，不应该和纯 SST 数值一致"
        )
        assert np.max(np.abs(k_sst - k_iddes)) > 1e-6, (
            "IDDES 的长度尺度替换应该真正改变 k_field 的演化，不应该和纯 SST 数值一致"
        )
        assert np.max(np.abs(k_ddes - k_iddes)) > 1e-6, (
            "DDES 和 IDDES 的长度尺度公式结构不同，不应该恰好数值一致"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
