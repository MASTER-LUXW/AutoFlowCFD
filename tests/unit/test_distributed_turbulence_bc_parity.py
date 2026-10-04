# -*- coding: utf-8 -*-
"""CPU 分布式（n_ranks=1）与单机在**带边界条件的湍流**算例上逐步一致。

2026-09-25 查出的真实缺陷：分布式湍流适配器（`distributed_turbulence.py::
DistributedTurbulenceSolverAdapter`）没有 `boundary_ghost_provider`，于是
湍流输运里一切按边界组类型取的条件——壁面 k=0、omega 的 Wilcox 壁面值
（对流 ghost 与显式壁面松弛）、来流条件——在分布式路径上全部静默退化成
零梯度。此前"分布式 == 单机"的对照测试只覆盖层流（合成网格、无边界组），
所以一直没有暴露。补上 provider 后本测试又查出第二处：分布式 compact 视图
（`build_distributed_turbulence_view`）按构造默认值建 `SSTModelFR`，
k_inf/omega_inf 是 1e-6/1.0，而来流 ghost 取的正是它们——来流面单元的
k/omega 第一步就与单机差 2e-4。

判据：通道（上下无滑移壁、两端远场、展向对称面），SST 用棱柱、SA-neg 用四面体（解点含落在壁面上
的顶点/棱点，覆盖其强 Dirichlet），同一初场、同一壁距来源（经两侧各自的生产入口施加），n_ranks=1 的
分布式 `step()` 与单机 `step()` 逐步一致：平均流与湍流输运场、涡粘一致到浮点重结合噪声（分布式按
compact 子集重排计算，算术顺序不同，见 `test_distributed_solver_main_init.py` 的容差说明）。
n_ranks=1 没有 halo 也没有跨 rank 通信，任何差异只能来自分布式实现自己。
"""

import numpy as np
import pytest

from tests.validation._channel_mesh import build_channel_mesh, build_channel_mesh_prism, channel_wall_source

RHO, U_INF, P, GAMMA = 1.225, 30.0, 101325.0, 1.4
LX, H, LZ = 0.4, 0.1, 0.08


def _bc():
    bc = {n: {"type": "SYMMETRY"} for n in ("z_min", "z_max")}
    for n in ("wall_bottom", "wall_top"):
        bc[n] = {"type": "WALL", "is_no_slip": True, "wall_velocity": [0.0, 0.0, 0.0]}
    for n in ("x_min", "x_max"):
        bc[n] = {"type": "FARFIELD", "Q_free": [RHO, U_INF, 0.0, 0.0, P]}
    return bc


def _uniform_U(n_cells, n_sps, n_vars):
    U = np.zeros((n_cells, n_sps, n_vars))
    U[..., 0] = RHO
    U[..., 1] = RHO * U_INF
    U[..., 4] = P / (GAMMA - 1.0) + 0.5 * RHO * U_INF ** 2
    return U


_KW = dict(mu_molecular=1.8e-5, rho_inf=RHO, vel_inf=U_INF, p_inf=P)


def _mesh(model, order):
    build = build_channel_mesh if model == "SA" else build_channel_mesh_prism
    return build(order, 3, 4, 2, LX, H, LZ)


def _single(scheme, order, model="SST", mesh=None):
    """单机求解器（均匀来流初场，壁距经生产入口由精确来源施加）。"""
    from autoflowcfd.core.fr_solver.solver import FRSolver
    from autoflowcfd.core.fr_solver.turbulence import apply_wall_distance_source
    from autoflowcfd.core.turbulence.registry import n_state_vars

    mesh = mesh or _mesh(model, order)
    single = FRSolver(mesh, order=order, turb_model_name=model, n_vars=n_state_vars(model), time_scheme=scheme,
                      bc_overrides=_bc(), **_KW)
    single.order_continuation_enabled = False
    single.state.U[...] = _uniform_U(*single.state.U.shape)
    single.state._update_primitives()
    apply_wall_distance_source(single, channel_wall_source(LX, H, LZ))
    return single


def _pair(scheme, order=1, model="SST"):
    from autoflowcfd.core.fr_residual.inviscid import conserved_to_primitive
    from autoflowcfd.core.mpi.distributed_solver import DistributedFRSolver
    from autoflowcfd.fr.operators import generate_fr_operators

    mesh = _mesh(model, order)
    ops = generate_fr_operators(order)
    single = _single(scheme, order, model, mesh)
    n_cells = single.state.U.shape[0]
    U0 = single.state.U

    dist = DistributedFRSolver(
        mesh=mesh, ops=ops, face_connectivity=mesh.face_connectivity, n_ranks=1,
        order=order, turb_model_name=model, n_vars=5, time_scheme=scheme,
        wall_distance_source=channel_wall_source(LX, H, LZ), bc_overrides=_bc(), **_KW)
    # 分布式状态只存平均流 5 变量（湍流场由湍流模型持有，见 DistributedFRState）
    dist.state.U[:n_cells] = U0[..., :5]
    dist.state.Q[:n_cells] = conserved_to_primitive(U0[..., :5])
    return single, dist


def _rel_diffs(ref, other):
    """`other` 相对单机 `ref` 的逐量最大相对差：`{"平均流", 各输运场, "nu_t"}`。"""
    n = ref.state.U.shape[0]
    got, exp = other.state.U[:n, :, :5], ref.state.U[..., :5]
    # 动量三分量共用动量模的尺度：均匀来流里 rho_v/rho_w 本身只有舍入量级，
    # 按各自最大值归一会把 1e-14 的重结合噪声放大成 1e-6
    scale = np.abs(exp).max(axis=(0, 1))
    scale[1:4] = np.abs(exp[..., 1:4]).max()
    out = {"平均流": float((np.abs(got - exp) / scale).max())}
    for name in tuple(ref.turb_model.TRANSPORTED_FIELDS) + ("nu_t",):
        a = np.asarray(getattr(other.turb_model, name))
        b = np.asarray(getattr(ref.turb_model, name))
        out[name] = float(np.abs(a - b).max() / max(np.abs(b).max(), 1e-300))
    return out


def _assert_same(single, dist, what, tol=1e-9):
    for name, r in _rel_diffs(single, dist).items():
        assert r <= tol, f"{what}：{name} 与单机不一致（相对 {r:.3e}，容差 {tol:.1e}）"


@pytest.mark.parametrize("model", ["SST", "SA"])
def test_explicit_turbulence_with_boundary_conditions_matches_single_machine(model):
    from autoflowcfd.core.time_integration.base import TimeIntegrationScheme

    single, dist = _pair(TimeIntegrationScheme.SSP_RK3, model=model)
    for k in range(2):
        single.step(1e-6)
        dist.step(1e-6)
        _assert_same(single, dist, f"显式 {model} 第 {k + 1} 步")


@pytest.mark.parametrize("model,order", [("SST", 0), ("SST", 1), ("SA", 0), ("SA", 1)])
def test_newton_krylov_turbulence_matches_single_machine(model, order):
    """隐式稳态（平均流与 k-omega 紧耦合 NK）：分布式与单机同一个算法，只换了
    归约对象、块 Jacobi 着色来源与湍流求值的 compact 视图。n_ranks=1 下三者
    都应退化为单机行为，差异只能来自浮点重结合。P0 另外覆盖差分装配截取耦合块
    的块 ILU（分布式耦合图 `distributed_coupling_graph` 与单机同一结构）。"""
    from autoflowcfd.core.time_integration.base import TimeIntegrationScheme

    single, dist = _pair(TimeIntegrationScheme.NEWTON_KRYLOV, order, model)
    # 容差按单机**自身**对舍入量级扰动的敏感度定：初场乘 (1 + 1e-15 xi) 的孪生单机。
    # P0 的湍流有了产生项之后（`fr_operators/corrected_gradient.py`），差分装配的块
    # 预处理把残差的舍入差异按 1/h 放大、GMRES 只走 1 次时直接进入 Newton 更新：孪生
    # 单机第 1/2/3 步 k 相对差 8.7e-9 / 2.6e-8 / 1.2e-7（此前 P0 产生项恒为零、k 恒为
    # 来流值，同一量只有 1e-13）。分布式与单机的差异只来自重结合时，必然落在这个
    # 敏感度的同一量级；判据取其 10 倍、下限 1e-9。
    twin = _single(TimeIntegrationScheme.NEWTON_KRYLOV, order, model)
    rng = np.random.default_rng(0)
    twin.state.U[...] *= 1.0 + 1e-15 * rng.standard_normal(twin.state.U.shape)
    twin.state._update_primitives()
    for k in range(3):
        single.step(1e-6)
        dist.step(1e-6)
        twin.step(1e-6)
        a, b = dist._newton_last_info, single._newton_last_info
        assert a["gmres_iters"] == b["gmres_iters"] and a["theta"] == b["theta"], (
            f"第 {k + 1} 步 Newton 轨迹不同：分布式 {a}，单机 {b}")
        tol = max(1e-9, 10.0 * max(_rel_diffs(single, twin).values()))
        _assert_same(single, dist, f"NK {model} 第 {k + 1} 步", tol)
    # CFL 由残差范数之比推出，继承状态的重结合差异：参照取孪生单机的状态偏差与 CFL 偏差中
    # 较大者的 10 倍（下限 1e-9）。只拿孪生的 CFL 偏差作参照会随线程数（并行归约顺序）偶然
    # 偏小：4 线程时孪生 CFL 只差 1.6e-8，而孪生状态差 1.2e-7、分布式 CFL 差 2.5e-7
    cfl = single._cfl_controller.cfl_number
    twin_dev = max(abs(twin._cfl_controller.cfl_number - cfl) / cfl, max(_rel_diffs(single, twin).values()))
    cfl_tol = max(1e-9, 10.0 * twin_dev)
    assert dist._cfl_controller.cfl_number == pytest.approx(cfl, rel=cfl_tol)
    if order == 0:
        assert single._newton_block_precond.coupling is not None
        assert dist._newton_block_precond.coupling is not None
