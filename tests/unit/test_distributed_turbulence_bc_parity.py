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

判据：棱柱通道（上下无滑移壁、两端远场、展向对称面）+ SST，同一初场、
同一解析壁面距离，n_ranks=1 的分布式 `step()` 与单机 `step()` 各走两步，
平均流与 k/omega/nu_t 必须一致到浮点重结合噪声（分布式按 compact 子集
重排计算，算术顺序不同，见 `test_distributed_solver_main_init.py` 的
容差说明）。n_ranks=1 没有 halo 也没有跨 rank 通信，任何差异只能来自分布式
实现自己。
"""

import numpy as np
import pytest

from tests.unit._wall_source import synthetic_wall_source
from tests.validation._channel_mesh import build_channel_mesh_prism

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


def _pair(scheme):
    from autoflowcfd.core.fr_residual.inviscid import conserved_to_primitive
    from autoflowcfd.core.fr_solver.solver import FRSolver
    from autoflowcfd.core.mpi.distributed_solver import DistributedFRSolver
    from autoflowcfd.fr.operators import generate_fr_operators

    mesh = build_channel_mesh_prism(1, nx=3, ny=4, nz=2, Lx=LX, H=H, Lz=LZ)
    ops = generate_fr_operators(1)
    kw = dict(mu_molecular=1.8e-5, rho_inf=RHO, vel_inf=U_INF, p_inf=P, bc_overrides=_bc())
    single = FRSolver(mesh, order=1, turb_model_name="SST", n_vars=7, time_scheme=scheme, **kw)
    single.order_continuation_enabled = False
    n_cells, n_sps, n_vars = single.state.U.shape
    U0 = _uniform_U(n_cells, n_sps, n_vars)
    single.state.U[...] = U0
    single.state._update_primitives()

    # 构造需要一个壁面距离来源（没有就报错）；随后两侧都换成同一个解析壁距
    dist = DistributedFRSolver(
        mesh=mesh, ops=ops, face_connectivity=mesh.face_connectivity, n_ranks=1,
        backend="cpu", order=1, turb_model_name="SST", n_vars=5, time_scheme=scheme,
        wall_distance_source=synthetic_wall_source(mesh), **kw)
    # 分布式状态只存平均流 5 变量（k/omega 由湍流模型持有，见 DistributedFRState）
    dist.state.U[:n_cells] = U0[..., :5]
    dist.state.Q[:n_cells] = conserved_to_primitive(U0[..., :5])

    y = np.asarray(mesh.sps_coords)[..., 1]
    d = np.ascontiguousarray(np.maximum(np.minimum(y, H - y), 1e-12))
    single.wall_distance = d
    dist.wall_distance_compact = d[dist.dist_flat_face.compact_global_ids]
    return single, dist


def _assert_same(single, dist, what):
    n = single.state.U.shape[0]
    got, exp = dist.state.U[:n, :, :5], single.state.U[..., :5]
    # 动量三分量共用动量模的尺度：均匀来流里 rho_v/rho_w 本身只有舍入量级，
    # 按各自最大值归一会把 1e-14 的重结合噪声放大成 1e-6
    scale = np.abs(exp).max(axis=(0, 1))
    scale[1:4] = np.abs(exp[..., 1:4]).max()
    rel = (np.abs(got - exp) / scale).max()
    assert rel <= 1e-9, f"{what}：平均流与单机不一致（相对 {rel:.3e}）"
    for name in ("k_field", "omega_field", "nu_t"):
        a = np.asarray(getattr(dist.turb_model, name))
        b = np.asarray(getattr(single.turb_model, name))
        r = np.abs(a - b).max() / max(np.abs(b).max(), 1e-300)
        assert r <= 1e-9, f"{what}：{name} 与单机不一致（相对 {r:.3e}）"


def test_explicit_sst_with_boundary_conditions_matches_single_machine():
    from autoflowcfd.core.time_integration.base import TimeIntegrationScheme

    single, dist = _pair(TimeIntegrationScheme.SSP_RK3)
    for k in range(2):
        single.step(1e-6)
        dist.step(1e-6)
        _assert_same(single, dist, f"显式 SST 第 {k + 1} 步")
