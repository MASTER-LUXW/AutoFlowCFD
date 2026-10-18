"""通量点物理坐标的唯一来源（I-180）与分布式 SEM 入口的面号空间（I-105）。

I-180：P0 每个面记录只有一个通量点，解点只有一个（单元中心），此前用解点外插求通量点坐标——壁面力的力臂、
SEM 入口采样坐标、WMLES 匹配点高度在 P0 下全部落在单元中心。现在 P0 位置在构造面几何时由网格节点算出
（拆分四边形的两条子面记录都取四边形中心）。

I-105：分布式路径把全局 provider 限制到本 rank 面序时只重切了组编码，SEM 入口坐标表仍以全局面号为键——本 rank
面号要么查不到（静默返回无脉动的均值态）、要么查到别处的坐标。
"""

import types

import numpy as np
import pytest

from autoflowcfd.core.fr_operators.face_kernels import get_flat_face_geometry
from autoflowcfd.core.fr_operators.face_points import flux_point_positions
from tests.unit._boundary_groups import tag_wall_cells
from tests.unit.test_fr_residual_inviscid import _build_synthetic_mixed_mesh
from tests.validation._channel_mesh import build_channel_mesh_prism


def _all_points(mesh, flat, faces):
    n_fp = int(flat.n_fp)
    xyz = flux_point_positions(mesh, flat, np.repeat(faces, n_fp), np.tile(np.arange(n_fp), faces.size))
    return xyz.reshape(faces.size, n_fp, 3)


@pytest.mark.parametrize("builder", [
    lambda o: build_channel_mesh_prism(o, nx=2, ny=2, nz=1, Lx=1.0, H=1.0, Lz=1.0),
    _build_synthetic_mixed_mesh,
])
def test_p0_point_is_the_face_centroid(builder):
    """P0 通量点 = 面记录所代表的整面（拆分四边形的主记录代表整个四边形）的形心：与 P1 精确通量点按面积权重
    求得的形心一致（这些网格的面都是平行四边形或三角形，形心 = 参考面中心的像）。修复前 P0 得到单元中心。"""
    m0, m1 = builder(0), builder(1)
    f0, f1 = get_flat_face_geometry(m0, m0.operators), get_flat_face_geometry(m1, m1.operators)
    faces = np.flatnonzero(f1.owner_is_primary)
    x1 = _all_points(m1, f1, faces)
    w = f1.true_area_weight[faces]
    centroid = (w[..., None] * x1).sum(axis=1) / w.sum(axis=1)[:, None]
    x0 = _all_points(m0, f0, faces)[:, 0]
    np.testing.assert_allclose(x0, centroid, rtol=0, atol=1e-13)
    cell_center = np.asarray(m0.sps_coords)[f0.owner_cell[faces], 0]
    assert np.min(np.linalg.norm(x0 - cell_center, axis=1)) > 0.1      # 不是单元中心


def test_p0_wall_force_moment_arms_lie_on_the_wall():
    """P0 壁面力的力臂坐标落在壁面上（槽道下壁 y = 0）；修复前是贴壁单元中心。"""
    from autoflowcfd.core.fr_residual.inviscid import primitive_to_conserved, conserved_to_primitive
    from autoflowcfd.postprocess.wall_forces import sample_wall, wall_group_codes
    from tests.validation._channel_mesh import build_face_exact_ghost_provider

    mesh = build_channel_mesh_prism(0, nx=3, ny=3, nz=2, Lx=0.4, H=0.1, Lz=0.08)
    bc = {n: {"type": "SYMMETRY"} for n in ("z_min", "z_max", "x_min", "x_max", "wall_top")}
    bc["wall_bottom"] = {"type": "WALL", "is_no_slip": True, "physical_no_slip": True}
    provider = build_face_exact_ghost_provider(mesh, 0.4, 0.1, 0.08, bc)
    U = np.tile(primitive_to_conserved(np.array([1.2, 10.0, 0.0, 0.0, 1e5])), (mesh.n_cells, 1, 1))
    host = types.SimpleNamespace(
        mesh=mesh, ops=mesh.operators, boundary_ghost_provider=provider, wall_model=None, mu_molecular=1.8e-5,
        freestream={"mach_ref": 0.1, "p_inf": 1e5}, state=types.SimpleNamespace(U=U, Q=conserved_to_primitive(U)),
        _get_turbulent_viscosity_field=lambda: None)
    samples = sample_wall(host, list(wall_group_codes(provider)))
    assert samples.position.shape[0] > 0
    np.testing.assert_allclose(samples.position[:, 1], 0.0, atol=1e-14)


def _sem_provider(mesh):
    from autoflowcfd.core.fr_solver.boundary import build_boundary_ghost_provider

    fc = mesh.face_connectivity
    inlet_cells = np.unique(fc.owner_cell[fc.is_boundary])[:2]
    tag_wall_cells(mesh, inlet_cells, wall_type="VELOCITY_INLET", rest_type="FARFIELD")
    stub = types.SimpleNamespace(mesh=mesh, ops=mesh.operators, turb_model_name="LES", wall_model=None,
                                 freestream={"rho_inf": 1.2, "vel_inf": 20.0, "p_inf": 1e5},
                                 _sem_num_eddies=40, _turbulence_intensity=0.05, _viscosity_ratio=5.0,
                                 mu_molecular=1.8e-5)
    return build_boundary_ghost_provider(stub, bc_overrides={})


@pytest.mark.parametrize("rank", [0, 1])
def test_restricted_provider_reproduces_global_sem_ghosts(rank):
    """本 rank 面序的 provider 在每个入口面上给出与全局 provider 在对应全局面上逐位相同的幽灵态。"""
    from autoflowcfd.boundary.fr_ghost_state import restrict_ghost_provider
    from autoflowcfd.core.mpi.partition import build_distributed_partition

    mesh = _build_synthetic_mixed_mesh(1)
    provider = _sem_provider(mesh)
    partition = build_distributed_partition(mesh.face_connectivity, np.array([0, 1, 0, 1], dtype=np.int32),
                                            rank=rank, n_ranks=2)
    local = restrict_ghost_provider(provider, partition.local_faces)
    inlet_code = [c for c, cfg in provider.code_to_config.items() if cfg.get("sem") is not None]
    assert inlet_code, "测试设置必须有 SEM 入口组"
    local_inlet = np.flatnonzero(np.isin(local.group_code, inlet_code))
    assert local_inlet.size > 0
    flat = get_flat_face_geometry(mesh, mesh.operators)
    rng = np.random.default_rng(rank)
    Q = np.tile(np.array([1.2, 20.0, 0.0, 0.0, 1e5]), (local_inlet.size, flat.n_fp, 1))
    Q[..., 1:4] += rng.normal(scale=0.5, size=Q[..., 1:4].shape)
    global_faces = np.asarray(partition.local_faces)[local_inlet]
    normals = flat.true_normal[global_faces]
    np.testing.assert_array_equal(local.ghost_states(local_inlet, Q, normals),
                                  provider.ghost_states(global_faces, Q, normals))


def test_sem_ghost_rejects_faces_outside_its_table():
    """坐标表里没有的面号说明面号空间错了：报错，不静默返回无脉动的均值态。"""
    mesh = _build_synthetic_mixed_mesh(1)
    provider = _sem_provider(mesh)
    sem = next(cfg["sem"] for cfg in provider.code_to_config.values() if cfg.get("sem") is not None)
    missing = max(sem.face_positions) + 1000
    with pytest.raises(KeyError):
        sem(missing, np.ones((4, 5)), np.tile([1.0, 0.0, 0.0], (4, 1)))
