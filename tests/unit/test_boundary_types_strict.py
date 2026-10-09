"""边界组的条件类型必须明确（2026-10-09，`core/fr_solver/boundary/ghost.py`）。

此前组在 `boundary_bc_types` 里缺类型、或类型不认识时，一律静默当成远场。几个测试就是这样把上下壁面当成远场跑了
很久（组名写成 y_min/y_max，而网格里叫 wall_bottom/wall_top）都没人发现。
"""

import types

import pytest

from autoflowcfd.core.fr_solver.boundary import build_boundary_ghost_provider
from autoflowcfd.fr.operators import generate_fr_operators
from tests.validation._channel_mesh import build_channel_mesh_prism

FULL = {"x_min": "VELOCITY_INLET", "x_max": "PRESSURE_OUTLET", "wall_bottom": "WALL", "wall_top": "WALL",
        "z_min": "SYMMETRY", "z_max": "SYMMETRY"}


def _provider(bc_types, bc_overrides=None):
    mesh = build_channel_mesh_prism(1, 2, 2, 2, 0.4, 0.1, 0.08)
    mesh.boundary_bc_types = dict(bc_types)
    stub = types.SimpleNamespace(
        mesh=mesh, ops=generate_fr_operators(1), turb_model_name="NONE", wmles_model=None,
        freestream={"rho_inf": 1.225, "vel_inf": 10.0, "p_inf": 101325.0},
        _sem_num_eddies=40, _turbulence_intensity=0.01)
    return build_boundary_ghost_provider(stub, bc_overrides=bc_overrides or {})


def test_complete_types_build():
    assert _provider(FULL) is not None


def test_group_without_a_type_is_an_error():
    partial = {k: v for k, v in FULL.items() if k != "wall_top"}
    with pytest.raises(ValueError, match="wall_top"):
        _provider(partial)


def test_explicit_override_supplies_the_missing_type():
    partial = {k: v for k, v in FULL.items() if k != "wall_top"}
    assert _provider(partial, bc_overrides={"wall_top": {"type": "SYMMETRY"}}) is not None


def test_unknown_type_is_an_error():
    with pytest.raises(ValueError, match="不支持"):
        _provider({**FULL, "z_min": "SYMETRY"})


def test_unpaired_periodic_faces_are_reported():
    with pytest.raises(ValueError, match="未配对"):
        _provider({**FULL, "z_min": "PERIODIC"})


def test_boundary_face_without_a_group_is_an_error():
    """没有组的边界面此前静默当成远场（网格内部缺口的面因此在流场内部施加来流条件）。"""
    mesh = build_channel_mesh_prism(1, 4, 4, 2, 0.4, 0.1, 0.08)     # 有只接触 z_max 的单元
    mesh.boundary_bc_types = dict(FULL)
    mesh.boundary_groups = {k: v for k, v in mesh.boundary_groups.items() if k != "z_max"}
    stub = types.SimpleNamespace(
        mesh=mesh, ops=generate_fr_operators(1), turb_model_name="NONE", wmles_model=None,
        freestream={"rho_inf": 1.225, "vel_inf": 10.0, "p_inf": 101325.0},
        _sem_num_eddies=40, _turbulence_intensity=0.01)
    with pytest.raises(ValueError, match="不属于任何边界组"):
        build_boundary_ghost_provider(stub, bc_overrides={})


def test_paired_periodic_groups_need_no_ghost_state_and_do_not_claim_other_faces():
    """周期面配对后是内部面：单元级标记不含周期组（否则同时贴着周期面与别的边界的角点单元，其余边界面会被
    标成周期组——此前这些面静默按远场处理），求解器对它们不建幽灵态。"""
    from autoflowcfd.grid.connectivity.face_connectivity_boundary_tags import tag_boundary_groups_for_mesh
    from tests.validation._periodic_mesh import build_periodic_channel_mesh_x

    mesh = build_periodic_channel_mesh_x(1, nx=3, ny=2, nz=1, Lx=1.0, H=1.0, Lz=0.3)
    fc = mesh.face_connectivity
    group_code, name_to_code = tag_boundary_groups_for_mesh(mesh, fc)
    assert "x_min" not in name_to_code and "x_max" not in name_to_code
    assert (group_code[fc.get_boundary_face_indices()] >= 0).all()

    stub = types.SimpleNamespace(
        mesh=mesh, ops=generate_fr_operators(1), turb_model_name="NONE", wmles_model=None,
        freestream={"rho_inf": 1.225, "vel_inf": 10.0, "p_inf": 101325.0},
        _sem_num_eddies=40, _turbulence_intensity=0.01)
    provider = build_boundary_ghost_provider(stub, bc_overrides={})
    assert {cfg["type"] for cfg in provider.code_to_config.values()} == {"SYMMETRY"}
