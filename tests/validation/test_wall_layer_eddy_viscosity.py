"""无滑移壁第一层里涡粘陡升时，壁面上外插出的负涡粘不能把动量注入贴壁层（2026-10-11）。

平板：底面无滑移壁，棱柱沿壁面法向挤出，第一层高 1 mm（y+ 约 100）、长细比 10，来流 30 m/s，SA，P1，隐式稳态。
来流粘性比取 50：第一层内涡粘从壁面的 0 升到约 50 mu，单元内涡粘多项式外插到壁面约 -15 mu。修复前这个算例
**收敛**（残差 6e-5）到贴壁层内最大速度 78.3 m/s 的解；修复后 30.8 m/s。cube_demo 上同一缺陷表现为 P1 稳态
停滞与贴壁单元速度增长到 307 m/s（`flux_kernels/viscous.py::viscous_physical_flux_point` 文档）。
"""

import numpy as np

from autoflowcfd.core.fr_solver import FRSolver
from autoflowcfd.core.fr_solver.turbulence.wall_distance import apply_wall_distance_source
from autoflowcfd.core.time_integration import TimeIntegrationScheme
from autoflowcfd.core.utils.wall_distance import WallDistanceSource, triangles_from_faces
from tests.validation._channel_mesh import build_channel_mesh_prism

U_INF = 30.0
NX, NY, NZ = 20, 20, 2
DX, DY = 1.0e-2, 1.0e-3


def test_velocity_in_the_wall_layer_stays_at_the_freestream_level():
    lx, h, lz = NX * DX, NY * DY, NZ * DX
    mesh = build_channel_mesh_prism(1, NX, NY, NZ, lx, h, lz)
    mesh.boundary_bc_types = {"x_min": "VELOCITY_INLET", "x_max": "PRESSURE_OUTLET", "wall_bottom": "WALL",
                              "wall_top": "SYMMETRY", "z_min": "SYMMETRY", "z_max": "SYMMETRY"}
    solver = FRSolver(mesh=mesh, order=1, turb_model_name="SA", time_scheme=TimeIntegrationScheme.NEWTON_KRYLOV,
                      vel_inf=U_INF, viscosity_ratio=50.0)
    solver.order_continuation_enabled = False
    corners = np.array([[0, 0, 0], [lx, 0, 0], [lx, 0, lz], [0, 0, lz]], dtype=float)
    apply_wall_distance_source(solver, WallDistanceSource(triangles_from_faces(corners, np.array([[0, 1, 2], [0, 2, 3]]))))

    first = None
    for _ in range(90):
        res = solver.step(1.0e-3)
        first = res if first is None else first
    speed = np.linalg.norm(np.asarray(solver.state.Q)[..., 1:4], axis=-1)
    assert res <= 1.0e-6 * first, (first, res)       # 实测 688 -> 5.5e-5（舍入底）；修复前同样收敛——判据在下一行
    assert speed.max() <= 1.1 * U_INF, float(speed.max())                # 修复前 78.3 m/s
    assert np.asarray(solver.turb_model.nu_t).max() >= 30.0 * solver.mu_molecular / 1.225   # 确实是高涡粘工况
