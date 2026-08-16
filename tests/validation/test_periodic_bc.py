"""周期边界条件（PERIODIC）自一致性验证。

背景：`pair_periodic_boundary_faces`（grid/face_connectivity.py）按几何
位置把 x_min/x_max 两组边界面配对合并成周期内部面。真实调试中发现过一个
角点单元候选面误选 bug：`BoundaryMap.groups` 是按 **owner 单元** 记录组
成员关系的（不是按面），角点单元（同时贴周期面和其他侧面的单元）的其他
边界面会被 `np.isin(boundary_owners, ...)` 一并选中、混入候选配对集合
——用一个 24 棱柱的小型合成网格直接复现过（22 个候选面里只有 6 个是
真正的周期面，其余 16 个是被误选的角点单元侧面，KD-tree 匹配距离 0 vs
0.13~0.22，界限清晰）。修复方式：在按单元筛出候选面之后，再用面法向量
与周期平移方向的对齐程度做一次几何二次筛选（平移周期性要求周期面法向量
必须与平移方向平行，这是定义本身而不是启发式阈值）——明显正交的候选面
判定为误选剔除，既不明显平行也不明显正交的候选面直接报错（不做静默
容忍，因为这种情况意味着周期面本身不平直，属于网格缺陷）。

本文件验证修复后的配对机制端到端工作正常：
1. 配对面数与预期一致（ny*nz*2 个棱柱封盖面）；
2. 均匀自由流场残差保持机器精度量级（无粘/粘性残差公式本身没有在
   周期面上引入虚假源项）；
3. 一个沿 x 周期、沿 y 非均匀的真实流场，残差在紧邻周期面的单元与
   其余单元相比没有异常放大的离群值——如果 face_translation 的符号/
   方向搞反了，最典型的失败模式就是紧邻周期面的单元残差异常放大。
"""
import numpy as np

from autoflowcfd.core.fr_solver import FRSolver
from autoflowcfd.core.time_integration import TimeIntegrationScheme

from ._periodic_mesh import build_periodic_channel_mesh_x, build_periodic_symmetry_ghost_provider

H = 1.0
Lz = 0.3
Lx = 1.0
NX, NY, NZ = 4, 3, 1
ORDER = 2
RHO_INF, P_INF, U_INF = 1.225, 101325.0, 20.0
GAMMA = 1.4


def _build_periodic_solver():
    mesh = build_periodic_channel_mesh_x(ORDER, nx=NX, ny=NY, nz=NZ, Lx=Lx, H=H, Lz=Lz)
    bc_overrides = {
        "wall_bottom": {"type": "SYMMETRY"}, "wall_top": {"type": "SYMMETRY"},
        "z_min": {"type": "SYMMETRY"}, "z_max": {"type": "SYMMETRY"},
    }
    solver = FRSolver(
        mesh=mesh, order=ORDER, turb_model_name="NONE", n_vars=5,
        time_scheme=TimeIntegrationScheme.SSP_RK3,
        rho_inf=RHO_INF, vel_inf=U_INF, p_inf=P_INF,
        bc_overrides=bc_overrides,
    )
    solver.order_continuation_enabled = False
    solver.boundary_ghost_provider = build_periodic_symmetry_ghost_provider(mesh, Lx, H, Lz)
    return solver, mesh


def test_periodic_faces_paired_correctly():
    """配对后应恰好有 ny*nz*2 个面带非零 face_translation（每个 (j,k)
    格子沿 x 方向的三角化封盖面拆成 2 个三角形，两端各一个配对）。
    """
    _, mesh = _build_periodic_solver()
    fc = mesh.face_connectivity
    n_translated = int(np.sum(np.any(fc.face_translation != 0, axis=1)))
    assert n_translated == NY * NZ * 2, (
        f"expected {NY * NZ * 2} periodic-paired faces, got {n_translated}"
    )
    # 配对面必须是内部面（否则残差组装不会把它们当跨单元耦合处理）。
    translated_idx = np.flatnonzero(np.any(fc.face_translation != 0, axis=1))
    assert not np.any(fc.is_boundary[translated_idx]), "paired periodic faces must not remain boundary faces"


def test_periodic_freestream_preservation():
    """均匀自由流场跨越周期面时残差应保持机器精度量级——与 Couette
    棱柱网格的自由流保持性测试（test_couette.py）同一判据、同一量级。
    """
    solver, mesh = _build_periodic_solver()

    solver.state.U[:, :, 0] = RHO_INF
    solver.state.U[:, :, 1] = RHO_INF * U_INF
    solver.state.U[:, :, 4] = P_INF / (GAMMA - 1.0) + 0.5 * RHO_INF * U_INF**2
    solver.state._update_primitives()

    inv_res = solver.compute_inviscid_residual()
    visc_res = solver.compute_viscous_residual()
    assert np.max(np.abs(inv_res)) < 1e-3
    assert np.max(np.abs(visc_res)) < 1e-6


def test_periodic_nonuniform_field_no_outlier_at_periodic_face():
    """构造一个沿 x 周期（u 含 sin(2*pi*x/Lx) 分量）、沿 y 线性变化的
    非均匀速度场，验证跨周期面的单元残差没有异常离群值。
    """
    solver, mesh = _build_periodic_solver()

    x = mesh.sps_coords[:, :, 0]
    y = mesh.sps_coords[:, :, 1]
    k_wave = 2.0 * np.pi / Lx
    u_field = U_INF + 0.3 * U_INF * (y / H) + 0.1 * U_INF * np.sin(k_wave * x)

    solver.state.U[:, :, 0] = RHO_INF
    solver.state.U[:, :, 1] = RHO_INF * u_field
    solver.state.U[:, :, 4] = P_INF / (GAMMA - 1.0) + 0.5 * RHO_INF * u_field**2
    solver.state._update_primitives()

    inv_res = solver.compute_inviscid_residual()

    fc = mesh.face_connectivity
    periodic_owner_cells = np.unique(fc.owner_cell[np.any(fc.face_translation != 0, axis=1)])
    max_res_per_cell = np.max(np.abs(inv_res[:, :, 1]), axis=1)

    median_res = np.median(max_res_per_cell)
    periodic_cell_res = max_res_per_cell[periodic_owner_cells]

    assert np.all(np.isfinite(max_res_per_cell))
    # 紧邻周期面的单元残差不应比全场中位数高出一个数量级以上——真实
    # 复现过的失败模式（周期平移方向/符号搞反）会让这里的比值达到
    # 10^3~10^6 量级，与"确有非均匀源项、残差普遍较大"的正常情况
    # （中位数本身抬升，但比值不会失控）有清晰区分。
    ratio = periodic_cell_res.max() / max(median_res, 1e-300)
    assert ratio < 50.0, (
        f"cell adjacent to periodic face shows anomalous residual: "
        f"max={periodic_cell_res.max():.3e}, median={median_res:.3e}, ratio={ratio:.3f}"
    )
