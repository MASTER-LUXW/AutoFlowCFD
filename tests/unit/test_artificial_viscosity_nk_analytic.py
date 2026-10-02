"""启用问题单元人工粘性的隐式稳态（NEWTON_KRYLOV）步走解析单元块装配。

2026-10-02 之前 `fr_residual/jacobian/backend.py::unsupported_reason` 对人工粘性返回
"不覆盖"，块 Jacobi 整体退回着色差分装配（每次装配 色数 x 解点数 x 5 次残差求值）。现在
人工粘性的 `-Gamma L_nu` 由与残差同一个标量扩散离散解析给出
（`fr_residual/jacobian/artificial_viscosity.py`）。判据：

1. 人工扩散系数确实非零（否则测试无检验力），且步内的块装配器存在（走解析路径）；
2. 解析路径的收敛不劣于差分路径：块只作预处理，两者精确到差分截断。注意不能要求两条
   轨迹逐步吻合——非精确 Newton（forcing `eta` 约 0.1）下线性解只精确到 `eta |R|`，不同的
   预处理子给出的更新本来就允许差到这个量级（实测 4 步内逐步差 8%）。
"""

import numpy as np

from autoflowcfd.core.time_integration import TimeIntegrationScheme
from tests.unit.test_artificial_diffusion_channel import _set_density_bump
from tests.validation._channel_mesh import build_channel_mesh_prism, build_face_exact_ghost_provider

H, LX, LZ = 1.0e-2, 2.0e-2, 2.5e-3
RHO, P, U_INF = 1.225, 101325.0, 30.0


def _solver():
    from autoflowcfd.core.fr_solver.solver import FRSolver

    mesh = build_channel_mesh_prism(1, nx=4, ny=4, nz=1, Lx=LX, H=H, Lz=LZ)
    bc = {n: {"type": "SYMMETRY"} for n in ("wall_bottom", "wall_top", "z_min", "z_max")}
    for n in ("x_min", "x_max"):
        bc[n] = {"type": "FARFIELD", "Q_free": [RHO, U_INF, 0.0, 0.0, P]}
    solver = FRSolver(mesh=mesh, order=1, turb_model_name="NONE", n_vars=5,
                      time_scheme=TimeIntegrationScheme.NEWTON_KRYLOV, rho_inf=RHO, vel_inf=U_INF, p_inf=P,
                      mu_molecular=1.8e-5, bc_overrides=bc, n_threads=1, artificial_viscosity_enabled=True)
    solver.order_continuation_enabled = False
    solver.boundary_ghost_provider = build_face_exact_ghost_provider(mesh, LX, H, LZ, bc)
    _set_density_bump(solver, mesh, amp=0.05)
    return solver


def _run(monkeypatch, force_difference: bool, n_steps: int = 4):
    from autoflowcfd.core.fr_residual.jacobian import backend
    from autoflowcfd.core.time_integration.implicit import mean_flow_step

    if force_difference:
        monkeypatch.setattr(backend, "unsupported_reason", lambda **kw: "测试：强制差分装配")
    else:
        monkeypatch.undo()
    solver = _solver()
    assert float(np.max(solver.compute_artificial_diffusivity_field())) > 0.0, "人工扩散系数恒为零，测试无检验力"
    assemblers, gmres, res = [], [], []
    original = mean_flow_step.step_mean_flow_newton

    def spy(*args, **kw):
        assemblers.append(kw.get("block_assembler"))
        return original(*args, **kw)

    monkeypatch.setattr(mean_flow_step, "step_mean_flow_newton", spy)
    for _ in range(n_steps):
        solver.step(0.0)
        gmres.append(int(solver._newton_last_info["gmres_iters"]))
        res.append(float(solver._newton_last_info["res_norm"]))
    monkeypatch.setattr(mean_flow_step, "step_mean_flow_newton", original)
    return assemblers, gmres, np.array(res)


def test_artificial_viscosity_nk_uses_analytic_blocks_and_matches_difference(monkeypatch):
    asm_a, gm_a, res_a = _run(monkeypatch, force_difference=False)
    asm_d, gm_d, res_d = _run(monkeypatch, force_difference=True)
    assert all(a is not None for a in asm_a), "启用人工粘性时仍未走解析装配"
    assert all(a is None for a in asm_d)
    assert res_a[0] == res_d[0]                       # 同一初场
    assert res_a[-1] < 0.1 * res_a[0]
    assert res_a[-1] <= 1.2 * res_d[-1], (res_a, res_d)
    assert sum(gm_a) <= sum(gm_d) + max(2, 0.1 * sum(gm_d)), (gm_a, gm_d)


def test_distributed_artificial_viscosity_nk_matches_single_machine():
    """CPU 分布式（n_ranks=1）的隐式步同样走解析装配（`distributed_mean_flow_assembler` 透传
    compact 排列的 `nu_av`），Newton 轨迹与单机一致到浮点重结合噪声。"""
    from autoflowcfd.core.fr_residual.inviscid import conserved_to_primitive
    from autoflowcfd.core.mpi.distributed_solver import DistributedFRSolver
    from autoflowcfd.fr.operators import generate_fr_operators

    single = _solver()
    mesh = single.mesh
    bc = {n: {"type": "SYMMETRY"} for n in ("wall_bottom", "wall_top", "z_min", "z_max")}
    for n in ("x_min", "x_max"):
        bc[n] = {"type": "FARFIELD", "Q_free": [RHO, U_INF, 0.0, 0.0, P]}
    dist = DistributedFRSolver(
        mesh=mesh, ops=generate_fr_operators(1), face_connectivity=mesh.face_connectivity, n_ranks=1,
        backend="cpu", order=1, turb_model_name="NONE", n_vars=5,
        time_scheme=TimeIntegrationScheme.NEWTON_KRYLOV, mu_molecular=1.8e-5, rho_inf=RHO, vel_inf=U_INF,
        p_inf=P, bc_overrides=bc, artificial_viscosity_enabled=True)
    dist.local_solver.boundary_ghost_provider = build_face_exact_ghost_provider(mesh, LX, H, LZ, bc)
    n = mesh.n_cells
    dist.state.U[:n] = single.state.U
    dist.state.Q[:n] = conserved_to_primitive(np.asarray(single.state.U))
    for k in range(3):
        single.step(0.0)
        dist.step(0.0)
        a, b = dist._newton_last_info, single._newton_last_info
        assert a["gmres_iters"] == b["gmres_iters"], f"第 {k + 1} 步 GMRES 次数不同：{a} vs {b}"
        # 残差范数含大量相消项（密度扰动场），compact 换序的重结合噪声被放大到 ~2e-9
        assert abs(a["res_norm"] / b["res_norm"] - 1) < 1e-8, (a["res_norm"], b["res_norm"])
    assert dist._newton_block_precond.assembler is not None, "分布式隐式步未走解析装配"
    rel = np.abs(dist.state.U[:n] - single.state.U).max() / np.abs(single.state.U).max()
    assert rel < 1e-9, rel
