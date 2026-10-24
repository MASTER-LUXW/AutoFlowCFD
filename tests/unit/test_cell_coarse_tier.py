"""块 Jacobi + 单元级粗校正档（`two+cc` / `single+cc`）接入求解路径：分档偏好、块缓存装配粗块并给出粗校正预处理、
dtau 复用判据、Newton 步里的部分灵活、紧耦合块对角组合。"""

import numpy as np
import pytest

from autoflowcfd.core.fr_residual.jacobian.backend import MeanFlowBlockAssembler
from autoflowcfd.core.time_integration.implicit.blocks import block_budget as bb
from autoflowcfd.core.time_integration.implicit.blocks.block_jacobi import BlockJacobiCache
from autoflowcfd.core.time_integration.implicit.blocks.coloring import greedy_cell_coloring
from autoflowcfd.core.time_integration.implicit.coarse.cell_coarse import (
    CoarseCorrectedPreconditioner, restriction_weights,
)
from autoflowcfd.core.time_integration.implicit.newton.jfnk import step_newton_krylov
from autoflowcfd.fr.native_padding import real_row_mask, real_sps_per_cell
from autoflowcfd.fr.operators import generate_fr_operators
from tests.unit.test_block_ilu_precond import _smooth_state
from tests.unit.test_mean_flow_analytic_jacobian import MACH_REF, MU, SCALES, _Residual, _channel


def test_mean_coarse_tiers_rank_between_ilu_and_plain_block_jacobi():
    """偏好 ilu > two+cc > single+cc > two > single：预算放得下单份 + 粗校正时它排在两份之前。"""
    sizes = (90649, 195744, 18, 10)                    # cube_demo 粗网格 P2
    cost = {m: bb.block_mode_bytes(m, *sizes, 5) for m in bb.MEAN_COARSE_MODES}
    assert cost["single+cc"] - cost["single"] == cost["two+cc"] - cost["two"] == bb.estimate_cell_coarse_bytes(*sizes)
    assert cost["two"] < cost["two+cc"] < cost["ilu"]
    assert bb.plan_block_modes(sizes, 5, 0, cost["two+cc"], mean_coarse=True) == ("two+cc", None)
    assert bb.plan_block_modes(sizes, 5, 0, cost["two+cc"] - 1, mean_coarse=True) == ("single+cc", None)
    assert bb.plan_block_modes(sizes, 5, 0, cost["single+cc"] - 1, mean_coarse=True) == ("single", None)
    # 不可用（P0 / 分布式）时与原来一样
    assert bb.plan_block_modes(sizes, 5, 0, cost["two+cc"], mean_coarse=False) == ("two", None)
    # 与湍流共享预算：平均流优先，湍流在剩余里取档
    single_sa = bb.block_mode_bytes("single", *sizes, 1)
    m, t = bb.plan_block_modes(sizes, 5, 1, cost["single+cc"] + single_sa, mean_coarse=True)
    assert (m, t) == ("single+cc", "single")


def _case(order=2):
    mesh, provider = _channel("tet", order)
    ops = generate_fr_operators(order)
    U = _smooth_state(mesh)
    res = _Residual(mesh, ops, provider, None, True, None)
    u0 = U.reshape(-1, 5)
    r0 = res(u0)
    npr, nte = real_sps_per_cell(order)
    cip = np.arange(mesh.n_cells) < mesh.n_prism_cells
    from autoflowcfd.core.fr_operators.face_kernels import get_flat_face_geometry

    flat = get_flat_face_geometry(mesh, ops)
    colors = greedy_cell_coloring(flat.owner_cell, flat.neighbor_cell, mesh.n_cells)
    real = real_row_mask(cip, mesh.n_sps_per_cell, order).reshape(mesh.n_cells, -1)
    from autoflowcfd.core.time_integration.positivity.limiter import build_positivity_limiter

    lim = build_positivity_limiter(mesh, ops, order=order)
    w = restriction_weights(np.asarray(lim.W), np.asarray(lim.norm_weights))

    def assembler():
        return MeanFlowBlockAssembler(mesh=mesh, ops=ops, ghost_provider=provider, mu=MU, mach_ref=MACH_REF,
                                      low_mach=True, mu_t=None, n_sps=mesh.n_sps_per_cell)

    return mesh, res, u0, r0, cip, npr, nte, colors, real, w, assembler


def _cache(mode, mesh, cip, npr, nte, colors, w):
    sizes = (int(cip.sum()), int((~cip).sum()), npr, nte)
    coarse = mode.endswith(bb.COARSE_SUFFIX)
    plan = bb.plan_for_budget(cip, npr, nte, 5, 0, bb.block_mode_bytes(mode, *sizes, 5), mean_coarse=coarse)
    assert plan.mean == mode
    return BlockJacobiCache(cell_is_prism=cip, colors=colors, plan=plan, n_sps=mesh.n_sps_per_cell,
                            n_real_prism=npr, n_real_tet=nte, n_var=5, coarse_weights=w if coarse else None)


@pytest.mark.parametrize("mode", ["two+cc", "single+cc"])
def test_cache_assembles_coarse_blocks_and_cuts_newton_gmres(mode):
    """`+cc` 档：块缓存经解析装配器拿到单元级粗块、给出粗校正预处理（部分灵活：只存粗层向量），大 CFL 下
    Newton 步的线性求解达到容差，GMRES 次数少于同一块 Jacobi 档（后者在 300 次内可能到不了容差）。"""
    mesh, res, u0, r0, cip, npr, nte, colors, real, w, assembler = _case()
    dtau = np.full(u0.shape[0], 5e-2)
    out = {}
    for m in (mode, mode[:-3]):
        cache = _cache(m, mesh, cip, npr, nte, colors, w)
        cache.assembler = assembler()
        u, info = step_newton_krylov(res, u0, dtau, SCALES, block_precond=cache, gmres_max_iter=300,
                                     real_rows=real.reshape(-1), norm_weights=np.where(real, 1.0, 0.0).reshape(-1),
                                     physicality=lambda u_, du, red: np.ones(u_.shape[0]))
        assert info["gmres_info"] == 0 or not m.endswith(bb.COARSE_SUFFIX), (m, info["gmres_iters"])
        out[m] = (cache, info["gmres_iters"])
    cache = out[mode][0]
    assert isinstance(cache.preconditioner(dtau, 5), CoarseCorrectedPreconditioner)
    assert cache.flexible and abs(cache.flexible_fraction - mesh.n_cells / (real.sum())) < 1e-12
    assert not out[mode[:-3]][0].flexible
    assert out[mode][1] < out[mode[:-3]][1], {k: v[1] for k, v in out.items()}


def test_coarse_solve_is_reused_until_dtau_drifts():
    mesh, res, u0, r0, cip, npr, nte, colors, real, w, assembler = _case(1)
    cache = _cache("two+cc", mesh, cip, npr, nte, colors, w)
    cache.assembler = assembler()
    n = u0.shape[0]
    cache.begin_step(res, u0, r0, SCALES, np.full(n, 1e-3))
    cache.preconditioner(np.full(n, 1e-3), 5)
    first = cache._coarse_solve
    cache.preconditioner(np.full(n, 1.5e-3), 5)
    assert cache._coarse_solve is first                      # 1.5 倍：复用
    cache.preconditioner(np.full(n, 3e-3), 5)
    assert cache._coarse_solve is not first                  # 3 倍：重建
    second = cache._coarse_solve
    cache.age = 1                                            # 复用中的块：补救重装配丢弃旧的粗层
    assert cache.recover(res, u0, r0, SCALES, np.full(n, 3e-3))
    assert cache._coarse_solve is None
    cache.preconditioner(np.full(n, 3e-3), 5)
    assert cache._coarse_solve is not second


def test_coupled_block_diagonal_passes_the_coarse_split_through():
    """紧耦合块对角组合：平均流带粗校正、湍流多层（整列非线性）时，只把湍流列标为非线性、粗层向量经
    `coarse_rows` 传给 GMRES；整体作用 = 拆分作用 + 延拓粗层向量。"""
    from autoflowcfd.core.time_integration.implicit.blocks.preconditioner import PseudoTransientDiagonal
    from autoflowcfd.core.time_integration.implicit.newton.coupled_step import _BlockDiagonal

    mesh, res, u0, r0, cip, npr, nte, colors, real, w, assembler = _case(1)
    cache = _cache("two+cc", mesh, cip, npr, nte, colors, w)
    cache.assembler = assembler()
    n = u0.shape[0]
    dtau = np.full(n, 1e-3)
    cache.begin_step(res, u0, r0, SCALES, dtau)
    mean = cache.preconditioner(dtau, 5)

    class _Flex(PseudoTransientDiagonal):
        flexible = True

    for turb, expect_cols in ((_Flex(dtau, 1), [False] * 5 + [True]), (PseudoTransientDiagonal(dtau, 1), None)):
        combo = _BlockDiagonal(mean, turb, dtau, 1)
        assert combo.flexible and combo.coarse_rows is mean.coarse_rows and combo.n_coarse_cols == 5
        assert (combo.nonlinear_columns is None) == (expect_cols is None)
        if expect_cols is not None:
            assert list(combo.nonlinear_columns) == expect_cols
        v = np.random.default_rng(3).normal(size=(n, 6))
        v[~real.reshape(-1)] = 0.0
        z, c = combo.apply_split(v)
        mean.prolong_add(z[:, :5], c)
        np.testing.assert_allclose(z, combo.apply(v), rtol=1e-12, atol=1e-30)
        lin = combo.apply_linear(v)
        np.testing.assert_array_equal(lin[:, :5], mean.apply_linear(v[:, :5]))
        assert np.all(lin[:, 5] == 0.0) == bool(turb.flexible)
