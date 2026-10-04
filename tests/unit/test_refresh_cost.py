# -*- coding: utf-8 -*-
"""块预处理刷新判据的确定性代价模型（`time_integration/implicit/refresh_cost.py`）。

2026-09-26 起刷新判据按实测墙钟耗时之比决定"一次装配折合多少次迭代"，同一输入跨进程的 Newton 轨迹
因此不可逐位复现（SST 黄金轨迹 P2 段在两个结果间切换）；2026-10-04 改为按块类型与阶数查实测标定表、
差分装配按精确的残差求值次数折算。
"""

import numpy as np
import pytest

from autoflowcfd.core.time_integration.implicit.refresh_cost import (
    ANALYTIC_ASSEMBLY_ITERS, FD_EVAL_ITERS, REFRESH_SLACK_MIN, assembly_cost_iters, block_kind,
)


def test_block_kinds():
    assert block_kind(5, False) == "mean_laminar"
    assert block_kind(5, True) == "mean_coupled"
    assert block_kind(1, False) == "turbulence_1"
    assert block_kind(2, False) == "turbulence_2"


@pytest.mark.parametrize("key,value", sorted(ANALYTIC_ASSEMBLY_ITERS.items()))
def test_analytic_table_lookup(key, value):
    assert assembly_cost_iters(key[0], key[1], True, 0) == max(REFRESH_SLACK_MIN, value)


def test_finite_difference_counts_residual_evaluations():
    assert assembly_cost_iters("mean_laminar", 0, False, 50) == round(FD_EVAL_ITERS * 50)
    assert assembly_cost_iters("mean_laminar", 0, False, 1) == REFRESH_SLACK_MIN


def test_missing_calibration_is_an_error_not_a_guess():
    with pytest.raises(KeyError, match="标定"):
        assembly_cost_iters("mean_laminar", 4, True, 0)


def test_cache_threshold_uses_the_table_after_a_build():
    """装配之后刷新阈值 = 基线 + 查表值（与耗时无关）。"""
    from autoflowcfd.core.time_integration.implicit.block_jacobi import BlockJacobiCache

    cache = BlockJacobiCache(cell_is_prism=np.ones(4, dtype=bool), colors=np.zeros(4, dtype=np.int64),
                             n_sps=8, n_real_prism=6, n_real_tet=4, n_var=5, with_turbulence=True)
    blocks = np.tile(np.eye(30, dtype=np.float32), (4, 1, 1))

    class _Assembler:
        want_coupling = False

        def __call__(self, u0, r0):
            return blocks, np.zeros((0, 20, 20), dtype=np.float32)

    cache.use_ilu = False          # 只测阈值：假装配器不给面邻居耦合块
    cache.assembler = _Assembler()
    cache.begin_step(None, np.zeros((32, 5)), np.zeros((32, 5)), np.ones(5), np.ones(32))
    assert cache.refresh_slack == ANALYTIC_ASSEMBLY_ITERS[("mean_coupled", 1)]
    assert cache._refresh_threshold(3) == 3 + ANALYTIC_ASSEMBLY_ITERS[("mean_coupled", 1)]


def test_newton_trajectory_is_bitwise_reproducible():
    """同一输入连跑两遍（含块装配、复用与刷新），每步的 GMRES 次数与最终状态逐位相同。"""
    from autoflowcfd.core.time_integration import TimeIntegrationScheme
    from tests.unit.test_implicit_sst_nk import _channel_solver

    runs = []
    for _ in range(2):
        s = _channel_solver(TimeIntegrationScheme.NEWTON_KRYLOV)
        gm = []
        for _ in range(25):
            s.step(2.0e-7)
            gm.append(s._newton_last_info["gmres_iters"])
        runs.append((gm, np.array(s.state.U), np.array(s.turb_model.omega_field), s._newton_block_precond.n_builds))
    assert runs[0][0] == runs[1][0]
    assert runs[0][3] == runs[1][3]
    np.testing.assert_array_equal(runs[0][1], runs[1][1])
    np.testing.assert_array_equal(runs[0][2], runs[1][2])
