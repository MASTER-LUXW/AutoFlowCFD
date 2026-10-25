"""阶段已走步数随 checkpoint 持久化（`order_continuation/checkpoint_state.py`，I-243）：续算出来的第一个阶段从
checkpoint 记录的步数接着数——非最终阶段的步数上限不被重复使用、日志步号连续；定阶循环同样续接；旧格式
（有阶段基准、没有阶段步数）报错。

2026-10-25 以前续算时阶段从第 1 步重新数：方柱 SA URANS 从 P0 阶段末的 checkpoint 续算，P0 又跑满 150 步。
"""

from unittest.mock import patch

import numpy as np
import pytest

from autoflowcfd.core.utils.order_continuation import run_order_continuation
from autoflowcfd.core.utils.order_continuation.checkpoint_state import (
    PHASE_INITIAL_RESIDUAL_KEY, PHASE_ITERATIONS_KEY, phase_state_metadata, restore_phase_state,
)
from tests.unit.test_order_continuation_resume import _fake_generate_ops, _fake_solver_with_residual_sequence
# 先于任何 `generate_fr_operators` 替换导入真实的网格与求解器模块：`high_order_mesh.py`、`fr_solver/solver/setup.py`
# 在模块顶部按名绑定该函数，若首次导入发生在替换期间（`run_order_continuation` 会连带导入求解器包），替身会永久
# 留在它们的命名空间里，单独运行本文件时定阶用例的真实网格与求解器随后取到替身算子
from autoflowcfd.core.fr_solver import FRSolver  # noqa: E402,F401
from tests.unit.test_solver_probes import DT, _channel_solver  # noqa: E402


def _slow(n=2000):
    """缓慢下降、不触发提前升阶 / 收敛 / 平台判据的残差序列。"""
    return [1.0 / (1.0 + 1e-3 * i) for i in range(n)]


def _run(solver, **kw):
    steps = []

    def record(s, k):
        steps.append((s.current_order, s._phase_iterations))

    with patch("autoflowcfd.fr.operators.generate_fr_operators", side_effect=_fake_generate_ops):
        result = run_order_continuation(solver, dt=1e-3, tol=1e-12, checkpoint_callback=record, **kw)
    return result, steps


def test_resume_inside_a_non_final_phase_uses_only_its_remaining_budget(capsys):
    solver = _fake_solver_with_residual_sequence(0, 2, True, _slow(), phase_initial_residual=1.0)
    solver._phase_iterations = 140
    result, steps = _run(solver, max_iter=40, phase_max_iter=150)
    p0 = [n for o, n in steps if o == 0]
    assert p0 == list(range(141, 151))                     # 只补满 P0 剩下的 10 步，步号接着 140 数
    p1 = [n for o, n in steps if o == 1]
    assert p1 == list(range(1, 31))                         # 升到 P1 后从 1 数，吃掉剩余预算
    assert result.iterations == 40
    out = capsys.readouterr().out
    assert "P0 Iter 141:" in out and "本阶段已走 140 步" in out


def test_resume_at_the_end_of_a_non_final_phase_advances_immediately():
    solver = _fake_solver_with_residual_sequence(0, 2, True, _slow(), phase_initial_residual=1.0)
    solver._phase_iterations = 150
    _, steps = _run(solver, max_iter=5, phase_max_iter=150)
    assert [o for o, _ in steps] == [1] * 5 and steps[0][1] == 1


def test_resume_in_the_final_phase_continues_its_numbering_with_the_full_budget():
    solver = _fake_solver_with_residual_sequence(2, 2, True, _slow(), phase_initial_residual=1.0)
    solver._phase_iterations = 37
    result, steps = _run(solver, max_iter=6, phase_max_iter=3)
    assert [n for _, n in steps] == list(range(38, 44)) and result.iterations == 6


def test_fresh_phases_count_from_one():
    solver = _fake_solver_with_residual_sequence(0, 1, False, _slow())
    _, steps = _run(solver, max_iter=8, phase_max_iter=3)
    assert steps == [(0, 1), (0, 2), (0, 3), (1, 1), (1, 2), (1, 3), (1, 4), (1, 5)]


def test_metadata_round_trip_and_old_format_is_rejected():
    from types import SimpleNamespace

    src = SimpleNamespace(_phase_initial_residual=2.5, _phase_iterations=17)
    meta = phase_state_metadata(src)
    assert meta[PHASE_ITERATIONS_KEY] == 17
    dst = SimpleNamespace()
    restore_phase_state(dst, meta)
    assert dst._phase_initial_residual == 2.5 and dst._phase_iterations == 17
    assert phase_state_metadata(SimpleNamespace()) == {}
    old = {k: v for k, v in meta.items() if k != PHASE_ITERATIONS_KEY}
    with pytest.raises(ValueError, match=PHASE_ITERATIONS_KEY):
        restore_phase_state(SimpleNamespace(), old)
    restore_phase_state(SimpleNamespace(), {})               # 没有阶段基准：什么都不设
    assert PHASE_INITIAL_RESIDUAL_KEY in meta


def test_fixed_order_loop_persists_and_continues_the_phase_count(tmp_path):
    """定阶循环（不走 Order Continuation）同样写阶段步数，续算接着数；checkpoint 回调仍按本次调用的步数。"""
    from autoflowcfd.cli.solve.checkpoint_io.write import write_checkpoint
    from autoflowcfd.core.utils.checkpoint import CheckpointManager

    solver = _channel_solver()
    solver.solve(max_iter=3, dt=DT, tol=0.0)
    assert solver._phase_iterations == 3
    meta = CheckpointManager().load(write_checkpoint(solver, str(tmp_path), 3, "mesh.nas", order=1,
                                                     turbulence_model="none", backend="cpu", quiet=True))[3]
    assert int(meta[PHASE_ITERATIONS_KEY]) == 3
    resumed = _channel_solver()
    restore_phase_state(resumed, meta)
    resumed._resumed_from_checkpoint = True
    calls = []
    result = resumed.solve(max_iter=2, dt=DT, tol=0.0, checkpoint_callback=lambda s, k: calls.append(k))
    assert resumed._phase_iterations == 5 and calls == [1, 2] and result.iterations == 2
    assert np.isfinite(result.final_residual)
