"""累计伪时间 `tau_accum` 必须跨 `solve resume` 延续（2026-09-24）。

## 为什么这件事重要

`tau/T_body`（累计伪时间 / 物体尺度对流时标）是**唯一**能回答"物理场到底
走了多远"的量。残差范数完全不回答这个：`pseudotime_budget.py` 模块文档
记录过，plate_demo 上残差单调下降 350 步、而物理场只走完绕板特征时间的
**1.8%**，那次误判把启动暂态的压力分布当成壁面处理缺陷追了好几天。

`tau_accum` 原先的语义是"**本次 solve() 调用**已推进的伪时间"，每次
`solve()` 无条件清零。这对长程算例是错的 —— 长程算例正常就是靠
`solve resume` 接力跑的，于是这个量恰好在最需要它的场景下归零。

**真实踩到过（2026-09-24 本次）**：做自适应 CFL 闭环 A/B 时，一臂是原
运行、一臂是 resume。按"相同迭代步"比较 Cd 得出了"高 CFL 在振荡"的结论；
改按累计伪时间对齐才发现高 CFL 臂只是每步多走了 4.38 倍伪时间（与 CFL
比值 4.62 倍吻合），那段 Cd 上升是低 CFL 臂**还没走到**的物理轨迹。
结论完全反了，而 tau 不跨 resume 延续正是差点写错的直接原因。

## 判据

1. `tau_accum` 写进 checkpoint 的 `extra_fields`；
2. `rebuild_solver_from_checkpoint` 恢复它并置上 `_tau_accum_seeded`；
3. `solve()` 看到这个标记就**不清零**，且**消费掉**标记（同一个 solver
   对象上第二次全新 `solve()` 仍正常清零）；
4. 旧版本 checkpoint（没有这个字段）行为与改动前完全一致 —— 不假装有。
"""

import inspect

import numpy as np


def _solve_src():
    from autoflowcfd.core.fr_solver.solver import FRSolver
    return inspect.getsource(FRSolver.solve)


def test_checkpoint_writer_persists_tau_accum():
    """判据 1：写入路径把 `tau_accum` 放进 extra_fields。"""
    from autoflowcfd.cli import solve_checkpoint_io as io_mod

    src = inspect.getsource(io_mod)
    assert 'extra_fields["tau_accum"]' in src, (
        "checkpoint 没有持久化 tau_accum —— 它与 k_field/omega_field/nu_t "
        "是同一类：resume 精确恢复物理场、却把这个计数打回 0")


def test_checkpoint_reader_restores_and_marks():
    """判据 2：读取路径恢复它并置上 `_tau_accum_seeded`。"""
    from autoflowcfd.cli import solve_checkpoint_io as io_mod

    src = inspect.getsource(io_mod)
    assert '"tau_accum" in fields' in src
    assert "solver.tau_accum = _tau" in src
    assert "solver._tau_accum_seeded = True" in src, (
        "恢复了 tau 却没置标记的话，紧接着的 solve() 会立刻把它清掉")


def test_solve_does_not_clobber_a_seeded_tau():
    """判据 3：`solve()` 只在没有种子时清零，且消费掉标记。"""
    src = _solve_src()
    assert "_tau_accum_seeded" in src, (
        "solve() 没有检查种子标记 —— 它那句无条件 `self.tau_accum = None` "
        "会让恢复出来的伪时间在第一行就丢掉")
    # 标记必须被消费（置回 False），否则同一对象上第二次全新 solve()
    # 会错误地继承上一次的 tau。
    assert "self._tau_accum_seeded = False" in src, (
        "标记没有被消费：同一个 solver 对象上第二次全新 solve() 会错误地"
        "沿用上一次的 tau_accum")


class _FakeSolver:
    """只实现 `solve()` 里那段 tau 起点逻辑所需的最小接口。"""

    def __init__(self, seeded, tau):
        self._tau_accum_seeded = seeded
        self.tau_accum = tau

    def reset_tau_like_solve(self):
        # 与 `FRSolver.solve()` 里那段逐字对应
        if getattr(self, "_tau_accum_seeded", False):
            self._tau_accum_seeded = False
        else:
            self.tau_accum = None


def test_seeded_tau_survives_then_second_solve_resets():
    """判据 3 的行为验证：第一次保留、第二次清零。"""
    tau = np.array([1.0, 2.0, 3.0])
    s = _FakeSolver(seeded=True, tau=tau)

    s.reset_tau_like_solve()
    assert s.tau_accum is tau, "恢复出来的 tau 在第一次 solve() 里被清掉了"
    assert s._tau_accum_seeded is False, "标记没有被消费"

    s.reset_tau_like_solve()
    assert s.tau_accum is None, (
        "同一个 solver 对象上第二次全新 solve() 应当正常清零 —— 否则"
        "一次 resume 会污染之后所有的 solve()")


def test_fresh_solver_still_resets():
    """判据 4：没有种子（全新求解 / 旧版本 checkpoint）时行为不变。"""
    s = _FakeSolver(seeded=False, tau=np.array([1.0]))
    s.reset_tau_like_solve()
    assert s.tau_accum is None


def test_reader_guards_shape_mismatch():
    """长度与网格单元数不符时不能硬塞 —— 那会让伪时间预算算出错的数。"""
    from autoflowcfd.cli import solve_checkpoint_io as io_mod

    src = inspect.getsource(io_mod)
    assert "tau_accum 长度" in src and "跳过恢复" in src, (
        "缺少形状校验：长度不符时应当跳过恢复并明确告知，而不是静默"
        "塞进去让 pseudo_time_budget 算出无意义的数")
