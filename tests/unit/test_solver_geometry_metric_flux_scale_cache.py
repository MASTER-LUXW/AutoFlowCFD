"""2026-08-23 发现的真实缺陷的回归测试（由用户要求追查 P1 发散根因引出，
det(J)/面失配/局部 CFL 几个假设逐一对照真实 cube_demo 数据被排除之后）：

`_SolverGeometryMixin._get_metric_flux_scale` 的缓存有效性检查只拿
`cached.shape[0]`（n_cells）与 `self.state.U.shape[0]`（也是 n_cells）
比较——两者在 Order Continuation 换阶前后都不变（P0->P1 等换阶时只有
`shape[1]`，即每单元的 n_sps 变化）。于是这个检查在任何换阶之后都报告
缓存仍然有效，静默地返回*旧*阶数的 `(n_cells, old_n_sps)` 数组。使用方
（cfl.py 的 `dt_geometric` 项）随后计算 `metric_flux_scale * wave_speed`，
其中 `wave_speed` 是 `(n_cells, new_n_sps)`——numpy 对大小为 1 的末维不
报错地广播，新阶数的每个解点都静默地复用较粗的旧阶数度量标度。这正是
专门为抓住退化单元解点级刚性而加的几何 CFL 项（见 cfl.py 里对它的大段
文档）——这个缺陷从换阶后的第一步起、直到运行结束都在静默地削弱这道
保护（代码库里没有别的地方显式让这个缓存失效——grep 过
`_metric_flux_scale_cache` 的全部引用确认）。

修复：比较完整的 `(n_cells, n_sps)` 形状，而不只是 `shape[0]`。
"""

from types import SimpleNamespace

import numpy as np

from autoflowcfd.core.fr_solver.solver_geometry import _SolverGeometryMixin


class _FakeSolver(_SolverGeometryMixin):
    def __init__(self, n_cells, n_sps):
        self.state = SimpleNamespace(U=np.zeros((n_cells, n_sps, 7)))
        det_jacs = np.full((n_cells, n_sps), 1e-6)
        inv_jacs = np.tile(np.eye(3), (n_cells, n_sps, 1, 1))
        self.mesh = SimpleNamespace(
            n_cells=n_cells, n_sps_per_cell=n_sps,
            jacobians={"det_jacs": det_jacs, "inv_jacs": inv_jacs},
        )


class TestMetricFluxScaleCacheInvalidatesOnOrderTransition:
    def test_cache_recomputes_with_correct_shape_after_order_change(self):
        n_cells = 5
        solver = _FakeSolver(n_cells, n_sps=1)  # P0
        scale_p0 = solver._get_metric_flux_scale()
        assert scale_p0.shape == (n_cells, 1)

        # 模拟 Order Continuation 换到 P1：n_cells 不变，n_sps 变化
        # （mesh.set_order + 状态插值）。
        solver.mesh.n_sps_per_cell = 8
        solver.mesh.jacobians["det_jacs"] = np.full((n_cells, 8), 1e-6)
        solver.mesh.jacobians["inv_jacs"] = np.tile(np.eye(3), (n_cells, 8, 1, 1))
        solver.state.U = np.zeros((n_cells, 8, 7))

        scale_p1 = solver._get_metric_flux_scale()
        assert scale_p1.shape == (n_cells, 8), (
            f"expected the cache to invalidate and recompute at the new "
            f"n_sps=8, got stale shape {scale_p1.shape}"
        )

    def test_cache_is_actually_reused_when_shape_is_unchanged(self):
        """防止矫枉过正成完全不缓存。"""
        n_cells = 5
        solver = _FakeSolver(n_cells, n_sps=8)
        first = solver._get_metric_flux_scale()
        # 原地改动网格的原始 Jacobian 而不碰 state.U——缓存被绕过的话会得到
        # 另一个数组对象；正确复用时，不论（现在已过时、但形状没变的）网格数据
        # 如何，拿回来的都是同一个缓存数组。
        solver.mesh.jacobians["det_jacs"] = solver.mesh.jacobians["det_jacs"] * 2
        second = solver._get_metric_flux_scale()
        assert second is first
