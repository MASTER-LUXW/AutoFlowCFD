"""2026-08-22 发现的真实缺陷的回归测试（用户在一次真实的 P0->P1 换阶日志
里直接看到了原始的 RuntimeWarning）：

`fr_solver/turbulence.py::compute_turbulence_source` 对 grad_k/grad_omega
模长的限幅（`np.linalg.norm(grad_k, axis=-1)` 之后截到 1e6）没有
`np.errstate` 保护。在退化单元上（坍缩坐标几何让度量比 adj(J)/det(J)
爆掉——见 troubled_cell.py 的退化单元诊断），理论上为常数的梯度里的浮点
噪声被放大到超过 1e150；`np.linalg.norm` 内部对它平方时 float64 溢出，
抛出一条到达用户控制台的 RuntimeWarning，尽管随后的限幅本来就能正确处理
`inf` 输入（inf > 1e6 恒为 True，值被缩小而不是变成 NaN）。

`turbulence/transport.py` 有*完全相同*的限幅写法，2026-08-21 已经包进
`np.errstate(over='ignore', invalid='ignore')`——这个文件（transport.py
的注释自己引用的、更早的原始位置）当时被漏掉了。本测试在隔离出来的数值
写法上钉住抑制行为（不是完整的 `compute_turbulence_source` 调用，那需要
带真实网格/算子几何的大型求解器替身来跑 `compute_scalar_gradient`——
只为重新证明 numpy 自己有文档的 errstate 语义而去造替身不值得）；要紧的
是这个文件的代码现在与已覆盖的 transport.py 位置包法相同。
"""

import warnings

import numpy as np



class TestGradClipErrstateWrapping:
    def test_source_file_wraps_grad_clip_in_errstate(self):
        """防止这层包装被以后的修改悄悄去掉。2026-09-26：限幅只在一个共用函数
        里（`sst/bounds.py::clip_gradient_magnitude`，CPU 源项、CPU 输运、单 GPU
        与多 GPU 路径都用它）；范数必须在那里的 `np.errstate` 之内，源项求值必须
        调用它。
        """
        import inspect

        from autoflowcfd.core.fr_solver.turbulence.source import evaluate_turbulence_rates
        from autoflowcfd.core.turbulence.limits import clip_gradient_magnitude

        src = inspect.getsource(clip_gradient_magnitude)
        errstate_idx = src.index('with np.errstate(over="ignore", invalid="ignore"):')
        norm_idx = src.index("mag = np.linalg.norm(grad, axis=-1)")
        assert errstate_idx < norm_idx, (
            "np.errstate wrapping must appear before the norm computation it's meant to protect")
        # k 与 w = ln(omega) 两个梯度都经过同一个裁剪（2026-09-27 起梯度对 ln(omega) 求）
        assert inspect.getsource(evaluate_turbulence_rates).count("clip_gradient_magnitude(") >= 2

    def test_overflow_prone_norm_and_clip_is_warning_free_under_errstate(self):
        """隔离地复现实际的数值失效方式：一个大到平方会让 float64 溢出
        （>~1.34e154）的梯度分量，过一遍共用的限幅函数——必须零警告，并给出
        正确限幅（不是 NaN）的结果。
        """
        from autoflowcfd.core.turbulence.limits import clip_gradient_magnitude

        grad_k = np.zeros((2, 1, 3))
        grad_k[0, 0, 0] = 1e200  # squaring this overflows float64
        grad_k[1, 0, 1] = 3.0

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            out = clip_gradient_magnitude(grad_k, np)

        assert len(caught) == 0, f"expected no warnings, got: {[str(w.message) for w in caught]}"
        assert np.isfinite(out).all()
        assert out[0, 0, 0] == 0.0     # 模长溢出成 inf -> 缩放 0（不是 NaN），与此前行为一致
        np.testing.assert_array_equal(out[1], grad_k[1])          # 未超限的点逐位不变
