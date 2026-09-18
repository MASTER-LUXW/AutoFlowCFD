"""GPU 两条路径的传感器门控（2026-09-18 接线）+ 一个真实静默缺陷的回归。

## 本机能验证什么、不能验证什么（如实说明）

本机没有 CUDA/CuPy，`build_gpu_filter_func` / `_build_sensor_gated_filter_gpu`
这类直接 `get_cupy()` 的入口无法执行（对应测试 skip）。但本次接线里**真正
可能算错**的两处都不是 CuPy API，而是纯算法/索引代数，两者都能在 NumPy 上
决定性验证：

1. **设备侧施加形态的数值等价性**。GPU 分支不用花式索引，而是"两个滤波
   矩阵都对全场各算一遍，再按单元类型选、再按 troubled 选"（理由见
   `gpu_modal_filter.py::filter_scalar_field_gated_gpu`：设备上
   gather/scatter 的开销高于多做一遍小矩阵乘）。这个三重混合的代数是否
   等于 CPU 的"按类型 gather、只在被标记单元上施加"，与在哪个设备上跑
   无关——`TestGpuApplyFormIsEquivalent` 直接对照。
2. **多 GPU 的单元类型索引代数**。`cell_is_prism = compact_cell_type
   [inv_perm][:n_local]`——这是修掉那个静默缺陷的核心一行，
   `TestMultiGpuCellTypeIndexAlgebra` 用构造好的置换对照真值。

真正留给真实硬件的只有 `cupyx.scatter_max/scatter_min` 与 `cp.einsum`
这两处 API 调用本身。

## 回归的那个真实缺陷

多 GPU 此前给滤波器传 `n_prism = self.mesh.n_prism_cells`（**全局**棱柱数），
而 `build_gpu_filter_func` 按 `U[:n_prism]` 切片、作用的数组只有 `n_local`
行。传统模式下 `self.mesh` 是完整全局网格，所以 `n_prism > n_local` 是常态，
切片被静默钳到 `n_local`——**每个 local 单元、包括四面体，都被施加了棱柱
滤波矩阵**，不报任何错。四面体走 native PKD/Dubiner 基（带零填充槽位），
与棱柱的张量积基完全不同，混用没有数值意义。
`TestNoSilentClamp` 既钉住新的护栏，也复现"静默钳会造成什么"。
"""

import numpy as np
import pytest

from autoflowcfd.core.fr_solver.filter import (
    build_sensor_gated_filter_func_arrays,
)


def _case(seed=5, n_cells=90, n_sps=6, n_var=5):
    """一个棱柱/四面体**交错**的小算例 + 面连接。

    交错是关键：`n_prism` 切片表达不了它，只有 `cell_is_prism` 能——
    分布式 local 排列正是这个形态。
    """
    rng = np.random.default_rng(seed)
    base = np.stack([
        1.2 + 0.05 * np.sin(np.linspace(0, 7, n_cells)),
        30.0 * np.linspace(0, 1, n_cells),
        0.4 * np.cos(np.linspace(0, 5, n_cells)),
        0.05 * np.linspace(0, 1, n_cells),
        101325.0 + 40.0 * np.linspace(0, 1, n_cells),
    ], axis=1)
    field = np.repeat(base[:, None, :], n_sps, axis=1)
    field += 2e-3 * rng.normal(size=field.shape) * np.abs(base)[:, None, :]
    bad = rng.choice(n_cells, size=max(n_cells // 10, 2), replace=False)
    field[bad] += 0.3 * np.abs(base)[bad][:, None, :] * rng.normal(
        size=(bad.size, n_sps, n_var))

    o = np.arange(n_cells - 1, dtype=np.int64)
    n = np.arange(1, n_cells, dtype=np.int64)
    bnd = np.zeros(n_cells - 1, dtype=bool)
    o = np.concatenate([o, [0, n_cells - 1]])
    n = np.concatenate([n, [-1, -1]])
    bnd = np.concatenate([bnd, [True, True]])
    cell_is_prism = (np.arange(n_cells) % 3 != 0)      # 交错
    return field, o, n, bnd, cell_is_prism


_FREESTREAM = {"rho_inf": 1.225, "vel_inf": 30.0, "p_inf": 101325.0}


def _two_distinct_filters(n_sps, seed=1):
    """两个**明显不同**的滤波矩阵。

    必须不同：若两者相同，"按单元类型选"这一步选错也看不出来——那正是
    多 GPU 那个缺陷能长期潜伏的原因。
    """
    rng = np.random.default_rng(seed)
    A = np.full((n_sps, n_sps), 1.0 / n_sps)                     # 投影到常数
    B = np.eye(n_sps) * 0.5 + 0.5 / n_sps
    B += 1e-3 * rng.normal(size=(n_sps, n_sps))
    return A, B


class TestGpuApplyFormIsEquivalent:
    """设备侧"全算再按掩码选"的代数 == CPU"按类型 gather 后施加"。"""

    def test_triple_blend_matches_gathered_apply(self):
        field, o, n, bnd, cip = _case()
        n_cells, n_sps = field.shape[0], field.shape[1]
        Fp, Ft = _two_distinct_filters(n_sps)

        # CPU 门控（numba gather 路径）的结果
        ff = build_sensor_gated_filter_func_arrays(
            n_cells, n_sps, 1, Fp, Ft, cell_is_prism=cip, sensor="bounds",
            owner_cell=o, neighbor_cell=n, is_boundary=bnd,
            freestream=_FREESTREAM)
        flat = field.reshape(n_cells * n_sps, 5).copy()
        cpu_out = ff(flat.copy()).reshape(n_cells, n_sps, 5)

        # 掩码本身（与门控回调用的是同一个内核，同一组参数）
        from autoflowcfd.core.fr_operators.bounds_sensor import (
            compute_bounds_violation_mask,
        )
        from autoflowcfd.core.fr_solver.residual_diagnostics import (
            _reference_scales,
        )
        troubled = compute_bounds_violation_mask(
            field[:, :, :5], o, n, bnd,
            ref_scales=_reference_scales(_FREESTREAM, 5))
        assert 0 < troubled.sum() < n_cells, (
            f"掩码必须既不空也不满才有区分力，实际 {troubled.sum()}/{n_cells}")

        # GPU 分支的形态，逐字照抄（只把 xp 换成 np）
        lead = field[:, :, :5]
        filt = np.where(cip[:, None, None],
                        np.einsum("sj,cjv->csv", Fp, lead),
                        np.einsum("sj,cjv->csv", Ft, lead))
        gpu_out = field.copy()
        gpu_out[:, :, :5] = np.where(troubled[:, None, None], filt, lead)

        np.testing.assert_allclose(
            gpu_out, cpu_out, rtol=1e-13, atol=1e-11,
            err_msg="设备侧三重混合与 CPU 的按类型 gather 施加不等价")

    def test_swapping_the_two_matrices_is_detectable(self):
        """把两个矩阵按类型选反必须给出不同结果。

        没有这条，上面那个等价性可能只是因为两个矩阵作用相近——而"四面体
        被施加棱柱矩阵"正是本次修掉的那个真实缺陷的形态。
        """
        field, o, n, bnd, cip = _case()
        n_cells, n_sps = field.shape[0], field.shape[1]
        Fp, Ft = _two_distinct_filters(n_sps)
        from autoflowcfd.core.fr_operators.bounds_sensor import (
            compute_bounds_violation_mask,
        )
        from autoflowcfd.core.fr_solver.residual_diagnostics import (
            _reference_scales,
        )
        troubled = compute_bounds_violation_mask(
            field[:, :, :5], o, n, bnd,
            ref_scales=_reference_scales(_FREESTREAM, 5))
        lead = field[:, :, :5]

        def blend(mat_prism, mat_tet):
            filt = np.where(cip[:, None, None],
                            np.einsum("sj,cjv->csv", mat_prism, lead),
                            np.einsum("sj,cjv->csv", mat_tet, lead))
            return np.where(troubled[:, None, None], filt, lead)

        assert not np.allclose(blend(Fp, Ft), blend(Ft, Fp)), (
            "两个滤波矩阵互换后结果相同——本测试对'按类型选错'没有区分力")


class TestMultiGpuCellTypeIndexAlgebra:
    """`cell_is_prism = compact_cell_type[inv_perm][:n_local]`。"""

    def _build(self, n_local=11, n_halo=5, seed=2):
        """构造一对自洽的 (紧凑"棱柱在前"排列, perm/inv_perm)。

        真值：原生排列下每个单元是不是棱柱。
        """
        rng = np.random.default_rng(seed)
        n_total = n_local + n_halo
        is_prism_native = rng.random(n_total) < 0.5
        # 与 build_distributed_flat_face 里完全同一个构造：
        #   perm = [棱柱的原生下标..., 四面体的原生下标...]
        perm = np.concatenate([np.flatnonzero(is_prism_native),
                               np.flatnonzero(~is_prism_native)])
        inv_perm = np.empty_like(perm)
        inv_perm[perm] = np.arange(n_total, dtype=perm.dtype)
        # compact_cell_type: 0=棱柱/1=四面体，按紧凑排列
        compact_cell_type = np.where(is_prism_native[perm], 0, 1).astype(np.int8)
        return is_prism_native, perm, inv_perm, compact_cell_type, n_local

    def test_derived_mask_matches_ground_truth(self):
        truth, perm, inv_perm, cct, n_local = self._build()
        derived = (cct[inv_perm][:n_local] == 0)
        np.testing.assert_array_equal(
            derived, truth[:n_local],
            err_msg="紧凑->原生换算错了：四面体会被施加棱柱滤波矩阵")

    def test_compact_ordering_really_is_prism_first(self):
        """自检：构造出来的紧凑排列确实是"棱柱在前"。

        否则上一条是在对照一个不成立的前提。
        """
        _, _, _, cct, _ = self._build()
        n_prism = int((cct == 0).sum())
        assert np.all(cct[:n_prism] == 0) and np.all(cct[n_prism:] == 1)

    def test_forgetting_inv_perm_is_detectably_wrong(self):
        """漏掉 `inv_perm`（直接切紧凑排列的前 n_local 项）必须被检出。

        这正是多 GPU 那个缺陷的另一种形态：紧凑排列的前 n_local 项与
        原生排列的 local 段是完全不同的两批单元。
        """
        truth, perm, inv_perm, cct, n_local = self._build()
        wrong = (cct[:n_local] == 0)
        assert not np.array_equal(wrong, truth[:n_local]), (
            "构造的算例区分不了'漏掉 inv_perm'，请换 seed")


class TestNoSilentClamp:
    """`n_prism > n_cells` 必须报错，不能被切片静默钳掉。"""

    def test_array_builder_rejects_out_of_range_n_prism(self):
        field, o, n, bnd, _ = _case()
        n_cells, n_sps = field.shape[0], field.shape[1]
        Fp, Ft = _two_distinct_filters(n_sps)
        with pytest.raises(ValueError, match="n_prism"):
            build_sensor_gated_filter_func_arrays(
                n_cells, n_sps, 1, Fp, Ft, n_prism=n_cells + 1,
                sensor="bounds", owner_cell=o, neighbor_cell=n,
                is_boundary=bnd, freestream=_FREESTREAM)

    def test_n_prism_equal_to_n_cells_is_allowed(self):
        """全棱柱网格是合法的，护栏不能把它一起拦掉。"""
        field, o, n, bnd, _ = _case()
        n_cells, n_sps = field.shape[0], field.shape[1]
        Fp, Ft = _two_distinct_filters(n_sps)
        ff = build_sensor_gated_filter_func_arrays(
            n_cells, n_sps, 1, Fp, Ft, n_prism=n_cells,
            sensor="bounds", owner_cell=o, neighbor_cell=n,
            is_boundary=bnd, freestream=_FREESTREAM)
        flat = field.reshape(n_cells * n_sps, 5).copy()
        assert ff(flat.copy()).shape == flat.shape

    def test_silent_clamp_would_have_applied_prism_matrix_to_tets(self):
        """复现"静默钳"的后果，说明这条护栏为什么不是形式主义。

        `U[:n_prism]` 在 `n_prism > n_cells` 时被钳到 `n_cells`，于是
        `U[n_prism:]`（四面体那一支）是**空切片**——四面体一个也没被
        它自己的矩阵处理，反而全被前一支的棱柱矩阵覆盖了。
        """
        n_cells, n_sps = 8, 4
        U = np.arange(n_cells * n_sps * 5, dtype=float).reshape(n_cells, n_sps, 5)
        Fp, Ft = _two_distinct_filters(n_sps)
        n_prism_wrong = 1000                        # "全局棱柱数"

        got = U.copy()
        got[:n_prism_wrong, :, :5] = np.einsum(
            "sj,cjv->csv", Fp, got[:n_prism_wrong, :, :5])
        got[n_prism_wrong:, :, :5] = np.einsum(
            "sj,cjv->csv", Ft, got[n_prism_wrong:, :, :5])

        all_prism = np.einsum("sj,cjv->csv", Fp, U[:, :, :5])
        np.testing.assert_allclose(got[:, :, :5], all_prism, rtol=0, atol=0)
        # 而正确结果（一半棱柱一半四面体）与它明显不同
        cip = np.arange(n_cells) < n_cells // 2
        correct = np.where(cip[:, None, None],
                           np.einsum("sj,cjv->csv", Fp, U[:, :, :5]),
                           np.einsum("sj,cjv->csv", Ft, U[:, :, :5]))
        assert not np.allclose(got[:, :, :5], correct)


class TestGpuEntryPointsRequireCupy:
    """直接吃 CuPy 的入口在本机 skip——但必须确认它们**存在且可导入**，
    而不是让"GPU 已接线"这句话无从核对。"""

    def test_single_gpu_method_exists(self):
        from autoflowcfd.core.gpu.solver.gpu_solver_init import (
            _GPUSolverInitMixin,
        )
        assert hasattr(_GPUSolverInitMixin, "_build_sensor_gated_filter_gpu")

    def test_multi_gpu_method_exists(self):
        from autoflowcfd.core.gpu.distributed.gpu_distributed_init import (
            _GPUDistributedInitMixin,
        )
        assert hasattr(_GPUDistributedInitMixin,
                       "_build_sensor_gated_filter_distributed_gpu")

    def test_build_gpu_filter_func_accepts_cell_is_prism(self):
        """签名里必须有 `cell_is_prism`——多 GPU 的修复依赖它。"""
        import inspect

        from autoflowcfd.core.gpu.gpu_modal_filter import build_gpu_filter_func
        sig = inspect.signature(build_gpu_filter_func)
        assert "cell_is_prism" in sig.parameters
