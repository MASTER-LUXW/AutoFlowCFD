"""2026-09-15 系统性审计第二批（A/B/D/E/F/G/H/I）的验证。

这一批修的是六个互相独立、但都属于同一组"缺陷类"的问题。每一类都有
一节测试，节的文档说明判据为什么成立。

## 覆盖的缺陷类与具体修复

- **第 10 类，"把填充/冻结槽位当成真实自由度"**（F/G/H/I）：native 四面体
  的 `n_sps` 槽位里只有前 `n_native=(p+1)(p+2)(p+3)/6` 个是真实解点，其余
  是零填充——它们在初始化时复制真实 SP #0、之后残差行被填零、滤波行是
  单位阵，于是**永远冻结在初始条件上**（实测推进 10 步后与真实 SP#0 相差
  3.4%，order=1 下占一半槽位）。对 SP 轴直接 `.mean(axis=1)` /
  `.min(axis=1)` 就把这些冻结值算进了结果。
- **第 7 类，"同一个开关在不同后端意味着不同的东西"**（A/B）：
  `AFCFD_FILTER_TURB_GATE` 此前只在 CPU 路径接线，粘性体积项去混叠此前只有
  CPU 实现——GPU 后端读不到就按默认路径跑，没有任何提示。（`AFCFD_VISC_OVERINT`
  开关本身 2026-10-01 删除，粘性体积项恒过积分。）
- **第 11 类，"留着但已无作用的代码/参数"**（D）：`compute_global_min_dt`
  零调用方。

## 方法论

GPU 侧的两节用与 `test_gpu_scalar_transport.py` 同一个手法：把
`get_cupy()` 换成"返回 numpy"的替身，直接跑**生产函数本身**，对照已
验证的 CPU 实现，要求逐位相等——同一套公式的两个后端只做了张量库
替换，不是近似关系。
"""

import numpy as np
import pytest

from autoflowcfd.fr.native_padding import (
    native_tet_n_real_sps,
    reduce_per_cell_over_real_sps,
    reduce_rows_over_real_sps,
)

from tests.unit._module_source import module_source


# ===========================================================================
# 第 10 类：只对真实自由度归约
# ===========================================================================
# 第 10 类：只对真实自由度归约
# ===========================================================================

class TestRealDofReductionHelpers:
    """两个归约辅助本身。

    构造方式刻意让"错"与"对"相差**三个**能手算、互不相同的值：真实槽位
    全 1、填充槽位全 50。order=1 下 `n_sps=8`，两类单元的真实前缀分别是
    棱柱 6（原生棱柱 `(p+1)^2(p+2)/2`）与四面体 4（`(p+1)(p+2)(p+3)/6`）：

      - 不屏蔽任何填充（错）：`(4*1 + 4*50)/8 = 25.5`
      - 按棱柱前缀屏蔽：`(4*1 + 2*50)/6 = 17.3333...`
      - 按四面体前缀屏蔽：`1.0`

    三个值两两不同，所以"把棱柱当成四面体切"与"棱柱不切"这两种错法都会
    被判据抓住 —— 这一点在两类单元的真实前缀都小于 `n_sps` 之后才成立
    （坍缩棱柱基时代棱柱用满 8 个槽位，那时前两个值重合）。

    坍缩棱柱基已于 2026-09-23 删除，所以本类不再分档；原先那个只在原生档
    下跑的对偶类 `TestNativePrismIsAlsoMasked` 已并入这里（同一语义两处
    覆盖是本项目明确要精简的重复）。
    """

    def _field(self, n_cells=4, order=1):
        n_sps = (order + 1) ** 3
        n_native = native_tet_n_real_sps(order)
        f = np.empty((n_cells, n_sps))
        f[:, :n_native] = 1.0
        f[:, n_native:] = 50.0
        return f

    def test_n_real_sps_formula(self):
        assert [native_tet_n_real_sps(p) for p in (0, 1, 2, 3)] == [1, 4, 10, 20]

    def test_real_prefix_per_cell_type(self):
        """`real_sps_per_cell` 是"哪些槽位是真的"的唯一判据来源。

        两类单元的真实前缀**都**小于 `n_sps=(p+1)^3`，这正是下面几条手算
        判据能分辨三种错法的前提（并入自原 `TestNativePrismIsAlsoMasked`）。
        """
        from autoflowcfd.fr.native_padding import real_sps_per_cell

        for p, expect in ((1, (6, 4)), (2, (18, 10)), (3, (40, 20))):
            assert real_sps_per_cell(p) == expect, (
                f"P{p} 的 (棱柱, 四面体) 真实自由度数应当是 {expect}")
            assert max(expect) < (p + 1) ** 3

    def test_per_cell_mean_masks_each_cell_types_padding(self):
        f = self._field()
        # 未修正的写法：两类单元都被污染成同一个 25.5
        np.testing.assert_allclose(f.mean(axis=1), 25.5)
        got = reduce_per_cell_over_real_sps(f, 2, 1, 'mean')
        # 前两个是棱柱（只看前 6 个：4 个 1 + 2 个 50），
        # 后两个是四面体（只看前 4 个，得 1.0）
        prism_expect = (4 * 1.0 + 2 * 50.0) / 6.0
        np.testing.assert_allclose(got, [prism_expect, prism_expect, 1.0, 1.0])

    def test_per_cell_min_max_masks_each_cell_types_padding(self):
        f = self._field()
        np.testing.assert_allclose(
            reduce_per_cell_over_real_sps(f, 2, 1, 'max'), [50.0, 50.0, 1.0, 1.0])
        np.testing.assert_allclose(
            reduce_per_cell_over_real_sps(f, 2, 1, 'min'), [1.0, 1.0, 1.0, 1.0])

    def test_rows_version_uses_per_row_mask(self):
        """按 owner_cell 索引出的逐面数组里单元类型任意混合，不能切片。"""
        f = self._field()
        row_is_prism = np.array([True, False, True, False])
        prism_expect = (4 * 1.0 + 2 * 50.0) / 6.0
        np.testing.assert_allclose(
            reduce_rows_over_real_sps(f, row_is_prism, 1, 'mean'),
            [prism_expect, 1.0, prism_expect, 1.0])

    def test_all_prism_slices_exactly_the_prism_prefix(self):
        """`n_prism == n_cells` 时必须**逐位等于**对棱柱前缀的朴素归约。

        这条判据在 2026-09-23 之前写的是"逐位等于全宽度朴素归约"，那只对
        已删除的坍缩棱柱基成立（棱柱用满 `(p+1)^3`）。原生棱柱有填充槽位，
        所以正确的对照是 `field[:, :n_prism_real]` —— 仍然是逐位判据
        （不引入任何多余的算术），只是前缀换对了。
        """
        from autoflowcfd.fr.native_padding import real_sps_per_cell

        n_prism_real, _ = real_sps_per_cell(1)
        rng = np.random.default_rng(0)
        f = rng.normal(size=(6, 8))
        for how in ('mean', 'min', 'max', 'sum'):
            masked = reduce_per_cell_over_real_sps(f, 6, 1, how)
            np.testing.assert_array_equal(
                masked, getattr(np, how)(f[:, :n_prism_real], axis=1))
            # 负对照：全宽度归约**不**应当等于它，否则填充没被屏蔽
            assert not np.allclose(masked, getattr(np, how)(f, axis=1)), (
                f"how={how}：全宽度归约与屏蔽后归约相等，判据失去分辨力")

    @pytest.mark.parametrize("how", ["median", "prod", ""])
    def test_rejects_unsupported_reduction(self, how):
        with pytest.raises(ValueError):
            reduce_per_cell_over_real_sps(np.zeros((2, 8)), 1, 1, how)

    def test_rejects_order_inconsistent_with_n_sps(self):
        """阶数与数组 SP 轴不自洽必须显式报错。

        这不是防御性冗余：Order Continuation 期间网格几何量的 `n_sps` 可以
        短暂地属于另一个阶数，按错的 `n_native` 切片会**静默**把真实解点
        当填充丢掉（或反过来）。本项目在跨阶数缓存上已经复现过同一类真实
        bug（见 `gpu_solver.py::_compute_local_time_step_gpu` 里
        metric_flux_scale 缓存那段注释）。
        """
        f = np.zeros((3, 8))  # n_sps=8 -> 只能是 order=1
        for bad_order in (0, 2, 3):
            with pytest.raises(ValueError, match="不自洽"):
                reduce_per_cell_over_real_sps(f, 1, bad_order, 'mean')
            with pytest.raises(ValueError, match="不自洽"):
                reduce_rows_over_real_sps(
                    f, np.zeros(3, dtype=bool), bad_order, 'mean')


class TestRealDofReductionCallSitesAreWired:
    """六处调用点确实改成了掩码版。

    源码断言在这里是**恰当**的判据而不是偷懒：这几处的"错"与"对"在
    只有棱柱、或者填充恰好还没变馊的合成数据上**数值完全相同**（填充
    初始化时就是真实 SP#0 的副本），差异只在真实长程推进之后出现。要在
    单元测试里制造出差异，就得把求解器推进到填充变馊——那是一个真实
    网格算例，不是单元测试。所以这里钉住"调用的是哪个函数"，数值正确性
    由上面 `TestRealDofReductionHelpers` 保证。
    """

    def test_artificial_viscosity_rho_and_vel_scale(self):
        # 读**真正含有调用点**的那个子模块（包 `__init__` 只 re-export）。人工
        # 粘性 2026-10-01 改为熵残差判据，单元类型可交错（分布式），所以用逐行
        # 掩码版的归约。
        from autoflowcfd.core.fr_operators.artificial_viscosity import (
            entropy_viscosity as av,
        )
        s = module_source(av)
        assert "reduce_rows_over_real_sps(speed, cip, order, 'mean', xp=xp)" in s
        assert "reduce_rows_over_real_sps(proj, cip, order, 'max', xp=xp)" in s
        assert ".mean(axis=1)" not in s and ".max(axis=1)" not in s

    def test_turbulence_transport_rho_owner(self):
        from autoflowcfd.core.turbulence import transport
        s = module_source(transport)
        assert "reduce_rows_over_real_sps(" in s
        assert "rho[owner_cells].mean(axis=1)" not in s

    def test_gpu_scalar_transport_rho_owner(self):
        from autoflowcfd.core.gpu.turbulence import gpu_scalar_transport as gst
        s = module_source(gst)
        assert "reduce_rows_over_real_sps(" in s

    def test_checkpoint_cell_average_both_sites(self):
        from autoflowcfd.cli.solve import checkpoint_io as solve_checkpoint_io
        from autoflowcfd.core.mpi import distributed_checkpoint
        s1 = module_source(solve_checkpoint_io)
        assert "reduce_per_cell_over_real_sps(" in s1
        assert "solver.state.U.mean(axis=1)" not in s1
        s2 = module_source(distributed_checkpoint)
        # 2026-10-04 起分布式写出端收集逐单元棱柱标志（两种分布式模式都适用），用逐行掩码版
        # 归约；此前完全分布式加载下退回含零填充槽位的全场平均
        assert "reduce_rows_over_real_sps(" in s2 and "global_cell_is_prism(" in s2
        assert "U_global.mean(axis=1)" not in s2

    def test_gpu_local_dt_both_sites(self):
        """GPU 的逐单元 dt 是 `min(axis=1)` 归约，必须掩码。

        CPU 侧 `cfl.py::compute_local_time_step` 返回的是逐 SP 的
        (n_cells, n_sps) 数组、填充槽位的 dt 只会乘到恒为零的残差上，所以
        那边不受影响；GPU 这边归约成逐单元一个标量，冻结槽位就真的参与了
        竞争（min 取"最大波速/最大 mu_eff"那一侧，典型来流初始化下会把 dt
        压得偏小）。
        """
        from autoflowcfd.core.gpu.distributed import gpu_distributed
        from autoflowcfd.core.gpu.solver import gpu_solver
        for mod in (gpu_solver, gpu_distributed):
            s = module_source(mod)
            assert "reduce_per_cell_over_real_sps(" in s, mod.__name__
            assert "cp.min(dt_all_sps, axis=1)" not in s, mod.__name__
            assert "cp.min(dt_all, axis=1)" not in s, mod.__name__


# ===========================================================================
# D 类：死代码已删除
# ===========================================================================

class TestDeadCodeRemoved:
    def test_compute_global_min_dt_is_gone(self):
        """零调用方的 `compute_global_min_dt`。

        分布式路径的步长一律走 `core/mpi/distributed_cfl.py` 的逐单元
        局部步长（与单机共用 `cfl.py::compute_local_time_step` 同一个
        函数），从来不需要归约成一个全局标量。
        """
        from autoflowcfd.core.mpi.distributed_solver import DistributedFRSolver
        assert not hasattr(DistributedFRSolver, "compute_global_min_dt")

    def test_unused_import_dropped(self):
        # **否定式**断言，必须拼上全部子模块：distributed_solver 一旦拆成
        # 子包，inspect.getsource(包) 只返回 __init__.py，这条会静默通过。
        assert "allreduce_min" not in module_source(
            "autoflowcfd.core.mpi.distributed_solver")
