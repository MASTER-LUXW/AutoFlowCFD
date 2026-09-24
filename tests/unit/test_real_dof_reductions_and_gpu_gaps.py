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
  `AFCFD_FILTER_TURB_GATE` 此前只在 CPU 路径接线，`AFCFD_VISC_OVERINT`
  此前只有 CPU 实现——GPU 后端读不到就按默认路径跑，没有任何提示。
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

from tests.unit._module_source import module_source

from autoflowcfd.fr.native_padding import (
    native_tet_n_real_sps,
    reduce_per_cell_over_real_sps,
    reduce_rows_over_real_sps,
)
from tests.unit._gpu_cupy_shim import patch_module_get_cupy


class _NumpyAsCupy:
    """把 numpy 伪装成 CuPy 模块接口（同 test_gpu_scalar_transport.py）。"""

    def __getattr__(self, name):
        return getattr(np, name)

    def scatter_add(self, a, indices, b):
        np.add.at(a, indices, b)

    def asnumpy(self, x):
        return np.asarray(x)


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
        # 读**真正含有调用点**的那个子模块：人工粘性模块 2026-09-20 拆成
        # 子包，包 `__init__` 只 re-export，源文本里没有调用点。
        from autoflowcfd.core.fr_operators.artificial_viscosity import (
            viscosity as av,
        )
        s = module_source(av)
        assert "reduce_per_cell_over_real_sps(rho, n_prism, order, 'mean')" in s
        assert "reduce_per_cell_over_real_sps(vel_mag, n_prism, order, 'mean')" in s
        assert "rho.mean(axis=1)" not in s

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
        from autoflowcfd.cli import solve_checkpoint_io
        from autoflowcfd.core.mpi import distributed_checkpoint
        s1 = module_source(solve_checkpoint_io)
        assert "reduce_per_cell_over_real_sps(" in s1
        assert "solver.state.U.mean(axis=1)" not in s1
        s2 = module_source(distributed_checkpoint)
        assert "reduce_per_cell_over_real_sps(" in s2
        # 完全分布式模式下拿不到全局棱柱数，退回全场平均是**显式记录**的
        # 有界失真，不是静默兜底——判据是那段说明必须在源码里，且必须说明
        # 精确数据走的是 U_sps。
        assert "_fully_distributed" in s2
        assert "U_sps" in s2

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
# A 类：AFCFD_FILTER_TURB_GATE 在 GPU 后端上生效
# ===========================================================================

class TestGpuTurbFilterGate:
    """GPU 版传感器 + 门控滤波必须与已验证的 CPU 版逐位一致。

    `AFCFD_FILTER_TURB_GATE=sensor` 此前只在单机 CPU 与 CPU MPI 上接线
    （后者经同一个 `compute_turbulence_source`，其 `turb_view`/
    `mesh_adapter` 都在 compact"棱柱在前"索引空间，所以同一段 n_prism
    切片代码两条路径都正确）；单 GPU 与多 GPU 直接调用
    `filter_scalar_field_gpu` 无条件全场滤波。
    """

    @pytest.fixture(autouse=True)
    def _patch(self, monkeypatch):
        import autoflowcfd.core.gpu.gpu_modal_filter as gmf
        import autoflowcfd.core.gpu.gpu_troubled_cell as gtc
        shim = _NumpyAsCupy()
        patch_module_get_cupy(monkeypatch, gmf, shim)
        patch_module_get_cupy(monkeypatch, gtc, shim)
        gtc._gpu_sensor_cache.clear()

    @pytest.fixture
    def fields(self):
        order = 2
        n_sps = (order + 1) ** 3
        n_prism, n_cells = 5, 11
        rng = np.random.default_rng(7)
        # 一半单元光滑（低阶多项式，最高阶模态能量几乎为零）、一半单元
        # 是白噪声（最高阶模态占比大）——这样传感器必须给出**非平凡**的
        # 掩码，而不是全 True 或全 False（那样 gated 与全场版就无法区分）。
        k = np.empty((n_cells, n_sps))
        om = np.empty((n_cells, n_sps))
        for c in range(n_cells):
            if c % 2 == 0:
                k[c] = 1.0 + 0.01 * np.arange(n_sps) / n_sps
                om[c] = 100.0 + 0.5 * np.arange(n_sps) / n_sps
            else:
                k[c] = rng.uniform(1e-3, 1.0, n_sps)
                om[c] = rng.uniform(10.0, 1e4, n_sps)
        return order, n_prism, n_cells, k, om

    def test_troubled_mask_matches_cpu_bitwise(self, fields):
        from autoflowcfd.core.fr_solver.filter import compute_turb_troubled_mask
        from autoflowcfd.core.gpu.gpu_troubled_cell import (
            compute_turb_troubled_mask_gpu,
        )
        order, n_prism, n_cells, k, om = fields
        cpu = compute_turb_troubled_mask(k, om, order, n_prism=n_prism)
        gpu = compute_turb_troubled_mask_gpu(k, om, n_prism, order)
        np.testing.assert_array_equal(np.asarray(gpu), cpu)
        # 判据必须非平凡，否则这个测试什么都没测
        assert 0 < cpu.sum() < n_cells

    def test_order_zero_mask_is_all_false(self, fields):
        from autoflowcfd.core.gpu.gpu_troubled_cell import (
            compute_troubled_cell_mask_gpu,
        )
        m = compute_troubled_cell_mask_gpu(np.ones((4, 1)), 2, 0)
        assert not np.any(np.asarray(m))

    def test_gated_filter_matches_cpu_bitwise(self, fields):
        from autoflowcfd.core.fr_solver.filter import filter_scalar_field_gated
        from autoflowcfd.core.gpu.gpu_modal_filter import (
            filter_scalar_field_gated_gpu,
        )
        from autoflowcfd.fr.operators import generate_fr_operators
        order, n_prism, n_cells, k, om = fields
        ops = generate_fr_operators(order)
        troubled = np.zeros(n_cells, dtype=bool)
        troubled[1::2] = True
        cpu = filter_scalar_field_gated(
            k, ops.filter_prism, ops.filter_tet, troubled, n_prism=n_prism)
        gpu = filter_scalar_field_gated_gpu(
            k, n_prism, ops.filter_prism, ops.filter_tet, troubled)
        np.testing.assert_allclose(np.asarray(gpu), cpu, rtol=1e-13, atol=1e-300)

    def test_gated_all_true_equals_ungated(self, fields):
        """全 True 时必须与无门控版一致（门控只改"对哪些单元施加"）。"""
        from autoflowcfd.core.gpu.gpu_modal_filter import (
            filter_scalar_field_gated_gpu, filter_scalar_field_gpu,
        )
        from autoflowcfd.fr.operators import generate_fr_operators
        order, n_prism, n_cells, k, om = fields
        ops = generate_fr_operators(order)
        a = filter_scalar_field_gpu(k, n_prism, ops.filter_prism, ops.filter_tet)
        b = filter_scalar_field_gated_gpu(
            k, n_prism, ops.filter_prism, ops.filter_tet,
            np.ones(n_cells, dtype=bool))
        np.testing.assert_array_equal(np.asarray(a), np.asarray(b))

    def test_gated_all_false_is_identity(self, fields):
        from autoflowcfd.core.gpu.gpu_modal_filter import (
            filter_scalar_field_gated_gpu,
        )
        from autoflowcfd.fr.operators import generate_fr_operators
        order, n_prism, n_cells, k, om = fields
        ops = generate_fr_operators(order)
        out = filter_scalar_field_gated_gpu(
            k, n_prism, ops.filter_prism, ops.filter_tet,
            np.zeros(n_cells, dtype=bool))
        np.testing.assert_array_equal(np.asarray(out), k)

    def test_both_gpu_paths_are_wired(self):
        """两条 GPU 调用路径都必须真的读了这个开关。"""
        from autoflowcfd.core.gpu.distributed import gpu_distributed_init
        from autoflowcfd.core.gpu.solver import gpu_solver_io
        # gpu_distributed_init 2026-09-24 拆成子包；inspect.getsource(包)
        # 只返回 __init__.py。
        for mod in (gpu_solver_io, gpu_distributed_init):
            s = module_source(mod)
            assert "resolve_turb_filter_gate()" in s, mod.__name__
            assert "filter_scalar_field_gated_gpu" in s, mod.__name__
            assert "compute_turb_troubled_mask_gpu" in s, mod.__name__

    def test_default_gate_keeps_ungated_path(self):
        """默认 "all" 必须仍走无门控分支（既有行为逐位不变）。"""
        from autoflowcfd.core.fr_solver.filter import resolve_turb_filter_gate
        assert resolve_turb_filter_gate() == "all"


# ===========================================================================
# B 类：AFCFD_VISC_OVERINT 在 GPU 后端上生效
# ===========================================================================

class TestGpuViscousOverintegration:
    """GPU 版粘性体积项去混叠必须与已验证的 CPU 版一致。

    对照的是 CPU 的 `_viscous_volume_overintegrated`——同一条五步链路
    （插值到细点 / 在细点重新求值非线性通量 / 细点度量 / 细网格微分 /
    限制回 coarse），GPU 版只把张量库换掉，所以要求的是到浮点重排误差
    的相等，不是"接近"。
    """

    @pytest.fixture(scope="class")
    def case(self):
        from autoflowcfd.core.fr_operators.volume_contract import (
            get_overintegration_context,
        )
        from autoflowcfd.fr.operators import generate_fr_operators
        from tests.unit.test_fr_residual_inviscid import _build_synthetic_mixed_mesh
        order = 2
        mesh = _build_synthetic_mixed_mesh(order)
        ops = generate_fr_operators(order)
        oi = get_overintegration_context(mesh, ops)
        assert oi is not None, "order=2 必须有过积分上下文，否则本测试无意义"
        n_cells, n_sps = mesh.n_cells, mesh.n_sps_per_cell
        rng = np.random.default_rng(11)
        Q = np.empty((n_cells, n_sps, 5))
        Q[..., 0] = rng.uniform(1.0, 1.4, (n_cells, n_sps))
        Q[..., 1:4] = rng.uniform(-30.0, 30.0, (n_cells, n_sps, 3))
        Q[..., 4] = rng.uniform(9.9e4, 1.03e5, (n_cells, n_sps))
        grad_vel = rng.uniform(-50.0, 50.0, (n_cells, n_sps, 3, 3))
        grad_T = rng.uniform(-200.0, 200.0, (n_cells, n_sps, 3))
        mu_t = rng.uniform(0.0, 1e-3, (n_cells, n_sps))
        return mesh, ops, oi, Q, grad_vel, grad_T, mu_t

    def _gpu_div(self, monkeypatch, case, mu_t_arg):
        import autoflowcfd.core.gpu.residual.gpu_flux as gpu_flux_mod
        import autoflowcfd.core.gpu.residual.gpu_viscous as gv
        import autoflowcfd.core.gpu.residual.gpu_volume_contract as gvc
        shim = _NumpyAsCupy()
        patch_module_get_cupy(monkeypatch, gv, shim)
        patch_module_get_cupy(monkeypatch, gvc, shim)
        patch_module_get_cupy(monkeypatch, gpu_flux_mod, shim)

        mesh, ops, oi, Q, grad_vel, grad_T, mu_t = case
        n_cells, n_sps = mesh.n_cells, mesh.n_sps_per_cell
        # 细点度量现在由 `segs` 每段自带（2026-09-17）：四面体过积分的
        # 细网格轴不再填充到棱柱宽度，两段 n_fine 不同，所以不再有一份
        # 共享的 adj_j_fine 可以整场预乘。这里把 CPU 上下文的每段
        # (det, inv) 预乘成 adj，拼成 GPU helper 的段格式。
        gpu_segs = tuple(
            (lo, hi, n_fine,
             np.ascontiguousarray(det_seg)[..., None, None]
             * np.ascontiguousarray(inv_seg),
             c2f, D_fine, f2c)
            for (lo, hi, n_fine, det_seg, inv_seg,
                 c2f, D_fine, f2c) in oi["segs"]
        )
        return gv._viscous_volume_overintegrated_gpu(
            shim, Q, grad_vel, grad_T, mu_t_arg, 1.8e-5, 0.72, 0.9,
            gpu_segs, n_cells, n_sps)

    def test_matches_cpu_with_turbulent_viscosity(self, monkeypatch, case):
        from autoflowcfd.core.fr_residual.viscous_flux import (
            _viscous_volume_overintegrated,
        )
        mesh, ops, oi, Q, grad_vel, grad_T, mu_t = case
        cpu = _viscous_volume_overintegrated(
            Q, grad_vel, grad_T, mu_t, 1.8e-5, 0.72, 0.9, oi,
            mesh.n_sps_per_cell)
        gpu = self._gpu_div(monkeypatch, case, mu_t)
        np.testing.assert_allclose(np.asarray(gpu), cpu, rtol=1e-11,
                                   atol=1e-11 * np.abs(cpu).max())

    def test_matches_cpu_laminar_zero_mu_t(self, monkeypatch, case):
        """层流（mu_t 全零数组）与"标量 0.0"两种传参必须给同一个结果。"""
        from autoflowcfd.core.fr_residual.viscous_flux import (
            _viscous_volume_overintegrated,
        )
        mesh, ops, oi, Q, grad_vel, grad_T, mu_t = case
        zero = np.zeros_like(mu_t)
        cpu = _viscous_volume_overintegrated(
            Q, grad_vel, grad_T, zero, 1.8e-5, 0.72, 0.9, oi,
            mesh.n_sps_per_cell)
        tol = 1e-11 * np.abs(cpu).max()
        g_arr = self._gpu_div(monkeypatch, case, zero)
        g_scalar = self._gpu_div(monkeypatch, case, 0.0)
        np.testing.assert_allclose(np.asarray(g_arr), cpu, rtol=1e-11, atol=tol)
        np.testing.assert_allclose(np.asarray(g_scalar), cpu, rtol=1e-11, atol=tol)

    def test_differs_from_coarse_path(self, monkeypatch, case):
        """去混叠必须真的改变结果，否则这个开关是无操作。"""
        from autoflowcfd.core.fr_operators.volume_contract import (
            contract_shared_operator_2axis, contravariant_flux_from_metric,
        )
        from autoflowcfd.core.fr_operators.flux_kernels import (
            viscous_physical_flux_batch,
        )
        mesh, ops, oi, Q, grad_vel, grad_T, mu_t = case
        n_cells, n_sps = mesh.n_cells, mesh.n_sps_per_cell
        det = mesh.jacobians['det_jacs'].reshape(n_cells, n_sps)
        inv = mesh.jacobians['inv_jacs'].reshape(n_cells, n_sps, 3, 3)
        G = viscous_physical_flux_batch(
            np.ascontiguousarray(Q.reshape(-1, 5)),
            np.ascontiguousarray(grad_vel.reshape(-1, 3, 3)),
            np.ascontiguousarray(grad_T.reshape(-1, 3)),
            1.8e-5, 0.72, np.ascontiguousarray(mu_t.reshape(-1)), 0.9,
        ).reshape(n_cells, n_sps, 3, 5)
        G_tilde = contravariant_flux_from_metric(det, inv, G)
        coarse = np.zeros((n_cells, n_sps, 5))
        _tet_D = (ops.D_native_tet_padded
                  if getattr(ops, "D_native_tet_padded", None) is not None
                  else ops.D_3d_tet)
        coarse[:mesh.n_prism_cells] = contract_shared_operator_2axis(
            ops.D_3d_prism, G_tilde[:mesh.n_prism_cells])
        coarse[mesh.n_prism_cells:] = contract_shared_operator_2axis(
            _tet_D, G_tilde[mesh.n_prism_cells:])
        fine = np.asarray(self._gpu_div(monkeypatch, case, mu_t))
        assert not np.allclose(fine, coarse, rtol=1e-6)

    def test_switch_is_read_in_gpu_path(self):
        from autoflowcfd.core.gpu.residual import gpu_viscous
        s = module_source(gpu_viscous)
        assert "resolve_viscous_overintegration()" in s
        assert "_viscous_volume_overintegrated_gpu(" in s

    def test_default_is_on_so_gpu_must_mirror_the_fine_path(self):
        """默认 2026-09-17 从 `off` 改成 `on`（依据见
        `viscous_flux.py::resolve_viscous_overintegration`）。

        对 GPU 侧的含义变了：过积分分支从"默认不走、只需存在"变成
        **默认路径**，所以 GPU 的分段必须与 CPU 的
        `get_overintegration_context` 同构（下一条测试查这个）。
        """
        from autoflowcfd.core.fr_residual.viscous_flux import (
            resolve_viscous_overintegration,
        )
        assert resolve_viscous_overintegration() == "on"

    def test_shared_overint_context_is_backend_consistent(self):
        """GPU 的分段必须与 CPU 的 `get_overintegration_context` 同构。"""
        from autoflowcfd.core.fr_operators.volume_contract import (
            get_overintegration_context,
        )
        from autoflowcfd.core.gpu.gpu_overintegration import (
            OVERINT_OPS_KEYS, get_overintegration_segs_gpu,
        )
        from autoflowcfd.fr.operators import generate_fr_operators
        from tests.unit.test_fr_residual_inviscid import _build_synthetic_mixed_mesh
        order = 2
        mesh = _build_synthetic_mixed_mesh(order)
        ops = generate_fr_operators(order)
        cpu = get_overintegration_context(mesh, ops)
        ops_data = {k: getattr(ops, k) for k in OVERINT_OPS_KEYS}
        # GPU helper 现在要从 `adj_j_fine` 的形状读棱柱段的细点宽度、并按段
        # 切度量，所以替身必须有真实形状（2026-09-17）。
        n_fine_prism = cpu["segs"][0][2]
        gpu = get_overintegration_segs_gpu(
            {'adj_j_fine': np.zeros((mesh.n_cells, n_fine_prism, 3, 3))},
            ops_data, mesh.n_cells, mesh.n_prism_cells)
        assert gpu is not None
        assert [(lo, hi) for lo, hi, *_ in gpu] == \
               [(lo, hi) for lo, hi, *_ in cpu["segs"]]
        # 两端每段的 n_fine 必须一致——这正是"同一份配置下两个后端跑的是
        # 同一个数值方案"的核心不变量
        assert [nf for _lo, _hi, nf, *_ in gpu] == \
               [nf for _lo, _hi, nf, *_ in cpu["segs"]]

    def test_missing_keys_fall_back_to_coarse(self):
        """缺任何一个算子/细点度量都必须返回 None（退回 coarse），而不是
        崩在半路——`order == 0` 是这条分支的正常情形。"""
        from autoflowcfd.core.gpu.gpu_overintegration import (
            OVERINT_OPS_KEYS, get_overintegration_segs_gpu,
        )
        full = {k: object() for k in OVERINT_OPS_KEYS}
        assert get_overintegration_segs_gpu({}, full, 4, 2) is None
        for k in OVERINT_OPS_KEYS:
            partial = dict(full)
            del partial[k]
            assert get_overintegration_segs_gpu(
                {'adj_j_fine': np.zeros((4, 64, 3, 3))},
                partial, 4, 2) is None, k


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
