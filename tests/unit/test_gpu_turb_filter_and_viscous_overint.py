"""GPU 湍流滤波门控与 GPU 粘性过积分（从 test_real_dof_reductions_and_gpu_gaps.py 拆出，背景见该文件模块文档）。"""

import numpy as np
import pytest

from autoflowcfd.core.gpu.residual.gpu_viscous.volume import _viscous_volume_term_gpu

from tests.unit._module_source import module_source
from tests.unit._gpu_cupy_shim import patch_module_get_cupy
from tests.unit._numpy_as_cupy import NumpyAsCupy


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
        shim = NumpyAsCupy()
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
        cpu = compute_turb_troubled_mask((k, om), order, n_prism=n_prism)
        gpu = compute_turb_troubled_mask_gpu((k, om), n_prism, order)
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
        """单机与多 GPU 都走全部后端共用的 `core/turbulence/unknown_filter.py::filter_turbulence_unknowns`
        并注入 GPU 滤波核（2026-10-04 起；此前是 SST 专用的 `GPUTurbulenceSST.filter_fields_gpu`）。"""
        from autoflowcfd.core.gpu.distributed import gpu_distributed_init
        from autoflowcfd.core.gpu.solver import gpu_solver_io
        # gpu_distributed_init 2026-09-24 拆成子包；inspect.getsource(包)
        # 只返回 __init__.py。
        for mod in (gpu_solver_io, gpu_distributed_init):
            src = module_source(mod)
            assert "filter_turbulence_unknowns(" in src and "GpuFilterKernels()" in src, mod.__name__

    @pytest.mark.parametrize("model_name", ["SST", "SA"])
    def test_gpu_kernels_match_cpu_kernels(self, fields, monkeypatch, model_name):
        """sensor 门控下 GPU 滤波核与 CPU 滤波核在同一状态上给出同一结果：传感器看 Newton 未知量
        （SST 为 (k, ln omega)、SA 为 nu_tilde），滤波也作用在它们上。"""
        from types import SimpleNamespace

        import autoflowcfd.core.gpu.gpu_modal_filter as gmf
        import autoflowcfd.core.gpu.gpu_troubled_cell as gtc
        from autoflowcfd.core.fr_solver.filter import CpuFilterKernels
        from autoflowcfd.core.turbulence.sa import SAModel
        from autoflowcfd.core.turbulence.sst import SSTModelFR
        from autoflowcfd.core.turbulence.unknown_filter import filter_turbulence_unknowns

        shim = NumpyAsCupy()
        patch_module_get_cupy(monkeypatch, [gmf, gtc], shim)
        monkeypatch.setenv("AFCFD_FILTER_TURB_GATE", "sensor")
        order, n_prism, n_cells, k, om = fields
        n_sps = k.shape[1]
        rng = np.random.default_rng(3)
        # 滤波档在导入时定（默认 off 即单位阵、整段跳过）：直接给一对非单位矩阵
        ops = SimpleNamespace(filter_prism=np.eye(n_sps) - 0.05 * rng.random((n_sps, n_sps)),
                              filter_tet=np.eye(n_sps) - 0.05 * rng.random((n_sps, n_sps)))

        def build(xp):
            if model_name == "SA":
                m = SAModel(n_cells, n_sps, nu_ref=1.5e-5, viscosity_ratio=3.0, xp=xp)
                m.nu_tilde_field = k * 1e-4
            else:
                m = SSTModelFR(n_cells, n_sps, k_inf=1e-3, omega_inf=10.0)
                m.omega_max = 1e8
                m.k_field, m.omega_field = k.copy(), om.copy()
            return m

        mc, mg = build(np), build(shim)
        frac_c = filter_turbulence_unknowns(mc, CpuFilterKernels, n_prism, order, ops)
        frac_g = filter_turbulence_unknowns(mg, gmf.GpuFilterKernels(), n_prism, order, ops)
        assert 0.0 < frac_c < 1.0 and frac_g == frac_c
        for fg, fc, f0 in zip(mg.transported_fields(), mc.transported_fields(), build(np).transported_fields()):
            assert not np.array_equal(fc, f0), "滤波应改变被标记单元"
            np.testing.assert_allclose(fg, fc, rtol=1e-12, atol=1e-300)

    def test_default_gate_keeps_ungated_path(self):
        """默认 "all" 必须仍走无门控分支（既有行为逐位不变）。"""
        from autoflowcfd.core.fr_solver.filter import resolve_turb_filter_gate
        assert resolve_turb_filter_gate() == "all"


# ===========================================================================
# B 类：粘性体积项去混叠在 GPU 后端上与 CPU 同一个离散
# ===========================================================================

class TestGpuViscousOverintegration:
    """GPU 版粘性体积项必须与已验证的 CPU 版一致。

    对照的是 CPU 的 `viscous_volume_term`——同一条链路（插值到细点 / 在细点重新
    求值非线性通量 / 细点度量 / 与体积算子 K 收缩），GPU 版只把张量库换掉，所以
    要求的是到浮点重排误差的相等，不是"接近"。
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
        shim = NumpyAsCupy()
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
        return _viscous_volume_term_gpu(
            shim, Q, grad_vel, grad_T, mu_t_arg, 1.8e-5, 0.72, 0.9,
            gpu_segs, oi["lifted_div"], n_cells, n_sps)

    def test_matches_cpu_with_turbulent_viscosity(self, monkeypatch, case):
        from autoflowcfd.core.fr_residual.viscous_flux import viscous_volume_term
        mesh, ops, oi, Q, grad_vel, grad_T, mu_t = case
        cpu = viscous_volume_term(
            Q, grad_vel, grad_T, mu_t, 1.8e-5, 0.72, 0.9, oi,
            mesh.n_sps_per_cell)
        gpu = self._gpu_div(monkeypatch, case, mu_t)
        np.testing.assert_allclose(np.asarray(gpu), cpu, rtol=1e-11,
                                   atol=1e-11 * np.abs(cpu).max())

    def test_matches_cpu_laminar_zero_mu_t(self, monkeypatch, case):
        """层流（mu_t 全零数组）与"标量 0.0"两种传参必须给同一个结果。"""
        from autoflowcfd.core.fr_residual.viscous_flux import viscous_volume_term
        mesh, ops, oi, Q, grad_vel, grad_T, mu_t = case
        zero = np.zeros_like(mu_t)
        cpu = viscous_volume_term(
            Q, grad_vel, grad_T, zero, 1.8e-5, 0.72, 0.9, oi,
            mesh.n_sps_per_cell)
        tol = 1e-11 * np.abs(cpu).max()
        g_arr = self._gpu_div(monkeypatch, case, zero)
        g_scalar = self._gpu_div(monkeypatch, case, 0.0)
        np.testing.assert_allclose(np.asarray(g_arr), cpu, rtol=1e-11, atol=tol)
        np.testing.assert_allclose(np.asarray(g_scalar), cpu, rtol=1e-11, atol=tol)

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
        # GPU helper 从上传的棱柱细点度量形状对账细点宽度，所以替身必须有真实形状。
        n_fine_prism = cpu["segs"][0][2]
        n_prism = mesh.n_prism_cells
        gpu = get_overintegration_segs_gpu(
            {'adj_j_fine_prism': np.zeros((n_prism, n_fine_prism, 3, 3)),
             'adj_j_fine_tet': np.zeros((mesh.n_cells - n_prism, 3, 3))},
            ops_data, mesh.n_cells, n_prism)
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
                {'adj_j_fine_prism': np.zeros((2, 64, 3, 3)), 'adj_j_fine_tet': np.zeros((2, 3, 3))},
                partial, 4, 2) is None, k
