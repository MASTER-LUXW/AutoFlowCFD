"""GPU 粘性体积项过积分与 CPU 同一个离散（从 test_real_dof_reductions_and_gpu_gaps.py 拆出，背景见该文件模块文档）。"""

import numpy as np
import pytest

from autoflowcfd.core.gpu.residual.gpu_viscous.volume import _viscous_volume_term_gpu

from tests.unit._module_source import module_source
from tests.unit._gpu_cupy_shim import patch_module_get_cupy
from tests.unit._numpy_as_cupy import NumpyAsCupy


# ===========================================================================
# 粘性体积项去混叠在 GPU 后端上与 CPU 同一个离散
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
