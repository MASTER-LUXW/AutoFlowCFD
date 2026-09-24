"""GPU 版 k/omega 标量输运（gpu_scalar_transport.py）`tet_basis_mode=
"native"` 组合的完整补全验证（2026-09-03，"把 native 全部补充完整"排查
的最后一块拼图）。

排查前的现状（见 `native_vs_collapsed_gpu_readiness_2026_09_03` 记忆）：
本模块此前**完全没有 native 分派**——不是"crash bug"级别的缺陷，是真正
未实现的功能缺口：
1. `_extrapolate_scalar_to_faces_gpu` 自身外插无条件用
   `boundary_extrap[celltype,axis,side_idx]`，native 面 `axis`（复用的
   excluded_vertex，取值 0~3）会在只有 3 个轴的表上越界。
2. `_distribute_scalar_correction_gpu` 面校正分配同样无条件走 collapsed
   的 1D 修正函数分布（`distribute_face_correction_to_sps`），既会在
   同一张轴表上越界，修复越界后也仍然是错误结果——native 单纯形基没有
   "坍缩计算方向"，必须走 `lift_native` DG 提升算子。
3. 附带发现：`_distribute_scalar_correction_gpu` 此前完全没有
   `owner_is_primary`/`neighbor_is_primary` 过滤（旧模块文档曾声称这是
   "刻意保留的、与 CPU 一致的行为差异"，但去核对 CPU 参考
   `transport_kernel.py::distribute_corrections_to_cells_kernel` 才发现
   CPU 早在 2026-09-02 就已经补上了这个过滤——旧文档的说法是过时信息，
   不是事实）。
4. 附带发现：面校正的"|adj_row| 加权"此前在调用方
   （`compute_scalar_convection/diffusion_residual_gpu`）统一提前完成，
   对 native 面是错误的（native 面应该用 `true_area_weight` 加权，不是
   `|adj_row|`）——必须像 CPU 版一样传**未加权**的 `raw_jump_fp`，加权
   方式延后到分配阶段按面类型分派。
5. `compute_scalar_convection_residual_gpu`/`compute_scalar_diffusion_
   residual_gpu` 的四面体段 divergence 收缩无条件用 `ops_data['D_3d_
   tet']`（坍缩坐标微分算子），与 `gpu_gradients.py::compute_physical_
   gradient_gpu` 同一类遗漏（该处已于同日早些时候修复）。

全部 5 处已按 CPU 参考 `core/turbulence/transport.py`/
`transport_kernel.py` 逐字补齐，本文件用真实含 native 四面体的合成
网格 + numpy-as-cupy 替身决定性验证（不是代码走查）。
"""

import numpy as np
import pytest
from tests.unit._gpu_cupy_shim import patch_module_get_cupy


class _NumpyAsCupy:
    def __getattr__(self, name):
        return getattr(np, name)

    def scatter_add(self, a, indices, b):
        np.add.at(a, indices, b)

    def asnumpy(self, x):
        return np.asarray(x)

    class cuda:
        class Device:
            def __init__(self, device_id=0):
                pass

            def __enter__(self):
                return self

            def __exit__(self, *args):
                return False


@pytest.fixture(autouse=True)
def _patch_gpu_modules(monkeypatch):
    shim = _NumpyAsCupy()

    import autoflowcfd.core.gpu as core_gpu_mod
    import autoflowcfd.core.gpu.residual.gpu_inviscid as gpu_inviscid_mod
    import autoflowcfd.core.gpu.residual.gpu_inviscid_volume as gpu_inviscid_volume_mod
    import autoflowcfd.core.gpu.residual.gpu_volume_contract as gpu_volume_contract_mod
    import autoflowcfd.core.gpu.residual.gpu_gradients as gpu_gradients_mod
    import autoflowcfd.core.gpu.turbulence.gpu_scalar_transport as gst_mod
    import autoflowcfd.core.gpu.turbulence.gpu_turbulence_sst as gpu_turbulence_sst_mod
    import autoflowcfd.core.gpu.gpu_face_geometry as gfg_mod

    # `gpu_turbulence_sst_mod` 必须显式列入（2026-09-03 发现的测试脚手架
    # 缺口，不是生产代码 bug）：`GPUTurbulenceSST` 用
    # `from autoflowcfd.core.gpu import get_cupy` 在模块导入时把 `get_cupy`
    # 绑定到自己的命名空间，绑定发生在"该模块第一次被任何测试文件导入"那
    # 一刻——如果本文件漏掉这个模块，且它此前已被 `test_gpu_scalar_
    # transport.py`（正确地）导入过，本文件复用的是 sys.modules 缓存的
    # 同一个模块对象，`monkeypatch.setattr(core_gpu_mod, "get_cupy", ...)`
    # 这时已经晚了——只有当本文件独立在一个全新解释器进程内第一个导入
    # 该模块时才会"侥幸"绑到已打过补丁的 `get_cupy`，与真实测试套件的
    # 执行顺序（多个测试文件共享一个进程）不一致，是假阳性。
    mods = [
        gpu_inviscid_mod, gpu_inviscid_volume_mod, gpu_volume_contract_mod,
        gpu_gradients_mod, gst_mod, gpu_turbulence_sst_mod, gfg_mod,
    ]
    # get_cupy 一次性整批替换（含各包的子模块）；断言是聚合的，所以
    # `mods` 里含 gpu_inviscid_volume 这类本就没有 get_cupy 的模块无妨。
    patch_module_get_cupy(monkeypatch, [core_gpu_mod] + mods, shim)
    # `gpu_available` 是各类 __init__ 单独检查的模块级标志，与 get_cupy
    # 无关，仍需逐模块 patch。
    monkeypatch.setattr(core_gpu_mod, "gpu_available", True)
    for m in mods:
        if hasattr(m, "gpu_available"):
            monkeypatch.setattr(m, "gpu_available", True)


def _prepare_mesh_ops_data(mesh, ops):
    """独立于 GPUArrayManager 的 numpy 版 mesh_data/ops_data 构造，与
    `test_gpu_scalar_transport.py::_prepare_mesh_ops_data` 同一约定，
    额外补上 native 模式需要的 `D_native_tet_padded`。"""
    n_cells = mesh.n_cells
    n_sps = mesh.n_sps_per_cell
    det_jacs = mesh.jacobians['det_jacs'].reshape(n_cells, n_sps)
    inv_jacs = mesh.jacobians['inv_jacs'].reshape(n_cells, n_sps, 3, 3)
    adj_j = det_jacs[..., None, None] * inv_jacs
    mesh_data = {
        'det_jacs': det_jacs,
        'inv_jacs': inv_jacs,
        'adj_j': adj_j,
        'n_cells': n_cells,
        'n_prism': mesh.n_prism_cells,
    }
    ops_data = {
        'D_3d_prism': ops.D_3d_prism,
        'D_3d_tet': ops.D_3d_tet,
    }
    # 补齐生产 GPU 路径会上传、替身容易漏掉的键（native 算子 + 过积分
    # 算子 + 细点度量），见 _gpu_standin_helpers 模块文档。
    from ._gpu_standin_helpers import complete_gpu_standin
    complete_gpu_standin(mesh, ops, ops_data, mesh_data)
    return mesh_data, ops_data


def _synthetic_scalar_field(mesh, seed):
    n_cells, n_sps = mesh.n_cells, mesh.n_sps_per_cell
    rng = np.random.default_rng(seed)
    return rng.uniform(1e-4, 1e-2, size=(n_cells, n_sps))


class TestNativeScalarTransportMatchesCpu:
    @pytest.mark.parametrize("order", [1, 2])
    def test_native_convection_residual_matches_cpu(self, order):
        from tests.unit.test_fr_residual_inviscid import _build_synthetic_mixed_mesh
        from autoflowcfd.fr.operators import generate_fr_operators
        from autoflowcfd.core.fr_operators.face_kernels import get_flat_face_geometry
        from autoflowcfd.core.turbulence.transport import compute_scalar_convection_residual as _cpu_convection
        import autoflowcfd.core.gpu.turbulence.gpu_scalar_transport as gst

        mesh = _build_synthetic_mixed_mesh(order, tet_basis_mode="native")
        ops = generate_fr_operators(order)
        flat = get_flat_face_geometry(mesh, ops)
        mesh_data, ops_data = _prepare_mesh_ops_data(mesh, ops)
        n_cells, n_sps = mesh.n_cells, mesh.n_sps_per_cell

        scalar = _synthetic_scalar_field(mesh, seed=order * 1000 + 11)
        rho = np.full((n_cells, n_sps), 1.225)
        rng = np.random.default_rng(order * 1000 + 12)
        velocity = rng.uniform(-30.0, 30.0, size=(n_cells, n_sps, 3))

        expected = _cpu_convection(scalar, rho, velocity, mesh, ops)
        actual = np.asarray(gst.compute_scalar_convection_residual_gpu(
            scalar, rho, velocity, mesh_data, ops_data, flat, n_cells, mesh.n_prism_cells, n_sps,
        ))

        max_diff = np.max(np.abs(actual - expected))
        scale = max(np.max(np.abs(expected)), 1.0)
        rel = max_diff / scale
        assert rel < 1e-9, f"P={order} native scalar convection: max|cpu-gpu|={max_diff:.3e}, rel={rel:.3e}"
        assert np.all(np.isfinite(actual))
        # 反向对照：确认真的在检验对流物理（速度清零应显著改变残差）。
        zero_vel = np.asarray(gst.compute_scalar_convection_residual_gpu(
            scalar, rho, np.zeros_like(velocity), mesh_data, ops_data, flat, n_cells, mesh.n_prism_cells, n_sps,
        ))
        assert not np.allclose(actual, zero_vel)
        np.testing.assert_allclose(zero_vel, 0.0, atol=1e-9)

    @pytest.mark.parametrize("order", [1, 2])
    def test_native_diffusion_residual_matches_cpu(self, order):
        from tests.unit.test_fr_residual_inviscid import _build_synthetic_mixed_mesh
        from autoflowcfd.fr.operators import generate_fr_operators
        from autoflowcfd.core.fr_operators.face_kernels import get_flat_face_geometry
        from autoflowcfd.core.turbulence.transport import compute_scalar_diffusion_residual as _cpu_diffusion
        import autoflowcfd.core.gpu.turbulence.gpu_scalar_transport as gst

        mesh = _build_synthetic_mixed_mesh(order, tet_basis_mode="native")
        ops = generate_fr_operators(order)
        flat = get_flat_face_geometry(mesh, ops)
        mesh_data, ops_data = _prepare_mesh_ops_data(mesh, ops)
        n_cells, n_sps = mesh.n_cells, mesh.n_sps_per_cell

        scalar = _synthetic_scalar_field(mesh, seed=order * 2000 + 21)
        gamma = np.full((n_cells, n_sps), 1.8e-5) + 0.5 * scalar

        expected = _cpu_diffusion(scalar, gamma, mesh, ops)
        actual = np.asarray(gst.compute_scalar_diffusion_residual_gpu(
            scalar, gamma, mesh_data, ops_data, flat, n_cells, mesh.n_prism_cells, n_sps,
        ))

        max_diff = np.max(np.abs(actual - expected))
        scale = max(np.max(np.abs(expected)), 1.0)
        rel = max_diff / scale
        assert rel < 1e-9, f"P={order} native scalar diffusion: max|cpu-gpu|={max_diff:.3e}, rel={rel:.3e}"
        assert np.all(np.isfinite(actual))

    @pytest.mark.parametrize("order", [1, 2])
    def test_native_full_turbulence_transport_residual_matches_cpu(self, order):
        """完整入口函数级别的端到端验证（对流+扩散+omega壁面目标+机制3
        离群值抑制全部链路），与 `test_gpu_scalar_transport.py::
        TestTurbulenceTransportResidualGpuMatchesCpu` 同一测试模式，仅
        换成 native 网格。"""
        import types
        from tests.unit.test_fr_residual_inviscid import _build_synthetic_mixed_mesh
        from autoflowcfd.fr.operators import generate_fr_operators
        from autoflowcfd.core.fr_operators.face_kernels import get_flat_face_geometry
        from autoflowcfd.core.turbulence.transport import (
            compute_turbulence_transport_residual as _cpu_transport_residual,
        )
        from autoflowcfd.core.turbulence.sst import SSTModelFR
        from autoflowcfd.core.gpu.turbulence.gpu_turbulence_sst import GPUTurbulenceSST
        from autoflowcfd.core.fr_residual.inviscid import primitive_to_conserved
        import autoflowcfd.core.gpu.turbulence.gpu_scalar_transport as gst

        mesh = _build_synthetic_mixed_mesh(order, tet_basis_mode="native")
        ops = generate_fr_operators(order)
        flat = get_flat_face_geometry(mesh, ops)
        mesh_data, ops_data = _prepare_mesh_ops_data(mesh, ops)
        n_cells, n_sps = mesh.n_cells, mesh.n_sps_per_cell
        mu = 1.8e-5

        rng = np.random.default_rng(order * 3000 + 31)
        rho_inf, u_inf, v_inf, w_inf, p_inf = 1.225, 30.0, 5.0, -3.0, 101325.0
        Q = np.zeros((n_cells, n_sps, 5))
        Q[..., 0] = rho_inf * (1.0 + rng.uniform(-0.02, 0.02, size=(n_cells, n_sps)))
        Q[..., 1] = u_inf + rng.uniform(-3.0, 3.0, size=(n_cells, n_sps))
        Q[..., 2] = v_inf + rng.uniform(-2.0, 2.0, size=(n_cells, n_sps))
        Q[..., 3] = w_inf + rng.uniform(-2.0, 2.0, size=(n_cells, n_sps))
        Q[..., 4] = p_inf * (1.0 + rng.uniform(-0.01, 0.01, size=(n_cells, n_sps)))
        U = np.stack(
            [primitive_to_conserved(Q[c, s]) for c in range(n_cells) for s in range(n_sps)]
        ).reshape(n_cells, n_sps, 5)
        k_field = 1.0 * (1.0 + rng.uniform(-0.3, 0.3, size=(n_cells, n_sps)))
        omega_field = 500.0 * (1.0 + rng.uniform(-0.3, 0.3, size=(n_cells, n_sps)))
        d_wall = np.full((n_cells, n_sps), 0.05)

        turb_gpu = types.SimpleNamespace(
            k_field=k_field.copy(), omega_field=omega_field.copy(),
            nu_t=np.zeros_like(k_field),
            sigma_k1=0.85, sigma_k2=1.0, sigma_w1=0.5, sigma_w2=0.856,
            beta_star=0.09, beta1=0.075,
        )
        turb_gpu.compute_blending_F1_gpu = types.MethodType(GPUTurbulenceSST.compute_blending_F1_gpu, turb_gpu)
        gpu_solver = types.SimpleNamespace(
            turb_model_gpu=turb_gpu,
            mesh=mesh,
            mesh_data=mesh_data,
            ops_data=ops_data,
            Q_gpu=Q,
            U_gpu=U,
            mu_molecular=mu,
            wall_distance_gpu=d_wall,
            flat_face_gpu=flat,
            _wall_mask_k_gpu=np.zeros(flat.n_faces, dtype=bool),
        )
        dk_gpu, domega_gpu = gst.compute_turbulence_transport_residual_gpu(gpu_solver)

        turb_ref = SSTModelFR(n_cells, n_sps)
        turb_ref.k_field = k_field.copy()
        turb_ref.omega_field = omega_field.copy()

        class _CpuSolverStub:
            def _compute_gradients(self_inner):
                from autoflowcfd.core.fr_operators.gradients import compute_physical_gradient
                return compute_physical_gradient(U[..., :5], mesh, ops)

        cpu_solver = _CpuSolverStub()
        cpu_solver.mesh = mesh
        cpu_solver.ops = ops
        cpu_solver.turb_model = turb_ref
        cpu_solver.mu_molecular = mu
        cpu_solver.boundary_ghost_provider = None
        cpu_solver.wall_distance = d_wall
        cpu_solver.state = types.SimpleNamespace(Q=Q, U=U)

        dk_cpu, domega_cpu = _cpu_transport_residual(cpu_solver)

        max_diff_k = np.max(np.abs(dk_gpu - dk_cpu))
        scale_k = max(np.max(np.abs(dk_cpu)), 1.0)
        max_diff_w = np.max(np.abs(domega_gpu - domega_cpu))
        scale_w = max(np.max(np.abs(domega_cpu)), 1.0)
        assert max_diff_k / scale_k < 1e-9, f"P={order} native dk/dt: rel={max_diff_k / scale_k:.3e}"
        assert max_diff_w / scale_w < 1e-9, f"P={order} native domega/dt: rel={max_diff_w / scale_w:.3e}"
        assert np.all(np.isfinite(dk_gpu))
        assert np.all(np.isfinite(domega_gpu))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
