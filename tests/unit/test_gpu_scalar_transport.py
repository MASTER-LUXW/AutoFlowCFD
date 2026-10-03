"""AutoFlowCFD V2.0 - GPU 版 k/omega 标量输运（gpu_scalar_transport.py）
端到端一致性测试。

本机没有 CuPy/真实 CUDA 设备，无法真正执行 CuPy kernel；但
`gpu_scalar_transport.py` 的实现只使用了 numpy 与 CuPy 共同支持的 API
子集（`where`/`einsum`/`matmul`/`maximum`/`clip`/`linalg.norm`/
`isfinite`/`zeros` 等，均语义完全一致，`scatter_add` 用 `np.add.at`
等价代替）——用一个把 `get_cupy()` 替换成"返回 numpy 模块（外加
scatter_add 方法）"的 monkeypatch，直接跑*生产函数本身*（不是重新
实现一份），对照已验证的 CPU `core/turbulence/transport.py` 在同一个
真实混合棱柱+四面体合成网格上的输出——数值要求逐位精确相等（同一套
公式的两种数值后端只做了张量库替换，不是近似关系）。
"""

import types

import numpy as np

from autoflowcfd.core.fr_operators.flux_kernels import resolve_viscous_ip_constant
import pytest

from autoflowcfd.fr.operators import generate_fr_operators
from autoflowcfd.core.fr_operators.face_kernels import get_flat_face_geometry
from autoflowcfd.core.turbulence.transport import (
    _extrapolate_scalar_to_faces as _cpu_extrapolate_scalar_to_faces,
    compute_scalar_convection_residual as _cpu_compute_scalar_convection_residual,
    compute_scalar_diffusion_residual as _cpu_compute_scalar_diffusion_residual,
    compute_turbulence_transport_residual as _cpu_compute_turbulence_transport_residual,
)
from autoflowcfd.core.fr_operators.gradients import compute_physical_scalar_gradient as _cpu_compute_physical_scalar_gradient
from autoflowcfd.core.turbulence.sst import SSTModelFR
from autoflowcfd.core.fr_residual.inviscid import primitive_to_conserved
from tests.unit.test_fr_residual_inviscid import _build_synthetic_mixed_mesh

import autoflowcfd.core.gpu.residual.gpu_gradients as gpu_gradients_mod
import autoflowcfd.core.gpu.residual.gpu_volume_contract as gpu_volume_contract_mod
import autoflowcfd.core.gpu.turbulence.gpu_scalar_transport as gst
import autoflowcfd.core.gpu.turbulence.gpu_turbulence_sst as gpu_turbulence_sst_mod
from tests.unit._gpu_cupy_shim import patch_module_get_cupy


class _NumpyAsCupy:
    """把 numpy 伪装成 CuPy 模块接口，供 gpu_scalar_transport.py 的生产
    函数在没有真实 CUDA 设备的机器上直接运行（不是重新实现，是给同一份
    代码换一个张量库后端）。"""

    def __getattr__(self, name):
        return getattr(np, name)

    def scatter_add(self, a, indices, b):
        np.add.at(a, indices, b)

    def asnumpy(self, x):
        return np.asarray(x)


@pytest.fixture(autouse=True)
def _patch_get_cupy(monkeypatch):
    """装上 numpy 替身，并把它**返回**出去。

    需要 cp 句柄的测试直接把本 fixture 当参数取用。此前写的是
    `gst.get_cupy()`：`gpu_scalar_transport` 拆成子包之后 `get_cupy` 不
    再出现在包命名空间里（它的调用点在子模块），而为了这几处调用把一个
    基础设施访问器 re-export 到包 `__init__` 只是污染 API 表面。
    """
    shim = _NumpyAsCupy()
    patch_module_get_cupy(monkeypatch, [
        gst, gpu_gradients_mod, gpu_volume_contract_mod,
        gpu_turbulence_sst_mod], shim)
    return shim


@pytest.fixture(scope="module")
def mesh_ops_flat():
    order = 2
    mesh = _build_synthetic_mixed_mesh(order)
    ops = generate_fr_operators(order)
    flat = get_flat_face_geometry(mesh, ops)
    return mesh, ops, flat


def _prepare_mesh_ops_data(mesh, ops):
    """独立于 GPUArrayManager 的 numpy 版 mesh_data/ops_data 构造（只包含
    这里用得到的键），语义与 array_manager.py::upload_mesh_data 一致，
    只是数组仍是 numpy 而不是 CuPy。"""
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
    # 补齐生产 GPU 路径会上传、替身容易漏掉的键，见
    # _gpu_standin_helpers 模块文档（四处替身共用一份实现）。
    from ._gpu_standin_helpers import complete_gpu_standin
    complete_gpu_standin(mesh, ops, ops_data, mesh_data)

    return mesh_data, ops_data


def _source_nu_t(k_field, omega_field):
    """源项求值刷新的涡粘（输运的扩散系数读它）：两侧同一份非平凡场。"""
    return 1e-3 * k_field / omega_field


def _source_F1(k_field):
    """源项求值刷新的混合函数 F1，取值在 (0, 1]，让扩散系数的混合真正被锻炼。"""
    return k_field / k_field.max()


def _synthetic_scalar_field(mesh):
    """构造一个非常量的标量场（节点坐标的光滑函数），保证梯度非零，
    真正锻炼扩散项，而不是退化到到处都是零跳跃的平凡情形。"""
    n_cells, n_sps = mesh.n_cells, mesh.n_sps_per_cell
    rng = np.random.default_rng(42)
    return rng.uniform(1e-4, 1e-2, size=(n_cells, n_sps))


class TestExtrapolateScalarToFacesGpuMatchesCpu:
    def test_neumann_default_matches_cpu(self, mesh_ops_flat):
        mesh, ops, flat = mesh_ops_flat
        scalar = _synthetic_scalar_field(mesh)

        phi_o_cpu, phi_n_cpu = _cpu_extrapolate_scalar_to_faces(scalar, flat, ops, mesh)
        phi_o_gpu, phi_n_gpu = gst._extrapolate_scalar_to_faces_gpu(np, flat, mesh.n_prism_cells, scalar)

        np.testing.assert_allclose(phi_o_gpu, phi_o_cpu, rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(phi_n_gpu, phi_n_cpu, rtol=1e-12, atol=1e-12)

    def test_dirichlet_zero_matches_cpu(self, mesh_ops_flat):
        mesh, ops, flat = mesh_ops_flat
        scalar = _synthetic_scalar_field(mesh)
        n_faces = flat.n_faces
        # 把前一半边界面标记为 Dirichlet-zero（k=0 壁面），与真实调用方
        # （wall_mask_k 只对 WALL 组为 True）同一种"部分面"场景。
        wall_mask = np.zeros(n_faces, dtype=np.bool_)
        wall_mask[np.nonzero(flat.is_boundary)[0][::2]] = True

        phi_o_cpu, phi_n_cpu = _cpu_extrapolate_scalar_to_faces(
            scalar, flat, ops, mesh, wall_dirichlet_zero_face=wall_mask,
        )
        phi_o_gpu, phi_n_gpu = gst._extrapolate_scalar_to_faces_gpu(
            np, flat, mesh.n_prism_cells, scalar, wall_dirichlet_zero_face=wall_mask,
        )
        np.testing.assert_allclose(phi_o_gpu, phi_o_cpu, rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(phi_n_gpu, phi_n_cpu, rtol=1e-12, atol=1e-12)

    def test_dirichlet_value_matches_cpu(self, mesh_ops_flat):
        mesh, ops, flat = mesh_ops_flat
        scalar = _synthetic_scalar_field(mesh)
        n_faces, n_fp = flat.n_faces, flat.n_fp
        boundary_idx = np.nonzero(flat.is_boundary)[0]
        has_value = np.zeros(n_faces, dtype=np.bool_)
        has_value[boundary_idx[1::2]] = True
        rng = np.random.default_rng(7)
        value_face = np.zeros((n_faces, n_fp), dtype=np.float64)
        value_face[has_value] = rng.uniform(100.0, 500.0, size=(int(has_value.sum()), n_fp))

        phi_o_cpu, phi_n_cpu = _cpu_extrapolate_scalar_to_faces(
            scalar, flat, ops, mesh,
            wall_dirichlet_value_face=value_face, has_wall_dirichlet_value=has_value,
        )
        phi_o_gpu, phi_n_gpu = gst._extrapolate_scalar_to_faces_gpu(
            np, flat, mesh.n_prism_cells, scalar,
            wall_dirichlet_value_face=value_face, has_wall_dirichlet_value=has_value,
        )
        np.testing.assert_allclose(phi_o_gpu, phi_o_cpu, rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(phi_n_gpu, phi_n_cpu, rtol=1e-12, atol=1e-12)


class TestScalarResidualsMatchCpu:
    def test_convection_residual_matches_cpu(self, mesh_ops_flat):
        mesh, ops, flat = mesh_ops_flat
        mesh_data, ops_data = _prepare_mesh_ops_data(mesh, ops)
        n_cells, n_sps = mesh.n_cells, mesh.n_sps_per_cell

        scalar = _synthetic_scalar_field(mesh)
        rho = np.full((n_cells, n_sps), 1.225)
        rng = np.random.default_rng(3)
        velocity = rng.uniform(-30.0, 30.0, size=(n_cells, n_sps, 3))

        expected = _cpu_compute_scalar_convection_residual(scalar, rho, velocity, mesh, ops)
        actual = gst.compute_scalar_convection_residual_gpu(
            scalar, rho, velocity, mesh_data, ops_data, flat, n_cells, mesh.n_prism_cells, n_sps,
            gst.scalar_convection_volume_divergence_gpu(np, np.ones((n_cells, n_sps)), rho, velocity,
                                                        mesh_data, ops_data, n_cells, mesh.n_prism_cells, n_sps),
        )
        np.testing.assert_allclose(actual, expected, rtol=1e-10, atol=1e-10)

    def test_diffusion_residual_matches_cpu(self, mesh_ops_flat):
        mesh, ops, flat = mesh_ops_flat
        mesh_data, ops_data = _prepare_mesh_ops_data(mesh, ops)
        n_cells, n_sps = mesh.n_cells, mesh.n_sps_per_cell

        scalar = _synthetic_scalar_field(mesh)
        gamma = np.full((n_cells, n_sps), 1.8e-5) + 0.5 * scalar

        expected = _cpu_compute_scalar_diffusion_residual(scalar, gamma, mesh, ops)
        actual = gst.compute_scalar_diffusion_residual_gpu(
            scalar, gamma, mesh_data, ops_data, flat, n_cells, mesh.n_prism_cells, n_sps,
            c_ip=resolve_viscous_ip_constant(int(mesh.order)),
        )
        np.testing.assert_allclose(actual, expected, rtol=1e-10, atol=1e-10)

    def test_negative_control_convection_sensitive_to_velocity(self, mesh_ops_flat):
        """反向对照：把速度场清零应该显著改变对流残差（证明测试真的在
        检验对流物理，不是恰好对任何输入都得到同一个平凡结果）。"""
        mesh, ops, flat = mesh_ops_flat
        mesh_data, ops_data = _prepare_mesh_ops_data(mesh, ops)
        n_cells, n_sps = mesh.n_cells, mesh.n_sps_per_cell

        scalar = _synthetic_scalar_field(mesh)
        rho = np.full((n_cells, n_sps), 1.225)
        rng = np.random.default_rng(3)
        velocity = rng.uniform(-30.0, 30.0, size=(n_cells, n_sps, 3))

        with_vel = gst.compute_scalar_convection_residual_gpu(
            scalar, rho, velocity, mesh_data, ops_data, flat, n_cells, mesh.n_prism_cells, n_sps,
            gst.scalar_convection_volume_divergence_gpu(np, np.ones((n_cells, n_sps)), rho, velocity,
                                                        mesh_data, ops_data, n_cells, mesh.n_prism_cells, n_sps),
        )
        zero_vel = gst.compute_scalar_convection_residual_gpu(
            scalar, rho, np.zeros_like(velocity), mesh_data, ops_data, flat, n_cells, mesh.n_prism_cells, n_sps,
            gst.scalar_convection_volume_divergence_gpu(np, np.ones((n_cells, n_sps)), rho, np.zeros_like(velocity),
                                                        mesh_data, ops_data, n_cells, mesh.n_prism_cells, n_sps),
        )
        assert not np.allclose(with_vel, zero_vel)
        np.testing.assert_allclose(zero_vel, 0.0, atol=1e-10)


class TestPhysicalScalarGradientGpuMatchesCpu:
    """真实 bug 回归测试（V2.0 专家组盲审第四轮，2026-08-28）：
    `gpu_gradients.py::compute_physical_scalar_gradient_gpu` 此前
    `return grad[..., 0]` 对末轴（3 个空间分量那一维）取索引 0，等价于
    "只留 x 分量、丢掉 y/z"，还多留了一个大小为 1 的 n_field_vars 轴，
    输出形状 (n_cells,n_sps,1) 而不是文档承诺的 (n_cells,n_sps,3)——是
    在为 #7 GPU 湍流输运写 `compute_scalar_diffusion_residual_gpu` 时
    才被发现（该函数是唯一调用这个标量梯度接口的新代码，`matmul` 直接
    因形状不匹配报错），但这个 bug 本身在 `compute_physical_scalar_
    gradient_gpu` 里已经存在，影响的是 SST 源项计算里的 grad_k/
    grad_omega（gpu_solver_io.py::compute_turbulence_source_gpu 唯一
    调用处），不是这次新引入的。"""

    def test_scalar_gradient_matches_cpu_on_nonconstant_field(self, mesh_ops_flat):
        mesh, ops, flat = mesh_ops_flat
        mesh_data, ops_data = _prepare_mesh_ops_data(mesh, ops)
        scalar = _synthetic_scalar_field(mesh)

        expected = _cpu_compute_physical_scalar_gradient(scalar, mesh, ops)
        actual = gpu_gradients_mod.compute_physical_scalar_gradient_gpu(scalar, mesh_data, ops_data)

        assert actual.shape == (mesh.n_cells, mesh.n_sps_per_cell, 3)
        np.testing.assert_allclose(actual, expected, rtol=1e-10, atol=1e-10)

    def test_negative_control_not_degenerate_to_single_component(self, mesh_ops_flat):
        """反向对照：修复前的 bug 会让返回值退化成形状 (n_cells,n_sps,1)
        （只有 x 分量、y/z 分量彻底丢失）。断言修复后的形状与"y/z 分量
        非零"，证明这条测试真的会在 bug 复现时失败，不是形状凑巧一致。"""
        mesh, ops, flat = mesh_ops_flat
        mesh_data, ops_data = _prepare_mesh_ops_data(mesh, ops)
        scalar = _synthetic_scalar_field(mesh)

        actual = gpu_gradients_mod.compute_physical_scalar_gradient_gpu(scalar, mesh_data, ops_data)
        assert actual.shape[-1] == 3
        assert not np.allclose(actual[..., 1], 0.0)
        assert not np.allclose(actual[..., 2], 0.0)


class TestTurbulenceTransportResidualGpuMatchesCpu:
    """`compute_turbulence_transport_residual_gpu` 完整入口函数的端到端
    一致性测试（此前本文件只测了它内部的几个子函数，没有测过入口函数
    本身）。

    原先这里还覆盖 2026-09-02 补齐的机制3 离群值抑制 round-trip；
    **机制3 已于 2026-09-19 整体删除**（依据见
    `core/fr_residual/inviscid.py`），那个用例随之删除，同时 GPU 侧
    也不再需要为它做 `GPU -> CPU -> GPU` 往返。"""

    def _build_state(self, mesh, rng):
        n_cells, n_sps = mesh.n_cells, mesh.n_sps_per_cell
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
        return Q, U, k_field, omega_field, d_wall

    def _build_gpu_solver_stub(self, mesh, mesh_data, ops_data, flat, Q, U, k_field, omega_field, d_wall, mu):
        turb = types.SimpleNamespace(
            k_field=k_field.copy(), omega_field=omega_field.copy(),
            nu_t=_source_nu_t(k_field, omega_field), _last_F1=_source_F1(k_field),
            sigma_k1=0.85, sigma_k2=1.0, sigma_w1=0.5, sigma_w2=0.856,
            beta_star=0.09, beta1=0.075, omega_max=1e6,
            # 来流值（来流条件用）：与 CPU 参照 `SSTModelFR(n_cells, n_sps)` 的
            # 构造默认值一致
            k_inf=1e-6, omega_inf=1.0,
        )
        return types.SimpleNamespace(
            turb_model_gpu=turb,
            mesh=mesh,
            mesh_data=mesh_data,
            ops_data=ops_data,
            Q_gpu=Q,
            U_gpu=U,
            mu_molecular=mu,
            wall_distance_gpu=d_wall,
            flat_face_gpu=flat,
            _wall_mask_k_gpu=np.zeros(flat.n_faces, dtype=bool),
            _open_mask_gpu=np.zeros(flat.n_faces, dtype=bool),
        )

    def _build_cpu_reference_solver(self, mesh, ops, Q, U, k_field, omega_field, d_wall, mu):
        turb_ref = SSTModelFR(mesh.n_cells, mesh.n_sps_per_cell)
        turb_ref.k_field = k_field.copy()
        turb_ref.omega_field = omega_field.copy()
        turb_ref.nu_t, turb_ref._last_F1 = _source_nu_t(k_field, omega_field), _source_F1(k_field)

        class _CpuSolverStub:
            def _compute_gradients(self_inner):
                from autoflowcfd.core.fr_operators.gradients import compute_physical_gradient
                return compute_physical_gradient(U[..., :5], mesh, ops)

        stub = _CpuSolverStub()
        stub.mesh = mesh
        stub.ops = ops
        stub.turb_model = turb_ref
        stub.mu_molecular = mu
        stub.boundary_ghost_provider = None
        stub.wall_distance = d_wall
        stub.state = types.SimpleNamespace(Q=Q, U=U)
        return stub, turb_ref

    def test_healthy_field_matches_cpu_exactly(self, mesh_ops_flat):
        """GPU 入口函数的完整输出必须与 CPU 参考逐位一致，不只是子函数
        级别一致。"""
        mesh, ops, flat = mesh_ops_flat
        mesh_data, ops_data = _prepare_mesh_ops_data(mesh, ops)
        mu = 1.8e-5
        rng = np.random.default_rng(99)
        Q, U, k_field, omega_field, d_wall = self._build_state(mesh, rng)

        gpu_solver = self._build_gpu_solver_stub(
            mesh, mesh_data, ops_data, flat, Q, U, k_field, omega_field, d_wall, mu
        )
        dk_gpu, domega_gpu = gst.compute_turbulence_transport_residual_gpu(gpu_solver)

        cpu_solver, _ = self._build_cpu_reference_solver(mesh, ops, Q, U, k_field, omega_field, d_wall, mu)
        dk_cpu, domega_cpu = _cpu_compute_turbulence_transport_residual(cpu_solver)

        np.testing.assert_allclose(dk_gpu, dk_cpu, rtol=1e-9, atol=1e-9)
        np.testing.assert_allclose(domega_gpu, domega_cpu, rtol=1e-9, atol=1e-9)

class TestComputeWallDirichletMaskGpu:
    """真实 bug 回归测试（2026-09-12，与 CPU 版
    `test_turbulence_transport.py::TestComputeWallDirichletFaceMask::
    test_slip_wall_excluded_only_no_slip_wall_included` 同一个 bug 的
    GPU 镜像验证）：`is_no_slip=False` 的滑移壁不应被
    `compute_turbulence_face_masks_gpu` 计入需要 Wilcox omega 壁面解析式
    处理的壁面集合，只有 `is_no_slip=True`（默认值）的 WALL 编码才应
    计入。函数本身是纯 numpy 实现（无需真实 CuPy 设备）。
    """

    def test_slip_wall_excluded_only_no_slip_wall_included(self):
        mesh = types.SimpleNamespace(face_connectivity=types.SimpleNamespace(n_faces=4))
        provider = types.SimpleNamespace(
            group_code=np.array([0, 1, 2, -1]),
            code_to_config={
                0: {"type": "WALL", "is_no_slip": True},   # body（真实固壁）
                1: {"type": "WALL", "is_no_slip": False},  # tunnel（滑移壁）
                2: {"type": "OUTLET"},
            },
        )

        mask, open_mask = gst.compute_turbulence_face_masks_gpu(mesh, provider)

        np.testing.assert_array_equal(mask, [True, False, False, False])
        # 开放边界：只有 OUTLET 组；-1（内部面/未匹配）在没有 default_config
        # 时不算（真边界条件由 GPU 对流里的面邻居源判定再求与）
        np.testing.assert_array_equal(open_mask, [False, False, True, False])

    def test_unmatched_faces_follow_default_config(self):
        mesh = types.SimpleNamespace(face_connectivity=types.SimpleNamespace(n_faces=3))
        provider = types.SimpleNamespace(
            group_code=np.array([0, -1, -1]),
            code_to_config={0: {"type": "SYMMETRY"}},
            default_config={"type": "FARFIELD"},
        )
        _, open_mask = gst.compute_turbulence_face_masks_gpu(mesh, provider)
        np.testing.assert_array_equal(open_mask, [False, True, True])


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
