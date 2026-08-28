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
import pytest

from autoflowcfd.fr.operators import generate_fr_operators
from autoflowcfd.core.fr_operators.face_kernels import get_flat_face_geometry
from autoflowcfd.core.turbulence.transport import (
    _extrapolate_scalar_to_faces as _cpu_extrapolate_scalar_to_faces,
    compute_scalar_convection_residual as _cpu_compute_scalar_convection_residual,
    compute_scalar_diffusion_residual as _cpu_compute_scalar_diffusion_residual,
)
from autoflowcfd.core.fr_operators.gradients import compute_physical_scalar_gradient as _cpu_compute_physical_scalar_gradient
from tests.unit.test_fr_residual_inviscid import _build_synthetic_mixed_mesh

import autoflowcfd.core.gpu.residual.gpu_gradients as gpu_gradients_mod
import autoflowcfd.core.gpu.residual.gpu_volume_contract as gpu_volume_contract_mod
import autoflowcfd.core.gpu.turbulence.gpu_scalar_transport as gst


class _NumpyAsCupy:
    """把 numpy 伪装成 CuPy 模块接口，供 gpu_scalar_transport.py 的生产
    函数在没有真实 CUDA 设备的机器上直接运行（不是重新实现，是给同一份
    代码换一个张量库后端）。"""

    def __getattr__(self, name):
        return getattr(np, name)

    def scatter_add(self, a, indices, b):
        np.add.at(a, indices, b)


@pytest.fixture(autouse=True)
def _patch_get_cupy(monkeypatch):
    shim = _NumpyAsCupy()
    monkeypatch.setattr(gst, "get_cupy", lambda: shim)
    monkeypatch.setattr(gpu_gradients_mod, "get_cupy", lambda: shim)
    monkeypatch.setattr(gpu_volume_contract_mod, "get_cupy", lambda: shim)


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
    return mesh_data, ops_data


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
        )
        zero_vel = gst.compute_scalar_convection_residual_gpu(
            scalar, rho, np.zeros_like(velocity), mesh_data, ops_data, flat, n_cells, mesh.n_prism_cells, n_sps,
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


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
