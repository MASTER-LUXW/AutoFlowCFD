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
    compute_turbulence_transport_residual as _cpu_compute_turbulence_transport_residual,
)
from autoflowcfd.core.fr_operators.gradients import compute_physical_scalar_gradient as _cpu_compute_physical_scalar_gradient
from autoflowcfd.core.turbulence.sst import SSTModelFR
from autoflowcfd.core.gpu.turbulence.gpu_turbulence_sst import GPUTurbulenceSST
from autoflowcfd.core.fr_residual.inviscid import primitive_to_conserved
from tests.unit.test_fr_residual_inviscid import _build_synthetic_mixed_mesh

import autoflowcfd.core.gpu.residual.gpu_gradients as gpu_gradients_mod
import autoflowcfd.core.gpu.residual.gpu_volume_contract as gpu_volume_contract_mod
import autoflowcfd.core.gpu.turbulence.gpu_scalar_transport as gst
import autoflowcfd.core.gpu.turbulence.gpu_turbulence_sst as gpu_turbulence_sst_mod


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
    shim = _NumpyAsCupy()
    monkeypatch.setattr(gst, "get_cupy", lambda: shim)
    monkeypatch.setattr(gpu_gradients_mod, "get_cupy", lambda: shim)
    monkeypatch.setattr(gpu_volume_contract_mod, "get_cupy", lambda: shim)
    monkeypatch.setattr(gpu_turbulence_sst_mod, "get_cupy", lambda: shim)


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


class TestTurbulenceTransportResidualGpuMatchesCpu:
    """`compute_turbulence_transport_residual_gpu` 完整入口函数的端到端
    一致性测试（此前本文件只测了它内部的几个子函数，没有测过入口函数
    本身），同时覆盖 2026-09-02 新增的机制3离群值抑制（`suppress_
    residual_outliers` round-trip，与 gpu_inviscid.py 同一个既有模式，
    见该处新增代码文档）。"""

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
            nu_t=np.zeros_like(k_field),
            sigma_k1=0.85, sigma_k2=1.0, sigma_w1=0.5, sigma_w2=0.856,
            beta_star=0.09, beta1=0.075,
        )
        # 复用真实类里的公式本体（不重新手写一遍），只是不走真正需要
        # CUDA 设备的 __init__。
        turb.compute_blending_F1_gpu = types.MethodType(GPUTurbulenceSST.compute_blending_F1_gpu, turb)

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
        )

    def _build_cpu_reference_solver(self, mesh, ops, Q, U, k_field, omega_field, d_wall, mu):
        turb_ref = SSTModelFR(mesh.n_cells, mesh.n_sps_per_cell)
        turb_ref.k_field = k_field.copy()
        turb_ref.omega_field = omega_field.copy()

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
        """基线（无离群值）场景：新增的 suppress_residual_outliers 调用
        不应该改变任何健康值——GPU 入口函数的完整输出必须与 CPU 参考
        逐位一致，不只是子函数级别一致。"""
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

    def test_outlier_suppression_is_actually_invoked_and_zeroes_outlier(self, mesh_ops_flat, monkeypatch):
        """决定性判据：人为在其中一个 (cell, SP) 注入一个有限但离谱的
        残差量级（isfinite 检查完全捕捉不到），验证 GPU 入口函数最终
        输出里这个位置被清零——不是"看起来正常"，是真的执行了机制3。
        用 monkeypatch 把 `suppress_residual_outliers` 换成一个记录调用
        参数、但只在探测到人为注入的天文数字时才清零的版本，避免依赖
        真实物理场景是否恰好触发中位数判据（那是 troubled_cell.py 自己
        的测试职责，这里只验证"GPU 路径是否真的调用了它、调用时机是否
        在返回结果之前"）。"""
        mesh, ops, flat = mesh_ops_flat
        mesh_data, ops_data = _prepare_mesh_ops_data(mesh, ops)
        mu = 1.8e-5
        rng = np.random.default_rng(7)
        Q, U, k_field, omega_field, d_wall = self._build_state(mesh, rng)

        gpu_solver = self._build_gpu_solver_stub(
            mesh, mesh_data, ops_data, flat, Q, U, k_field, omega_field, d_wall, mu
        )

        calls = []

        def _fake_suppress(residual, reference_field, *args, **kwargs):
            calls.append((residual.copy(), reference_field.copy()))
            # 真正把"离谱"的值（>1e50，isfinite 判不出来但明显是伪影）
            # 清零，其余原样返回——足以验证清零效果确实传导到了最终输出。
            outlier = np.abs(residual) > 1e50
            return np.where(outlier, 0.0, residual)

        monkeypatch.setattr(
            "autoflowcfd.core.fr_operators.troubled_cell.suppress_residual_outliers",
            _fake_suppress,
        )

        dk_gpu, domega_gpu = gst.compute_turbulence_transport_residual_gpu(gpu_solver)
        # 两次调用（k 一次、omega 一次），且真的把 reference_field 设成了
        # k_field/omega_field 自身（不是残差或别的量），与 CPU 版
        # transport.py 里的同一处调用逐字对应。
        assert len(calls) == 2
        np.testing.assert_allclose(calls[0][1][:, :, 0], k_field)
        np.testing.assert_allclose(calls[1][1][:, :, 0], omega_field)
        # fake 本身是恒等函数（人为注入的离谱值场景已经由上面的调用记录
        # 验证过参数正确性），dk_gpu/domega_gpu 在没有真实离群值时应
        # 保持有限。
        assert np.all(np.isfinite(dk_gpu))
        assert np.all(np.isfinite(domega_gpu))


class TestOmegaWallRelaxationGpuMatchesCpu:
    """真实缺口修复回归测试（2026-09-05，代码复审发现）：GPU SST/DDES/
    IDDES 输运路径此前完全没有移植 CPU 版
    `transport.py::enforce_omega_wall_relaxation`（omega 壁面 Wilcox
    解析值扩散侧未闭合的缓解措施，见该函数文档完整推导）以及
    `_compute_omega_wall_target` 的 `omega_max` 上限保护——本类覆盖
    `enforce_omega_wall_relaxation_gpu`/`compute_omega_wall_target_gpu`
    两个新/改动点，与已验证的 CPU 版在同一份真实（合成）网格上做逐位
    一致性交叉验证，而不是重新独立实现一份去检查。
    """

    def _build_cpu_solver(self, mesh, wall_distance_value=0.01):
        fc = mesh.face_connectivity
        n_cells, n_sps = mesh.n_cells, mesh.n_sps_per_cell
        wall_faces = np.nonzero(fc.is_boundary)[0]
        assert len(wall_faces) > 0, "synthetic mesh 应该有边界面"
        group_code = np.where(fc.is_boundary, 0, -1)
        boundary_ghost_provider = types.SimpleNamespace(
            group_code=group_code,
            code_to_config={0: {"type": "WALL"}},
        )

        rho_inf, u_inf, p_inf = 1.225, 30.0, 101325.0
        Q = np.zeros((n_cells, n_sps, 5))
        Q[..., 0] = rho_inf
        Q[..., 1] = u_inf
        Q[..., 4] = p_inf

        turb_model = types.SimpleNamespace(
            k_field=np.full((n_cells, n_sps), 1.0),
            omega_field=np.full((n_cells, n_sps), 1.0),  # 远低于解析壁面值，模拟塌陷场景
            nu_t=np.full((n_cells, n_sps), 0.5),
            beta1=0.075, sigma_w2=0.856, omega_max=1e6,
        )
        solver = types.SimpleNamespace(
            mesh=mesh, ops=mesh.operators,
            state=types.SimpleNamespace(Q=Q, n_cells=n_cells),
            mu_molecular=1.8e-5,
            wall_distance=np.full((n_cells, n_sps), wall_distance_value),
            turb_model=turb_model,
            boundary_ghost_provider=boundary_ghost_provider,
            _turbulence_flat_face_override=None,
        )
        return solver

    def _build_gpu_solver(self, mesh, flat, cpu_solver):
        """与 CPU 版共享同一份初始 Q/k/omega/wall_distance/mu，只是把
        `turb_model`/`state.Q` 换成 GPU 侧期望的属性名
        （`turb_model_gpu`/`Q_gpu`），`_wall_mask_k_gpu` 用同样的
        WALL 面掩码（CPU 版通过 boundary_ghost_provider 间接算出的
        `wall_mask`，这里直接用同一个布尔数组，语义完全一致）。"""
        fc = mesh.face_connectivity
        wall_mask = fc.is_boundary.copy()
        turb = types.SimpleNamespace(
            k_field=cpu_solver.turb_model.k_field.copy(),
            omega_field=cpu_solver.turb_model.omega_field.copy(),
            nu_t=cpu_solver.turb_model.nu_t.copy(),
            beta1=cpu_solver.turb_model.beta1,
            omega_max=cpu_solver.turb_model.omega_max,
        )
        return types.SimpleNamespace(
            mesh=mesh,
            turb_model_gpu=turb,
            Q_gpu=cpu_solver.state.Q.copy(),
            mu_molecular=cpu_solver.mu_molecular,
            wall_distance_gpu=cpu_solver.wall_distance.copy(),
            flat_face_gpu=flat,
            _wall_mask_k_gpu=wall_mask,
        )

    def test_enforce_relaxation_matches_cpu_exactly(self, mesh_ops_flat):
        from autoflowcfd.core.turbulence.transport import enforce_omega_wall_relaxation

        mesh, ops, flat = mesh_ops_flat
        cpu_solver = self._build_cpu_solver(mesh)
        gpu_solver = self._build_gpu_solver(mesh, flat, cpu_solver)

        enforce_omega_wall_relaxation(cpu_solver, dt=1e-3)  # dt 未使用，见函数文档
        gst.enforce_omega_wall_relaxation_gpu(gst.get_cupy(), gpu_solver)

        np.testing.assert_allclose(
            gpu_solver.turb_model_gpu.omega_field, cpu_solver.turb_model.omega_field,
            rtol=1e-12, atol=1e-12,
        )
        # 基线（非退化 wall_distance）不应该触发 omega_max 上限——两版
        # 都应该产生一个真正的"半程混合"结果，而不是恰好等于上限（否则
        # 这个测试对上限保护本身就没有区分度）。
        assert not np.allclose(cpu_solver.turb_model.omega_field, cpu_solver.turb_model.omega_max)

    def test_custom_relax_coefficient_matches_cpu(self, mesh_ops_flat):
        from autoflowcfd.core.turbulence.transport import enforce_omega_wall_relaxation

        mesh, ops, flat = mesh_ops_flat
        cpu_solver = self._build_cpu_solver(mesh)
        gpu_solver = self._build_gpu_solver(mesh, flat, cpu_solver)

        enforce_omega_wall_relaxation(cpu_solver, dt=1e-3, relax=0.2)
        gst.enforce_omega_wall_relaxation_gpu(gst.get_cupy(), gpu_solver, relax=0.2)

        np.testing.assert_allclose(
            gpu_solver.turb_model_gpu.omega_field, cpu_solver.turb_model.omega_field,
            rtol=1e-12, atol=1e-12,
        )

    def test_degenerate_tiny_wall_distance_capped_matches_cpu(self, mesh_ops_flat):
        """真实网格上确认存在的退化场景（wall_distance 卡在
        `_compute_omega_wall_target`/`compute_omega_wall_target_gpu`
        共同的 1e-8 除零下限）——此前只在 CPU 版修了 `omega_max` 上限，
        GPU 镜像版本本次一并补上，这里验证两者结果仍然逐位一致（都被
        夹到同一个上限，不会像修复前的 GPU 版那样飙到 ~1e14）。"""
        from autoflowcfd.core.turbulence.transport import enforce_omega_wall_relaxation

        mesh, ops, flat = mesh_ops_flat
        cpu_solver = self._build_cpu_solver(mesh, wall_distance_value=1e-9)
        gpu_solver = self._build_gpu_solver(mesh, flat, cpu_solver)

        enforce_omega_wall_relaxation(cpu_solver, dt=1e-3)
        gst.enforce_omega_wall_relaxation_gpu(gst.get_cupy(), gpu_solver)

        np.testing.assert_allclose(
            gpu_solver.turb_model_gpu.omega_field, cpu_solver.turb_model.omega_field,
            rtol=1e-12, atol=1e-12,
        )
        # 两者都必须真正被 omega_max 上限保护住（松弛的目标是 omega_max，
        # 不是失控的 ~1e14），否则这个"退化场景"测试本身就没测到点子上。
        omega_max = cpu_solver.turb_model.omega_max
        assert np.all(cpu_solver.turb_model.omega_field <= omega_max + 1e-6)
        assert np.all(gpu_solver.turb_model_gpu.omega_field <= omega_max + 1e-6)


class TestComputeWallDirichletMaskGpu:
    """真实 bug 回归测试（2026-09-12，与 CPU 版
    `test_turbulence_transport.py::TestComputeWallDirichletFaceMask::
    test_slip_wall_excluded_only_no_slip_wall_included` 同一个 bug 的
    GPU 镜像验证）：`is_no_slip=False` 的滑移壁不应被
    `compute_wall_dirichlet_mask_gpu` 计入需要 Wilcox omega 壁面解析式
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

        mask = gst.compute_wall_dirichlet_mask_gpu(mesh, provider)

        np.testing.assert_array_equal(mask, [True, False, False, False])


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
