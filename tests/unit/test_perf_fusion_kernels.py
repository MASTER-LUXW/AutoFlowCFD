"""2026-09-13 性能优化引入的 numba 融合 kernel 的数值等价性回归测试。

背景：用户反馈"每步计算时长太长、且计算效率对 CPU 数量不敏感"后做的真实
剖析（79 万单元 cube_demo 网格、P1）定位到一批热点全都是 **numpy 批量
微型 gemm / 逐元素运算**——这两类在 numpy 里都是单线程执行，加 CPU 核
完全无效，正是"对核数不敏感"的直接原因。这批优化把它们换成按 cell
`prange` 并行的 numba kernel（见各 kernel 文档的实测数据）。

本文件的作用：把"换计算路径但不改数学公式"这个前提**钉住**。每个用例都
用被替换掉的原始 numpy 公式当参照实现，在多种真实会出现的形状
（n_sps=1/4/8/27 对应 P0~P2、n_vars=1/3/5）上逐元素比对。凡是能做到逐位
相同的（求和顺序与原路径一致）就断言 `array_equal`；因浮点重结合只能做到
机器精度的，断言相对误差 <= 1e-14 并在注释里说明原因——避免后续有人
"顺手"改动求和顺序或引入 fastmath 而无人察觉。
"""
import numpy as np
import pytest

from autoflowcfd.core.fr_operators.volume_contract import (
    compute_adj_j,
    contract_shared_operator_1axis,
    contract_shared_operator_2axis,
    contravariant_flux_from_metric,
    grad_computational_to_physical,
)
from autoflowcfd.core.fr_operators.troubled_cell import (
    RESIDUAL_OUTLIER_FACTOR,
    RESIDUAL_OUTLIER_FIELD_REL_FLOOR,
    suppress_residual_outliers,
)
# 私有 kernel 从它**真正的**所在模块导入，而不是靠 troubled_cell
# 的 re-export（那里只 re-export 三个公开名；机制3 已于 2026-09-19
# 拆到 residual_outliers.py，见该模块文档）。
from autoflowcfd.core.fr_operators.residual_outliers import (
    _median_abs_over_sps_kernel,
)
from autoflowcfd.core.turbulence.sst import compute_strain_and_vorticity_magnitude
from autoflowcfd.core.turbulence.transport_kernel import scalar_convection_volume_kernel


def _rel_err(ref: np.ndarray, got: np.ndarray) -> float:
    denom = max(float(np.abs(ref).max()), 1e-300)
    return float(np.abs(ref - got).max()) / denom


class TestContractSharedOperator:
    """`contract_shared_operator_1axis/2axis`：原实现是 `np.matmul` 的批量
    小矩阵乘广播（(F,K)@(C,K,V)），numpy 对 batch 维度不并行。"""

    @pytest.mark.parametrize("n_sps", [1, 4, 8, 27])
    @pytest.mark.parametrize("n_vars", [1, 5])
    def test_1axis_matches_matmul(self, n_sps, n_vars):
        rng = np.random.default_rng(0)
        D = rng.standard_normal((n_sps, n_sps))
        X = rng.standard_normal((37, n_sps, n_vars))
        ref = np.ascontiguousarray(np.matmul(D, X))
        got = contract_shared_operator_1axis(D, X)
        assert got.shape == ref.shape
        assert _rel_err(ref, got) <= 1e-14

    @pytest.mark.parametrize("n_sps", [1, 4, 8, 27])
    @pytest.mark.parametrize("n_vars", [1, 5])
    def test_2axis_matches_matmul(self, n_sps, n_vars):
        rng = np.random.default_rng(1)
        D = rng.standard_normal((n_sps, n_sps, 3))
        X = rng.standard_normal((23, n_sps, 3, n_vars))
        F, J, M = D.shape
        C = X.shape[0]
        ref = np.ascontiguousarray(
            np.matmul(D.reshape(F, J * M), X.reshape(C, J * M, n_vars))
        )
        got = contract_shared_operator_2axis(D, X)
        assert got.shape == ref.shape
        assert _rel_err(ref, got) <= 1e-14

    def test_non_contiguous_input_handled(self):
        """真实调用点会传 `Q[c0:c1]`/`div_comp_fine[:n_prism]` 这类切片视图，
        以及（过积分链路里）非连续的中间结果——kernel 内部统一
        `ascontiguousarray`，这里显式钉住这个行为。"""
        rng = np.random.default_rng(2)
        D = rng.standard_normal((8, 8))
        X_big = rng.standard_normal((40, 8, 5))
        X = X_big[::2]  # 非连续视图
        assert not X.flags["C_CONTIGUOUS"]
        ref = np.ascontiguousarray(np.matmul(D, X))
        got = contract_shared_operator_1axis(D, X)
        assert _rel_err(ref, got) <= 1e-14


class TestContravariantFluxFromMetric:
    """`contravariant_flux_from_metric`：原实现是
    `np.matmul(compute_adj_j(det,inv), F_phys)`——先物化一份 (C,P,3,3) 的
    adj(J)（79 万单元 P1 过积分细点下约 1.5GiB），再做逐点 3x3@3xV 微型
    gemm。融合后 adj(J) 只存在于寄存器里。"""

    @pytest.mark.parametrize("n_pts", [1, 8, 27])
    @pytest.mark.parametrize("n_vars", [1, 5])
    def test_matches_adj_j_matmul(self, n_pts, n_vars):
        rng = np.random.default_rng(3)
        det = rng.standard_normal((29, n_pts)) + 5.0
        inv = rng.standard_normal((29, n_pts, 3, 3))
        F_phys = rng.standard_normal((29, n_pts, 3, n_vars))
        ref = np.matmul(compute_adj_j(det, inv), F_phys)
        got = contravariant_flux_from_metric(det, inv, F_phys)
        assert got.shape == ref.shape
        if n_vars >= 2:
            # 对 j 的求和顺序（j=0,1,2）与 np.matmul 的 gemm 路径一致
            # -> 逐位相同（生产路径的平均流就是 n_vars=5）
            assert np.array_equal(ref, got)
        else:
            # n_vars==1 时 np.matmul 退化成矩阵-向量（gemv）路径，累加方式
            # 与 gemm 不同，只能到机器精度：实测 9.3e-17 相对误差（1 ULP）。
            # 生产路径里 n_vars=1 只出现在标量对流的共享逆变质量通量
            # （transport.py::precompute_scalar_convection_geometry）。
            assert _rel_err(ref, got) <= 1e-14

    def test_zero_metric_gives_zero_flux(self):
        """退化单元（det(J)->0）不应产生 NaN/Inf——纯乘法链路，det=0 就该
        得到恰好 0（原 matmul 路径同理），这里钉住它。"""
        det = np.zeros((5, 8))
        inv = np.ones((5, 8, 3, 3))
        F_phys = np.ones((5, 8, 3, 5))
        got = contravariant_flux_from_metric(det, inv, F_phys)
        assert np.all(got == 0.0)


class TestGradComputationalToPhysical:
    """`grad_computational_to_physical`：原实现
    `np.matmul(np.swapaxes(grad_comp,-1,-2), inv_jacs)` 的左操作数是非连续
    转置视图，numpy 必须先物化一份连续副本再做逐点 (V,3)@(3,3) 微型 gemm。"""

    @pytest.mark.parametrize("n_sps", [1, 8, 27])
    @pytest.mark.parametrize("n_vars", [1, 3, 5])
    def test_matches_swapaxes_matmul(self, n_sps, n_vars):
        rng = np.random.default_rng(4)
        grad_comp = rng.standard_normal((31, n_sps, 3, n_vars))
        inv_jacs = rng.standard_normal((31, n_sps, 3, 3))
        ref = np.matmul(np.swapaxes(grad_comp, -1, -2), inv_jacs)
        got = grad_computational_to_physical(grad_comp, inv_jacs)
        assert got.shape == ref.shape
        # 对 m 的求和顺序（m=0,1,2）与 np.matmul 一致 -> 逐位相同
        assert np.array_equal(ref, got)


class TestStrainVorticityMagnitude:
    """`compute_strain_and_vorticity_magnitude`：原实现对 grad_u 先物化一份
    对称化/反对称化张量（79 万单元 P1 下各约 456MiB，含转置拷贝）再
    einsum 收缩，且两个量各来一遍。融合后只读一遍 grad_u。"""

    @pytest.mark.parametrize("n_sps", [1, 8, 27])
    def test_matches_einsum_reference(self, n_sps):
        rng = np.random.default_rng(5)
        g = rng.standard_normal((43, n_sps, 3, 3))
        S_ij = 0.5 * (g + np.transpose(g, (0, 1, 3, 2)))
        W_ij = 0.5 * (g - np.transpose(g, (0, 1, 3, 2)))
        s_ref = np.sqrt(2.0 * np.einsum("nijm,nijm->ni", S_ij, S_ij))
        w_ref = np.sqrt(2.0 * np.einsum("nijm,nijm->ni", W_ij, W_ij))
        s_got, w_got = compute_strain_and_vorticity_magnitude(g)
        # einsum 内部成对/SIMD 求和 vs 这里顺序双重循环 -> 只能到机器精度，
        # 实测 ~1e-16 相对误差（见 kernel 文档）
        assert _rel_err(s_ref, s_got) <= 1e-14
        assert _rel_err(w_ref, w_got) <= 1e-14

    def test_pure_rotation_has_zero_strain(self):
        """刚体旋转（grad_u 反对称）应给出 |S|=0、|Omega|>0——这是两个量
        物理定义的基本判据，融合后必须保持。"""
        g = np.zeros((3, 4, 3, 3))
        g[:, :, 0, 1] = 2.0
        g[:, :, 1, 0] = -2.0
        s, w = compute_strain_and_vorticity_magnitude(g)
        assert np.allclose(s, 0.0, atol=1e-14)
        assert np.all(w > 1.0)

    def test_pure_dilatation_has_zero_vorticity(self):
        """各向同性膨胀（grad_u 为对角）应给出 |Omega|=0。"""
        g = np.zeros((3, 4, 3, 3))
        for i in range(3):
            g[:, :, i, i] = 1.5
        s, w = compute_strain_and_vorticity_magnitude(g)
        assert np.all(s > 1.0)
        assert np.allclose(w, 0.0, atol=1e-14)


class TestScalarConvectionVolumeKernel:
    """`scalar_convection_volume_kernel`：原实现是 Python 层按块
    `adj_j*(rho*vel*phi)` -> `np.matmul` 逐点 3x3@3x1 -> 3 次
    `np.tensordot` 累加。这里用那条原始公式当参照实现。

    同时钉住 `adj_j@(rho*u*phi) == phi*(adj_j@(rho*u))` 这个把 phi 提到
    度量乘法外面的数学恒等式——它是 k/omega 两次调用能共享同一份
    `rho_u_tilde` 的依据（见 transport.py::ScalarConvectionGeometry）。
    """

    @pytest.mark.parametrize("n_sps", [1, 8, 27])
    def test_matches_original_numpy_chain(self, n_sps):
        rng = np.random.default_rng(6)
        n_cells = 19
        det = rng.standard_normal((n_cells, n_sps)) + 5.0
        inv = rng.standard_normal((n_cells, n_sps, 3, 3))
        rho = rng.standard_normal((n_cells, n_sps)) + 3.0
        vel = rng.standard_normal((n_cells, n_sps, 3))
        phi = rng.standard_normal((n_cells, n_sps)) + 2.0
        op_D = rng.standard_normal((n_sps, n_sps, 3))

        # --- 原始实现（被替换掉的那条 numpy 链路）---
        adj_j = det[:, :, None, None] * inv
        rho_u_phi = rho[:, :, None] * vel * phi[:, :, None]
        F_tilde = np.matmul(adj_j, rho_u_phi[..., None]).squeeze(-1)
        ref = np.zeros((n_cells, n_sps))
        for m in range(3):
            ref += np.tensordot(F_tilde[:, :, m], op_D[:, :, m], axes=([1], [1]))

        # --- 融合 kernel（配合"phi 提到度量外"的共享 rho_u_tilde）---
        rho_u = rho[:, :, None] * vel
        rho_u_tilde = contravariant_flux_from_metric(det, inv, rho_u[..., None])[..., 0]
        got = np.empty((n_cells, n_sps))
        scalar_convection_volume_kernel(
            np.ascontiguousarray(phi), np.ascontiguousarray(rho_u_tilde),
            np.ascontiguousarray(op_D), got,
        )
        # phi 从度量乘法里提出来改变了乘加顺序 -> 机器精度等价
        assert _rel_err(ref, got) <= 1e-13

    def test_uniform_scalar_field_reproduces_mass_divergence(self):
        """phi 恒为常数 c 时，散度必须恰好等于 c 乘以"phi=1 时的散度"
        （对流算子对标量的线性齐次性）——这是共享 rho_u_tilde 的另一个
        直接后果，用它排除 kernel 里把 phi 用错索引之类的错误。"""
        rng = np.random.default_rng(7)
        n_cells, n_sps = 11, 8
        rho_u_tilde = rng.standard_normal((n_cells, n_sps, 3))
        op_D = rng.standard_normal((n_sps, n_sps, 3))
        out_one = np.empty((n_cells, n_sps))
        out_c = np.empty((n_cells, n_sps))
        scalar_convection_volume_kernel(
            np.ones((n_cells, n_sps)), rho_u_tilde, op_D, out_one)
        scalar_convection_volume_kernel(
            np.full((n_cells, n_sps), 3.25), rho_u_tilde, op_D, out_c)
        np.testing.assert_allclose(out_c, 3.25 * out_one, rtol=1e-13, atol=0.0)


class TestSuppressResidualOutliersFused:
    """`suppress_residual_outliers`：原实现是 "numba 中位数 kernel + 5~7 趟
    numpy 全场遍历"（np.mean/np.abs/比较/np.any/np.where 各一趟，每趟读写
    253MiB 且全部单线程）。融合成两个 prange kernel 后仍要与原判据逐位一致。
    """

    @staticmethod
    def _reference(residual, reference_field,
                   factor=RESIDUAL_OUTLIER_FACTOR,
                   field_rel_floor=RESIDUAL_OUTLIER_FIELD_REL_FLOOR):
        ref_sibling = _median_abs_over_sps_kernel(residual)[:, np.newaxis, :]
        ref_field = field_rel_floor * np.mean(np.abs(reference_field), axis=1, keepdims=True)
        ref = np.maximum(np.maximum(ref_sibling, ref_field), 1e-300)
        with np.errstate(over="ignore", invalid="ignore"):
            outlier = np.abs(residual) > factor * ref
        if not np.any(outlier):
            return residual
        return np.where(outlier, 0.0, residual)

    @pytest.mark.parametrize("n_sps", [1, 4, 8, 27])
    @pytest.mark.parametrize("n_vars", [1, 5])
    def test_matches_reference_no_outlier(self, n_sps, n_vars):
        rng = np.random.default_rng(8)
        r = rng.standard_normal((97, n_sps, n_vars)) * 1e3
        f = rng.standard_normal((97, n_sps, n_vars)) * 10.0
        ref = self._reference(r.copy(), f.copy())
        got = suppress_residual_outliers(r.copy(), f.copy(), r.shape[0])
        assert np.array_equal(ref, got)

    @pytest.mark.parametrize("n_sps", [4, 8, 27])
    def test_matches_reference_with_single_outlier(self, n_sps):
        """真实病理形态：同一单元里绝大多数 SP 正常、个别 SP 量级暴涨。"""
        rng = np.random.default_rng(9)
        r = rng.standard_normal((53, n_sps, 5)) * 1e2
        f = rng.standard_normal((53, n_sps, 5)) * 10.0
        r[7, 0, 4] = 1e12
        ref = self._reference(r.copy(), f.copy())
        got = suppress_residual_outliers(r.copy(), f.copy(), r.shape[0])
        assert np.array_equal(ref, got)
        assert got[7, 0, 4] == 0.0

    def test_all_zero_residual_returns_unchanged(self):
        r = np.zeros((13, 8, 5))
        f = np.ones((13, 8, 5))
        got = suppress_residual_outliers(r.copy(), f.copy(), r.shape[0])
        assert np.array_equal(got, r)

    def test_input_not_mutated(self):
        """原实现返回 `np.where(...)` 的新数组、从不就地改输入；融合版本
        必须保持这个契约（调用方 transport.py 传的是带新轴的视图）。"""
        rng = np.random.default_rng(10)
        r = rng.standard_normal((29, 8, 5)) * 1e2
        r[3, 2, 1] = 1e13
        r_copy = r.copy()
        f = np.ones((29, 8, 5))
        suppress_residual_outliers(r, f, r.shape[0])
        assert np.array_equal(r, r_copy)


class TestModalFilterFusion:
    """`_filter_flat_U` / `filter_scalar_field`：原实现用的是**未开
    `optimize=True`** 的 `np.einsum`（走 numpy 通用逐元素求和路径、不是
    BLAS gemm，单线程），且 `U[...,:5]` 在 SST（n_vars=7）下是跨步长切片，
    赋值两端各有一次大拷贝。79 万单元 P1 实测每次 ~0.55s、SSP-RK3 每步
    调 3 次。"""

    @staticmethod
    def _ref_flat_U(U_flat, n_cells, n_sps, n_prism, F1, F2):
        U = U_flat.reshape(n_cells, n_sps, -1)
        if n_prism > 0:
            U[:n_prism, :, :5] = np.einsum("sj,cjv->csv", F1, U[:n_prism, :, :5])
        if n_cells > n_prism:
            U[n_prism:, :, :5] = np.einsum("sj,cjv->csv", F2, U[n_prism:, :, :5])
        return U.reshape(U_flat.shape)

    @pytest.mark.parametrize("n_sps,n_vars,n_prism,n_cells", [
        (8, 7, 40, 100),   # P1 + SST（n_vars=7 -> `:5` 是跨步长切片）
        (8, 5, 0, 80),     # P1 纯平均流、全四面体
        (27, 7, 30, 60),   # P2 + SST
        (1, 7, 20, 50),    # P0（滤波矩阵退化为单位阵的场景形状）
    ])
    def test_flat_U_filter_bit_identical(self, n_sps, n_vars, n_prism, n_cells):
        from autoflowcfd.core.fr_solver.filter import _filter_flat_U
        rng = np.random.default_rng(11)
        F1 = rng.standard_normal((n_sps, n_sps))
        F2 = rng.standard_normal((n_sps, n_sps))
        U0 = rng.standard_normal((n_cells * n_sps, n_vars))
        ref = self._ref_flat_U(U0.copy(), n_cells, n_sps, n_prism, F1, F2)
        got = _filter_flat_U(U0.copy(), n_cells, n_sps, n_prism, F1, F2)
        # 对 j 的求和顺序与非优化 einsum 一致 -> 实测全部形状逐位相同
        assert np.array_equal(ref, got)

    @pytest.mark.parametrize("n_sps", [1, 4, 8, 27])
    def test_scalar_filter_machine_precision(self, n_sps):
        from autoflowcfd.core.fr_solver.filter import filter_scalar_field
        rng = np.random.default_rng(12)
        n_cells, n_prism = 70, 30
        F1 = rng.standard_normal((n_sps, n_sps))
        F2 = rng.standard_normal((n_sps, n_sps))
        phi = rng.standard_normal((n_cells, n_sps))
        ref = phi.copy()
        ref[:n_prism] = np.einsum("sj,cj->cs", F1, phi[:n_prism])
        ref[n_prism:] = np.einsum("sj,cj->cs", F2, phi[n_prism:])
        got = filter_scalar_field(phi, n_prism, F1, F2)
        # 2D einsum 的累加方式与顺序循环略有不同 -> 机器精度（实测 ~1e-16）
        assert _rel_err(ref, got) <= 1e-14

    def test_scalar_filter_does_not_mutate_input(self):
        """`filter_scalar_field` 的既有契约是返回新数组、不改输入
        （调用方 turbulence.py 依赖这一点）。"""
        from autoflowcfd.core.fr_solver.filter import filter_scalar_field
        rng = np.random.default_rng(13)
        phi = rng.standard_normal((25, 8))
        phi_copy = phi.copy()
        F = rng.standard_normal((8, 8))
        filter_scalar_field(phi, 10, F, F)
        assert np.array_equal(phi, phi_copy)

    def test_identity_filter_is_noop(self):
        """滤波矩阵为单位阵时（P0 的真实情形）必须原样返回。"""
        from autoflowcfd.core.fr_solver.filter import _filter_flat_U, filter_scalar_field
        rng = np.random.default_rng(14)
        n_cells, n_sps, n_vars, n_prism = 30, 8, 7, 12
        I = np.eye(n_sps)
        U0 = rng.standard_normal((n_cells * n_sps, n_vars))
        got = _filter_flat_U(U0.copy(), n_cells, n_sps, n_prism, I, I)
        np.testing.assert_allclose(got, U0, rtol=0, atol=1e-15)
        phi = rng.standard_normal((n_cells, n_sps))
        np.testing.assert_allclose(filter_scalar_field(phi, n_prism, I, I), phi,
                                   rtol=0, atol=1e-15)
