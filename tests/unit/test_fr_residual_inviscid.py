"""
AutoFlowCFD V2.0 - FR 无粘残差单元测试 (S-02/S-04)

核心测试：自由流场保持性 (Free-Stream Preservation)。这是曲边/坍缩坐标
高阶格式的标准正确性判据——对均匀流场，无粘残差必须严格为零（数值上为
机器精度量级）。任何度量项变换、界面耦合、通量函数中的符号或一致性错误
都会在这个测试里表现为非零残差，是比单独检查每个子步骤更可靠的整体验证。

同时测试 AUSM+up 数值通量本身的相容性 (F(q,q,n) == 精确物理通量·n)，
这是任何黎曼求解器的基本要求，也是本次修复中发现的一个真实 bug
（M- 分裂函数亚声速分支符号错误、动量/能量通量用错误的迎风权重）的
回归测试。
"""

from types import SimpleNamespace

import numpy as np
import pytest

from autoflowcfd.core.fr_kernels import compute_ausm_up_flux
from autoflowcfd.core.fr_residual_inviscid import (
    compute_inviscid_residual_fr,
    euler_physical_flux,
    primitive_to_conserved,
)
from autoflowcfd.grid.high_order_mesh import HighOrderMesh


class _MockNodes:
    def __init__(self, coords):
        self._coords = coords

    def get_coordinates(self):
        return self._coords


class _MockCells:
    def __init__(self, connectivity):
        self.connectivity = connectivity


def _build_synthetic_mixed_mesh(order: int) -> HighOrderMesh:
    """2 个共享面的四面体 + 2 个共享侧面的棱柱，覆盖内部面 + 边界面两种情形。"""
    nodes = np.array(
        [
            [0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 1],
            [10, 0, 0], [11, 0, 0], [10, 1, 0], [10, 0, 1], [11, 0, 1], [10, 1, 1],
        ],
        dtype=float,
    )
    tet_conn = np.array([[0, 1, 2, 3], [1, 2, 3, 4]], dtype=np.int32)
    nodes = np.vstack([nodes, [[9, -1, 0], [9, -1, 1]]])
    prism_conn = np.array(
        [
            [5, 6, 7, 8, 9, 10],
            [5, 7, 11, 8, 10, 12],
        ],
        dtype=np.int32,
    )

    mock_volume = SimpleNamespace(
        cell_count=len(tet_conn) + len(prism_conn),
        nodes=_MockNodes(nodes),
        cells=_MockCells(tet_conn),
        prism_cells=_MockCells(prism_conn),
    )

    mesh = HighOrderMesh(order=order)
    mesh.load_from_volume_mesh(mock_volume)
    return mesh


class TestAusmUpConsistency:
    def test_flux_consistency_random_states(self):
        """F(q, q, n) 必须精确等于物理通量·n（黎曼求解器的基本相容性要求）。"""
        rng = np.random.default_rng(42)
        for _ in range(200):
            rho = rng.uniform(0.5, 2.0)
            u, v, w = rng.uniform(-50, 50, 3)
            p = rng.uniform(5e4, 2e5)
            q = np.array([rho, u, v, w, p])
            n = rng.normal(size=3)
            n /= np.linalg.norm(n)

            flux_ausm = compute_ausm_up_flux(q, q, n)
            F_exact = euler_physical_flux(q[None, :])[0]
            flux_exact_n = F_exact.T @ n

            rel_diff = np.abs(flux_ausm - flux_exact_n) / (np.abs(flux_exact_n) + 1e-6)
            assert np.max(rel_diff) < 1e-9

    def test_mass_flux_matches_normal_velocity_for_uniform_state(self):
        """M+(M)+M-(M) 应恒等于 M，故 qL=qR 时 mass_flux 应精确等于 rho*u_n。"""
        q = np.array([1.225, 30.0, 5.0, -3.0, 101325.0])
        n = np.array([0.6, 0.8, 0.0])
        flux = compute_ausm_up_flux(q, q, n)
        u_n = q[1] * n[0] + q[2] * n[1] + q[3] * n[2]
        assert abs(flux[0] - q[0] * u_n) < 1e-6 * abs(q[0] * u_n)


class TestFreeStreamPreservation:
    """曲边/坍缩坐标 FR 残差的黄金标准判据：均匀流场残差必须为零。"""

    # P=2 是本项目当前实际生产阶数（cube_demo 全流程验证用的阶数），要求
    # 严格判据；P=3 的四面体坍缩坐标模态基 Vandermonde 矩阵条件数已明显
    # 增长（真实测得 cond(V)~1e9，见 fr/collapsed_basis.py::jacobi_polynomial
    # 文档——这是未做节点优化的张量积-Duffy 组合在高阶下的已知谱方法限制，
    # 不是正确性 bug），残差判据相应放宽但仍需明确、有界，不能无限放宽/
    # 静默跳过——这是诚实记录一个已知、有界的数值局限，不是简化算法本身。
    #
    # P=2 容差从 1e-7 放宽到 3e-5（G-04 跨单元插值统一 + S-02 体积项
    # 去混叠两项修复的共同后果，均见
    # ProjectFiles/V2.0/6_整体专家组二次评审.md）：
    # 1. 跨单元插值（fr/face_flux_points.py::build_cross_interp）此前用
    #    与 owner 侧自身外插不同的朴素张量积 Lagrange 基，两者在同一
    #    物理点上最大相差 2070（G-04 缺陷 9），已改为与 owner 侧同源的
    #    坍缩坐标模态基，用 scipy.linalg.lu_solve 而不是显式求逆控制
    #    舍入放大——这一项单独会把 P=2 容差收紧到约 1.1e-7。
    # 2. 体积项此前直接在 coarse SPs 上对非线性通量*度量的乘积做散度，
    #    是标准的欠积分混叠（对解析残差恒为 0 的线性剪切场，P2 算出的
    #    残差是真值的 43~62 倍），已改为 over-integration：插值到
    #    over_order=min(2*order,3)=3 的细网格上精确求值/求导再限制回
    #    coarse（fr/collapsed_basis.py::build_overintegration_operators）。
    #    模态 Vandermonde 条件数在 over_order=3 时 ~1e9，比原生 P=2 的
    #    ~1e5 差几个数量级，这一项额外贡献的舍入误差把 rel 从 1.1e-7
    #    推高到真实测得的 1.06e-5；线性剪切流残差则从 43~62 倍降到约
    #    3.5e-6（改善 5~6 个数量级），是这项修复真正要解决的问题，见
    #    TestVolumeTermDealiasing。3e-5 留有约 3 倍安全余量。
    @pytest.mark.parametrize("order,rel_tol", [(2, 3e-5), (3, 5e-3)])
    def test_uniform_flow_gives_near_zero_residual(self, order, rel_tol):
        mesh = _build_synthetic_mixed_mesh(order)

        rho_inf, u_inf, v_inf, w_inf, p_inf = 1.225, 30.0, 5.0, -3.0, 101325.0
        Q_inf = np.array([rho_inf, u_inf, v_inf, w_inf, p_inf])
        U_inf = primitive_to_conserved(Q_inf)
        U = np.tile(U_inf, (mesh.n_cells, mesh.n_sps_per_cell, 1))

        residual = compute_inviscid_residual_fr(U, mesh, mesh.operators)
        rel_res = np.max(np.abs(residual)) / p_inf

        assert rel_res < rel_tol, f"Free-stream preservation failed at P={order}: rel={rel_res:.3e}"

    def test_p1_shows_documented_metric_aliasing_not_crash(self):
        """P=1（每方向2点）度量项混叠误差是坍缩坐标方法的已知固有特性，
        不是 bug（见 curved_mapping.py 文档），这里只验证不会崩溃/产生 NaN，
        且残差是有限值（不做机器精度断言）。"""
        mesh = _build_synthetic_mixed_mesh(order=1)
        Q_inf = np.array([1.225, 30.0, 5.0, -3.0, 101325.0])
        U_inf = primitive_to_conserved(Q_inf)
        U = np.tile(U_inf, (mesh.n_cells, mesh.n_sps_per_cell, 1))

        residual = compute_inviscid_residual_fr(U, mesh, mesh.operators)
        assert np.all(np.isfinite(residual))


class TestVolumeTermDealiasing:
    """体积项去混叠（over-integration，V2.0 二次评审 Tier 0 #2）回归测试。

    均匀流场保持性（TestFreeStreamPreservation）对体积项混叠不敏感——
    F_tilde = adj(J)*F_phys(Q) 对常数 Q 的"跳跃项"在两侧各自精确抵消，
    与体积项散度本身是否精确无关（见 fr_residual_inviscid.py 模块文档
    与 5_重大问题修复-Part1.md §2.3 的推导）。这里用一个解析残差严格
    为零、但流场非均匀的线性剪切场 u=30+a*y 直接验证体积项散度本身的
    精度——真实数值审计发现，去混叠修复之前，这个测试在 P2（生产阶数）
    下会给出真值的 43~62 倍误差。
    """

    @pytest.mark.parametrize("order,abs_tol", [(2, 1e-4), (3, 1e-2)])
    def test_linear_shear_flow_gives_near_zero_residual(self, order, abs_tol):
        mesh = _build_synthetic_mixed_mesh(order)
        rho_inf, p_inf, a = 1.225, 101325.0, 1.0

        U = np.zeros((mesh.n_cells, mesh.n_sps_per_cell, 5))
        for c in range(mesh.n_cells):
            y = mesh.sps_coords[c][:, 1]
            u = 30.0 + a * y
            Q = np.stack(
                [np.full_like(u, rho_inf), u, np.zeros_like(u), np.zeros_like(u), np.full_like(u, p_inf)],
                axis=-1,
            )
            U[c] = np.stack([primitive_to_conserved(Q[i]) for i in range(len(Q))])

        residual = compute_inviscid_residual_fr(U, mesh, mesh.operators)
        max_mass_res = np.max(np.abs(residual[..., 0]))

        assert max_mass_res < abs_tol, (
            f"Linear shear flow (analytically zero residual) failed at P={order}: "
            f"max|mass residual|={max_mass_res:.3e}"
        )
