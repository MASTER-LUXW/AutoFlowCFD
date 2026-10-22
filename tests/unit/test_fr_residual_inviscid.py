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

from autoflowcfd.boundary.fr_ghost_state import DefaultGhostProvider
from autoflowcfd.core.fr_operators.flux_kernels import euler_physical_flux_point
from autoflowcfd.core.fr_operators.kernels import compute_ausm_up_flux

from autoflowcfd.core.fr_residual.inviscid import (
    compute_inviscid_residual_fr,
    primitive_to_conserved,
)
from autoflowcfd.grid.high_order.high_order_mesh import HighOrderMesh


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
    # 全部边界面显式归远场组（每个边界面都必须有组，见 build_boundary_ghost_provider）；
    # 需要别的边界条件的测试自行覆盖 boundary_groups/boundary_bc_types
    from tests.unit._boundary_groups import tag_all_boundary_faces

    tag_all_boundary_faces(mesh)
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

            F_exact = euler_physical_flux_point(q)
            flux_exact_n = F_exact.T @ n

            flux_ausm = compute_ausm_up_flux(q, q, n, 0.2)
            rel_diff = np.abs(flux_ausm - flux_exact_n) / (np.abs(flux_exact_n) + 1e-6)
            assert np.max(rel_diff) < 1e-9, "相容性被破坏"

    def test_mass_flux_matches_normal_velocity_for_uniform_state(self):
        """M+(M)+M-(M) 应恒等于 M，故 qL=qR 时 mass_flux 应精确等于 rho*u_n。"""
        q = np.array([1.225, 30.0, 5.0, -3.0, 101325.0])
        n = np.array([0.6, 0.8, 0.0])
        u_n = q[1] * n[0] + q[2] * n[1] + q[3] * n[2]
        flux = compute_ausm_up_flux(q, q, n, 0.2)
        assert abs(flux[0] - q[0] * u_n) < 1e-6 * abs(q[0] * u_n)


class TestAusmUpMachRef:
    """mach_ref 只经 AUSM+up 自带的低马赫标度函数 fa 进入通量。"""

    def test_consistency_unaffected_by_mach_ref(self):
        """F(q,q,n)=F(q) 相容性对任意 mach_ref 都必须成立（qL=qR 时 (pR-pL)=(unR-unL)=0，Mp/pu 修正项恒为零）。"""
        q = np.array([1.225, 30.0, 5.0, -3.0, 101325.0])
        n = np.array([0.6, 0.8, 0.0])
        u_n = q[1] * n[0] + q[2] * n[1] + q[3] * n[2]
        for mach_ref in [1.0, 0.5, 0.1, 0.05, 0.01, 0.001]:
            flux = compute_ausm_up_flux(q, q, n, mach_ref)
            assert abs(flux[0] - q[0] * u_n) < 1e-6 * abs(q[0] * u_n), (
                f"mach_ref={mach_ref}: mass flux consistency broken")


class TestAusmUpM4P5CoefficientsMatchLiterature:
    """真实 bug 修复回归测试（V2.0 专家组盲审第四次评审，2026-08-28，#12）：
    此前 M4±（质量分裂）用了 0.1875、P5±（压力分裂）用了 0.5——两个标准
    记号的系数被交换错配，且 0.5 本身超出 P5± 的有效区间 [-3/4,3/16]。

    已用 Liou (2006) AUSM+up 原始文献的独立生产实现交叉核实（SU2
    `CUpwAUSMPLUSUP_Flow::ComputeMassAndPressureFluxes`，su2code/SU2
    GitHub 仓库 ausm_slau.cpp）确认正确值：
        beta_mass = 1/8 = 0.125（固定常数，质量分裂）
        alpha_pressure = 3/16*(-4+5*fa^2)（随 fa 变化，压力分裂；
            fa=1 时退化为标准值 3/16=0.1875，fa->0 时趋于 -3/4）

    这里独立重新转录一份只含 M4±/P5± 分裂函数的最小参照实现（不是从
    kernels.py 复制过来的同一份代码，避免"验证代码抄了被验证代码的
    同一个 bug"），与 `compute_ausm_up_flux` 在受控状态下逐位比对。
    """

    @staticmethod
    def _reference_split_functions(M, alpha_pressure_val, beta_mass_val=1.0 / 8.0):
        """独立转录的 M4±/P5± 分裂函数（Liou 2006 / SU2 ausm_slau.cpp），
        与 kernels.py 内部同名闭包分开维护，供交叉核对。"""
        if abs(M) >= 1.0:
            Mp = 0.5 * (M + abs(M))
            Mm = 0.5 * (M - abs(M))
            Pp = 0.5 * (1.0 + np.sign(M))
            Pm = 0.5 * (1.0 - np.sign(M))
        else:
            Mp = 0.25 * (M + 1.0) ** 2 + beta_mass_val * (M ** 2 - 1.0) ** 2
            Mm = -0.25 * (M - 1.0) ** 2 - beta_mass_val * (M ** 2 - 1.0) ** 2
            Pp = 0.25 * (M + 1.0) ** 2 * (2.0 - M) + alpha_pressure_val * M * (M ** 2 - 1.0) ** 2
            Pm = 0.25 * (M - 1.0) ** 2 * (2.0 + M) - alpha_pressure_val * M * (M ** 2 - 1.0) ** 2
        return Mp, Mm, Pp, Pm

    def test_beta_mass_is_one_eighth(self):
        """质量分裂系数必须是固定常数 1/8——用两个不同的（非对称、非
        ±互为相反数——那种选择会让 M-(-M)=-M+(M) 恒成立，M_half 恒为
        零，与 beta_mass 取值无关，是一个退化的反例）亚声速马赫数
        M_L=0.5、M_R=0.2，验证 mass_flux 与独立参照实现一致。"""
        n = np.array([1.0, 0.0, 0.0])
        rho, a_target = 1.0, 340.0
        p = a_target ** 2 * rho / 1.4
        M_L_target, M_R_target = 0.5, 0.2
        qL = np.array([rho, M_L_target * a_target, 0.0, 0.0, p])
        qR = np.array([rho, M_R_target * a_target, 0.0, 0.0, p])

        # mach_ref=1.0：M0_sq 恒为 1（fa=1）；pL=pR 时 Mp 修正项恒为零。
        flux_correct = compute_ausm_up_flux(qL, qR, n, 1.0)

        Mp_expected, _, _, _ = self._reference_split_functions(M_L_target, alpha_pressure_val=0.1875)
        _, Mm_expected, _, _ = self._reference_split_functions(M_R_target, alpha_pressure_val=0.1875)
        mass_flux_expected = rho * a_target * (Mp_expected + Mm_expected)

        assert abs(flux_correct[0] - mass_flux_expected) < 1e-6 * abs(mass_flux_expected), (
            f"mass_flux={flux_correct[0]:.6f} 与 beta_mass=1/8 独立参照值 "
            f"{mass_flux_expected:.6f} 不符"
        )

    def test_negative_control_old_swapped_coefficients_would_differ(self):
        """反向对照：如果仍用修复前那套被交换错配的系数（0.1875 用在质量
        分裂、0.5 用在压力分裂），压力通量 p_half 会明显偏离修复后的值
        ——证明这处修复真的改变了数值行为，不是无意义的重命名。"""
        n = np.array([1.0, 0.0, 0.0])
        qL = np.array([1.2, 100.0, 0.0, 0.0, 101300.0])
        qR = np.array([1.2, -80.0, 0.0, 0.0, 101500.0])

        flux_fixed = compute_ausm_up_flux(qL, qR, n, 1.0)

        # 用修复前的（交换错配）系数独立重算一遍 p_half，只替换 P+/P-
        # 系数、其余（a_half/beta2/fa/Kp/Ku 等）沿用与生产代码相同的
        # 构造方式，隔离出 alpha_pressure 取值本身造成的差异。
        gamma = 1.4
        rhoL, rhoR = qL[0], qR[0]
        pL, pR = qL[4], qR[4]
        unL, unR = qL[1], qR[1]
        aL = np.sqrt(gamma * pL / rhoL)
        aR = np.sqrt(gamma * pR / rhoR)
        a_half = 0.5 * (aL + aR)
        M0_sq = 1.0  # mach_ref=1.0
        fa = np.sqrt(M0_sq) * (2.0 - np.sqrt(M0_sq))
        beta2 = 1.0  # mach_ref=1.0 时恒被 clip 到 1.0
        aL_p, aR_p, a_half_p = np.sqrt(beta2) * aL, np.sqrt(beta2) * aR, np.sqrt(beta2) * a_half
        M_L, M_R = unL / aL_p, unR / aR_p

        _, _, Pp_L_old, _ = self._reference_split_functions(M_L, alpha_pressure_val=0.5)
        _, _, _, Pm_R_old = self._reference_split_functions(M_R, alpha_pressure_val=0.5)
        Ku = 0.75
        p_half_old_buggy = (
            Pp_L_old * pL + Pm_R_old * pR
            - Ku * Pp_L_old * Pm_R_old * (rhoL + rhoR) * fa * a_half_p * (unR - unL)
        )

        # flux[1] = mass_flux*u_upwind + p_half*nx，nx=1 → p_half 是唯一
        # 依赖 alpha_pressure 的分量（mass_flux 只依赖 beta_mass，已在
        # 上一个测试单独验证）。
        p_half_fixed = flux_fixed[1] - flux_fixed[0] * (qL[1] if flux_fixed[0] >= 0 else qR[1])

        assert abs(p_half_fixed - p_half_old_buggy) > 1.0, (
            f"修复后 p_half={p_half_fixed:.6f} 与修复前（交换错配系数）"
            f"p_half={p_half_old_buggy:.6f} 差异过小，#12 修复疑似未生效"
        )


class TestFreeStreamPreservation:
    """FR 残差的黄金标准判据：均匀流场残差必须为零（到舍入）。

    容差按原生基实测标定（2026-10-22 I-210，`_build_synthetic_mixed_mesh`，相对 p_inf）：P1 2.6e-12、P2 1.1e-11、
    P3 6.7e-11、P4 3.1e-10，取约 30 倍余量。此前 P2 3e-5、P3 2e-2、P1 只查有限值，是按坍缩坐标模态基
    （Vandermonde 条件数 ~1e9，该基已删除）的舍入地板标定的，比现在的实测值宽 6~8 个量级、测不出度量或
    界面插值上 1e-7 量级的缺陷。
    """

    @pytest.mark.parametrize("order,rel_tol", [(1, 1e-10), (2, 3e-10), (3, 2e-9), (4, 1e-8)])
    def test_uniform_flow_gives_near_zero_residual(self, order, rel_tol):
        mesh = _build_synthetic_mixed_mesh(order)

        rho_inf, u_inf, v_inf, w_inf, p_inf = 1.225, 30.0, 5.0, -3.0, 101325.0
        Q_inf = np.array([rho_inf, u_inf, v_inf, w_inf, p_inf])
        U_inf = primitive_to_conserved(Q_inf)
        U = np.tile(U_inf, (mesh.n_cells, mesh.n_sps_per_cell, 1))

        residual = compute_inviscid_residual_fr(U, mesh, mesh.operators, boundary_ghost_provider=DefaultGhostProvider())
        rel_res = np.max(np.abs(residual)) / p_inf

        assert rel_res < rel_tol, f"Free-stream preservation failed at P={order}: rel={rel_res:.3e}"

    def test_p0_uniform_flow_gives_near_zero_residual(self):
        """P0 有限体积路径（compute_inviscid_residual_fv_p0）的均匀流场
        保持性——单独覆盖，因为它走的是完全不同的代码路径
        （inviscid_p0.py，不经过 compute_inviscid_residual_fr/over-
        integration）。

        这是 P0 路径的通用正确性回归，*不是* 2026-08-23 棱柱四边形侧面
        重复计数 bug（见 inviscid_p0.py::p0_face_geometry 修复
        文档）的可靠保护网——事后核实过，这里用的小合成网格（2 个棱柱
        共享一条侧面，理论上覆盖"重复记录、同一真实邻居"的场景）哪怕
        不做那次修复，闭合误差也巧合地接近机器精度（原因未深挖，判断
        是这个手搭小网格几何上的巧合，不是普适现象），这个测试在修复
        前后都是绿的，测不出那个回归。那个 bug 真正有效、已验证的回归
        保护是 test_inviscid_p0_prism_quad_dedup.py（直接构造 mock
        _KernelFaceData 断言去重/multi-source 回退分支的行为，不依赖
        具体网格的巧合）；真实网格上的定量验证见该修复的对话记录
        （cube_demo 单元 17790 处 Σ(n̂·A) 相对误差从 6.76% 降到 6.2e-16，
        全网 P0 均匀流场 max|residual|/p_inf 从 ~160 降到 1.19e-9）。
        """
        from autoflowcfd.core.fr_residual.inviscid_p0 import compute_inviscid_residual_fv_p0

        mesh = _build_synthetic_mixed_mesh(order=0)

        rho_inf, u_inf, v_inf, w_inf, p_inf = 1.225, 30.0, 5.0, -3.0, 101325.0
        Q_inf = np.array([rho_inf, u_inf, v_inf, w_inf, p_inf])
        U_inf = primitive_to_conserved(Q_inf)
        U = np.tile(U_inf, (mesh.n_cells, mesh.n_sps_per_cell, 1))

        from autoflowcfd.core.fr_operators.face_kernels import get_flat_face_geometry

        residual = compute_inviscid_residual_fv_p0(U, mesh, get_flat_face_geometry(mesh, mesh.operators),
                                                   boundary_ghost_provider=DefaultGhostProvider())
        rel_res = np.max(np.abs(residual)) / p_inf

        assert rel_res < 1e-10, f"P0 free-stream preservation failed: rel={rel_res:.3e}"


class TestVolumeTermDealiasing:
    """体积项去混叠（over-integration，V2.0 二次评审 Tier 0 #2）回归测试。

    均匀流场保持性（TestFreeStreamPreservation）对体积项混叠不敏感——
    F_tilde = adj(J)*F_phys(Q) 对常数 Q 的"跳跃项"在两侧各自精确抵消，
    与体积项散度本身是否精确无关（见 fr_residual_inviscid.py 模块文档
    与 5_重大问题修复-Part1.md §2.3 的推导）。这里用一个解析残差严格
    为零、但流场非均匀的线性剪切场 u=30+a*y 直接验证体积项散度本身的
    精度——真实数值审计发现，去混叠修复之前，这个测试在 P2（生产阶数）
    下会给出真值的 43~62 倍误差。

    容差按原生基实测标定（2026-10-22 I-210，质量方程残差绝对值）：P2 2.5e-10、P3 4.8e-10、P4 7.5e-10，取约
    15~40 倍余量（此前 P2 1e-4、P3 1e-2 是坍缩基时期的地板）。
    """

    @pytest.mark.parametrize("order,abs_tol", [(2, 1e-8), (3, 1e-8), (4, 1e-8)])
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

        residual = compute_inviscid_residual_fr(U, mesh, mesh.operators, boundary_ghost_provider=DefaultGhostProvider())
        max_mass_res = np.max(np.abs(residual[..., 0]))

        assert max_mass_res < abs_tol, (
            f"Linear shear flow (analytically zero residual) failed at P={order}: "
            f"max|mass residual|={max_mass_res:.3e}"
        )
