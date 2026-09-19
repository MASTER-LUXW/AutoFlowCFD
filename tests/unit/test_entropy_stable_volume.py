"""AutoFlowCFD V2.0 - 体积项 entropy-stable/split-form 通量单元测试。

见 core/fr_operators/flux_kernels.py::chandrashekar_flux_point /
entropy_stable_volume_divergence_batch，`8_算法重构-Entropy-Stable_
Split-Form通量重构-Part1/2.md`。端到端（真实 FRSolver、真实网格）验证见
tests/validation/test_entropy_stable_volume_stability.py。
"""

import numpy as np

from autoflowcfd.core.fr_operators.flux_kernels import (
    chandrashekar_flux_point,
    euler_physical_flux_point,
    entropy_stable_volume_divergence_batch,
)
from autoflowcfd.fr.native_tet.basis import (
    build_native_tet_operators, map_native_tet_to_physical, compute_native_tet_jacobian,
)
from autoflowcfd.fr.operators import generate_fr_operators


def _random_Q(rng):
    return np.array([
        1.0 + rng.random(),
        10.0 * rng.standard_normal(),
        10.0 * rng.standard_normal(),
        10.0 * rng.standard_normal(),
        101325.0 + 5000.0 * rng.standard_normal(),
    ])


def test_chandrashekar_flux_reduces_to_euler_flux_when_states_equal():
    """两点通量的标准一致性要求：QL==QR 时精确退化为普通物理通量
    （数值上到 log_mean 泰勒展开阈值附近的浮点噪声，见
    `_log_mean_point` 文档）。"""
    rng = np.random.default_rng(1)
    for _ in range(20):
        Q = _random_Q(rng)
        f_pair = chandrashekar_flux_point(Q, Q)
        f_std = euler_physical_flux_point(Q)
        assert np.max(np.abs(f_pair - f_std)) < 1e-8


def test_chandrashekar_flux_symmetric_mass_momentum():
    """质量/动量分量的两点通量必须关于 (QL,QR) 对称（对数平均、算术
    平均都是对称的，这是熵守恒两点通量的代数结构性要求）。"""
    rng = np.random.default_rng(2)
    for _ in range(20):
        QL, QR = _random_Q(rng), _random_Q(rng)
        f1 = chandrashekar_flux_point(QL, QR)
        f2 = chandrashekar_flux_point(QR, QL)
        assert np.allclose(f1[:, :4], f2[:, :4], atol=1e-10)


def _real_tet_geometry(order, cell_nodes):
    """用真实生产 native 单纯形基几何（不是随便编的度量数据——度量项
    adj_j 和微分矩阵 D 必须满足离散 GCL 恒等式，均匀场散度恒零这个
    检验才有意义，随便拼凑的 adj_j/D 数据不满足这个恒等式，测出的
    非零散度是这套拼凑数据本身不自洽，不是被测函数的 bug）。

    2026-09-03 更正：四面体坍缩坐标基已删除（见 fr/operators.py 模块
    文档），`ops.D_3d_tet` 现在是 `D_native_tet_padded` 的别名——对
    (r,s,t) 参考单纯形坐标求导，不再是对 (a,b,c) 坍缩坐标求导。度量项
    改用与生产 `high_order_mesh_order.py::compute_native_tet_jacobians`
    完全同一套构造：native 直边单元 Jacobian 是不依赖参考点位置的
    常数（`compute_native_tet_jacobian` 解析闭式），直接广播到全部
    `n_sps_global` 槽位（含填充槽位）——不能像坍缩坐标方案那样反过来
    用 `D @ phys` 数值求导算 Jacobian：`D` 的填充行/列是零（"零填充块
    对角"约定），若填充槽位的 `phys` 走真实仿射映射，`D@phys` 在填充
    行会得到退化的零 Jacobian（奇异，`inv_jacs` 出 NaN/Inf），而这个
    NaN 会通过 entropy-stable 核里的两两配对求和污染真实行的结果（
    `0*NaN=NaN`，不会被后续乘零自动清零）——生产代码从一开始就不会
    触碰到这个陷阱，因为它压根不用 D 矩阵算四面体 Jacobian，这里必须
    复刻同一个"常数解析 Jacobian 广播"策略才自洽。
    """
    ops = generate_fr_operators(order)
    D = ops.D_3d_tet  # 别名到 D_native_tet_padded，见模块文档
    n_sps_global = D.shape[0]

    det_j, adj_const = compute_native_tet_jacobian(cell_nodes)
    det_jacs = np.full(n_sps_global, det_j)
    adj_j = np.broadcast_to(adj_const, (n_sps_global, 3, 3)).copy()
    return D, adj_j, det_jacs


def test_entropy_stable_volume_divergence_batch_zero_on_uniform_field_real_geometry():
    """均匀流场在真实坍缩坐标几何（满足离散 GCL）下，entropy-stable
    体积项散度必须恰好为零——最基本的自由流场保持检验。"""
    order = 2
    cell_nodes = np.array([[0.1, -0.2, 0.3], [1.1, 0.0, -0.1], [-0.2, 1.2, 0.1], [0.0, 0.1, 1.3]])
    D, adj_j, det_jacs = _real_tet_geometry(order, cell_nodes)
    n_sps = D.shape[0]

    rng = np.random.default_rng(3)
    Q0 = _random_Q(rng)
    Q_uniform = np.tile(Q0, (n_sps, 1))
    Q = Q_uniform[None, :, :]

    div = entropy_stable_volume_divergence_batch(Q, adj_j[None], D)
    residual = div / np.maximum(det_jacs, 1e-300)[None, :, None]
    # 相对判据：残差相对场量本身的物理量级（Q0，含 ~1e5 量级的能量分量），
    # 而不是相对 det_jacs 本身（det_jacs 只是分母、不是场的物理量级，
    # 单独用它归一化在能量分量上会显得"残差偏大"，实际上是分母选择不
    # 恰当，不是残差真的大——见下面数值：绝对残差 ~1e-5 量级，相对
    # Q0 的 ~1e5 能量量级是 ~1e-10，接近机器精度）。
    scale = np.maximum(np.abs(Q0), 1e-300)
    assert np.max(np.abs(residual) / scale) < 1e-6


def test_entropy_stable_volume_divergence_batch_matches_manual_reference_real_geometry():
    """与手写的纯 numpy 参考实现（`8_算法重构-Entropy-Stable_Split-Form
    通量重构-Part2.md` 决定性验证脚本同一套公式）逐位交叉核对，真实
    native 单纯形基几何、非均匀（Couette 剪切）流场。

    2026-09-03 更正：Q 场按物理坐标求值必须用 `map_native_tet_to_
    physical`（与 `D`——现在是 `D_native_tet_padded`——参考的同一套
    (r,s,t) 单纯形坐标），不能再用坍缩坐标 `cube_to_tet_rst`/
    `tet_barycentric` 那套往返（见 `_real_tet_geometry` 文档同一处
    更正说明）。填充槽位（`[n_native:]`）复制真实 SP #0 的物理坐标
    （与 build_order_geometry 同一约定），Q 场在这些槽位的取值本身
    不影响真实自由度的结果——`D` 的填充列恒为零，真实行的散度求和
    不会读到填充槽位的贡献。
    """
    order = 2
    cell_nodes = np.array([[0.0, 0.0, 0.0], [1.3, -0.1, 0.2], [-0.1, 1.1, 0.0], [0.1, 0.0, 0.9]])
    D, adj_j, det_jacs = _real_tet_geometry(order, cell_nodes)
    n_sps = D.shape[0]

    ref_rst, _ = build_native_tet_operators(order)
    n_native = ref_rst.shape[0]
    phys_native = map_native_tet_to_physical(ref_rst, cell_nodes)
    phys = np.zeros((n_sps, 3))
    phys[:n_native] = phys_native
    phys[n_native:] = phys_native[0]

    Q = np.zeros((n_sps, 5))
    Q[:, 0] = 1.225
    u = 30.0 * phys[:, 1]
    Q[:, 1] = 1.225 * u
    Q[:, 4] = 101325.0 / 0.4 + 0.5 * 1.225 * u**2

    div_kernel = entropy_stable_volume_divergence_batch(Q[None], adj_j[None], D)[0]

    # 手写参考实现（纯 numpy，O(n^2) pair，独立于生产 numba kernel）
    n = n_sps
    QL = np.repeat(Q, n, axis=0)
    QR = np.tile(Q, (n, 1))
    F_pair = np.stack([_chandrashekar_flux_3d_reference(QL, QR, c) for c in range(3)], axis=1)
    adj_i = np.repeat(adj_j, n, axis=0)
    adj_jj = np.tile(adj_j, (n, 1, 1))
    adj_sym = 0.5 * (adj_i + adj_jj)
    F_tilde_pair = np.matmul(adj_sym, F_pair).reshape(n, n, 3, 5)
    div_ref = np.zeros((n, 5))
    for m in range(3):
        div_ref += np.einsum("ij,ijv->iv", D[:, :, m], F_tilde_pair[:, :, m, :])
    div_ref *= 2.0

    rel = np.max(np.abs(div_kernel - div_ref)) / max(np.max(np.abs(div_ref)), 1e-300)
    assert rel < 1e-10


def _log_mean_reference(a, b):
    xi = a / b
    f = (xi - 1) / (xi + 1)
    u = f * f
    F = np.where(
        u < 1e-4,
        1.0 + u / 3.0 + u**2 / 5.0 + u**3 / 7.0,
        np.log(np.maximum(xi, 1e-300)) / np.where(np.abs(f) > 1e-300, 2.0 * f, 1.0),
    )
    return (a + b) / (2.0 * F)


def _chandrashekar_flux_3d_reference(QL, QR, direction):
    GAMMA = 1.4
    rhoL, uL_vec, pL = QL[:, 0], QL[:, 1:4], QL[:, 4]
    rhoR, uR_vec, pR = QR[:, 0], QR[:, 1:4], QR[:, 4]
    betaL, betaR = rhoL / (2 * pL), rhoR / (2 * pR)
    rho_ln = _log_mean_reference(rhoL, rhoR)
    beta_ln = _log_mean_reference(betaL, betaR)
    rho_bar = 0.5 * (rhoL + rhoR)
    beta_bar = 0.5 * (betaL + betaR)
    u_bar_vec = 0.5 * (uL_vec + uR_vec)
    p_tilde = rho_bar / (2 * beta_bar)
    f_rho = rho_ln * u_bar_vec[:, direction]
    f_mom = u_bar_vec * f_rho[:, None]
    f_mom[:, direction] += p_tilde
    ke_bar = 0.5 * np.sum(u_bar_vec**2, axis=1)
    f_e = (1.0 / (2.0 * (GAMMA - 1.0) * beta_ln) - ke_bar) * f_rho + np.sum(u_bar_vec * f_mom, axis=1)
    F = np.zeros((QL.shape[0], 5))
    F[:, 0] = f_rho
    F[:, 1:4] = f_mom
    F[:, 4] = f_e
    return F
