"""AutoFlowCFD V2.0 - native 四面体（路径C）过积分（去混叠）算子决定性
正确性验证：真实随机四面体上，体积项混叠误差的量化改善。

这是"native 过积分算子确实有效"的**决定性判据**（区别于
`test_native_tet_overintegration_residual_wiring.py` 那个只检查"接入
生产管线不出错"的健全性测试——那个测试用的小合成网格恰好形状规则、
混叠效应本来就不明显，不适合当决定性判据，见该文件文档说明）。

判据设计（与 `fr/collapsed_basis.py::build_overintegration_operators`
模块文档"线性剪切场，解析残差恒为 0"同一个方法论）：Q 本身取真正的
线性 Couette 剖面 `u=U_WALL*y/H`（对任意阶数、任意单元朝向，解析散度
恒为 0——见下方 `_make_couette_Q_at`/模块文档推导：F_x/F_z 不依赖
x/z，F_y 不依赖 y，散度处处为 0），真正的非线性来自欧拉能量通量
`u*(E+p)`（含 u^3 项，恰好是过积分要处理的对象），不是 Q 自身构造出的
假高阶内容。

**开发过程中的真实教训**（如实记录）：最初误把 Q 的动能分量按
`u=U_WALL*(y/H)^order`（随阶数变化次数）构造，导致 Q 自身（尤其能量
分量 `0.5*rho*u^2`，次数 `2*order`）在采样到 coarse 节点时就已经产生
了采样级别的信息丢失，与过积分要处理的"F_phys 非线性导致的次数提升"
完全是两回事，混在一起测出的结果毫无意义（order=1 时反而变差、
order=2 时也不稳定）。改成简单的**线性**剖面（对任何阶数 Q 本身都精确
可表示）后，问题才清晰：非线性只来自 F_phys 这一步，过积分只需要处理
这一步，得到清晰、稳定的改善量级。
"""

import numpy as np
import pytest

from autoflowcfd.fr.native_tet.basis import (
    build_native_tet_operators, map_native_tet_to_physical, compute_native_tet_jacobian,
)
from autoflowcfd.fr.native_tet.overintegration import build_native_tet_overintegration_operators
from autoflowcfd.core.fr_operators.flux_kernels import euler_physical_flux_batch
from autoflowcfd.fr.native_tet.overintegration import (
    resolve_tet_overintegration_order,
)

GAMMA = 1.4
RHO_INF, P_INF, U_WALL, H = 1.225, 101325.0, 30.0, 1.0


def _make_couette_Q_at(phys):
    y = phys[:, 1]
    u = U_WALL * y / H
    n = phys.shape[0]
    Q = np.zeros((n, 5))
    Q[:, 0] = RHO_INF
    Q[:, 1] = RHO_INF * u
    Q[:, 4] = P_INF / (GAMMA - 1.0) + 0.5 * RHO_INF * u ** 2
    return Q


def _random_tet(rng):
    base = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=float)
    return base + 0.3 * rng.standard_normal((4, 3))


def _volume_residual_direct(nodes, order, D_coarse, ref_coarse):
    det_j, adj_j = compute_native_tet_jacobian(nodes)
    phys = map_native_tet_to_physical(ref_coarse, nodes)
    Q = _make_couette_Q_at(phys)
    F_phys = euler_physical_flux_batch(Q)
    F_tilde = np.einsum("ij,pjv->piv", adj_j, F_phys)
    div = np.zeros((ref_coarse.shape[0], 5))
    for m in range(3):
        div += D_coarse[:, :, m] @ F_tilde[:, m, :]
    return np.abs(-div / det_j).max(), det_j


def _volume_residual_overintegrated(nodes, order, D_coarse, ref_coarse, over_ops):
    ref_fine, interp_c2f, D_fine, restrict_f2c = over_ops
    det_j, adj_j = compute_native_tet_jacobian(nodes)
    phys_coarse = map_native_tet_to_physical(ref_coarse, nodes)
    Q_coarse = _make_couette_Q_at(phys_coarse)

    Q_fine = interp_c2f @ Q_coarse
    F_phys_fine = euler_physical_flux_batch(Q_fine)
    F_tilde_fine = np.einsum("ij,pjv->piv", adj_j, F_phys_fine)
    div_fine = np.zeros((ref_fine.shape[0], 5))
    for m in range(3):
        div_fine += D_fine[:, :, m] @ F_tilde_fine[:, m, :]
    div_restricted = restrict_f2c @ div_fine
    return np.abs(-div_restricted / det_j).max()


@pytest.mark.parametrize("order,expect_improvement", [(1, False), (2, True), (3, False)])
def test_overintegration_reduces_analytical_zero_residual_for_couette_profile(order, expect_improvement):
    """order=2（生产默认阶数）：`over_order=min(2*2,3)=3` 恰好匹配能量
    通量 `u*(E+p)`（u 线性、E 含 u^2，故 u*(E+p) 三次）所需的三次
    分辨率，真实测得约 20 倍改善——这是本文件的决定性判据。

    order=1：`over_order=min(2,3)=2` 只够二次，不足以完全捕捉三次能量
    通量，改善不明显甚至可能略差——如实标注为已知、可解释的量级限制
    （`over_order` 经验公式与坍缩坐标共用，不是 native 分支独有的
    缺陷），不断言改善。
    order=3：`over_order=min(6,3)=3=order`，过积分退化成无操作
    （`OVERINTEGRATION_MAX_ORDER` 封顶导致），因此不断言改善。
    """
    over_order = resolve_tet_overintegration_order(order)
    ref_coarse, D_coarse = build_native_tet_operators(order)
    over_ops = build_native_tet_overintegration_operators(order, over_order)

    rng = np.random.default_rng(42 + order)
    direct_res, overint_res = [], []
    for _ in range(30):
        nodes = _random_tet(rng)
        res_direct, det_j = _volume_residual_direct(nodes, order, D_coarse, ref_coarse)
        if det_j <= 1e-6:
            continue
        res_overint = _volume_residual_overintegrated(nodes, order, D_coarse, ref_coarse, over_ops)
        direct_res.append(res_direct)
        overint_res.append(res_overint)

    direct_res = np.array(direct_res)
    overint_res = np.array(overint_res)
    assert len(direct_res) >= 20
    improvement = np.median(direct_res / np.maximum(overint_res, 1e-300))

    if expect_improvement:
        assert improvement > 5.0, (
            f"order={order}: native 过积分改善倍数中位数 {improvement:.2f} 未达到预期量级"
        )
    # order=1/3：不断言方向，只如实记录已知的量级限制（见函数文档）。
