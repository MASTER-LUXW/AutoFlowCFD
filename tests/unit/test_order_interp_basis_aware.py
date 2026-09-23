"""Order Continuation 的跨阶数延拓算子必须按**基**分派（2026-09-20）。

## 这份测试钉的是一个真实生产缺陷

`core/utils/order_continuation.py::interpolate_to_new_order` 一直用一维
Gauss 点的张量积 Lagrange 插值，而那个矩阵只对**坍缩棱柱基**的解点成立。
native 四面体（自 2026-09-03 起是四面体唯一实现）与 native 棱柱
（2026-09-20 起是默认）的解点都不在那个张量积网格上。

修复前实测（线性场 P1->P2 的最大相对误差）：

    棱柱基   棱柱单元    四面体单元
    坍缩     4.3e-16     **7.0e-01**
    原生     **1.4e-01**  **7.0e-01**

判据是**精确可表示性**：线性场在 P1 与 P2 空间里都精确可表示，所以正确
的延拓必须给出机器精度。这条与分辨率、与网格质量都无关。

P0->P1 恰好不受影响（P0 是常数场），所以生产的 `P0 -> P1 -> P2` 只在最后
那一跳被污染 —— 这正是它能长期不被发现的原因，也是本文件要覆盖到
`P1->P2` 的理由。
"""

import numpy as np
import pytest

from autoflowcfd.fr.native_padding import real_sps_per_cell
from autoflowcfd.fr.order_interp import (
    apply_order_interp, build_order_interp_matrices,
)

from tests.unit.test_fr_residual_inviscid import _build_synthetic_mixed_mesh

_A = np.array([2.0, -3.0, 5.0])


def _linear(mesh):
    return mesh.sps_coords @ _A + 7.0


@pytest.mark.parametrize("p,q", [(1, 2), (2, 3), (1, 3)])
def test_linear_field_is_lifted_exactly(p, q):
    """线性场延拓到更高阶必须是机器精度（两类单元、两条基）。"""
    mesh_p = _build_synthetic_mixed_mesh(p)
    mesh_q = _build_synthetic_mixed_mesh(q)
    got = apply_order_interp(_linear(mesh_p), mesh_p.n_prism_cells, p, q)
    want = _linear(mesh_q)

    n_real_prism, n_real_tet = real_sps_per_cell(q)
    n_prism = mesh_p.n_prism_cells
    for name, sl, n_real in (("prism", slice(0, n_prism), n_real_prism),
                             ("tet", slice(n_prism, None), n_real_tet)):
        err = np.abs(got[sl][:, :n_real] - want[sl][:, :n_real]).max()
        scale = float(np.abs(want[sl][:, :n_real]).max())
        assert err / scale < 1e-13, (
            f"{basis} {name} P{p}->P{q}: 线性场延拓相对误差 {err/scale:.3e} "
            f"不是机器精度 —— 延拓算子与该单元类型的解点/模态族不匹配")


def test_constant_field_survives_p0_to_p1():
    """P0->P1：常数场必须逐位保持（这一跳对任何插值都成立，是护栏）。"""
    mesh0 = _build_synthetic_mixed_mesh(0)
    f0 = np.full((mesh0.n_cells, mesh0.n_sps_per_cell), 7.0)
    got = apply_order_interp(f0, mesh0.n_prism_cells, 0, 1)
    np.testing.assert_array_equal(got, np.full_like(got, 7.0))


@pytest.mark.parametrize("p,q", [(1, 2), (2, 3)])
def test_padding_rows_equal_real_sp0(p, q):
    """延拓后的零填充槽位必须等于该单元真实 SP #0 —— 与
    `fr/native_padding.py` 的初始化约定一致（否则新阶数的填充块从第一步
    起就带着一个与约定不符的值）。"""
    mesh_p = _build_synthetic_mixed_mesh(p)
    got = apply_order_interp(_linear(mesh_p), mesh_p.n_prism_cells, p, q)
    n_real_prism, n_real_tet = real_sps_per_cell(q)
    n_prism = mesh_p.n_prism_cells
    n_sps_q = got.shape[1]
    for name, sl, n_real in (("prism", slice(0, n_prism), n_real_prism),
                             ("tet", slice(n_prism, None), n_real_tet)):
        if n_real >= n_sps_q:
            continue
        np.testing.assert_array_equal(
            got[sl][:, n_real:], np.repeat(got[sl][:, :1],
                                           n_sps_q - n_real, axis=1),
            err_msg=f"{name}: 填充槽位不等于真实 SP #0")


def test_padding_columns_of_the_source_do_not_contribute():
    """**源**的填充列必须零贡献：把填充槽位改成任意值，延拓结果的真实
    自由度部分必须逐位不变。

    这条对应"填充槽位的值在初始化后被冻结、不是该处多项式的取值"这个
    不变量；让它们参与插值就是把馊值搬进新阶数（坍缩档没有填充槽位，
    这条在那一档下恒真）。
    """
    p, q = 1, 2
    mesh_p = _build_synthetic_mixed_mesh(p)
    f = _linear(mesh_p)
    n_real_prism, n_real_tet = real_sps_per_cell(p)
    n_prism = mesh_p.n_prism_cells
    f_dirty = f.copy()
    f_dirty[:n_prism, n_real_prism:] = 1.0e6
    f_dirty[n_prism:, n_real_tet:] = -1.0e6

    a = apply_order_interp(f, n_prism, p, q)
    b = apply_order_interp(f_dirty, n_prism, p, q)
    n_real_q_prism, n_real_q_tet = real_sps_per_cell(q)
    np.testing.assert_array_equal(a[:n_prism, :n_real_q_prism],
                                  b[:n_prism, :n_real_q_prism])
    np.testing.assert_array_equal(a[n_prism:, :n_real_q_tet],
                                  b[n_prism:, :n_real_q_tet])


# 原来这里有一条 `test_collapsed_prism_matrix_is_bit_identical_to_the_legacy_one`
# ：坍缩档的棱柱延拓矩阵必须与既有的一维 Lagrange 张量积实现逐位相同，
# 用来保证 2026-09-20 那次"延拓算子按基分派"的改动不触动已验证的坍缩档。
# 坍缩棱柱基已于 2026-09-23 删除，该判据随之移除 —— 它当时是**通过**的
# （`np.testing.assert_array_equal`，逐位相同）。
#
# 同时移除了本文件四组 `parametrize(["collapsed", "native"])` 的坍缩臂：
# 删除坍缩档之后 `build_order_interp_matrices` 不再读
# `AFCFD_PRISM_BASIS`，那些用例设了环境变量却**不被任何代码消费**，
# 名字说在测坍缩、实际把原生跑了两遍 —— 这种"配置被静默忽略"的测试比
# 没有测试更糟（本项目已因同类问题出过真实缺陷，见项目记忆
# `defaults_retuned_2026_09_17` 里"固定 CFL 请求被静默丢弃"那条）。


@pytest.mark.parametrize("bad", [(-1, 1), (1, -2)])
def test_negative_order_raises(bad):
    with pytest.raises(ValueError, match="阶数必须非负"):
        build_order_interp_matrices(*bad)


def test_shape_mismatch_raises(monkeypatch):
    monkeypatch.setenv("AFCFD_PRISM_BASIS", "native")
    f = np.zeros((4, 27))
    with pytest.raises(ValueError, match="全局宽度"):
        apply_order_interp(f, 2, 1, 2)      # 27 是 P2 的宽度，不是 P1 的
