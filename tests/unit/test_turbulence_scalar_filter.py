"""真实 bug 回归测试（2026-09-12）：`fr_solver/filter.py::filter_scalar_field`
——k/omega 场此前完全不参与模态滤波（`_filter_flat_U` 只滤平均流 5 个
守恒变量），理由是"湍流方程走独立单步显式更新，不经过多级 RK 因此不
积累混叠"。cube_demo 791,492 单元真实网格 P1 直连（无 Order
Continuation）长程测试决定性证伪这个理由：全新、干净的 P1 启动在数十
步内 omega 大范围失控增长（8.6% 单元逼近 1e6 安全上限），定位到具体
单元（如 153869）P1 多项式在相邻解点间出现数量级跳变（同一单元 8 个
解点里部分 ~22 万、部分卡在物理下限），外插到面后被上风格式放大成
巨大虚假对流残差——是与平均流完全同一类"坍缩坐标节点配置法高阶模态
混叠"病理，只是发生在标量湍流场而不是守恒变量上。

修复：k_field/omega_field 复用与平均流完全相同的 filter_prism/
filter_tet 矩阵，在每次 update_fields 之后滤波一次（P0 下 n_sps=1，
矩阵退化为单位矩阵，天然是无操作，见 filter_scalar_field 文档）。

注意（已有 tests/unit/test_modal_filter.py 记录的真实教训）：这个
滤波矩阵本身有 2026-08-29 调查记录过的已知张力（过度削弱中间模态
压制会导致真实网格数步内混叠失稳到 ~1e78 量级）——本次修复只是把
*现有、已验证生效* 的滤波矩阵应用范围扩大到 k/omega，不改动滤波矩阵
本身的构造/压制强度，因此不重新触发那个已经解决的张力。
"""
import numpy as np
import pytest

from autoflowcfd.core.fr_solver.filter import filter_scalar_field
from autoflowcfd.fr.modal_filter import build_prism_modal_filter, build_tet_modal_filter
from autoflowcfd.fr.operators import gauss_legendre


def _ref_cube_sps(order: int) -> np.ndarray:
    n1d = order + 1
    sps_1d, _ = gauss_legendre(n1d)
    xx, yy, zz = np.meshgrid(sps_1d, sps_1d, sps_1d, indexing="ij")
    return np.column_stack([xx.ravel(), yy.ravel(), zz.ravel()])


class TestFilterScalarFieldConstantPreserved:
    """常数场必须严格不受滤波影响（与平均流滤波的核心不变量完全一致，
    见 test_modal_filter.py::TestConstantFieldPreserved 同名判据）。"""

    def test_constant_field_preserved_mixed_mesh(self):
        order = 1
        ref = _ref_cube_sps(order)
        n_sps = ref.shape[0]
        filter_tet = build_tet_modal_filter(order, ref)
        filter_prism = build_prism_modal_filter(order, ref)

        n_prism, n_tet = 3, 4
        n_cells = n_prism + n_tet
        phi = np.full((n_cells, n_sps), 17.5)

        out = filter_scalar_field(phi, n_prism, filter_prism, filter_tet)

        np.testing.assert_allclose(out, phi, atol=1e-9)

    def test_p0_is_noop(self):
        """P0（n_sps=1）下滤波矩阵退化为单位矩阵，必须是纯粹的 no-op。"""
        order = 0
        ref = _ref_cube_sps(order)
        n_sps = ref.shape[0]
        assert n_sps == 1
        filter_tet = build_tet_modal_filter(order, ref)
        filter_prism = build_prism_modal_filter(order, ref)

        rng = np.random.default_rng(0)
        n_prism, n_tet = 2, 3
        phi = rng.uniform(0.01, 5.0, size=(n_prism + n_tet, n_sps))

        out = filter_scalar_field(phi, n_prism, filter_prism, filter_tet)

        np.testing.assert_allclose(out, phi, atol=1e-12)


class TestFilterScalarFieldMatchesMeanFlowFilter:
    """接线正确性：`filter_scalar_field` 必须与平均流已经过充分验证的
    `_filter_flat_U`（见 test_modal_filter.py）在数学上完全等价——把
    同一个标量场当成"5 个变量都相同"的守恒变量数组喂给 `_filter_flat_U`，
    每个变量分量的滤波结果必须与 `filter_scalar_field` 逐位一致。这样
    真正验证矩阵本身的压制强度、常数场恒等、最高阶模态压制等物理性质
    （已经由 test_modal_filter.py 详尽覆盖，包括 2026-08-29 调查记录的
    已知张力）无需重新验证，本文件只需确认"接线正确"这一件事。"""

    def test_equivalent_to_mean_flow_filter(self):
        from autoflowcfd.core.fr_solver.filter import _filter_flat_U

        order = 1
        ref = _ref_cube_sps(order)
        n_sps = ref.shape[0]
        filter_tet = build_tet_modal_filter(order, ref)
        filter_prism = build_prism_modal_filter(order, ref)

        rng = np.random.default_rng(2)
        n_prism, n_tet = 2, 3
        n_cells = n_prism + n_tet
        phi = rng.uniform(-50, 50, size=(n_cells, n_sps))

        out_scalar = filter_scalar_field(phi, n_prism, filter_prism, filter_tet)

        # 把同一个标量场复制成 5 份塞进 _filter_flat_U 期望的 (n_cells,
        # n_sps, 5) 形状，取任意一个分量（5 个分量此前都相同、结果也
        # 该都相同）与 filter_scalar_field 的输出比较。
        U_flat = np.repeat(phi[:, :, None], 5, axis=2).reshape(n_cells * n_sps, 5)
        out_meanflow = _filter_flat_U(U_flat.copy(), n_cells, n_sps, n_prism, filter_prism, filter_tet)
        out_meanflow = out_meanflow.reshape(n_cells, n_sps, 5)

        for v in range(5):
            np.testing.assert_allclose(
                out_scalar, out_meanflow[:, :, v], atol=1e-10,
                err_msg=f"filter_scalar_field 与平均流滤波在分量 {v} 上结果不一致",
            )
