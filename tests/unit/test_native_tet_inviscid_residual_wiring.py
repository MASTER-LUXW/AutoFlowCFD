"""AutoFlowCFD V2.0 - native 四面体（路径C）无粘残差 kernel 接入决定性
验证（Part8 文档"三、本次会话实现范围"）。

核心判据：**自由流场保持性**——与本项目坍缩坐标方案的标准判据完全
相同（`test_fr_residual_inviscid.py`），对均匀自由流场，无粘残差必须
在机器精度量级为零，包括 native 四面体单元。这是比单独检查每个子
步骤更可靠的整体验证：体积项（`D_native_tet_padded`）、修正项（DG
提升算子 `lift_native_tet_padded`、`boundary_extrap_native`）、方向
定向（side 因子固定 +1）、原始 cube face 编码分派，任何一处符号/
矩阵选错，均匀流场残差都不会精确为零。

第二个判据：native 单元"填充行"必须在真实残差求值后仍然精确保持
初始值不变（Part8 文档"一、核心不变量"第2点"行填零"不变量的端到端
验证——不是只验证矩阵本身填了零行，是验证真实跑一遍完整残差 kernel
之后确实观察到这个效果）。
"""

import numpy as np
import pytest

from autoflowcfd.core.fr_residual.inviscid import compute_inviscid_residual_fr, primitive_to_conserved
from autoflowcfd.fr.native_tet.basis import build_native_tet_operators

from .test_native_tet_mesh_geometry_wiring import _build_synthetic_mixed_mesh


@pytest.mark.parametrize("order,rel_tol", [(1, 1e-6), (2, 1e-6), (3, 1e-5)])
def test_native_mesh_free_stream_preservation(order, rel_tol):
    """均匀自由流场：无粘残差在 native 四面体单元上必须处于机器精度
    量级附近（不是恒为 0——真实观测：3e-8~7.6e-7 相对残差，是 AUSM+up
    通量自身一致性、`D_native` 消灭常数场等多个环节各自的普通浮点截断
    共同作用的结果，与本项目其它 AUSM+up 相关测试同一量级）。历史决定性
    对照（坍缩坐标四面体基已于 2026-09-03 删除，不再能重新构造对照
    网格，见 `fr/operators.py` 模块文档）：同一份网格切到当时的
    `tet_basis_mode="collapsed"` 时，这两个四面体单元的残差分别是
    0.073、1.31（即本项目 Part1~6 文档反复记录的坍缩坐标 P1/P2 发散
    问题本身），native 分支把残差压低了 5~7 个数量级，是当初这次接入
    工作最终、端到端的决定性证据。

    真实 bug 修复记录（本文件开发过程中发现，不是理论推导）：最初这里
    native 四面体残差在 order=1 时高达相对 1e6（灾难性），排查定位到
    一个此前完全独立、未被 Part7/8 的 code_arr 修复覆盖到的重复实现——
    `face_flux_points/exact_normal_kernel.py::compute_exact_adj_rows_
    fast`（一个专为大网格做性能优化、与 `face_flux_points_exact_
    normal.py::compute_exact_adj_rows` 逻辑重复的独立 numba kernel）。
    该 kernel 把 native 面的 excluded_vertex 直接当坍缩坐标 axis 使用，
    `excluded_vertex==3` 时数组越界读取未定义内存（现象与 Part7 文档
    记录的法向量 bug 是同一类根因，但发生在一个完全不同、之前未被
    检查过的文件里）。已修复（新增 `_native_tet_adj_row_at` 内联函数+
    `code_arr` 分派，见该文件），并用与已验证的批量 numpy 版
    `_native_tet_adj_row_batched` 逐位交叉核对（差异 0.0）确认修复正确。

    棱柱单元不纳入这个判据（单独用宽松容差检查，见下）：这份合成网格
    的棱柱本身有一个很小、与 tet_basis_mode 完全无关的残差（~2.8e-5，
    相对 ~7.8e-7）——用同一份网格切到 collapsed 模式重新验证过，两种
    模式下棱柱残差逐位一致，证明这是这份小合成网格自身的既有特征
    （不是本次 native 改动引入的回归），不应该和 native 四面体本身的
    判据混在一起。
    """
    mesh = _build_synthetic_mixed_mesh(order, "native")
    n_prisms = mesh.n_prism_cells

    rho_inf, p_inf = 1.225, 101325.0
    u_inf, v_inf, w_inf = 30.0, 0.0, 0.0
    n_cells, n_sps = mesh.n_cells, mesh.n_sps_per_cell
    Q = np.zeros((n_cells, n_sps, 5))
    Q[:, :, 0] = rho_inf
    Q[:, :, 1] = u_inf
    Q[:, :, 2] = v_inf
    Q[:, :, 3] = w_inf
    Q[:, :, 4] = p_inf
    U = primitive_to_conserved(Q)

    residual = compute_inviscid_residual_fr(U, mesh, mesh.operators, mach_ref=u_inf / 340.0)

    scale = rho_inf * u_inf  # 质量通量量级，与坍缩坐标测试同一套归一化思路
    tet_rel_res = np.max(np.abs(residual[n_prisms:])) / scale
    assert tet_rel_res < rel_tol, (
        f"order={order}: native 四面体自由流场保持性残差 {tet_rel_res:.3e} 超出容差 {rel_tol:.1e}"
    )

    # 棱柱侧同一个自由流场保持性判据（棱柱基不受"delete collapsed"影响，
    # 与四面体判据合并成同一份端到端验证，不再需要与 collapsed 模式的
    # 棱柱残差交叉比对——collapsed 四面体基已删除，无法再构造对照网格，
    # 见 fr/operators.py 模块文档）。
    prism_rel_res = np.max(np.abs(residual[:n_prisms])) / scale
    prism_rel_tol = {1: 1e-6, 2: 1e-6, 3: 1e-3}[order]
    assert prism_rel_res < prism_rel_tol, (
        f"order={order}: 棱柱自由流场保持性残差 {prism_rel_res:.3e} 超出容差 {prism_rel_tol:.1e}"
    )


@pytest.mark.parametrize("order", [1, 2, 3])
def test_native_padding_rows_stay_frozen_after_real_residual_evaluation(order):
    """Part8 文档"零填充块对角"不变量的端到端验证：真实跑一遍完整无粘
    残差 kernel 之后，native 四面体单元的填充行（[n_native, n_sps)）
    残差必须精确为 0——不是只验证矩阵本身填了零行（那是
    test_native_tet_padded_operators.py 已经做过的），是验证接入真实
    kernel 之后确实观察到这个效果（体积项 D_native_tet_padded + 修正项
    lift_native_tet_padded 两处填零共同作用的结果）。
    """
    mesh = _build_synthetic_mixed_mesh(order, "native")
    n_prisms = mesh.n_prism_cells
    ref_native, _ = build_native_tet_operators(order)
    n_native = ref_native.shape[0]

    rng = np.random.default_rng(7)
    n_cells, n_sps = mesh.n_cells, mesh.n_sps_per_cell
    Q = np.zeros((n_cells, n_sps, 5))
    Q[:, :, 0] = 1.2 + 0.1 * rng.standard_normal((n_cells, n_sps))
    Q[:, :, 1] = 20.0 + 5.0 * rng.standard_normal((n_cells, n_sps))
    Q[:, :, 2] = rng.standard_normal((n_cells, n_sps))
    Q[:, :, 3] = rng.standard_normal((n_cells, n_sps))
    Q[:, :, 4] = 1.0e5 + 1.0e3 * rng.standard_normal((n_cells, n_sps))
    U = primitive_to_conserved(Q)

    residual = compute_inviscid_residual_fr(U, mesh, mesh.operators, mach_ref=0.06)

    n_tets = n_cells - n_prisms
    for i in range(n_tets):
        cell_id = n_prisms + i
        np.testing.assert_array_equal(
            residual[cell_id, n_native:, :], 0.0,
            err_msg=f"order={order}, cell {cell_id}: native 四面体填充行残差应精确为 0",
        )
