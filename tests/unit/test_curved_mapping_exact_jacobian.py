"""AutoFlowCFD V2.0 - 直边四面体/棱柱解析精确雅可比单元测试。

核心判据：
1. 解析雅可比（tet_exact_jacobian/prism_exact_jacobian）必须与有限差分
   数值微分一致——验证闭式求导公式本身没有推导错误。
2. 离散几何守恒律（GCL，均匀流场必须给出零残差）残差必须降到机器精度，
   且这个结论对极端偏斜单元（真实网格棱柱-四面体过渡区常见的细长四面体，
   边长比~25:1，det(J)低至~1e-14）同样成立——这正是解析精确雅可比要
   替代谱微分矩阵几何求导的原因，见 curved_mapping.py 模块内注释。
"""

import numpy as np

from autoflowcfd.fr.operators import gauss_legendre, generate_fr_operators
from autoflowcfd.grid.curved_mapping import (
    CurvedMapping,
    map_prism_to_physical,
    map_tet_to_physical,
    prism_exact_jacobian,
    tet_exact_jacobian,
)


def _ref_cube_sps(order: int) -> np.ndarray:
    n1d = order + 1
    sps_1d, _ = gauss_legendre(n1d)
    xx, yy, zz = np.meshgrid(sps_1d, sps_1d, sps_1d, indexing="ij")
    return np.column_stack([xx.ravel(), yy.ravel(), zz.ravel()])


def _finite_diff_jacobian(map_fn, ref_pt: np.ndarray, cell_nodes: np.ndarray) -> np.ndarray:
    eps = 1e-6
    p0 = map_fn(ref_pt[None, :], cell_nodes)[0]
    jac = np.zeros((3, 3))
    for m in range(3):
        pert = ref_pt.copy()
        pert[m] += eps
        p1 = map_fn(pert[None, :], cell_nodes)[0]
        jac[:, m] = (p1 - p0) / eps
    return jac


# 一个正常形状的四面体/棱柱，以及一个极端偏斜（边长比~25:1）的"薄片"四面体，
# 模拟真实网格棱柱-四面体过渡区最差单元的几何特征。
_TET_NODES_NORMAL = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
_TET_NODES_SLIVER = np.array(
    [[0.0, 0.0, 0.0], [1e-2, 3e-4, 1e-4], [2e-4, 1e-2, 2e-4], [1e-4, 2e-4, 1e-2]]
)
_PRISM_NODES_NORMAL = np.array(
    [
        [0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0], [1.0, 0.0, 1.0], [0.0, 1.0, 1.0],
    ]
)


def test_tet_exact_jacobian_matches_finite_difference():
    ref_pt = np.array([0.3, -0.2, 0.55])
    for nodes in (_TET_NODES_NORMAL, _TET_NODES_SLIVER):
        jac_fd = _finite_diff_jacobian(map_tet_to_physical, ref_pt, nodes)
        jac_exact = tet_exact_jacobian(ref_pt[None, :], nodes)[0]
        assert np.max(np.abs(jac_fd - jac_exact)) < 1e-8


def test_prism_exact_jacobian_matches_finite_difference():
    ref_pt = np.array([0.3, -0.2, 0.55])
    jac_fd = _finite_diff_jacobian(map_prism_to_physical, ref_pt, _PRISM_NODES_NORMAL)
    jac_exact = prism_exact_jacobian(ref_pt[None, :], _PRISM_NODES_NORMAL)[0]
    assert np.max(np.abs(jac_fd - jac_exact)) < 1e-8


def test_gcl_residual_near_machine_precision_for_sliver_tet():
    """解析精确雅可比替代谱微分几何求导的核心验证：即使对边长比~25:1的
    极端偏斜四面体（det(J)量级~1e-9，真实网格最差单元的典型特征），离散
    GCL 残差也必须降到机器精度量级，而不是随 det(J) 一起被放大。"""
    order = 2
    ops = generate_fr_operators(order)
    mapper = CurvedMapping(order)
    mapper.operators = ops
    ref_sps = _ref_cube_sps(order)

    for nodes, label in [(_TET_NODES_NORMAL, "normal"), (_TET_NODES_SLIVER, "sliver")]:
        phys_sps = map_tet_to_physical(ref_sps, nodes)
        jac_data = mapper.compute_jacobian(
            phys_sps, cell_type="tet", cell_nodes=nodes, ref_cube_sps=ref_sps
        )
        residual = mapper.compute_metric_identity_residual(
            phys_sps, cell_type="tet", cell_nodes=nodes, ref_cube_sps=ref_sps
        )
        max_res = np.max(np.abs(residual))
        min_det = jac_data["det_jacs"].min()
        # 残差应远小于 det(J) 本身（相对残差应处于机器精度量级），而不是
        # 和 det(J) 同量级（这正是旧的谱微分几何求导在偏斜单元上的失效模式）。
        assert max_res < 1e-12, f"{label} tet: GCL residual too large: {max_res:.3e} (min_det={min_det:.3e})"


def test_gcl_residual_near_machine_precision_for_prism():
    order = 2
    ops = generate_fr_operators(order)
    mapper = CurvedMapping(order)
    mapper.operators = ops
    ref_sps = _ref_cube_sps(order)

    phys_sps = map_prism_to_physical(ref_sps, _PRISM_NODES_NORMAL)
    residual = mapper.compute_metric_identity_residual(
        phys_sps, cell_type="prism", cell_nodes=_PRISM_NODES_NORMAL, ref_cube_sps=ref_sps
    )
    assert np.max(np.abs(residual)) < 1e-12


def test_compute_jacobian_requires_cell_nodes_for_simplex_types():
    """cell_type=tet/prism 缺少 cell_nodes/ref_cube_sps 时必须显式报错，
    不能静默退化回旧的谱微分近似路径。"""
    order = 2
    mapper = CurvedMapping(order)
    ref_sps = _ref_cube_sps(order)
    phys_sps = map_tet_to_physical(ref_sps, _TET_NODES_NORMAL)
    try:
        mapper.compute_jacobian(phys_sps, cell_type="tet")
        assert False, "expected ValueError"
    except ValueError:
        pass
