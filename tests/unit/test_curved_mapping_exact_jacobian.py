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

import pytest

from autoflowcfd.grid.curved_mapping.curved_mapping import (
    CurvedMapping,
    map_prism_to_physical,
    map_tet_to_physical,
)
from autoflowcfd.grid.curved_mapping.curved_mapping_exact_jacobian import prism_exact_jacobian, tet_exact_jacobian


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
    """即使对边长比~25:1 的极端偏斜四面体（真实网格最差单元的典型特征），离散 GCL 残差（原生四面体基、与残差
    组装同一套体积微分算子）也必须在机器精度量级，而不是随 det(J) 一起被放大。"""
    mapper = CurvedMapping(2)
    for nodes, label in [(_TET_NODES_NORMAL, "normal"), (_TET_NODES_SLIVER, "sliver")]:
        max_res = float(np.max(np.abs(mapper.compute_metric_identity_residual("tet", nodes))))
        assert max_res < 1e-12, f"{label} tet: GCL residual too large: {max_res:.3e}"


def test_gcl_residual_near_machine_precision_for_prism():
    residual = CurvedMapping(2).compute_metric_identity_residual("prism", _PRISM_NODES_NORMAL)
    assert np.max(np.abs(residual)) < 1e-12


def test_unknown_cell_type_is_rejected():
    """只有原生四面体/棱柱：其它单元类型（此前的六面体张量积分支）报错。"""
    with pytest.raises(ValueError, match="cell_type"):
        CurvedMapping(2).compute_metric_identity_residual("hex", _TET_NODES_NORMAL)
