"""平均流的过积分细->粗限制是参考单元上到求解空间的 L2 投影（四面体与棱柱原生基）。

离散守恒依赖它（`fr/native_tet/overintegration.py` 模块文档"限制：平均流投影、湍流插值"）：

1. 求解空间内的场原样保持（`f2c c2f = I`）；
2. 细层多项式的参考单元积分保持（`w_c^T f2c = w_f^T`）——全域积分 `Σ w_s J_s R_s`
   等于解点求积作用在限制结果上，这一条就是守恒所需的全部；
3. 与独立构造对照：PKD 三角形 / 四面体模态与 Legendre 挤出模态在参考单元上两两正交，
   L2 投影就是"细层模态系数只保留粗层那几个模态"（算子本身用 Duffy 求积构造，两条
   路径互不依赖）。
"""

import numpy as np
import pytest

from autoflowcfd.fr.native_prism.basis import (
    build_native_prism_nodes, build_native_prism_vandermonde, restricted_prism_modes,
)
from autoflowcfd.fr.native_prism.overintegration import (
    build_native_prism_l2_projection, build_native_prism_overintegration_operators,
)
from autoflowcfd.fr.native_prism.quadrature import build_native_prism_sp_weights
from autoflowcfd.fr.native_tet.basis import build_native_tet_operators, restricted_tet_modes
from autoflowcfd.fr.native_tet.overintegration import (
    _native_modal_vandermonde, build_native_tet_l2_projection, build_native_tet_overintegration_operators,
)
from autoflowcfd.fr.native_tet.quadrature import build_native_tet_sp_weights


def _tet(order, oo):
    ref_c, _ = build_native_tet_operators(order)
    ref_f, c2f, _, _ = build_native_tet_overintegration_operators(order, oo)
    f2c = build_native_tet_l2_projection(order, oo)
    modes_c, modes_f = restricted_tet_modes(order), restricted_tet_modes(oo)
    return (c2f, f2c, build_native_tet_sp_weights(order), build_native_tet_sp_weights(oo),
            _native_modal_vandermonde(ref_c, modes_c), _native_modal_vandermonde(ref_f, modes_f), modes_c, modes_f)


def _prism(order, oo):
    ref_c = build_native_prism_nodes(order)
    ref_f, c2f, _, _ = build_native_prism_overintegration_operators(order, oo)
    f2c = build_native_prism_l2_projection(order, oo)
    return (c2f, f2c, build_native_prism_sp_weights(order), build_native_prism_sp_weights(oo),
            build_native_prism_vandermonde(order, ref_c)[0], build_native_prism_vandermonde(oo, ref_f)[0],
            restricted_prism_modes(order), restricted_prism_modes(oo))


@pytest.mark.parametrize("build", [_tet, _prism], ids=["tet", "prism"])
@pytest.mark.parametrize("order", [1, 2, 3])
def test_restrict_f2c_is_l2_projection(build, order):
    c2f, f2c, w_c, w_f, V_c, V_f, modes_c, modes_f = build(order, 2 * order)
    np.testing.assert_allclose(f2c @ c2f, np.eye(c2f.shape[1]), atol=1e-12)
    np.testing.assert_allclose(w_c @ f2c, w_f, atol=1e-13)
    keep = [modes_f.index(m) for m in modes_c]
    truncation = V_c @ np.linalg.inv(V_f)[keep]
    np.testing.assert_allclose(f2c, truncation, atol=1e-11)
