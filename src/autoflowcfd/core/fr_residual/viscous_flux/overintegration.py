"""AutoFlowCFD V2.0 - 粘性体积项（过积分去混叠 + 修正项本侧通量迹，体积算子 K）。

## 为什么粘性体积项必须过积分

粘性通量 `G(Q, grad u, grad T, mu_t)` 里有 `tau = mu_eff (grad u + grad u^T)`、`u·tau`、
`k grad T`，`u = rho_u/rho`、`T = p/(rho R)` 都是 Q 的有理函数，再乘 `adj(J)`，真实次数远高
于求解阶数；直接在解点上微分等价于"先混叠再求导"。平板边界层（2304 单元、P1，400 步的真实
粘性梯度状态）以更高过积分阶数（3x）为参照：

    解点上微分      能量分量相对差 0.632442
    过积分 @ 2x     能量分量相对差 0.000000（与 3x 逐位相同 -> 已收敛）

动量分量两者逐位相同（P1 常粘度下 tau 逐单元为常数，零混叠），差别全在能量。代价是整步约
+21%（同一算例背靠背实测 119.7 -> 145.3 ms/step）。2026-09-17 起默认过积分；2026-10-01 起
"解点上微分"那一档（`AFCFD_VISC_OVERINT=off`）删除——它给出的是被混叠污染的能量通量，不是
另一种可选取舍，而且修正项本侧通量改成体积通量多项式的迹之后，两档都已不能逐位复现历史结果，
保留它只剩一套要同步维护的第二离散。

仍然存在的上游局限（如实写明）：`grad_vel`/`grad_T` 是 `compute_physical_gradient` 在解点上
算出的（`adj(J) D Q / det J` 同样是乘积），这里把它们精确插值到细点，去掉的是通量与度量
这一层的混叠。

## 体积算子 K

与无粘同一个 `K = f2c·D_fine - Σ_面 lift·diag(w)·Tn`（`fr/face_flux_trace.py`）：体积散度的
L2 投影减去修正项的本侧通量迹，界面核只施加公共通量与内罚项——离散守恒。
"""

import numpy as np

from autoflowcfd.core.fr_operators.flux_kernels import viscous_physical_flux_batch
from autoflowcfd.core.fr_operators.volume_contract import (
    OVERINT_CHUNK_CELLS,
    contract_shared_operator_1axis,
    contract_shared_operator_2axis,
    contravariant_flux_from_metric,
)


def viscous_volume_term(Q, grad_vel, grad_T, mu_t_field, mu, Pr, Pr_t, oi, n_sps):
    """粘性体积项 `K G~`（参考空间，调用方再除以 det J），返回 (n_cells, n_sps, 5)。

    ① Q / grad_vel / grad_T / mu_t 各自精确插值到细点（各自次数 <= order）；
    ② 在细点**重新求值**粘性通量（非线性函数本身在细点求值——去混叠的全部内容）；
    ③ 用解析精确的细点度量算逆变通量；
    ④ 与体积算子 K 一次收缩。
    按单元分块（`OVERINT_CHUNK_CELLS`），块内细点数组用完即弃。
    """
    n_cells = Q.shape[0]
    div_comp = np.zeros((n_cells, n_sps, 5))
    # 每段自带自己的 n_fine 与已切好的细点度量；度量按**段内局部**索引切
    # （`i0 = c0 - seg_lo`）——用全局 c0 去切段内数组会静默取到错误的单元。
    for (seg_lo, seg_hi, n_fine, det_seg, inv_seg, op_c2f, _D_fine, _f2c), op_K in zip(oi["segs"],
                                                                                       oi["lifted_div"]):
        for c0 in range(seg_lo, seg_hi, OVERINT_CHUNK_CELLS):
            c1 = min(c0 + OVERINT_CHUNK_CELLS, seg_hi)
            nb = c1 - c0
            i0, i1 = c0 - seg_lo, c1 - seg_lo
            Q_f = contract_shared_operator_1axis(op_c2f, np.ascontiguousarray(Q[c0:c1]))            # (nb,n_fine,5)
            gv_f = contract_shared_operator_1axis(
                op_c2f, np.ascontiguousarray(grad_vel[c0:c1].reshape(nb, n_sps, 9))).reshape(nb, n_fine, 3, 3)
            gT_f = contract_shared_operator_1axis(op_c2f, np.ascontiguousarray(grad_T[c0:c1]))     # (nb,n_fine,3)
            mut_f = contract_shared_operator_1axis(
                op_c2f, np.ascontiguousarray(mu_t_field[c0:c1][:, :, None]))[..., 0]               # (nb,n_fine)
            G_phys_f = viscous_physical_flux_batch(
                np.ascontiguousarray(Q_f.reshape(-1, 5)),
                np.ascontiguousarray(gv_f.reshape(-1, 3, 3)),
                np.ascontiguousarray(gT_f.reshape(-1, 3)),
                mu, Pr,
                np.ascontiguousarray(mut_f.reshape(-1)),
                Pr_t,
            ).reshape(nb, n_fine, 3, 5)
            del Q_f, gv_f, gT_f, mut_f
            G_tilde_f = contravariant_flux_from_metric(det_seg[i0:i1], inv_seg[i0:i1], G_phys_f)
            del G_phys_f
            div_comp[c0:c1] = contract_shared_operator_2axis(op_K, G_tilde_f)
            del G_tilde_f
    return div_comp
