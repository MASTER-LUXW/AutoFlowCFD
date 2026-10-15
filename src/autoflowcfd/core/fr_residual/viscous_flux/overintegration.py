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

与无粘同一个 `K = f2c·D_fine - Σ_面 C_面`（`fr/face_flux_trace.py`）：体积散度的
L2 投影减去修正项本侧通量迹的精确面积分 `C_面`，界面核只施加公共通量与内罚项——离散守恒。
"""

import numpy as np
from numba import njit, prange

from autoflowcfd.core.fr_operators.flux_kernels.viscous import viscous_physical_flux_point
from autoflowcfd.core.fr_operators.volume_contract import (
    cell_blocks, contract_local, contravariant_point, lifted_segments,
)


@njit(cache=True, parallel=True)
def _viscous_volume_kernel(Q, grad_vel, grad_T, mu_t, mu, Pr, Pr_t, lo, c2f, det_f, inv_f, K2, combo, out,
                           n_blocks):
    """段内单元 `lo + i`：细点上重新求值粘性通量，`out[c] = K2[combo[i]] : adj(J) G`。"""
    n = combo.shape[0]
    n_fine, n_sps = c2f.shape
    for b in prange(n_blocks):
        gv_c = np.empty((n_sps, 9))
        mt_c = np.empty((n_sps, 1))
        q_f = np.empty((n_fine, 5))
        gv_f = np.empty((n_fine, 9))
        gt_f = np.empty((n_fine, 3))
        mt_f = np.empty((n_fine, 1))
        g_t = np.empty((3 * n_fine, 5))
        for i in range(b * n // n_blocks, (b + 1) * n // n_blocks):
            c = lo + i
            for s in range(n_sps):
                for a in range(3):
                    for d in range(3):
                        gv_c[s, 3 * a + d] = grad_vel[c, s, a, d]
                mt_c[s, 0] = mu_t[c, s]
            contract_local(c2f, Q[c], q_f)
            contract_local(c2f, gv_c, gv_f)
            contract_local(c2f, grad_T[c], gt_f)
            contract_local(c2f, mt_c, mt_f)
            for p in range(n_fine):
                g = viscous_physical_flux_point(q_f[p], gv_f[p].reshape((3, 3)), gt_f[p], mu, Pr, mt_f[p, 0], Pr_t)
                contravariant_point(det_f[i, p], inv_f[i, p], g, g_t[3 * p:3 * p + 3])
            contract_local(K2[combo[i]], g_t, out[c])


def viscous_volume_term(Q, grad_vel, grad_T, mu_t_field, mu, Pr, Pr_t, oi, n_sps):
    """粘性体积项 `K G~`（参考空间，调用方再除以 det J），返回 (n_cells, n_sps, 5)。

    ① Q / grad_vel / grad_T / mu_t 各自精确插值到细点（各自次数 <= order）；
    ② 在细点**重新求值**粘性通量（非线性函数本身在细点求值——去混叠的全部内容）；
    ③ 用解析精确的细点度量算逆变通量；
    ④ 与体积算子 K 一次收缩。
    逐单元在块内局部工作数组里走完（`fr_residual/inviscid.py::inviscid_volume_divergence` 文档"逐单元融合"），
    与此前按 32768 单元分块、每步各自分配块大小临时数组的版本逐位相同。`grad_vel` 可以是梯度数组的切片视图。
    """
    Q = np.ascontiguousarray(Q)
    out = np.empty((Q.shape[0], n_sps, 5))
    mu_t = np.asarray(mu_t_field, dtype=np.float64)
    for lo, hi, c2f, det_f, inv_f, K2, combo in lifted_segments(oi):
        _viscous_volume_kernel(Q, grad_vel, grad_T, mu_t, float(mu), float(Pr), float(Pr_t), lo, c2f, det_f, inv_f,
                               K2, combo, out, cell_blocks(hi - lo))
    return out
