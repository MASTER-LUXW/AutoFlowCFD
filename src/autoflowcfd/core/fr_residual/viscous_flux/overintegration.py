"""AutoFlowCFD V2.0 - 粘性体积项过积分（去混叠）开关与细点路径

从 `src/autoflowcfd/core/fr_residual/viscous_flux.py`(原 560 行)拆出(2026-09-24, 项目"单文件不超 500 行"规范)。**纯搬家, 逻辑未改**。
"""

import os

import numpy as np



from autoflowcfd.core.fr_operators.flux_kernels import viscous_physical_flux_batch

from autoflowcfd.core.fr_operators.volume_contract import (
    contract_shared_operator_1axis,
    contract_shared_operator_2axis,
    contravariant_flux_from_metric,
    OVERINT_CHUNK_CELLS,
)


def resolve_viscous_overintegration() -> str:
    """粘性体积项是否走过积分（去混叠）：`AFCFD_VISC_OVERINT = off | on`，
    **默认 on（2026-09-17 从 off 改）**。

    ## 为什么改默认（实测，平板边界层算例 2304 单元，跑到 400 步的真实
    ## 粘性梯度状态上求一次粘性残差）

    以 `on` + `AFCFD_OVERINT_ORDER_RULE=3x`（更高的过积分阶数）为参照：

        off @ 2x（原默认）   能量分量相对差 0.632442
        on  @ 2x（新默认）   能量分量相对差 0.000000   <- 逐位相同

    也就是说**过积分的结果在生产过积分阶数上就已经收敛**（提到 3x 逐位
    不变），而不过积分的结果差 63%。这是去混叠收敛性的标准签名：`on` 是
    收敛值，`off` 是被混叠污染的值。

    逐分量看更清楚（off vs on）：

        rho / rho_u / rho_v / rho_w   逐位相同（0.0）
        rho_E                          相对差 0.632442

    动量分量不变是对的：P1 下常粘度的 `tau = mu*(grad u + grad u^T)` 是
    逐单元 P0（线性场的导数），乘常数 `adj(J)` 仍是 P0，精确可微分、零
    混叠。而能量通量含 `u·tau` 与 `k_cond*grad_T`，其中 `u = rho_u/rho`、
    `T = p/(rho*R)` 都是 Q 的**有理**函数——真实非多项式，必然混叠。

    代价：**整步 +21.4%**（同一算例背靠背实测：`off` 119.7 ms/step、
    `on` 145.3 ms/step，nx=32、2304 单元、P1、预热 30 步后计时 80 步）。

    （**更正**：首次记录写的是"整步约 +5.3%"，那是把"`compute_viscous_
    residual` 单次调用的增量 6.4 ms"除以整步耗时得到的——漏了粘性残差
    每步被 RK 调用 **3 次**。正确的数是直接测整步得到的 +21.4%。）

    这个代价仍然值得付：`on@2x` 与 `on@3x` 逐位相同说明它是**收敛值**，
    而 `off` 的能量分量差 63%——那是错误而不是"另一种取舍"。

    ## 为什么"off 无所谓"这个前提已经不成立

    原文案写过（如实保留在下面）：legacy 模态滤波器下 `grad_vel` 是机器零
    （实测胞内 |grad u|/(U/h) = 7.7e-16），粘性体积项几乎只剩边界 IP 罚项。
    **2026-09-17 把 `AFCFD_FILTER_MODE` 默认从 `legacy` 改成 `sensor`
    之后**（未被传感器标记的单元等于不滤波），`grad_vel` 在绝大部分域里
    变成 O(0.08) 的真实量——这一项立刻活跃。所以把它打开是那次默认值改动
    的**一致性要求**，不是可选的附加项。

    `off` 保留为合法档，供回归对照与逐位复现历史结果。

    ## 以下是原有的动机记录（仍然有效）

    **为什么需要它（2026-09-15）**：去混叠机制
    （`fr/collapsed_basis.py::build_overintegration_operators` 有完整动机
    与实测数字——不去混叠的 P2 体积项对解析残差恒为 0 的线性剪切场算出
    的残差是真值的 43~62 倍）一直**只接在平均流的无粘体积项**上。粘性项
    完全没有（grep 确认本文件 overint/jacobians_fine 零命中）。

    而粘性通量 `G(Q, grad_vel, grad_T, mu_t)` 里有 `tau = mu_eff*(...)`
    与 `u·tau`、`k_cond*grad_T` 这些乘积，再乘 `adj(J)`，真实多项式次数
    远高于 order。

    它此前无所谓的原因**已经消失**：legacy 模态滤波器下 `grad_vel` 是
    机器零（实测胞内 |grad u|/(U/h)=7.7e-16），粘性体积项几乎只剩边界
    IP 罚项；一旦真正关掉滤波器（零阶数损失），`grad_vel` 变成 O(0.08)
    的真实量，这一项立刻活跃。

    与 k/omega 那边不同的是**这里没有"Gamma 自身混叠"那种局限**：本函数
    手上有 Q/grad_vel/grad_T/mu_t，可以像无粘路径那样在 FINE 点**重新
    求值非线性通量函数本身**，不是只能插值一个已经组装好的乘积。

    仍然存在的一处上游局限（如实写明）：`grad_vel`/`grad_T` 本身是
    `compute_physical_gradient` 在 coarse SPs 上算出来的，而那个算子
    **也没有**去混叠（`adj(J)*D*Q/det(J)` 同样是乘积）。这里把它们精确
    插值到 FINE 点，去掉的是**通量与度量项这一层**的混叠，梯度自身在
    coarse 上就带进来的混叠还在。补那一层是独立的一步。
    """
    v = os.environ.get("AFCFD_VISC_OVERINT", "on").lower()
    if v not in ("off", "on"):
        raise ValueError(
            f"AFCFD_VISC_OVERINT={v!r} 不是合法取值（off | on）。"
            f"'off' 是既有行为（体积项直接在 coarse SPs 上微分），"
            f"'on' 把粘性体积项改走 FINE 点去混叠。")
    return v


def _viscous_volume_overintegrated(Q, grad_vel, grad_T, mu_t_field,
                                   mu, Pr, Pr_t, oi, n_sps):
    """粘性体积项 `div(adj(J)*G(Q,grad_vel,grad_T,mu_t))` 的去混叠版，
    返回 (n_cells, n_sps, 5)。

    链路与 `fr_residual/inviscid.py` 的过积分分支逐项对应：
      ① Q/grad_vel/grad_T/mu_t 各自精确插值到 FINE 点（各自次数 <= order）；
      ② 在 FINE 点**重新求值** `viscous_physical_flux_batch`（非线性函数
         本身在细点求值，不是把 coarse 上的乘积插过去——这正是去混叠的
         全部内容）；
      ③ 用解析精确的 FINE 点度量算逆变通量；
      ④ 用 FINE 网格自己的微分矩阵求散度；
      ⑤ 精确插值限制回 coarse SPs。
    """
    n_cells = Q.shape[0]
    div_comp = np.zeros((n_cells, n_sps, 5))
    # 每段自带自己的 n_fine 与已切好的细点度量（2026-09-17）：native
    # 四面体的过积分细网格轴不再填充到棱柱的 (oo+1)^3 宽度，两段的 n_fine
    # 不同了。度量按**段内局部**索引切（`i0 = c0 - seg_lo`）——用全局 c0
    # 去切段内数组会静默取到错误的单元。
    for (seg_lo, seg_hi, n_fine, det_seg, inv_seg,
         op_c2f, op_D_fine, op_f2c) in oi["segs"]:
        for c0 in range(seg_lo, seg_hi, OVERINT_CHUNK_CELLS):
            c1 = min(c0 + OVERINT_CHUNK_CELLS, seg_hi)
            nb = c1 - c0
            i0, i1 = c0 - seg_lo, c1 - seg_lo
            Q_f = contract_shared_operator_1axis(
                op_c2f, np.ascontiguousarray(Q[c0:c1]))              # (nb,n_fine,5)
            gv_f = contract_shared_operator_1axis(
                op_c2f, np.ascontiguousarray(
                    grad_vel[c0:c1].reshape(nb, n_sps, 9))).reshape(nb, n_fine, 3, 3)
            gT_f = contract_shared_operator_1axis(
                op_c2f, np.ascontiguousarray(grad_T[c0:c1]))         # (nb,n_fine,3)
            mut_f = contract_shared_operator_1axis(
                op_c2f, np.ascontiguousarray(
                    mu_t_field[c0:c1][:, :, None]))[..., 0]          # (nb,n_fine)
            G_phys_f = viscous_physical_flux_batch(
                np.ascontiguousarray(Q_f.reshape(-1, 5)),
                np.ascontiguousarray(gv_f.reshape(-1, 3, 3)),
                np.ascontiguousarray(gT_f.reshape(-1, 3)),
                mu, Pr,
                np.ascontiguousarray(mut_f.reshape(-1)),
                Pr_t,
            ).reshape(nb, n_fine, 3, 5)
            del Q_f, gv_f, gT_f, mut_f
            G_tilde_f = contravariant_flux_from_metric(
                np.ascontiguousarray(det_seg[i0:i1]),
                np.ascontiguousarray(inv_seg[i0:i1]), G_phys_f)
            del G_phys_f
            div_f = contract_shared_operator_2axis(op_D_fine, G_tilde_f)
            del G_tilde_f
            div_comp[c0:c1] = contract_shared_operator_1axis(op_f2c, div_f)
            del div_f
    return div_comp
