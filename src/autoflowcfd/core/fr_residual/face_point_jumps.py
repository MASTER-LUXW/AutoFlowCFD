"""AutoFlowCFD V2.0 - 界面项的逐通量点跳变量（残差核与解析 Jacobian 共用）。

FR/DG 界面项在每个通量点上算一个"公共通量减本侧通量"的跳变量 `J_i`，再经
提升算子 `lift_native` 回到解点。`J_i` 只依赖该点两侧的迹（与本侧的逆变
度量行），与单元、面、线程划分无关——这里把它写成**唯一一份**逐点函数：

* 残差核（`inviscid_kernel*.py`、`viscous_flux_kernel*.py`，着色与逐线程
  缓冲两种调度）逐点调用它求值；
* 解析单元块 Jacobian（`fr_residual/jacobian/`）在同一个函数上做逐点差分，
  得到 `dJ_i/d(迹)`。

两处用的是同一段机器码，Jacobian 不可能与残差悄悄脱节（本项目多次出现过
"两份实现只改了一份"的缺陷）。边界面上"另一侧"的梯度如何由本侧构造、以及
各粘性边界种类的公共通量，见 `flux_kernels/viscous_bc.py`（同样一份定义）。

各式的物理依据见被调用的通量函数文档（`kernels.compute_ausm_up_flux`、
`flux_kernels.viscous_physical_flux_point`、`viscous_ip_penalty_tilde`、
`flux_kernels/viscous_bc.py`）；本模块只负责把它们组合成跳变量。
"""

import numpy as np
from numba import njit

from autoflowcfd.core.fr_operators.kernels import compute_ausm_up_flux
from autoflowcfd.core.fr_operators.flux_kernels import (
    CP_AIR, VBC_DIRICHLET, VBC_INTERIOR, VBC_NEUMANN, resolve_point_kind,
    euler_physical_flux_point, viscous_physical_flux_point, viscous_ip_penalty_tilde,
)


@njit(cache=True, inline='always')
def inviscid_jump_point(Q_s, Q_x, adjrow, mach_ref, precond_mode):
    """无粘跳变量 `|a| F_AUSM(Q_s, Q_x, a/|a|) - a . F(Q_s)`，形状 (5,)。

    `Q_s`/`Q_x`：本侧/另一侧原始变量 (rho,u,v,w,p)；`adjrow`：本侧逐 FP 精确
    逆变度量行（已 outward 定向，法向一律取自本侧，理由见
    `inviscid_kernel.compute_inviscid_interface_correction_kernel` 文档）。
    """
    a0 = adjrow[0]
    a1 = adjrow[1]
    a2 = adjrow[2]
    adj_mag = np.sqrt(a0 * a0 + a1 * a1 + a2 * a2)
    if adj_mag < 1e-300:
        adj_mag = 1e-300
    normal = np.empty(3)
    normal[0] = a0 / adj_mag
    normal[1] = a1 / adj_mag
    normal[2] = a2 / adj_mag
    F_common = compute_ausm_up_flux(Q_s, Q_x, normal, mach_ref, precond_mode)
    F_phys = euler_physical_flux_point(Q_s)
    jump = np.empty(5)
    for v in range(5):
        jump[v] = F_common[v] * adj_mag - (a0 * F_phys[0, v] + a1 * F_phys[1, v] + a2 * F_phys[2, v])
    return jump


@njit(cache=True, inline='always')
def viscous_jump_point(Q_s, gv_s, gT_s, mut_s, Q_x, gv_x, gT_x, mut_x, adjrow, h_ip,
                       bkind, mu, Pr, Pr_t, c_ip):
    """粘性跳变量（面平均通量减本侧通量，加 IP 罚项），形状 (5,)。

    `bkind`：该面（混合拆分面的边界半区取配对面）的粘性边界种类
    （`flux_kernels/viscous_bc.py`），内部面传 `VBC_INTERIOR`；入口在这里按本点
    状态落到 Dirichlet / Neumann（`resolve_point_kind`）。边界面上 `Q_x` 是幽灵态、
    `(gv_x, gT_x)` 由 `boundary_other_gradients` 按同一个种类构造。

    * `VBC_NEUMANN`：公共法向粘性通量为零（无牵引、绝热），跳变量就是
      `-a·G(本侧)`，不加罚项；
    * 其余：`a·G(平均态) - a·G(本侧)` 加罚项。内部面罚项用面平均涡粘与热传导率；
      边界面用本侧涡粘，热传导率只在 `VBC_DIRICHLET` 时给（温度被弱施加到
      幽灵态温度），无滑移壁/镜像类传 0（绝热由 ∇T 镜像精确施加）。
    """
    bkind = resolve_point_kind(bkind, Q_s, adjrow)
    a0 = adjrow[0]
    a1 = adjrow[1]
    a2 = adjrow[2]
    jump = np.empty(5)
    G_self = viscous_physical_flux_point(Q_s, gv_s, gT_s, mu, Pr, mut_s, Pr_t)
    if bkind == VBC_NEUMANN:
        for v in range(5):
            jump[v] = -(a0 * G_self[0, v] + a1 * G_self[1, v] + a2 * G_self[2, v])
        return jump

    Q_avg = np.empty(5)
    for v in range(5):
        Q_avg[v] = 0.5 * (Q_s[v] + Q_x[v])
    gv_avg = np.empty((3, 3))
    for a in range(3):
        for b in range(3):
            gv_avg[a, b] = 0.5 * (gv_s[a, b] + gv_x[a, b])
    gT_avg = np.empty(3)
    for a in range(3):
        gT_avg[a] = 0.5 * (gT_s[a] + gT_x[a])
    mut_avg = 0.5 * (mut_s + mut_x)

    G_common = viscous_physical_flux_point(Q_avg, gv_avg, gT_avg, mu, Pr, mut_avg, Pr_t)
    for v in range(5):
        jump[v] = ((a0 * G_common[0, v] + a1 * G_common[1, v] + a2 * G_common[2, v])
                   - (a0 * G_self[0, v] + a1 * G_self[1, v] + a2 * G_self[2, v]))

    adj_mag = np.sqrt(a0 * a0 + a1 * a1 + a2 * a2)
    if bkind == VBC_INTERIOR:
        k_tot = mu * CP_AIR / Pr + mut_avg * CP_AIR / Pr_t
        pen = viscous_ip_penalty_tilde(Q_s, Q_x, mu + mut_avg, k_tot, h_ip, adj_mag, 1.0, c_ip)
    elif bkind == VBC_DIRICHLET:
        k_tot = mu * CP_AIR / Pr + mut_s * CP_AIR / Pr_t
        pen = viscous_ip_penalty_tilde(Q_s, Q_x, mu + mut_s, k_tot, h_ip, adj_mag, 1.0, c_ip)
    else:
        pen = viscous_ip_penalty_tilde(Q_s, Q_x, mu + mut_s, 0.0, h_ip, adj_mag, 1.0, c_ip)
    for v in range(1, 5):
        jump[v] += pen[v]
    return jump
