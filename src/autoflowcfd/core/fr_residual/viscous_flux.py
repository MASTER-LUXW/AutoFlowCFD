"""
AutoFlowCFD V2.0 - FR 粘性物理通量与界面耦合 (Tier-0 重建版, 对应 S-03)

牛顿流体应力张量 + 傅里叶热传导的物理通量函数，以及基于真实单元-面连接
关系的界面耦合（BR1 格式：界面处的原始变量与梯度取相邻单元外插值的平均，
这是规范文档 3_系统实现方式-算法流程.md §2.3 明确允许的做法——
"在单元界面处，Θ̂ 取左右单元的平均值（或加权平均值）"）。

取代旧版本 fr_residual_viscous.py 中：
1. 从未被满足的 `hasattr(mesh,'face_connectivity')` 分支（死代码，从未执行）
2. 唯一实际执行的 fallback——用**单元内部梯度模长**冒充界面跳跃
   （`jump_estimate = h_local * |grad_u|`），这不是任何邻居信息，纯粹是
   同一个单元自己的局部量，物理上不构成"界面耦合"
3. 体积项用 D_3d 直接当物理导数使用（缺少度量项变换，见
   core/fr_gradients.py 文档），对本代码库的每个曲边/坍缩坐标单元都是
   错误导数

正确性通过「均匀常数流场（零梯度）粘性残差应严格为零」验证——牛顿粘性
应力和热传导对常数场恒为零，这是比自由流场保持性更基础但同样严格的
判据，见 tests/unit/test_fr_residual_viscous.py。

问题单元保护：`compute_physical_gradient` 用 `inv_jac`（近似正比于
adj(J)/det(J)）把参考空间导数转成物理梯度，坍缩坐标退化 SP 处 det(J)
极小，`inv_jac` 对应地极大——梯度本身在这类点先被放大一次，随后
`residual = div_comp/det(J)`（体积项）与 `correction/det(J)`（界面项）
在同一个退化 det(J) 上再放大一次，是比无粘残差更严重的*双重*放大（真实
Couette 合成算例复现：粘性残差 3 步内从 4e-2 量级放大到 1.16e7）。

此前曾仿照无粘那边"先用 det(J)/法向失配几何量预判、按整个单元降阶"的
机制1/2 实现过一版保护，但发现该判据有两个真实缺陷（见
fr_troubled_cell.py 模块文档"机制3"一节）：(1) 绝对 det(J) 阈值是照一个
特定网格的绝对尺度标定的，换个尺度就可能失效（真实复现：det(J) 比阈值
高 828 倍仍被放大到灾难量级）；(2) 按整个单元降阶，会在网格所有单元
恰好同一绝对尺度、以至于机制1对*每个*单元都命中时（合成验证网格常见），
把全网格的粘性物理都拍平成零梯度，等于关掉了粘性扩散本身。

此后改用过机制3（按 (cell,SP,变量) 粒度检测残差量级异常并清零），
**机制3 也已于 2026-09-19 删除** —— 真实网格消融对照证明它触发了但只把
残差轨迹改变 ~1e-10 相对量、且不改变发散这个结局，完整依据见
`fr_residual/inviscid.py` 里那段记录。

所以本函数现在**不对残差做任何异常抑制**：退化单元的对策是网格质量门
（项目记忆 `industry_practice_degenerate_cell_gcl`），不是运行期限制器。
"""

import os
import numpy as np

from autoflowcfd.core.fr_operators.gradients import compute_physical_gradient
from autoflowcfd.core.fr_operators.flux_kernels import viscous_physical_flux_batch
from autoflowcfd.core.fr_operators.volume_contract import (
    contract_shared_operator_1axis, contract_shared_operator_2axis,
    compute_adj_j, contravariant_flux_from_metric,
    OVERINT_CHUNK_CELLS, get_overintegration_context,
)

GAMMA = 1.4
R_AIR = 287.0  # 空气比气体常数 J/(kg*K)


def compute_temperature(Q: np.ndarray) -> np.ndarray:
    """T = p/(rho*R)。"""
    rho = np.maximum(Q[..., 0], 1e-10)
    return Q[..., 4] / (rho * R_AIR)


def viscous_physical_flux(
    Q: np.ndarray,
    grad_vel: np.ndarray,
    grad_T: np.ndarray,
    mu: float,
    Pr: float,
    mu_t=0.0,
    Pr_t: float = 0.9,
) -> np.ndarray:
    """计算粘性物理通量张量 G_i，与 euler_physical_flux 同样的 (...,3,5) 约定。

    Args:
        Q: (...,5) 原始变量 (rho,u,v,w,p)
        grad_vel: (...,3,3) 速度梯度，grad_vel[...,i,j] = d(u_i)/d(x_j)
        grad_T: (...,3) 温度梯度
        mu: 分子动力粘度（标量）
        Pr: 分子普朗特数
        mu_t: 湍流涡粘度（标量或可广播到 Q.shape[:-1] 的数组），默认0
            （层流/未提供湍流模型时）。应力张量按 Boussinesq 假设用
            mu_total=mu+mu_t 统一处理；热传导的湍流贡献用湍流普朗特数
            Pr_t（标准值0.9，非分子普朗特数）单独换算，两者不能共用同一
            个 Pr——这是本次修复把湍流涡粘度真正耦合进粘性应力张量
            （T-01/T-04/T-06）的核心：此前调用方从不传湍流粘度，
            粘性通量永远只用分子粘度。
        Pr_t: 湍流普朗特数

    Returns:
        G: (...,3,5)，G[...,i,:] 是方向 i 的粘性通量向量
           （质量分量恒为0；动量分量 G[...,i,1+j]=tau_ij；能量分量含粘性功+热传导）
    """
    mu_total = mu + mu_t
    mu_total = mu_total * np.ones(Q.shape[:-1]) if np.isscalar(mu_total) else mu_total

    S = 0.5 * (grad_vel + np.swapaxes(grad_vel, -1, -2))  # (...,3,3)
    div_u = grad_vel[..., 0, 0] + grad_vel[..., 1, 1] + grad_vel[..., 2, 2]
    lam = -2.0 / 3.0 * mu_total

    eye3 = np.eye(3)
    tau = 2.0 * mu_total[..., None, None] * S + lam[..., None, None] * div_u[..., None, None] * eye3  # (...,3,3)

    cp = GAMMA * R_AIR / (GAMMA - 1.0)
    k_cond = mu * cp / Pr + mu_t * cp / Pr_t
    q = -k_cond * grad_T if np.isscalar(k_cond) else -k_cond[..., None] * grad_T  # (...,3)

    vel = Q[..., 1:4]  # (...,3)
    work = np.einsum("...i,...ij->...j", vel, tau)  # work[...,j] = sum_i u_i*tau_ij

    shape = Q.shape[:-1]
    G = np.zeros(shape + (3, 5))
    G[..., :, 1:4] = np.swapaxes(tau, -1, -2)  # G[...,i,1+j] = tau[...,j,i] = tau[...,i,j] (对称)
    G[..., :, 4] = work + q
    return G


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


def compute_viscous_residual_fr(U: np.ndarray, mesh, ops, mu: float, Pr: float,
                                 mu_t_field=None, Pr_t: float = 0.9,
                                 boundary_ghost_provider=None,
                                 flat_face_override=None) -> np.ndarray:
    """计算真实面耦合的 FR 粘性残差 dU/dt（物理空间，已除以 det(J)）。

    Args:
        U: 守恒变量 (n_cells,n_sps,n_vars)，只使用前5个欧拉变量
        mesh: HighOrderMesh（需要 face_connectivity, face_flux_points, jacobians）
        ops: FROperators
        mu: 分子动力粘度（标量）
        Pr: 分子普朗特数
        mu_t_field: 湍流涡粘度场，形状 (n_cells, n_sps) 或 None（层流/未激活
            湍流模型时视为全零）。这是 T-01/T-04/T-06 湍流-平均流耦合的
            接入点——调用方（core/fr_solver.py）负责把 SST/DDES/WALE 算出
            的 nu_t 乘以密度后传入，见该模块修复说明。
        Pr_t: 湍流普朗特数
        boundary_ghost_provider: 边界面幽灵态提供者，与无粘残差
            （fr_residual_inviscid.py）共用同一个实例/同一套 WALL/INLET/
            OUTLET/FARFIELD/SYMMETRY 逻辑（见 boundary/fr_ghost_state.py），
            签名 (face_idx, Q_owner_fp, true_normal) -> Q_ghost_fp。
            None 时退化为 DefaultGhostProvider（零梯度外插，Q_ghost=Q_owner，
            BR1 跳跃恒为零）。

            此前这里完全不使用该参数、边界面统一取 Q_n=Q_o（内部值原样
            镜像），导致 BR1 平均 Q_avg 恒等于 Q_o、跳跃项 jump_owner 恒为
            零——即固壁上不存在任何由粘性方程施加的边界约束，无滑移
            剪应力不存在（真实数值验证：真实网格上 WALL 面的粘性校正项
            逐面精确为 0，与是否退化/网格质量无关，是恒等式）。现在真正
            调用 boundary_ghost_provider 取得反映边界条件的幽灵原始变量
            （例如 WALL 用速度镜像取反构造 Q_avg 速度=0，真正的无滑移）。

            **速度梯度 gv_n / mut_n 仍取内部值镜像**——标准 BR1/LDG 做法：
            它们没有独立的边界"真值"，边界约束通过状态跳跃在通量里体现
            （动量方向另有 IP 罚项，见 fr_operators/flux_kernels.py::
            viscous_boundary_penalty_tilde）。特别**不能**"顺手"把无滑移壁
            的 gv 也做法向镜像：壁面切向速度的法向导数**就是**壁面剪应力
            本身，镜像掉等于把它抹成零。

            **温度梯度 gT_n 已按热边界类型分派（2026-09-15 修复）**：
            此前 gT_n 也一律取内部值，对能量方程留下一处真实的不自洽——
            壁面 ghost 态复制 rho/p（温度无跳跃，见 fr_ghost_state.py::
            wall_ghost_state 的"热边界条件"一节，语义上是绝热壁），而
            IP 罚项只覆盖动量分量（`for v in range(1,4)`），于是面上平均
            法向温度梯度等于内部值、一般非零：实现出来的壁面热条件既不是
            绝热（q_w=0）也不是等温，而是"按内部梯度透射"。

            现在 WALL/SYMMETRY（见 fr_ghost_state.py::
            ADIABATIC_THERMAL_BC_TYPES）改为法向分量镜像
            `∇T_ghost = ∇T_int − 2(∇T_int·n)n`，于是 BR1 面平均
            `∇T_avg = ∇T_int − (∇T_int·n)n` 的法向分量**精确为零**，投影到
            该面的离散传导热通量 `a·q_avg` 恒等于零——是恒等式，不是
            "收敛到某个容差"。镜像用的 n 取自逆变行 `adj_row`（与面法向
            平行），这样"恒为零"的正是真正进入残差的那个投影量本身，而
            不是一个与之差一个截断误差的替代量。INLET/OUTLET/FARFIELD
            保持透射（流入/流出边界上法向热通量本就应该非零）。非
            `BoundaryGhostStateProvider` 的 provider（DefaultGhostProvider、
            测试 stub）没有 BC 语义，一律按透射处理，既有行为逐位不变。

            **修复前的实测量级，以及为什么那个数字不能再被引用**：
            79 万单元 cube_demo P1 iter=300 检查点上，温度场全域变化
            5.086 K 但**胞内** |∇T| 只有 mean=5.9e-12 / p99=6.4e-11 /
            max=3.7e-10 K/m，据此虚假壁面热通量密度
            |q_n| = k|∇T| ≈ 6.1e-14 W/m^2，相对来流焓通量密度
            rho*U*cp*T = 1.18e7 W/m^2 约 5e-21。**这个 5e-21 是被另一个
            缺陷压出来的**：legacy 模态滤波器每个 RK stage 清掉一整阶模态
            （见 fr/modal_filter.py 模块文档），P1 的解实际退化成分片常数，
            胞内梯度当然是机器零。滤波器一旦放开，实测胞内 |∇u|/(U/h) 从
            7.7e-16 跳到 O(1)，这一项的真实影响必须重新测量。
            （校验说明：上述 |∇T| 数字所用的 `compute_physical_gradient`
            调用方式已在已知线性场 T=x / T=3y 上验证到机器精度
            7.6e-16，不是算子用错导致的虚低。）

    Returns:
        residual: (n_cells, n_sps, 5)
    """
    from autoflowcfd.core.fr_residual.inviscid import conserved_to_primitive, DefaultGhostProvider

    ghost_provider = boundary_ghost_provider if boundary_ghost_provider is not None else DefaultGhostProvider()

    n_cells = mesh.n_cells
    n_sps = mesh.n_sps_per_cell
    n1d = mesh.n_points_1d

    mu_t_field = np.zeros((n_cells, n_sps)) if mu_t_field is None else mu_t_field

    Q = conserved_to_primitive(U[..., :5])  # (n_cells,n_sps,5)
    T = compute_temperature(Q)  # (n_cells,n_sps)

    grad_Q = compute_physical_gradient(Q, mesh, ops)  # (n_cells,n_sps,5,3)
    grad_vel = grad_Q[:, :, 1:4, :]  # (n_cells,n_sps,3,3): grad_vel[...,i,j]=d(u_i)/dx_j
    grad_T = compute_physical_gradient(T[:, :, None], mesh, ops)[:, :, 0, :]  # (n_cells,n_sps,3)

    det_jacs = mesh.jacobians["det_jacs"].reshape(n_cells, n_sps)
    inv_jacs = mesh.jacobians["inv_jacs"].reshape(n_cells, n_sps, 3, 3)
    adj_j = compute_adj_j(det_jacs, inv_jacs)

    # 体积项：整条 物理通量→逆变通量→散度 链按单元分块执行（B-12 P2 OOM
    # 修复第④级，2026-08-26）。历史背景：本函数体积项最初是纯 einsum 实现，
    # 后按与 fr_residual_inviscid.py 相同的性能优化路径改为 `viscous_physical_
    # flux_batch`（复用已逐位验证过的 numba 逐点 kernel）+ `np.matmul`/
    # `contract_shared_operator_2axis`（与原 einsum 公式严格等价，见
    # fr_volume_contract.py 模块文档）。retest5（cube 79万单元生产网格，
    # P0+P1 全部走完、P2 首次无粘残差求值也已通过——证明过积分链分块修复③
    # 生效）显示失败点前移到这里：SSP-RK3 第二阶段（L2）粘性残差求值时全场
    # 分配 G_phys（~2.6GiB = 791492 单元×27 SP×15 分量）MemoryError 崩溃。
    # 根因与内存账同 fr_residual/inviscid.py 过积分链分块注释（仪表化实测见
    # _tmp_review/diag_p2_13_prod.log：峰值 freeCommit 仅 0.52GB）：本段的两个
    # 全场数组 G_phys/G_tilde 各 ~2.6GiB，加上 grad_vel_flat 全场强制拷贝
    # ~1.4GiB，与第一阶段驻留量共存时超出剩余 commit 额度。这条链每一步都是
    # cell 局部的（`viscous_physical_flux_batch` 是 numba 逐点纯 gather、
    # `np.matmul`/`contract_shared_operator_2axis` 按 cell 批量，cell 之间零数据
    # 依赖），把 cell 轴切块、块内走完整条链，每步计算形状/求和顺序与全场版
    # 逐位一致，数值结果不变（等价性判据见
    # tests/unit/test_fr_viscous_flux_kernel_crosscheck.py）。块大小同样取 32768：
    # 块内瞬态 ≈0.3GiB（G_phys+G_tilde ~0.2GiB + grad_vel 块拷贝 ~0.07GiB），
    # 对比全场版同时驻留 ~6.6GiB。prism/tet 两段分开切块是因为两者用不同的
    # 坍缩坐标微分矩阵（见 FROperators.D_3d_tet/D_3d_prism 文档），且单元存储
    # 本来就是 prism 在前 tet 在后，块不会跨类型。附带收益：原全场 `np.
    # ascontiguousarray(grad_vel.reshape(-1,3,3))` 强制拷贝（~1.4GiB 真实新分配，
    # grad_vel 是 grad_Q 的非连续切片）降为每块一次小块拷贝；grad_T 同为梯度
    # 输出的非连续切片，按块拷贝量级相同（~21MiB/块）。Q/mu_t_field 是连续数组，
    # 块切片的 ascontiguousarray 是 no-op view。Q/grad_vel/grad_T/adj_j/det_jacs
    # 在下方界面项 kernel 里还要用，不删。
    n_prism = mesh.n_prism_cells
    _VISC_CHUNK_CELLS = 32768
    div_comp = np.zeros((n_cells, n_sps, 5))
    # native 四面体（路径C）：体积项散度同样必须改用 D_native_tet_padded，
    # 理由与 gradients.py::compute_physical_gradient 同一处文档——D_3d_tet
    # 是坍缩坐标专属微分矩阵，对 native 单纯形基节点没有意义。
    _tet_op_D = ops.D_native_tet_padded if getattr(ops, "D_native_tet_padded", None) is not None else ops.D_3d_tet
    # 去混叠（AFCFD_VISC_OVERINT=on）：粘性通量是 tau/u·tau/k_cond*grad_T
    # 这些乘积再乘 adj(J)，直接在 coarse SPs 上微分等价于"先混叠再求导"。
    # 理由、实测背景与保留的上游局限见 `resolve_viscous_overintegration`。
    # 默认 off，行为逐位不变。
    _oi = (get_overintegration_context(mesh, ops)
           if resolve_viscous_overintegration() == "on" else None)
    if _oi is not None:
        div_comp = _viscous_volume_overintegrated(
            Q, grad_vel, grad_T, mu_t_field, mu, Pr, Pr_t, _oi, n_sps)
        residual = div_comp / det_jacs[..., None]
        del div_comp
    else:
      for seg_lo, seg_hi, op_D in (
        (0, n_prism, ops.D_3d_prism),
        (n_prism, n_cells, _tet_op_D),
      ):
        for c0 in range(seg_lo, seg_hi, _VISC_CHUNK_CELLS):
            c1 = min(c0 + _VISC_CHUNK_CELLS, seg_hi)
            G_phys = viscous_physical_flux_batch(
                np.ascontiguousarray(Q[c0:c1].reshape(-1, 5)),
                np.ascontiguousarray(grad_vel[c0:c1].reshape(-1, 3, 3)),
                np.ascontiguousarray(grad_T[c0:c1].reshape(-1, 3)),
                mu, Pr,
                np.ascontiguousarray(mu_t_field[c0:c1].reshape(-1)),
                Pr_t,
            ).reshape(c1 - c0, n_sps, 3, 5)
            # 度量×通量融合 kernel（性能优化 2026-09-13，见 volume_contract.py
            # ::contravariant_flux_from_metric 文档：原 `np.matmul(adj_j, G_phys)`
            # 是逐点 3x3@3x5 批量微型 gemm，调用开销主导且不随核数并行）。
            # 数学上逐位等价（adj_j 本身就是 det_jacs*inv_jacs，这里直接从
            # 两者融合算出同一个乘积）；`adj_j` 仍保留物化，界面项 kernel 要用。
            G_tilde = contravariant_flux_from_metric(
                det_jacs[c0:c1], inv_jacs[c0:c1], G_phys
            )
            del G_phys  # 块内用完即弃，下一轮迭代变量重新绑定
            div_comp[c0:c1] = contract_shared_operator_2axis(op_D, G_tilde)
            del G_tilde
      # 注意：粘性项是 +div(G)（见模块文档的符号约定）
      residual = div_comp / det_jacs[..., None]
      del div_comp  # ~854MiB，用完即弃

    # --- 界面项：numba 逐点标量 kernel（性能优化，替代原纯 Python
    # `for f in range(fc.n_faces)` 逐面循环，理由/验证方式与
    # fr_residual_inviscid.py 的同类改动完全一致，见
    # fr_viscous_flux_kernel.py 模块文档、
    # tests/unit/test_fr_viscous_flux_kernel_crosscheck.py 的新旧实现
    # 逐位对比验证）。
    from autoflowcfd.core.fr_operators.face_kernels import get_flat_face_geometry
    from autoflowcfd.core.fr_residual.inviscid_kernel import compute_boundary_ghost_states

    # #2（2026-08-28）：见 core/fr_residual/inviscid.py::
    # compute_inviscid_residual_fr 同名参数文档——CPU 分布式路径必须传入
    # 预先按 local+halo 压缩索引空间构造好的 flat_face_override，不能让
    # 这里对 DistributedMeshAdapter 重新调用 get_flat_face_geometry。
    flat = flat_face_override if flat_face_override is not None else get_flat_face_geometry(mesh, ops)
    Q_ghost = compute_boundary_ghost_states(flat, Q, adj_j, ghost_provider)
    # 边界温度梯度按热边界类型分派（WALL/SYMMETRY 法向镜像 ⇒ 离散壁面热通量
    # 精确为零；INLET/OUTLET/FARFIELD 保持透射），见
    # boundary/fr_ghost_state.py::build_boundary_adiabatic_mask。
    from autoflowcfd.boundary.fr_ghost_state import build_boundary_adiabatic_mask
    bnd_adiabatic = build_boundary_adiabatic_mask(flat.n_faces, flat.is_boundary, ghost_provider)

    if n_sps == 1:
        # P0 专用路径：使用 n_sps=1 特化 kernel（消除 SP 循环，外插写成
        # 标量乘——n_sps=1 下与矩阵乘是同一个运算，不是近似，见
        # viscous_p0_kernel.py 模块文档的用词更正）
        # 性能优化：Order Continuation P0 阶段 n_sps=1，通用 kernel 的
        # for s in range(n_sps) 循环虽只有 1 次迭代但仍有分支/索引开销，
        # P0 专用 kernel 在编译期消除所有 SP 循环。
        from autoflowcfd.core.fr_residual.viscous_p0_kernel import (
            compute_viscous_interface_correction_p0_kernel,
        )
        import numba
        n_threads = numba.get_num_threads()
        correction = compute_viscous_interface_correction_p0_kernel(
            Q, grad_vel, grad_T, mu_t_field,
            adj_j, det_jacs, mu, Pr, Pr_t,
            flat.owner_cell, flat.neighbor_cell, flat.is_boundary,
            flat.owner_axis, flat.owner_side, flat.neighbor_axis, flat.neighbor_side,
            flat.owner_is_primary, flat.neighbor_is_primary,
            flat.neighbor_src0_cell, flat.neighbor_src0_mat,
            flat.neighbor_src1_idx, flat.neighbor_src1_cell, flat.neighbor_src1_mat,
            flat.owner_src0_cell, flat.owner_src0_mat,
            flat.owner_src1_idx, flat.owner_src1_cell, flat.owner_src1_mat,
            flat.mixed_nb_partner, flat.mixed_nb_mask,
            flat.mixed_ow_partner, flat.mixed_ow_mask,
            flat.boundary_extrap, flat.g_left, flat.g_right, Q_ghost, bnd_adiabatic,
            flat.dist_fp_of_sp, flat.dist_axis_coord_of_sp,
            n_prism, n_threads,
            flat.owner_cube_face, flat.neighbor_cube_face,
            flat.ref_area_weight,
            flat.boundary_extrap_native, flat.lift_native,
        )
    else:
        # P≥1 通用路径：图着色或 per-thread buffer
        from autoflowcfd.core.fr_residual.viscous_flux_kernel import (
            compute_viscous_interface_correction_kernel,
        )
        from autoflowcfd.core.fr_residual.viscous_flux_kernel_colored import (
            compute_viscous_interface_correction_kernel_colored,
        )
        # 图着色方案：同色面无 owner_cell 冲突，直接写入共享 buffer
        # 内存从 O(n_threads * n_cells * n_sps * 5) 降至 O(n_cells * n_sps * 5)
        # 着色结果已缓存在 flat 中（build 时一次性计算，不再重复着色）
        # 通过环境变量或配置可切换回 per-thread buffer 方案
        use_coloring = os.environ.get("AFCFD_USE_COLORING", "1") == "1"
        
        if use_coloring:
            correction = np.zeros((n_cells, n_sps, 5))
            for c in range(flat.n_colors):
                face_indices = flat.color_face_indices[c]
                if len(face_indices) == 0:
                    continue
                compute_viscous_interface_correction_kernel_colored(
                    Q, grad_vel, grad_T, mu_t_field,
                    det_jacs, mu, Pr, Pr_t,
                    flat.owner_cell, flat.neighbor_cell, flat.is_boundary,
                    flat.owner_axis, flat.owner_side, flat.neighbor_axis, flat.neighbor_side,
                    flat.owner_is_primary, flat.neighbor_is_primary,
                    flat.owner_adj_row_exact, flat.neighbor_adj_row_exact,
                    flat.neighbor_src0_cell, flat.neighbor_src0_mat,
                    flat.neighbor_src1_idx, flat.neighbor_src1_cell, flat.neighbor_src1_mat,
                    flat.owner_src0_cell, flat.owner_src0_mat,
                    flat.owner_src1_idx, flat.owner_src1_cell, flat.owner_src1_mat,
                    flat.mixed_nb_partner, flat.mixed_nb_mask,
                    flat.mixed_ow_partner, flat.mixed_ow_mask,
                    flat.boundary_extrap, flat.g_left, flat.g_right, Q_ghost, bnd_adiabatic,
                    flat.dist_fp_of_sp, flat.dist_axis_coord_of_sp,
                    n_prism, face_indices, correction,
                    flat.owner_cube_face, flat.neighbor_cube_face,
                    flat.ref_area_weight,
                    flat.boundary_extrap_native, flat.lift_native,
                )
        else:
            # 回退到 per-thread buffer 方案（小网格 + 低线程数可能更快）
            import numba
            n_threads = numba.get_num_threads()
            correction = compute_viscous_interface_correction_kernel(
                Q, grad_vel, grad_T, mu_t_field,
                det_jacs, mu, Pr, Pr_t,
                flat.owner_cell, flat.neighbor_cell, flat.is_boundary,
                flat.owner_axis, flat.owner_side, flat.neighbor_axis, flat.neighbor_side,
                flat.owner_is_primary, flat.neighbor_is_primary,
                flat.owner_adj_row_exact, flat.neighbor_adj_row_exact,
                flat.neighbor_src0_cell, flat.neighbor_src0_mat,
                flat.neighbor_src1_idx, flat.neighbor_src1_cell, flat.neighbor_src1_mat,
                flat.owner_src0_cell, flat.owner_src0_mat,
                flat.owner_src1_idx, flat.owner_src1_cell, flat.owner_src1_mat,
                flat.mixed_nb_partner, flat.mixed_nb_mask,
                flat.mixed_ow_partner, flat.mixed_ow_mask,
                flat.boundary_extrap, flat.g_left, flat.g_right, Q_ghost, bnd_adiabatic,
                flat.dist_fp_of_sp, flat.dist_axis_coord_of_sp,
                n_prism, n_threads,
                flat.owner_cube_face, flat.neighbor_cube_face,
                flat.ref_area_weight,
                flat.boundary_extrap_native, flat.lift_native,
            )
    residual = residual + correction

    # 机制3 已删除，依据见 `fr_residual/inviscid.py` 同一处的完整记录
    # （真实网格消融对照：触发 15 次但残差轨迹只差 ~1e-10、结局不变）。
    return residual
