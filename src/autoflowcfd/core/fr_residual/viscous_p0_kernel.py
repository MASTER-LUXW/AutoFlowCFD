"""
AutoFlowCFD V2.0 - P0 专用粘性界面校正 numba kernel

从 viscous_flux_kernel.py 拆出。当 n_sps=1（P0 order continuation 阶段）
时使用本 kernel 替代通用 kernel，消除所有 SP 循环（编译期常量 n_sps=1），
把外插矩阵乘写成标量乘。

**用词更正（2026-09-14）**：本文件多处把这种写法称为「简化」，那个说法
不准确——n_sps=1 时 `E (n_fp,1) @ field (1,k)` 与 `E[i,0]*field[0,...]`
是**同一个矩阵乘**（求和只有一项），不是任何形式的近似。本 kernel 相对
通用 kernel 是 n_sps=1 的**特化**（specialization）：去掉编译期已知为 1
的循环、把单项求和写成乘法，数学上逐项等价，只有浮点运算顺序不同。
现已把各处「简化」改为「特化」/「n_sps=1 恒等式」这类准确表述。

性能收益（791K 单元 / 188 万面网格，P0 阶段）：
- 消除 for s in range(n_sps) 循环（n_sps=1 时仍有一次迭代开销）
- 外插 E@field 写成 E[i,0]*field[0]（n_sps=1 下的同一个乘法，避免矩阵乘调用开销）
- 输出 correction 形状 (n_cells, 1, 5) 而非 (n_cells, n_sps, 5)

算法与通用 viscous_flux_kernel.py 完全一致（n_sps=1 的特化），
数学等价，仅浮点重排顺序不同。

确认不受棱柱四边形侧面重复计数问题影响（2026-08-23 核实，交叉引用
fr_residual/inviscid_p0.py::_extract_p0_face_geometry 文档记录的那次
回归/修复）：本 kernel 的两处 scatter-add 累加块本来就用
`owner_is_primary[f]`/`neighbor_is_primary[f]` 门控（见下方
`compute_viscous_interface_correction_p0_kernel` 的 owner/neighbor
两段累加代码），对棱柱四边形侧面被三角化拆分出的非 primary 重复记录
天然贡献为零；multi-source（~5% 拆分面两条记录指向 2 个不同真实
相邻单元）情形也已经通过 `owner_src0_cell`/`owner_src1_idx` 等插值
权重机制在 primary 记录内部正确混合，不需要像 inviscid_p0.py 那样
额外回退到三角化半面几何。P0 无粘 kernel 修复前唯一遗漏的正是这个
primary 门控，本 kernel 从一开始就没有这个问题，不需要改动。
"""

import numpy as np
from numba import njit, prange, get_thread_id

from autoflowcfd.core.fr_operators.small_dense import matmul_small
from autoflowcfd.core.fr_operators.flux_kernels import (
    CP_AIR, viscous_physical_flux_point,
    viscous_ip_penalty_tilde, mirror_normal_component,
)



@njit(cache=True, parallel=True)
def compute_viscous_interface_correction_p0_kernel(
    Q: np.ndarray,             # (n_cells, 1, 5)
    grad_vel: np.ndarray,      # (n_cells, 1, 3, 3)
    grad_T: np.ndarray,        # (n_cells, 1, 3)
    mu_t_field: np.ndarray,    # (n_cells, 1)
    det_jacs: np.ndarray,      # (n_cells, 1)
    mu: float, Pr: float, Pr_t: float,
    owner_cell: np.ndarray, neighbor_cell: np.ndarray, is_boundary: np.ndarray,
    owner_is_primary: np.ndarray, neighbor_is_primary: np.ndarray,
    # 逐 FP 精确度量伴随行（与 P>=1 kernel 同一个来源，见函数文档
    # "2026-09-23 修复的真实缺陷"）。
    owner_adj_row_exact: np.ndarray, neighbor_adj_row_exact: np.ndarray,
    neighbor_src0_cell: np.ndarray, neighbor_src0_mat: np.ndarray,
    neighbor_src1_idx: np.ndarray, neighbor_src1_cell: np.ndarray, neighbor_src1_mat: np.ndarray,
    owner_src0_cell: np.ndarray, owner_src0_mat: np.ndarray,
    owner_src1_idx: np.ndarray, owner_src1_cell: np.ndarray, owner_src1_mat: np.ndarray,
    mixed_nb_partner: np.ndarray, mixed_nb_mask: np.ndarray,
    mixed_ow_partner: np.ndarray, mixed_ow_mask: np.ndarray,
    Q_ghost: np.ndarray,          # (n_boundary_faces, n_fp, 5)
    bnd_adiabatic: np.ndarray,    # (n_faces,) 该边界面要求法向 dT/dn=0
    n_threads: int,
    owner_cube_face: np.ndarray, neighbor_cube_face: np.ndarray,
    ref_area_weight: np.ndarray,
    boundary_extrap_native: np.ndarray, lift_native: np.ndarray,
    # IP 罚项的长度尺度 `h_f = cell_volume[cell] / face_area[face]`
    # （面法向的单元厚度）。见 `flux_kernels.viscous_ip_penalty_tilde`
    # 的"长度尺度"一节：此前用 `mean(det_jacs)**(1/3)`，两处都错 ——
    # `mean(det_jacs)` 不是体积而是体积/参考体积（参考体积基相关：
    # 坍缩 8 / 原生棱柱 4 / 原生四面体 4/3），且几何平均在各向异性
    # 贴壁单元上比壁法向厚度大 2.48 倍（实测平板算例）。
    face_area: np.ndarray, cell_volume: np.ndarray,
    # IP 罚项常数，按阶数解析（见 `flux_kernels.resolve_viscous_ip_constant`）。
    # 做成形参而不是模块全局：njit 把全局当编译期常量、且
    # `cache=True` 的磁盘缓存**不因全局值变化而失效**，那样
    # 扫参数必须隔离缓存目录重编译，极易误读成"改了没生效"。
    c_ip: float,
) -> np.ndarray:
    """P0 专用粘性界面校正 kernel。

    与通用 compute_viscous_interface_correction_kernel 数学等价，
    但 n_sps=1 时：
    - 外插写成标量乘：E[i,0]*field[0,...]（n_sps=1 恒等式，非近似）
    - 分布特化为单 SP：out[0,v] = g * fp_data[fp_i, v]
    - correction 形状 (n_cells, 1, 5)

    原生基（四面体 + 棱柱）：与通用 kernel（viscous_flux_kernel.py）
    同一套做法 —— `owner_cube_face`/`neighbor_cube_face` 减 6 索引原生
    算子表（四面体 [6,10)、棱柱 [10,15) 在同一张表里），自身面外插用
    `boundary_extrap_native[code-6]`，面修正项用
    `lift_native[code-6] @ (ref_area_weight⊙jump)`。order=0 时原生真实
    自由度数恰好也是 1（受限 PKD 模态数 `(0+1)(0+2)(0+3)/6 = 1`），
    `lift_native[code-6]` 形状 `(1, n_fp)`，矩阵乘本身就是这里 P0 特化
    要的标量化形式。

    **2026-09-23 修复的真实缺陷（度量伴随行用错了量）**：这里此前用
    `adj_o_s0 = adj_j[oc, 0, oax]`，也就是**坍缩坐标那条"取 SP 网格度量
    的第 oax 个轴行"**的做法。两个独立后果：

    1. `owner_axis` 对原生面是**复用槽位**（原生四面体存 excluded_vertex
       ∈ {0,1,2,3}，原生棱柱存面序号 ∈ {0..4}），而 `adj_j` 的轴维只有
       3 —— `excluded_vertex == 3` 的四面体面直接**越界读**（numba
       nopython 不做边界检查，读到的是相邻单元的内存）。
    2. 即使没越界，那个量本身也是错的：合成混合网格实测，`adj_j[oc,0,
       oax]` 与精确行 `owner_adj_row_exact[f,i]` 的相对差在**每一个**面
       上都是 O(1)（0.5 ~ 3.0，好几个面方向完全不同）。

    P>=1 的通用 kernel 早在 2026-08-23 就改用了逐 FP 精确
    `owner_adj_row_exact`/`neighbor_adj_row_exact`（见
    `fr/face_flux_points/exact_normal.py` 模块文档），**P0 这份拷贝被
    漏掉了** —— 又一次"同一语义两份实现、只改了一份"。现已改成同一个
    来源并逐 FP 读取，`adj_j`/`owner_axis`/`neighbor_axis` 三个参数因此
    从签名里移除。

    同批（2026-09-23）删除的还有坍缩坐标那条并行路径
    （`boundary_extrap[celltype,axis,side]` 外插 + 1D Radau/VCJH 分布），
    生产不可达，完整论证见 `inviscid_kernel.py::
    compute_inviscid_interface_correction_kernel` 文档。
    """
    n_cells = Q.shape[0]
    n_faces = owner_cell.shape[0]
    n_fp = Q_ghost.shape[1]

    correction_per_thread = np.zeros((n_threads, n_cells, 1, 5))

    for f in prange(n_faces):
        tid = get_thread_id()
        oc = owner_cell[f]
        oc_code = owner_cube_face[f]
        # **原生面的罚项 side 因子必须是 +1**（2026-09-22 修复真实缺陷）：
        # 原生面的 `owner_adj_row_exact` 已按 outward 定向（见
        # `native_prism/face.py::native_prism_face_adj_rows` 与
        # `exact_normal.py::compute_exact_face_normals_and_weights` 里
        # `side_factor = np.where(owner_code >= 6, 1.0, owner_side)` 的同一
        # 处理），而坍缩面的 adj_row 未定向、外向性由 `oside` 给出。此前
        # 这里直接乘 `oside`，于是 `owner_side = -1` 的原生面（棱柱 f0/f3/f4、
        # 四面体全部 4 个面）罚项符号反了 —— 从耗散变成往壁面单元注入动量。

        if owner_is_primary[f]:
            E_o = boundary_extrap_native[oc_code - 6]  # (n_fp, 1)

            # P0 外插：n_sps=1 时 E (n_fp,1) @ field (1,k) 与
            # E[i,0]*field[0,...] 是**同一个矩阵乘**，不是近似——原注释
            # 写成「P0 简化外插」不准确（2026-09-14 更正）。
            # 而且 P0 的唯一基函数是常数 1，正确的插值算子在这里恒有
            # E[i,0]=1（已实测核实：order=0 下 boundary_extrap_prism 与
            # boundary_extrap_native_tet 全为 1.0），所以外插结果就等于
            # 单元自身的值——这正是 P0 常数重构应有的行为。
            Q_o_s0 = Q[oc, 0]  # (5,)
            gv_o_s0 = grad_vel[oc, 0]  # (3,3)
            gT_o_s0 = grad_T[oc, 0]  # (3,)
            mut_o_s0 = mu_t_field[oc, 0]  # scalar
            adjrow_o = owner_adj_row_exact[f]  # (n_fp,3)，逐 FP 精确值

            jump_owner = np.zeros((n_fp, 5))
            for i in range(n_fp):
                # 混合拆分面（B-8，与通用 kernel 同步，见 viscous_flux_kernel.py 同名注释）。
                mp = mixed_nb_partner[f]
                is_bnd_i = is_boundary[f] or (mp >= 0 and mixed_nb_mask[f, i])
                e_i = E_o[i, 0]  # 标量
                adj_o_i = adjrow_o[i]  # (3,)，本 FP 的精确度量伴随行

                # 外插到 FP i：n_sps=1 下标量乘 == 矩阵乘（见上方说明，
                # 不是简化）
                Q_o_i = np.empty(5)
                for v in range(5):
                    Q_o_i[v] = e_i * Q_o_s0[v]
                gv_o_i = np.empty((3, 3))
                for a in range(3):
                    for b in range(3):
                        gv_o_i[a, b] = e_i * gv_o_s0[a, b]
                gT_o_i = np.empty(3)
                for a in range(3):
                    gT_o_i[a] = e_i * gT_o_s0[a]
                mut_o_i = e_i * mut_o_s0

                # 邻居状态（源矩阵插值；n_sps=1 时同样是标量乘 == 矩阵乘
                # 的恒等式，不是简化）
                if is_boundary[f]:
                    # 边界温度梯度按热边界类型分派，见 viscous_flux_kernel.py
                    # 模块文档"边界温度梯度"一节（三处 kernel 必须同步）。
                    Q_n = Q_ghost[f, i]
                    gv_n = gv_o_i.copy()
                    if bnd_adiabatic[f]:
                        gT_n = mirror_normal_component(gT_o_i, adj_o_i)
                    else:
                        gT_n = gT_o_i.copy()
                    mut_n = mut_o_i
                else:
                    Q_n = np.zeros(5)
                    gv_n = np.zeros((3, 3))
                    gT_n = np.zeros(3)
                    mut_n = 0.0
                    c0 = neighbor_src0_cell[f]
                    if c0 >= 0:
                        mat0 = neighbor_src0_mat[f]
                        w = mat0[i, 0]  # n_sps=1，只有 s=0
                        if w != 0.0:
                            for v in range(5):
                                Q_n[v] += w * Q[c0, 0, v]
                            for a in range(3):
                                for b in range(3):
                                    gv_n[a, b] += w * grad_vel[c0, 0, a, b]
                                gT_n[a] += w * grad_T[c0, 0, a]
                            mut_n += w * mu_t_field[c0, 0]
                    idx1 = neighbor_src1_idx[f]
                    if idx1 >= 0:
                        c1 = neighbor_src1_cell[idx1]
                        mat1 = neighbor_src1_mat[idx1]
                        w = mat1[i, 0]
                        if w != 0.0:
                            for v in range(5):
                                Q_n[v] += w * Q[c1, 0, v]
                            for a in range(3):
                                for b in range(3):
                                    gv_n[a, b] += w * grad_vel[c1, 0, a, b]
                                gT_n[a] += w * grad_T[c1, 0, a]
                            mut_n += w * mu_t_field[c1, 0]
                    # 混合拆分面边界半区（B-8）：状态取配对面幽灵态，梯度镜像内部值。
                    if mp >= 0 and mixed_nb_mask[f, i]:
                        for v in range(5):
                            Q_n[v] = Q_ghost[mp, i, v]
                        if bnd_adiabatic[mp]:
                            gT_bnd = mirror_normal_component(gT_o_i, adj_o_i)
                        else:
                            gT_bnd = gT_o_i
                        for a in range(3):
                            for b in range(3):
                                gv_n[a, b] = gv_o_i[a, b]
                            gT_n[a] = gT_bnd[a]
                        mut_n = mut_o_i

                # 算术平均
                Q_avg = np.empty(5)
                for v in range(5):
                    Q_avg[v] = 0.5 * (Q_o_i[v] + Q_n[v])
                gv_avg = np.empty((3, 3))
                for a in range(3):
                    for b in range(3):
                        gv_avg[a, b] = 0.5 * (gv_o_i[a, b] + gv_n[a, b])
                gT_avg = np.empty(3)
                for a in range(3):
                    gT_avg[a] = 0.5 * (gT_o_i[a] + gT_n[a])
                mut_avg = 0.5 * (mut_o_i + mut_n)

                # 粘性通量
                G_common = viscous_physical_flux_point(Q_avg, gv_avg, gT_avg, mu, Pr, mut_avg, Pr_t)
                a0 = adj_o_i[0]
                a1 = adj_o_i[1]
                a2 = adj_o_i[2]
                G_tilde_common = np.empty(5)
                for v in range(5):
                    G_tilde_common[v] = a0 * G_common[0, v] + a1 * G_common[1, v] + a2 * G_common[2, v]

                G_phys_o = viscous_physical_flux_point(Q_o_i, gv_o_i, gT_o_i, mu, Pr, mut_o_i, Pr_t)
                G_tilde_own = np.empty(5)
                for v in range(5):
                    G_tilde_own[v] = a0 * G_phys_o[0, v] + a1 * G_phys_o[1, v] + a2 * G_phys_o[2, v]

                for v in range(5):
                    jump_owner[i, v] = G_tilde_common[v] - G_tilde_own[v]

                # `adj_mag_o` 与罚项长度尺度两条分支都要用，提到 if 之外。
                adj_mag_o = np.sqrt(a0 * a0 + a1 * a1 + a2 * a2)
                h_ip_o = cell_volume[oc] / face_area[f]
                if is_bnd_i:
                    # 边界 IP 罚项，见 viscous_flux_kernel.py::
                    # compute_viscous_interface_correction_kernel 同名分支
                    # 文档。（原注释写"P0 特化：vol_o 直接是 det_jacs[oc,0]"，
                    # 2026-09-23 起罚项长度尺度改用 `cell_volume/face_area`，
                    # 与阶数无关，那条特化说明已不适用。）
                    pen = viscous_ip_penalty_tilde(
                        Q_o_i, Q_n, mu + mut_o_i, 0.0, h_ip_o, adj_mag_o,
                        1.0, c_ip, False,
                    )
                else:
                    # **内部面 IP 罚项**（2026-09-23 修复真实缺陷，完整依据见
                    # `viscous_ip_penalty_tilde` 的"为什么内部面也必须加"）：
                    # `sigma = grad(u)` 是纯单元内局部梯度
                    # （`compute_physical_gradient(field, mesh, ops)` 的签名里
                    # 没有任何面数据），没有 BR1 要求的提升项；界面耦合只有
                    # "粘性通量取两侧算术平均"这一层，内部面此前**零罚项** ——
                    # 正是 ABCM(2002) 框架里"提升项与罚项都为零"的那一档，
                    # 不满足强制性（实测均匀基态纯粘性算子谱正实部 328/2160）。
                    # 涡粘与热传导率都取**面平均**，与 `G_common` 一致。
                    k_tot_o = mu * CP_AIR / Pr + mut_avg * CP_AIR / Pr_t
                    pen = viscous_ip_penalty_tilde(
                        Q_o_i, Q_n, mu + mut_avg, k_tot_o, h_ip_o, adj_mag_o,
                        1.0, c_ip, True,
                    )
                for v in range(1, 5):
                    jump_owner[i, v] += pen[v]

            dj = det_jacs[oc, 0]
            # DG 提升算子：`lift_native[code-6]` 形状 (1,n_fp)，@ 之后直接
            # 得到 (1,5)——本身已经是 P0 需要的标量化形式。
            weighted_jump_o = np.empty((n_fp, 5))
            for i in range(n_fp):
                w_area = ref_area_weight[i]
                for v in range(5):
                    weighted_jump_o[i, v] = w_area * jump_owner[i, v]
            contrib_owner = matmul_small(lift_native[oc_code - 6], weighted_jump_o)  # (1,5)
            for v in range(5):
                correction_per_thread[tid, oc, 0, v] += contrib_owner[0, v] / dj

        # Neighbor 侧（与通用 kernel 相同逻辑，n_sps=1 特化）
        if (not is_boundary[f]) and neighbor_is_primary[f]:
            nc = neighbor_cell[f]
            nc_code = neighbor_cube_face[f]
            E_n = boundary_extrap_native[nc_code - 6]

            Q_n_s0 = Q[nc, 0]
            gv_n_s0 = grad_vel[nc, 0]
            gT_n_s0 = grad_T[nc, 0]
            mut_n_s0 = mu_t_field[nc, 0]
            adjrow_n = neighbor_adj_row_exact[f]  # (n_fp,3)，逐 FP 精确值

            jump_neighbor = np.zeros((n_fp, 5))
            for i in range(n_fp):
                e_i = E_n[i, 0]
                adj_n_i = adjrow_n[i]  # (3,)，本 FP 的精确度量伴随行

                Q_n_i = np.empty(5)
                for v in range(5):
                    Q_n_i[v] = e_i * Q_n_s0[v]
                gv_n_i = np.empty((3, 3))
                for a in range(3):
                    for b in range(3):
                        gv_n_i[a, b] = e_i * gv_n_s0[a, b]
                gT_n_i = np.empty(3)
                for a in range(3):
                    gT_n_i[a] = e_i * gT_n_s0[a]
                mut_n_i = e_i * mut_n_s0

                # Owner 侧插值（n_sps=1 特化）
                Q_o_at_n = np.zeros(5)
                gv_o_at_n = np.zeros((3, 3))
                gT_o_at_n = np.zeros(3)
                mut_o_at_n = 0.0
                c0 = owner_src0_cell[f]
                if c0 >= 0:
                    mat0 = owner_src0_mat[f]
                    w = mat0[i, 0]
                    if w != 0.0:
                        for v in range(5):
                            Q_o_at_n[v] += w * Q[c0, 0, v]
                        for a in range(3):
                            for b in range(3):
                                gv_o_at_n[a, b] += w * grad_vel[c0, 0, a, b]
                            gT_o_at_n[a] += w * grad_T[c0, 0, a]
                        mut_o_at_n += w * mu_t_field[c0, 0]
                idx1 = owner_src1_idx[f]
                if idx1 >= 0:
                    c1 = owner_src1_cell[idx1]
                    mat1 = owner_src1_mat[idx1]
                    w = mat1[i, 0]
                    if w != 0.0:
                        for v in range(5):
                            Q_o_at_n[v] += w * Q[c1, 0, v]
                        for a in range(3):
                            for b in range(3):
                                gv_o_at_n[a, b] += w * grad_vel[c1, 0, a, b]
                            gT_o_at_n[a] += w * grad_T[c1, 0, a]
                        mut_o_at_n += w * mu_t_field[c1, 0]
                # 混合拆分面边界半区（B-8）：neighbor 侧对称处理，规则同通用 kernel。
                mp_o = mixed_ow_partner[f]
                if mp_o >= 0 and mixed_ow_mask[f, i]:
                    for v in range(5):
                        Q_o_at_n[v] = Q_ghost[mp_o, i, v]
                    if bnd_adiabatic[mp_o]:
                        gT_bnd_n = mirror_normal_component(gT_n_i, adj_n_i)
                    else:
                        gT_bnd_n = gT_n_i
                    for a in range(3):
                        for b in range(3):
                            gv_o_at_n[a, b] = gv_n_i[a, b]
                        gT_o_at_n[a] = gT_bnd_n[a]
                    mut_o_at_n = mut_n_i

                Q_avg_n = np.empty(5)
                for v in range(5):
                    Q_avg_n[v] = 0.5 * (Q_n_i[v] + Q_o_at_n[v])
                gv_avg_n = np.empty((3, 3))
                for a in range(3):
                    for b in range(3):
                        gv_avg_n[a, b] = 0.5 * (gv_n_i[a, b] + gv_o_at_n[a, b])
                gT_avg_n = np.empty(3)
                for a in range(3):
                    gT_avg_n[a] = 0.5 * (gT_n_i[a] + gT_o_at_n[a])
                mut_avg_n = 0.5 * (mut_n_i + mut_o_at_n)

                G_common_native = viscous_physical_flux_point(Q_avg_n, gv_avg_n, gT_avg_n, mu, Pr, mut_avg_n, Pr_t)
                a0 = adj_n_i[0]
                a1 = adj_n_i[1]
                a2 = adj_n_i[2]
                G_tilde_common_n = np.empty(5)
                for v in range(5):
                    G_tilde_common_n[v] = a0 * G_common_native[0, v] + a1 * G_common_native[1, v] + a2 * G_common_native[2, v]

                G_phys_n = viscous_physical_flux_point(Q_n_i, gv_n_i, gT_n_i, mu, Pr, mut_n_i, Pr_t)
                G_tilde_own_n = np.empty(5)
                for v in range(5):
                    G_tilde_own_n[v] = a0 * G_phys_n[0, v] + a1 * G_phys_n[1, v] + a2 * G_phys_n[2, v]

                for v in range(5):
                    jump_neighbor[i, v] = G_tilde_common_n[v] - G_tilde_own_n[v]

                # 混合拆分面边界半区（B-8）：neighbor 侧边界 IP 罚项，规则同通用 kernel。
                adj_mag_n = np.sqrt(a0 * a0 + a1 * a1 + a2 * a2)
                h_ip_n = cell_volume[nc] / face_area[f]
                if mp_o >= 0 and mixed_ow_mask[f, i]:
                    pen_n = viscous_ip_penalty_tilde(
                        Q_n_i, Q_o_at_n, mu + mut_n_i, 0.0, h_ip_n, adj_mag_n,
                        1.0, c_ip, False,
                    )
                else:
                    # **内部面 IP 罚项**（2026-09-23 修复真实缺陷，完整依据见
                    # `viscous_ip_penalty_tilde` 的"为什么内部面也必须加"）：
                    # `sigma = grad(u)` 是纯单元内局部梯度
                    # （`compute_physical_gradient(field, mesh, ops)` 的签名里
                    # 没有任何面数据），没有 BR1 要求的提升项；界面耦合只有
                    # "粘性通量取两侧算术平均"这一层，内部面此前**零罚项** ——
                    # 正是 ABCM(2002) 框架里"提升项与罚项都为零"的那一档，
                    # 不满足强制性（实测均匀基态纯粘性算子谱正实部 328/2160）。
                    # 涡粘与热传导率都取**面平均**，与 `G_common` 一致。
                    k_tot_n = mu * CP_AIR / Pr + mut_avg_n * CP_AIR / Pr_t
                    pen_n = viscous_ip_penalty_tilde(
                        Q_n_i, Q_o_at_n, mu + mut_avg_n, k_tot_n, h_ip_n, adj_mag_n,
                        1.0, c_ip, True,
                    )
                for v in range(1, 5):
                    jump_neighbor[i, v] += pen_n[v]

            dj = det_jacs[nc, 0]
            weighted_jump_n = np.empty((n_fp, 5))
            for i in range(n_fp):
                w_area = ref_area_weight[i]
                for v in range(5):
                    weighted_jump_n[i, v] = w_area * jump_neighbor[i, v]
            contrib_neighbor = matmul_small(lift_native[nc_code - 6], weighted_jump_n)  # (1,5)
            for v in range(5):
                correction_per_thread[tid, nc, 0, v] += contrib_neighbor[0, v] / dj

    return correction_per_thread.sum(axis=0)
