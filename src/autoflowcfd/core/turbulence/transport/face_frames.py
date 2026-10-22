"""AutoFlowCFD V2.0 - 湍流标量输运的面数据：按"本侧坐标系"外插与提升（numba）。

## 每一侧都在自己的通量点顺序里算（2026-09-26，真实缺陷修复）

同一个物理面上，owner 与 neighbor 各自的面局部参数化给出的通量点**顺序
不同**（plate_demo 真实网格：53.3% 的内部面两侧顺序不一致——三角面的
点可以旋转/镜像）。DG 提升算子 `lift_native[op]` 按**该侧自己的**顺序
消费跳变量，所以两侧的跳变量必须分别在各自的顺序里构造：

    owner 侧      self  = E_owner    @ phi[owner]             （owner 顺序）
                  other = neighbor_src @ phi[neighbor 来源]    （同上）
    neighbor 侧   self  = E_neighbor @ phi[neighbor]          （neighbor 顺序）
                  other = owner_src    @ phi[owner 来源]       （同上）

这与平均流无粘/粘性 kernel（`fr_residual/inviscid_kernel.py` 等）的
neighbor-primary 分支是同一个结构。此前本模块把 owner 顺序下算出的跳变量
直接喂给 neighbor 的提升算子：对流算子在均匀来流下线性不稳定（CFL 0.1 的
SSP-RK3 下扰动 200 步 rms 从 8e-3 涨到 4e-2 并扩散到 13% 的单元，修正后
单调衰减），这是 P1 湍流场单元内锯齿、贴下限、逐单元松弛冻结的根因。

## 提升的符号约定只有一个

两侧都按 `J_side = (F* - F_side) . n_side`（`n_side` 为该侧外法向）构造跳变量，
提升统一为 `corr[cell] += sign * lift_side @ (w_side * J_side) / det`：对流
（`dphi/dt = -div F`）`sign=-1`，扩散（`dphi/dt = +div G`）`sign=+1`。面积权重
`w_side = ref_area_weight * |adj_row_side|`（物理面积权重，跳变量是物理通量
密度差；owner 侧它逐位等于既有的 `true_area_weight`）。此前 owner "-=" /
neighbor "+=" 的非对称约定只对反对称跳变量成立，扩散（BR1 跳变量两侧对称）
因此在 neighbor 侧是**反扩散**。
"""

import numpy as np
from numba import njit, prange

from autoflowcfd.core.fr_operators.flux_kernels.viscous import smooth_nonnegative


@njit(cache=True, parallel=True)
def extrapolate_scalar_pair_kernel(
    scalar_sps,
    self_cell, self_face_op, boundary_extrap_native,
    other_src0_cell, other_src0_tpl, other_src0_tid,
    other_src1_idx, other_src1_cell, other_src1_mat,
    apply_boundary_ghost,
    wall_dirichlet_zero_face, has_wall_dirichlet_value, wall_dirichlet_value_face,
    mixed_partner, mixed_mask,
):
    """某一侧坐标系下的 `(phi_self, phi_other)`，各 `(n_faces, n_fp)`。

    `self_cell[f] < 0` 的面（neighbor 坐标系下的边界面）两者都留 0，调用方不消费。

    ghost 规则（`apply_boundary_ghost` 只在 owner 坐标系为真——真边界面只有 owner
    一侧）：`other` 没有任何来源的面，Dirichlet-zero 壁面取镜像 `-self`（k=0），
    非零 Dirichlet 取 `2*target - self`（omega 壁面解析值），其余取零梯度 `self`。
    混合拆分面（B-8，`fr/face_flux_points/merge.py`）的边界半区按配对边界面同一
    规则逐通量点覆盖；配对边界面的 owner 就是本侧单元，所以它的逐通量点目标值
    与本侧同一顺序。
    """
    n_faces = self_cell.shape[0]
    n_fp = boundary_extrap_native.shape[1]
    n_sps = scalar_sps.shape[1]
    phi_self = np.zeros((n_faces, n_fp))
    phi_other = np.zeros((n_faces, n_fp))
    for f in prange(n_faces):
        sc = self_cell[f]
        if sc < 0:
            continue
        E = boundary_extrap_native[self_face_op[f]]
        for i in range(n_fp):
            v = 0.0
            for s in range(n_sps):
                v += E[i, s] * scalar_sps[sc, s]
            phi_self[f, i] = v
        c0 = other_src0_cell[f]
        idx1 = other_src1_idx[f]
        if c0 >= 0:
            m0 = other_src0_tpl[other_src0_tid[f]]
            for i in range(n_fp):
                v = 0.0
                for s in range(n_sps):
                    v += m0[i, s] * scalar_sps[c0, s]
                phi_other[f, i] = v
        if idx1 >= 0:
            c1 = other_src1_cell[idx1]
            m1 = other_src1_mat[idx1]
            for i in range(n_fp):
                v = 0.0
                for s in range(n_sps):
                    v += m1[i, s] * scalar_sps[c1, s]
                phi_other[f, i] += v
        if apply_boundary_ghost and c0 < 0 and idx1 < 0:
            for i in range(n_fp):
                if wall_dirichlet_zero_face[f]:
                    phi_other[f, i] = -phi_self[f, i]
                elif has_wall_dirichlet_value[f]:
                    phi_other[f, i] = 2.0 * wall_dirichlet_value_face[f, i] - phi_self[f, i]
                else:
                    phi_other[f, i] = phi_self[f, i]
        mp = mixed_partner[f]
        if mp >= 0:
            for i in range(n_fp):
                if mixed_mask[f, i]:
                    if wall_dirichlet_zero_face[mp]:
                        phi_other[f, i] = -phi_self[f, i]
                    elif has_wall_dirichlet_value[mp]:
                        phi_other[f, i] = 2.0 * wall_dirichlet_value_face[mp, i] - phi_self[f, i]
                    else:
                        phi_other[f, i] = phi_self[f, i]
    return phi_self, phi_other


@njit(cache=True, parallel=True)
def face_mass_flux_kernel(
    rho_u, owner_cell, owner_face_op, neighbor_cell, neighbor_face_op, boundary_extrap_native,
    owner_src0_cell, owner_src0_tpl, owner_src0_tid, owner_src1_idx, owner_src1_cell, owner_src1_mat,
    mixed_ow_partner, mixed_ow_mask, normal_owner, normal_neighbor,
):
    """两侧坐标系下通量点上的质量通量 `(m_owner, m_neighbor)`，各 `(n_faces, n_fp)`。

    都取 **owner 的迹** `(rho*u)|_owner`：owner 坐标系点乘 owner 外法向，neighbor
    坐标系（经 `owner_src` 取到 neighbor 通量点上）点乘 neighbor 外法向，于是同一
    物理点上 `m_neighbor = -m_owner`——公共通量单值（守恒）、上风选择两侧一致。
    混合拆分面的边界半区没有 owner 覆盖，取 neighbor 自身的迹（与 owner 坐标系在
    真边界面上用自身迹同一原则）。

    这个取法随面的 owner / neighbor 取向而变，跳量 `[rho u]` 为 O(h^(p+1))，与离散误差同阶。2026-10-22（I-67）
    把它与两侧迹的平均 `{rho u} . n`（同样单值、但与取向无关）做了 A/B：
    - 湍流平板 P0 -> P1 -> P2（NK）：SA 两种取法都 105 步完成三个阶段，摩阻逐位相同；SST 两种取法停在同一平台。
    - 槽道 SST P3（NK，`test_implicit_sst_nk.py`）：步数 54 / 55 相当；两侧平均的 k 负欠冲从峰值的 2.9% 增加到
      5.2%，且位置从出口与壁面交界处移进壁面层内部。

    两侧平均没有稳定的优势，保留 owner 侧的迹。
    """
    n_faces = owner_cell.shape[0]
    n_fp = boundary_extrap_native.shape[1]
    n_sps = rho_u.shape[1]
    m_o = np.zeros((n_faces, n_fp))
    m_n = np.zeros((n_faces, n_fp))
    for f in prange(n_faces):
        oc = owner_cell[f]
        E = boundary_extrap_native[owner_face_op[f]]
        for i in range(n_fp):
            acc = 0.0
            for d in range(3):
                t = 0.0
                for s in range(n_sps):
                    t += E[i, s] * rho_u[oc, s, d]
                acc += t * normal_owner[f, i, d]
            m_o[f, i] = acc
        nc = neighbor_cell[f]
        if nc < 0:
            continue
        c0 = owner_src0_cell[f]
        idx1 = owner_src1_idx[f]
        mp = mixed_ow_partner[f]
        En = boundary_extrap_native[neighbor_face_op[f]]
        for i in range(n_fp):
            acc = 0.0
            for d in range(3):
                t = 0.0
                if mp >= 0 and mixed_ow_mask[f, i]:
                    for s in range(n_sps):
                        t += En[i, s] * rho_u[nc, s, d]
                else:
                    if c0 >= 0:
                        for s in range(n_sps):
                            t += owner_src0_tpl[owner_src0_tid[f], i, s] * rho_u[c0, s, d]
                    if idx1 >= 0:
                        c1 = owner_src1_cell[idx1]
                        for s in range(n_sps):
                            t += owner_src1_mat[idx1, i, s] * rho_u[c1, s, d]
                acc += t * normal_neighbor[f, i, d]
            m_n[f, i] = acc
    return m_o, m_n


@njit(cache=True, inline='always')
def diffusion_jump_point(ps, gs, dns, po, go, dno, h, c_ip, is_bnd, is_dir, target, gamma_band):
    """一个通量点上的扩散跳变量 `J = (G* - G_self) . n_self`（公式见
    `diffusion_face_jumps_kernel`）。残差核与湍流解析 Jacobian 共用这一份。

    `gs`/`go` 是单元内扩散系数多项式**外插**到面上的值，在这里换成取用值 `smooth_nonnegative(., gamma_band)`
    （`gamma_band` 取分子粘度；解点上的扩散系数不小于它，所以只改负的与窄带内的外插值）。扩散系数在贴壁
    第一层里随湍流量陡升时外插到壁面为负——cube_demo SA P1 上 91% 的第一层棱柱如此，最负到分子值的 -5 倍——
    壁面内罚 Dirichlet 的罚项系数 `c_ip Gamma / h` 随之为负，把壁面处的湍流量推离给定值；与平均流的涡粘是
    同一个缺陷（`fr_operators/flux_kernels/viscous.py::viscous_physical_flux_point` 文档"涡粘按非负取用"）。
    """
    gs = smooth_nonnegative(gs, gamma_band)
    go = smooth_nonnegative(go, gamma_band)
    if is_bnd:
        if is_dir:
            return -(c_ip * gs / h) * (ps - target)
        return -gs * dns
    eta = c_ip * 0.5 * (gs + go) / h
    return 0.5 * (go * dno - gs * dns) - eta * (ps - po)


@njit(cache=True, inline='always')
def convection_jump_point(m, ps, po):
    """一个通量点上的对流跳变量 `J = m (phi_upwind - phi_self)`，`m` 为本侧外法向
    质量通量（见 `convection.py::compute_scalar_convection_residual`）。"""
    if m >= 0.0:
        return 0.0
    return m * (po - ps)


@njit(cache=True, parallel=True)
def diffusion_face_jumps_kernel(
    phi, gamma, grad_phi,
    self_cell, self_face_op, boundary_extrap_native,
    other_src0_cell, other_src0_tpl, other_src0_tid, other_src1_idx, other_src1_cell, other_src1_mat,
    normal, h_face, c_ip, is_boundary, is_dirichlet, target, gamma_band,
):
    """某一侧坐标系下扩散界面项的跳变量 `J = (G* - G_self) . n_self`，`(n_faces, n_fp)`。

    与 `diffusion.py` 文档的公式逐项对应，一趟算完（自身/对侧的 phi、Gamma、
    法向梯度各自外插再组合，不物化十份中间数组）：

        内部点   J = 1/2 (Gamma_o dphi_o/dn - Gamma_s dphi_s/dn) - eta (phi_s - phi_o)
                 eta = c_ip * (Gamma_s + Gamma_o) / 2 / h_face
        壁面点   J = -(c_ip Gamma_s / h_face) (phi_s - target)       （内罚 Dirichlet）
        其余边界 J = -Gamma_s dphi_s/dn                               （齐次 Neumann）

    边界点分类由 `faces.boundary_diffusion_targets` 给出（CPU/GPU 共用那份规则）。
    """
    n_faces = self_cell.shape[0]
    n_fp = boundary_extrap_native.shape[1]
    n_sps = phi.shape[1]
    J = np.zeros((n_faces, n_fp))
    for f in prange(n_faces):
        sc = self_cell[f]
        if sc < 0:
            continue
        E = boundary_extrap_native[self_face_op[f]]
        c0 = other_src0_cell[f]
        idx1 = other_src1_idx[f]
        h = h_face[f]
        for i in range(n_fp):
            n0 = normal[f, i, 0]
            n1 = normal[f, i, 1]
            n2 = normal[f, i, 2]
            ps = 0.0
            gs = 0.0
            dns = 0.0
            for s in range(n_sps):
                w = E[i, s]
                ps += w * phi[sc, s]
                gs += w * gamma[sc, s]
                dns += w * (grad_phi[sc, s, 0] * n0 + grad_phi[sc, s, 1] * n1 + grad_phi[sc, s, 2] * n2)
            if is_boundary[f, i]:
                J[f, i] = diffusion_jump_point(ps, gs, dns, 0.0, 0.0, 0.0, h, c_ip, True,
                                               is_dirichlet[f, i], target[f, i], gamma_band)
                continue
            po = 0.0
            go = 0.0
            dno = 0.0
            if c0 >= 0:
                for s in range(n_sps):
                    w = other_src0_tpl[other_src0_tid[f], i, s]
                    po += w * phi[c0, s]
                    go += w * gamma[c0, s]
                    dno += w * (grad_phi[c0, s, 0] * n0 + grad_phi[c0, s, 1] * n1 + grad_phi[c0, s, 2] * n2)
            if idx1 >= 0:
                c1 = other_src1_cell[idx1]
                for s in range(n_sps):
                    w = other_src1_mat[idx1, i, s]
                    po += w * phi[c1, s]
                    go += w * gamma[c1, s]
                    dno += w * (grad_phi[c1, s, 0] * n0 + grad_phi[c1, s, 1] * n1 + grad_phi[c1, s, 2] * n2)
            J[f, i] = diffusion_jump_point(ps, gs, dns, po, go, dno, h, c_ip, False, False, 0.0, gamma_band)
    return J


@njit(cache=True, inline='always')
def _lift_one_side(jump_f, adj_row_f, ref_area_weight, lift, n_fp):
    """`lift @ (ref_area_weight * |adj_row| * jump)`：`jump_f (n_fp, n_comp)` -> `(n_sps, n_comp)`。

    多个分量共用同一侧的权重与提升矩阵（P0 速度梯度一次提升 9 个分量）；单分量时逐位
    等于此前的标量版本（同一乘法次序、同一通量点累加顺序）。
    """
    n_sps = lift.shape[0]
    n_comp = jump_f.shape[1]
    w = np.empty(n_fp)
    for i in range(n_fp):
        a0 = adj_row_f[i, 0]
        a1 = adj_row_f[i, 1]
        a2 = adj_row_f[i, 2]
        w[i] = ref_area_weight[i] * np.sqrt(a0 * a0 + a1 * a1 + a2 * a2)
    out = np.empty((n_sps, n_comp))
    for s in range(n_sps):
        for m in range(n_comp):
            acc = 0.0
            for i in range(n_fp):
                acc += lift[s, i] * (w[i] * jump_f[i, m])
            out[s, m] = acc
    return out


@njit(cache=True, parallel=True)
def lift_side_jumps_kernel(
    jump_owner, jump_neighbor, sign,
    owner_cell, neighbor_cell, owner_face_op, neighbor_face_op,
    owner_adj_row_exact, neighbor_adj_row_exact, ref_area_weight, lift_native,
    owner_is_primary, neighbor_is_primary, det_jacs,
    face_indices, out,
):
    """两侧跳变量 `(n_faces, n_fp, n_comp)` 提升回解点 `out (n_cells, n_sps, n_comp)`
    （图着色版：同色面不共享单元，直接写 `out`）。"""
    n_fp = jump_owner.shape[1]
    n_sps = out.shape[1]
    n_comp = out.shape[2]
    for fi in prange(face_indices.shape[0]):
        f = face_indices[fi]
        if owner_is_primary[f]:
            oc = owner_cell[f]
            c = _lift_one_side(jump_owner[f], owner_adj_row_exact[f], ref_area_weight,
                               lift_native[owner_face_op[f]], n_fp)
            for s in range(n_sps):
                for m in range(n_comp):
                    out[oc, s, m] += sign * c[s, m] / det_jacs[oc, s]
        nc = neighbor_cell[f]
        if nc >= 0 and neighbor_is_primary[f]:
            c = _lift_one_side(jump_neighbor[f], neighbor_adj_row_exact[f], ref_area_weight,
                               lift_native[neighbor_face_op[f]], n_fp)
            for s in range(n_sps):
                for m in range(n_comp):
                    out[nc, s, m] += sign * c[s, m] / det_jacs[nc, s]
