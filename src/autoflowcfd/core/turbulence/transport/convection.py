"""AutoFlowCFD V2.0 - 湍流标量的**对流**残差（含去混叠体积项）。

从 `core/turbulence/transport.py` 拆出（2026-09-24）。纯搬家，逻辑未改。

残差约定：`-div(rho*U*phi)/det(J)`（含界面上风校正），见包 `__init__.py`
的"符号约定"一节。
"""

import os
import numpy as np


from autoflowcfd.core.fr_operators.volume_contract import (
    OVERINT_CHUNK_CELLS, contract_shared_operator_1axis,
    contract_shared_operator_2axis, contravariant_flux_from_metric,
    get_overintegration_context,
)
from autoflowcfd.core.fr_operators.face_kernels import get_flat_face_geometry
from autoflowcfd.core.turbulence.transport_kernel import (
    scalar_convection_volume_kernel,
)

from .faces import (
    ScalarConvectionGeometry,
    _distribute_correction_to_cells,
    _extrapolate_owner_only_to_faces,
    _extrapolate_scalar_to_faces,
)


#: k/omega 输运体积项是否走过积分（去混叠），由 `AFCFD_TURB_OVERINT` 选择：
#:   "off"（默认，与此前行为逐位一致）
#:   "on" —— 对流/扩散体积项都在 FINE 点上算完再限制回 coarse
#:
#: **为什么需要它（2026-09-15）**：`fr/collapsed_basis.py::
#: build_overintegration_operators` 文档记录了去混叠的动机，并给出实测
#: 数字——对解析残差恒为 0 的线性剪切场，不去混叠的 P2 体积项算出的残差
#: 是真值的 43~62 倍。那套机制一直**只接在平均流的无粘体积项**上
#: （`fr_residual/inviscid.py` 的 `mesh.jacobians_fine is not None` 分支），
#: 粘性项与本模块的 k/omega 输运项**完全没有**。
#:
#: 而 k/omega 对流体积项算的是 `div(adj(J)·rho·u·phi)`——一个**三重**
#: 非线性乘积，真实多项式次数约 `3*order + 度量次数`，远高于 order。
#: 直接在 coarse SPs 上对它做微分就是"先混叠再求导"。
#:
#: 这与 `filter_scalar_field` 文档记录的 2026-09-12 真实 P1 发散症状
#: 直接对应：那次的诊断原文是"P1 多项式在单元内部相邻解点间出现数量级
#: 跳变 -> 外插到面通量点后被上风格式放大成巨大的虚假对流残差"，正是
#: 对流项混叠的形态。当时的处置是给 k/omega 加模态滤波器，而那个滤波器
#: 每个 RK stage 清掉一整阶（见 `fr/modal_filter.py`）——用牺牲阶数换
#: 稳定。去混叠是同一个问题的**不牺牲阶数**的正解。
#:
#: **默认已于 2026-09-15 改为 "on"**，三条证据齐备后才改的：
#:
#: 1. **解析判据**：`rho`/`u` 取常数、`phi` 取线性（`rho*u*phi` 只有一次、
#:    完全落在 P1 解空间内，任何非零误差都只能来自离散算子本身），
#:    `d(rho*phi)/dt = -rho*(u·G)` 有闭式解。走公开接口
#:    `compute_scalar_convection_residual` 实测：
#:
#:        默认 off: order=1 棱柱相对误差 **1.1294（113%）**
#:                  （残差范围 [-12.26, +1.11]，解析值 -8.5750）
#:                  order=1 四面体 6.84e-15（机器零）
#:        打开 on : order=1 棱柱 **4.61e-10**（改善约 2.4e9 倍）
#:                  order=2 两档逐位相同（2.2597e-09 / 2.2576e-09）
#:
#:    棱柱正是边界层单元、k/omega 最要紧的地方；而 order=2 在两档下都
#:    已足够精确，说明那个 113% 不是"这套离散本来就这么差"，是 order=1
#:    特有的混叠（非仿射棱柱上 adj(J) 是非平凡多项式，`adj(J)*rho*u*phi`
#:    真实次数高于 1；四面体在该网格上仿射、adj(J) 常数，所以差 14 个
#:    数量级）。边界条件不是原因：order=1 与 order=2 的边界面数完全相同。
#: 2. **代价已量化**：微基准（5 万单元、3 线程、V=1）单次标量链路
#:    85.5ms，按 79 万单元线性外推 1.354s；每步 4 次（k/omega 对流 +
#:    k/omega 扩散）约 5.42s，相对实测 ~53s/步约 **+10.2%**。
#: 3. **真实网格已验证**：79 万单元 cube_demo 的两个 250 步运行
#:    （`true_cfl03`/`true_adaptive`，零阶数损失 + 逐点 omega 下限）本来
#:    就是带 `TURB_OVERINT=on` 跑的，已单调收敛 137+ 步、om_max 全程不动。
#:
#: +10% 的单步代价换掉生产阶数上 113% 的残差误差——这个权衡不需要再等。
#: `off` 保留为受控 A/B 与回归复现的入口。
def resolve_turb_overintegration() -> str:
    """返回 k/omega 输运体积项的去混叠开关，并校验取值。

    每次调用都重读环境变量（不缓存模块级常量）：这一维不影响算子构造
    （过积分三件套与 `jacobians_fine` 在 `order>=1` 时本来就无条件构造
    好了，见 `fr/operators.py` 与 `grid/high_order/high_order_mesh_order.py`），
    运行期读取是安全的，也让测试可以直接 monkeypatch 环境变量。
    """
    v = os.environ.get("AFCFD_TURB_OVERINT", "on").lower()
    if v not in ("off", "on"):
        raise ValueError(
            f"AFCFD_TURB_OVERINT={v!r} 不是合法取值（off | on）。"
            f"'on'（默认）把对流/扩散体积项改走 FINE 点去混叠，"
            f"'off' 是 2026-09-15 之前的行为（直接在 coarse SPs 上微分，"
            f"order=1 棱柱有 113% 相对误差），保留作受控 A/B 入口。")
    return v


#: `get_overintegration_context` 的旧名别名：三个消费方（无粘体积项、
#: 本模块的 k/omega 输运、粘性体积项）共用同一份实现，见
#: `fr_operators/volume_contract.py::get_overintegration_context`。
_turb_overint_ops = get_overintegration_context

_TURB_OVERINT_CHUNK_CELLS = OVERINT_CHUNK_CELLS


def _scalar_convection_volume_overintegrated(
    scalar_field, rho, velocity, oi, n_sps,
):
    """对流体积项 `div(adj(J)*rho*u*phi)` 的去混叠版，返回 (n_cells,n_sps)。

    链路与 `fr_residual/inviscid.py` 的过积分分支逐项对应：
      ① phi/rho/u 各自精确插值到 FINE 点（它们各自次数 <= order，
         插值不引入误差——**乘积必须在 FINE 点上做**，这正是去混叠的
         全部内容：先插值再相乘，而不是先相乘再插值）；
      ② 用解析精确的 FINE 点度量算逆变通量；
      ③ 用 FINE 网格自己的微分矩阵求散度；
      ④ 精确插值限制回 coarse SPs。
    """
    n_cells = scalar_field.shape[0]
    div_F = np.empty((n_cells, n_sps))
    # 每段自带自己的 n_fine 与已切好的细点度量（2026-09-17）：native
    # 四面体的过积分细网格轴不再填充到棱柱的 (oo+1)^3 宽度，两段的
    # n_fine 不同了。度量按**段内局部**索引切（`i0 = c0 - seg_lo`）——
    # 用全局 c0 去切段内数组会静默取到错误的单元。
    for (seg_lo, seg_hi, n_fine, det_seg, inv_seg,
         op_c2f, op_D_fine, op_f2c) in oi["segs"]:
        for c0 in range(seg_lo, seg_hi, _TURB_OVERINT_CHUNK_CELLS):
            c1 = min(c0 + _TURB_OVERINT_CHUNK_CELLS, seg_hi)
            i0, i1 = c0 - seg_lo, c1 - seg_lo
            phi_f = contract_shared_operator_1axis(
                op_c2f, np.ascontiguousarray(scalar_field[c0:c1, :, None]))
            rho_f = contract_shared_operator_1axis(
                op_c2f, np.ascontiguousarray(rho[c0:c1, :, None]))
            u_f = contract_shared_operator_1axis(
                op_c2f, np.ascontiguousarray(velocity[c0:c1]))       # (n,n_fine,3)
            # rho*u*phi 在 FINE 点上相乘（去混叠的核心）
            F_phys_f = (rho_f * phi_f * u_f)[..., None]              # (n,n_fine,3,1)
            del phi_f, rho_f, u_f
            # ascontiguousarray：段内度量是带偏移的非连续视图
            # （`det_all[n_prism:, :n_fine_tet]`），numba kernel 要连续输入
            F_tilde_f = contravariant_flux_from_metric(
                np.ascontiguousarray(det_seg[i0:i1]),
                np.ascontiguousarray(inv_seg[i0:i1]), F_phys_f)
            del F_phys_f
            div_f = contract_shared_operator_2axis(op_D_fine, F_tilde_f)  # (n,n_fine,1)
            del F_tilde_f
            div_F[c0:c1] = contract_shared_operator_1axis(op_f2c, div_f)[..., 0]
            del div_f
    return div_F


def compute_scalar_convection_residual(
    scalar_field: np.ndarray,
    rho: np.ndarray,
    velocity: np.ndarray,
    mesh,
    ops,
    wall_dirichlet_zero_face: np.ndarray = None,
    wall_dirichlet_value_face: np.ndarray = None,
    has_wall_dirichlet_value: np.ndarray = None,
    flat_face_override=None,
    conv_geom: "ScalarConvectionGeometry" = None,
) -> np.ndarray:
    """计算标量对流 FR 残差（体积项 + 界面上风校正）。

    对流方程: d(rho*phi)/dt + div(rho*U*phi) = 0
    残差 = -div(rho*U*phi)/det(J) + interface_correction/det(J)

    Args:
        scalar_field: (n_cells, n_sps) 标量场（k 或 omega）
        rho: (n_cells, n_sps) 密度
        velocity: (n_cells, n_sps, 3) 速度
        mesh: HighOrderMesh
        ops: FROperators
        wall_dirichlet_zero_face: (n_faces,) bool，可选，见
            `_extrapolate_scalar_to_faces` 文档——只影响 scalar_field
            自身外插到面的 ghost 值（决定上风通量的物理量），不影响
            rho/velocity 的外插（无滑移壁面上 u=0 已经由平均流的
            WALL 幽灵态保证，这里不需要重复处理）。
        wall_dirichlet_value_face, has_wall_dirichlet_value: 见
            `_extrapolate_scalar_to_faces` 文档，omega 壁面解析式用，
            与 wall_dirichlet_zero_face 互斥。
        flat_face_override: 显式传入时优先使用，不再调用
            `get_flat_face_geometry(mesh, ops)`（2026-09-02 分布式湍流
            移植新增，与 `core/fr_residual/inviscid.py::compute_
            inviscid_residual_fr` 同名参数同一个道理）——分布式路径下
            `mesh` 是 `DistributedMeshAdapter`，其 `face_connectivity`
            是 local+halo 压缩索引空间下的 `DistributedFlatFaceGeometry`，
            不是 `get_flat_face_geometry` 内部期望的完整全局
            `FRFaceConnectivity`，用它重新构建一遍会得到错误的面几何。
            调用方需要传入已经按同一套压缩索引空间构造好的
            `dist_fc.base_flat`。单机路径不传，行为完全不变。

    Returns:
        residual: (n_cells, n_sps) 对流残差（dphi/dt 量纲，已除以 rho 前的
                  原始残差，调用方需自行除以 rho）
    """
    n_cells = mesh.n_cells
    n_sps = mesh.n_sps_per_cell
    n_prism = mesh.n_prism_cells

    det_jacs = mesh.jacobians["det_jacs"].reshape(n_cells, n_sps)
    inv_jacs = mesh.jacobians["inv_jacs"].reshape(n_cells, n_sps, 3, 3)

    # === 体积项：按单元分块执行 adj_j 构造→物理通量→逆变通量→散度 全链路
    # （真实内存修复，2026-09-01，见项目 memory：cube_demo 79万单元 P2+SST
    # 组合内存峰值实测约需 37GB，超过常见 32GB 工作站配置，会在 P2 阶段
    # 第一步内触发 numpy ArrayMemoryError）。此前这里一次性对全场分配
    # `adj_j`（(n_cells,n_sps,3,3)~1.7GB）+ `rho_u_phi`/`F_tilde`
    # （各~500MB），SST 每步要独立调用本函数 4 次（k 对流/k 扩散/omega
    # 对流/omega 扩散，见 compute_turbulence_transport_residual），即使
    # 每次都及时 del，瞬时峰值仍会与同一步里平均流 inviscid.py/
    # viscous_flux.py 自己的大数组同时驻留，合计推高总峰值。这条链上
    # 每一步都是 cell 局部的（`np.matmul`/`np.tensordot` 按 cell 批量，
    # cell 之间零数据依赖），把 cell 轴切块、块内走完链路再进下一块，
    # 与 `fr_residual/inviscid.py`/`viscous_flux.py` 的过积分/体积项
    # 分块修复同一个原理、同一个分块大小（`_VISC_CHUNK_CELLS`），块内
    # 计算形状/求和顺序与全场版逐位一致，数值结果不变。
    # native 四面体（路径C）：D_3d_tet 是坍缩坐标专属微分矩阵，对 native
    # 单纯形基节点没有意义，需改用已零填充到全局宽度的 D_native_tet_padded
    # （理由与 gradients.py::compute_physical_gradient/viscous_flux.py
    # 体积项同一处文档，此前这里从未适配，见本模块 native 分支引入记录）。
    _tet_op_D = ops.D_native_tet_padded if getattr(ops, "D_native_tet_padded", None) is not None else ops.D_3d_tet
    # 性能优化（2026-09-13）：整条"度量×通量→散度"链压进 `scalar_convection_
    # volume_kernel`（按 cell prange，见该 kernel 文档），并复用调用方预先
    # 算好的、与标量无关的 `rho_u_tilde`（k/omega 两次调用共享，见
    # `ScalarConvectionGeometry` 文档）。原实现是 Python 层分块 +
    # `np.matmul` 逐点 3x3@3x1 微型 gemm + 3 次 `np.tensordot`，既物化多份
    # 块中间数组、又完全不随 CPU 核数并行。
    if conv_geom is not None:
        rho_u_tilde = conv_geom.rho_u_tilde
    else:
        rho_u = rho[:, :, None] * velocity
        rho_u_tilde = contravariant_flux_from_metric(det_jacs, inv_jacs, rho_u[..., None])[..., 0]
        del rho_u
    # 去混叠（AFCFD_TURB_OVERINT=on）：`rho*u*phi` 是三重非线性乘积，
    # 直接在 coarse SPs 上微分等价于"先混叠再求导"，理由与实测数字见
    # `resolve_turb_overintegration` 文档。默认 off，行为逐位不变。
    _oi = _turb_overint_ops(mesh, ops) if resolve_turb_overintegration() == "on" else None
    if _oi is not None:
        div_F = _scalar_convection_volume_overintegrated(
            scalar_field, rho, velocity, _oi, n_sps)
    else:
        div_F = np.empty((n_cells, n_sps))
        if n_prism > 0:
            scalar_convection_volume_kernel(
                np.ascontiguousarray(scalar_field[:n_prism]),
                np.ascontiguousarray(rho_u_tilde[:n_prism]),
                np.ascontiguousarray(ops.D_3d_prism), div_F[:n_prism],
            )
        if n_cells > n_prism:
            scalar_convection_volume_kernel(
                np.ascontiguousarray(scalar_field[n_prism:]),
                np.ascontiguousarray(rho_u_tilde[n_prism:]),
                np.ascontiguousarray(_tet_op_D), div_F[n_prism:],
            )
    if conv_geom is None:
        del rho_u_tilde

    # 真实复现（2026-08-21，79万单元生产网格，Order Continuation P0->P1
    # 切换后）：退化单元（坍缩坐标/BL 挤出导致 det(J) 局部极小，见
    # troubled_cell.py 模块文档）上这里会真的溢出到 inf，`np.errstate` 只是
    #
    # **本段已两次更正，当前状态（2026-09-19）**：
    # 最初写的是"平均流残差有 mechanism-1/2 两道专门保护，本模块至今
    # 没有"；2026-09-15 更正成"实际干预全部由机制3 承担，本模块已接入
    # 机制3"。现在**机制3 也已整体删除**（真实网格消融对照证明它触发了
    # 但只把残差轨迹改变 ~1e-10、不改变结局，见
    # `fr_residual/inviscid.py`）。
    #
    # 所以现状是：平均流与湍流输运**都不做**残差量级异常抑制；机制1/2
    # 只产出诊断报告；退化单元的对策是网格质量门。三者对称，没有哪条
    # 路径比另一条多一层保护。
    #
    # 这段注释被改过两次都是因为"照搬旧注释"差点把过时表述当成事实 ——
    # 核实代码而不是读注释。
    # 让这个*已知、已经在下游处理*的溢出不再往 stderr 打印 RuntimeWarning
    # 噪音——不改变任何数值结果：`compute_turbulence_transport_residual`
    # 末尾的 `np.where(np.isfinite(...), ..., 0.0)` 本来就会把这类 inf/nan
    # 结果清零，交给 SST.update_fields 的正性限制器接管（见该函数文档
    # "NaN/Inf 隔离"一节），此前只是没有抑制这个警告，容易被误读成新
    # 出现的异常。真正的解析修复（把 troubled-cell 的 mechanism-1/2
    # 保护接入湍流标量输运）是比这更大的独立工作，见
    # transport_kernel.py::extrapolate_scalar_to_faces_kernel 文档同类
    # 说明。
    with np.errstate(over='ignore', invalid='ignore'):
        residual = -div_F / det_jacs  # 体积项对流残差
    del div_F  # 体积项已经收尾，界面项不再需要它（F_tilde_chunk 已在分块循环内逐块释放）

    # === 界面项（上风校正）===
    flat = flat_face_override if flat_face_override is not None else get_flat_face_geometry(mesh, ops)
    n_fp = flat.n_fp

    # 外插 rho, velocity, scalar 到面通量点——rho/velocity 只需要 owner 侧
    # （下面 mass_flux 只用 owner 侧状态，见 _extrapolate_owner_only_to_faces
    # 文档"真实内存修复"一节：neighbor 侧此前算出来从未被读取，纯浪费）；
    # 只有标量场本身需要两侧（上风选择要比较 owner/neighbor）。
    if conv_geom is None:
        rho_owner_fp = _extrapolate_owner_only_to_faces(rho, flat)
        vel_owner_fp = np.zeros((flat.n_faces, n_fp, 3))
        for d in range(3):
            vel_owner_fp[:, :, d] = _extrapolate_owner_only_to_faces(velocity[:, :, d], flat)
    phi_owner_fp, phi_neighbor_fp = _extrapolate_scalar_to_faces(
        scalar_field, flat, ops, mesh, wall_dirichlet_zero_face,
        wall_dirichlet_value_face, has_wall_dirichlet_value,
    )

    # 计算每个面通量点的物理质量通量（使用 true_normal）
    # mass_flux_phys[f,fp] = (rho * U) . n̂_true
    # `mass_flux` 与标量无关，k/omega 共享（见 `ScalarConvectionGeometry`
    # 文档）——传入 conv_geom 时直接复用，省掉 rho 与 3 个速度分量各一次
    # 全场面外插（187 万面×n_fp 个通量点）。
    if conv_geom is not None:
        mass_flux = conv_geom.mass_flux
    else:
        rho_u_owner = rho_owner_fp[..., None] * vel_owner_fp  # (n_faces, n_fp, 3)
        mass_flux = np.sum(rho_u_owner * flat.true_normal, axis=-1)  # (n_faces, n_fp)
        del rho_owner_fp, vel_owner_fp, rho_u_owner  # 同上"真实内存修复"一节，及时释放

    # 迎风选择
    phi_upwind = np.where(mass_flux >= 0, phi_owner_fp, phi_neighbor_fp)

    # 通量差（用于校正分配）——真实 bug 修复（2026-09-12，cube_demo
    # 791,492 单元真实网格 P0 阶段长程续算 k_max/omega_max 复合增长排查
    # 发现，见下方"owner/neighbor 跳变量不对称"完整推导）：owner 侧和
    # neighbor 侧的校正必须分别相对各自的面值计算，不能共用同一个相对
    # phi_owner_fp 的跳变量。
    #
    # 完整推导（DG/FR 标准做法，见 core/fr_residual/inviscid_kernel.py
    # 里 AUSM+up 通量分两次、分别用 owner/neighbor 各自状态当"内部通量"
    # 计算 jump_owner/jump_neighbor 的既有正确实现——本函数此前没有跟
    # 那个模式对齐，是本模块独立实现、独立踩坑的同一类问题）：
    #   owner 侧："体积项"（P1+ 时由 D_3d 算出，P0 时恒为 0）隐含假设
    #     该单元通过每个面的"自身内部通量"是 mass_flux*phi_owner_fp
    #     （用 owner 自己的面值）；界面校正 = 真实通量(common,用上风值)
    #     减去这个假设值 = mass_flux*(phi_upwind-phi_owner_fp)，即现有
    #     的 raw_jump_fp，continue 用于 owner 侧不变。
    #   neighbor 侧：同一个面，站在 neighbor 自己的体积项视角，它的
    #     "自身内部通量"假设是（用 neighbor 自己的面值，法向对调一次，
    #     两次取负抵消）mass_flux*phi_neighbor_fp——不是 phi_owner_fp！
    #     neighbor 侧真正的界面校正 = mass_flux*(phi_upwind-phi_neighbor_fp)，
    #     是一个独立于 raw_jump_fp 的量，只在 phi_owner_fp==phi_neighbor_fp
    #     （面上没有跳变）时才恰好相等。
    #
    # 此前代码把 owner 侧算出的 raw_jump_fp 原样又给 neighbor 侧用（只是
    # 累加符号相反），在 owner 恰好处于该面上风侧时（mass_flux>=0）
    # phi_upwind==phi_owner_fp，raw_jump_fp 恒为 0——这种情形下 neighbor
    # （下风侧，真实网格中占全部内部面里的一半）完全没有从这个面收到
    # 任何对流稀释/浓缩效果，等价于凭空丢弃了这部分物理输运。这是 P0
    # 阶段（体积项恒零、全部输运只能靠界面项）唯一的输运来源，这个缺口
    # 因此完全暴露：局部单元一旦（哪怕因为其他机制）值略高于周围，会因为
    # 系统性地收不到来自上风侧的正确稀释而持续偏高，若同时有其他面
    # 贡献哪怕很小的净流入，缺乏正确抵消的稀释就会造成真实、持续、复合
    # 的数值增长（真实观测：cube_demo 单元 87035，k 从 iter 2600 的 0.99
    # 复合增长到 iter 3100 的 2.05，约 1.2x/100步；最小 2-四面体合成网格
    # 决定性复现：owner 上风时 neighbor 完全收不到本该有的稀释，conv_k
    # 恒为 0，与真实物理应有的非零稀释矛盾）。P1+ 阶段体积项非零、能
    # 部分补偿这个缺口，这也是该问题只在长期停留 P0 的场景下才充分暴露
    # 的原因。
    # 注（2026-09-12，尝试并撤销的一次修复，记录下来避免后续重蹈覆辙）：
    # 这里曾经尝试在 P0（n_sps==1）时改用"直接有限体积"公式（跳变量直接
    # 取 phi_upwind，不减任何一侧自身面值），模仿 core/fr_residual/
    # inviscid_p0.py 的做法，理由是 P0 架构上 volume_term 恒为 0（D_3d
    # 是 1x1 零矩阵），下面这套"跳变量=上风值-自身面值"的 DG 差额公式
    # 因此永远缺失一部分理应由 volume_term 提供的贡献。用合成网格做
    # 独立正确性检验（均匀标量场、无跳变，纯对流残差必须处处精确为
    # 零——这是不依赖具体公式、只依赖"无源无跳变"这个前提的基本不变量）
    # 时**决定性证伪**：改成"不减自身面值"后，均匀场在某些单元上给出
    # 高达 451 的伪残差（应为 0），根源是这个简化后的公式又变得对
    # `sum_faces(mass_flux)`（局部质量守恒残差，真实网格计算中后期该量
    # 远非零，尤其是本次真实排查最早定位到的"驻点残差"单元）敏感——
    # inviscid_p0.py 的直接公式之所以安全，是因为它求解的是完整
    # 5 变量 Euler 方程组本身（该公式与连续性方程自洽是同一个解），而
    # k/omega 是附着在已经独立演化、瞬时并不精确满足连续性的密度场上的
    # 被动标量方程，直接照搬会重新引入当天早些时候已经用真实数据证伪过
    # 的"质量不守恒交叉项"敏感性——绕了一圈证实那次证伪的判断没错，只是
    # 用错了地方（旧公式的"减自身面值"设计正是为了规避这个敏感性，代价
    # 是缺一部分 dilution；不能两者都要）。已撤销，保留下面的原始形式，
    # 只保留 owner/neighbor 跳变量分别独立计算这一个改动（见上方修复
    # 说明），未继续尝试消除"缺失 dilution"这个更深的架构缺口。
    delta_phi_owner = phi_upwind - phi_owner_fp  # (n_faces, n_fp)
    delta_phi_neighbor = phi_upwind - phi_neighbor_fp  # (n_faces, n_fp)
    del phi_upwind, phi_owner_fp, phi_neighbor_fp
    # 面元幅值因子（真实修复，2026-08-25 代码审查）：上面用单位法向算出的是物理
    # 通量密度差，而平均流无粘/粘性界面项送进同一套分配链路的跳越量都是协变
    # 通量（物理通量 × |adj_row|，含面元幅值）：inviscid_kernel.py L197
    # `F_common_n * adj_mag`（adj_mag 归一化只用于方向对齐检查，幅值随后乘回）、
    # viscous_flux_kernel.py L182 `adjrow_o · G`。缺这个 ~O(h²) 因子会把校正放大
    # ~1/h²（细网格 10²~10³ 倍）。true_normal 是单位向量（见
    # face_flux_points/exact_normal.py），必须补回 |adj_row|。
    #
    # native 四面体（路径C）真实 bug 修复（2026-08-30，见
    # transport_kernel.py::distribute_corrections_to_cells_kernel 文档）：
    # 此前这里统一用 |owner_adj_row_exact| 当面元幅值因子——但 native 面
    # 需要的是真实物理面积权重 `true_area_weight`（DG 提升算子弱形式积分
    # 用的量），与 collapsed 面的度量张量 adj 行模长不是同一个量，不能
    # 在这里统一预乘后再传给下游。改为只传"未加权"的 `mass_flux*delta_phi`，
    # 加权方式按面类型分派下沉到 `_distribute_correction_to_cells`/
    # kernel 内部。
    raw_jump_fp = mass_flux * delta_phi_owner  # (n_faces, n_fp)，owner 侧未加权物理通量密度差
    raw_jump_fp_neighbor = mass_flux * delta_phi_neighbor  # neighbor 侧独立的未加权跳变量
    del mass_flux, delta_phi_owner, delta_phi_neighbor

    # 分配回 SPs
    interface_correction = _distribute_correction_to_cells(
        raw_jump_fp, flat, ops, mesh, raw_jump_fp_neighbor=raw_jump_fp_neighbor,
    )
    with np.errstate(over='ignore', invalid='ignore'):
        residual = residual + interface_correction

    return residual
