"""
AutoFlowCFD V2.0 - 物理空间梯度计算（度量项一致，修复版）

此前 fr_residual_viscous.py 的 compute_gradients / compute_scalar_gradient
把计算立方体微分算子 D_3d（给出 ∂φ/∂ξ_m，即相对计算坐标 a,b,c 的导数）
直接当作物理空间导数 ∂φ/∂x_n 使用，没有做任何度量项变换。这对笛卡尔
张量积（未曲变的规则六面体）单元恰好凑巧正确，但对本代码库中的每一个
四面体/棱柱单元（曲边/坍缩坐标映射，见 grid/curved_mapping.py）都是
错误的导数——物理正确的链式法则是：

    ∂φ/∂x_n = Σ_m (∂ξ_m/∂x_n) * ∂φ/∂ξ_m = Σ_m inv_jac[m,n] * (D_3d[:,:,m] @ φ)

已用线性函数解析解验证（对任意非退化四面体/棱柱，线性物理函数的梯度
应精确恢复为其真实常数梯度，验证误差在机器精度量级，见
tests/unit/test_fr_gradients.py）。
"""

import numpy as np

from autoflowcfd.core.fr_operators.volume_contract import (
    contract_shared_operator_1axis, grad_computational_to_physical,
)


def compute_physical_gradient(field: np.ndarray, mesh, ops) -> np.ndarray:
    """计算场变量在物理空间中的梯度，正确处理曲边/坍缩坐标度量项。

    Args:
        field: 形状 (n_cells, n_sps, n_field_vars)，SPs 上的场值
        mesh: HighOrderMesh 实例（需要 jacobians['inv_jacs']）
        ops: FROperators（需要 D_3d）

    Returns:
        grad: 形状 (n_cells, n_sps, n_field_vars, 3)，物理空间梯度
    """
    n_cells, n_sps, n_field_vars = field.shape
    inv_jacs = mesh.jacobians["inv_jacs"].reshape(n_cells, n_sps, 3, 3)

    # 四面体/棱柱专用坍缩坐标微分矩阵（不能用朴素张量积 D_3d，理由见
    # fr/operators.py::FROperators.D_3d_tet/D_3d_prism 文档）。
    #
    # 性能优化：D（形状 (s,j,m)）不依赖 cell，只有 field（c,j,v）依赖
    # cell——把 D 的 (s,m) 两个输出轴摊平、j 挪到最后一维，转成
    # `contract_shared_operator_1axis` 认识的 (S*M, J) 2D 形状，收缩后
    # 再 reshape 回 (c,s,m,v)；数学上与 `np.einsum("sjm,cjv->csmv", D,
    # field)` 严格等价（同一个求和，只是换一条 BLAS gemm 计算路径），
    # 原因/验证方式见 fr_volume_contract.py 模块文档——生产网格上
    # `compute_physical_gradient` 的 einsum 是体积项性能优化里的另一个
    # 主要热点（py-spy 采样证实）。
    # 链式法则转物理空间：grad_phys[c,s,v,n] = sum_m inv_jac[c,s,m,n] * grad_comp[c,s,m,v]
    # 输出维度顺序 (n_cells,n_sps,n_field_vars,3)，与代码库既有 grad_U 约定一致。
    # 计算用 `grad_computational_to_physical` 融合 kernel（性能优化
    # 2026-09-13，见 volume_contract.py 同名函数文档：原
    # `np.matmul(np.swapaxes(grad_comp,-1,-2), inv_jacs)` 要先物化一份
    # 非连续转置副本、再做数百万次 (V,3)@(3,3) 微型 gemm，开销主导且
    # 不随核数并行）。对 m 求和顺序与 np.matmul 一致，结果逐位相同。
    #
    # **按单元分块**（内存优化 2026-09-14，为 P2 OOM 排查新增）：本函数
    # 此前先为全场物化 `grad_comp`（(n_cells,n_sps,3,V)），再整体转物理
    # 空间，于是 `grad_comp` + 收缩临时结果 + 输出 `grad_phys` 三份大数组
    # 同时存活。79 万单元 P2（n_sps=27）、V=5 时每份约 2.56GiB，实测峰值
    # 约 7.2GiB——而这条链每一步都是**逐单元独立**的（收缩只在单元内的
    # SPs 之间、链式法则只在同一 (cell,SP) 上），完全可以切块：现在只有
    # 输出是全场数组，块内临时量按 32768 单元计只有约 106MiB×2。P2/V=5
    # 下峰值从约 7.2GiB 降到约 2.8GiB（省约 4.4GiB）。分块不改变任何
    # 逐单元的计算顺序与形状，结果与全场版**逐位相同**（见
    # tests/unit/test_fr_gradients.py 与
    # tests/unit/test_perf_fusion_kernels.py 的等价性用例）。
    # prism/tet 两段分开切块：两者用不同的微分矩阵，且单元存储本来就是
    # prism 在前 tet 在后，块不会跨类型。
    n_prism = mesh.n_prism_cells
    grad_phys = np.empty((n_cells, n_sps, n_field_vars, 3))
    D2_prism = None
    if n_prism > 0:
        D2_prism = np.ascontiguousarray(
            np.transpose(ops.D_3d_prism, (0, 2, 1))).reshape(n_sps * 3, n_sps)
    # native 四面体（路径C）：`D_3d_tet`（坍缩坐标专属微分矩阵）对
    # native 单纯形基节点毫无意义（native 节点不是坍缩坐标张量积
    # 采样点），必须改用已经零填充到全局 n_sps 宽度的
    # `D_native_tet_padded`（Part8 文档"零填充块对角"不变量：填充行
    # 的散度贡献恒为 0，与这里"物理梯度"用途——同样是对体积节点场
    # 求导——完全兼容，不需要额外处理）。这是 Part8 native 支持范围
    # 此前遗漏的一处：`compute_physical_gradient` 是粘性残差
    # （viscous_flux.py）以及 SST/DES 湍流输运（transport.py）梯度
    # 计算共用的唯一入口，此前一直无条件读取 `D_3d_tet`，对 native
    # 网格会产生完全错误的梯度（用坍缩坐标基函数的导数系数去解释
    # native 节点上的场值）。
    D2_tet = None
    if n_cells > n_prism:
        tet_ops = (ops.D_native_tet_padded
                   if getattr(ops, "D_native_tet_padded", None) is not None else ops.D_3d_tet)
        D2_tet = np.ascontiguousarray(
            np.transpose(tet_ops, (0, 2, 1))).reshape(n_sps * 3, n_sps)

    _GRAD_CHUNK_CELLS = 32768
    for seg_lo, seg_hi, D2 in ((0, n_prism, D2_prism), (n_prism, n_cells, D2_tet)):
        if D2 is None or seg_hi <= seg_lo:
            continue
        for c0 in range(seg_lo, seg_hi, _GRAD_CHUNK_CELLS):
            c1 = min(c0 + _GRAD_CHUNK_CELLS, seg_hi)
            gc = contract_shared_operator_1axis(D2, field[c0:c1]).reshape(
                c1 - c0, n_sps, 3, n_field_vars)
            grad_phys[c0:c1] = grad_computational_to_physical(gc, inv_jacs[c0:c1])
            del gc  # 块内用完即弃，下一轮迭代变量重新绑定
    return grad_phys


def compute_physical_scalar_gradient(scalar_field: np.ndarray, mesh, ops) -> np.ndarray:
    """标量场版本（去掉 field_vars 维度的便捷包装）。

    Args:
        scalar_field: 形状 (n_cells, n_sps) 或 (n_cells, n_sps, 1)

    Returns:
        grad: 形状 (n_cells, n_sps, 3)

    修复记录：此前只在输入是 2D (n_cells,n_sps) 时才补一个 field_vars 轴再
    在最后挤掉；但唯一真实调用方（core/fr_solver_turbulence.py 计算
    SST/DDES 的 grad_k/grad_omega）传入的是已经手动加过 (n_cells,n_sps,1)
    这个轴的 3D 数组，导致这里的挤压条件从未触发，返回值多出一个不该有
    的轴 (n_cells,n_sps,1,3)——下游 turbulence_sst.py 的交叉扩散项
    `np.sum(grad_k*grad_omega, axis=2)` 会把这个多余的单位轴当成待求和的
    维度（axis=2 现在指向它而不是本该求和的空间维），真正的 3 分量空间
    向量完全没被点积掉；再与形状 (n_cells,n_sps) 的 rho/F1 等场相乘时，
    错位广播成 (n_cells,n_cells,3)（真实网格已复现该崩溃）。现在统一在
    末尾挤掉这个轴，与输入是 2D 还是"3D 且末轴为1"无关。
    """
    if scalar_field.ndim == 2:
        scalar_field = scalar_field[:, :, np.newaxis]
    grad = compute_physical_gradient(scalar_field, mesh, ops)  # (n_cells,n_sps,1,3)
    return grad[:, :, 0, :]
