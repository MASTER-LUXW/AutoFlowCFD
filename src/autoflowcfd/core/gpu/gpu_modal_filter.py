"""
AutoFlowCFD V2.0 - GPU 版模态滤波

与 core/fr_solver_filter.py 对应的 CuPy 版本。
在每个 RK stage 后对守恒变量施加谱滤波，抑制坍缩坐标节点配置法
固有的混叠噪声放大。

设计：
- 使用 CuPy einsum 替代 numpy einsum
- 只作用于前 5 个欧拉变量（湍流量不参与滤波）
- 与 CPU 版公式完全一致
"""

from typing import Callable, Optional

from autoflowcfd.core.gpu import get_cupy


def build_gpu_filter_func(
    n_cells: int,
    n_sps: int,
    n_prism: int,
    filter_prism,
    filter_tet,
    device_id: int = 0,
    cell_is_prism=None,
) -> Callable:
    """构造 GPU 版滤波回调函数（全局施加，非门控）。

    Args:
        n_cells: 单元数
        n_sps: 每单元解点数
        n_prism: 棱柱单元数。**只在"棱柱在前"排列下有效**（单 GPU 的
            全局编号满足这条约定）。分布式 local 排列不满足，必须走
            `cell_is_prism`。
        filter_prism: 棱柱滤波矩阵 (n_sps, n_sps) CuPy 数组
        filter_tet: 四面体滤波矩阵 (n_sps, n_sps) CuPy 数组
        device_id: GPU 设备 ID
        cell_is_prism: (n_cells,) 布尔，True=棱柱。给出时按掩码分派，
            忽略 `n_prism`。

            **为什么必须有这条路（2026-09-18 真实缺陷）**：多 GPU 分布式
            的 `U_gpu` 是 halo 交换的**原生**排列（local 在前、halo 在后，
            棱柱与四面体**交错**），而这里此前无条件按 `U[:n_prism]` 切片，
            且调用方传的是**全局** `mesh.n_prism_cells`。两处叠加的后果是：
            `n_prism > n_cells` 时切片被静默钳到 n_cells，于是**每个 local
            单元、包括四面体，都被施加了棱柱滤波矩阵**，不报任何错。
            四面体走的是 native PKD/Dubiner 基（带零填充槽位），与棱柱的
            张量积基完全不同，混用在数值上没有意义。
            与 CPU 分布式 `build_filter_func_by_cell_type` 同一处理由。

    Returns:
        filter_func: 接受 CuPy 数组 (N, n_vars)，返回滤波后的同形状数组

    Raises:
        ValueError: 未给 `cell_is_prism` 且 `n_prism` 不在 [0, n_cells]
            之内——见上面那条缺陷，不静默钳。
    """
    cp = get_cupy()

    with cp.cuda.Device(device_id):
        # 确保滤波矩阵在正确的设备上
        if not hasattr(filter_prism, 'device'):
            filter_prism = cp.asarray(filter_prism)
        if not hasattr(filter_tet, 'device'):
            filter_tet = cp.asarray(filter_tet)
        if cell_is_prism is not None:
            cip = cp.asarray(cell_is_prism).astype(bool)
            if cip.shape != (n_cells,):
                raise ValueError(
                    f"cell_is_prism 形状 {cip.shape} 与 n_cells={n_cells} 不符")
            cip3 = cip[:, None, None]
        else:
            if not 0 <= n_prism <= n_cells:
                raise ValueError(
                    f"n_prism={n_prism} 超出 [0, n_cells={n_cells}]。"
                    f"分布式 local 排列里棱柱与四面体交错、且棱柱数是本 "
                    f"rank 的局部量，必须用 cell_is_prism 而不是 n_prism。")
            cip3 = None

    def gpu_filter_func(U_flat):
        """对展平的守恒变量施加模态滤波。

        Args:
            U_flat: CuPy 数组 (n_cells * n_sps, n_vars)

        Returns:
            filtered_U: CuPy 数组 (n_cells * n_sps, n_vars)
        """
        cp = get_cupy()
        n_vars = U_flat.shape[1]

        # 重塑为 (n_cells, n_sps, n_vars)
        U = U_flat.reshape(n_cells, n_sps, n_vars)

        # 只滤波前 5 个欧拉变量
        if cip3 is not None:
            # 交错排列：两个矩阵都对全场算一遍再按类型选。理由与
            # `filter_scalar_field_gated_gpu` 同一条实测结论（设备上
            # 花式索引的 gather/scatter 开销高于多做一遍小矩阵乘）。
            lead = U[:, :, :5]
            U[:, :, :5] = cp.where(
                cip3,
                cp.einsum("sj,cjv->csv", filter_prism, lead),
                cp.einsum("sj,cjv->csv", filter_tet, lead),
            )
            return U.reshape(n_cells * n_sps, n_vars)
        if n_prism > 0:
            # prism: einsum("sj,cjv->csv", filter, U)
            U[:n_prism, :, :5] = cp.einsum(
                "sj,cjv->csv", filter_prism, U[:n_prism, :, :5]
            )
        if n_cells > n_prism:
            U[n_prism:, :, :5] = cp.einsum(
                "sj,cjv->csv", filter_tet, U[n_prism:, :, :5]
            )

        return U.reshape(n_cells * n_sps, n_vars)

    return gpu_filter_func


def filter_scalar_field_gpu(phi, n_prism: int, filter_prism, filter_tet):
    """GPU 版湍流标量场（k 或 omega）模态滤波，与 CPU 版
    `core/fr_solver/filter.py::filter_scalar_field` 完全同一套公式/矩阵
    （真实 bug 修复，2026-09-12，见 CPU 版文档完整推导：k/omega 此前
    不参与模态滤波，cube_demo 真实网格 P1 直连长程测试发现全新、干净的
    P1 启动会在数十步内 omega 大范围失控增长，根因是同一类"坍缩坐标
    节点配置法高阶模态混叠"病理）。

    CPU/GPU 一致性（本项目一贯要求，见 test_gpu_distributed_turbulence.py
    等大量既有交叉验证测试）：只给 CPU 侧加这个滤波、GPU 侧不加，会让
    两者从这里开始产生真实数值分歧——本函数与单 GPU
    （gpu_solver_io.py）、GPU 分布式（gpu_distributed_init.py）两条
    调用路径必须同步接入，缺一个都会破坏 CPU/GPU 交叉验证。

    Args:
        phi: (n_cells, n_sps) CuPy 数组
        n_prism: 棱柱单元数
        filter_prism, filter_tet: 与平均流共用的同一套 CuPy 滤波矩阵

    Returns:
        滤波后的标量场，形状不变
    """
    cp = get_cupy()
    n_cells = phi.shape[0]
    if not hasattr(filter_prism, 'device'):
        filter_prism = cp.asarray(filter_prism)
    if not hasattr(filter_tet, 'device'):
        filter_tet = cp.asarray(filter_tet)
    out = phi.copy()
    if n_prism > 0:
        out[:n_prism] = cp.einsum("sj,cj->cs", filter_prism, phi[:n_prism])
    if n_cells > n_prism:
        out[n_prism:] = cp.einsum("sj,cj->cs", filter_tet, phi[n_prism:])
    return out


def filter_scalar_field_gated_gpu(phi, n_prism: int, filter_prism, filter_tet,
                                  troubled):
    """`filter_scalar_field_gpu` 的逐单元门控版（`AFCFD_FILTER_TURB_GATE=
    sensor`）：只对 `troubled` 为真的单元施加滤波矩阵，其余单元逐位原样
    返回。

    与 CPU 版 `fr_solver/filter.py::filter_scalar_field_gated` 语义完全
    一致：矩阵相同、分组顺序相同，`troubled` 全 True 时结果与
    `filter_scalar_field_gpu` 逐位一致。

    实现上用 `cp.where` 做整场混合而不是花式索引（`out[sel] = ...`）：
    GPU 上布尔/整数索引会触发额外的 gather/scatter 与同步，而滤波矩阵
    乘法本身对全部单元算一遍的成本远低于此（(n_sps,n_sps) 的小矩阵，
    n_sps<=64），所以"全算再按掩码选"是更快且数值等价的形式——被选中
    的单元取滤波结果、未选中的取原值，与逐单元施加逐位相同。

    Args:
        phi: (n_cells, n_sps) CuPy 数组
        n_prism: 棱柱单元数（前 n_prism 个）
        filter_prism, filter_tet: 滤波矩阵
        troubled: (n_cells,) CuPy 布尔数组
    """
    cp = get_cupy()
    n_cells = phi.shape[0]
    if not hasattr(filter_prism, 'device'):
        filter_prism = cp.asarray(filter_prism)
    if not hasattr(filter_tet, 'device'):
        filter_tet = cp.asarray(filter_tet)
    out = phi.copy()
    if not bool(cp.any(troubled)):
        return out
    sel2 = troubled[:, cp.newaxis]
    if n_prism > 0:
        filt = cp.einsum("sj,cj->cs", filter_prism, phi[:n_prism])
        out[:n_prism] = cp.where(sel2[:n_prism], filt, phi[:n_prism])
    if n_cells > n_prism:
        filt = cp.einsum("sj,cj->cs", filter_tet, phi[n_prism:])
        out[n_prism:] = cp.where(sel2[n_prism:], filt, phi[n_prism:])
    return out
