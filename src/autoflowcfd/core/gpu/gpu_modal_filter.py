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
) -> Callable:
    """构造 GPU 版滤波回调函数。

    Args:
        n_cells: 单元数
        n_sps: 每单元解点数
        n_prism: 棱柱单元数
        filter_prism: 棱柱滤波矩阵 (n_sps, n_sps) CuPy 数组
        filter_tet: 四面体滤波矩阵 (n_sps, n_sps) CuPy 数组
        device_id: GPU 设备 ID

    Returns:
        filter_func: 接受 CuPy 数组 (N, n_vars)，返回滤波后的同形状数组
    """
    cp = get_cupy()

    with cp.cuda.Device(device_id):
        # 确保滤波矩阵在正确的设备上
        if not hasattr(filter_prism, 'device'):
            filter_prism = cp.asarray(filter_prism)
        if not hasattr(filter_tet, 'device'):
            filter_tet = cp.asarray(filter_tet)

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
