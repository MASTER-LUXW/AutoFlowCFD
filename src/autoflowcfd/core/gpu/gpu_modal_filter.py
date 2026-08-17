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
