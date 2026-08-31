"""AutoFlowCFD V2.0 - native 四面体（路径C）算子零填充到全局统一 SPs 宽度。

从 `native_simplex_basis.py` 拆出（控制单文件行数，>400 行需拆分的项目
规范）。完整架构设计见
`ProjectFiles/V2.0/8_算法重构-微分算子对坍缩坐标退化参考轴的病态条件数-Part8.md`
"一、核心不变量：零填充块对角"一节——本文件只提供该不变量要求的、单一
的填充实现，供 `fr/operators.py::generate_fr_operators` 消费，避免各处
各写一份、未来不一致。
"""

import numpy as np


def pad_native_tet_matrix_to_global(matrix: np.ndarray, n_sps: int, pad_axes: tuple) -> np.ndarray:
    """把 native 四面体算子矩阵嵌入到全局统一宽度 `n_sps`（=`n1d**3`，
    棱柱/坍缩坐标四面体共用的张量积节点数）的零矩阵左上角，其余槽位
    （"填充行/列"）恒为零——Part8 文档"零填充块对角"不变量的具体实现：

    - 对体积->体积算子（如 `D_native_tet`，形状 `(n_native,n_native,3)`，
      前两个轴都需要填充）：`pad_axes=(0,1)`，产出 `(n_sps,n_sps,3)`，
      只有 `[:n_native,:n_native,:]` 块非零——"列填零"确保填充行里任何
      有限数值都不贡献真实输出，"行填零"确保填充槽位残差恒为零、不会
      被时间推进意外改写。
    - 对体积->面算子（如 `lift_native_tet`，形状 `(n_native,n_fp)`，
      只有输出（体积）轴需要填充）：`pad_axes=(0,)`，产出 `(n_sps,n_fp)`。
    - 对面->体积算子（如 `boundary_extrap_native_tet`，形状
      `(n_fp,n_native)`，只有输入（体积）轴需要填充）：`pad_axes=(1,)`，
      产出 `(n_fp,n_sps)`——**本函数目前未在 `FROperators` 里为
      `boundary_extrap_native_tet` 主动调用**：它的消费点
      （`build_cross_interp`/`_native_interp_matrix_nb`）已经各自按需
      现场填充到 `n_sps` 宽度（Part6/7 既有设计），这里仍然支持这个用法
      是为了让本函数覆盖三种矩阵形状的调用方保持同一个实现来源，供未来
      需要"预先"（而非现场）填充时直接复用，不需要再写一份。

    Args:
        matrix: 待填充矩阵，`pad_axes` 指定的轴当前长度必须是
            `n_native`（<=`n_sps`），未列出的轴（如 `D_native_tet` 的
            第三个"方向"轴）原样保留。
        n_sps: 全局统一宽度（`n1d**3`）。
        pad_axes: 需要从 `n_native` 填充到 `n_sps` 的轴下标元组。

    Returns:
        padded: 与 `matrix` 同 dtype、`pad_axes` 指定轴长度变为
            `n_sps`、其余槽位为 0 的新数组。
    """
    n_native = matrix.shape[pad_axes[0]]
    for ax in pad_axes:
        if matrix.shape[ax] != n_native:
            raise ValueError(
                f"pad_native_tet_matrix_to_global: 待填充的轴长度不一致（axes={pad_axes}，"
                f"shape={matrix.shape}）——同一个矩阵里被要求填充的几个轴理应是同一个"
                f"n_native_sps，不一致说明调用方传错了矩阵或轴下标。"
            )
    if n_sps < n_native:
        raise ValueError(
            f"pad_native_tet_matrix_to_global: n_sps={n_sps} 小于 n_native={n_native}——"
            "native 基自由度理应严格不多于全局统一宽度（(order+1)(order+2)(order+3)/6 "
            "<= (order+1)^3），出现相反的情形说明上游传错了参数，不应该静默截断。"
        )

    new_shape = list(matrix.shape)
    for ax in pad_axes:
        new_shape[ax] = n_sps
    padded = np.zeros(tuple(new_shape), dtype=matrix.dtype)

    slices = tuple(slice(0, n_native) if ax in pad_axes else slice(None) for ax in range(matrix.ndim))
    padded[slices] = matrix
    return padded


def pad_native_tet_filter_matrix_to_global(filter_matrix: np.ndarray, n_sps: int) -> np.ndarray:
    """滤波矩阵专用填充——与 `pad_native_tet_matrix_to_global` 用途不同，
    不能直接复用：滤波矩阵直接作用在 `U` 本身（`U_new = F @ U`，不是像
    `D_native_tet`/`lift_native_tet` 那样产出一份*残差贡献*）。如果填充
    槽位照搬"行填零"（`D`/`lift` 用的约定），滤波器每次调用都会把填充
    行的值**重置为 0**，而不是"冻结在初值"——density=0 会在下游
    `conserved_to_primitive` 触发 Part8 文档"一、核心不变量"第4点警告
    过的 0 除风险，不是安全的选择。

    正确的填充块是**单位矩阵**（不是零）：`padded[n_native:,n_native:]
    =I`，让填充行经过滤波器后原样不变（保持初始化时写入的有限占位值，
    见 `high_order_mesh_order.py::build_order_geometry` native 分支
    "填充行复制真实 SP #0"的约定），真实行与填充行之间的交叉块仍然是
    零（滤波器不应该把体积场的值和占位内容混在一起）。

    Args:
        filter_matrix: (n_native, n_native) native 滤波矩阵。
        n_sps: 全局统一宽度。

    Returns:
        padded: (n_sps, n_sps)，`[:n_native,:n_native]=filter_matrix`，
            `[n_native:,n_native:]=I`，其余为 0。
    """
    n_native = filter_matrix.shape[0]
    if filter_matrix.shape[1] != n_native:
        raise ValueError(
            f"pad_native_tet_filter_matrix_to_global: 滤波矩阵必须是方阵，实际 {filter_matrix.shape}"
        )
    if n_sps < n_native:
        raise ValueError(
            f"pad_native_tet_filter_matrix_to_global: n_sps={n_sps} 小于 n_native={n_native}"
        )
    padded = np.eye(n_sps, dtype=filter_matrix.dtype)
    padded[:n_native, :n_native] = filter_matrix
    padded[:n_native, n_native:] = 0.0
    padded[n_native:, :n_native] = 0.0
    return padded
