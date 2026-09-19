"""AutoFlowCFD V2.0 - 原生基算子零填充到全局统一 SPs 宽度，及"只统计真实
自由度"的归约。

从 `native_simplex_basis.py` 拆出（控制单文件行数，>400 行需拆分的项目
规范）。完整架构设计见
`ProjectFiles/V2.0/8_算法重构-微分算子对坍缩坐标退化参考轴的病态条件数-Part8.md`
"一、核心不变量：零填充块对角"一节——本文件只提供该不变量要求的、单一
的填充实现，供 `fr/operators.py::generate_fr_operators` 消费，避免各处
各写一份、未来不一致。

**2026-09-18 起同时服务原生棱柱基**（`AFCFD_PRISM_BASIS=native`，见
`native_prism/mode.py`）。填充函数本身与单元类型无关（只是把矩阵某些轴补到
全局宽度），原先名字里带 `tet` 纯属历史，已正名；真正与单元类型绑定的只有
"真实自由度个数"，见 `real_sps_per_cell`。

**这一点是引入原生棱柱时必须同时改的**：本文件的
`reduce_rows_over_real_sps`/`reduce_per_cell_over_real_sps` 原先假设
**棱柱用满全部槽位**（只对四面体块切 `:n_native`）。原生棱柱一上线棱柱
也有填充槽位，那批消费点（checkpoint 的单元均值、人工粘性的尺度因子、
omega 壁面目标值里的 rho、GPU 局部 dt 的 min）会**静默算错**——填充槽位
冻结在初值、随推进变馊（实测 10 步后偏差 3.4%）。所以两个归约现在都按
`real_sps_per_cell` 给出的**两个**真实长度分别切片。
"""

from typing import Tuple

import numpy as np


def pad_native_matrix_to_global(matrix: np.ndarray, n_sps: int, pad_axes: tuple) -> np.ndarray:
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
                f"pad_native_matrix_to_global: 待填充的轴长度不一致（axes={pad_axes}，"
                f"shape={matrix.shape}）——同一个矩阵里被要求填充的几个轴理应是同一个"
                f"n_native_sps，不一致说明调用方传错了矩阵或轴下标。"
            )
    if n_sps < n_native:
        raise ValueError(
            f"pad_native_matrix_to_global: n_sps={n_sps} 小于 n_native={n_native}——"
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


def pad_native_filter_matrix_to_global(filter_matrix: np.ndarray, n_sps: int) -> np.ndarray:
    """滤波矩阵专用填充——与 `pad_native_matrix_to_global` 用途不同，
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
            f"pad_native_filter_matrix_to_global: 滤波矩阵必须是方阵，实际 {filter_matrix.shape}"
        )
    if n_sps < n_native:
        raise ValueError(
            f"pad_native_filter_matrix_to_global: n_sps={n_sps} 小于 n_native={n_native}"
        )
    padded = np.eye(n_sps, dtype=filter_matrix.dtype)
    padded[:n_native, :n_native] = filter_matrix
    padded[:n_native, n_native:] = 0.0
    padded[n_native:, :n_native] = 0.0
    return padded


def native_tet_n_real_sps(order: int) -> int:
    """native 四面体的**真实自由度**个数 `(p+1)(p+2)(p+3)/6`。

    其余槽位是零填充（见 `pad_native_matrix_to_global` 的"零填充块
    对角"不变量）：它们不贡献任何真实输出，但**值本身会变馊**——按
    `high_order_mesh_order.py::build_order_geometry` 的约定它们在初始化时
    复制真实 SP #0，之后残差行填零、滤波行是单位阵，于是永远冻结在初值。
    实测 79 万单元合成算例推进 10 步后，填充块与真实 SP#0 已相差 3.4%。
    """
    return (order + 1) * (order + 2) * (order + 3) // 6


def native_prism_n_real_sps(order: int) -> int:
    """原生棱柱的**真实自由度**个数 `(p+1)^2 (p+2)/2`。

    公式**读** `fr/native_prism/basis.py::native_prism_n_sps`，不在这里
    再写一遍——同一语义只允许一个事实来源（自由度数同时决定节点生成、
    Vandermonde 尺寸、填充布局与这里的归约切片，任何一处与其它处不一致
    都会静默算错）。
    """
    from .native_prism.basis import native_prism_n_sps

    return native_prism_n_sps(order)


def real_sps_per_cell(order: int) -> Tuple[int, int]:
    """`(棱柱真实自由度, 四面体真实自由度)`，按**当前生效的棱柱基**。

    全局统一 SPs 宽度恒为 `(p+1)^3`；这两个数是那个宽度里真正携带自由度
    的前缀长度，其余是零填充。

      * 坍缩棱柱（迁移期默认）：棱柱**用满** `(p+1)^3`，没有填充；
      * 原生棱柱：棱柱只用前 `(p+1)^2(p+2)/2` 个。

    四面体恒为 `(p+1)(p+2)(p+3)/6`（native 是唯一实现）。

    这是"哪些槽位是真的"的**唯一判据来源**：归约、checkpoint 的单元均值、
    人工粘性尺度、GPU 局部 dt 全都读它，任何一处自己判断都会在切换基的
    时候漏改。
    """
    from .native_prism.mode import prism_basis_is_native

    n_prism_real = (native_prism_n_real_sps(order) if prism_basis_is_native()
                    else (order + 1) ** 3)
    return n_prism_real, native_tet_n_real_sps(order)


def order_from_n_sps(n_sps: int) -> int:
    """从每单元解点数反解多项式阶数：`n_sps = (p+1)^3` => `p`。

    为什么优先用这个而不是 `solver.current_order`/`solver.order`：本项目
    的填充/真实自由度划分完全由**数组自身的 SP 轴长度**决定，而 solver
    上的阶数属性是另一条信息来源——两者在 Order Continuation 期间可以
    短暂不一致，而且不是所有调用点都拿得到一个完整的 solver（GPU 侧只有
    数组与 flat face，分布式 checkpoint 只有 gather 出来的全局数组，测试
    替身更是只有必需的几个属性）。从数组反解则**恒与被归约的数组自洽**。

    Raises:
        ValueError: `n_sps` 不是某个整数的立方——那说明调用方传进来的不是
            张量积解点布局的数组，不能猜。
    """
    p = int(round(float(n_sps) ** (1.0 / 3.0))) - 1
    for cand in (p - 1, p, p + 1):  # 立方根浮点误差的邻域
        if cand >= 0 and (cand + 1) ** 3 == n_sps:
            return cand
    raise ValueError(
        f"order_from_n_sps: n_sps={n_sps} 不是 (order+1)^3 形式，"
        f"无法反解阶数（调用方传进来的不是张量积解点布局的数组）")


def _check_order_matches_n_sps(order: int, n_sps: int, who: str) -> int:
    """确认传入的 `order` 与数组的 SP 轴长度自洽，并返回真实自由度个数。

    零填充的行数是 `n_sps - n_native`，两者都由**当前解阶数**决定
    (`n_sps=(p+1)^3`)。调用方通常从 solver 上取 `current_order`，而在
    Order Continuation 期间网格几何量的 `n_sps` 可以短暂地属于另一个
    阶数——那种情形下按错的 `n_native` 切片会静默地把真实解点当成填充
    丢掉（或反过来把填充当真实算进去），是一个无声的精度损失。本项目
    在跨阶数缓存上已经复现过同一类真实 bug（见
    `gpu_solver.py::_compute_local_time_step_gpu` 里 metric_flux_scale
    缓存那段注释），所以这里显式报错而不是静默继续。
    """
    if (order + 1) ** 3 != n_sps:
        raise ValueError(
            f"{who}: order={order} 对应 n_sps=(order+1)^3={(order + 1) ** 3}，"
            f"但数组的 SP 轴长度是 {n_sps}——阶数与数组不自洽，"
            f"不能静默按其中之一继续（会把真实解点当填充丢掉，或反之）")
    return native_tet_n_real_sps(order)


def reduce_rows_over_real_sps(field, row_is_prism, order: int, how: str,
                              xp=None):
    """`reduce_per_cell_over_real_sps` 的**逐行掩码**版。

    用于行不是"棱柱在前"排列的情形——典型是按 `owner_cell` 索引出来的
    逐面数组（`rho[owner_cells]`，每行的单元类型任意混合），这时按
    `n_prism` 切片没有意义。

    Args:
        field: `(n_rows, n_sps)`
        row_is_prism: `(n_rows,)` 布尔，True=该行对应棱柱单元
        order, how, xp: 同 `reduce_per_cell_over_real_sps`
    """
    if xp is None:
        import numpy as _np
        xp = _np
    if how not in ('mean', 'min', 'max', 'sum'):
        raise ValueError(f"reduce_rows_over_real_sps: how={how!r} 不支持")
    n_sps = field.shape[1]
    _check_order_matches_n_sps(order, n_sps, 'reduce_rows_over_real_sps')
    n_prism_real, n_tet_real = real_sps_per_cell(order)
    fn = getattr(xp, how)
    if n_prism_real >= n_sps and n_tet_real >= n_sps:
        return fn(field, axis=1)
    prism_val = (fn(field, axis=1) if n_prism_real >= n_sps
                 else fn(field[:, :n_prism_real], axis=1))
    tet_val = (fn(field, axis=1) if n_tet_real >= n_sps
               else fn(field[:, :n_tet_real], axis=1))
    return xp.where(row_is_prism, prism_val, tet_val)


def reduce_per_cell_over_real_sps(field, n_prism: int, order: int, how: str,
                                  xp=None):
    """对 `(n_cells, n_sps[, ...])` 的逐单元场在 **SP 轴**做归约，**只统计
    真实自由度**。

    **为什么必须有这个函数（2026-09-15 系统性审计）**：对 SP 轴直接做
    `.mean(axis=1)` / `.min(axis=1)` 会把 native 四面体的零填充槽位一起
    算进去，而那些槽位冻结在初值、会随推进变馊（见
    `native_tet_n_real_sps`）。审计中确认受影响的真实调用点：

    - `core/mpi/distributed_checkpoint.py`：写进检查点/后处理的**单元
      平均**被污染（填充占一半槽位、实测偏差 3.4%）；
    - `core/fr_operators/artificial_viscosity.py`：`rho_local`/`vel_local`
      （人工粘性上限的尺度因子）；
    - `core/turbulence/transport.py` 与 GPU 对应处：`rho_owner`
      （omega 壁面目标值里的 nu = mu/rho）；
    - `core/gpu/solver/gpu_solver.py` 与多 GPU 对应处：局部 dt 的
      `min(axis=1)`（会被填充槽位的馊状态过度限制）。

    注意 `mean` 与 `min/max` 的差别：把填充同步成 SP#0 的副本能让
    `min/max` 恰好正确（重复一个真实值不改变极值），但**修不了 `mean`**
    ——权重会变成"4 个真实 + 4 份 SP#0 副本"。所以统一按掩码处理，而不是
    去维护填充值。

    单元存储是"棱柱在前、四面体在后"（本项目的既有不变量），所以直接切片
    就够，不需要构造布尔掩码。**两段各按自己的真实长度切**（见
    `real_sps_per_cell`）：原生棱柱上线后棱柱段也有填充槽位。

    Args:
        field: `(n_cells, n_sps)` 或 `(n_cells, n_sps, V)`
        n_prism: 棱柱单元数（前 n_prism 个）
        order: 多项式阶数
        how: 'mean' | 'min' | 'max' | 'sum'
        xp: 数组模块（numpy 或 cupy）；None 时按 numpy

    Returns:
        `(n_cells,)` 或 `(n_cells, V)`
    """
    if xp is None:
        import numpy as _np
        xp = _np
    if how not in ('mean', 'min', 'max', 'sum'):
        raise ValueError(f"reduce_per_cell_over_real_sps: how={how!r} 不支持"
                         f"（mean | min | max | sum）")
    n_cells = field.shape[0]
    n_sps = field.shape[1]
    _check_order_matches_n_sps(
        order, n_sps, 'reduce_per_cell_over_real_sps')
    # 两个真实长度都从 `real_sps_per_cell` 取：坍缩棱柱是 (p+1)^3（用满、
    # 无填充），原生棱柱是 (p+1)^2(p+2)/2（有填充）。写死"棱柱用满"会在
    # 切换棱柱基时静默算错，见本模块文档。
    n_prism_real, n_tet_real = real_sps_per_cell(order)
    fn = getattr(xp, how)
    out_prism = (fn(field[:n_prism, :n_prism_real], axis=1)
                 if n_prism > 0 else None)
    out_tet = (fn(field[n_prism:, :n_tet_real], axis=1)
               if n_cells > n_prism else None)
    if out_prism is None:
        return out_tet
    if out_tet is None:
        return out_prism
    return xp.concatenate([out_prism, out_tet], axis=0)
