"""AutoFlowCFD V2.0 - 标量场（k/omega）滤波与湍流门控。

从 `core/fr_solver/filter.py` 拆出（2026-09-24）。纯搬家，逻辑未改。

与平均流滤波**分开**的理由：k/omega 只在每步的湍流源项之后施加一次，
而平均流要逐 RK stage 施加；门控开关也是独立的
（`AFCFD_FILTER_TURB_GATE`）。
"""

import os

import numpy as np

from .apply import _filter_scalar_kernel


def filter_scalar_field(phi: np.ndarray, n_prism: int, filter_prism, filter_tet) -> np.ndarray:
    """对湍流标量场（k 或 omega，形状 (n_cells, n_sps)）施加与平均流
    完全同一套模态滤波矩阵（真实 bug 修复，2026-09-12，cube_demo
    791,492 单元真实网格 P1 直连（无 Order Continuation）长程测试发现：
    全新、干净的 P1 启动在数十步内 omega 场大范围失控增长，8.6% 的单元
    omega 逼近 1e6 安全上限，对流残差量级达到 ~1e9~1e10（应为 O(1e2~
    1e4)），阻力系数 Cd 单调线性漂移到 4.9~16.6（正常方块应在 1~2.5），
    残差整体不收敛——决定性诊断定位到具体单元（如 153869）P1 多项式在
    单元内部相邻解点间出现数量级跳变（如同一单元 8 个解点里 2 个
    ~22 万、2 个恰好卡在 omega_inf 现实性下限 226.8、4 个恰好是某个
    壁面目标值 2252.6，物理上不可能这样分片跳跃），外插到面通量点后
    被上风格式放大成巨大的虚假对流残差，形成真实的复合增长——这正是
    `_filter_flat_U` 文档所述"坍缩坐标节点配置法对高阶模态混叠噪声
    天然敏感"的同一种病理，只是发生在 k/omega 而不是平均流守恒变量上。

    此前 k/omega 被排除在滤波之外的理由（"湍流方程走独立单步显式更新，
    不经过多级 RK 残差重算，因此不会积累混叠"）经真实数据证伪：混叠
    振铃是*空间*离散在陡峭梯度区的固有行为（Hesthaven & Warburton
    2008 §5.3；Boyd 2001 Ch.11 关于坍缩坐标/配置点法高阶混叠的标准
    结论，对任何标量场都成立，不区分它用几级龙格库塔推进），"单步
    显式"只是少了多级 RK residual 重新求值这一条*额外*的放大途径，
    从未意味着完全免疫——真实数据已经证明并非如此。

    与平均流滤波用的是完全相同的 `ops.filter_prism`/`ops.filter_tet`
    矩阵（对常数场恒等，P0 下 n_sps=1 时矩阵退化为 1x1 单位矩阵，
    天然是无操作，不需要额外的阶数判断分支）。

    Args:
        phi: (n_cells, n_sps) 标量场（k_field 或 omega_field）
        n_prism: 棱柱单元数（前 n_prism 个单元用 filter_prism）
        filter_prism, filter_tet: 与平均流共用的同一套模态滤波矩阵

    Returns:
        滤波后的标量场，形状不变
    """
    n_cells = phi.shape[0]
    out = phi.copy()
    # 同上并行 kernel（机器精度等价，见 `_filter_scalar_kernel` 文档）
    if n_prism > 0:
        _filter_scalar_kernel(np.ascontiguousarray(phi[:n_prism]),
                              np.ascontiguousarray(filter_prism), out[:n_prism])
    if n_cells > n_prism:
        _filter_scalar_kernel(np.ascontiguousarray(phi[n_prism:]),
                              np.ascontiguousarray(filter_tet), out[n_prism:])
    return out


def filter_scalar_field_gated(
    phi: np.ndarray, filter_prism, filter_tet, troubled: np.ndarray,
    *, n_prism=None, cell_is_prism=None,
) -> np.ndarray:
    """`filter_scalar_field` 的逐单元门控版：只对 `troubled` 为真的单元
    施加滤波矩阵，其余单元逐位原样返回。

    矩阵与全局版完全相同，唯一区别是"对哪些单元施加"——`troubled` 全 True
    时结果与 `filter_scalar_field` 逐位一致（同一个 kernel、同一个矩阵、
    同样的分组顺序）。
    """
    if (n_prism is None) == (cell_is_prism is None):
        raise ValueError("n_prism 与 cell_is_prism 必须且只能给一个")
    n_cells = phi.shape[0]
    out = phi.copy()
    if not np.any(troubled):
        return out
    if cell_is_prism is not None:
        cip = np.asarray(cell_is_prism, dtype=bool)
        groups = ((np.flatnonzero(cip), filter_prism),
                  (np.flatnonzero(~cip), filter_tet))
    else:
        groups = ((np.arange(0, n_prism), filter_prism),
                  (np.arange(n_prism, n_cells), filter_tet))
    for sel_all, mat in groups:
        sel = sel_all[troubled[sel_all]]
        if sel.size == 0:
            continue
        sub_out = np.ascontiguousarray(phi[sel])
        _filter_scalar_kernel(np.ascontiguousarray(phi[sel]),
                              np.ascontiguousarray(mat), sub_out)
        out[sel] = sub_out
    return out


#: k/omega 滤波的门控方式，由 `AFCFD_FILTER_TURB_GATE` 选择：
#:   "all"（默认，与此前行为逐位一致）—— 所有单元都滤波
#:   "sensor"                         —— 只对传感器判定欠分辨的单元滤波
#:
#: **为什么这一维必须与 `AFCFD_FILTER_MODE` 独立**（2026-09-15 实测结论）：
#: `filter_scalar_field` 直接用 `ops.filter_prism`、完全不经过平均流那边
#: 的传感器门控，所以 `AFCFD_FILTER_MODE=sensor` 实际的语义一直是
#: "平均流门控 + k/omega 仍被完整清掉一整阶"。79 万单元真实网格 250 步
#: 对照决定性分离出了这一点：
#:
#:   档       平均流       k/omega     om_max 轨迹（起始 1.63e4）
#:   legacy   全局清零     全局清零    受控
#:   off      不滤波       **不滤波**  step100 达 1.65e5，增速持续加速
#:   sensor   门控(≈不滤波) 全局清零    step140 才 7.32e4，增速持续减速
#:
#: off 与 sensor 的**平均流**轨迹几乎逐位相同（step100 残差都是 2.211e9、
#: Cd 3.0831 vs 3.0830），说明传感器在平均流上几乎不触发；两者 om_max 的
#: 巨大差异**全部**来自 k/omega 那一维。也就是说 sensor 档的好处与"传感器
#: 在平均流上起作用"无关，而是来自 k/omega 仍被滤波——这正是
#: `filter_scalar_field` 文档记录的 2026-09-12 真实 P1 发散所需要的保护。
#:
#: 既然两维的效果可以完全分离，就不能再让一个环境变量同时决定它们。
def resolve_turb_filter_gate() -> str:
    """返回 k/omega 滤波的门控方式，并校验取值。

    每次调用都重读环境变量（不缓存模块级常量）：与
    `AFCFD_FILTER_MODE` 不同，这一维不影响算子构造，运行期读取是安全的，
    而且让测试可以用 monkeypatch 切换而不必 reload 模块。

    已接线的后端：**全部四条**——单机 CPU 与 CPU MPI 分布式
    （都经 `fr_solver/turbulence.py::compute_turbulence_source`，后者的
    `turb_view`/`mesh_adapter` 都在 compact"棱柱在前"索引空间，所以同一段
    `n_prism` 切片代码两条路径都正确）、单 GPU（`gpu_solver_io.py`）、
    多 GPU 分布式（`gpu_distributed_init.py`，后两条用
    `core/gpu/gpu_troubled_cell.py` 的 GPU 版同一套传感器，2026-09-15
    补齐）。所以这一维**不需要** `resolve_filter_mode` 那样的后端白名单
    ——平均流的 `sensor` 档要在 RK stage 内部逐 stage 求指标，k/omega 这
    一维只在每步湍流源项之后施加一次，补齐成本低得多。
    """
    gate = os.environ.get("AFCFD_FILTER_TURB_GATE", "all").lower()
    if gate not in ("all", "sensor"):
        raise ValueError(
            f"AFCFD_FILTER_TURB_GATE={gate!r} 不是合法取值（all | sensor）。"
            f"'all' 是既有行为（所有单元都滤波），'sensor' 只对传感器判定"
            f"欠分辨的单元滤波。注意 AFCFD_FILTER_MODE=off 会把滤波矩阵本身"
            f"变成单位阵，此时这一维无论取什么都是无操作。")
    return gate


def compute_turb_troubled_mask(k_field: np.ndarray, omega_field: np.ndarray,
                               order: int, *, n_prism=None,
                               cell_is_prism=None) -> np.ndarray:
    """k/omega 门控用的欠分辨掩码：对 k 与 omega **分别**求
    Persson-Peraire 指示器后取并集。

    为什么取并集而不是只看一个：2026-09-12 记录的真实发散发生在 omega
    （单元内部相邻解点间数量级跳变），而更早一轮攻关里失控的是 k（局部
    单元撞 k_max 上限，见 cube_demo omega realizability 那条记录）。两个
    场各自都会混叠，任一出问题都需要该单元被滤波，所以取并集而不是交集。

    为什么不复用平均流的掩码：一个单元完全可以密度光滑而 omega 有尖峰
    （实测 off/sensor 两档平均流轨迹几乎逐位相同、om_max 却差一个量级，
    就是这件事的直接证据）。
    """
    from autoflowcfd.core.fr_operators.artificial_viscosity import (
        compute_troubled_cell_mask,
    )
    kw = dict(n_prism=n_prism, cell_is_prism=cell_is_prism)
    mask_k = compute_troubled_cell_mask(np.ascontiguousarray(k_field), order, **kw)
    mask_om = compute_troubled_cell_mask(np.ascontiguousarray(omega_field), order, **kw)
    return mask_k | mask_om
