"""AutoFlowCFD V2.0 - 原生（非坍缩）基过积分阶数的**共享**解析逻辑。

## 为什么单独一个模块

`fr/collapsed_basis.py::OVERINTEGRATION_MAX_ORDER = 3` 是**坍缩坐标模态
Vandermonde 的数值条件数极限**（cond 在 N=4 达约 1e14、`max|D|` 暴涨约
6.3 万倍，见该常量处的完整记录），不是去混叠本身的要求。原生基没有这个
问题，所以四面体（2026-09-17）和棱柱（2026-09-19）各自有**自己的**上限
环境变量。

两者的 env 解析与"`rule*order` 取 min 再对 order 取 max"这套规则逐字
相同。此前只有四面体一份，写在 `native_tet/overintegration.py` 里；棱柱
接入时把它抄第二遍是本项目明确禁止的（同一语义两份实现，改一份漏一份
已经出过真实缺陷——滤波档双解析器、CFL 三处硬编码兜底）。所以这里做成
参数化的唯一实现，两个基各自只保留一行"传自己的 env 名与默认值"的薄
包装。

## 实测依据（为什么原生基可以不设额外上限）

    基          阶数  cond(V)    max|D|
    坍缩棱柱    oo=3  6.19e+03    560.1
    坍缩棱柱    oo=4  4.72e+05  33928.8   <- 条件数崩掉的地方
    原生棱柱    oo=4  6.33e+01      7.7
    原生棱柱    oo=8  1.86e+03     24.7   <- 比坍缩 oo=4 还好 250 倍
    原生四面体  oo=6  3.86e+03      -     （见 native_tet_overintegration）

而去混叠精度在 `oo = 2*order` 处断崖式下降（四面体 P3: oo=5 3.37e-3 vs
oo=6 4.80e-6，18600 倍），所以"不额外设限、让 `rule*order` 说话"是两个
原生基共同的正确取法。
"""

import os

#: 常识护栏上界：`n_fine` 随阶数三次增长、体积项收缩是 O(n_fine^2)，
#: 超过这个值不是数值失效而是没有实用意义。两个原生基共用同一个护栏。
NATIVE_OVERINT_SANITY_MAX = 8


def resolve_overintegration_order_rule() -> int:
    """过积分阶数规则的倍率：`AFCFD_OVERINT_ORDER_RULE = 2x | 3x`，默认 `2x`。

    实际 `over_order = min(rule * order, 该基自己的上限)`，见本模块的
    `resolve_native_overintegration_order`。

    **为什么是可切换的而不是直接改成 3x（2026-09-15）**：`3x` 在 order=1
    上带来很大的精度收益（细点 27 -> 64）——

        k/omega 对流  prism 1.360e-1 -> 2.124e-3（64x）  tet -> 机器零
        粘性体积项    prism 4.615e-3 -> 3.617e-6（1276x）tet -> 机器零

    ——而且自由流场保持性没有回归（order=1 相对残差 4.1101e-11 ->
    4.3839e-11，噪声级）。注意「order>=2 逐位不变」那半句只对已删除的坍缩棱柱基成立：
    原生基的上限是 6，P2 会被这条规则从 oo=4 抬到 oo=6。

    ## 真实网格实测已补齐（2026-09-17）：`3x` 在 P1 上买不到任何东西

    上面那组精度数字是在**合成场/隔离判据**上量的。真实网格 + SST 的
    对照（plate_coarse_volume，168,340 单元，P1+SST，同一初场，预热 4 步
    后计时 8 步，壁面距离场按 CLI 同一路径算好）：

        rule=2x   prism_oo=2 tet_oo=2 n_fine_tet=10   11.996 s/step  res 4.425745e+08
        rule=3x   prism_oo=3 tet_oo=3 n_fine_tet=20   18.967 s/step  res 4.425745e+08

    **残差 7 位有效数字完全相同，而每步贵 1.58 倍。** 也就是说 `oo = 2*order`
    对这条真实算例上的全部项——包括 k/omega 输运与含 mu_t 的粘性通量那两个
    三重乘积——都已经足够。同一结论在平板边界层算例上独立复现过：粘性
    体积项 `AFCFD_VISC_OVERINT=on` 时 `2x` 与 `3x` 的能量分量**逐位相同**
    （见 `core/fr_residual/viscous_flux.py::resolve_viscous_overintegration`）。

    （旧的代价估算——微基准 3.47s -> 14.30s、外推每步 ~1.6x——是在**细网格
    轴还被零填充**的代码上做的，那份填充已于 2026-09-17 去掉（P1 实测整链
    3.04x 加速），所以那个估算已不适用；现在这个 1.58x 是真实网格直接测的。）

    因此默认保持 `2x`，依据从"代价未量化所以保守"升级为"**收益实测为零、
    代价实测 1.58 倍**"。`3x` 保留为受控 A/B 的入口（order>=2 上是否仍然
    为零尚未测，那需要 P2 的真实网格运行）。
    """
    v = os.environ.get("AFCFD_OVERINT_ORDER_RULE", "2x").lower()
    if v not in ("2x", "3x"):
        raise ValueError(
            f"AFCFD_OVERINT_ORDER_RULE={v!r} 不是合法取值（2x | 3x）。"
            f"'2x' 是既有行为（为平均流的二次非线性设计），'3x' 针对标量"
            f"输运/粘性项里的三重乘积，见 resolve_overintegration_order_rule。")
    return 2 if v == "2x" else 3


def resolve_native_overintegration_max_order(env_name: str, default: int) -> int:
    """读 `env_name` 指定的环境变量，缺省返回 `default`。

    非法取值直接报错、不静默回退——静默回退会让一次拼写错误伪装成默认
    行为，把 A/B 的两条运行悄悄变成同一档（同一原则见
    `fr_operators/kernels.py::resolve_ausm_precond_mode`；本项目已经因为
    "两条'不同 CFL'给出逐位相同轨迹"吃过一次亏）。
    """
    raw = os.environ.get(env_name, "").strip()
    if not raw:
        return int(default)
    try:
        v = int(raw)
    except ValueError:
        raise ValueError(f"{env_name}={raw!r} 不是整数") from None
    if not (1 <= v <= NATIVE_OVERINT_SANITY_MAX):
        raise ValueError(
            f"{env_name}={v} 超出 [1, {NATIVE_OVERINT_SANITY_MAX}]。上界"
            f"不是数值极限而是常识护栏：n_fine 随阶数三次增长、体积项"
            f"收缩是 O(n_fine^2)")
    return v


def resolve_native_overintegration_order(
    order: int, env_name: str, default_max: int
) -> int:
    """定某个原生基实际使用的 `over_order`。

    两条约束：
      1. 去混叠经验法则 `rule * order`（**与基无关**，见
         `resolve_overintegration_order_rule`，本模块）；
      2. 该基自己的上限（`env_name`，默认 `default_max`）。

    Returns:
        实际 `over_order`（>= order；等于 order 时过积分退化为恒等）。
    """
    oo = min(resolve_overintegration_order_rule() * int(order),
             resolve_native_overintegration_max_order(env_name, default_max))
    return max(oo, int(order))


# ---------------------------------------------------------------------------
# 棱柱：跨基档的唯一入口
# ---------------------------------------------------------------------------
#
# 棱柱的 `over_order` 与"细点数"此前在两个文件里各算一遍同一个式子
# （`fr/operators.py::generate_fr_operators` 与 `grid/high_order/
# high_order_mesh_order.py::build_order_geometry`），两处的注释里都写着
# "必须与另一处逐字一致，否则算子形状与 mesh.jacobians_fine 对不上"。
# 那种"靠注释维持一致"的写法在本项目已经出过真实缺陷（滤波档双解析器、
# CFL 三处硬编码兜底），而且现在还要再叠一层原生/坍缩分档 —— 所以合并成
# 下面这两个函数，两个调用点都改成引用它们。

#: 原生棱柱过积分阶数上限，含义是"不再额外设限"（实际阶数由
#: `rule*order` 决定）。`fr/collapsed_basis.py::OVERINTEGRATION_MAX_ORDER
#: = 3` 是**坍缩坐标** Vandermonde 的条件数极限，原生基实测 oo=8 的
#: cond(V)=1.86e3 比坍缩 oo=4 的 4.72e5 还小 250 倍（见模块文档表格），
#: 没有理由继承它。与四面体取同一个值是因为是同一个结论，不是抄默认值
#: —— 两个基各有自己的 env，可以独立调。
NATIVE_PRISM_OVERINTEGRATION_MAX_ORDER = 6

#: 原生棱柱过积分上限的环境变量名（唯一事实来源，别处只引用这个常量）。
NATIVE_PRISM_OVERINT_MAX_ORDER_ENV = "AFCFD_PRISM_OVERINT_MAX_ORDER"


def resolve_prism_overintegration_max_order() -> int:
    """原生档下棱柱过积分阶数上限（读
    `AFCFD_PRISM_OVERINT_MAX_ORDER`，默认
    `NATIVE_PRISM_OVERINTEGRATION_MAX_ORDER`）。"""
    return resolve_native_overintegration_max_order(
        NATIVE_PRISM_OVERINT_MAX_ORDER_ENV,
        NATIVE_PRISM_OVERINTEGRATION_MAX_ORDER)


def resolve_prism_overintegration_order(order: int) -> int:
    """棱柱实际使用的 `over_order`，**按当前生效的棱柱基分档**。

    `min(rule*order, AFCFD_PRISM_OVERINT_MAX_ORDER)`，默认上限 6 = 不额外
    设限，所以 P1/P2/P3 拿到理想的 2/4/6。

    **不再有 `OVERINTEGRATION_MAX_ORDER = 3` 那条上限**（坍缩棱柱基已于
    2026-09-23 删除）：那是坍缩坐标模态 Vandermonde 的条件数极限（放宽到
    4 会让 P2 均匀自由流残差从 1.06e-5 恶化到 5.6e-3），原生基没有这个
    问题 —— 实测 oo=8 时 cond(V)=1.86e+03，比坍缩 oo=4 的 4.72e+05 还
    好 250 倍。

    `order == 0` 时返回 0（P0 走独立的有限体积残差路径，没有可去混叠的
    内容）。
    """
    order = int(order)
    if order <= 0:
        return 0
    # 棱柱恒为原生基（坍缩棱柱基已于 2026-09-23 删除），所以不再有
    # `OVERINTEGRATION_MAX_ORDER = 3` 那条上限 —— 那是**坍缩坐标模态
    # Vandermonde 的条件数极限**，原生基没有这个问题（实测 oo=8 时
    # cond(V)=1.86e+03，比坍缩 oo=4 的 4.72e+05 还好 250 倍，见模块文档）。
    return resolve_native_overintegration_order(
        order, NATIVE_PRISM_OVERINT_MAX_ORDER_ENV,
        NATIVE_PRISM_OVERINTEGRATION_MAX_ORDER)


def prism_n_fine(over_order: int) -> int:
    """棱柱在 `over_order` 下的细点数 `(oo+1)^2 (oo+2)/2`。

    这个值同时是 `mesh.n_sps_per_cell_fine`（`jacobians_fine` 的每单元
    布局宽度）。原生棱柱的细点是 Warp&Blend 三角形点 x Gauss 挤出点，
    **不填充到立方体宽度 `(oo+1)^3`** —— 填充槽位恒为零、零贡献，却让整条
    过积分链在空点上白算，而 `D_fine` 的收缩是 O(n_fine^2)。

    四面体段不受这个宽度约束：它的细点度量是逐单元常数、只取第 0 列广播
    （见 `core/fr_operators/volume_contract.get_overintegration_context`）。
    """
    from .native_prism.basis import native_prism_n_sps
    oo = int(over_order)
    if oo <= 0:
        return 0
    return native_prism_n_sps(oo)
