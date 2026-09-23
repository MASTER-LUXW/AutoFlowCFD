"""AutoFlowCFD V2.0 - FR 算子数据容器 `FROperators`。

从 `fr/operators.py`（原 705 行）拆出（2026-09-19，项目"单文件不超
500 行"规范）：这里只有容器与它的取算子方法，构建流程在 `build.py`。
纯搬家，未改任何逻辑。

容器本身的设计背景（坍缩四面体基已删除、两类原生面算子的分派约定等）
见 `__init__.py` 的模块文档。
"""

import numpy as np
from typing import Dict
from dataclasses import dataclass


@dataclass
class FROperators:
    """
    FR 算子容器，存储预计算的所有矩阵。

    Attributes:
        D_1d: 一维微分矩阵，形状 (n_pts, n_pts)
        D_3d: 三维微分算子，形状 (n_pts^3, n_pts^3, 3)——朴素张量积 Lagrange
            微分矩阵，只对六面体（无坍缩坐标退化面）正确；四面体/棱柱的
            体积散度/梯度/几何 Jacobian 计算必须改用 D_3d_tet/D_3d_prism。
        D_3d_tet: 四面体体积微分矩阵，形状同 D_3d——**别名到
            `D_native_tet_padded`**（2026-09-03 起，collapsed 四面体基
            已删除，见模块文档），不再是独立构造的坍缩坐标矩阵。
        D_3d_prism: 棱柱专用体积微分矩阵，形状同 D_3d，用坍缩坐标模态基
            （fr/collapsed_basis.py）通过 Vandermonde 矩阵构造，SPs
            位置与 D_3d 完全相同（不改变点集，只改变"如何对这些点上的
            节点值求导"）——原因见该模块文档：朴素张量积基与坍缩坐标
            退化边（棱柱 b=+1）附近真实存在的高阶（度量项、通量的坍缩
            坐标依赖是有理式而非低阶多项式）结构不匹配，插值多项式在
            退化边附近的混叠误差会被同样在该处偏小的真实几何 Jacobian
            放大到灾难量级；坍缩坐标模态基内建与退化因子匹配的结构，
            能显著降低这一混叠误差（真实网格验证：某棱柱单元体积项
            残差从 3.15e-11 降到 6.53e-12，约 5 倍）。棱柱没有 native
            方案可换，这里的坍缩坐标构造不受本次删除影响。
        filter_tet: (n_sps,n_sps) 指数模态滤波矩阵——**别名到
            `filter_native_tet_padded`**（2026-09-03 起，不再是独立
            构造的坍缩坐标滤波器）。
        filter_prism: (n_sps,n_sps) 指数模态滤波矩阵（见
            fr/modal_filter.py 文档）——坍缩坐标节点配置法对高阶模态的
            混叠噪声天然敏感，重复微分（体积散度、粘性梯度+散度两次）
            会把这个噪声逐步放大，真实网格上复现过在几步显式时间推进内
            从机器精度噪声放大到 NaN；每个 RK 阶段结束后对解场施加一次
            滤波是标准谱/DG 方法对策，不影响已解析到的低阶物理精度。
    """
    D_1d: np.ndarray
    D_3d: np.ndarray
    D_3d_tet: np.ndarray = None
    D_3d_prism: np.ndarray = None
    filter_tet: np.ndarray = None
    filter_prism: np.ndarray = None
    # 体积项去混叠（over-integration，见 collapsed_basis.build_overintegration_operators
    # 文档）：order==0 时全部为 None（P0 走独立的有限体积残差路径，不需要）。
    #: 棱柱**实际**使用的过积分阶数（原名 `overint_order`，2026-09-19
    #: 改名对称化）。取值由 `fr/overintegration_order.py::
    #: resolve_prism_overintegration_order` 按当前棱柱基分档给出：坍缩档
    #: `min(rule*order, 3)`（P1/P2/P3 = 2/3/3），原生档不设额外上限
    #: （2/4/6）。order==0 时为 0。
    overint_order_prism: int = 0
    #: 棱柱过积分的**真实**细点数，同时是 `mesh.n_sps_per_cell_fine`
    #: （`jacobians_fine` 的每单元布局宽度）。坍缩档 `(oo+1)^3`、原生档
    #: `(oo+1)^2(oo+2)/2`，见 `overintegration_order.prism_n_fine`。
    overint_n_fine_prism: int = 0
    #: native 四面体过积分的**真实**细点数 `(oo+1)(oo+2)(oo+3)/6`。与
    #: 棱柱的宽度**无关**——细网格轴不填充，且四面体段的细点度量是逐单元
    #: 常数取第 0 列广播，见 `generate_fr_operators` 里那段说明与
    #: `volume_contract.get_overintegration_context`。
    overint_n_fine_tet: int = 0
    #: 四面体**实际**使用的过积分阶数（2026-09-17）。与
    #: `overint_order_prism` 可以不同：两者各走自己的基、各有自己的上限
    #: env，见 `native_tet.overintegration.
    #: NATIVE_TET_OVERINTEGRATION_MAX_ORDER`。order>=1 时 >0。
    overint_order_tet: int = 0
    overint_interp_c2f_tet: np.ndarray = None
    overint_interp_c2f_prism: np.ndarray = None
    overint_D_fine_tet: np.ndarray = None
    overint_D_fine_prism: np.ndarray = None
    overint_restrict_f2c_tet: np.ndarray = None
    overint_restrict_f2c_prism: np.ndarray = None
    # 四面体路径C（独立于坍缩坐标，见 fr/native_tet/basis.py 与
    # `8_算法重构-微分算子对坍缩坐标退化参考轴的病态条件数-Part6.md`
    # 阶段0/1）：2026-09-03 起恒为非 None（native 现在是四面体唯一
    # 实现，不再有 collapsed 分支可选，`tet_basis_mode` 字段保留仅为
    # 兼容仍在读取它当"是否 native"判据的旧调用点，取值恒为 "native"）。
    # D_native_tet 形状 (n_native_sps_tet, n_native_sps_tet, 3)，参考
    # 坐标是四面体自己的 (r,s,t) 单纯形，节点数一般少于 (order+1)^3。
    tet_basis_mode: str = "native"
    D_native_tet: np.ndarray = None
    ref_native_tet: np.ndarray = None
    n_native_sps_tet: int = None
    # native 四面体体积->自身面外插矩阵（Part7 阶段2设计文档"二·五"节+
    # `native_tet/basis.py::build_native_tet_boundary_extrap`），
    # 键是被排除的局部顶点 0~3。
    boundary_extrap_native_tet: Dict[int, np.ndarray] = None
    # native 四面体 DG 提升算子（`native_tet/basis.py::
    # build_native_tet_lift` 文档），把面通量跳跃提升成体积节点修正
    # 贡献——是坍缩坐标方案"1D Radau/VCJH 修正函数 + _distribute_point"
    # 对非张量积单纯形基的唯一正确推广（native 基没有"坍缩计算方向"，
    # 1D 修正函数沿某一轴分布这个概念不适用）。键同样是被排除的局部
    # 顶点 0~3。
    lift_native_tet: Dict[int, np.ndarray] = None
    # `D_native_tet`/`lift_native_tet` 零填充到全局统一 SPs 宽度 `n_sps`
    # 之后的版本（`fr/native_padding.py::pad_native_matrix_to_
    # global`，见 Part8 文档"一、核心不变量：零填充块对角"）——生产
    # 残差 kernel（`inviscid.py`/`inviscid_kernel.py` 等）要消费的是
    # 这两个已经填充好的版本，不是上面两个原始（n_native 宽）版本；
    # 保留原始版本是因为部分测试/诊断代码可能只关心真实自由度本身，
    # 不需要每次都从填充版本反推。`D_3d_tet`/`filter_tet` 字段（上方
    # dataclass 开头）现在就是这两者的别名，见模块文档。
    D_native_tet_padded: np.ndarray = None
    lift_native_tet_padded: Dict[int, np.ndarray] = None
    # native 四面体指数模态滤波器（`native_tet_filter.py::build_native_
    # tet_modal_filter`，抑制混叠失稳，见该模块与 fr/modal_filter.py
    # 文档），填充到全局 n_sps 宽度（`native_padding.py::pad_native_
    # tet_filter_matrix_to_global`——填充块是单位矩阵，不是零，与
    # D_native_tet_padded/lift_native_tet_padded 的"零填充"约定不同，
    # 见该函数文档）。
    filter_native_tet_padded: np.ndarray = None

    # ======================= 原生棱柱基（2026-09-18） =======================
    #
    # `AFCFD_PRISM_BASIS=native` 时构造并**别名到** `D_3d_prism`/
    # `filter_prism`（与四面体那套完全同一个做法），所以任何无条件读这两个
    # 旧字段名的消费点自动拿到原生结果。面算子（外插/提升）键是 0~4 的
    # `face_id`，与坍缩棱柱的 `(axis, side)` 之间的换算**只允许**走
    # `native_prism_face.cube_face_to_native_prism_face`。
    #
    # 坍缩棱柱基已于 2026-09-23 删除，所以这一整组**恒为非 None**；原先
    # 记录档位的那个字段随之删除（只有一条基，再记"当前是哪条"就是冗余）。
    # 删除前后的对照数据见 `fr/native_prism/mode.py` 的模块函数文档。
    D_native_prism: np.ndarray = None
    ref_native_prism: np.ndarray = None
    n_native_sps_prism: int = None
    #: `face_id (0~4)` -> `(n_fp, n_native_sps_prism)` 体积->面外插。
    boundary_extrap_native_prism: Dict[int, np.ndarray] = None
    #: `face_id (0~4)` -> `(n_native_sps_prism, n_fp)` DG 提升。
    lift_native_prism: Dict[int, np.ndarray] = None
    #: 上面两个零填充到全局统一宽度 `(order+1)^3` 之后的版本 —— 生产残差
    #: kernel 要消费的是这些，不是原始 `n_native` 宽的版本。
    D_native_prism_padded: np.ndarray = None
    lift_native_prism_padded: Dict[int, np.ndarray] = None
    filter_native_prism_padded: np.ndarray = None

    # ---- 原生面算子的唯一分派入口（2026-09-18）----
    #
    # 两类单元的原生面算子键不同（四面体是 excluded_vertex 0~3，棱柱是
    # face_id 0~4），而消费点拿到的是统一的 `cube_face_code`：
    #
    #     [6, 10)  -> native 四面体面，excluded_vertex = code - 6
    #     [10, 15) -> native 棱柱面，  face_id        = code - 10
    #
    # **必须走这两个方法**，不要在消费点自己判断：面 id 配错不会报错，
    # 只会静默地对某个面用错矩阵（与当年多 GPU"四面体拿到棱柱矩阵"完全
    # 同一类缺陷）。numba kernel 不能调方法，它们读的是
    # `face_kernels.build_flat_face_geometry` 按同一套规则**叠好**的
    # flat 数组（0~3 四面体、4~8 棱柱，按 `code - 6` 连续索引，于是那边
    # 所有 `code >= 6` / `code - 6` 的既有写法原样成立）。

    def native_face_extrap(self, cube_face_code: int) -> np.ndarray:
        """按 cube face code 取原生面的体积->面外插矩阵（**未填充**，
        形状 `(n_fp, n_native)`，`n_native` 随单元类型不同）。

        Raises:
            ValueError: 不是原生面编码，或对应的基没有启用。
        """
        return self._native_face_op(cube_face_code, lift=False)

    def native_face_lift(self, cube_face_code: int) -> np.ndarray:
        """按 cube face code 取原生面的 DG 提升矩阵（**未填充**，
        形状 `(n_native, n_fp)`）。"""
        return self._native_face_op(cube_face_code, lift=True)

    def native_face_lift_padded(self, cube_face_code: int) -> np.ndarray:
        """按 cube face code 取**已填充到全局 `n_sps` 宽度**的 DG 提升矩阵，
        形状 `(n_sps, n_fp)`。

        生产残差路径消费的是填充版本（见 `lift_native_tet_padded` /
        `lift_native_prism_padded` 字段说明）。
        """
        code = int(cube_face_code)
        if 6 <= code < 10:
            if self.lift_native_tet_padded is None:
                raise ValueError(
                    f"cube_face_code={code} 是 native 四面体面，但算子集里"
                    f"没有填充好的提升算子")
            return self.lift_native_tet_padded[code - 6]
        if 10 <= code < 15:
            if self.lift_native_prism_padded is None:
                raise ValueError(
                    f"cube_face_code={code} 是 native 棱柱面，但算子集里"
                    f"没有填充好的提升算子 —— 棱柱只有原生基一种实现"
                    f"（2026-09-23 起），出现这个说明算子构造本身失败了")
            return self.lift_native_prism_padded[code - 10]
        raise ValueError(
            f"cube_face_code={code} 不是原生面编码（四面体 [6,10)、"
            f"棱柱 [10,15)）")

    def _native_face_op(self, cube_face_code: int, lift: bool) -> np.ndarray:
        code = int(cube_face_code)
        if 6 <= code < 10:
            table = (self.lift_native_tet if lift
                     else self.boundary_extrap_native_tet)
            if table is None:
                raise ValueError(
                    f"cube_face_code={code} 是 native 四面体面，但算子集里"
                    f"没有对应的表 —— native 是四面体唯一实现，这说明算子"
                    f"构造被跳过了")
            return table[code - 6]
        if 10 <= code < 15:
            table = (self.lift_native_prism if lift
                     else self.boundary_extrap_native_prism)
            if table is None:
                raise ValueError(
                    f"cube_face_code={code} 是 native 棱柱面，但算子集里"
                    f"没有原生棱柱面算子 —— 棱柱只有原生基一种实现"
                    f"（2026-09-23 起），出现这个说明算子构造本身失败了")
            return table[code - 10]
        raise ValueError(
            f"cube_face_code={code} 不是原生面编码（四面体 [6,10)、"
            f"棱柱 [10,15)）——原生是唯一实现，没有第二个入口")
