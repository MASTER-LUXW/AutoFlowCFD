"""AutoFlowCFD V2.0 - FR 算子数据容器 `FROperators`。

这里只有容器与它的取算子方法，构建流程在 `build.py`；两类原生面算子的分派约定见 `__init__.py` 的模块文档。
"""

from typing import Optional
import numpy as np
from dataclasses import dataclass


@dataclass
class FROperators:
    """FR 算子容器（四面体与三棱柱都是原生基，算子零填充块对角到统一解点宽度，`fr/native_padding.py`）。

    2026-10-19 起删除张量积时代的 `D_1d`/`D_3d`（朴素 Lagrange 张量积微分矩阵，只对六面体正确，生产里只被拿来
    读解点宽度）、别名 `D_3d_tet`/`D_3d_prism`（与 `D_native_tet_padded`/`D_native_prism_padded` 是同一个对象，
    两个名字并存让消费点各写一种兜底）与恒为 "native" 的 `tet_basis_mode`；解点宽度改为显式字段 `n_sps`。
    """
    #: 每单元解点存储宽度（全部按单元的数组与填充算子的 SP 轴长度）
    n_sps: int
    # 原生四面体（`fr/native_tet/basis.py`）：D_native_tet 形状 (n_native_sps_tet, n_native_sps_tet, 3)，参考坐标是
    # 四面体自己的 (r,s,t) 单纯形。
    D_native_tet: np.ndarray
    ref_native_tet: np.ndarray
    n_native_sps_tet: int
    # `D_native_tet` 零填充到解点宽度 `n_sps` 之后的体积微分算子（`fr/native_padding.py::pad_native_matrix_to_global`）。
    # 面外插 / DG 提升见下方 `face_extrap_by_op`/`face_lift_by_op`。
    D_native_tet_padded: np.ndarray

    # ======================= 原生棱柱基（2026-09-18） =======================
    #
    # 面算子（外插/提升）在下方整表里按面算子索引取（棱柱 face_id 0~4 即面编码 10~14）。删除坍缩棱柱基前后的对照
    # 数据见 `fr/native_prism/__init__.py`。
    D_native_prism: np.ndarray
    ref_native_prism: np.ndarray
    n_native_sps_prism: int
    #: 零填充到解点宽度 `n_sps` 之后的体积微分算子。
    D_native_prism_padded: np.ndarray
    #: 全部原生面（含三角形面的 3 个坍缩顶点槽位）按面算子索引排好的外插 / 提升整表，
    #: 填充到全局 `n_sps` 宽度：`(N_FACE_OPS, n_fp, n_sps)` / `(N_FACE_OPS, n_sps, n_fp)`
    #: （`fr/operators/face_ops.py`）。逐面取用走 `native_face_extrap` 或平面面几何里的
    #: 逐面面算子索引。
    face_extrap_by_op: np.ndarray
    face_lift_by_op: np.ndarray

    # 体积项去混叠（over-integration）：order==0 时全部为 None（P0 走独立的有限体积残差路径，不需要）。
    #: 棱柱使用的过积分阶数，取自 `fr/overintegration_order.py::overintegration_order`（`2*order`，
    #: P1/P2/P3 = 2/4/6）。order==0 时为 0。
    overint_order_prism: int = 0
    #: 棱柱过积分的**真实**细点数 `(oo+1)^2(oo+2)/2`（`overintegration_order.prism_n_fine`），同时是
    #: `mesh.n_sps_per_cell_fine`（`jacobians_fine` 的每单元布局宽度）。
    overint_n_fine_prism: int = 0
    #: native 四面体过积分的**真实**细点数 `(oo+1)(oo+2)(oo+3)/6`。与
    #: 棱柱的宽度**无关**——细网格轴不填充，且四面体段的细点度量是逐单元
    #: 常数取第 0 列广播，见 `generate_fr_operators` 里那段说明与
    #: `volume_contract.get_overintegration_context`。
    overint_n_fine_tet: int = 0
    #: 四面体使用的过积分阶数，与棱柱同取 `fr/overintegration_order.py::overintegration_order`（`2*order`）；
    #: 两者分设字段是因为各走自己的基与细点集。order==0 时为 0。
    overint_order_tet: int = 0
    overint_interp_c2f_tet: Optional[np.ndarray] = None
    overint_interp_c2f_prism: Optional[np.ndarray] = None
    overint_D_fine_tet: Optional[np.ndarray] = None
    overint_D_fine_prism: Optional[np.ndarray] = None
    overint_restrict_f2c_tet: Optional[np.ndarray] = None
    overint_restrict_f2c_prism: Optional[np.ndarray] = None
    #: 无粘体积算子 `K = f2c·D_fine - Σ_面 C_面` `(n_sps, n_fine, 3)`：修正项
    #: 的本侧通量取细层通量多项式的法向迹并并进体积项（离散守恒，见
    #: `fr/face_flux_trace.py`）。
    overint_lifted_div_tet: Optional[np.ndarray] = None
    overint_lifted_div_prism: Optional[np.ndarray] = None

    # ---- 原生面算子的逐面分派入口 ----
    #
    # 两类单元的原生面算子键不同（四面体是 excluded_vertex 0~3，棱柱是
    # face_id 0~4），三角形面还要按坍缩顶点槽位分 3 份，而消费点拿到的是统一的
    # `cube_face_code` 与该侧槽位（`mesh.face_flux_points.owner_tri_slot`）：
    #
    #     [6, 10)  -> native 四面体面，excluded_vertex = code - 6
    #     [10, 15) -> native 棱柱面，  face_id        = code - 10
    #
    # **必须走这个方法**，不要在消费点自己判断：面 id 配错不会报错，只会静默地对
    # 某个面用错矩阵。numba kernel 不能调方法，它们读平面面几何里的整表与逐面
    # 面算子索引（`face_kernels.build_flat_face_geometry`，`fr/operators/face_ops.py`）。

    def native_face_extrap(self, cube_face_code: int, slot: int = 0) -> np.ndarray:
        """按 cube face code 与坍缩顶点槽位取原生面的体积->面外插矩阵（**未填充**，
        形状 `(n_fp, n_native)`，`n_native` 随单元类型不同）。

        Raises:
            ValueError: 不是原生面编码，或给四边形面传了非零槽位。
        """
        from .face_ops import face_op_index, is_triangle_face_code

        code = int(cube_face_code)
        if 6 <= code < 10:
            n_native = self.n_native_sps_tet
        elif 10 <= code < 15:
            n_native = self.n_native_sps_prism
        else:
            raise ValueError(
                f"cube_face_code={code} 不是原生面编码（四面体 [6,10)、"
                f"棱柱 [10,15)）——原生是唯一实现，没有第二个入口")
        if slot != 0 and not bool(is_triangle_face_code(code)):
            raise ValueError(f"四边形面（cube_face_code={code}）没有坍缩顶点，槽位只能是 0，收到 {slot}")
        if self.face_extrap_by_op is None:
            raise ValueError("算子集里没有原生面算子整表 —— 算子构造本身失败了")
        return self.face_extrap_by_op[int(face_op_index(code, slot))][:, :int(n_native)]
