"""AutoFlowCFD V2.0 - 面通量点（Flux Points）几何与插值算子子包。

从 `fr/` 顶层的 10 个平铺 `face_flux_points*.py` 收进本子包
（2026-09-19，项目"按功能逻辑分文件夹、单文件不超 500 行"的规范）。
搬家只改位置与导入，不改任何逻辑。

## 文件分工

    geometry.py             面通量点的参考/物理坐标、`face_ref_grid`、
                            跨单元插值矩阵 `build_cross_interp`、
                            `CUBE_FACE_AXIS_SIDE` 查找表
    locate.py               Newton 在目标单元参考空间里定位面点
    data.py                 逐面数据容器 `_KernelFaceData` 与棱柱四边形
                            侧面的三角化半区分类
    merge.py                总装入口 `build_face_flux_points`：分组、
                            调 numba kernel、拼 flat 数组
    kernel.py               主 numba 并行 kernel（Newton + 单源插值）
    kernel_multisource.py   多源面（棱柱四边形侧面被对侧三角化成两片）
                            的 numba kernel
    ref_geometry_nb.py      两个 kernel 共用的**坍缩坐标**njit 原语
                            （坐标变换/物理映射/参考面网格/Newton 定位）
    native_geometry_nb.py   同上的**原生**基版本，外加三条基共用的
                            插值入口 `interp_matrix_from_cube_coords_nb`
    face_code_tables.py     cube face 编码的 numba 查找表（15 个编码全覆盖）
    exact_normal.py         逐通量点精确法向/面积权重（numpy 桶循环版）
    exact_normal_kernel.py  同上的 numba 并行版 —— **生产走这一条**
    validation.py           面点定位残差校验与阈值

## 为什么 `__init__.py` 只做 re-export

全仓库最常见的一条写法是 `from autoflowcfd.fr.face_flux_points import
X`（`geometry.py` 里的东西）。把 `face_flux_points/geometry.py` 搬成子包里的
`geometry.py` 再在这里原样 re-export，那条写法一个字都不用改。

包内互相引用一律走**相对导入**（`from .geometry import ...`），不绕回
包根 —— 避免 `__init__` 初始化顺序带来的循环导入隐患。
"""

from .geometry import (  # noqa: F401
    ACCEPT_STRICT_REL,
    CUBE_FACE_AXIS_SIDE,
    FaceFluxPointGeometry,
    # 两个私有 LU 缓存：`tests/unit/test_native_tet_numba_kernel_parity.py`
    # 直接从包根导入它们来复现 kernel 内部的插值矩阵构造。
    _get_v_sps_lu,
    _get_v_sps_lu_native,
    build_cross_interp,
    extrapolate_to_face,
    face_ref_grid,
    native_tet_face_points_physical,
)

__all__ = [
    "ACCEPT_STRICT_REL",
    "CUBE_FACE_AXIS_SIDE",
    "FaceFluxPointGeometry",
    "build_cross_interp",
    "extrapolate_to_face",
    "face_ref_grid",
    "native_tet_face_points_physical",
]
