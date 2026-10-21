"""平衡壁面模型（Kawai & Larsson 2012 的匹配点选取 + Spalding 壁面律）：WMLES 与 RANS 自动壁面处理共用。

    law.py        Spalding 全 y+ 壁面律与 u+ 的带区间保护 Newton 求解
    matching.py   壁面通量点的匹配点几何（owner 单元形心高度、参考坐标反解、插值行）
    stress.py     匹配点解 -> 壁面牵引 -> 平均流动量残差
    model.py      选物理无滑移壁面、缓存几何、对外接口 `EquilibriumWallModel`
    treatment.py  RANS 的近壁处理方式（resolved | auto）

CPU、单 GPU、CPU 分布式、多 GPU 共用这一份（数组模块无关）；壁面力积分（`postprocess/wall_forces.py`）用同一个
牵引。SGS 涡粘由 WALE 给出（`core/turbulence/sgs.py`）。
"""

from .law import B_LOG, KAPPA, solve_u_plus, spalding_y_plus, wall_shear_stress  # noqa: F401
from .matching import WallMatching, build_wall_matching  # noqa: F401
from .model import EquilibriumWallModel, wall_model_codes  # noqa: F401
from .treatment import (  # noqa: F401
    DEFAULT_WALL_TREATMENT, WALL_TREATMENTS, build_wall_model, needs_wall_model, validate_wall_treatment)

__all__ = [
    "B_LOG", "DEFAULT_WALL_TREATMENT", "build_wall_model", "KAPPA", "EquilibriumWallModel", "WALL_TREATMENTS",
    "WallMatching",
    "build_wall_matching", "needs_wall_model", "solve_u_plus", "spalding_y_plus", "validate_wall_treatment",
    "wall_model_codes", "wall_shear_stress",
]
