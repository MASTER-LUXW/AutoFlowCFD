"""
AutoFlowCFD V2.0 - FR 算子生成器主接口

本模块是 Flux Reconstruction 方法算子生成的统一入口，
整合了点集生成、矩阵算子计算等功能。

核心功能：
1. 统一的FR算子生成接口
2. 算子数据容器定义

四面体与三棱柱都只有原生基（四面体：受限 PKD/Dubiner + Warp&Blend 节点，`fr/native_tet/`；三棱柱：原生棱柱基，
`fr/native_prism/`）。全部算子零填充块对角到统一的解点宽度（`fr/native_padding.py`），`D_3d_tet`/`D_3d_prism`
即对应的原生填充版本。

## 文件分工（2026-09-19 拆包）

    container.py   `FROperators` 数据容器与它的取算子方法
    build.py       `generate_fr_operators` 构建流程

本 `__init__.py` re-export 三个既有公开名（`FROperators`/
`generate_fr_operators`/`gauss_legendre`），所以全仓库
`from autoflowcfd.fr.operators import ...` 一个字都不用改。
"""

from ..quadrature_points import gauss_legendre, gauss_lobatto  # noqa: F401
from .container import FROperators  # noqa: F401
from .build import generate_fr_operators  # noqa: F401

__all__ = ["FROperators", "generate_fr_operators",
           "gauss_legendre", "gauss_lobatto"]
