"""AutoFlowCFD V2.0 - 四面体原生基（受限 PKD/Dubiner + Warp&Blend 节点）。

从 `fr/` 顶层的三个平铺文件收进本子包（2026-09-19，项目"按功能逻辑分
文件夹"的规范），与 `fr/native_prism/` 对称：

    basis.py            节点/模态/Vandermonde/体积微分算子/几何映射与
                        精确雅可比/自身面外插/DG 提升（原
                        `native_simplex_basis.py`）
    filter.py           模态滤波矩阵（原 `native_tet_filter.py`）
    overintegration.py  体积项去混叠三件套与阶数策略
                        （原 `native_tet_overintegration.py`）

`fr/native_padding.py`（零填充块对角）与 `fr/warp_blend_nodes.py`（节点
生成）两者被**两种**原生基共用，所以留在 `fr/` 顶层而不是收进任一子包。
阶数策略（上限常量/env/`rule*order` 规则）住在 `fr/overintegration_order.py`
—— 那是棱柱两档与四面体共用的唯一事实来源。

搬家只改位置与导入，不改任何逻辑：搬家前后同一算例 20 步的解场
SHA256 逐位相同（坍缩/原生 × P1/P2 四种组合）。

本 `__init__.py` **刻意不做 re-export**：四面体这三个模块的调用点本来
就是按模块导入的（不像 `face_flux_points` 那样存在"从包根导入"的大量
既有写法），再加一层 re-export 只会多出一份要同步的名单。
"""
