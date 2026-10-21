"""AutoFlowCFD V2.0 - 四面体原生基（受限 PKD/Dubiner + Warp&Blend 节点），与 `fr/native_prism/` 对称：

    basis.py            节点/模态/Vandermonde/体积微分算子/几何映射与
                        精确雅可比/自身面外插/DG 提升
    overintegration.py  体积项去混叠三件套

`fr/native_padding.py`（零填充块对角）与 `fr/warp_blend_nodes.py`（节点生成）被两种原生基共用，留在 `fr/`
顶层。过积分阶数策略在 `fr/overintegration_order.py`（棱柱与四面体共用的唯一事实来源）。

本 `__init__.py` 不做 re-export：调用点按模块导入。
"""
