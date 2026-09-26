"""AutoFlowCFD V2.0 - 平均流 FR 残差的解析单元块 Jacobian（隐式预处理用）。

    pointwise.py   逐点导数（dQ/dU、dT/dQ、物理通量对输入的差分导数）
    volume.py      体积项（无粘 + 粘性，含过积分）对对角块的贡献
    faces.py       界面项对对角块的贡献（逐点跳变量与残差核共用同一函数）
    assemble.py    组装、dQ/dU 与低马赫 Gamma 链接

背景与正确性判据见 `assemble.py` 模块文档。
"""

from .assemble import MeanFlowLinearization, assemble_mean_flow_blocks

__all__ = ["MeanFlowLinearization", "assemble_mean_flow_blocks"]
