"""AutoFlowCFD V2.0 - 湍流场的单元平均输出（checkpoint 与 VTK 后处理共用的唯一实现）。

## 为什么需要它（2026-10-04）

此前 VTK 导出的 `k`/`omega` 取自守恒解的第 6、7 列，而那两列是 SST 求解器状态数组里从未被
更新的历史槽位（`time_integration/implicit/mean_flow_step.py` 模块文档："全仓库无人读取、
残差恒为零"）——导出的永远是初值；`nut` 期望 checkpoint 里有 `mu_t` 字段，而写 checkpoint 的
一方从未写过它，于是退化成用那两个初值列算的 `k/omega` 估计。真正的湍流场在模型对象上
（`TransportedTurbulence`）。这里把模型的输运场与涡粘按**真实自由度**求单元平均（与守恒解
单元平均同一个归约，`fr/native_padding.py::reduce_per_cell_over_real_sps`），键名取模型声明的
`OUTPUT_FIELD_KEYS`（SST: `k`、`omega`；SA-neg: `nu_tilde`），外加 `nut`（运动涡粘）。
"""

import numpy as np

from autoflowcfd.fr.native_padding import reduce_per_cell_over_real_sps

#: 写进 checkpoint 时的字段名前缀（`turb_cell_<键>`）。
CHECKPOINT_PREFIX = "turb_cell_"


def turbulence_cell_means(turb_model, n_prism: int, order: int) -> dict:
    """`{输出键: (n_cells,) 单元平均}`；模型为 None 或不是输运模型时返回空字典。"""
    if turb_model is None or not hasattr(turb_model, "transported_fields"):
        return {}
    out = {}
    named = list(zip(turb_model.OUTPUT_FIELD_KEYS, turb_model.transported_fields()))
    if getattr(turb_model, "nu_t", None) is not None:
        named.append(("nut", turb_model.nu_t))
    for key, field in named:
        host = np.asarray(field.get() if hasattr(field, "get") else field, dtype=np.float64)
        out[key] = reduce_per_cell_over_real_sps(host, int(n_prism), int(order), "mean")
    return out


def turbulence_fields_from_checkpoint(fields: dict) -> dict:
    """从 checkpoint 的 extra fields 里取出 `turbulence_cell_means` 写入的那一组（去掉前缀）。"""
    return {name[len(CHECKPOINT_PREFIX):]: np.asarray(v, dtype=np.float64)
            for name, v in (fields or {}).items() if name.startswith(CHECKPOINT_PREFIX)}
