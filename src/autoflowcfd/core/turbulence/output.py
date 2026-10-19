"""AutoFlowCFD V2.0 - 湍流场的单元平均输出（checkpoint 与 VTK 后处理共用的唯一实现）。

## 为什么需要它（2026-10-04）

此前 VTK 导出的 `k`/`omega` 取自守恒解的第 6、7 列，而那两列是 SST 求解器状态数组里从未被
更新的历史槽位（`time_integration/implicit/newton/mean_flow_step.py` 模块文档："全仓库无人读取、
残差恒为零"）——导出的永远是初值；`nut` 期望 checkpoint 里有 `mu_t` 字段，而写 checkpoint 的
一方从未写过它，于是退化成用那两个初值列算的 `k/omega` 估计。真正的湍流场在模型对象上
（`TransportedTurbulence`）。这里把模型的输运场与涡粘按**真实自由度**求单元平均（与守恒解
单元平均同一个积分平均，`fr/native_padding.py::weighted_cell_average`），键名取模型声明的
`OUTPUT_FIELD_KEYS`（SST: `k`、`omega`；SA-neg: `nu_tilde`），外加 `nut`（运动涡粘）。
"""

import numpy as np

from autoflowcfd.fr.native_padding import weighted_cell_average

#: 写进 checkpoint 时的字段名前缀（`turb_cell_<键>`）。
CHECKPOINT_PREFIX = "turb_cell_"


def turbulence_sp_fields(turb_model) -> dict:
    """`{输出键: (n_cells, n_sps) 解点值}`（主机数组）；模型为 None 或不是输运模型时返回空字典。高阶 VTK 导出用
    解点值本身，单元平均（`turbulence_cell_means`）由它积分得到。"""
    if turb_model is None or not hasattr(turb_model, "transported_fields"):
        return {}
    named = list(zip(turb_model.OUTPUT_FIELD_KEYS, turb_model.transported_fields()))
    if getattr(turb_model, "nu_t", None) is not None:
        named.append(("nut", turb_model.nu_t))
    return {key: np.asarray(field.get() if hasattr(field, "get") else field, dtype=np.float64)
            for key, field in named}


def turbulence_cell_means(turb_model, W) -> dict:
    """`{输出键: (n_cells,) 单元积分平均}`；模型为 None 或不是输运模型时返回空字典。

    Args:
        W: `fr/native_padding.py::sp_integration_weights` 给出的解点积分权重（与模型场同一单元顺序）
    """
    return {key: weighted_cell_average(field[:W.shape[0]], W)
            for key, field in turbulence_sp_fields(turb_model).items()}


def turbulence_fields_from_checkpoint(fields: dict) -> dict:
    """从 checkpoint 的 extra fields 里取出 `turbulence_cell_means` 写入的那一组（去掉前缀）。"""
    return {name[len(CHECKPOINT_PREFIX):]: np.asarray(v, dtype=np.float64)
            for name, v in (fields or {}).items() if name.startswith(CHECKPOINT_PREFIX)}
