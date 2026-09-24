"""AutoFlowCFD V2.0 - 枚举到求解器字符串的映射与工厂函数

从 `src/autoflowcfd/api.py` 拆出（2026-09-24，项目「单文件不超 500 行」规范）。
"""



def _turbulence_model_str(turb_config_value) -> str:
    """把 `config.solver_config.TurbulenceModel` 枚举值映射到求解器/CLI
    真正使用的字符串取值（none/sst/ddes/wmles/les，见
    fr_solver/turbulence.py::init_turbulence_models）。

    配套 #6（配置层接入）：`SteadyConfig`/`TransientConfig.turbulence`
    字段用的是这套独立枚举，命名（sst_kw vs sst）和求解器实际接受的
    字符串不完全一致，真正把 config 对象喂给 FRSolver 构造之前必须
    先做这层转换，而不是直接 `.value` 传下去（那样 "sst_kw" 会被当成
    未知湍流模型字符串处理，构造函数从不认识这个值）。
    """
    from autoflowcfd.config.solver_config import TurbulenceModel
    mapping = {
        TurbulenceModel.NONE: "none",
        TurbulenceModel.SST_KW: "sst",
        TurbulenceModel.DDES: "ddes",
        TurbulenceModel.IDDES: "iddes",
        TurbulenceModel.WMLES: "wmles",
        TurbulenceModel.LES: "les",
    }
    if turb_config_value not in mapping:
        raise ValueError(
            f"Turbulence model '{turb_config_value.value}' is representable in "
            f"SteadyConfig/TransientConfig but is not actually implemented by "
            f"FRSolver/GPUFRSolver — only none/sst_kw/ddes/iddes/wmles/les are real "
            f"solver options."
        )
    return mapping[turb_config_value]
