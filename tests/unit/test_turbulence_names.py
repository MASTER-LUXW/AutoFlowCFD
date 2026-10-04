# -*- coding: utf-8 -*-
"""湍流模型的配置层枚举与求解器 / CLI 模型名之间的唯一映射（`config/turbulence_names.py`）。

此前四份映射各写一份，`api_create_steady_config` 那份用 `.get(name, SST_KW)` 把写错的模型名静默
换成 SST；SA-neg 接入后四份都要同步。
"""

import pytest

from autoflowcfd.config.solver_config import TurbulenceModel
from autoflowcfd.config.turbulence_names import turbulence_from_name, turbulence_solver_name


@pytest.mark.parametrize("model", list(TurbulenceModel))
def test_every_enum_value_roundtrips(model):
    assert turbulence_from_name(turbulence_solver_name(model)) is model
    assert turbulence_from_name(model.value) is model


def test_sa_is_a_real_option():
    assert turbulence_solver_name(TurbulenceModel.SA) == "sa"
    assert turbulence_from_name("SA") is TurbulenceModel.SA


def test_unknown_name_is_an_error_not_sst():
    with pytest.raises(ValueError, match="Unknown turbulence model"):
        turbulence_from_name("kepsilon")


def test_steady_api_config_rejects_unknown_model():
    """fail 半边：此前 `api_create_steady_config(turbulence="kepsilon")` 静默得到 SST 配置。"""
    from autoflowcfd.api import AutoFlowCFDAPI

    api = AutoFlowCFDAPI()
    with pytest.raises(ValueError):
        api.create_steady_config(turbulence="kepsilon")
    assert api.create_steady_config(turbulence="sa").turbulence is TurbulenceModel.SA


def test_cli_choices_accept_sa():
    from autoflowcfd.cli.solve.steady.command import solve_steady
    from autoflowcfd.cli.solve.transient import transient

    for cmd in (solve_steady, transient):
        opt = next(p for p in cmd.params if p.name == "turbulence_model")
        assert "sa" in opt.type.choices
