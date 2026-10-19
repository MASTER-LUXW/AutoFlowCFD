"""checkpoint 物理参数直接读求解器属性（I-87）。

此前 `physics_metadata` 对分子粘度、Tu/VR、人工粘性开关用 getattr 缺省值"为测试替身保留"：某条路径真的丢了属性时
按缺省值写进 checkpoint，续算即是另一个物理算例。
"""

from types import SimpleNamespace

import pytest

from autoflowcfd.core.utils.checkpoint_physics import physics_metadata

_FULL = dict(freestream={"rho_inf": 1.2, "vel_inf": 30.0, "p_inf": 1.0e5}, wall_treatment="auto",
             mu_molecular=2.0e-5, _turbulence_intensity=0.02, _viscosity_ratio=3.0,
             artificial_viscosity_enabled=True, artificial_viscosity_alpha=0.5)


def test_values_are_read_from_the_solver():
    meta = physics_metadata(SimpleNamespace(**_FULL))
    assert (meta["mu_molecular"], meta["turbulence_intensity"], meta["viscosity_ratio"]) == (2.0e-5, 0.02, 3.0)
    assert meta["artificial_viscosity_enabled"] is True and meta["artificial_viscosity_alpha"] == 0.5


@pytest.mark.parametrize("missing", ["mu_molecular", "_turbulence_intensity", "_viscosity_ratio",
                                     "artificial_viscosity_enabled", "artificial_viscosity_alpha"])
def test_missing_attribute_is_an_error_not_a_default(missing):
    attrs = dict(_FULL)
    del attrs[missing]
    with pytest.raises(AttributeError):
        physics_metadata(SimpleNamespace(**attrs))
