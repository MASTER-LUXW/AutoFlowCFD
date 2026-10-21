"""湍流模型按名字构造只有一处（`build_turbulence_models`，I-82）：此前 CPU 单机、CPU 完全分布式、单 GPU、多 GPU、
多 GPU 完全分布式各写一条 if/elif 链（DES 模型、SST 上界、SA-neg 来流各自重复）。"""

import types

import numpy as np
import pytest

from autoflowcfd.core.fr_solver.turbulence.init import build_turbulence_models, describe_turbulence_models
from autoflowcfd.core.turbulence.des import DDESModel, IDDESModel
from autoflowcfd.core.turbulence.registry import SUPPORTED_MODELS
from autoflowcfd.core.turbulence.sa import SAModel
from autoflowcfd.core.turbulence.sgs import WALEModel
from autoflowcfd.core.turbulence.sst import SSTModelFR


def _owner():
    return types.SimpleNamespace(freestream={"rho_inf": 1.2, "vel_inf": 40.0, "p_inf": 1.0e5}, mu_molecular=1.8e-5,
                                 _turbulence_intensity=0.01, _viscosity_ratio=5.0, wall_treatment="auto")


@pytest.mark.parametrize("name", SUPPORTED_MODELS)
def test_every_supported_model_is_built(name):
    models = build_turbulence_models(_owner(), name, 4, 6)
    if name in ("SST", "DDES", "IDDES"):
        assert isinstance(models.transport, SSTModelFR) and models.transport.k_field.shape == (4, 6)
        assert models.transport.k_max == pytest.approx(0.5 * 40.0 ** 2)          # configure_sst_model
        assert models.transport.wall_treatment == "auto"
        np.testing.assert_allclose(models.transport.k_inf, 1.5 * (40.0 * 0.01) ** 2)
    elif name == "SA":
        assert isinstance(models.transport, SAModel) and models.transport.viscosity_ratio == 5.0
    else:
        assert models.transport is None
    assert isinstance(models.des, {"DDES": DDESModel, "IDDES": IDDESModel}.get(name, type(None)))
    assert (models.sgs is not None) == (name in ("LES", "WMLES"))
    assert models.sgs is None or isinstance(models.sgs, WALEModel)
    assert describe_turbulence_models(name, models)


def test_unknown_model_is_rejected():
    with pytest.raises(ValueError, match="未知湍流模型"):
        build_turbulence_models(_owner(), "kepsilon", 4, 6)


def test_gpu_construction_needs_both_cupy_and_device():
    with pytest.raises(ValueError, match="cupy"):
        build_turbulence_models(_owner(), "SST", 4, 6, device_id=0)
