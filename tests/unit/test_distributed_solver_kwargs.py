"""DistributedFRSolver 的 solver_kwargs：未知键报错、缺省值取自单机 FRSolver 签名（I-109）。

此前以 `.get(键, 缺省)` 读取：键名拼错（如 'aoa' 而不是 'aoa_deg'）不报错、静默换成缺省物理量；缺省值在这里
另写一份（Tu=0.01、VR=5、人工粘性、近壁处理……），与单机构造签名各自维护。
"""

import inspect

import pytest

from autoflowcfd.core.fr_solver.solver import FRSolver
from autoflowcfd.core.mpi.distributed_solver import core as dcore
from autoflowcfd.core.time_integration import TimeIntegrationScheme

_REQ = dict(order=1, turb_model_name="none", time_scheme=TimeIntegrationScheme.SSP_RK3, rho_inf=1.2, vel_inf=30.0,
            p_inf=1.0e5, mu_molecular=1.8e-5)


def test_misspelled_key_is_an_error():
    with pytest.raises(TypeError, match="aoa"):
        dcore._check_solver_kwargs({**_REQ, "aoa": 5.0})


def test_missing_required_key_is_an_error():
    kw = dict(_REQ)
    del kw["vel_inf"]
    with pytest.raises(TypeError, match="vel_inf"):
        dcore._check_solver_kwargs(kw)


@pytest.mark.parametrize("name", ["turbulence_intensity", "viscosity_ratio", "artificial_viscosity_enabled",
                                  "artificial_viscosity_alpha", "wall_treatment", "dual_time_inner_iter",
                                  "time_accurate", "adaptive_cfl", "low_mach_precond", "aoa_deg", "aos_deg"])
def test_defaults_come_from_the_single_node_signature(name):
    expected = inspect.signature(FRSolver.__init__).parameters[name].default
    assert dcore._kwarg(dict(_REQ), name) == expected
    assert dcore._kwarg({**_REQ, name: "given"}, name) == "given"


def test_no_literal_fallbacks_and_no_runtime_asserts():
    src = inspect.getsource(dcore)
    assert "solver_kwargs.get(" not in src
    assert "\n        assert " not in src and "\n                assert " not in src
