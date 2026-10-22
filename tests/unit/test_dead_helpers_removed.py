"""零调用的辅助函数已删除（I-3 / I-15）：保留它们只会让读者以为存在第二条执行路径。"""

import pytest


@pytest.mark.parametrize("module, name", [
    ("autoflowcfd.core.fr_residual.inviscid", "ausm_up_flux_batch"),        # 逐点循环 AUSM+up，生产走核内同一函数
    ("autoflowcfd.core.gpu", "gpu_device_count"),
    ("autoflowcfd.core.gpu", "get_device_info"),
    ("autoflowcfd.config.loader", "save_config_template"),                 # CLI 直接用 ConfigLoader.save_template
])
def test_removed(module, name):
    import importlib

    assert not hasattr(importlib.import_module(module), name)


def test_reference_area_mixin_is_gone():
    """V1 遗留的参考面积 mixin（I-84）：宿主类早已不存在，按组名含 BODY/CAR 认车身、只沿 X 投影、异常时静默返回
    固定值。生产路径是 `cli/solve/aero_coefficients.py::_compute_reference_area_auto`。"""
    import importlib.util

    import autoflowcfd.core.utils as utils

    assert importlib.util.find_spec("autoflowcfd.core.utils.aero_coeffs") is None
    assert not hasattr(utils, "ReferenceAreaMixin")
