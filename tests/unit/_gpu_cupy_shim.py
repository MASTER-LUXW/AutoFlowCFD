# -*- coding: utf-8 -*-
"""GPU 测试共用：把 `get_cupy` 换成 numpy 替身 —— **连子模块一起换**。

`get_cupy()` 的调用点在子模块里（`gpu_inviscid`/`gpu_viscous`/
`gpu_scalar_transport` 等千行文件已按职责拆成子包），对**包对象**做
setattr 不会改变子模块内部的调用。遍历子模块这件事由
`tests/unit/_patch_pkg.py::patch_pkg_attr` 统一实现（那里记了两种失效
方式，以及为什么断言必须是聚合的）；本文件只是它在 `get_cupy` 上的
一层薄封装，保持 16 个测试文件的调用写法不变。
"""

from tests.unit._patch_pkg import patch_pkg_attr


def patch_module_get_cupy(monkeypatch, mods, shim, *, required=True):
    """在 `mods` 及其全部子模块上把 `get_cupy` 换成 `lambda: shim`。

    Args:
        mods: 单个模块，或模块序列（断言是聚合的，见 `patch_pkg_attr`）。
        required: True 时整批一处都没换上就断言失败。

    Returns:
        实际打上补丁的模块数。
    """
    return patch_pkg_attr(monkeypatch, mods, "get_cupy", lambda: shim,
                          required=required)
