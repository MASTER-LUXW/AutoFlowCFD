# -*- coding: utf-8 -*-
"""GPU 测试共用：把 `get_cupy` 换成 numpy 替身 —— **连子模块一起换**。

## 为什么需要这个 helper（2026-09-24）

GPU 侧几个千行文件已按职责拆成子包（`gpu_inviscid`/`gpu_viscous`/
`gpu_scalar_transport` 等），而 `get_cupy()` 的**调用点在子模块里**。
于是原先的写法全都失效：

    monkeypatch.setattr(gst_mod, "get_cupy", lambda: shim)

对**包对象**做 setattr 不会改变子模块内部的调用。两种失效方式都出现过：

* 不带守卫的写法 -> `AttributeError: module ... has no attribute 'get_cupy'`
  （能被发现）；
* 带 `if hasattr(m, "get_cupy")` 守卫的写法 -> **静默跳过**，替身根本没装上，
  测试于是去用真的 cupy。这比报错更糟。

这与 `ProjectFiles/V2.0/29` 记的检查 (3) 是同一件事：**patch 目标必须是
真正持有调用点的模块**。本文件把"遍历包的全部子模块"这件事做成唯一实现，
避免每个测试文件各写一遍。

## 断言必须是**聚合**的

第一版对每个模块单独断言"至少装上一处"，在混合列表上直接误报：调用方
常常把 `gpu_inviscid_volume`、`gpu_volume_contract` 这类**本来就不含**
`get_cupy` 的模块一起传进来（原写法的 `if hasattr` 守卫同时兼着"过滤
这类模块"和"容忍拆包失效"两个职责）。正确语义是：**整批**至少装上一处，
单个模块没有不算错。所以本函数接受模块列表，一次调用一次断言。
"""

import importlib
import pkgutil


def _walk(mod):
    """`mod` 自身，以及（若它是包）它的全部子模块。"""
    yield mod
    path = getattr(mod, "__path__", None)
    if path is None:
        return
    for info in pkgutil.iter_modules(path):
        try:
            yield importlib.import_module(mod.__name__ + "." + info.name)
        except Exception:
            # 子模块导入失败不该让 patch 整体失败（例如可选依赖缺失）；
            # 真正重要的是下面的聚合断言。
            continue


def patch_module_get_cupy(monkeypatch, mods, shim, *, required=True):
    """在 `mods` 及其全部子模块上把 `get_cupy` 换成 `lambda: shim`。

    Args:
        mods: 单个模块，或模块序列（推荐一次传一批，见模块文档"聚合断言"）。
        required: True 时若**整批**一处都没装上就断言失败 —— 这一条是关键，
            它把"静默跳过"变成显式失败。

    Returns:
        实际打上补丁的模块数。
    """
    if hasattr(mods, "__name__"):          # 单个模块
        mods = [mods]
    mods = list(mods)
    n = 0
    for top in mods:
        for m in _walk(top):
            if hasattr(m, "get_cupy"):
                monkeypatch.setattr(m, "get_cupy", lambda: shim)
                n += 1
    if required:
        names = ", ".join(getattr(m, "__name__", str(m)) for m in mods)
        assert n > 0, (
            "以下模块（含其子模块）里没有任何一处 `get_cupy` 被替换 —— "
            "替身没装上，测试会去用真的 cupy：" + names + "。"
            "若这批模块确实都不该有 get_cupy，请显式传 required=False")
    return n
