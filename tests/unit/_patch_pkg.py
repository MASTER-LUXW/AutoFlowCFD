# -*- coding: utf-8 -*-
"""在一个模块**及其全部子模块**上替换某个名字 —— 拆包后 patch 的唯一正确姿势。

## 为什么需要它（2026-09-24）

本项目把超 500 行的模块陆续拆成子包。`monkeypatch.setattr(pkg, "Name", fake)`
对**包对象**做替换，**不会**改变子模块内部对同一名字的引用 —— 子模块有
自己的全局命名空间。于是有两种结局，危害差别很大：

* 那个名字没有被 `__init__` re-export -> `AttributeError`，能被发现；
* 被 re-export 了 -> setattr **成功**，但真正的调用点（子模块的全局）毫发
  无损。**patch 静默失效，而测试照样"通过"**。这比没有测试更糟。

判断哪种情况取决于调用点怎么取这个名字：

* 子模块顶层 `from X import name` -> 名字进了**子模块**的全局，必须 patch
  子模块（本文件干的事）；
* 函数体内 `from X import name`（惰性导入）-> 每次调用都重新从 X 取，
  patch **X 本身**才对，patch 子模块无效。

所以本文件的做法是"包和它的全部子模块，凡有这个属性就都换掉"，两种情况
一起覆盖，并用**聚合断言**保证至少换掉了一处。

记在 `ProjectFiles/V2.0/29_transport拆包与SST黄金轨迹判据.md` 三·五 (3)。
"""

import importlib
import pkgutil


def walk_modules(mod):
    """`mod` 自身，以及（若它是包）它的全部子模块（递归）。"""
    yield mod
    path = getattr(mod, "__path__", None)
    if path is None:
        return
    for info in pkgutil.iter_modules(path):
        try:
            sub = importlib.import_module(mod.__name__ + "." + info.name)
        except Exception:
            # 子模块导入失败不该让 patch 整体失败（例如可选依赖缺失）；
            # 真正重要的是调用方的聚合断言。
            continue
        for m in walk_modules(sub):
            yield m


def patch_pkg_attr(monkeypatch, mods, name, value, *, required=True):
    """在 `mods`（模块或模块序列）及其全部子模块上把 `name` 换成 `value`。

    Args:
        mods: 单个模块，或模块序列（推荐一次传一批 —— 断言是**聚合**的，
            所以列表里含本就不带这个属性的模块无妨）。
        required: True 时若**整批**一处都没换上就断言失败，把"静默跳过"
            变成显式失败。

    Returns:
        实际打上补丁的模块数。
    """
    if hasattr(mods, "__name__"):
        mods = [mods]
    mods = list(mods)
    n = 0
    for top in mods:
        for m in walk_modules(top):
            if hasattr(m, name):
                monkeypatch.setattr(m, name, value)
                n += 1
    if required:
        names = ", ".join(getattr(m, "__name__", str(m)) for m in mods)
        assert n > 0, (
            "以下模块（含其子模块）里没有任何一处 `" + name + "` 被替换 —— "
            "替身没装上，测试用的是真实实现：" + names + "。"
            "若这批模块确实都不该有这个属性，请显式传 required=False")
    return n
