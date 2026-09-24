# -*- coding: utf-8 -*-
"""单元测试会话级守卫。

## `inspect.getsource(包)` 的运行时拦截（2026-09-24）

`inspect.getsource(package)` 只返回 `__init__.py`。本项目把超 500 行的模块
陆续拆成子包，于是"读源码做文本断言"的测试会在拆包后失效，而 `not in`
那一半是**静默通过**的 —— 字符串只是搬到了子模块，并不是真的不存在。

`test_no_stale_package_refs.py` 已经有一条**静态**规则查这件事，但它只认
直接写出模块别名的形式。2026-09-24 真实漏掉两处，写法都是

    for mod in (gpu_solver_io, gpu_distributed_init):
        s = inspect.getsource(mod)

模块来自循环变量，静态分析解析不到它是谁。运行时拦截没有这个盲区：不管
怎么写，只要调用方在 `tests/` 下、参数是包，就立刻失败并指明替代写法。

两者不冗余：静态规则覆盖**被跳过**的测试（本机没有 CuPy，一批 GPU 测试
整体 skip，运行时拦截对它们永远不会触发）；运行时拦截覆盖静态分析解析
不了的写法。

`tests/unit/_module_source.py` 自己需要读包的 `__init__.py`（作为拼接的
第一段），所以对它放行。
"""

import inspect
import os
import sys

_TESTS_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_ALLOWED_CALLERS = {
    os.path.normcase(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                  "_module_source.py")),
}
_ORIGINAL_GETSOURCE = inspect.getsource


def _guarded_getsource(obj):
    if inspect.ismodule(obj) and hasattr(obj, "__path__"):
        caller = os.path.normcase(
            os.path.abspath(sys._getframe(1).f_code.co_filename))
        if (caller.startswith(os.path.normcase(_TESTS_ROOT))
                and caller not in _ALLOWED_CALLERS):
            raise AssertionError(
                "测试里对**包** `" + obj.__name__ + "` 调用了 inspect.getsource"
                " —— 它只返回 __init__.py，于是 `assert \"x\" not in src` 会"
                "静默通过（字符串只是搬到了子模块）。改用 "
                "tests/unit/_module_source.py 的 module_source（拼全部子模块）"
                "或 module_sources（逐个分开，用于顺序类断言）。调用方："
                + caller)
    return _ORIGINAL_GETSOURCE(obj)


# 会话开始即生效（conftest 在收集任何测试模块之前加载，所以测试文件里
# `from inspect import getsource` 绑定到的也是这个版本）。
inspect.getsource = _guarded_getsource
