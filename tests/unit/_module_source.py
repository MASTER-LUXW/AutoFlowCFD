# -*- coding: utf-8 -*-
"""读取模块源码用于文本断言 —— **包要拼上全部子模块**。

## 为什么必须有这一份共用实现（2026-09-24）

`inspect.getsource(package)` 只返回 `__init__.py`。本项目把超 500 行的
模块陆续拆成子包，于是所有"读源码做文本断言"的测试都受影响，而两种失效
方式的危害完全不同：

* `assert "xxx" in src` —— 拆包后**直接失败**，能被发现；
* `assert "xxx" not in src` —— 拆包后**静默通过**，理由却是错的：那个字符串
  只是搬到了子模块，并不是真的不存在。**这比没有测试更糟**。

同一个坑本会话内撞了 4 次（`order_interp` 的四组参数化、`test_viscous_*`
的两组、`test_omega_wall_blended_mode` 一处、`test_overintegration_
unpadded_fine_axis` 的硬编码文件路径清单）。记在
`ProjectFiles/V2.0/29_transport拆包与SST黄金轨迹判据.md` 三·五 (4)。

另一种等价的失效是**按文件路径**读源码（`root / "src/.../gpu_viscous.py"`）：
拆包后那个 `.py` 不存在了，`read_text` 抛 FileNotFoundError。所以本模块
一律按**模块名**取源码，不用文件路径 —— 模块名在拆包前后都是稳定的。
"""

import importlib
import inspect
import pkgutil


def module_source(mod, *, recursive=True):
    """模块源码；若 `mod` 是包，则拼接它（递归的）全部子模块源码。

    Args:
        mod: 模块对象，或可导入的模块名字符串。
        recursive: True 时连子包的子模块一起拼（本项目有 `core/gpu/
            residual/gpu_viscous` 这类嵌套情形）。

    Returns:
        拼接后的源码字符串。
    """
    if isinstance(mod, str):
        mod = importlib.import_module(mod)
    parts = [inspect.getsource(mod)]
    path = getattr(mod, "__path__", None)
    if path is not None:
        for info in pkgutil.iter_modules(path):
            sub = importlib.import_module(mod.__name__ + "." + info.name)
            if info.ispkg and recursive:
                parts.append(module_source(sub, recursive=True))
            else:
                parts.append(inspect.getsource(sub))
    return "\n".join(parts)
