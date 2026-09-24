# -*- coding: utf-8 -*-
"""守卫：测试自己不许用"拆包后会失效"的引用方式。

## 为什么要有这个守卫（2026-09-24）

本项目把超 500 行的模块陆续拆成子包。每拆一批，都有一批测试因为**引用
方式**而失效，而且失效方式的危害差别极大：

| 写法 | 拆包后 |
|---|---|
| `(root / "src/.../yyy.py").read_text()` | FileNotFoundError（能发现） |
| `inspect.getsource(pkg)` + `assert "x" in src` | 直接失败（能发现） |
| `inspect.getsource(pkg)` + `assert "x" not in src` | **静默通过**，理由却是错的 |
| `monkeypatch.setattr(pkg, "N", fake)`，N 未被 re-export | AttributeError（能发现） |
| `monkeypatch.setattr(pkg, "N", fake)`，N 被 re-export | **静默失效**，替身没装上 |

两种"静默"比没有测试更糟 —— 它们让对照臂看起来通过了。2026-09-24 这一轮
真实撞上：`test_bounds_sensor_mirror` 的"表只重建一次"计数恒为 0（setattr
成功、但真正的调用在同一个子模块内部），以及 `test_flow_direction` 的两条
否定式断言。所以把这三类写法做成守卫，而不是每次拆包再逐个救火。

## 三条规则

1. 测试里硬编码的 `src/autoflowcfd/....py` 路径必须真实存在 —— 更推荐
   直接换成模块名 + `tests/unit/_module_source.py::module_source`。
2. 不许对**包**调用 `inspect.getsource` —— 它只返回 `__init__.py`。
   用 `module_source`（拼全部子模块）或 `module_sources`（逐个分开，
   用于顺序类断言）。
3. 不许对**包**做裸 `monkeypatch.setattr` / `patch.object` —— 用
   `tests/unit/_patch_pkg.py::patch_pkg_attr`，它把包和全部子模块一起
   换掉（两种取名方式都覆盖），并用聚合断言保证至少换上了一处。
"""

import ast
import io
import os
import pathlib

import pytest

_ROOT = pathlib.Path(__file__).resolve().parents[2]


def _all_packages():
    """`src/` 下所有包的导入路径。"""
    out = set()
    src = _ROOT / "src"
    for root, dirs, files in os.walk(src):
        dirs[:] = [d for d in dirs if d != "__pycache__"]
        if "__init__.py" in files and pathlib.Path(root) != src:
            rel = pathlib.Path(root).relative_to(src).as_posix()
            out.add(rel.replace("/", "."))
    return out


def _test_files():
    out = []
    for root, dirs, files in os.walk(_ROOT / "tests"):
        dirs[:] = [d for d in dirs if d != "__pycache__"]
        for f in files:
            if f.endswith(".py"):
                out.append(pathlib.Path(root) / f)
    return sorted(out)


def _module_aliases(tree):
    """测试文件里 `名字 -> 它绑定的模块全名` 的映射（含函数体内的 import）。"""
    a = {}
    for n in ast.walk(tree):
        if isinstance(n, ast.Import):
            for x in n.names:
                a[x.asname or x.name.split(".")[0]] = x.name
        elif isinstance(n, ast.ImportFrom) and n.module and n.level == 0:
            for x in n.names:
                a[x.asname or x.name] = n.module + "." + x.name
    return a


def _resolve(node, aliases):
    if isinstance(node, ast.Name):
        return aliases.get(node.id)
    if isinstance(node, ast.Attribute):
        parts = []
        cur = node
        while isinstance(cur, ast.Attribute):
            parts.append(cur.attr)
            cur = cur.value
        if isinstance(cur, ast.Name):
            base = aliases.get(cur.id) or cur.id
            return ".".join([base] + list(reversed(parts)))
    return None


_FILES = _test_files()
_IDS = [p.relative_to(_ROOT).as_posix() for p in _FILES]


@pytest.mark.parametrize("path", _FILES, ids=_IDS)
def test_no_dead_src_file_paths(path):
    """规则 1：硬编码的源文件路径必须存在。"""
    tree = ast.parse(io.open(path, encoding="utf-8").read())
    bad = []
    for n in ast.walk(tree):
        if isinstance(n, ast.Constant) and isinstance(n.value, str):
            v = n.value
            if v.startswith("src/autoflowcfd/") and v.endswith(".py"):
                if not (_ROOT / v).exists():
                    bad.append((n.lineno, v))
    assert not bad, (
        "硬编码的源文件路径已不存在（很可能是那个模块被拆成了子包）："
        + "；".join(f"第 {ln} 行 {v}" for ln, v in bad)
        + "。改成模块名 + tests/unit/_module_source.py::module_source，"
          "模块名跨拆包稳定，且会自动把新增子模块拼进来")


@pytest.mark.parametrize("path", _FILES, ids=_IDS)
def test_no_getsource_on_a_package(path):
    """规则 2：不许对包调用 `inspect.getsource`。"""
    tree = ast.parse(io.open(path, encoding="utf-8").read())
    aliases = _module_aliases(tree)
    pkgs = _all_packages()
    bad = []
    for n in ast.walk(tree):
        if (isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
                and n.func.attr == "getsource" and len(n.args) == 1):
            m = _resolve(n.args[0], aliases)
            if m in pkgs:
                bad.append((n.lineno, m))
    assert not bad, (
        "对**包**调用了 inspect.getsource —— 它只返回 `__init__.py`，"
        "于是 `assert \"x\" not in src` 会**静默通过**（字符串只是搬到了"
        "子模块）："
        + "；".join(f"第 {ln} 行 {m}" for ln, m in bad)
        + "。改用 tests/unit/_module_source.py 的 module_source（拼全部"
          "子模块）或 module_sources（逐个分开，用于顺序类断言）")


@pytest.mark.parametrize("path", _FILES, ids=_IDS)
def test_no_bare_setattr_on_a_package(path):
    """规则 3：不许对包做裸 setattr / patch.object。"""
    tree = ast.parse(io.open(path, encoding="utf-8").read())
    aliases = _module_aliases(tree)
    pkgs = _all_packages()
    bad = []
    for n in ast.walk(tree):
        if (isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
                and n.func.attr in ("setattr", "delattr", "object")
                and len(n.args) >= 2
                and isinstance(n.args[1], ast.Constant)
                and isinstance(n.args[1].value, str)):
            m = _resolve(n.args[0], aliases)
            if m in pkgs:
                bad.append((n.lineno, m, n.args[1].value))
    assert not bad, (
        "对**包**做了裸 setattr/patch.object —— 若那个名字被 `__init__` "
        "re-export 了，setattr 会**成功**但子模块内部的引用毫发无损，"
        "patch 静默失效而测试照样通过："
        + "；".join(f"第 {ln} 行 {m}.{k}" for ln, m, k in bad)
        + "。改用 tests/unit/_patch_pkg.py::patch_pkg_attr —— 它把包和"
          "全部子模块一起换掉，并用聚合断言保证至少换上了一处")


# ---------------------------------------------------------------------------
# 运行时守卫（tests/unit/conftest.py）本身的判据
# ---------------------------------------------------------------------------


def test_runtime_guard_rejects_getsource_on_a_package():
    """运行时守卫必须真的咬人 —— 否则它就是个摆设。

    用循环变量取模块，正是静态规则解析不了、2026-09-24 真实漏掉的写法。
    """
    import inspect

    import autoflowcfd.core.fr_solver.boundary as pkg
    for mod in (pkg,):
        with pytest.raises(AssertionError, match="只返回 __init__.py"):
            inspect.getsource(mod)


def test_runtime_guard_lets_module_source_through():
    """`module_source` 自己要读包的 __init__.py，不能被拦。"""
    from tests.unit._module_source import module_source, module_sources

    src = module_source("autoflowcfd.core.fr_solver.boundary")
    assert "def build_boundary_ghost_provider(" in src
    names = [n for n, _s in module_sources("autoflowcfd.core.fr_solver.boundary")]
    assert "autoflowcfd.core.fr_solver.boundary.ghost" in names


def test_runtime_guard_leaves_plain_modules_and_functions_alone():
    """对普通模块、函数、类调用 getsource 是合法的，不能误伤。"""
    import inspect

    from autoflowcfd.core.fr_solver.boundary import ghost
    from autoflowcfd.core.fr_solver.boundary.ghost import (
        build_boundary_ghost_provider,
    )
    assert "build_boundary_ghost_provider" in inspect.getsource(ghost)
    assert inspect.getsource(build_boundary_ghost_provider).startswith("def ")
