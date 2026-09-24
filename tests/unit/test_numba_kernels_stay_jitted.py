# -*- coding: utf-8 -*-
"""守卫：名为 `*_kernel` 的模块级函数必须是 numba 编译的。

## 为什么要有这个守卫（2026-09-24）

SST 拆包时拆包工具丢掉了 `_strain_vorticity_magnitude_kernel` 的
`@njit(cache=True, parallel=True)`（工具取源码范围从 `def` 行开始，漏了装饰
器行）。**数值逐位不变**——两条黄金轨迹、全量单元测试全部通过——但这个核
退化成纯 Python 三重循环：单次调用 0.002s -> 6.07s，plate_demo 长程运行每步
从 10 秒变成 26 秒。它是所有既有检查都看不见的一类：丢装饰器只改性能、不改
结果，而单元测试的网格太小，慢几千倍也察觉不到。

## 规则

仓库里模块级 `*_kernel` 函数共 24 个，其中没有 numba 装饰器的 4 个全是
`_get_*_kernel` 这种**返回核的工厂函数**（构造并缓存 CUDA/numba 核）。所以
规则可以没有例外：名为 `*_kernel`（含 `*_kernel_colored` 等变体）、且不以
`_get_` 开头的模块级函数，静态上
必须带 numba 装饰器，运行时导入后必须是 numba 的 Dispatcher。

静态与运行时两层都查：静态层不需要导入（覆盖本机无 CuPy 导不进来的 GPU
模块）；运行时层能抓住"装饰器还在但被别的东西覆盖/包裹掉"的情形。
"""

import ast
import importlib
import io
import os
import pathlib

import pytest

_SRC = pathlib.Path(__file__).resolve().parents[2] / "src"
_NUMBA_DECORATORS = ("njit", "jit", "numba.njit", "numba.jit", "cuda.jit",
                     "numba.cuda.jit", "vectorize", "guvectorize",
                     "numba.vectorize", "numba.guvectorize")


def _kernels():
    out = []
    for p in sorted((_SRC / "autoflowcfd").rglob("*.py")):
        if "__pycache__" in p.parts:
            continue
        tree = ast.parse(io.open(p, encoding="utf-8").read())
        mod = p.relative_to(_SRC).with_suffix("").as_posix().replace("/", ".")
        for n in tree.body:
            if not isinstance(n, ast.FunctionDef):
                continue
            # `*_kernel` 与 `*_kernel_colored` 这类变体都算核
            is_kernel = n.name.endswith("_kernel") or "_kernel_" in n.name
            if is_kernel and not n.name.startswith("_get_"):
                decs = [ast.unparse(d.func if isinstance(d, ast.Call) else d)
                        for d in n.decorator_list]
                out.append((mod, n.name, decs))
    return out


_KERNELS = _kernels()


def test_there_are_kernels_to_check():
    """防止判据因为路径变化而静默地"什么都没查"。"""
    assert len(_KERNELS) >= 20, f"只找到 {len(_KERNELS)} 个 *_kernel，判据的扫描范围可能失效了"


@pytest.mark.parametrize("mod,name,decs", _KERNELS,
                         ids=[f"{m.split('.')[-1]}.{n}" for m, n, _d in _KERNELS])
def test_kernel_has_a_numba_decorator(mod, name, decs):
    assert any(d in _NUMBA_DECORATORS for d in decs), (
        f"{mod}.{name} 没有 numba 装饰器（现有：{decs}）—— 它会以纯 Python 运行，"
        f"数值不变但慢几个数量级，黄金轨迹与单元测试都看不出来")


def _importable(mod):
    try:
        return importlib.import_module(mod)
    except ImportError:
        return None


@pytest.mark.parametrize("mod,name,decs", _KERNELS,
                         ids=[f"{m.split('.')[-1]}.{n}" for m, n, _d in _KERNELS])
def test_kernel_is_a_numba_dispatcher_at_runtime(mod, name, decs):
    m = _importable(mod)
    if m is None:
        pytest.skip(f"{mod} 在本机导入不了（多为缺 CuPy 的 GPU 模块），由静态判据覆盖")
    obj = getattr(m, name)
    from numba.core.dispatcher import Dispatcher
    try:
        from numba.cuda.dispatcher import CUDADispatcher
    except Exception:           # 本机没有 CUDA 工具链时这个类可能导入不了
        CUDADispatcher = ()
    assert isinstance(obj, (Dispatcher,) + ((CUDADispatcher,) if CUDADispatcher else ())), (
        f"{mod}.{name} 运行时不是 numba Dispatcher（是 {type(obj).__name__}）")
