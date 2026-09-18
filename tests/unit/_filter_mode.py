"""在指定 `AFCFD_FILTER_MODE` / `AFCFD_FILTER_SIGMA_TOP` 下重载滤波模块。

## 为什么需要这个共享助手

`fr/modal_filter.py` 的 `FILTER_MODE`/`FILTER_ALPHA` 是**模块级常量**，在
import 时求值；`fr/operators.py` 构造滤波矩阵时读它们。所以"测某一档的
矩阵"必须设好环境变量再重载这两个模块。

**做成共享模块而不是各文件抄一份**（项目规则：同一语义只允许一个事实
来源）：`test_modal_filter_order_loss.py` 原本有一份本地
`_reload_with_env`，而 `test_modal_filter.py` / `test_native_tet_filter.py`
根本没有——后两者于是直接用**默认档**去测"滤波器把顶模态压到机器精度"
这类性质。

那是一个真实的测试写法缺陷，2026-09-18 被暴露：默认档从
`legacy`（顶模态清零）改成 `sensor`（顶模态 sigma=0.99 的有界衰减）之后，
那些测试全部失败——它们从来不是在测自己名字里那一档，只是历史上默认档
恰好也清零顶模态。`TestLegacyFilterLosesExactlyOneOrder` 尤其明显：类名
写着 legacy，用的却是 `AFCFD_FILTER_MODE=None`。

所以规则是：**任何刻画"某一档矩阵长什么样"的测试，必须显式指定那一档**；
只有专门检查"默认值是什么"的测试才可以不指定。
"""

import contextlib
import importlib
import os


def reload_filter_modules(**env):
    """在给定环境变量下重载 `modal_filter` 与 `operators`，返回两个模块。

    `None` 表示删除该环境变量（用默认值）。调用结束时环境变量被还原，
    但**模块保持在新状态**——需要恢复默认请用 `filter_mode` 上下文管理器
    或各文件里的 autouse fixture。

    Args:
        **env: 例如 `AFCFD_FILTER_MODE="legacy"`,
            `AFCFD_FILTER_SIGMA_TOP="0.99"`

    Returns:
        `(modal_filter 模块, operators 模块)`
    """
    old = {k: os.environ.get(k) for k in env}
    try:
        for k, v in env.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        import autoflowcfd.fr.modal_filter as mf
        importlib.reload(mf)
        import autoflowcfd.fr.operators as ops_mod
        importlib.reload(ops_mod)
        # native 四面体滤波器读的是 modal_filter 的模块级常量，但它自己
        # 缓存了矩阵（按 order），重载后必须清掉，否则拿到上一档的矩阵。
        import autoflowcfd.fr.native_tet_filter as ntf
        importlib.reload(ntf)
        return mf, ops_mod
    finally:
        for k, v in old.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def restore_default_filter_modules():
    """把两个模块恢复成**默认环境**下的状态。

    必须在每个改过档位的用例之后调用：其它测试文件 import 的是这两个
    模块的模块级常量/算子，留在非默认状态会污染它们。
    """
    reload_filter_modules(AFCFD_FILTER_MODE=None, AFCFD_FILTER_SIGMA_TOP=None)


@contextlib.contextmanager
def filter_mode(mode=None, sigma_top=None):
    """`with filter_mode("legacy"): ...` —— 退出时自动恢复默认档。"""
    env = {"AFCFD_FILTER_MODE": mode}
    if sigma_top is not None:
        env["AFCFD_FILTER_SIGMA_TOP"] = str(sigma_top)
    try:
        yield reload_filter_modules(**env)
    finally:
        restore_default_filter_modules()
