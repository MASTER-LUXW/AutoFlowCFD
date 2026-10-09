"""模态滤波测试（test_modal_filter_order_loss.py、test_modal_filter_modes.py）共用的辅助函数。"""

import numpy as np
import pytest

from tests.unit._filter_mode import (
    reload_filter_modules,
    restore_default_filter_modules,
)


# 本文件原有一份本地 `_reload_with_env`，2026-09-18 抽到
# `tests/unit/_filter_mode.py` 作为唯一事实来源——另外两个滤波测试文件
# 此前根本没有这个能力，于是直接用**默认档**去测"顶模态被压到机器精度"
# 这类性质，默认档一改就全体失败。理由见那个模块的文档。
_reload_with_env = reload_filter_modules


@pytest.fixture(autouse=True)
def _restore_modules():
    """每个用例结束后把两个模块恢复成默认环境下的状态，避免污染其它
    测试文件（它们会 import 这两个模块的模块级常量/算子）。"""
    yield
    restore_default_filter_modules()


def _prism_filter(ops_mod, order):
    return np.asarray(ops_mod.generate_fr_operators(order).filter_prism)


def _real_block(F, order):
    """取滤波矩阵里**真实自由度**的方块子矩阵。

    零填充槽位的滤波行按 `fr/native_padding.py` 的约定是单位阵（填充块
    必须保持初值不变），把它们算进"还剩多少非常数内容"这类判据里，等于
    在问"单位阵有没有清零"。坍缩档 `n_real == n_sps`，这一步是恒等。
    """
    from autoflowcfd.fr.native_padding import real_sps_per_cell

    n_real, _ = real_sps_per_cell(order)
    return F[:n_real, :n_real]


def _expected_legacy_rank(order):
    """legacy 档滤波矩阵在**全局 `(order+1)^3` 宽度**下的期望秩。

    两条棱柱基的"保留集"不同，但**语义相同**（保留 `eta < 1` 的模态，
    即"每根轴/每个因子都没到顶"的那些）：

    * 坍缩档：保留集 `{i,j,k <= order-1}`，大小 `order^3`，没有填充槽位；
    * 原生档：模态是 `psi_ij(r,s) * P_k(t)`，判据 `max(i+j, k) < order`
      的模态数是"三角形次数 <= order-1 的模态数"× `order`
      = `order(order+1)/2 * order`；另外零填充槽位的滤波行是**单位阵**
      （`fr/native_padding.py` 的约定），每个填充槽位贡献 1 个秩。

    实测（2026-09-20）：原生档 P1 秩 3 = 1 + 2 个填充、P2 秩 15 = 6 + 9、
    P3 秩 42 = 18 + 24 —— 与这里的公式逐一对上。也就是说**"每阶损失一整
    阶"这条结论在两条基上都成立**（原生档 P1 的真实保留秩同样只有 1，
    即只剩常数），这正是本文件要钉的事实。
    """
    from autoflowcfd.fr.native_padding import real_sps_per_cell
    from autoflowcfd.fr.native_prism.mode import prism_basis_is_native

    n_global = (order + 1) ** 3
    if not prism_basis_is_native():
        return order ** 3
    n_real, _ = real_sps_per_cell(order)
    kept_real = (order * (order + 1) // 2) * order
    return kept_real + (n_global - n_real)


#: 全部后端标识。哪些**已接线**不在这里硬编码——从
#: `_SENSOR_MODE_SUPPORTED_BACKENDS` 读（同一个语义只允许一个事实来源；
#: 硬编码一份会让"接线完成但测试仍钉着旧状态"这种失败反复出现，
#: 2026-09-18 接线 cpu-mpi 时就是这么被绊了一次）。
_ALL_FILTER_BACKENDS = ("cpu-single", "cpu-mpi", "gpu-single", "gpu-mpi")
