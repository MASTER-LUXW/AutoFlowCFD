# -*- coding: utf-8 -*-
"""单进程测试用的 halo 交换替身（CPU `HaloExchange` / GPU `GPUHaloExchange` 共用）。

真实交换器同一个实例交换各种逐单元形状的数据（平均流 `(n_sps, n_vars)`、k/omega
`(n_sps, 2)`、DES 长度尺度 `(n_sps,)`、P0 速度梯度 `(n_sps, 3, 3)`），见
`core/mpi/halo.py` 模块文档。替身按输入的逐单元形状返回预先拼好的 local+halo 数组
（从全局真值切出来的，等价于另一个 rank 已经算好并交换过来）。
"""


class ShapeKeyedFakeHalo:
    """`exchange(local)` 返回逐单元形状与 `local` 相同的那份预拼数组。"""

    def __init__(self, *extended_arrays):
        self._by_shape = {}
        for a in extended_arrays:
            shape = tuple(a.shape[1:])
            if shape in self._by_shape:
                raise ValueError(f"两份预拼数组的逐单元形状相同 {shape}，替身无法区分")
            self._by_shape[shape] = a

    def set(self, extended):
        """替换（或新增）某一逐单元形状的预拼数组（多步测试里每步更新）。"""
        self._by_shape[tuple(extended.shape[1:])] = extended

    def exchange(self, local):
        return self._by_shape[tuple(local.shape[1:])]
