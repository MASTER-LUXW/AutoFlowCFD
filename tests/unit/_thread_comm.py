"""本机没有 MPI 时的多 rank 模拟：每个线程就是一个 rank，同时执行同一段代码，集合通信用
`threading.Barrier` 实现（全部 rank 进入、全部 rank 得到同一结果，与真实 MPI 的集合语义相同）。"""

import threading

import numpy as np


class ThreadComm:
    """n 个线程之间的集合通信。"""

    def __init__(self, n):
        self.n = n
        self.barrier = threading.Barrier(n)
        self.slots = [None] * n

    def collective(self, rank, value, combine):
        """各 rank 交出 `value`，全部到齐后各自得到 `combine(按 rank 顺序的列表)`。"""
        self.slots[rank] = value
        self.barrier.wait()
        out = combine(list(self.slots))
        self.barrier.wait()
        return out

    def allgather(self, rank, a):
        return self.collective(rank, np.asarray(a), np.concatenate)

    def run(self, fn):
        """`fn(rank)` 在 n 个线程里同时执行，返回按 rank 排列的结果；任一线程出错时中止屏障
        并在主线程重新抛出。"""
        out, errors = [None] * self.n, []

        def work(r):
            try:
                out[r] = fn(r)
            except Exception as e:  # 线程里的异常带回主线程
                errors.append(e)
                self.barrier.abort()

        threads = [threading.Thread(target=work, args=(r,)) for r in range(self.n)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        if errors:
            raise errors[0]
        return out
