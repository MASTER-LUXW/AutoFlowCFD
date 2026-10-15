"""AutoFlowCFD V2.0 - 本进程还能用多少内存（主机或 GPU 设备）。

隐式步的单元块预处理按它定档（`time_integration/implicit/block_budget.py`）：块预处理是 Newton 步里最大的
一块内存，此前用写死的 11 GiB 合计预算选档，不看求解器其余部分占了多少——cube_demo 粗网格（19.6 万单元）
SA P2 在 31.5 GB 的机器上平均流选了"两份块 Jacobi"（9.12 GiB），第 2 步 GMRES 时进程 29.4 GB、内存不足，
而"单份"档（4.56 GiB）放得下。

不引入新依赖（psutil 不是项目依赖）：Windows 用 `GlobalMemoryStatusEx`，Linux 读 `/proc/meminfo` 的
`MemAvailable`，其余 POSIX 用 `sysconf`；GPU 用 CuPy 的设备空闲内存加内存池里已缓存、可复用的空闲块。
"""

import os
import sys


def _host_available_bytes() -> int:
    """主机当前可分配而不换页的物理内存（字节）。"""
    if sys.platform == "win32":
        import ctypes
        from ctypes import wintypes

        class _MemoryStatusEx(ctypes.Structure):
            _fields_ = [("dwLength", wintypes.DWORD), ("dwMemoryLoad", wintypes.DWORD),
                        ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                        ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                        ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                        ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]

        status = _MemoryStatusEx()
        status.dwLength = ctypes.sizeof(_MemoryStatusEx)
        if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
            raise OSError("GlobalMemoryStatusEx 调用失败，无法取得可用物理内存")
        return int(status.ullAvailPhys)
    meminfo = "/proc/meminfo"
    if os.path.exists(meminfo):
        with open(meminfo, encoding="ascii") as f:
            for line in f:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) * 1024
        raise OSError(f"{meminfo} 里没有 MemAvailable（内核早于 3.14），无法取得可用物理内存")
    return int(os.sysconf("SC_AVPHYS_PAGES")) * int(os.sysconf("SC_PAGE_SIZE"))


def _device_available_bytes(xp) -> int:
    """当前 GPU 设备的空闲内存加 CuPy 内存池里已缓存的空闲块（字节）。"""
    free, _total = xp.cuda.runtime.memGetInfo()
    return int(free) + int(xp.get_default_memory_pool().free_bytes())


def _ranks_sharing_host() -> int:
    """与本进程共享同一台主机内存的 MPI rank 数（本进程没有用 MPI 或单进程时为 1）。

    只在本进程已经加载了 `mpi4py.MPI` 时查询：导入它默认就会初始化 MPI，单机运行不该有这个副作用；分布式
    运行在构造求解器之前早已导入。"""
    MPI = sys.modules.get("mpi4py.MPI")
    if MPI is None or not MPI.Is_initialized() or MPI.COMM_WORLD.Get_size() == 1:
        return 1
    node = MPI.COMM_WORLD.Split_type(MPI.COMM_TYPE_SHARED)
    try:
        return int(node.Get_size())
    finally:
        node.Free()


def available_bytes(xp) -> int:
    """本进程（本 rank）可以再分配的内存：`xp` 是 numpy 时为主机内存按同一主机上的 rank 数均分，是 cupy 时为
    当前设备内存（多 GPU 每个 rank 独占一块设备）。"""
    import numpy as np

    if xp is np:
        return _host_available_bytes() // _ranks_sharing_host()
    return _device_available_bytes(xp)
