# AutoFlowCFD 安装

依赖以 `pyproject.toml` 为准（`[tool.poetry.dependencies]` 与 `[tool.poetry.extras]`）；`autoflowcfd utils doctor` 按同一份
声明逐项核对已安装的版本。

---

## 1. 用 Poetry 安装（推荐）

```bash
pip install poetry
git clone https://github.com/AutoFlowCFD/AutoFlowCFD.git
cd AutoFlowCFD

poetry install              # 核心依赖 + 开发工具（pytest、black、mypy……）
poetry install -E gpu       # 另装 GPU 后端：cupy-cuda12x
poetry install -E mpi       # 另装分布式求解：mpi4py、pymetis（还需系统 MPI：OpenMPI / MPICH / MS-MPI）
poetry install -E perf      # 另装 psutil（utils benchmark 的内存统计）
```

国内网络可在 `pyproject.toml` 所在目录配置镜像源后再安装：

```bash
poetry source add --priority=primary tuna https://pypi.tuna.tsinghua.edu.cn/simple
```

## 2. 用 pip 安装（已有环境）

```bash
pip install -e .                 # 核心依赖
pip install -e ".[gpu,mpi,perf]" # 按需加可选组
```

---

## 依赖

### 核心（必需）

| 包 | 版本约束 | 用途 |
|---|---|---|
| numpy | ^1.26 | 数组计算 |
| scipy | ^1.10 | 稀疏线性代数、KD 树、信号处理（Welch 谱） |
| numba / llvmlite | ^0.60 / 0.43.0 | CPU 计算核（JIT 编译、`prange` 并行） |
| tetgen | ^0.8.4 | 体网格生成的核心区四面体 |
| pyvista | ^0.44 | VTK 读写（体网格 / 高阶单元导出） |
| h5py | ^3.9 | HDF5 checkpoint |
| click | ^8.1 | 命令行 |
| pyyaml | ^6.0 | YAML 配置文件 |
| loguru | ^0.7 | 日志 |
| packaging | >=23.0 | `utils doctor` 解析依赖声明 |

### 可选组

| 组 | 包 | 用途 |
|---|---|---|
| `gpu` | cupy-cuda12x ^12.2 | 单 GPU / 多 GPU 后端（需要 NVIDIA 驱动与 CUDA 12.x） |
| `mpi` | mpi4py >=3.1、pymetis >=2023.1 | `--n-ranks > 1` 的进程间通信与 METIS 图划分 |
| `perf` | psutil >=5.9 | `utils benchmark` 的内存统计 |

Python 版本：3.10 或 3.11（scipy 的声明只覆盖 Python < 3.12）。

---

## 验证

```bash
poetry run autoflowcfd --version     # AutoFlowCFD, version 0.1.0
poetry run autoflowcfd utils doctor  # 必需依赖缺失或版本不符为问题，可选组缺失为警告；另报告 GPU / MPI 是否可用
poetry run pytest tests/unit -q
```

`pyproject.toml` 的 pytest 缺省选项带覆盖率统计（`--cov`）；只想快速跑测试时可加 `-o addopts=` 覆盖掉。

GPU 是否可用：

```bash
poetry run python -c "import cupy; print(cupy.cuda.runtime.getDeviceCount())"   # 应大于 0
```

MPI 是否可用（两个进程各打印自己的 rank）：

```bash
mpirun -np 2 poetry run python -c "from mpi4py import MPI; print(MPI.COMM_WORLD.Get_rank())"
```

---

## 常见问题

**h5py 在 Linux 上编译失败** —— 先装 HDF5 开发包（Debian/Ubuntu：`sudo apt-get install libhdf5-dev`）。

**CuPy 装不上或运行时报找不到 CUDA** —— 确认 NVIDIA 驱动与 CUDA 12.x 已安装、`nvidia-smi` 能看到设备；CuPy 的包名与
CUDA 大版本对应（本项目声明的是 `cupy-cuda12x`）。

**mpi4py 装不上** —— 需要先有系统 MPI 及其编译器包装（`mpicc`）；Windows 用 MS-MPI。

**numba 首次运行很慢** —— 计算核首次调用时编译并写入磁盘缓存（缓存目录按核源码内容区分，源码改动后自动重新编译），
之后的运行直接加载。

---

下一步：[QUICKSTART.md](QUICKSTART.md)；开发者见 [docs/DEVELOPER_GUIDE.md](docs/DEVELOPER_GUIDE.md)。
