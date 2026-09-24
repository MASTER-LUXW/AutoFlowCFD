# -*- coding: utf-8 -*-
"""`solve transient` 的四条路径都必须收到并转发攻角与三个 CFL 边界。

## 这里钉住的两个真实缺陷（2026-09-24 发现）

排查中用 `flake8 --select=F821` 扫全仓库时发现：

    src/autoflowcfd/cli/solve_transient_distributed.py:221: F821 undefined name 'aoa_deg'
    src/autoflowcfd/cli/solve_transient_distributed.py:325: F821 undefined name 'aoa_deg'

1. **硬崩溃**：`_solve_transient_fully_distributed` 与
   `_solve_transient_multi_gpu` 的函数体里就在用 `aoa_deg`/`aos_deg`
   （下方那个 `freestream = {...}` 字典），却都不是形参，分发处也没转发。
   于是 `solve transient --fully-distributed` 与 `solve transient
   --multi-gpu` **100% 必现 `NameError`** —— 两条路径完全跑不起来。

2. **静默丢弃**：`--cfl-start/--cfl-max/--cfl-min` 从未转发到这两条路径。
   `solve steady` 的四处分布式构造点已于 2026-09-15 补齐（见
   `solve_steady_command.py` 里同主题注释：「此前在**全部分布式路径**上
   被静默丢弃」），`solve transient` 的这两条当时被漏掉了。这类缺陷在
   日志里完全看不出来，只表现为"我明明设了 CFL 0.01 却还是按默认值跑"。

两者是同一个根因的两面：这四条路径的参数转发从来没有被逐项核对过。所以
这里按**形参 + 转发 + 真正到达消费点**三层分别钉住，而不是只测其中一层。

## 为什么用静态检查

让这四条路径真的跑起来需要 MPI + 多 GPU 真实硬件（本机没有），而"参数有
没有接上"是个结构性事实，AST 检查就足够，而且能在将来新增第五条路径时
立刻提醒作者。
"""

import ast
import io
import pathlib

import pytest

_SRC = pathlib.Path(__file__).resolve().parents[2] / (
    "src/autoflowcfd/cli/solve_transient_distributed.py")
_TREE = ast.parse(io.open(_SRC, encoding="utf-8").read())
_FUNCS = {n.name: n for n in _TREE.body if isinstance(n, ast.FunctionDef)}

#: 五个必须逐层接上的参数
_KEYS = ("aoa_deg", "aos_deg", "cfl_start", "cfl_max", "cfl_min")

#: 四条路径。`_solve_transient_distributed` 是分发入口，其余三条是实现。
_ENTRIES = [
    "_solve_transient_distributed",
    "_solve_transient_cpu_traditional",
    "_solve_transient_fully_distributed",
    "_solve_transient_multi_gpu",
]


@pytest.mark.parametrize("fname", _ENTRIES)
@pytest.mark.parametrize("key", _KEYS)
def test_entry_declares_the_parameter(fname, key):
    """第一层：形参必须存在。

    缺形参时，若函数体又在用这个名字，就是必现 `NameError`（缺陷 1）；
    若函数体不用它，则是静默丢弃（缺陷 2）。两种都要拦。
    """
    fn = _FUNCS[fname]
    args = {a.arg for a in fn.args.args} | {a.arg for a in fn.args.kwonlyargs}
    assert key in args, (
        f"{fname} 没有 `{key}` 形参 —— 若函数体里用到它就是必现 NameError，"
        f"否则是用户设的值被静默丢弃")


@pytest.mark.parametrize("key", _KEYS)
def test_dispatcher_forwards_to_every_branch(key):
    """第二层：分发入口必须把参数转发给**全部三个**实现分支。

    缺陷 1/2 正是栽在这里：分发处只给 `_solve_transient_cpu_traditional`
    传了这五个，另外两个分支一个都没给。
    """
    disp = _FUNCS["_solve_transient_distributed"]
    branches = ("_solve_transient_cpu_traditional",
                "_solve_transient_fully_distributed",
                "_solve_transient_multi_gpu")
    seen = {b: False for b in branches}
    for n in ast.walk(disp):
        if not (isinstance(n, ast.Call) and isinstance(n.func, ast.Name)):
            continue
        if n.func.id not in seen:
            continue
        kws = {k.arg for k in n.keywords if k.arg}
        pos = {a.id for a in n.args if isinstance(a, ast.Name)}
        seen[n.func.id] = key in kws or key in pos
    missing = [b for b, ok in seen.items() if not ok]
    assert not missing, (
        f"分发入口没有把 `{key}` 转发给：{missing} —— 这正是 2026-09-24 那两个"
        f"缺陷的位置（aoa/aos 必现 NameError、CFL 静默丢弃）")


def test_freestream_dicts_all_carry_the_angles():
    """第三层（攻角）：每个 `freestream = {...}` 字典都必须带上两个角。

    这是 `aoa_deg` 的真正消费点 —— 它决定初场速度方向。少了它，
    `solve transient` 的那条路径要么崩溃、要么静默退回零攻角。
    """
    src = io.open(_SRC, encoding="utf-8").read()
    n_dicts = src.count('freestream = {"rho_inf"')
    # 本文件里有两处 `freestream = {...}`（完全分布式、多 GPU 完全分布式）；
    # CPU 传统模式不建这个字典，它把两个角**直接**传给 `DistributedFRSolver`
    # 构造函数，由上面 test_entry_declares_the_parameter 与
    # test_dispatcher_forwards_to_every_branch 覆盖。
    assert n_dicts == 2, (
        f"freestream 字典数变成了 {n_dicts}（原为 2）——新增/删除了路径，"
        f"判据需要跟着更新")
    for key in ("aoa_deg", "aos_deg"):
        n = src.count(f'"{key}": {key}')
        assert n == n_dicts, (
            f"{n_dicts} 个 freestream 字典里只有 {n} 个带 {key} —— 漏掉的那条"
            f"路径要么必现 NameError，要么静默退回零攻角")


@pytest.mark.parametrize("key", ("cfl_start", "cfl_max", "cfl_min"))
def test_every_solver_construction_site_receives_cfl(key):
    """第三层（CFL）：每个求解器/package 构造点都必须真正收到三个边界。

    构造点有三类：`distributed_mesh_load_v2`（完全分布式，靠 package 把
    值送到各 rank）、`MultiGPUDistributedSolver(...)`、以及 CPU 传统模式的
    `DistributedFRSolver(...)`。只要有一处漏掉，那条路径就会静默使用
    `AdaptiveCFLController` 的签名默认值 —— 而默认值的单一事实来源就该是
    那个签名，所以这里查的是"有没有传"，不是"传了什么字面量"。
    """
    src = io.open(_SRC, encoding="utf-8").read()
    sites = ("distributed_mesh_load_v2(", "MultiGPUDistributedSolver(",
             "DistributedFRSolver(")
    total = sum(src.count(x) for x in sites)
    assert total >= 3, f"构造点只找到 {total} 处，判据需要更新"
    n = src.count(key + "=" + key)
    assert n >= total, (
        f"`{key}={key}` 只出现 {n} 次，而构造点/转发点共 {total} 处 —— "
        f"很可能有路径漏传，那会让它静默使用控制器默认值")
