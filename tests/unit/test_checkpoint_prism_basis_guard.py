"""checkpoint 必须记录棱柱基，非原生基的 checkpoint 必须**硬失败**。

## 这份测试钉的是一个真实危险组合

原生棱柱基 P1 每单元只有前 6 个槽位是自由度（其余按
`fr/native_padding.py` 约定复制真实 SP#0），已删除的坍缩基 8 个全是自由
度。同一份 `(n_cells, n_sps, n_vars)` 数组在两条基下**含义不同**，而
**形状恰好相同** —— 所以跨基 resume 既不会报形状不符、也不会报任何错，
只是静默给出一个看不出异常的错解。

这个组合在 2026-09-20 默认值从 `collapsed` 改成 `native` 的那一刻变成
现实：磁盘上所有既有 checkpoint 都是 collapsed 时代写的。坍缩棱柱基已于
2026-09-23 整体删除，但**这条护栏刻意保留**：磁盘上那些 checkpoint 不会
随代码删除而消失，它们必须继续被拒绝，而不是被静默重解释。

判据三条：写入记的是 `"native"`；同基放行；`"collapsed"` 标签与属性缺失
两种历史 checkpoint 都拒绝。
"""

import h5py
import numpy as np
import pytest

from autoflowcfd.core.utils.checkpoint import CheckpointManager
from autoflowcfd.core.utils.checkpoint_load import load_checkpoint


def _manager(tmp_path):
    """最小 CheckpointManager：`config` 只被 `_compute_config_hash` 读，
    用 `getattr(..., 默认值)` 取，所以给一个空对象即可。"""
    from types import SimpleNamespace

    return CheckpointManager(config=SimpleNamespace(),
                             output_dir=str(tmp_path), quiet=True)


def _write(tmp_path):
    """写一份当前（原生）基下的 checkpoint。

    不再有 `basis` 形参：`AFCFD_PRISM_BASIS=collapsed` 现在会让
    `resolve_prism_basis_mode()` 直接报错（坍缩棱柱基已删除），所以坍缩
    时代的 checkpoint 只能靠下面 `_relabel` 直接改 HDF5 属性来伪造。
    """
    mgr = _manager(tmp_path)
    path = mgr.save(
        solution=np.ones((3, 5)),
        # history 的 schema 见 `CheckpointManager.save` 文档：
        # residuals 是 {方程名: [值]} 的字典，不是列表。
        history={"iterations": [1, 2],
                 "residuals": {"rho": [1.0, 0.5]},
                 "coefficients": {},
                 "cfl_history": [0.03, 0.03]},
        iteration=7,
    )
    assert path is not None
    return path


def _relabel(path, basis):
    """把已写好的 checkpoint 的 `prism_basis` 属性改成 `basis`；
    `basis is None` 表示删掉这个属性（模拟 2026-09-20 之前写的文件）。"""
    with h5py.File(path, "r+") as f:
        if basis is None:
            del f["metadata"].attrs["prism_basis"]
        else:
            f["metadata"].attrs["prism_basis"] = np.string_(basis)


def test_basis_is_recorded(tmp_path):
    """保存时必须把生效的棱柱基写进元数据 —— 删掉坍缩档之后**仍然要写**：
    这个标签是未来任何一次基变更的唯一判据来源。"""
    path = _write(tmp_path)
    with h5py.File(path, "r") as f:
        saved = f["metadata"].attrs["prism_basis"]
    if isinstance(saved, bytes):
        saved = saved.decode("utf-8")
    assert saved == "native"


def test_same_basis_loads(tmp_path):
    """当前基下写的 checkpoint 正常加载（护栏不能把正常路径也堵掉）。"""
    path = _write(tmp_path)
    mgr = _manager(tmp_path)
    sol, _hist, it, _meta = load_checkpoint(mgr, path)
    assert it == 7
    assert sol.shape == (3, 5)


def test_collapsed_labelled_checkpoint_is_refused(tmp_path):
    """**核心判据**：坍缩时代的 checkpoint 必须抛错，不是 warning。

    与 `config_hash` 不一致刻意不同级别：配置哈希不一致只是"可能影响
    结果"（例如 CFL 上限变了），棱柱基不一致是"这份数据的含义变了"。
    """
    path = _write(tmp_path)
    _relabel(path, "collapsed")
    mgr = _manager(tmp_path)
    with pytest.raises(RuntimeError, match="棱柱基"):
        load_checkpoint(mgr, path)


def test_missing_attr_is_refused(tmp_path):
    """属性缺失（2026-09-20 之前写的 checkpoint）同样拒绝。

    那之前**唯一**的默认值是 collapsed，所以"缺失"等价于"坍缩"。
    """
    path = _write(tmp_path)
    _relabel(path, None)
    mgr = _manager(tmp_path)
    with pytest.raises(RuntimeError, match="没有记录棱柱基"):
        load_checkpoint(mgr, path)


def test_refusal_does_not_advise_the_deleted_collapsed_mode(tmp_path):
    """报错信息不能再教用户"设 AFCFD_PRISM_BASIS=collapsed 续算" ——
    那条建议在 2026-09-23 删除坍缩基之后是**错的建议**（照做只会撞上
    `resolve_prism_basis_mode` 的报错），而错的建议比没有建议更费时间。
    """
    mgr = _manager(tmp_path)
    for label in ("collapsed", None):
        path = _write(tmp_path)
        _relabel(path, label)
        with pytest.raises(RuntimeError) as exc:
            load_checkpoint(mgr, path)
        msg = str(exc.value)
        assert "AFCFD_PRISM_BASIS=collapsed" not in msg
        assert "从头开始" in msg


def test_collapsed_mode_cannot_be_requested_at_all():
    """护栏的另一半：环境变量本身已经不接受 `collapsed`。

    否则"拒绝旧 checkpoint"这条判据可以被一次 `AFCFD_PRISM_BASIS=collapsed`
    绕过成"两边都是 collapsed、放行"。
    """
    import os

    from autoflowcfd.fr.native_prism.mode import resolve_prism_basis_mode

    old = os.environ.get("AFCFD_PRISM_BASIS")
    os.environ["AFCFD_PRISM_BASIS"] = "collapsed"
    try:
        with pytest.raises(ValueError, match="collapsed"):
            resolve_prism_basis_mode()
    finally:
        if old is None:
            os.environ.pop("AFCFD_PRISM_BASIS", None)
        else:
            os.environ["AFCFD_PRISM_BASIS"] = old
