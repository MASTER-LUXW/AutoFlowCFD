"""AutoFlowCFD V2.0 - 从 checkpoint 恢复湍流场时"缺了哪些字段"的判定与提示（单机与分布式恢复端共用）。

单机（`cli/solve/checkpoint_io/restore.py::restore_turbulence_from_fields`）与分布式
（`core/mpi/distributed_checkpoint/turbulence.py::restore_turbulence_fields`）的恢复主体不同（前者核对当前几何的
形状，后者按本 rank 的 local 段核对并上传设备），判定规则与提示文字是同一份。2026-10-22（I-224）以前两边各写
一份且已分叉：单机只在模型有 `nu_t` 属性时才把它算作必需字段，分布式无条件算。
"""

from typing import Iterable, Tuple


def missing_turbulence_fields(transported: Tuple[str, ...], with_nu_t: bool,
                              available: Iterable[str]) -> Tuple[Tuple[str, ...], bool]:
    """`(缺少的字段, 输运场是否不全)`。`with_nu_t`：模型带涡粘场（`nu_t` 也是要恢复的字段）。"""
    have = set(available)
    needed = tuple(transported) + (("nu_t",) if with_nu_t else ())
    missing = tuple(name for name in needed if name not in have)
    return missing, any(name in missing for name in transported)


def report_missing_turbulence(missing: Tuple[str, ...], transported_incomplete: bool) -> None:
    """面向用户的提示：输运场不全时湍流场保持来流初值；只缺 `nu_t`（旧版本）时涡粘保持初值。"""
    if transported_incomplete:
        print(f"   ⚠️  Checkpoint 缺少湍流场 {'/'.join(missing)}（层流、另一个湍流模型或旧版本）："
              "湍流场保持来流初值，与已恢复的平均流场不连续。")
    elif "nu_t" in missing:
        print("   ⚠️  Checkpoint 缺少 nu_t（旧版本）：涡粘保持初值，第一步粘性应力可能突变。")
