"""AutoFlowCFD V2.0 - 续算的湍流场被上界大面积截断时的处理（全部后端共用的唯一实现）。

续算的 checkpoint 若有超过 10% 的真实自由度贴近湍流量上界（>= 0.9 倍），说明湍流场来自此前的失控、从未恢复。
**默认拒绝续算并报告**；用户确认要从来流值重新建立湍流场（平均流保留）时用 `solve resume --reset-turbulence`。

2026-10-19（I-91）以前满足判据时自动把整个湍流场重置为来流值并继续算，只打一行 WARN：被重置后的续算已不是原
算例，而上界本身（`turbulence/sst/bounds.py` 的 k_max 等）只是数值安全界，合法的强湍流区也可能触发。比例的分母
此前含四面体的零填充槽位（P2 占 63%），现在只计真实自由度。

## 判据的来历

2026-09-05 cube_demo 真实网格续算发现：此前按 k 的均值与 max(1% k_max, 10 k_inf) 比较，健康、充分发展的钝体尾流
（k_mean 为来流的 228 倍）被误判为失控；改为按贴近上界的自由度比例判断（那份健康 checkpoint 上 k / omega 分别只有
4.36% / 0.07% 贴近上界）。

## 分布式

湍流场按 rank / 设备本地分片存储。各 rank 必须用同一个全局比例做同一个决定（否则部分 rank 重置、部分不重置，halo
交换会把不一致扩散到全场），所以贴近上界的自由度数与真实自由度总数都经 `allreduce_sum` 求全局和（单进程原样返回）。
"""

#: 贴近上界的判据（相对上界）与触发比例（相对真实自由度总数）
CEILING_FRACTION = 0.9
CLAMPED_SHARE_THRESHOLD = 0.10


def check_resumed_turbulence(solver) -> None:
    """续算前检查湍流场（见模块文档）；满足判据时报错，`solver._reset_turbulence_on_resume` 为真时改为重置到来流值。

    `solver._reset_turbulence_on_resume` 由恢复路径设置（缺省 False，`solve resume --reset-turbulence` 置真），只在
    `_resumed_from_checkpoint` 时调用本函数。

    Raises:
        RuntimeError: 贴近上界的真实自由度超过 10% 且未要求重置。
    """
    from autoflowcfd.core.mpi import is_root
    from autoflowcfd.core.mpi.comm import allreduce_sum
    from autoflowcfd.core.utils.array_module import array_module
    from autoflowcfd.fr.native_padding import order_from_n_sps, real_row_mask

    turb = getattr(solver, 'turb_model', None) or getattr(solver, 'turb_model_gpu', None)
    if turb is None or not hasattr(turb, 'transported_fields'):
        return
    fields = turb.transported_fields()
    bounds = turb.upper_bounds()
    n_cells, n_sps = fields[0].shape
    real = real_row_mask(solver.owned_cell_is_prism(), n_sps, order_from_n_sps(n_sps)).reshape(n_cells, n_sps)
    xp_real = array_module(fields[0]).asarray(real)       # 湍流场所在的数组模块（CPU / GPU）
    n_real_global = allreduce_sum(float(real.sum()))
    shares = [allreduce_sum(float(((f >= CEILING_FRACTION * b) & xp_real).sum())) / max(n_real_global, 1.0)
              for f, b in zip(fields, bounds)]
    if max(shares) <= CLAMPED_SHARE_THRESHOLD:
        return
    detail = ", ".join(f"{name} {100 * sh:.2f}%（上界 {b:.4g}）"
                       for name, sh, b in zip(turb.TRANSPORTED_FIELDS, shares, bounds))
    if not solver._reset_turbulence_on_resume:
        raise RuntimeError(
            f"续算的湍流场有超过 {100 * CLAMPED_SHARE_THRESHOLD:.0f}% 的真实自由度贴近上界（>= {CEILING_FRACTION} 倍，"
            f"全域统计）：{detail}。湍流场来自此前的失控、未恢复，在它上面续算即是在被截断的场上继续。"
            f"确认要从来流值重新建立湍流场（平均流保留）时用 `solve resume --reset-turbulence`；否则从更早的 checkpoint "
            f"续算。")
    if is_root():
        print(f"[WARN] 续算：湍流场 {detail} 贴近上界，按 --reset-turbulence 重置为来流值（平均流保留）")
    for f, v in zip(fields, turb.freestream_values()):
        f[:] = v
    if hasattr(turb, 'nu_t'):
        turb.nu_t[:] = 0.0

