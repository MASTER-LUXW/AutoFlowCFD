"""AutoFlowCFD V2.0 - 把紧凑索引空间的壁面几何上传到设备

从 `src/autoflowcfd/core/gpu/distributed/gpu_distributed_fully_distributed.py`(原 557 行)拆出(2026-09-24, 项目"单文件不超 500 行"规范)。**纯搬家, 逻辑未改**。
"""











def _upload_wall_geometry_compact(solver, package, cp, device_id):
    """从 package 里取出 root 已经按本 rank 的 compact 索引空间算好、
    切好的 wall_distance_compact/iddes_h_max_compact/iddes_h_wn_compact
    并上传 GPU——SST/DDES/IDDES/WMLES 共用同一套逻辑，两处调用点
    （构造 + Order Continuation 重建）复用，避免写两遍。
    """
    wall_distance_compact = package.get('wall_distance_compact')
    if wall_distance_compact is None:
        raise RuntimeError(
            f"Rank {solver.rank}: 完全分布式加载模式下 turbulence_model="
            f"'{solver.turb_model_name}' 需要 wall_distance_compact，但 "
            f"package 里没有——build_fully_distributed_rank_package 应该"
            f"已经算好并放进 package，说明 root 侧构造有缺陷。"
        )
    with cp.cuda.Device(device_id):
        solver.wall_distance_gpu = cp.asarray(wall_distance_compact)

    h_max_compact = package.get('iddes_h_max_compact')
    h_wn_compact = package.get('iddes_h_wn_compact')
    with cp.cuda.Device(device_id):
        solver.iddes_h_max_compact = (
            cp.asarray(h_max_compact) if h_max_compact is not None else None
        )
        solver.iddes_h_wn_compact = (
            cp.asarray(h_wn_compact) if h_wn_compact is not None else None
        )
