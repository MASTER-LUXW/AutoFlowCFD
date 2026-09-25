"""求解命令的壁面距离场计算辅助函数 —— 从 solve_helpers.py 拆出，控制单文件行数。

见 solve_helpers.py 文档说明整体拆分结构。
"""

import click


def build_wall_distance_source(volume_data, use_eikonal=False):
    """CLI 各后端分支共用：由体网格构造壁面距离来源（单机、分布式、GPU 同一份）。

    WALL 组边界面取壁面节点的判据与为什么这么取，见
    `core/utils/wall_distance_source.py`。没有壁面时的 `ValueError` 转成
    可读的命令行错误，不退化成估计值继续求解。
    """
    from autoflowcfd.core.utils.wall_distance_source import WallDistanceSource

    try:
        return WallDistanceSource.from_volume_data(volume_data, use_eikonal=use_eikonal)
    except ValueError as e:
        raise click.ClickException(str(e)) from e


def wall_distance_source_if_needed(turbulence_model, volume_data, use_eikonal=False):
    """湍流模型需要壁距时构造来源，否则 None。CLI 的分布式 / GPU 分支用它把
    同一个来源交给求解器（单机分支经 `compute_wall_distance_for_solver`）。"""
    from autoflowcfd.core.fr_solver.turbulence.wall_distance import WALL_DISTANCE_MODELS

    if str(turbulence_model).upper() not in WALL_DISTANCE_MODELS:
        return None
    return build_wall_distance_source(volume_data, use_eikonal=use_eikonal)


def compute_wall_distance_for_solver(solver, volume_data, use_eikonal=False):
    """
    为求解器计算壁面距离场。

    Args:
        solver: FRSolver实例
        volume_data: load_mesh_for_solver 已经加载好的 VolumeMeshData - 直接
            复用，不重新解析一遍输入文件（这里以前是重新按 input_file 路径
            读一遍 .pkl，且只认 .pkl，.nas 体网格输入会直接跳过整个壁面距离
            计算、静默退化成"简化估计" - 现在 load_mesh_for_solver 两条路径
            都已经把 volume_data 解析好，直接传进来即可，同时对 .pkl/.nas
            两种输入路径都正确）
        use_eikonal: 是否使用 Eikonal 方程求解
    """
    turb_model = getattr(solver, 'turb_model_name', '').lower()
    if turb_model not in ['sst', 'ddes', 'iddes', 'wmles', 'les']:
        print(f"   ℹ️  Turbulence model '{turb_model}' does not require wall distance")
        return

    if volume_data is None or not hasattr(volume_data, 'boundaries'):
        raise click.ClickException(
            "无法访问体网格边界数据（volume_data 缺少 'boundaries' 属性），"
            f"无法为湍流模型 '{turb_model}' 计算壁面距离场。")
    try:
        print("\n🔍 Computing wall distance field...")
        source = build_wall_distance_source(volume_data, use_eikonal=use_eikonal)
        from autoflowcfd.core.fr_solver.turbulence import apply_wall_distance_source

        apply_wall_distance_source(solver, source)
        print(f"   ✅ Wall distance field computed ({source.kind}, "
              f"{source.n_wall_nodes} wall nodes)\n")
    except click.ClickException:
        raise
    except Exception as e:
        # 此前这里是裸 except Exception：任何失败（含 Eikonal 求解器内部
        # bug）都打印一行 warning 后静默降级为"simplified estimate"继续
        # 求解——但 solver.wall_distance 实际仍是 None，SST/DDES 下游会在
        # fr_solver_turbulence.py 里因 wall_distance is None 抛
        # RuntimeError（等于这里的"降级"从未真正发生），LES/WMLES 下游则
        # 没有这道保护、会真的带着错误的湍流模型悄悄跑完。与
        # load_mesh_for_solver 的质量门"宁可报错也不静默放行"原则矛盾，
        # 统一改为向上抛出可读错误。
        raise click.ClickException(
            f"壁面距离场计算失败，无法为湍流模型 '{turb_model}' 提供有效的 "
            f"d_w：{e}\n如需临时绕过做诊断，请改用 --turbulence-model none。"
        ) from e
