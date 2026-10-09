"""NAS 导出：边界组写入模块。

从 nas_export.py 拆分出来，专门负责把体网格的边界分组写成 ANSA 风格的
PSHELL 属性 + 真实 CTRIA3 面单元（外加 PSOLID/ANSA_PART 元数据），与
nas_export.py 里节点/单元几何写入的部分区分开。

主要组件：
    - write_boundaries：写出 PSHELL/CTRIA3/PSOLID/ANSA_PART 等边界元数据卡片（每组边界面逐面判定，
      见 mesh_boundary.exterior_faces_by_group）
"""

from loguru import logger


def write_boundaries(
    f, volume_mesh, solid_pid: int, start_eid: int
) -> None:
    """将边界组写为 PSHELL 属性和真实 CTRIA3 面单元，
    加上体网格的 PSOLID 卡片。

    Args:
        f: 文件句柄
        volume_mesh: VolumeMeshData——提供 `boundaries`（含 groups 和
            bc_types 的 BoundaryMap）；每组的边界面逐面判定（见
            mesh_boundary.exterior_faces_by_group）。
        solid_pid: 已用于 CTETRA/CPENTA 元素的 PSOLID 属性 ID
            （由调用方预留，不会与 PSHELL PID 冲突）。
        start_eid: 第一个可用的 Nastran 元素 ID (n_prism + n_tets + 1)，
            边界 CTRIA3 元素不会与 CPENTA/CTETRA 元素 ID 冲突。
    """
    boundaries = volume_mesh.boundaries
    if not boundaries.groups:
        logger.warning("No boundary groups found, skipping boundary export")
        return

    from ..mesh_gen.utils.mesh_boundary import exterior_faces_by_group

    faces_by_group = exterior_faces_by_group(volume_mesh)

    pid_counter = 1
    mid_counter = 1
    eid_counter = start_eid

    for group_name in faces_by_group:
        # PSHELL 名称写组名（2026-10-09 以前写的是边界条件类型的小写：两个 WALL 组同名，原始组名丢失；
        # 面网格读入时正是按组名识别边界条件类型的，组名保留了全部信息）
        ansa_name = group_name

        # PSHELL 小字段格式（每字段 8 字符）：字段 1 "PSHELL  "、2 PID、3 MID1、4 T、5 MID2、
        # 6 12I/T^3、7 MID3、8 TS/T
        f.write(
            f"PSHELL  {pid_counter:>8}{mid_counter:>8}{1.0:>8.1f}"
            f"{mid_counter:>8}{1.0:>8.1f}{mid_counter:>8}{0.8333:>8.4f}\n"
        )

        # 写入 ANSA 名称 comment
        f.write(f"$ANSA_NAME_COMMENT;{pid_counter};PSHELL;{ansa_name};;NO;NO;NO;NO;\n")

        # 写入分组实际边界面的 CTRIA3 卡片，使上面的属性
        # 引用真实几何而不是空定义。
        for face in faces_by_group.get(group_name, ()):
            n1, n2, n3 = int(face[0]) + 1, int(face[1]) + 1, int(face[2]) + 1
            f.write(f"CTRIA3{eid_counter:>10}{pid_counter:>8}{n1:>8}{n2:>8}{n3:>8}\n")
            eid_counter += 1

        pid_counter += 1
        mid_counter += 1

    logger.info(f"  Boundary face elements written: {eid_counter - start_eid:,}")

    # 写入体网格的 PSOLID 卡片（PID 由调用方预留，
    # 不会与上面写入的 PSHELL PID 冲突）
    solid_mid = mid_counter
    f.write(f"PSOLID{solid_pid:>8}{solid_mid:>8}\n")
    f.write(f"$ANSA_NAME_COMMENT;{solid_pid};PSOLID;Auto Detected Volume;;NO;NO;NO;NO;\n")

    # 写入 $ANSA_COLOR 显示颜色注释，每个 shell MID [1,
    # solid_mid) 一个——不包括 solid_mid 本身，它有自己的
    # （不同的）体颜色，就在下面；用 range(1, mid_counter + 1) 会
    # 重复计算它（solid_mid == mid_counter），为同一个 MID 发出
    # 两条冲突的颜色条目。
    for i in range(1, mid_counter):
        f.write(f"$ANSA_COLOR;{i};MAT1;.725490212440491;.035294119268656;0.20392157137394;1.;\n")

    f.write(f"$ANSA_COLOR;{solid_mid};MAT1;.635294139385223;0.34901961684227;.341176480054855;1.;\n")

    # 写入部件定义
    f.write("$ANSA_PART;GROUP;ID;2;NAME;Auto Detected Volumes Group;BELONGS_HERE;YES;PID_OFFS\n")
    f.write("$ET;0;COLOR;137;211;69;0;IS_COLOR_ACTIVE;0;PART_TYPE;Undefined;ATTRIBUTES;2;DM/F\n")
    f.write("$ile Type;ANSA;DM/Status;WIP;CONTAINS;ANSAPART;3;\n")

    # 表面部件
    if pid_counter > 1:
        shell_range = f"1-{pid_counter-1}" if pid_counter > 2 else "1"
        f.write("$ANSA_PART;PART;ID;1;NAME;Untitled;BELONGS_HERE;YES;STUDY_VERSION;0;PID_OFFSET;0\n")
        f.write("$;COLOR;185;9;52;0;IS_COLOR_ACTIVE;1;PART_TYPE;Undefined;ATTRIBUTES;2;DM/File Ty\n")
        f.write(f"$pe;ANSA;DM/Status;WIP;CONTAINS;PSHELL;{shell_range};\n")

    # 体积部件
    f.write("$ANSA_PART;PART;ID;3;NAME;Untitled_Volume_1;BELONGS_HERE;YES;STUDY_VERSION;0;PID\n")
    f.write("$_OFFSET;0;COLOR;215;68;166;0;IS_COLOR_ACTIVE;1;PART_TYPE;Undefined;ATTRIBUTES;2\n")
    f.write(f"$;DM/File Type;ANSA;DM/Status;WIP;CONTAINS;PSOLID;{solid_pid};\n")
    f.write("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$\n")
    f.write("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$\n")

    logger.info(f"  Boundary groups written: {len(faces_by_group)}")
