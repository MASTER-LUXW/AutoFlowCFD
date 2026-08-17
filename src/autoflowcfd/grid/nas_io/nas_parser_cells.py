"""NAS 解析器：面单元（单元）提取。

流式解析 Nastran 格式文件里的面单元：原生支持 CTRIA3 三角形面单元，
CQUAD4 四边形面单元会被拆成 2 个三角形（n1,n2,n3 + n1,n3,n4）——真实的
ANSA 汽车导出文件里，双曲率车身钣金面经常混用这两种单元类型；如果解析器
不认识 CQUAD4，会在用到四边形的地方静默产生一个带洞的面网格且没有任何
警告（CQUAD4 这一行不会匹配下面任何已有的正则模式，所以连"丢弃比例"这个
兜底安全网都发现不了它，这一点和真正格式错误的 CTRIA3 行不一样）。
"""

import re
from typing import Tuple
import numpy as np
from loguru import logger

from ..structures import CellArray
from .nas_parser_exceptions import NASParseError

# 超过这个比例的元素 (CTRIA3/CQUAD4) 行被丢弃（解析错误、不可解析的行
# 或悬空节点引用），就视为系统性的格式/编码不匹配——或 GRID/元素段不同步
# ——而不是偶然噪声，大声报错而不是静默返回一个缺失大块真实几何但仍然
# 报告成功的表面网格。仅在有意义的样本量 (MIN_LINES_FOR_DROP_CHECK) 时
# 才强制执行——原因见 nas_parser_nodes.py 中的匹配常量。以行数为单位
# 而不是单元数：丢弃一行 CQUAD4 损失一张卡片，而不是它会产生的两个三角形，
# 所以基于单元数的统计会低估其权重。
MAX_DROP_FRACTION = 0.05
MIN_LINES_FOR_DROP_CHECK = 20


def parse_cells_from_nas(
    file_path: str,
    node_id_to_index: dict,
    encoding: str = 'UTF-8'
) -> Tuple[CellArray, np.ndarray]:
    """解析单元数据（流式）。

    使用流式方式从 NAS 文件解析 CTRIA3 和 CQUAD4 卡片。
    支持逗号分隔、固定格式和空白分隔的 Nastran 卡片。
    CQUAD4 四边形每张拆成 2 个三角形（对角线 n1-n3），
    因为本项目的其余表面/体网格管线纯三角形。

    Args:
        file_path: NAS 文件路径
        node_id_to_index: 从 NAS 节点 ID 到数组索引的映射
        encoding: 文件编码

    Returns:
        Tuple[CellArray, np.ndarray]: 解析的单元连接/类型，以及每个
        幸存单元的属性 ID (PID)，顺序和长度与 CellArray 一致。
        因缺失节点引用而被跳过的单元两者都排除，所以索引 i
        始终指向同一个单元。

    Raises:
        NASParseError: 单元解析失败
    """
    connectivity_list = []
    cell_types = []
    cell_pids = []
    cell_count = 0
    parse_errors = 0
    skipped_cells = 0
    skipped_lines = 0
    total_lines_seen = 0
    quad_lines_parsed = 0
    eid_to_cell_idx: dict = {}
    duplicate_eids = 0

    def _record_triangle(key, pid: int, idx1: int, idx2: int, idx3: int) -> None:
        """存储一个解析的三角形（CTRIA3 卡片，或拆分的 CQUAD4 的一半），
        以 `key` 为键，应用 Nastran 文档化的"最后一个元素 ID 获胜"约定：
        重复键时覆盖而不是追加第二个重合的三角形（之前会静默发生——
        EID 已解析但从未跟踪，所以重新导出或重复的元素卡片膨胀了
        cell_count 并且可能给体网格器留下一个非流形表面）。

        `key` 对 CTRIA3 是裸 int EID，对 CQUAD4 的两个子三角形是
        (eid, 0|1) 元组——Nastran EID 每元素唯一（不是每三角形），
        所以重新发出的同 EID CQUAD4 必须覆盖其自身的两个先前子三角形，
        不能与碰巧共享裸 int 值的无关 CTRIA3 冲突。
        """
        nonlocal cell_count, duplicate_eids
        existing_idx = eid_to_cell_idx.get(key)
        if existing_idx is not None:
            connectivity_list[existing_idx] = [idx1, idx2, idx3]
            cell_pids[existing_idx] = pid
            duplicate_eids += 1
            return
        connectivity_list.append([idx1, idx2, idx3])
        cell_types.append(0)
        cell_pids.append(pid)
        eid_to_cell_idx[key] = cell_count
        cell_count += 1
        if cell_count % 10000 == 0:
            logger.debug(f"Parsed {cell_count:,} cells...")

    def _record_quad(eid: int, pid: int, n1: int, n2: int, n3: int, n4: int) -> bool:
        """将 CQUAD4 的 4 个节点拆成 2 个三角形 (n1,n2,n3) 和
        (n1,n3,n4) 并记录两者。如果 4 个节点中任何一个缺失则返回
        False（不记录任何东西）——整个四边形作为一个单元被跳过，
        不部分记录，以避免撕裂/自重叠的表面。"""
        if not (n1 in node_id_to_index and n2 in node_id_to_index
                and n3 in node_id_to_index and n4 in node_id_to_index):
            return False
        i1 = node_id_to_index[n1]
        i2 = node_id_to_index[n2]
        i3 = node_id_to_index[n3]
        i4 = node_id_to_index[n4]
        _record_triangle((eid, 0), pid, i1, i2, i3)
        _record_triangle((eid, 1), pid, i1, i3, i4)
        return True

    ctria3_pattern_comma = re.compile(
        r'^\s*CTRIA3\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)',
        re.IGNORECASE
    )
    ctria3_pattern_fixed = re.compile(
        r'^\s*CTRIA3\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)',
        re.IGNORECASE
    )
    cquad4_pattern_comma = re.compile(
        r'^\s*CQUAD4\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)',
        re.IGNORECASE
    )
    cquad4_pattern_fixed = re.compile(
        r'^\s*CQUAD4\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)',
        re.IGNORECASE
    )

    try:
        with open(file_path, 'r', encoding=encoding, errors='ignore') as f:
            for line in f:
                line_stripped = line.strip()

                if not line_stripped or line_stripped.startswith('$') or line_stripped.startswith('#'):
                    continue

                line_upper = line_stripped.upper()
                is_quad = line_upper.startswith('CQUAD4')
                if not is_quad and not line_upper.startswith('CTRIA3'):
                    continue

                total_lines_seen += 1

                try:
                    if is_quad:
                        match = cquad4_pattern_comma.match(line_stripped) \
                            or cquad4_pattern_fixed.match(line_stripped)
                        if match:
                            eid, pid, n1, n2, n3, n4 = (int(g) for g in match.groups())
                            if _record_quad(eid, pid, n1, n2, n3, n4):
                                quad_lines_parsed += 1
                            else:
                                skipped_cells += 1
                            continue

                        # Flexible parsing (whitespace-tokenized fallback)
                        parts = line_stripped[6:].split()
                        if len(parts) >= 6:
                            try:
                                eid, pid, n1, n2, n3, n4 = (int(p) for p in parts[:6])
                                if _record_quad(eid, pid, n1, n2, n3, n4):
                                    quad_lines_parsed += 1
                                else:
                                    skipped_cells += 1
                            except (ValueError, IndexError):
                                parse_errors += 1
                        else:
                            skipped_lines += 1
                        continue

                    # 尝试 comma-separated 格式
                    match = ctria3_pattern_comma.match(line_stripped)

                    if match:
                        eid = int(match.group(1))
                        pid = int(match.group(2))
                        n1 = int(match.group(3))
                        n2 = int(match.group(4))
                        n3 = int(match.group(5))

                        if n1 in node_id_to_index and n2 in node_id_to_index and n3 in node_id_to_index:
                            idx1 = node_id_to_index[n1]
                            idx2 = node_id_to_index[n2]
                            idx3 = node_id_to_index[n3]
                            _record_triangle(eid, pid, idx1, idx2, idx3)
                            continue
                        else:
                            skipped_cells += 1
                            continue

                    # 尝试 fixed-格式
                    match = ctria3_pattern_fixed.match(line_stripped)

                    if match:
                        eid = int(match.group(1))
                        pid = int(match.group(2))
                        n1 = int(match.group(3))
                        n2 = int(match.group(4))
                        n3 = int(match.group(5))

                        if n1 in node_id_to_index and n2 in node_id_to_index and n3 in node_id_to_index:
                            idx1 = node_id_to_index[n1]
                            idx2 = node_id_to_index[n2]
                            idx3 = node_id_to_index[n3]
                            _record_triangle(eid, pid, idx1, idx2, idx3)
                            continue
                        else:
                            skipped_cells += 1
                            continue

                    # Flexible parsing
                    parts = line_stripped[6:].split()

                    if len(parts) >= 5:
                        try:
                            eid = int(parts[0])
                            pid = int(parts[1])
                            n1 = int(parts[2])
                            n2 = int(parts[3])
                            n3 = int(parts[4])

                            if n1 in node_id_to_index and n2 in node_id_to_index and n3 in node_id_to_index:
                                idx1 = node_id_to_index[n1]
                                idx2 = node_id_to_index[n2]
                                idx3 = node_id_to_index[n3]
                                _record_triangle(eid, pid, idx1, idx2, idx3)
                                continue
                            else:
                                skipped_cells += 1
                                continue
                        except (ValueError, IndexError):
                            parse_errors += 1
                            continue
                    else:
                        skipped_lines += 1
                        continue

                except ValueError as e:
                    logger.debug(f"Invalid node ID: {line_stripped} - {e}")
                    parse_errors += 1
                except Exception as e:
                    logger.debug(f"Error parsing element card: {line_stripped} - {e}")
                    parse_errors += 1

    except Exception as e:
        raise NASParseError(f"Failed to read cells: {str(e)}") from e

    if cell_count == 0:
        logger.error("No valid CTRIA3/CQUAD4 cards found")
        return CellArray(
            connectivity=np.array([], dtype=np.int32).reshape(0, 3),
            cell_type=np.array([], dtype=np.int32)
        ), np.array([], dtype=np.int32)

    dropped = parse_errors + skipped_lines + skipped_cells
    drop_fraction = dropped / total_lines_seen if total_lines_seen else 0.0
    if total_lines_seen >= MIN_LINES_FOR_DROP_CHECK and drop_fraction > MAX_DROP_FRACTION:
        raise NASParseError(
            f"{dropped}/{total_lines_seen} ({drop_fraction:.1%}) CTRIA3/CQUAD4 "
            f"lines could not be parsed or reference missing nodes - this "
            f"exceeds the {MAX_DROP_FRACTION:.0%} threshold for incidental "
            f"noise. A large dangling-node-reference count ({skipped_cells} "
            f"cells skipped) usually means the GRID and element sections are "
            f"out of sync (e.g. nodes parsed with a different ID range/format "
            f"than the elements reference). Proceeding would silently produce "
            f"a surface mesh missing a large fraction of its true geometry "
            f"while still reporting 'success' - check the file's GRID/element "
            f"card layout and encoding."
        )

    if parse_errors > 0:
        logger.warning(f"Encountered {parse_errors} parsing errors")
    if skipped_lines > 0:
        logger.info(f"Skipped {skipped_lines} element lines")
    if skipped_cells > 0:
        logger.info(f"Skipped {skipped_cells} elements due to missing nodes")
    if quad_lines_parsed > 0:
        logger.info(f"Split {quad_lines_parsed} CQUAD4 quads into {quad_lines_parsed * 2} triangles")
    if duplicate_eids > 0:
        logger.warning(
            f"{duplicate_eids} element cards reused an already-seen element ID; "
            f"kept the last card's connectivity for each (Nastran convention) "
            f"instead of creating a duplicate coincident triangle"
        )

    connectivity_array = np.array(connectivity_list, dtype=np.int32)
    cell_type_array = np.array(cell_types, dtype=np.int32)
    pid_array = np.array(cell_pids, dtype=np.int32)

    logger.info(f"Successfully parsed {cell_count:,} cells")

    return CellArray(connectivity=connectivity_array, cell_type=cell_type_array), pid_array
