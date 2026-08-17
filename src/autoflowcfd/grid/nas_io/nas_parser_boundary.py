"""NAS 解析器：边界条件提取。

提供从 NAS 文件解析边界条件的专用函数，包括 属性 名称 识别与边界分组。
"""

import re
from typing import Dict, List, Tuple
import numpy as np
from loguru import logger

from ..structures import BoundaryMap
from .nas_parser_exceptions import NASParseError

# 用于捕获所有属性 ID 无法解析为名称的单元的
# "catch-all"分组名称（没有 $ANSA_NAME_COMMENT 也没有可解析的 PSHELL
# 注释）。没有这个分组，这些单元会静默从每个边界组消失——它们仍然
# 存在于网格中但完全没有分配边界条件。
_UNCLASSIFIED_GROUP = "UNCLASSIFIED"

# Shell 类型的属性卡片类型，用于边界命名识别。PCOMP/
# PCOMPG（复合/分层壳——常见于涂漆或复合材料车身板）之前直接落入
# _UNCLASSIFIED_GROUP/WALL，没有任何指示表明这是系统性的属性类型缺口
# 而不是真正的未命名属性，因为只检查了 PSHELL。
_SHELL_PROPERTY_TYPES = {'PSHELL', 'PCOMP', 'PCOMPG'}


def _leading_card_name(line_stripped: str) -> str:
    """提取 Nastran 卡片的首关键字（例如从
    "PSHELL,1,1001,0.001" 和 "PSHELL       1      1      1." 中提取
    "PSHELL"）——卡片名称是第一个逗号或空白之前的内容，取先出现的那个。"""
    return line_stripped.split(',', 1)[0].split()[0].upper() if line_stripped else ''


def parse_boundary_properties(
    file_path: str,
    encoding: str,
    cell_count: int,
    cells_data: list
) -> BoundaryMap:
    """解析边界条件（v2.0：支持属性名称识别）。

    使用属性 (PSHELL) 名称从 NAS 文件解析边界条件信息。

    Args:
        file_path: NAS 文件路径
        encoding: 文件编码
        cell_count: 从文件解析的单元总数，用于警告某些单元可能无法
            放入任何边界组
        cells_data: [cell_index, pid] 对列表，用于映射单元到属性

    Returns:
        BoundaryMap: 解析的边界组和 BC 类型，包含属性信息

    Raises:
        NASParseError: 边界解析失败
    """
    groups: Dict[str, np.ndarray] = {}
    bc_types: Dict[str, str] = {}
    property_ids: Dict[str, int] = {}
    property_names: Dict[int, str] = {}

    logger.info("Parsing boundary conditions from Properties...")

    try:
        # Step 1: Parse $ANSA_NAME_COMMENT cards to extract PID to Name mapping
        pid_to_name = _parse_property_names(file_path, encoding)

        # If no ANSA_NAME_COMMENT found, try parsing PSHELL cards directly
        if not pid_to_name:
            logger.warning("No $ANSA_NAME_COMMENT cards found. Trying alternative parsing...")
            pid_to_name = _parse_pshell_names(file_path, encoding)

        logger.info(f"Found {len(pid_to_name)} Properties with names")

        # Step 2: Build cell_index to PID mapping from the pre-parsed data
        cell_to_pid = _parse_cell_to_pid_mapping(cells_data)

        # Step 3: Group cells by Property ID
        pid_to_cells = _group_cells_by_pid(cell_to_pid)

        # Step 4: Map Property Names to boundary groups
        groups, bc_types, property_ids = _map_properties_to_boundaries(
            pid_to_name, pid_to_cells
        )

        # PID 从未解析到名称（或 PID 根本不在 pid_to_name 中）
        # 的单元会从每个边界组消失。收集到 UNCLASSIFIED 分组
        # 而不是静默丢弃它们，并大声警告——后续求解器找不到某些
        # 单元的 BC 时需要知道原因。
        classified_cells = set()
        for indices in groups.values():
            classified_cells.update(indices)
        unclassified = sorted(
            idx for idx in cell_to_pid if idx not in classified_cells
        )
        if unclassified:
            groups[_UNCLASSIFIED_GROUP] = unclassified
            bc_types[_UNCLASSIFIED_GROUP] = 'WALL'
            logger.warning(
                f"{len(unclassified)} cells could not be mapped to a named "
                f"Property (missing $ANSA_NAME_COMMENT/PSHELL name) and were "
                f"placed in the '{_UNCLASSIFIED_GROUP}' group as WALL"
            )
        if cell_count and len(cell_to_pid) < cell_count:
            logger.warning(
                f"Only {len(cell_to_pid)}/{cell_count} cells resolved a "
                f"Property ID during boundary parsing; the remainder have "
                f"no boundary group at all"
            )

        # 转换为 numpy 数组
        for name in groups:
            groups[name] = np.array(groups[name], dtype=np.int32)

        logger.info(f"Parsed {len(groups)} boundary groups")

        return BoundaryMap(
            groups=groups,
            bc_types=bc_types,
            property_ids=property_ids,
            property_names=property_names,
            detection_mode="auto",
            parameters={}
        )
        
    except Exception as e:
        raise NASParseError(f"Failed to parse boundaries: {str(e)}") from e


def _parse_property_names(file_path: str, encoding: str) -> Dict[int, str]:
    """解析 $ANSA_NAME_COMMENT 卡片以提取 PID 到名称的映射。"""
    pid_to_name = {}
    
    with open(file_path, 'r', encoding=encoding, errors='ignore') as f:
        for line in f:
            line_stripped = line.strip()
            
            if not line_stripped or not line_stripped.startswith('$'):
                continue
            
            # 检查 for ANSA_NAME_COMMENT card
            # Format: $ANSA_NAME_COMMENT;PID;PSHELL;name;;NO;NO;NO;NO;
            if line_stripped.upper().startswith('$ANSA_NAME_COMMENT'):
                parts = line_stripped.split(';')
                if len(parts) >= 5:
                    try:
                        pid = int(parts[1])
                        prop_type = parts[2].strip().upper()
                        prop_name = parts[3].strip()

                        # PSHELL, and composite/layered shells (PCOMP/
                        # PCOMPG) - common for painted or composite body
                        # panels, otherwise silently degraded to WALL via
                        # _UNCLASSIFIED_GROUP with no indication it was a
                        # property-TYPE gap rather than a genuinely unnamed
                        # property.
                        if prop_type in _SHELL_PROPERTY_TYPES and prop_name:
                            pid_to_name[pid] = prop_name
                            logger.debug(f"Found Property: PID={pid}, Name='{prop_name}'")
                    except (ValueError, IndexError):
                        pass
    
    return pid_to_name


def _parse_pshell_names(file_path: str, encoding: str) -> Dict[int, str]:
    """解析带基于注释命名的 PSHELL/PCOMP/PCOMPG 卡片。

    处理逗号分隔的自由字段卡片（``PSHELL,1,1001,0.001``）
    和固定宽度小字段卡片（``PSHELL       1      1      1.``，
    无逗号——本项目自己的 nas_export.py 在没有配对的
    $ANSA_NAME_COMMENT 时写入的格式）。仅逗号分割以前既匹配不了
    固定格式的情况，也匹配不了真正裸的 PID 列，对这样的文件静默
    返回无名称。"""
    pid_to_name = {}

    with open(file_path, 'r', encoding=encoding, errors='ignore') as f:
        prev_line = ""
        for line in f:
            line_stripped = line.strip()

            if not line_stripped or line_stripped.startswith('#'):
                continue

            # Check for a shell-like property card (PSHELL, or composite/
            # layered PCOMP/PCOMPG - same fallback naming convention).
            if _leading_card_name(line_stripped) in _SHELL_PROPERTY_TYPES:
                # 尝试从上一行注释获取名称
                # Format: $ PROPERTY NAME: XXXX
                if prev_line.startswith('$'):
                    match = re.search(r'PROPERTY\s+NAME:\s*(\S+)', prev_line, re.IGNORECASE)
                    if match:
                        # 逗号分隔自由字段格式优先...
                        parts = [p.strip() for p in line_stripped.split(',') if p.strip()]
                        if len(parts) < 2:
                            # ...fall back to whitespace-split fixed-width form.
                            parts = line_stripped.split()
                        if len(parts) >= 2:
                            try:
                                pid = int(parts[1])
                                pid_to_name[pid] = match.group(1)
                            except ValueError:
                                pass

            prev_line = line_stripped

    return pid_to_name


def _parse_cell_to_pid_mapping(cells_data: list) -> Dict[int, int]:
    """把预先解析好的 [cell_index, pid] 列表转换成 cell_index -> pid 字典。

    唯一调用方 parser_core.py 总是传入已经和 parse_cells_from_nas 结果对齐的
    cells_data（见调用处注释：独立重新扫描 CTRIA3 卡片无法知道哪些 单元 被
    跳过，索引会错位），因此这里不再保留"文件里独立重新扫描 CTRIA3"的分支。
    """
    return {cell_idx: pid for cell_idx, pid in cells_data}


def _group_cells_by_pid(cell_to_pid: Dict[int, int]) -> Dict[int, List[int]]:
    """按属性 ID 分组单元。"""
    pid_to_cells = {}
    
    for cell_idx, pid in cell_to_pid.items():
        if pid not in pid_to_cells:
            pid_to_cells[pid] = []
        pid_to_cells[pid].append(cell_idx)
    
    return pid_to_cells


def _map_properties_to_boundaries(
    pid_to_name: Dict[int, str],
    pid_to_cells: Dict[int, List[int]]
) -> Tuple[Dict[str, List[int]], Dict[str, str], Dict[str, int]]:
    """将属性名称映射到边界组并检测 BC 类型。"""
    groups = {}
    bc_types = {}
    property_ids = {}
    
    # Boundary keyword mapping for automatic detection. Order matters:
    # _detect_boundary_type returns the FIRST matching bc_type, so more
    # specific keyword sets are listed before the generic 'WALL' bucket -
    # otherwise a compound name like "TUNNEL_WALL" would match the plain
    # 'wall' substring before ever reaching the 'tunnel' keyword below.
    boundary_keywords = {
        'VELOCITY_INLET': ['inlet', 'inflow', 'entrance', '入口'],
        'PRESSURE_OUTLET': ['outlet', 'outflow', 'exit', '出口'],
        'SYMMETRY': ['symmetry', 'sym', '对称'],
        # 周期边界（见 grid/face_connectivity.py::pair_periodic_boundary_faces）
        # 只能靠属性名关键字识别出"这是一对周期面"，无法从几何/NAS 文件本身
        # 反推出配对的另一侧组名与平移向量——这两项必须通过 YAML 手动/混合
        # 配置补齐（写入 BoundaryMap.parameters[name]['paired_with'/'translation']），
        # 纯 NAS 自动模式无法单独完成周期边界的完整配置。
        'PERIODIC': ['periodic', '周期'],
        # A "tunnel" boundary is a frictionless duct wall (see
        # bc_handler.py's _classify: TUNNEL -> SYMMETRY/free-slip), not a
        # viscous no-slip wall - it must never get BL extrusion (there is
        # no velocity gradient at a slip wall to resolve). Previously
        # "tunnel" matched none of these keywords and silently fell
        # through to the 'WALL' default below, making it BL-extrude-
        # eligible - extruding a boundary layer on a domain-spanning
        # tunnel wall collapses almost immediately (hits the opposite
        # wall/body within 1-2 layers), producing hundreds of degenerate
        # tetrahedra and a non-manifold surface that crashes tetgen.
        'SLIP_WALL': ['slip', 'farfield', 'freestream', 'tunnel', '风洞', '洞壁'],
        'WALL': ['wall', 'body', 'surface', '车体', '车身', '壁面'],
    }
    
    for pid, name in pid_to_name.items():
        if pid not in pid_to_cells:
            continue

        # 确定 boundary 类型 基于 名称
        bc_type = _detect_boundary_type(name, boundary_keywords)

        # ANSA 导出经常将一个逻辑边界（例如 "WALL"）拆分到
        # 多个共享相同属性名称的 PID。这里用扩展而不是替换，
        # 否则静默丢弃每个 PID 的单元，只保留给定名称最后处理的那个。
        if name in groups:
            groups[name].extend(pid_to_cells[pid])
            if property_ids.get(name) != pid:
                logger.debug(
                    f"Property '{name}' spans multiple PIDs "
                    f"({property_ids.get(name)}, {pid}); merging cells "
                    f"into one boundary group"
                )
        else:
            groups[name] = list(pid_to_cells[pid])
            property_ids[name] = pid
        bc_types[name] = bc_type

        logger.debug(f"Mapped Property '{name}' (PID={pid}) to {bc_type}")
    
    return groups, bc_types, property_ids


def _detect_boundary_type(name: str, keywords: Dict[str, List[str]]) -> str:
    """使用关键字匹配从属性名称检测边界类型。"""
    name_lower = name.lower()
    
    for bc_type, keyword_list in keywords.items():
        for keyword in keyword_list:
            if keyword.lower() in name_lower:
                return bc_type
    
    # 默认到 WALL，如果没有匹配
    return 'WALL'
