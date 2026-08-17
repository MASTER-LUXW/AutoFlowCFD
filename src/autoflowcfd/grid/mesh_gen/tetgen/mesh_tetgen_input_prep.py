"""tetgen 核心填充调用前的 PLC 输入整理。

从 mesh_tetgen_core.py 的 fill_core_volume 里拆出来的纯数据预处理步骤：
背景点拼接、面索引越界校验、退化面（三个顶点里有重复）剔除。三者共同点是
只读/只变换 点/面/face_markers 这三个数组本身，不涉及 tetgen 对象
的创建或调用，拆成独立、可单独单元测试的纯函数。
"""

from typing import Optional, Tuple

import numpy as np
from loguru import logger


def prepare_plc_input(
    points: np.ndarray,
    faces: np.ndarray,
    background_points: Optional[np.ndarray] = None,
    face_markers: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
    """把 fill_core_volume 收到的原始输入整理成可以直接喂给 tetgen 的形式。

    Args:
        points: (n_points, 3) float64 PLC 顶点
        faces: (n_faces, 3) int32 PLC 三角面
        background_points: 可选，(q, 3) 额外自由点（不被 faces 任何一行引
            用），拼接在 点 末尾 - 见 fill_core_volume 自己的同名参数文档
        face_markers: 可选，(n_faces,) int32 每个面的标记，随 faces 一起被
            退化面剔除同步过滤

    Returns:
        (points, faces, face_markers) - points 可能因为拼接背景点而变长；
        面/face_markers 可能因为剔除退化面而变短。

    Raises:
        RuntimeError: faces 引用了越界的顶点索引 - 上游 bug 的早期信号，
            比让 tetgen 自己在 C 层面因越界索引崩溃更容易定位。
    """
    points = np.ascontiguousarray(points, dtype=np.float64)
    faces = np.ascontiguousarray(faces, dtype=np.int32)

    # 拼接在 faces 可能引用到的所有点之后,所以下面的越界/退化面检查（只看
    # faces 和原始 points）不需要跟着调整。见 fill_core_volume 自己的
    # `background_points` 参数文档，这些点为什么可以作为自由（非 facet）
    # 输入点安全加入。
    if background_points is not None and len(background_points) > 0:
        points = np.vstack([points, np.ascontiguousarray(background_points, dtype=np.float64)])
        logger.info(f"Adding {len(background_points)} background points to seed the initial tetrahedralization")

    if np.any(faces < 0) or np.any(faces >= len(points)):
        raise RuntimeError(
            f"Invalid face indices detected in PLC boundary. "
            f"Faces range [{faces.min()}, {faces.max()}], but points count is {len(points)}."
        )

    # 剔除退化面（三个顶点里有重复，面积恒为零）- 喂给 tetgen 会导致其在
    # 内部挂起而不是给出明确报错。
    sorted_faces = np.sort(faces, axis=1)
    degenerate_mask = (
        (sorted_faces[:, 0] == sorted_faces[:, 1]) |
        (sorted_faces[:, 1] == sorted_faces[:, 2]) |
        (sorted_faces[:, 0] == sorted_faces[:, 2])
    )
    n_degenerate = int(np.sum(degenerate_mask))
    if n_degenerate > 0:
        logger.warning(
            f"Found {n_degenerate} degenerate faces in PLC boundary. "
            f"Removing them before TetGen call to prevent hangs."
        )
        faces = faces[~degenerate_mask]
        if face_markers is not None:
            face_markers = face_markers[~degenerate_mask]

    return points, faces, face_markers
