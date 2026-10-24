"""AutoFlowCFD V2.0 - 求解器逐物理步时间序列 CSV（探针 `probes.csv`、气动力历史 `forces.csv` 共用）。

* 新建：写表头；
* 续算：保留不晚于 checkpoint 物理时间的行（中断前已写、续算会重算的那一段截掉），表头必须与本次一致；
* 追加：每行写完即关闭文件，中断时已写的样本完整。

只在 root rank 调用。
"""

import csv
from pathlib import Path
from typing import Optional, Sequence

#: 续算截断时判定"晚于 checkpoint 时刻"的相对容差（物理时间是逐步累加的浮点数）
RESUME_TIME_RTOL = 1e-9


def prepare_series_file(path: Path, header: Sequence[str], resume_time: Optional[float], what: str) -> None:
    """新建或续算截断（见模块文档）。`what`：表头不一致时报错信息里的说明（哪些列变了）。

    Raises:
        ValueError: 续算时已有文件的表头与本次不一致
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    kept = []
    if resume_time is not None and path.exists():
        with open(path, newline="") as f:
            rows = list(csv.reader(f))
        if rows and rows[0] != list(header):
            raise ValueError(f"{path} 的列与本次不一致（{what}）：换一个输出目录或删掉旧文件")
        limit = resume_time * (1.0 + RESUME_TIME_RTOL) + 1e-300
        kept = [r for r in rows[1:] if float(r[0]) <= limit]
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(list(header))
        writer.writerows(kept)


def append_series_row(path: Path, row: Sequence) -> None:
    with open(path, "a", newline="") as f:
        csv.writer(f).writerow(list(row))
