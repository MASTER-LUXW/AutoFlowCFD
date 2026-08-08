"""瞬态求解器的结果容器与统计量。

提供用于存储瞬态仿真结果的数据结构，并计算时间平均统计量。
"""

import numpy as np
from typing import Dict, List, Optional
from dataclasses import dataclass, field


@dataclass
class TransientResult:
    """瞬态仿真结果容器。

    Attributes:
        solution_final: 最终解状态
        total_time: 仿真的总物理时间
        n_steps: 已完成的时间步数
        cd_history: 阻力系数历史
        cl_history: 升力系数历史
        time_stamps: 每一步对应的时间戳
        checkpoint_path: 最近一次 checkpoint 的路径
    """
    solution_final: np.ndarray
    total_time: float
    n_steps: int
    cd_history: List[float] = field(default_factory=list)
    cl_history: List[float] = field(default_factory=list)
    time_stamps: List[float] = field(default_factory=list)
    checkpoint_path: Optional[str] = None

    def get_mean_coefficients(self) -> Dict[str, float]:
        """计算时间平均的气动系数。

        Returns:
            包含平均 Cd、Cl 的字典
        """
        if len(self.cd_history) == 0:
            return {"Cd": 0.0, "Cl": 0.0}

        # 跳过初始瞬态段（前 20%）
        n_skip = int(len(self.cd_history) * 0.2)

        cd_mean = float(np.mean(self.cd_history[n_skip:]))
        cl_mean = float(np.mean(self.cl_history[n_skip:]))

        return {"Cd": cd_mean, "Cl": cl_mean}

    def get_rms_coefficients(self) -> Dict[str, float]:
        """计算系数的 RMS 脉动量。

        Returns:
            包含 RMS Cd'、Cl' 的字典
        """
        if len(self.cd_history) < 10:
            return {"Cd_rms": 0.0, "Cl_rms": 0.0}

        n_skip = int(len(self.cd_history) * 0.2)

        cd_rms = float(np.std(self.cd_history[n_skip:]))
        cl_rms = float(np.std(self.cl_history[n_skip:]))

        return {"Cd_rms": cd_rms, "Cl_rms": cl_rms}
