"""
AutoFlowCFD - 求解器配置管理 (V2.0 Adapted)

本模块定义 SteadyConfig 类，支持 V2.0 FR 方法特有的配置项。
"""

import json


class SteadyConfig:
    """
    求解器配置容器。
    
    Attributes:
        polynomial_order: FR 多项式阶数 P (V2.0)
        turbulence_model: 湍流模型选择 ('SST', 'DDES', 'WMLES')
        time_method: 时间推进方法 ('RK4', 'IMEX', 'Dual-Time')
    """

    def __init__(self):
        # V1.0 基础配置
        self.max_iterations = 1000
        self.cfl = 1.0
        
        # V2.0 新增配置 (C-01)
        self.polynomial_order = 2
        self.turbulence_model = "SST"
        self.time_method = "RK4"
        self.flux_type = "AUSM+up"

    def load_from_json(self, file_path: str):
        """从 JSON 文件加载配置。"""
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
                for key, value in data.items():
                    if hasattr(self, key):
                        setattr(self, key, value)
            print(f"Configuration loaded from {file_path}")
        except Exception as e:
            print(f"Error loading config: {e}")

    def save_to_json(self, file_path: str):
        """保存配置到 JSON 文件。"""
        with open(file_path, 'w') as f:
            json.dump(self.__dict__, f, indent=4)