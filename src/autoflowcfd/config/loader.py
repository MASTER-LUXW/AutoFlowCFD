"""YAML 配置文件加载器。

提供从 YAML 文件加载、验证和合并求解器配置的功能。

核心组件:
    - ConfigLoader: 主配置加载器类
    - 定常和瞬态模拟的默认配置

示例:
    >>> from autoflowcfd.config import ConfigLoader
    >>> loader = ConfigLoader()
    >>> config = loader.load("simulation.yaml")
"""

import yaml
from enum import Enum
from pathlib import Path
from typing import Union, Dict, Any
from loguru import logger

from .solver_config import (
    SteadyConfig,
    TransientConfig,
    BackendType,
    TurbulenceModel,
    TimeIntegrationScheme,
)


class ConfigLoader:
    """YAML 配置文件加载器与验证器。

    支持从 YAML 文件加载定常和瞬态模拟配置，自动验证并合并默认值。

    属性:
        default_steady: 默认定常配置
        default_transient: 默认瞬态配置

    示例:
        >>> loader = ConfigLoader()
        >>> config = loader.load("config.yaml")
        >>> print(config.backend)
    """
    
    def __init__(self):
        """初始化配置加载器，设置默认配置。"""
        self.default_steady = SteadyConfig()
        self.default_transient = TransientConfig()
    
    def load(self, config_path: Union[str, Path]) -> Union[SteadyConfig, TransientConfig]:
        """从 YAML 文件加载配置。

        Args:
            config_path: YAML 配置文件路径

        Returns:
            SteadyConfig or TransientConfig: 加载的配置对象

        Raises:
            FileNotFoundError: 配置文件不存在
            ValueError: 配置格式无效
            yaml.YAMLError: YAML 解析错误

        示例:
            >>> loader = ConfigLoader()
            >>> config = loader.load("simulation.yaml")
        """
        config_path = Path(config_path)
        
        if not config_path.exists():
            raise FileNotFoundError(f"未找到配置文件: {config_path}")
        
        logger.info(f"正在从 {config_path} 加载配置")
        
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config_dict = yaml.safe_load(f)
        except yaml.YAMLError as e:
            raise ValueError(f"{config_path} 中的 YAML 格式无效: {e}")
        
        if not isinstance(config_dict, dict):
            raise ValueError(f"配置必须是 YAML 映射，得到的是 {type(config_dict)}")
        
        # 确定仿真模式
        mode = config_dict.get('mode', 'steady')
        
        if mode == 'steady':
            return self._load_steady_config(config_dict)
        elif mode == 'transient':
            return self._load_transient_config(config_dict)
        else:
            raise ValueError(f"未知的仿真模式: {mode}。必须是 'steady' 或 'transient'")
    
    def _load_steady_config(self, config_dict: Dict[str, Any]) -> SteadyConfig:
        """加载并验证定常配置。

        Args:
            config_dict: YAML 配置字典

        Returns:
            SteadyConfig: 验证后的定常配置
        """
        logger.debug("正在加载定常配置")
        
        # 与默认值合并
        merged = self._merge_defaults(config_dict, self.default_steady)
        
        # 将枚举字符串转换为枚举值
        merged = self._convert_enums(merged, SteadyConfig)
        
        # 创建配置对象
        try:
            config = SteadyConfig(**merged)
            logger.info(f"定常配置已加载: backend={config.backend}, order={config.order}")
            return config
        except Exception as e:
            raise ValueError(f"无效的定常配置: {e}")
    
    def _load_transient_config(self, config_dict: Dict[str, Any]) -> TransientConfig:
        """加载并验证瞬态配置。

        Args:
            config_dict: YAML 配置字典

        Returns:
            TransientConfig: 验证后的瞬态配置
        """
        logger.debug("正在加载瞬态配置")
        
        # 与默认值合并
        merged = self._merge_defaults(config_dict, self.default_transient)
        
        # 将枚举字符串转换为枚举值
        merged = self._convert_enums(merged, TransientConfig)
        
        # 创建配置对象
        try:
            config = TransientConfig(**merged)
            logger.info(
                f"瞬态配置已加载: backend={config.backend}, "
                f"dt={config.dt}, total_time={config.total_time}"
            )
            return config
        except Exception as e:
            raise ValueError(f"无效的瞬态配置: {e}")
    
    def _merge_defaults(self, user_config: Dict[str, Any], default_config) -> Dict[str, Any]:
        """合并用户配置与默认值。

        用户提供的值会覆盖默认值，缺失的值使用默认值。

        Args:
            user_config: 用户提供的配置字典
            default_config: 默认配置对象

        Returns:
            Dict[str, Any]: 合并后的配置字典
        """
        # 从默认值开始
        merged = {}
        for key in default_config.__dataclass_fields__.keys():
            if hasattr(default_config, key):
                merged[key] = getattr(default_config, key)
        
        # 用用户值覆盖
        for key, value in user_config.items():
            if key in merged:
                merged[key] = value
            elif key == "mode":
                # `mode` 是顶层路由键，本来就不属于 SteadyConfig/
                # TransientConfig 任何一个的字段——load() 在调用本方法
                # 之前已经用它选好了 _load_steady_config/_load_transient_
                # config 分支（见该方法文档），走到这里时它的使命已经
                # 完成。之前没有这个分支，导致 save_template() 生成的
                # 每一份配置模板自己都会触发"未知的配置键: mode"这个
                # 误报——不是真的未知，是本来就不该按 dataclass 字段处理。
                continue
            else:
                logger.warning(f"未知的配置键: {key}")
        
        return merged
    
    def _convert_enums(self, config_dict: Dict[str, Any], config_class) -> Dict[str, Any]:
        """将字符串值转换为枚举类型。

        Args:
            config_dict: 配置字典
            config_class: 目标配置类

        Returns:
            Dict[str, Any]: 枚举值已转换的字典
        """
        converted = config_dict.copy()
        
        # 获取字段注解
        annotations = getattr(config_class, '__annotations__', {})
        
        for key, value in config_dict.items():
            if key not in annotations:
                continue
            
            field_type = annotations[key]
            
            # 转换 BackendType
            if field_type == BackendType and isinstance(value, str):
                try:
                    converted[key] = BackendType(value.lower())
                except ValueError:
                    raise ValueError(
                        f"无效的后端类型: {value}。"
                        f"必须是以下之一: {[b.value for b in BackendType]}"
                    )
            
            # 转换 TurbulenceModel
            elif field_type == TurbulenceModel and isinstance(value, str):
                try:
                    converted[key] = TurbulenceModel(value.lower())
                except ValueError:
                    raise ValueError(
                        f"无效的湍流模型: {value}。"
                        f"必须是以下之一: {[t.value for t in TurbulenceModel]}"
                    )
            
            # 转换 TimeIntegrationScheme
            elif field_type == TimeIntegrationScheme and isinstance(value, str):
                try:
                    converted[key] = TimeIntegrationScheme(value.lower())
                except ValueError:
                    raise ValueError(
                        f"无效的时间积分方案: {value}。"
                        f"必须是以下之一: {[s.value for s in TimeIntegrationScheme]}"
                    )
        
        return converted
    
    def save_template(self, output_path: Union[str, Path], mode: str = 'steady'):
        """保存配置模板到 YAML 文件。

        Args:
            output_path: 输出文件路径
            mode: 仿真模式（'steady' 或 'transient'）

        示例:
            >>> loader = ConfigLoader()
            >>> loader.save_template("config_template.yaml", mode="steady")
        """
        output_path = Path(output_path)
        
        if mode == 'steady':
            config = self.default_steady
        elif mode == 'transient':
            config = self.default_transient
        else:
            raise ValueError(f"未知的模式: {mode}。必须是 'steady' 或 'transient'")
        
        # 将枚举转换为字符串以便 YAML 序列化
        config_dict = self._enum_to_string(config)
        config_dict['mode'] = mode
        
        # 添加注释作为 YAML 结构
        yaml_content = self._add_yaml_comments(config_dict, mode)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(yaml_content)
        
        logger.info(f"配置模板已保存至 {output_path}")
    
    def _enum_to_string(self, config) -> Dict[str, Any]:
        """将枚举值转换为字符串以便 YAML 序列化。

        Args:
            config: 配置对象

        Returns:
            Dict[str, Any]: 字符串值字典
        """
        result = {}
        for key, value in config.__dict__.items():
            if isinstance(value, Enum):
                result[key] = value.value
            else:
                result[key] = value
        return result
    
    def _add_yaml_comments(self, config_dict: Dict[str, Any], mode: str) -> str:
        """向 YAML 输出添加说明注释。

        Args:
            config_dict: 配置字典
            mode: 仿真模式

        Returns:
            str: 带注释的 YAML 内容
        """
        lines = [f"# AutoFlowCFD {mode.capitalize()} 仿真配置"]
        lines.append(f"# 由 ConfigLoader 生成")
        lines.append("")
        lines.append(f"mode: {mode}")
        lines.append("")
        
        for key, value in config_dict.items():
            if key == 'mode':
                continue
            
            # 为重要参数添加注释
            comment = self._get_parameter_comment(key, mode)
            if comment:
                lines.append(f"# {comment}")

            # 通过 yaml.safe_dump 本身序列化值，而不是使用 f-string，
            # 否则 None 值（例如默认的 max_cell_size）会呈现为字面文本 "None" - 
            # PyYAML 不将裸写的 "None" 识别为 null（只有 null/Null/NULL/~ /空可以），
            # 所以 safe_load 会将其读回为字符串 "None"，而不是 Python None，
            # 并且它会静默地因类型错误而验证失败，而不是加载为真正的默认值。
            # 转储为单键 {key: value} 流式映射（然后剥离大括号）- 而不是单独转储 `value` - 
            # 避免 PyYAML 在裸顶层标量后附加 "...\n" 文档结束标记，
            # 否则这会作为杂散行出现在这个手工组装的多行文件的中间，并破坏整个模板的每次解析。
            dumped = yaml.safe_dump({key: value}, default_flow_style=True).strip()
            assert dumped.startswith("{") and dumped.endswith("}")
            lines.append(dumped[1:-1])
            lines.append("")
        
        return "\n".join(lines)
    
    def _get_parameter_comment(self, param: str, mode: str) -> str:
        """获取配置参数的帮助注释。

        Args:
            param: 参数名
            mode: 仿真模式

        Returns:
            str: 帮助注释文本
        """
        comments = {
            'backend': '计算后端: cpu, gpu, 或 auto',
            'order': 'FR 离散阶数: 1, 2, 或 3',
            'turbulence': '湍流模型: sst_kw, sa, des, ddes, les',
            'max_iter': '最大迭代步数（仅定常）',
            'cfl_init': '初始 CFL 数（仅定常）',
            'cfl_max': '最大 CFL 数（仅定常）',
            'convergence_tol': '残差收敛容差（仅定常）',
            'dt': '时间步长，单位秒（仅瞬态）',
            'total_time': '总物理时间，单位秒（仅瞬态）',
            'time_scheme': '时间积分方案: backward_euler, rk2, rk3, ab3（仅瞬态）',
            'output_dir': '结果输出目录',
            'checkpoint_interval': '检查点保存间隔，单位步数',
            'growth_rate': '边界层几何增长率（仅定常）',
            'bl_layers': '在切换到（固定增长率）过渡阶段之前，计为精细 BL 阶段的层数（未设置 = 8）',
            'min_cell_size': '第一层（近壁）厚度，单位米（仅定常）',
            'target_cells': '目标总单元数（仅定常；被 tetgen 混合网格路径忽略）',
            'max_cell_size': '核心区域最大单元尺寸，单位米，从近壁尺寸向外渐变（未设置 = 无上限）',
            'rho_inf': '自由流密度，单位 kg/m^3',
            'vel_inf': '自由流速度大小，单位 m/s',
            'p_inf': '自由流静压，单位 Pa',
        }
        return comments.get(param, '')


def load_config(config_path: Union[str, Path]) -> Union[SteadyConfig, TransientConfig]:
    """从 YAML 文件加载配置的便捷函数。

    Args:
        config_path: YAML 配置文件路径

    Returns:
        SteadyConfig or TransientConfig: 加载的配置

    示例:
        >>> from autoflowcfd.config import load_config
        >>> config = load_config("simulation.yaml")
    """
    loader = ConfigLoader()
    return loader.load(config_path)


def save_config_template(output_path: Union[str, Path], mode: str = 'steady'):
    """保存配置模板的便捷函数。

    Args:
        output_path: 输出文件路径
        mode: 仿真模式（'steady' 或 'transient'）

    示例:
        >>> from autoflowcfd.config import save_config_template
        >>> save_config_template("config.yaml", mode="steady")
    """
    loader = ConfigLoader()
    loader.save_template(output_path, mode)
