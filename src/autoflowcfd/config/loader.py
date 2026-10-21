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

import dataclasses
from enum import Enum
from pathlib import Path
from typing import Union, Dict, Any

import yaml
from loguru import logger

from autoflowcfd.core.fr_solver.solver.threads import DEFAULT_N_THREADS
from autoflowcfd.core.utils.order_continuation.policy import MAX_ORDER, MIN_STEADY_ORDER, MIN_TRANSIENT_ORDER

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
        """从 YAML 文件加载配置（`load_with_keys` 的配置对象部分）。"""
        return self.load_with_keys(config_path)[0]

    def load_with_keys(self, config_path: Union[str, Path]):
        """从 YAML 文件加载配置，同时返回文件里显式写出的字段名（不含路由键 `mode`）。

        CLI `--config` 只用文件里显式写出的字段覆盖选项默认值（`cli/solve/config_file.py`）：配置类的默认值与
        CLI 默认值并不处处相同，不能拿前者去覆盖后者。

        Args:
            config_path: YAML 配置文件路径

        Returns:
            (SteadyConfig 或 TransientConfig, 显式写出的字段名集合)

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

        keys = {k for k in config_dict if k != 'mode'}
        if mode == 'steady':
            return self._load_steady_config(config_dict), keys
        elif mode == 'transient':
            return self._load_transient_config(config_dict), keys
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

        # 创建配置对象
        try:
            config = TransientConfig(**merged)
            logger.info(
                f"瞬态配置已加载: backend={config.backend}, "
                f"dt={config.dt}, 步数={config.n_steps}"
            )
            return config
        except Exception as e:
            raise ValueError(f"无效的瞬态配置: {e}")

    def _merge_defaults(self, user_config: Dict[str, Any], default_config) -> Dict[str, Any]:
        """以 dataclass 默认值为底，覆盖用户给出的字段；未知字段（拼写错误等）直接报错。

        `mode` 不是 dataclass 字段，由 `load_with_keys` 在分派前读走，这里跳过。
        """
        names = [f.name for f in dataclasses.fields(default_config) if f.init]
        merged = {name: getattr(default_config, name) for name in names}
        unknown = sorted(k for k in user_config if k != "mode" and k not in merged)
        if unknown:
            raise ValueError(f"未知的配置键 {unknown}（{type(default_config).__name__} 的字段：{names}）")
        merged.update({k: v for k, v in user_config.items() if k != "mode"})
        return merged

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

        # 只导出构造参数（`__post_init__` 派生的 `total_steps` 等不是可配置项，写进模板会让加载时报
        # "未知的配置键"）
        result = {}
        for f in dataclasses.fields(config):
            if not f.init:
                continue
            value = getattr(config, f.name)
            result[f.name] = value.value if isinstance(value, Enum) else value
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
        lines.append("# 由 ConfigLoader 生成")
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
            # 否则 None 值（例如默认的 phase_max_iter）会呈现为字面文本 "None" -
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
        from autoflowcfd.core.time_integration.base import STEADY_SCHEMES
        from autoflowcfd.core.time_integration.time_accurate import TIME_ACCURATE_SCHEMES

        steady = mode == 'steady'
        schemes = STEADY_SCHEMES if steady else (TimeIntegrationScheme.DUAL_TIME,) + TIME_ACCURATE_SCHEMES
        lo = MIN_STEADY_ORDER if steady else MIN_TRANSIENT_ORDER
        # 每个配置字段一条（`tests/unit/test_config_loader.py` 检查覆盖全部字段）；可选值从枚举/常量读（唯一来源）
        comments = {
            'backend': '计算后端: ' + ', '.join(b.value for b in BackendType) + '（auto：有可用 GPU 用 gpu）',
            'order': f'FR 目标阶数 {lo}~{MAX_ORDER}' + ('（>=1 时从 P0 逐阶爬坡）' if steady else ''),
            'turbulence': '湍流模型: ' + ', '.join(t.value for t in TurbulenceModel),
            'gpu_device': 'GPU 设备号（null：单 GPU 用 0，多 GPU 按 rank % GPU 数分配）',
            'n_threads': f'numba 线程数，-1 = {DEFAULT_N_THREADS}（实测扩展性甜点，不是核数）',
            'output_dir': '结果输出目录',
            'checkpoint_interval': '中间 checkpoint 保存间隔（步）',
            'turbulence_intensity': '来流湍流强度 Tu（默认 0.01=1%）',
            'viscosity_ratio': '来流粘性比 VR=nu_t/nu（默认 5.0）',
            'mu_molecular': '分子动力粘度，单位 Pa*s（默认标准状态下空气 1.8e-5）',
            'artificial_viscosity_enabled': '问题单元人工粘性（熵残差判据，P1 起有效）',
            'artificial_viscosity_alpha': '人工粘性强度标定常数',
            'wall_treatment': 'RANS 近壁处理: auto（平衡壁面模型）/ resolved（要求第一层 y+~1）',
            'phase_max_iter': 'Order Continuation 非最终阶段步数上限（null：按阶段数均分，目标阶数吃满剩余步数）',
            'residual_drop_threshold': 'Order Continuation 非最终阶段提前升阶所需的残差下降倍数',
            'max_iter': '最大迭代步数' if steady else '物理时间步数（total_time 为 null 时生效）',
            'time_scheme': '时间格式: ' + ', '.join(sch.value for sch in schemes),
            # 留空（null）= 按时间格式取 CFL 律的默认值（显式与隐式相差两个数量级，见 build_cfl_policy）
            'cfl_init': '自适应 CFL 初值（null：按时间格式取默认）',
            'cfl_max': '自适应 CFL 上限（null：按时间格式取默认）',
            'cfl_min': '自适应 CFL 下限（null：按时间格式取默认）',
            'convergence_tol': '相对收敛容差（对应 --tol）',
            'low_mach_precond': '低马赫数伪时间预处理（不改变定常解）',
            'dt': '物理时间步长，单位秒',
            'total_time': '总物理时间，单位秒（null：按 max_iter 个物理步）',
            'init_from_checkpoint': '以已有 checkpoint 的场为初场（null：来流均匀场）',
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
