"""配置管理子命令。

本模块提供用于管理仿真配置的 CLI 命令。

Commands:
    - init: 生成配置模板
    - show: 显示当前配置
    - validate: 验证配置文件

Example:
    $ autoflowcfd config init --template steady
    $ autoflowcfd config validate simulation.yaml
"""

import click
import json
from pathlib import Path
from loguru import logger


@click.group()
def config() -> None:
    """配置管理命令。
    
    创建、验证和管理 YAML 配置文件。
    
    Examples:
        # 创建配置模板
        $ autoflowcfd config init --template steady
        
        # 验证配置
        $ autoflowcfd config validate simulation.yaml
    """
    pass


@config.command()
@click.option("--template", "-t", type=click.Choice(["steady", "transient"]),
              required=True, help="配置模板类型")
@click.option("--output", "-o", type=click.Path(), default="config.yaml",
              help="输出文件路径")
def init(template: str, output: str) -> None:
    """生成配置文件模板。
    
    为稳态或瞬态仿真创建带有合理默认值的 YAML 配置模板。
    
    Args:
        template: 模板类型（稳态/瞬态）
        output: 输出文件路径
    
    Examples:
        # 稳态模板
        $ autoflowcfd config init --template steady -o steady_config.yaml
        
        # 瞬态模板
        $ autoflowcfd config init --template transient -o transient_config.yaml
    """
    from autoflowcfd.config import SteadyConfig, TransientConfig
    
    logger.info(f"正在生成 {template} 配置模板")
    
    try:
        output_path = Path(output)
        
        if template == "steady":
            config_obj = SteadyConfig()
            template_content = f"""# AutoFlowCFD 稳态配置
# 由 'autoflowcfd config init' 生成

mode: steady

# 求解器设置
backend: auto  # cpu, gpu, 或 auto
order: 3       # FR 阶数 (1, 2, 或 3)
turbulence: sst_kw  # sst_kw, sa

# 收敛设置
max_iter: 5000
cfl_init: 0.1
cfl_max: 5.0
convergence_tol: 1.0e-6

# 输出设置
output_dir: ./results
checkpoint_interval: 100
verbose: false
"""
        else:  # transient
            config_obj = TransientConfig()
            template_content = f"""# AutoFlowCFD 瞬态配置
# 由 'autoflowcfd config init' 生成

mode: transient

# 求解器设置
backend: auto  # cpu, gpu, 或 auto
order: 3       # FR 阶数 (1, 2, 或 3)
turbulence: des  # des, ddes, les

# 时间积分
time_integration: backward_euler  # backward_euler, rk2, rk3, ab3
dt: 1.0e-4     # 时间步长 (s)
total_time: 0.3  # 总物理时间 (s)

# 采样设置
sample_interval: 10

# 输出设置
output_dir: ./transient_results
checkpoint_interval: 100
verbose: false
"""
        
        # 写入文件
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(template_content)
        
        logger.info(f"配置模板已保存至 {output_path}")
        click.echo(f"✓ 配置模板已创建: {output}")
        click.echo(f"  类型: {template}")
        click.echo(f"\n编辑此文件以自定义您的仿真设置。")
    
    except Exception as e:
        logger.error(f"创建配置模板失败: {e}")
        raise click.ClickException(f"创建配置模板失败: {e}")


@config.command()
@click.argument("config_file", type=click.Path(exists=True))
@click.option("--json", "-j", "json_output", is_flag=True, help="以 JSON 格式输出")
def show(config_file: str, json_output: bool) -> None:
    """显示配置文件内容。
    
    加载并显示 YAML 配置文件，并进行验证。
    
    Args:
        config_file: YAML 配置文件路径
        json_output: 以 JSON 格式输出
    
    Examples:
        # 显示配置
        $ autoflowcfd config show simulation.yaml
        
        # JSON 输出
        $ autoflowcfd config show simulation.yaml --json
    """
    from autoflowcfd.config import ConfigLoader
    
    logger.info(f"正在加载配置: {config_file}")
    
    try:
        loader = ConfigLoader()
        config_obj = loader.load(config_file)
        
        # 转换为字典以便显示
        if hasattr(config_obj, '__dict__'):
            config_dict = vars(config_obj)
        else:
            config_dict = config_obj.__dict__
        
        if json_output:
            # 将枚举转换为字符串
            def enum_serializer(obj):
                if hasattr(obj, 'value'):
                    return obj.value
                raise TypeError(f"Object of type {type(obj)} is not JSON serializable")
            
            click.echo(json.dumps(config_dict, indent=2, default=enum_serializer))
        else:
            click.echo(f"\n配置: {config_file}")
            click.echo(f"{'='*60}")
            for key, value in config_dict.items():
                # 格式化枚举值
                if hasattr(value, 'value'):
                    value = value.value
                click.echo(f"{key:<25} {value}")
            click.echo(f"{'='*60}")
    
    except Exception as e:
        logger.error(f"加载配置失败: {e}")
        raise click.ClickException(f"加载配置失败: {e}")


@config.command()
@click.argument("config_file", type=click.Path(exists=True))
@click.option("--json", "-j", "json_output", is_flag=True, help="以 JSON 格式输出")
def validate(config_file: str, json_output: bool) -> None:
    """验证配置文件。
    
    检查配置文件是否存在语法错误和无效值。
    
    Args:
        config_file: YAML 配置文件路径
        json_output: 以 JSON 格式输出
    
    Examples:
        # 验证配置
        $ autoflowcfd config validate simulation.yaml
        
        # JSON 输出
        $ autoflowcfd config validate simulation.yaml --json
    """
    from autoflowcfd.config import ConfigLoader
    
    logger.info(f"正在验证配置: {config_file}")
    
    try:
        loader = ConfigLoader()
        config_obj = loader.load(config_file)
        
        result = {
            "command": "config.validate",
            "status": "valid",
            "file": config_file,
            "mode": config_obj.mode if hasattr(config_obj, 'mode') else "unknown",
        }
        
        if json_output:
            click.echo(json.dumps(result, indent=2))
        else:
            mode = config_obj.mode if hasattr(config_obj, 'mode') else "unknown"
            click.echo(f"✓ 配置有效")
            click.echo(f"  文件: {config_file}")
            click.echo(f"  模式: {mode}")
    
    except Exception as e:
        logger.error(f"配置验证失败: {e}")
        result = {
            "command": "config.validate",
            "status": "invalid",
            "file": config_file,
            "error": str(e),
        }
        
        if json_output:
            click.echo(json.dumps(result, indent=2))
        
        raise click.ClickException(f"配置验证失败: {e}")
