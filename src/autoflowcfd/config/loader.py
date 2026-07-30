"""Configuration loader for YAML files.

This module provides functionality to load, validate, and merge solver
configurations from YAML files.

Key Components:
    - ConfigLoader: Main configuration loader class
    - Default configs for steady and transient simulations

Example:
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
    """YAML configuration file loader and validator.
    
    Supports loading steady-state and transient simulation configurations
    from YAML files, with automatic validation and default value merging.
    
    Attributes:
        default_steady: Default steady-state configuration
        default_transient: Default transient configuration
    
    Example:
        >>> loader = ConfigLoader()
        >>> config = loader.load("config.yaml")
        >>> print(config.backend)
    """
    
    def __init__(self):
        """Initialize configuration loader with default configs."""
        self.default_steady = SteadyConfig()
        self.default_transient = TransientConfig()
    
    def load(self, config_path: Union[str, Path]) -> Union[SteadyConfig, TransientConfig]:
        """Load configuration from YAML file.
        
        Args:
            config_path: Path to YAML configuration file
            
        Returns:
            SteadyConfig or TransientConfig: Loaded configuration object
            
        Raises:
            FileNotFoundError: Configuration file not found
            ValueError: Invalid configuration format
            yaml.YAMLError: YAML parsing error
            
        Example:
            >>> loader = ConfigLoader()
            >>> config = loader.load("simulation.yaml")
        """
        config_path = Path(config_path)
        
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
        logger.info(f"Loading configuration from {config_path}")
        
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config_dict = yaml.safe_load(f)
        except yaml.YAMLError as e:
            raise ValueError(f"Invalid YAML format in {config_path}: {e}")
        
        if not isinstance(config_dict, dict):
            raise ValueError(f"Configuration must be a YAML mapping, got {type(config_dict)}")
        
        # Determine simulation mode
        mode = config_dict.get('mode', 'steady')
        
        if mode == 'steady':
            return self._load_steady_config(config_dict)
        elif mode == 'transient':
            return self._load_transient_config(config_dict)
        else:
            raise ValueError(f"Unknown simulation mode: {mode}. Must be 'steady' or 'transient'")
    
    def _load_steady_config(self, config_dict: Dict[str, Any]) -> SteadyConfig:
        """Load and validate steady-state configuration.
        
        Args:
            config_dict: Configuration dictionary from YAML
            
        Returns:
            SteadyConfig: Validated steady-state configuration
        """
        logger.debug("Loading steady-state configuration")
        
        # Merge with defaults
        merged = self._merge_defaults(config_dict, self.default_steady)
        
        # Convert enum strings to enum values
        merged = self._convert_enums(merged, SteadyConfig)
        
        # Create config object
        try:
            config = SteadyConfig(**merged)
            logger.info(f"Steady config loaded: backend={config.backend}, order={config.order}")
            return config
        except Exception as e:
            raise ValueError(f"Invalid steady configuration: {e}")
    
    def _load_transient_config(self, config_dict: Dict[str, Any]) -> TransientConfig:
        """Load and validate transient configuration.
        
        Args:
            config_dict: Configuration dictionary from YAML
            
        Returns:
            TransientConfig: Validated transient configuration
        """
        logger.debug("Loading transient configuration")
        
        # Merge with defaults
        merged = self._merge_defaults(config_dict, self.default_transient)
        
        # Convert enum strings to enum values
        merged = self._convert_enums(merged, TransientConfig)
        
        # Create config object
        try:
            config = TransientConfig(**merged)
            logger.info(
                f"Transient config loaded: backend={config.backend}, "
                f"dt={config.dt}, total_time={config.total_time}"
            )
            return config
        except Exception as e:
            raise ValueError(f"Invalid transient configuration: {e}")
    
    def _merge_defaults(self, user_config: Dict[str, Any], default_config) -> Dict[str, Any]:
        """Merge user configuration with default values.
        
        User-provided values override defaults. Missing values use defaults.
        
        Args:
            user_config: User-provided configuration dictionary
            default_config: Default configuration object
            
        Returns:
            Dict[str, Any]: Merged configuration dictionary
        """
        # Start with defaults
        merged = {}
        for key in default_config.__dataclass_fields__.keys():
            if hasattr(default_config, key):
                merged[key] = getattr(default_config, key)
        
        # Override with user values
        for key, value in user_config.items():
            if key in merged:
                merged[key] = value
            else:
                logger.warning(f"Unknown configuration key: {key}")
        
        return merged
    
    def _convert_enums(self, config_dict: Dict[str, Any], config_class) -> Dict[str, Any]:
        """Convert string values to enum types.
        
        Args:
            config_dict: Configuration dictionary
            config_class: Target configuration class
            
        Returns:
            Dict[str, Any]: Dictionary with enum values converted
        """
        converted = config_dict.copy()
        
        # Get field annotations
        annotations = getattr(config_class, '__annotations__', {})
        
        for key, value in config_dict.items():
            if key not in annotations:
                continue
            
            field_type = annotations[key]
            
            # Convert BackendType
            if field_type == BackendType and isinstance(value, str):
                try:
                    converted[key] = BackendType(value.lower())
                except ValueError:
                    raise ValueError(
                        f"Invalid backend type: {value}. "
                        f"Must be one of: {[b.value for b in BackendType]}"
                    )
            
            # Convert TurbulenceModel
            elif field_type == TurbulenceModel and isinstance(value, str):
                try:
                    converted[key] = TurbulenceModel(value.lower())
                except ValueError:
                    raise ValueError(
                        f"Invalid turbulence model: {value}. "
                        f"Must be one of: {[t.value for t in TurbulenceModel]}"
                    )
            
            # Convert TimeIntegrationScheme
            elif field_type == TimeIntegrationScheme and isinstance(value, str):
                try:
                    converted[key] = TimeIntegrationScheme(value.lower())
                except ValueError:
                    raise ValueError(
                        f"Invalid time integration scheme: {value}. "
                        f"Must be one of: {[s.value for s in TimeIntegrationScheme]}"
                    )
        
        return converted
    
    def save_template(self, output_path: Union[str, Path], mode: str = 'steady'):
        """Save configuration template to YAML file.
        
        Args:
            output_path: Output file path
            mode: Simulation mode ('steady' or 'transient')
            
        Example:
            >>> loader = ConfigLoader()
            >>> loader.save_template("config_template.yaml", mode="steady")
        """
        output_path = Path(output_path)
        
        if mode == 'steady':
            config = self.default_steady
        elif mode == 'transient':
            config = self.default_transient
        else:
            raise ValueError(f"Unknown mode: {mode}. Must be 'steady' or 'transient'")
        
        # Convert enums to strings for YAML serialization
        config_dict = self._enum_to_string(config)
        config_dict['mode'] = mode
        
        # Add comments as YAML structure
        yaml_content = self._add_yaml_comments(config_dict, mode)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(yaml_content)
        
        logger.info(f"Configuration template saved to {output_path}")
    
    def _enum_to_string(self, config) -> Dict[str, Any]:
        """Convert enum values to strings for YAML serialization.
        
        Args:
            config: Configuration object
            
        Returns:
            Dict[str, Any]: Dictionary with string values
        """
        result = {}
        for key, value in config.__dict__.items():
            if isinstance(value, Enum):
                result[key] = value.value
            else:
                result[key] = value
        return result
    
    def _add_yaml_comments(self, config_dict: Dict[str, Any], mode: str) -> str:
        """Add helpful comments to YAML output.
        
        Args:
            config_dict: Configuration dictionary
            mode: Simulation mode
            
        Returns:
            str: YAML content with comments
        """
        lines = [f"# AutoFlowCFD {mode.capitalize()} Simulation Configuration"]
        lines.append(f"# Generated by ConfigLoader")
        lines.append("")
        lines.append(f"mode: {mode}")
        lines.append("")
        
        for key, value in config_dict.items():
            if key == 'mode':
                continue
            
            # Add comment for important parameters
            comment = self._get_parameter_comment(key, mode)
            if comment:
                lines.append(f"# {comment}")
            
            lines.append(f"{key}: {value}")
            lines.append("")
        
        return "\n".join(lines)
    
    def _get_parameter_comment(self, param: str, mode: str) -> str:
        """Get help comment for configuration parameter.
        
        Args:
            param: Parameter name
            mode: Simulation mode
            
        Returns:
            str: Help comment text
        """
        comments = {
            'backend': 'Compute backend: cpu, gpu, or auto',
            'order': 'FR discretization order: 1, 2, or 3',
            'turbulence': 'Turbulence model: sst_kw, sa, des, ddes, les',
            'max_iter': 'Maximum iteration steps (steady only)',
            'cfl_init': 'Initial CFL number (steady only)',
            'cfl_max': 'Maximum CFL number (steady only)',
            'convergence_tol': 'Convergence tolerance for residuals (steady only)',
            'dt': 'Time step size in seconds (transient only)',
            'total_time': 'Total physical time in seconds (transient only)',
            'time_scheme': 'Time integration scheme: backward_euler, rk2, rk3, ab3 (transient only)',
            'output_dir': 'Output directory for results',
            'checkpoint_interval': 'Checkpoint save interval in steps',
            'growth_rate': 'Boundary-layer geometric growth rate (steady only)',
            'max_layers': 'Max boundary-layer + transition layer count (steady only)',
            'min_cell_size': 'First (near-wall) layer thickness in meters (steady only)',
            'target_cells': 'Target total cell count (steady only; ignored by the tetgen hybrid mesh path)',
            'rho_inf': 'Freestream density in kg/m^3',
            'vel_inf': 'Freestream velocity magnitude in m/s',
            'p_inf': 'Freestream static pressure in Pa',
        }
        return comments.get(param, '')


def load_config(config_path: Union[str, Path]) -> Union[SteadyConfig, TransientConfig]:
    """Convenience function to load configuration from YAML file.
    
    Args:
        config_path: Path to YAML configuration file
        
    Returns:
        SteadyConfig or TransientConfig: Loaded configuration
        
    Example:
        >>> from autoflowcfd.config import load_config
        >>> config = load_config("simulation.yaml")
    """
    loader = ConfigLoader()
    return loader.load(config_path)


def save_config_template(output_path: Union[str, Path], mode: str = 'steady'):
    """Convenience function to save configuration template.
    
    Args:
        output_path: Output file path
        mode: Simulation mode ('steady' or 'transient')
        
    Example:
        >>> from autoflowcfd.config import save_config_template
        >>> save_config_template("config.yaml", mode="steady")
    """
    loader = ConfigLoader()
    loader.save_template(output_path, mode)
