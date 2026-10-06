#!/usr/bin/env python3
"""
Configuration loader for HNSW benchmark.
Loads parameters from config.yaml with sensible defaults.
"""
import os
import yaml
from typing import Any, Dict, Optional


class Config:
    """Configuration manager with dot notation access."""
    
    def __init__(self, config_dict: Dict[str, Any]):
        self._config = config_dict
    
    def __getattr__(self, name: str) -> Any:
        if name in self._config:
            value = self._config[name]
            if isinstance(value, dict):
                return Config(value)
            return value
        raise AttributeError(f"Config has no attribute '{name}'")
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get value with dot notation (e.g., 'hnsw.M')."""
        keys = key.split('.')
        value = self._config
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        return value
    
    def to_dict(self) -> Dict[str, Any]:
        return self._config


def load_config(config_path: str = "config.yaml") -> Config:
    """
    Load configuration from YAML file.
    
    Args:
        config_path: Path to config.yaml
        
    Returns:
        Config object with dot notation access
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        config_dict = yaml.safe_load(f)
    
    return Config(config_dict)


# Global config instance (loaded on first import)
_config: Optional[Config] = None


def get_config(config_path: str = "config.yaml") -> Config:
    """Get global config instance (singleton pattern)."""
    global _config
    if _config is None:
        _config = load_config(config_path)
    return _config


def reset_config():
    """Reset global config (useful for testing)."""
    global _config
    _config = None


if __name__ == "__main__":
    # Test config loading
    cfg = load_config("config.yaml")
    print("Config loaded successfully!")
    print(f"HNSW M_values: {cfg.hnsw.M_values}")
    print(f"HNSW ef_search_values: {cfg.hnsw.ef_search_values}")
    print(f"Data base_path: {cfg.data.base_path}")