"""Core configuration management for the Breadfruit Dataset Engineering Pipeline."""
import yaml
import os
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, Optional


def _to_namespace(d):
    """Recursively convert dictionaries to SimpleNamespace for attribute access."""
    if isinstance(d, dict):
        for key, value in d.items():
            d[key] = _to_namespace(value)
        return SimpleNamespace(**d)
    if isinstance(d, list):
        return [_to_namespace(item) if isinstance(item, dict) else item for item in d]
    return d


class ConfigManager:
    """Manages pipeline configuration loading and access."""
    
    _instance: Optional['ConfigManager'] = None
    _config: Optional[SimpleNamespace] = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        if self._config is None:
            self._config = None
    
    def load(self, config_path: str) -> SimpleNamespace:
        """Load configuration from YAML file."""
        with open(config_path, 'r') as f:
            raw_config = yaml.safe_load(f)
        
        self._config = _to_namespace(raw_config)
        return self._config
    
    @property
    def config(self) -> SimpleNamespace:
        if self._config is None:
            raise RuntimeError("Configuration not loaded. Call load() first.")
        return self._config
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get nested config value using dot notation."""
        keys = key.split('.')
        value = self._config
        for k in keys:
            if hasattr(value, k):
                value = getattr(value, k)
            elif isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        return value


def get_config(config_path: Optional[str] = None) -> SimpleNamespace:
    """Get the global configuration instance."""
    manager = ConfigManager()
    if manager._config is None:
        if config_path is None:
            config_path = Path(__file__).parent.parent / "config" / "config.yaml"
        manager.load(str(config_path))
    return manager.config
