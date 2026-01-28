"""Configuration management for the MIA framework."""

from typing import Dict, Any, Optional
from attack.misc.io import load_config


class ConfigManager:
    """Manages configuration for the MIA framework."""

    def __init__(self, config_path: str):
        """Initialize the configuration manager."""
        self.config = load_config(config_path)
        self.global_config = self.config.get('global', {})

    def get_global_config(self) -> Dict[str, Any]:
        """Get the global configuration."""
        return self.global_config

    def get_attack_config(self, attack_name: str) -> Dict[str, Any]:
        """Get configuration for a specific attack."""
        attack_config = self.config.get(attack_name, {}).copy()
        attack_config.update(self.global_config)
        return attack_config

    def get_available_attacks(self) -> Dict[str, str]:
        """Get available attacks from config while preserving order."""
        return {k: v['module'] for k, v in self.config.items() if k != 'global'}

    def get_datasets(self) -> list:
        """Get dataset configurations."""
        return self.global_config.get('datasets', [])