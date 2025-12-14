"""
Configuration settings
"""

import os
from typing import Optional
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings"""
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8"
    )
    
    # API Settings
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    api_reload: bool = False
    
    # Database Settings
    lci_database_path: Optional[str] = None
    grid_data_api_key: Optional[str] = None
    grid_data_source: str = "electricity_maps"
    
    # API Keys
    electricity_maps_api_key: Optional[str] = None
    watttime_username: Optional[str] = None
    watttime_password: Optional[str] = None
    entsoe_security_token: Optional[str] = None
    
    # Cache Settings
    cache_dir: Optional[str] = "data/cache"
    cache_enabled: bool = True
    cache_max_age_hours: int = 1
    
    # Model Settings
    model_checkpoint_path: Optional[str] = None
    enable_uncertainty: bool = True
    
    # Logging
    log_level: str = "INFO"


_settings: Optional[Settings] = None


def get_settings() -> Settings:
    """Get application settings (singleton)"""
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings

