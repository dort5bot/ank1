# utils/binance_api/bi_config.py class BinanceConfig
from __future__ import annotations
import os
from typing import Optional, Dict, Any
from dataclasses import dataclass, field
from pathlib import Path

@dataclass
class BinanceConfig:
    """Centralized configuration management"""
    base_path: str = field(default_factory=lambda: os.getenv("BINANCE_MAPS_PATH", ""))
    api_timeout: int = field(default_factory=lambda: int(os.getenv("BINANCE_API_TIMEOUT", "30")))
    max_retries: int = field(default_factory=lambda: int(os.getenv("BINANCE_MAX_RETRIES", "3")))
    cache_ttl: int = field(default_factory=lambda: int(os.getenv("BINANCE_CACHE_TTL", "300")))
    cleanup_interval: int = field(default_factory=lambda: int(os.getenv("BINANCE_CLEANUP_INTERVAL", "300")))
    
    def __post_init__(self):
        """Post-initialization validation"""
        if not self.base_path:
            self.base_path = self._discover_maps_path()
        
        self._validate_path()

    def _discover_maps_path(self) -> str:
        """Automatically discover maps directory"""
        possible_paths = [
            # Docker container path
            "/app/binance_maps",
            # Local development
            Path(__file__).parent.parent.parent / "binance_maps",
            # Python package path
            Path(__file__).parent / "maps",
            # Current directory
            Path.cwd() / "binance_maps"
        ]
        
        for path in possible_paths:
            path_str = str(path)
            if Path(path_str).exists():
                return path_str
        
        # Create default directory
        default_path = Path.cwd() / "binance_maps"
        default_path.mkdir(exist_ok=True)
        return str(default_path)

    def _validate_path(self):
        """Validate base path exists"""
        if not Path(self.base_path).exists():
            raise ValueError(f"Maps directory not found: {self.base_path}")

    @classmethod
    def from_env(cls) -> BinanceConfig:
        """Create config from environment variables"""
        return cls()