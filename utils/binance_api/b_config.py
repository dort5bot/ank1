from __future__ import annotations

import os
from dataclasses import dataclass, field


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, default))
    except ValueError:
        return default


@dataclass(slots=True)
class BinanceConfig:
    """
    Production-ready Binance runtime configuration.

    - Stateless
    - Filesystem bağımsız
    - Container / serverless uyumlu
    """

    # ⏱ HTTP & Network
    api_timeout: int = field(default_factory=lambda: _env_int("BINANCE_API_TIMEOUT", 30))
    max_retries: int = field(default_factory=lambda: _env_int("BINANCE_MAX_RETRIES", 3))

    # 🧠 Cache & Performance
    cache_ttl: int = field(default_factory=lambda: _env_int("BINANCE_CACHE_TTL", 300))

    # 🧹 Background cleanup
    cleanup_interval: int = field(default_factory=lambda: _env_int("BINANCE_CLEANUP_INTERVAL", 300))

    # 🔐 Security / Limits
    max_concurrent_requests: int = field(
        default_factory=lambda: _env_int("BINANCE_MAX_CONCURRENT_REQUESTS", 10)
    )

    # 🧪 Debug / Diagnostics
    debug: bool = field(
        default_factory=lambda: os.getenv("BINANCE_DEBUG", "false").lower() == "true"
    )

    def __post_init__(self) -> None:
        self._validate()

    def _validate(self) -> None:
        if self.api_timeout <= 0:
            raise ValueError("api_timeout must be > 0")

        if self.max_retries < 0:
            raise ValueError("max_retries must be >= 0")

        if self.cache_ttl < 0:
            raise ValueError("cache_ttl must be >= 0")

        if self.cleanup_interval <= 0:
            raise ValueError("cleanup_interval must be > 0")

        if self.max_concurrent_requests <= 0:
            raise ValueError("max_concurrent_requests must be > 0")

    @classmethod
    def from_env(cls) -> "BinanceConfig":
        """
        Explicit factory method.
        (İleride config source değişirse tek yerden kontrol edilir)
        """
        return cls()
