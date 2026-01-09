# config.py
import os
import platform
import logging
from typing import List, Optional
from enum import Enum
from pathlib import Path
from functools import lru_cache

from pydantic import Field, field_validator, computed_field
from pydantic_settings import BaseSettings, SettingsConfigDict

logger = logging.getLogger("BotConfig")


class Environment(str, Enum):
    PRODUCTION = "production"
    TESTNET = "testnet"
    DEVELOPMENT = "development"


class Settings(BaseSettings):
    """
    Tüm bot yapılandırmasını tek merkezden yöneten ana sınıf.
    Pydantic v2 kullanır.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # ------------------------------------------------------------------
    # CORE BOT SETTINGS
    # ------------------------------------------------------------------
    TELEGRAM_TOKEN: str
    TELEGRAM_NAME: str = "binance_bot"
    ADMIN_IDS: List[int] = Field(default_factory=list)
    DEBUG: bool = False
    ENV: Environment = Environment.PRODUCTION

    # ------------------------------------------------------------------
    # BINANCE
    # ------------------------------------------------------------------
    BINANCE_API_KEY: str = ""
    BINANCE_API_SECRET: str = ""
    ENABLE_TRADING: bool = False

    # ------------------------------------------------------------------
    # WEBHOOK / DEPLOYMENT
    # ------------------------------------------------------------------
    PORT: int = 3000
    WEBHOOK_HOST: Optional[str] = None
    WEBHOOK_SECRET: str = ""

    @computed_field
    @property
    def BOT_MODE(self) -> str:
        return "webhook" if self.ENV == Environment.PRODUCTION else "polling"

    @computed_field
    @property
    def WEBHOOK_URL(self) -> str:
        if not self.WEBHOOK_HOST:
            return ""
        return f"{self.WEBHOOK_HOST.rstrip('/')}/webhook"

    # ------------------------------------------------------------------
    # RUNTIME / DATA
    # ------------------------------------------------------------------
    RUNTIME_DIR: Path = Field(default=Path("/tmp/zbot1"))

    @computed_field
    @property
    def DATA_DIR(self) -> Path:
        env_path = os.getenv("DATABASE_PATH")
        if env_path:
            return Path(env_path)

        if platform.system() == "Windows":
            return Path.cwd() / "data"

        base_dir = Path("/home/ubuntu/bot_persistence")
        if not base_dir.parent.exists():
            return Path.cwd() / "data"

        return base_dir / "data"

    @computed_field
    @property
    def DATABASE_URL(self) -> str:
        self.DATA_DIR.mkdir(parents=True, exist_ok=True)
        return str(self.DATA_DIR / "apikeys.db")

    # ------------------------------------------------------------------
    # SECURITY
    # ------------------------------------------------------------------
    MASTER_KEY: str

    @field_validator("MASTER_KEY", mode="before")
    @classmethod
    def validate_master_key(cls, v: str) -> str:
        """
        MASTER_KEY:
        - ZORUNLU
        - Fallback YOK
        - Otomatik üretim YOK
        """
        if not v or not v.strip():
            raise ValueError(
                "❌ MASTER_KEY tanımlı değil! "
                ".env veya environment variable olarak SET EDİLMELİ."
            )
        return v

    # ------------------------------------------------------------------
    # OTHER SETTINGS
    # ------------------------------------------------------------------
    COLLECT_INTERVAL_SECONDS: int = Field(
        default=600,
        description="Market collector çalışma aralığı (saniye)",
    )

    SCAN_SYMBOLS: List[str] = Field(
        default=[
            "BTCUSDT","ETHUSDT","BNBUSDT",
            "SOLUSDT","ARPAUSDT","PEPEUSDT",
            "FETUSDT","TURBOUSDT","SUIUSDT",
        ]
    )

    SCAN_DEFAULT_COUNT: int = 100
    MAX_LEVERAGE: int = 3

    USE_REDIS: bool = False
    REDIS_URL: str = "redis://localhost:6379/0"

    BINANCE_BASE_URL: str = "https://api.binance.com"
    BINANCE_TESTNET_URL: str = "https://testnet.binance.vision"
    RECV_WINDOW: int = 5000

    # ------------------------------------------------------------------
    # FINAL VALIDATION
    # ------------------------------------------------------------------
    def validate_setup(self) -> None:
        """
        Kritik kontroller.
        UYGULAMA ÖLDÜRMEZ.
        SADECE HATA FIRLATIR.
        """
        if not self.TELEGRAM_TOKEN:
            raise RuntimeError("❌ TELEGRAM_TOKEN eksik!")

        if self.ENABLE_TRADING and (
            not self.BINANCE_API_KEY or not self.BINANCE_API_SECRET
        ):
            logger.warning(
                "⚠️ Trading aktif ama BINANCE API anahtarları eksik!"
            )


# ----------------------------------------------------------------------
# SINGLETON
# ----------------------------------------------------------------------
@lru_cache
def get_settings() -> Settings:
    settings = Settings()
    settings.validate_setup()
    return settings


# Kolay erişim
config = get_settings()
