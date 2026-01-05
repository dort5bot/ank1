# config.py
import os
import logging
import sys
from enum import Enum
from typing import List, Optional, Any, Dict, ClassVar
from pathlib import Path
from functools import lru_cache
from dotenv import load_dotenv
from pydantic import Field, field_validator, computed_field
from pydantic_settings import BaseSettings, SettingsConfigDict

from cryptography.fernet import Fernet

# .env dosyasını yükle
load_dotenv()

# Logger Kurulumu
logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("BotConfig")

class Environment(str, Enum):
    PRODUCTION = "production"
    TESTNET = "testnet"
    DEVELOPMENT = "development"

class Settings(BaseSettings):
    """
    Tüm bot yapılandırmasını tek merkezden yöneten ana sınıf.
    Pydantic V2 kullanarak otomatik tip dönüşümü ve validasyon sağlar.
    """
    model_config = SettingsConfigDict(
        env_file=".env", 
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False
    )

    # --- CORE BOT SETTINGS ---
    TELEGRAM_TOKEN: str
    TELEGRAM_NAME: str = "binance_bot"
    ADMIN_IDS: List[int] = Field(default_factory=list)
    DEBUG: bool = False
    ENV: Environment = Environment.PRODUCTION
    
    # --- BINANCE CREDENTIALS ---
    BINANCE_API_KEY: str = ""
    BINANCE_API_SECRET: str = ""
    ENABLE_TRADING: bool = False
    
    # --- WEBHOOK / DEPLOYMENT ---
    # Render, Oracle, Heroku gibi platformlarda otomatik PORT atanır
    PORT: int = Field(default=3000, alias="PORT") 
    WEBHOOK_HOST: Optional[str] = None
    WEBHOOK_SECRET: str = ""
    
    @computed_field
    @property
    def BOT_MODE(self) -> str:
        """Render veya Oracle gibi platformlarda otomatik olarak webhook moduna geçer."""
        platforms = ['RENDER', 'RAILWAY', 'HEROKU', 'ORACLE_CLOUD', 'DIGITALOCEAN']
        if any(env in os.environ for env in platforms) or self.WEBHOOK_HOST:
            return "webhook"
        return "polling"

    @computed_field
    @property
    def WEBHOOK_URL(self) -> str:
        if not self.WEBHOOK_HOST: return ""
        return f"{self.WEBHOOK_HOST.rstrip('/')}/webhook/{self.TELEGRAM_TOKEN}"

    # --- ENCRYPTION & SECURITY ---
    MASTER_KEY: str = Field(default="")
    
    # DATABASE_URL: str = "data/apikeys.db"
    # RUNTIME_DIR = Path(os.getenv("RUNTIME_DIR", "/tmp/zbot1"))
    RUNTIME_DIR: Path = Field(default=Path("/tmp/zbot1"))

    # DATABASE_URL: str = str(RUNTIME_DIR / "data" / "apikeys.db")

    @computed_field
    @property
    def DATABASE_URL(self) -> str:
        return str(self.RUNTIME_DIR / "data" / "apikeys.db")



    @field_validator("MASTER_KEY", mode="before")
    @classmethod
    def validate_master_key(cls, v: str) -> str:
        """Anahtar yoksa oluşturur veya geçerli olup olmadığını kontrol eder."""
        if not v:
            # Fallback mantığı: Çevresel değişkenlerde ara
            for alt in ["ENCRYPTION_KEY", "FERNET_KEY"]:
                if os.getenv(alt): return os.getenv(alt)
            
            # Hala yoksa geçici anahtar üret (Data klasörüne yaz)
            logger.warning("🚨 MASTER_KEY bulunamadı! Geçici anahtar üretiliyor.")
            new_key = Fernet.generate_key().decode()
            return new_key
        return v


    # market_collector  için zamanlayıcı
    COLLECT_INTERVAL_SECONDS: int = 600 # 10 dk


    # --- SCAN & TRADING PARAMS ---
    # p12_handler, a11_handler için - sadece sembol listesi
    SCAN_SYMBOLS: List[str] = Field(default = [
        "BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT", "ARPAUSDT", 
        "PEPEUSDT", "FETUSDT", "TURBOUSDT", "SUIUSDT"
    ])
    # SİL: Diğer handler'lar için gerekebilecek ayarlar KULLANILMIYORSA
    SCAN_DEFAULT_COUNT: int = 50
    MAX_LEVERAGE: int = 3
    
    # --- REDIS (AIOGRAM FSM) ---
    USE_REDIS: bool = False
    REDIS_URL: str = "redis://localhost:6379/0"

    # --- BINANCE API INTERNAL (Sabitler) ---
    BINANCE_BASE_URL: str = "https://api.binance.com"
    BINANCE_TESTNET_URL: str = "https://testnet.binance.vision"
    RECV_WINDOW: int = 5000

    def validate_setup(self):
        """Kritik kontrolleri yapar."""
        if not self.TELEGRAM_TOKEN:
            logger.error("❌ TELEGRAM_TOKEN eksik!")
            sys.exit(1)
        
        if self.ENABLE_TRADING and (not self.BINANCE_API_KEY or not self.BINANCE_API_SECRET):
            logger.warning("⚠️ Trading aktif ama API anahtarları eksik!")

        # Veritabanı klasörünü oluştur
        db_path = Path(self.DATABASE_URL).parent
        if db_path: db_path.mkdir(parents=True, exist_ok=True)

# --- SINGLETON INSTANCE ---
@lru_cache()
def get_settings() -> Settings:
    settings = Settings()
    settings.validate_setup()
    return settings

# Kolay erişim için instance
config = get_settings()