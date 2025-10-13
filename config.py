# config.py - ank1
import os
import logging
import sys
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Set, Tuple
from dotenv import load_dotenv
from enum import Enum
from typing import get_type_hints   # type safety için ek validation
from pathlib import Path
from pydantic_settings import BaseSettings  # v2 - doğru
from functools import lru_cache
#from cryptography.fernet import Fernet, InvalidKey
from cryptography.fernet import Fernet, InvalidToken


# Environment variables'ı yükle
load_dotenv()

# Logging yapılandırması
def _setup_logger():
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.getLevelName(os.getenv("LOG_LEVEL", "INFO")))
    return logger

logger = _setup_logger()


# Global async-safe config instance
_CONFIG_INSTANCE: Optional["BotConfig"] = None


class Environment(Enum):
    """Runtime environment options."""
    PRODUCTION = "production"
    TESTNET = "testnet"
    DEVELOPMENT = "development"



# Config içine ekleyebilirsiniz (opsiyonel) polling/ webhook
@property
def suggested_bot_mode(self) -> str:
    """Ortama göre önerilen bot modu"""
    if any(env_var in os.environ for env_var in ['RENDER', 'RAILWAY', 'HEROKU', 'ORACLE']):
        return "webhook"
    return "polling"
    

@dataclass
class EnvironmentConfig:
    """Environment-specific configuration."""
    ENV: Environment = Environment.PRODUCTION
    DEBUG: bool = False
    TESTNET: bool = False


# Binance alt class - icine eklenemeyenler
@dataclass
class SecurityConfig:
    """Security and authentication configuration."""
    # Bot default credentials
    BOT_API_KEY: Optional[str] = None
    BOT_API_SECRET: Optional[str] = None
    
    # Multi-user API key management
    API_KEY_MANAGER_ENABLED: bool = True
    API_KEY_CACHE_TTL: int = 3600  # seconds
    
    # Security settings
    DEFAULT_RECV_WINDOW: int = 5000  # ms
    ENABLE_SIGNATURE_VALIDATION: bool = True
    REQUEST_SIGNING_REQUIRED: bool = True


class BinanceConfig:
    """
    Binance API configuration with organized class blocks for maintainability.
    
    This class provides structured configuration for all Binance API related settings
    including URLs, rate limiting, WebSocket, and trading parameters.
    """

    def __init__(self):
        """Initialize Binance configuration with nested config classes."""
        self.environment = EnvironmentConfig()
        self.security = SecurityConfig()

    class URLs:
        """Binance API URL configurations."""
        BASE_URL: str = "https://api.binance.com"
        FUTURES_URL: str = "https://fapi.binance.com"
        MARGIN_URL: str = "https://api.binance.com"
        TESTNET_BASE_URL: str = "https://testnet.binance.vision"
        TESTNET_FUTURES_URL: str = "https://testnet.binancefuture.com"
        
        # WebSocket URLs
        WS_BASE_URL: str = "wss://stream.binance.com:9443/ws"
        WS_FUTURES_URL: str = "wss://fstream.binance.com/ws"
        WS_TESTNET_BASE_URL: str = "wss://testnet.binance.vision/ws"
        WS_TESTNET_FUTURES_URL: str = "wss://stream.binancefuture.com/ws"

    class RateLimiting:
        """Rate limiting configuration."""
        ENABLED: bool = True
        REQUESTS_PER_SECOND: float = 10.0
        MIN_REQUEST_INTERVAL: float = 0.1  # seconds
        
        # Binance specific limits
        IP_LIMIT_PER_MINUTE: int = 1200
        ORDER_LIMIT_PER_SECOND: int = 10
        RAW_REQUESTS_PER_5MIN: int = 5000
        
        # Adaptive rate limiting
        ADAPTIVE_RATE_LIMITING: bool = True
        WEIGHT_BUFFER_PERCENTAGE: float = 0.1  # 10% buffer

    class CircuitBreaker:
        """Circuit breaker configuration for fault tolerance."""
        ENABLED: bool = True
        FAILURE_THRESHOLD: int = 5
        RESET_TIMEOUT: float = 60.0  # seconds
        HALF_OPEN_TIMEOUT: float = 30.0  # seconds
        MAX_HALF_OPEN_CALLS: int = 1
        
        # Multi-user circuit breaker
        USER_CIRCUIT_BREAKER_ENABLED: bool = True
        MAX_CACHE_SIZE: int = 1000
        TTL_SECONDS: int = 3600

    class HTTPClient:
        """HTTP client configuration for async requests."""
        TIMEOUT: int = 30  # seconds
        CONNECT_TIMEOUT: int = 5
        SOCK_CONNECT_TIMEOUT: int = 5
        SOCK_READ_TIMEOUT: int = 10
        
        # Connection pooling
        CONNECTOR_LIMIT: int = 100
        CONNECTOR_LIMIT_PER_HOST: int = 20
        ENABLE_CLEANUP_CLOSED: bool = True
        
        # Retry configuration
        MAX_RETRIES: int = 3
        RETRY_DELAY: float = 1.0  # seconds
        EXPONENTIAL_BACKOFF: bool = True

    class WebSocket:
        """WebSocket configuration for real-time data."""
        ENABLED: bool = True
        RECONNECT_INTERVAL: int = 5  # seconds
        MAX_RECONNECT_DELAY: int = 60  # seconds
        PING_TIMEOUT: float = 30.0  # seconds
        
        # Multi-user WebSocket
        MULTI_USER_ENABLED: bool = True
        USER_STREAM_KEEPALIVE_INTERVAL: int = 1800  # 30 minutes
        
        # Message handling
        MAX_QUEUE_SIZE: int = 1000
        PROCESS_CONCURRENT_MESSAGES: bool = True

    class Metrics:
        """Metrics and monitoring configuration."""
        ENABLED: bool = True
        COLLECTION_INTERVAL: int = 60  # seconds
        
        # Response time tracking
        RESPONSE_TIME_WINDOW_SIZE: int = 5000
        TRACK_PERCENTILES: bool = True
        PERCENTILES: List[float] = [0.5, 0.95, 0.99]
        
        # Health monitoring
        HEALTH_CHECK_INTERVAL: int = 30  # seconds
        PERFORMANCE_GRADING_ENABLED: bool = True
        
        # Prometheus integration
        PROMETHEUS_ENABLED: bool = False
        EXPOSITION_PORT: int = 8000

    class Logging:
        """Logging configuration for Binance operations."""
        ENABLED: bool = True
        LEVEL: str = "INFO"
        JSON_FORMAT: bool = False
        INCLUDE_METRICS: bool = True
        
        # File logging
        FILE_LOGGING_ENABLED: bool = False
        LOG_FILE_PATH: str = "logs/binance_bot.log"
        MAX_LOG_FILE_SIZE: int = 100  # MB
        BACKUP_COUNT: int = 5
        
        # Sensitive data masking
        MASK_SENSITIVE_DATA: bool = True
        MASKED_FIELDS: List[str] = ["api_key", "secret_key", "signature"]

    class Trading:
        """Trading configuration and defaults."""
        # Order defaults
        DEFAULT_ORDER_TYPE: str = "LIMIT"
        DEFAULT_TIME_IN_FORCE: str = "GTC"
        DEFAULT_RESPONSE_TYPE: str = "ACK"
        
        # Validation
        VALIDATE_ORDERS: bool = True
        VALIDATE_SYMBOLS: bool = True
        VALIDATE_BALANCES: bool = True
        
        # Risk management
        MAX_ORDER_QUANTITY: Optional[float] = None
        MAX_POSITION_SIZE: Optional[float] = None
        ENABLE_RISK_CHECKS: bool = True

    class MarketData:
        """Market data and caching configuration."""
        # Supported intervals
        SPOT_INTERVALS: List[str] = [
            "1m", "3m", "5m", "15m", "30m", "1h", "2h", "4h", "6h", "8h", "12h",
            "1d", "3d", "1w", "1M"
        ]
        
        FUTURES_INTERVALS: List[str] = [
            "1s", "1m", "3m", "5m", "15m", "30m", "1h", "2h", "4h", "6h", "8h", "12h",
            "1d", "3d", "1w", "1M"
        ]
        
        # Data caching
        CACHE_EXCHANGE_INFO: bool = True
        CACHE_TTL: int = 300  # seconds
        SYMBOL_INFO_CACHE_SIZE: int = 1000

    class ErrorHandling:
        """Error handling and recovery configuration."""
        # Exception mapping
        MAP_ERROR_CODES: bool = True
        INCLUDE_RESPONSE_IN_ERROR: bool = True
        
        # Specific error handling
        RETRY_ON_RATE_LIMIT: bool = True
        RETRY_ON_TIMEOUT: bool = True
        RETRY_ON_NETWORK_ERROR: bool = True
        
        # Error reporting
        REPORT_ERRORS: bool = True
        ERROR_REPORTING_WEBHOOK: Optional[str] = None

    class Performance:
        """Performance optimization settings."""
        # Caching
        ENABLE_CACHING: bool = True
        CACHE_BACKEND: str = "memory"  # memory, redis
        
        # Connection reuse
        REUSE_HTTP_SESSIONS: bool = True
        SESSION_CLEANUP_INTERVAL: int = 300  # seconds
        
        # Async optimization
        MAX_CONCURRENT_REQUESTS: int = 100
        USE_CONNECTION_POOLING: bool = True

    class Storage:
        """Storage and persistence configuration."""
        # API Key storage
        API_KEY_STORAGE_BACKEND: str = "memory"  # memory, database, redis
        ENCRYPT_API_KEYS: bool = True
        
        # Metrics storage
        STORE_METRICS: bool = False
        METRICS_RETENTION_DAYS: int = 30
        
        # Order history
        STORE_ORDER_HISTORY: bool = False
        ORDER_HISTORY_BACKEND: str = "memory"

    class Advanced:
        """Advanced features and experimental settings."""
        # WebSocket message validation
        VALIDATE_WS_MESSAGES: bool = True
        USE_PYDANTIC_VALIDATION: bool = True
        
        # Request signing
        ALTERNATIVE_SIGNING_METHODS: List[str] = ["HMAC_SHA256"]
        
        # Custom endpoints
        CUSTOM_ENDPOINTS: Dict[str, str] = {}
        
        # Feature flags
        ENABLE_EXPERIMENTAL_FEATURES: bool = False
        PREVIEW_API_SUPPORT: bool = False

    class Bot:
        """Bot-specific operational settings."""
        # Multi-user support
        MULTI_USER_ENABLED: bool = True
        USER_SESSION_TIMEOUT: int = 3600  # seconds
        
        # Bot identification
        BOT_NAME: str = "binance_bot"
        BOT_VERSION: str = "1.0.0"
        USER_AGENT: str = "BinancePythonBot/1.0"
        
        # Operation modes
        TRADING_ENABLED: bool = False
        READ_ONLY_MODE: bool = True
        DRY_RUN: bool = True


@dataclass
class AioGramConfig:
    """Aiogram 3.x configuration for Telegram bot."""
    API_TOKEN: str = field(default_factory=lambda: os.getenv("TELEGRAM_TOKEN", ""))
    PARSE_MODE: str = "HTML"
    USE_REDIS: bool = field(default_factory=lambda: os.getenv("USE_REDIS_FSM", "true").lower() == "true")
    REDIS_URL: Optional[str] = None
    
    # Redis configuration
    REDIS_HOST: str = field(default_factory=lambda: os.getenv("AIOGRAM_REDIS_HOST", "localhost"))
    REDIS_PORT: int = field(default_factory=lambda: int(os.getenv("AIOGRAM_REDIS_PORT", "6379")))
    REDIS_DB: int = field(default_factory=lambda: int(os.getenv("AIOGRAM_REDIS_DB", "0")))
    FSM_STORAGE_TTL: int = field(default_factory=lambda: int(os.getenv("FSM_STORAGE_TTL", "3600")))


@dataclass
class AnalysisConfig:
    """Technical analysis configuration."""
    ENABLED: bool = True
    DEFAULT_TIMEFRAME: str = "1h"
    MAX_LOOKBACK: int = 500


@dataclass
class OnChainConfig:
    """On-chain data configuration."""
    ENABLED: bool = False
    ETHERSCAN_API_KEY: Optional[str] = None
    BLOCKCHAIN_EXPLORER_URL: str = "https://etherscan.io"



@dataclass
class EncryptionConfig:
    """Encryption configuration with robust fallback handling."""
    
    def _get_master_key(self) -> str:
        """Get master key with comprehensive fallback options"""
        master_key = os.getenv("MASTER_KEY")
        
        # Primary key validation
        if master_key:
            try:
                # Validate key format by attempting to create Fernet
                Fernet(master_key.encode())
                logger.info("✅ MASTER_KEY validation successful")
                return master_key
            except (ValueError, InvalidKey) as e:
                logger.warning(f"⚠️ Invalid MASTER_KEY format: {e}, checking fallbacks")
        
        # Alternative environment variables
        alternative_sources = [
            ("ENCRYPTION_KEY", "Using ENCRYPTION_KEY as fallback"),
            ("FERNET_KEY", "Using FERNET_KEY as fallback"),
            ("APP_KEY", "Using APP_KEY as fallback"),
            ("SECRET_KEY", "Using SECRET_KEY as fallback")
        ]
        
        for env_var, message in alternative_sources:
            alt_key = os.getenv(env_var)
            if alt_key:
                try:
                    Fernet(alt_key.encode())
                    logger.info(f"✅ {message}")
                    return alt_key
                except (ValueError, InvalidKey):
                    continue
        
        # File-based fallback
        key_file_paths = [
            ".encryption_key",
            "config/encryption.key",
            "data/master.key"
        ]
        
        for key_path in key_file_paths:
            if os.path.exists(key_path):
                try:
                    with open(key_path, 'r') as f:
                        file_key = f.read().strip()
                    Fernet(file_key.encode())
                    logger.info(f"✅ Using encryption key from file: {key_path}")
                    return file_key
                except (ValueError, InvalidKey, IOError):
                    continue
        
        # Final emergency fallback
        logger.warning("🚨 No valid encryption key found in environment or files")
        
        # Generate persistent temporary key
        temp_key = Fernet.generate_key().decode()
        temp_key_path = "data/temp_encryption.key"
        
        try:
            os.makedirs(os.path.dirname(temp_key_path), exist_ok=True)
            with open(temp_key_path, 'w') as f:
                f.write(temp_key)
            logger.warning(f"🔑 Generated persistent temporary key at: {temp_key_path}")
            logger.warning("⚠️ WARNING: Using auto-generated key - NOT SECURE FOR PRODUCTION!")
            logger.warning("⚠️ Set MASTER_KEY environment variable for production use")
        except Exception as e:
            logger.error(f"❌ Failed to save temporary key: {e}")
        
        return temp_key

    @property
    def MASTER_KEY(self) -> str:
        """Get validated master key."""
        return self._get_master_key()

    @property
    def DATABASE_URL(self) -> str:
        """Get database URL with directory creation."""
        db_url = os.getenv("DATABASE_URL", "data/apikeys.db")
        db_dir = os.path.dirname(db_url)
        if db_dir and not os.path.exists(db_dir):
            os.makedirs(db_dir, exist_ok=True)
            logger.info(f"✅ Created database directory: {db_dir}")
        return db_url



class ApikeyManagerSettings(BaseSettings):
    BINANCE_API_KEY: str
    BINANCE_API_SECRET: str
    DEBUG: bool = False
    ENABLE_TRADING: bool = False
    LOG_LEVEL: str = "INFO"
    MAX_REQUESTS_PER_MINUTE: int = 1200
    PORT: int = 3000
    REQUEST_TIMEOUT: int = 30
    TELEGRAM_NAME: str
    TELEGRAM_TOKEN: str
    MASTER_KEY: str
    DATABASE_URL: str = "data/apikeys.db"
    USE_WEBHOOK: bool = False
    WEBHOOK_SECRET: str
    WEBHOOK_PATH: str = "/webhook"
    WEBAPP_HOST: str
    NGROK_URL: str = None
    WEBHOOK_HOST: str = None
    FAILURE_THRESHOLD: int = 5
    RESET_TIMEOUT: int = 30
    HALF_OPEN_TIMEOUT: int = 15
    SCAN_SYMBOLS: str
    GLASSNODE_API_KEY: str
    ONCHAIN_LOG_LEVEL: str = "INFO"
    ONCHAIN_CACHE_ENABLED: bool = False
    

    class Config:
        env_file = ".env"
        case_sensitive = False

@lru_cache()
def get_apikey_config() -> ApikeyManagerSettings:
    config = ApikeyManagerSettings()
    db_dir = os.path.dirname(config.DATABASE_URL)
    if db_dir and not os.path.exists(db_dir):
        os.makedirs(db_dir, exist_ok=True)
    return config



# config.py'ye main için olmalı 
class CacheConfig:
    USER_SESSION_TTL = 3600  # 1 saat
    MAX_CONCURRENT_USERS = 1000
    RATE_LIMIT_REQUESTS_PER_SECOND = 10


@dataclass
class BotConfig:
    """
    Main bot configuration class.
    
    This class follows singleton pattern and provides async-safe access
    to all configuration settings.
    """
    
    # ========================
    # 🤖 CORE BOT SETTINGS
    # ========================
    TELEGRAM_TOKEN: str = field(default_factory=lambda: os.getenv("TELEGRAM_TOKEN", ""))
    TELEGRAM_NAME: str = field(default_factory=lambda: os.getenv("TELEGRAM_NAME", ""))
    NGROK_URL: str = field(default_factory=lambda: os.getenv("NGROK_URL", "https://2fce5af7336f.ngrok-free.app"))
    
    DEFAULT_LOCALE: str = field(default_factory=lambda: os.getenv("DEFAULT_LOCALE", "en"))
 
    # Admin ID validation
    ADMIN_IDS: List[int] = field(default_factory=lambda: [
        int(x.strip()) for x in os.getenv("ADMIN_IDS", "").split(",") 
        if x.strip().isdigit() and int(x.strip()) > 0  # Pozitif ID kontrolü
    ])
    
    # ========================
    # 🔐 BINANCE API SETTINGS
    # ========================
    BINANCE_API_KEY: str = field(default_factory=lambda: os.getenv("BINANCE_API_KEY", ""))
    BINANCE_API_SECRET: str = field(default_factory=lambda: os.getenv("BINANCE_API_SECRET", ""))
    
    # ========================
    # 🌐 WEBHOOK SETTINGS
    # ========================
    USE_WEBHOOK: bool = field(default_factory=lambda: os.getenv("USE_WEBHOOK", "false").lower() == "true")
    WEBHOOK_HOST: str = field(default_factory=lambda: os.getenv("WEBHOOK_HOST", ""))
    WEBHOOK_SECRET: str = field(default_factory=lambda: os.getenv("WEBHOOK_SECRET", ""))
    WEBAPP_HOST: str = field(default_factory=lambda: os.getenv("WEBAPP_HOST", "0.0.0.0"))
    WEBAPP_PORT: int = field(default_factory=lambda: int(os.getenv("PORT", "3000")))
    


    
    # ========================
    # 🔐 ENCRYPTION SETTINGS
    # ========================
    ENCRYPTION: EncryptionConfig = field(default_factory=EncryptionConfig)
    
    
    # ========================
    # 📊 TRADING SETTINGS
    # ========================
    SCAN_SYMBOLS: List[str] = field(default_factory=lambda: [
        symbol.strip() for symbol in os.getenv(
            "SCAN_SYMBOLS", 
            "BTCUSDT,ETHUSDT,BNBUSDT,SOLUSDT,TRXUSDT,CAKEUSDT,SUIUSDT,PEPEUSDT,ARPAUSDT,TURBOUSDT"
        ).split(",") if symbol.strip()
    ])
    
    ENABLE_TRADING: bool = field(default_factory=lambda: os.getenv("ENABLE_TRADING", "false").lower() == "true")
    TRADING_STRATEGY: str = field(default_factory=lambda: os.getenv("TRADING_STRATEGY", "conservative"))
    MAX_LEVERAGE: int = field(default_factory=lambda: int(os.getenv("MAX_LEVERAGE", "3")))
    
    # Alert settings
    ALERT_PRICE_CHANGE_PERCENT: float = field(default_factory=lambda: float(os.getenv("ALERT_PRICE_CHANGE_PERCENT", "5.0")))
    ENABLE_PRICE_ALERTS: bool = field(default_factory=lambda: os.getenv("ENABLE_PRICE_ALERTS", "true").lower() == "true")
    ALERT_COOLDOWN: int = field(default_factory=lambda: int(os.getenv("ALERT_COOLDOWN", "300")))
    
    
    # other settings
    DEBUG: bool = field(default_factory=lambda: os.getenv("DEBUG", "false").lower() == "true")
    
    # ========================
    # 🏗️ COMPONENT CONFIGS
    # ========================
    BINANCE: BinanceConfig = field(default_factory=BinanceConfig)
    AIOGRAM: AioGramConfig = field(default_factory=AioGramConfig)
    ANALYSIS: AnalysisConfig = field(default_factory=AnalysisConfig)
    ONCHAIN: OnChainConfig = field(default_factory=OnChainConfig)

    @property
    def WEBHOOK_PATH(self) -> str:
        """
        Generate dynamic webhook path in Telegram format.
        
        Returns:
            str: Webhook path
        """
        if not self.TELEGRAM_TOKEN:
            return "/webhook/default"
        return f"/webhook/{self.TELEGRAM_TOKEN}"

    @property
    def WEBHOOK_URL(self) -> str:
        """
        Generate complete webhook URL.
        
        Returns:
            str: Full webhook URL
        """
        if not self.USE_WEBHOOK or not self.WEBHOOK_HOST:
            return ""
        return f"{self.WEBHOOK_HOST.rstrip('/')}{self.WEBHOOK_PATH}"




    @classmethod
    def load(cls) -> "BotConfig":
        """
        Load configuration from environment variables.
        
        Returns:
            BotConfig: Loaded configuration instance
        """
        return cls()

    def validate(self) -> bool:
        """
        Validate configuration values.
        
        Returns:
            bool: True if validation passes
            
        Raises:
            SystemExit: If validation fails with critical errors
        """
        errors = []
        
        # Telegram bot validation
        if not self.TELEGRAM_TOKEN:
            errors.append("❌ TELEGRAM_TOKEN is required")
        
        # Webhook validation
        if self.USE_WEBHOOK and not self.WEBHOOK_HOST:
            errors.append("❌ WEBHOOK_HOST is required when USE_WEBHOOK=true")
        
        # Binance validation for trading
        if self.ENABLE_TRADING:
            if not self.BINANCE_API_KEY:
                errors.append("❌ BINANCE_API_KEY is required when trading is enabled")
            if not self.BINANCE_API_SECRET:
                errors.append("❌ BINANCE_API_SECRET is required when trading is enabled")
        
        if errors:
            logger.critical("Configuration validation failed:\n%s", "\n".join(errors))
            sys.exit(1)
        
        return True

    #  Type safety için ek validation
    def validate_types(self) -> List[str]:
        """Type validation for configuration fields"""
        errors = []
        type_hints = get_type_hints(type(self))
        
        for field_name, expected_type in type_hints.items():
            value = getattr(self, field_name)
            if not isinstance(value, expected_type):
                errors.append(f"Field {field_name} should be {expected_type}, got {type(value)}")
        
        return errors


    def _validate_critical(self) -> List[str]:
        """Validate critical configuration values that would prevent bot from starting."""
        errors = []
        
        # Telegram bot validation
        if not self.TELEGRAM_TOKEN:
            errors.append("❌ TELEGRAM_TOKEN is required")
        
        # Webhook validation
        if self.USE_WEBHOOK and not self.WEBHOOK_HOST:
            errors.append("❌ WEBHOOK_HOST is required when USE_WEBHOOK=true")
        
        # Binance validation for trading
        if self.ENABLE_TRADING:
            if not self.BINANCE_API_KEY:
                errors.append("❌ BINANCE_API_KEY is required when trading is enabled")
            if not self.BINANCE_API_SECRET:
                errors.append("❌ BINANCE_API_SECRET is required when trading is enabled")
        
        return errors

    # Validasyon sistemini genişletmek için:
    def comprehensive_validate(self) -> Dict[str, List[str]]:
        """
        Kapsamlı validasyon sonuçlarını döndür.
        
        Returns:
            Dict[str, List[str]]: errors ve warnings ayrı ayrı
        """
        return {
            "errors": self._validate_critical(),
            "warnings": self.get_warnings(),
            "type_errors": self.validate_types()  # Yukarıda eklediğimiz
        }
 
    def get_warnings(self) -> List[str]:
        """
        Get configuration warnings (non-critical issues).
        
        Returns:
            List[str]: List of warning messages
        """
        warnings = []
        
        if self.ENABLE_TRADING and not self.USE_WEBHOOK:
            warnings.append("⚠️ Trading enabled but webhook disabled - consider using webhook for production")
            
        if self.DEBUG and self.ENABLE_TRADING:
            warnings.append("⚠️ Debug mode with trading enabled - be cautious with real funds")
        
        # Webhook URL validation warning
        if self.USE_WEBHOOK and self.WEBHOOK_URL and not self.WEBHOOK_URL.startswith(('https://', 'http://')):
            warnings.append("⚠️ Webhook URL might be malformed")
        
        # Testnet warning for production
        if self.ENABLE_TRADING and self.BINANCE.environment.ENV == Environment.PRODUCTION:
            warnings.append("⚠️ Trading enabled in PRODUCTION environment - real funds at risk!")
        
        return warnings
        
    
    def is_admin(self, user_id: int) -> bool:
        """
        Check if user is admin.
        
        Args:
            user_id: Telegram user ID to check
            
        Returns:
            bool: True if user is admin
        """
        return user_id in self.ADMIN_IDS

    def to_dict(self, include_sensitive: bool = False) -> Dict[str, Any]:
        """
        Convert configuration to dictionary for debugging/logging.
        
        Args:
            include_sensitive: Whether to include sensitive fields
            
        Returns:
            Dict[str, Any]: Configuration as dictionary
        """
        sensitive_fields = {
            "TELEGRAM_TOKEN", "BINANCE_API_KEY", "BINANCE_API_SECRET", 
            "WEBHOOK_SECRET"
        }
        result = {}
        
        for field_name in self.__dataclass_fields__:
            value = getattr(self, field_name)
            if field_name in sensitive_fields and value and not include_sensitive:
                result[field_name] = "***HIDDEN***"
            else:
                result[field_name] = value
        
        # Add properties
        result["WEBHOOK_PATH"] = self.WEBHOOK_PATH
        result["WEBHOOK_URL"] = self.WEBHOOK_URL
        
        return result

    def to_safe_dict(self) -> Dict[str, Any]:
        """
        Get safe configuration dictionary without sensitive data.
        
        Returns:
            Dict[str, Any]: Safe configuration dictionary
        """
        return self.to_dict(include_sensitive=False)


# ===============================
# 🎯 CONFIG MANAGEMENT FUNCTIONS
# ===============================

async def get_config() -> BotConfig:
    """
    Get global config instance (async wrapper).
    
    Returns:
        BotConfig: Global configuration instance
    """
    return get_config_sync()


def get_config_sync() -> BotConfig:
    """
    Get synchronized config instance.
    
    Returns:
        BotConfig: Global configuration instance
    """
    global _CONFIG_INSTANCE
    if _CONFIG_INSTANCE is None:
        _CONFIG_INSTANCE = BotConfig.load()
        _CONFIG_INSTANCE.validate()
        logger.info("✅ Bot configuration loaded and validated")
        
        # Debug logging with safe data only
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug("Configuration (safe): %s", _CONFIG_INSTANCE.to_safe_dict())
    
    return _CONFIG_INSTANCE


async def reload_config() -> BotConfig:
    """
    Reload configuration and clear cache.
    
    Returns:
        BotConfig: Reloaded configuration instance
    """
    global _CONFIG_INSTANCE
    _CONFIG_INSTANCE = None
    logger.info("🔄 Configuration cache cleared, reloading...")
    return await get_config()


def get_telegram_token() -> str:
    """
    Get Telegram bot token for Aiogram.
    
    Returns:
        str: Telegram bot token
    """
    config = get_config_sync()
    return config.TELEGRAM_TOKEN


def get_admins() -> List[int]:
    """
    Get admin user IDs.
    
    Returns:
        List[int]: List of admin user IDs
    """
    config = get_config_sync()
    return config.ADMIN_IDS


async def get_webhook_config() -> Dict[str, Any]:
    """
    Get webhook configuration.
    
    Returns:
        Dict[str, Any]: Webhook configuration dictionary
    """
    config = await get_config()
    return {
        "path": config.WEBHOOK_PATH,
        "url": config.WEBHOOK_URL,
        "secret_token": config.WEBHOOK_SECRET,
        "host": config.WEBAPP_HOST,
        "port": config.WEBAPP_PORT,
        "enabled": config.USE_WEBHOOK,
    }


async def get_redis_config() -> Dict[str, Any]:
    """
    Get Redis configuration for Aiogram.
    
    Returns:
        Dict[str, Any]: Redis configuration dictionary
    """
    config = await get_config()
    return {
        "host": config.AIOGRAM.REDIS_HOST,
        "port": config.AIOGRAM.REDIS_PORT,
        "db": config.AIOGRAM.REDIS_DB,
        "ttl": config.AIOGRAM.FSM_STORAGE_TTL,
    }





# ===============================
# 📦 MODULE INITIALIZATION
# ===============================
# Environment variable fallback iyileştirmesi
def __get_env_bool(key: str, default: bool = False) -> bool:  
    value = os.getenv(key, "").lower()
    if value in ("true", "1", "yes", "on"):
        return True
    elif value in ("false", "0", "no", "off"):
        return False
    return default





# Pre-load configuration for immediate access
config = get_config_sync()

"""
Bot Configuration Module

Centralized configuration management for the async trading bot.
Uses singleton pattern with async support and type hints.

# def validate Mevcut kullanım (geriye uyumlu)
config.validate()  # Kritik hatalarda exit

# def validate Yeni kullanım - uyarıları almak için
warnings = config.get_warnings()
for warning in warnings:
    logger.warning(warning)
"""
