# utils/apikey_manager.py
# v1011
# utils/apikey_manager.py
r"""
Tüm manager tamamen async yapıda
Sadece CPU-bound ve tek seferlik işlemler sync
"""
import os
import asyncio
import json
import logging

from enum import Enum
from dataclasses import dataclass
from dotenv import load_dotenv
from pathlib import Path
from typing import Optional, Tuple, List, Dict, Any, Union
from typing import TypeAlias

import aiosqlite
from cryptography.fernet import Fernet, InvalidToken
from cryptography.exceptions import InvalidKey
from binance import AsyncClient
from binance.exceptions import BinanceAPIException, BinanceRequestException

from config import config


UserID: TypeAlias = int  # UserID tip tanımı
GLOBAL_USER: int = 0     # Global kullanıcı sabiti

load_dotenv()  # <-- Bunu çağırmadan .env okunmaz
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class APITier(Enum):
    PUBLIC = "public"
    PRIVATE = "private" 
    SYSTEM = "system"

# ✅ ÖZEL HATA SINIFLARI
class APIKeyValidationError(Exception):
    """API key validation failed"""
    pass

class CredentialNotFoundError(Exception):
    """Credentials not found exception"""
    pass

class EncryptionError(Exception):
    """Encryption/decryption error"""
    pass

class EncryptionKeyError(Exception):
    """Encryption key related errors"""
    pass
    
# ============================================================
# BASE CLASS (DB + Encryption altyapısı)
# ============================================================

class BaseManager:
    _db_path: str = None
    _fernet: Fernet = None
    _init_lock: asyncio.Lock = asyncio.Lock()
    _db_connections: Dict[str, Any] = {}
    _db_initialized: bool = False
    _db_init_lock: asyncio.Lock = asyncio.Lock()
    _instance = None

    def __init__(self):
        cfg = config
        
        if not BaseManager._db_path:
            BaseManager._db_path = cfg.DATABASE_URL or "data/apikeys.db"

        if not BaseManager._fernet:
            BaseManager._fernet = self._initialize_encryption(cfg)
        
        # ✅ CONFIG'TEN GÜVENLİK AYARLARINI AL
        self._secure_permissions = getattr(cfg, 'SECURE_DB_PERMISSIONS', True)
        self._db_timeout = getattr(config, 'DB_CONNECTION_TIMEOUT', 30)

    #tablo başlıkları
      
    async def init_db(self) -> bool:
        """Initialize database with all required tables - DÜZELTİLMİŞ"""
        await self._ensure_db_exists()
        
        db = await self.get_db_connection()
        try:
            # ✅ ÖNCE users tablosu (diğerleri buna bağlı)
            await db.execute('''
                CREATE TABLE IF NOT EXISTS users (
                    user_id INTEGER PRIMARY KEY,
                    username TEXT,
                    first_name TEXT,
                    language_code TEXT DEFAULT 'en',
                    registered_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    last_active TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # ✅ SONRA apikeys (users'a foreign key OPSİYONEL)
            await db.execute('''
                CREATE TABLE IF NOT EXISTS apikeys (
                    user_id INTEGER NOT NULL,
                    exchange TEXT NOT NULL DEFAULT 'binance',
                    api_key_encrypted TEXT NOT NULL,
                    api_secret_encrypted TEXT NOT NULL,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    PRIMARY KEY (user_id, exchange)
                    -- FOREIGN KEY (user_id) REFERENCES users(user_id) ON DELETE CASCADE -- OPSİYONEL
                )
            ''')
            
            # ✅ Alarms tablosu
            await db.execute('''
                CREATE TABLE IF NOT EXISTS alarms (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id INTEGER NOT NULL,
                    symbol TEXT NOT NULL,
                    price REAL NOT NULL CHECK(price > 0),
                    condition TEXT NOT NULL CHECK(condition IN ('above', 'below')),
                    is_active BOOLEAN DEFAULT 1,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    triggered_at TIMESTAMP NULL,
                    notes TEXT
                    -- FOREIGN KEY (user_id) REFERENCES users(user_id) ON DELETE CASCADE -- OPSİYONEL
                )
            ''')
            
            # ✅ Trade Settings
            await db.execute('''
                CREATE TABLE IF NOT EXISTS trade_settings (
                    user_id INTEGER NOT NULL,
                    setting_key TEXT NOT NULL CHECK(setting_key IN ('risk_level', 'notifications', 'auto_trade')),
                    setting_value TEXT NOT NULL,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    PRIMARY KEY (user_id, setting_key)
                    CHECK(setting_key IN ('risk_level', 'notifications', 'auto_trade', 'language', 'timezone'))
                )
            ''')
            
            # ✅ Audit Log
            await db.execute('''
                CREATE TABLE IF NOT EXISTS audit_log (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id INTEGER NOT NULL,
                    action TEXT NOT NULL,
                    details TEXT,
                    ip_address TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # ✅ PERFORMANS INDEX'LERİ - BURAYA EKLE
            await self._create_indexes(db)
            
            await db.commit()
            logger.info("✅ All database tables and indexes initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"❌ Database initialization failed: {e}")
            await db.rollback()
            return False

    async def _create_indexes(self, db) -> None:
        """Create performance indexes"""
        indexes = [
            "CREATE INDEX IF NOT EXISTS idx_alarms_user_active ON alarms(user_id, is_active)",
            "CREATE INDEX IF NOT EXISTS idx_alarms_symbol ON alarms(symbol)",
            "CREATE INDEX IF NOT EXISTS idx_apikeys_user ON apikeys(user_id)",
            "CREATE INDEX IF NOT EXISTS idx_audit_user_time ON audit_log(user_id, created_at)",
            "CREATE INDEX IF NOT EXISTS idx_trade_settings_user ON trade_settings(user_id)",
            "CREATE INDEX IF NOT EXISTS idx_alarms_created_at ON alarms(created_at)",
            "CREATE INDEX IF NOT EXISTS idx_apikeys_updated ON apikeys(updated_at)"
        ]
        
        for index_sql in indexes:
            try:
                await db.execute(index_sql)
            except Exception as e:
                logger.warning(f"⚠️ Index creation failed for {index_sql}: {e}")

    async def ensure_db_initialized(self) -> bool:
        """Ensure database is initialized"""
        return await self.initialize_database()         


    # utils/apikey_manager.py'de BaseManager.initialize_database()'i düzelt
    @classmethod
    async def initialize_database(cls) -> bool:
        """Basitleştirilmiş database initialization"""
        if cls._db_initialized:
            return True
            
        try:
            # Database path kontrolü
            db_path = Path(cls._db_path or "data/apikeys.db")
            db_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Basit tablo oluşturma
            async with aiosqlite.connect(db_path) as db:
                await db.execute("PRAGMA journal_mode=WAL")
                await db.execute("PRAGMA foreign_keys=ON")
                
                # Sadece gerekli tabloları oluştur
                await db.execute('''
                    CREATE TABLE IF NOT EXISTS apikeys (
                        user_id INTEGER PRIMARY KEY,
                        api_key_encrypted TEXT NOT NULL,
                        api_secret_encrypted TEXT NOT NULL,
                        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                ''')
                
            cls._db_initialized = True
            logger.info("✅ Database initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"❌ Database initialization failed: {e}")
            return False
            



    @classmethod
    async def close_connections(cls) -> None:  # ✅ class method olmalı
        """Close all database connections"""
        for path, conn in cls._db_connections.items():
            try:
                await conn.close()
                logger.info(f"✅ Database connection closed: {path}")
            except Exception as e:
                logger.error(f"❌ Error closing connection {path}: {e}")
        cls._db_connections.clear()

    @classmethod
    async def cleanup_all(cls) -> None:
        """Tüm kaynakları temizle"""
        await cls.close_connections()
        
        cls._db_initialized = False
        cls._fernet = None
        logger.info("✅ All BaseManager resources cleaned up")


    @staticmethod
    def _initialize_encryption(config) -> Fernet:
        """
        Initialize and validate encryption using MASTER_KEY.
        Falls back to temporary key only in dev environments.
        """
        master_key = config.MASTER_KEY

        if not master_key:
            logger.critical("❌ MASTER_KEY not defined in environment (.env)")
            raise RuntimeError("❌ MASTER_KEY tanımlı olmalı (.env)!")

        if len(master_key) < 32:
            raise ValueError("❌ MASTER_KEY too short — must be 32+ characters")

        try:
            fernet = Fernet(master_key.encode())
            
            # Test encryption-decryption
            test_data = b"test_encryption"
            encrypted = fernet.encrypt(test_data)
            decrypted = fernet.decrypt(encrypted)

            if decrypted != test_data:
                raise RuntimeError("Encryption test failed: decrypted data mismatch")

            logger.info("✅ Fernet encryption initialized successfully")
            return fernet

        except (InvalidToken, InvalidKey, ValueError, RuntimeError) as e:
            logger.critical(f"❌ Encryption initialization failed: {e}")
            raise

    @property
    def db_path(self) -> str:
        return BaseManager._db_path

    @property
    def fernet(self) -> Fernet:
        return BaseManager._fernet

    def _encrypt(self, data: str) -> str:
        """Encrypt sensitive data"""
        if not data or not isinstance(data, str):
            raise ValueError("Encryption data must be non-empty string")
        return self.fernet.encrypt(data.encode()).decode()

    def _decrypt(self, data: str) -> str:
        """Decrypt sensitive data"""
        if not data or not isinstance(data, str):
            raise ValueError("Decryption data must be non-empty string")
        try:
            return self.fernet.decrypt(data.encode()).decode()
        except InvalidToken:
            logger.error("❌ Invalid token during decryption - possible key mismatch")
            raise
        except Exception as e:
            logger.error(f"❌ Decryption failed: {e}")
            raise



    async def _ensure_db_exists(self) -> None:
        """Database directory and file exist with config-based permissions"""
        db_path = Path(self.db_path)
        db_dir = db_path.parent

        if not db_dir.exists():
            db_dir.mkdir(parents=True, exist_ok=True)
            # ✅ CONFIG'E GÖRE İZİN AYARLA
            permissions = 0o700 if self._secure_permissions else 0o755
            os.chmod(db_dir, permissions)
            logger.info(f"📁 Database directory created: {db_dir}")

        if not db_path.exists():
            db_path.touch()
            # ✅ CONFIG'E GÖRE DOSYA İZİNLERİ
            file_permissions = 0o600 if self._secure_permissions else 0o644
            os.chmod(db_path, file_permissions)
            logger.info(f"🆕 Database file created: {db_path}")
            

    @classmethod
    async def get_db_connection(cls) -> aiosqlite.Connection:
        """Basitleştirilmiş connection management"""
        if not cls._db_initialized:
            await cls.initialize_database()
        
        db_path = cls._db_path
        if db_path not in cls._db_connections:
            cls._db_connections[db_path] = await aiosqlite.connect(
                db_path, 
                check_same_thread=False,
                timeout=30.0
            )
            # WAL mode ve foreign keys
            await cls._db_connections[db_path].execute("PRAGMA journal_mode=WAL")
            await cls._db_connections[db_path].execute("PRAGMA foreign_keys=ON")
        
        return cls._db_connections[db_path]
        

# ============================================================
# API KEY MANAGER
# ============================================================

@dataclass
class UserCredentials:
    user_id: UserID
    api_key: str
    api_secret: str
    tier: APITier
    is_valid: bool = False


class APIKeyManager(BaseManager):
    _instance: Optional["APIKeyManager"] = None
    _initialized: bool = False  # ✅ Initialization flag ekle 25/10
    _validation_lock: asyncio.Lock = asyncio.Lock()
    _cache: Dict[int, Tuple[str, str]] = {}
    _cache_lock: asyncio.Lock = asyncio.Lock()

    def __init__(self):
        super().__init__()
        # Trading flag
        self.enable_trading: bool = getattr(config, "ENABLE_TRADING", False)
    

    @classmethod
    async def get_instance(cls) -> "APIKeyManager":
        if cls._instance is None:
            cls._instance = cls()
            await cls._instance._async_init()  # ✅ Async initialization
        return cls._instance

    async def _async_init(self):
        if not self._initialized:
            await self.ensure_db_initialized()
            self._initialized = True
            

    # APIKeyManager sınıfına özel cleanup:
    async def cleanup(self) -> None:
        """APIKeyManager özel cleanup"""
        async with self._cache_lock:
            self._cache.clear()
        logger.info("✅ APIKeyManager cache cleared")

# ---Global API Key .env içeriğine göre -----

    """async def get_global_apikey(self) -> Optional[Tuple[str, str]]:
        try:
            api_key = os.getenv("BINANCE_API_KEY")
            secret_key = os.getenv("BINANCE_API_SECRET")
            
            if api_key and secret_key:
                if len(api_key) >= 16 and len(secret_key) >= 32:
                    logger.info("Global API credentials retrieved", extra={
                        "user_id": 0,  # Global user
                        "source": "environment",
                        "tier": "system"
                    })
                    return api_key, secret_key
            
            logger.warning("Global API credentials not found or invalid", extra={
                "user_id": 0,
                "source": "environment",
                "tier": "system",
                "api_key_length": len(api_key) if api_key else 0,
                "secret_key_length": len(secret_key) if secret_key else 0
            })
            return None
        except Exception as e:
            logger.error("Failed to get global API keys", extra={
                "user_id": 0,
                "source": "environment", 
                "tier": "system",
                "error": str(e)
            })
            return None
   """         
    

    async def get_global_apikey(self) -> Optional[Tuple[str, str]]:
        if not getattr(self, "enable_trading", False):
            logger.info("Trading is disabled - global API key not returned")
            return None

        try:
            api_key = os.getenv("BINANCE_API_KEY")
            secret_key = os.getenv("BINANCE_API_SECRET")
            
            if api_key and secret_key:
                if len(api_key) >= 16 and len(secret_key) >= 32:
                    logger.info("Global API credentials retrieved", extra={
                        "user_id": 0,  # Global user
                        "source": "environment",
                        "tier": "system"
                    })
                    return api_key, secret_key
            
            logger.warning("Global API credentials not found or invalid", extra={
                "user_id": 0,
                "source": "environment",
                "tier": "system",
                "api_key_length": len(api_key) if api_key else 0,
                "secret_key_length": len(secret_key) if secret_key else 0
            })
            return None
        except Exception as e:
            logger.error("Failed to get global API keys", extra={
                "user_id": 0,
                "source": "environment", 
                "tier": "system",
                "error": str(e)
            })
            return None

    
    
    async def validate_global_credentials(self) -> bool:
        """Validate global .env credentials"""
        try:
            creds = await self.get_global_apikey()
            if not creds:
                return False
                
            api_key, secret_key = creds
            client = await AsyncClient.create(api_key, secret_key)
            
            try:
                account_info = await client.get_account()
                if account_info and 'canTrade' in account_info:
                    logger.info("✅ Global API credentials validated")
                    return True
                return False
            finally:
                await client.close_connection()
                
        except Exception as e:
            logger.error(f"❌ Global credentials validation failed: {e}")
            return False

# ----------kişisel api+secret--------

    async def add_or_update_apikey(self, user_id: int, api_key: str, secret_key: str) -> None:
        if not all([user_id, api_key, secret_key]):
            logger.error("Validation failed for API key addition", extra={
                "user_id": user_id,
                "api_key_provided": bool(api_key),
                "secret_key_provided": bool(secret_key)
            })
            raise ValueError("User ID, API key and secret key are required")

        # Ensure database is initialized
        #await self.init_db()
        # ✅ Sadece ilk seferde init_db çağır
        await self.ensure_db_initialized()

        # ✅ DÜZELTME: Tek bir encrypted field yerine ayrı ayrı encrypt et
        encrypted_api_key = self._encrypt(api_key)
        encrypted_secret_key = self._encrypt(secret_key)
        
        db = await self.get_db_connection()
        
        try:
            async with db.cursor() as cursor:
                await cursor.execute(
                    """INSERT INTO apikeys (user_id, exchange, api_key_encrypted, api_secret_encrypted, updated_at)
                       VALUES (?, 'binance', ?, ?, CURRENT_TIMESTAMP)
                       ON CONFLICT(user_id, exchange) DO UPDATE SET
                       api_key_encrypted=excluded.api_key_encrypted,
                       api_secret_encrypted=excluded.api_secret_encrypted,
                       updated_at=CURRENT_TIMESTAMP""",
                    (user_id, encrypted_api_key, encrypted_secret_key)
                )

                
                await db.commit()
                
                # Update cache
                async with self._cache_lock:
                    self._cache[user_id] = (api_key, secret_key)
                
                logger.info("API credentials added/updated successfully", extra={
                    "user_id": user_id,
                    "source": "database",
                    "tier": "private",
                    "action": "upsert"
                })
        except Exception as e:
            await db.rollback()
            logger.exception(f"❌ DB update failed for user {user_id}")
            raise

    async def get_apikey(self, user_id: int) -> Optional[Tuple[str, str]]:
        # Check cache first
        async with self._cache_lock:
            if user_id in self._cache:
                logger.info("API credentials retrieved from cache", extra={
                    "user_id": user_id,
                    "source": "cache",
                    "tier": "private"
                })
                return self._cache[user_id]
        
        # Query database
        db = await self.get_db_connection()
        try:
            async with db.execute(
                "SELECT api_key_encrypted, api_secret_encrypted FROM apikeys WHERE user_id=?", 
                (user_id,)
            ) as cursor:
                row = await cursor.fetchone()
                if row:
                    api_key = self._decrypt(row[0])
                    secret_key = self._decrypt(row[1])
                    
                    # Update cache
                    async with self._cache_lock:
                        self._cache[user_id] = (api_key, secret_key)
                    
                    logger.info("API credentials retrieved from database", extra={
                        "user_id": user_id,
                        "source": "database",
                        "tier": "private"
                    })
                    return api_key, secret_key
                
                logger.warning("API credentials not found", extra={
                    "user_id": user_id,
                    "source": "none",
                    "tier": "private"
                })
                return None
                    
        except Exception as e:
            logger.error(f"❌ Error retrieving API key for user {user_id}: {e}")
            return None
                   
      
    """async def validate_binance_credentials(self, user_id: int) -> bool:

        async with self._validation_lock:
            try:
                creds = await self.get_apikey(user_id)
                if not creds:
                    logger.warning(f"⚠️ No credentials found for user {user_id}")
                    return False
                    
                api_key, secret_key = creds
                
                # Basic validation
                if not api_key.startswith(('api_key_', 'binance_')) and len(api_key) < 16:
                    logger.warning(f"⚠️ Invalid API key format for user {user_id}")
                    return False
                
                client = await AsyncClient.create(api_key, secret_key)
                
                try:
                    # Test API connectivity with account info
                    account_info = await client.get_account()
                    if account_info and 'canTrade' in account_info:
                        logger.info(f"✅ Binance credentials validated for user {user_id}")
                        return True
                    return False
                    
                except BinanceAPIException as e:
                    logger.error(f"❌ Binance API error for user {user_id}: {e.code} - {e.message}")
                    return False
                except BinanceRequestException as e:
                    logger.error(f"❌ Binance request error for user {user_id}: {e}")
                    return False
                except Exception as e:
                    logger.error(f"❌ Unexpected Binance error for user {user_id}: {e}")
                    return False
                finally:
                    await client.close_connection()
                    
            except Exception as e:
                logger.error(f"❌ Validation process failed for user {user_id}: {e}")
                return False
    """

    async def validate_binance_credentials(self, user_id: int) -> bool:
        """Validate Binance credentials with proper error handling"""
        if not getattr(self, "enable_trading", False):
            logger.warning(f"Trading is disabled - skipping Binance validation for user {user_id}")
            return False

        async with self._validation_lock:
            try:
                creds = await self.get_apikey(user_id)
                if not creds:
                    logger.warning(f"⚠️ No credentials found for user {user_id}")
                    return False
                    
                api_key, secret_key = creds
                
                # Basic validation
                if not api_key.startswith(('api_key_', 'binance_')) and len(api_key) < 16:
                    logger.warning(f"⚠️ Invalid API key format for user {user_id}")
                    return False
                
                client = await AsyncClient.create(api_key, secret_key)
                
                try:
                    account_info = await client.get_account()
                    if account_info and 'canTrade' in account_info:
                        logger.info(f"✅ Binance credentials validated for user {user_id}")
                        return True
                    return False
                    
                except BinanceAPIException as e:
                    logger.error(f"❌ Binance API error for user {user_id}: {e.code} - {e.message}")
                    return False
                except BinanceRequestException as e:
                    logger.error(f"❌ Binance request error for user {user_id}: {e}")
                    return False
                except Exception as e:
                    logger.error(f"❌ Unexpected Binance error for user {user_id}: {e}")
                    return False
                finally:
                    await client.close_connection()
                    
            except Exception as e:
                logger.error(f"❌ Validation process failed for user {user_id}: {e}")
                return False



    async def rotate_keys(self, user_id: int, new_api_key: str, new_secret: str) -> bool:
        """Rotate API keys with validation"""
        try:
            await self.add_or_update_apikey(user_id, new_api_key, new_secret)
            return await self.validate_binance_credentials(user_id)
        except Exception as e:
            logger.error(f"❌ Key rotation failed for user {user_id}: {e}")
            return False

    async def cleanup_cache(self, max_size: int = 1000) -> None:
        """Cleanup cache to prevent memory leaks"""
        async with self._cache_lock:
            if len(self._cache) > max_size:
                # Remove oldest entries (simple FIFO)
                keys_to_remove = list(self._cache.keys())[:len(self._cache) - max_size]
                for key in keys_to_remove:
                    del self._cache[key]
                logger.info(f"🧹 Cache cleaned up, removed {len(keys_to_remove)} entries")

    async def delete_apikey(self, user_id: int) -> bool:
        """Delete API key for user"""
        db = await self.get_db_connection()
        try:
            async with db.cursor() as cursor:
                await cursor.execute("DELETE FROM apikeys WHERE user_id=?", (user_id,))
                await db.commit()
                
                # Remove from cache
                async with self._cache_lock:
                    self._cache.pop(user_id, None)
                
                logger.info(f"🗑️ API key deleted for user {user_id}")
                return True
        except Exception as e:
            await db.rollback()
            logger.error(f"❌ Failed to delete API key for user {user_id}: {e}")
            return False

    async def get_database_info(self) -> Dict[str, Any]:
        """Database bilgilerini getir"""
        try:
            db_path = Path(self.db_path)
            info = {
                "exists": db_path.exists(),
                "size": db_path.stat().st_size if db_path.exists() else 0,
                "cache_size": len(self._cache)
            }
            
            if db_path.exists():
                async with aiosqlite.connect(self.db_path) as db:
                    # Tablo sayıları
                    for table in ['apikeys', 'alarms', 'trade_settings']:
                        cursor = await db.execute(f"SELECT COUNT(*) FROM {table}")
                        info[f"{table}_count"] = (await cursor.fetchone())[0]
                        
            return info
        except Exception as e:
            return {"error": str(e)}
            

# ============================================================
# ALARM MANAGER
# ============================================================

class AlarmManager(BaseManager):
    _instance: Optional["AlarmManager"] = None
    _initialized: bool = False
    _cache: Dict[int, Dict[str, Any]] = {}
    _cache_lock: asyncio.Lock = asyncio.Lock()

    def __init__(self):
        super().__init__()

    @classmethod
    async def get_instance(cls) -> "AlarmManager":  # ✅ Async yap
        if cls._instance is None:
            cls._instance = AlarmManager()
            await cls._instance._async_init()
        return cls._instance

    async def _async_init(self):
        if not self._initialized:
            await self.ensure_db_initialized()
            self._initialized = True


    async def set_alarm_settings(self, user_id: int, settings: dict) -> int:
        """Alarm ekle ve alarm_id döndür"""
        db = await self.get_db_connection()
        
        try:
            cursor = await db.execute(
                """INSERT INTO alarms (user_id, symbol, price, condition) 
                   VALUES (?, ?, ?, ?)""",
                (user_id, settings.get('symbol'), settings.get('price'), settings.get('condition'))
            )
            await db.commit()
            alarm_id = cursor.lastrowid
            logger.info(f"🔔 Alarm eklendi: user_id={user_id}, alarm_id={alarm_id}")
            return alarm_id
        except Exception as e:
            await db.rollback()
            logger.error(f"❌ Alarm eklenemedi: {e}")
            raise


    async def add_alarm(self, user_id: int, symbol: str, price: float, condition: str) -> int:
        """Alarms tablosuna uygun alarm ekle"""
        db = await self.get_db_connection()
        try:
            cursor = await db.execute(
                "INSERT INTO alarms (user_id, symbol, price, condition) VALUES (?, ?, ?, ?)",
                (user_id, symbol, price, condition)
            )
            await db.commit()
            return cursor.lastrowid
        except Exception as e:
            await db.rollback()
            raise

    async def get_user_alarms(self, user_id: int, active_only: bool = True) -> List[dict]:
        """Kullanıcının alarmlarını getir - TABLO YAPISINA UYGUN"""
        db = await self.get_db_connection()
        try:
            if active_only:
                query = "SELECT * FROM alarms WHERE user_id=? AND is_active=1 ORDER BY created_at DESC"
            else:
                query = "SELECT * FROM alarms WHERE user_id=? ORDER BY created_at DESC"
                
            async with db.execute(query, (user_id,)) as cursor:
                results = []
                async for row in cursor:
                    # ✅ Tablo yapısına uygun dict oluştur
                    alarm_dict = {
                        'id': row[0],
                        'user_id': row[1],
                        'symbol': row[2],
                        'price': row[3],
                        'condition': row[4],
                        'is_active': bool(row[5]),
                        'created_at': row[6],
                        'triggered_at': row[7]
                    }
                    results.append(alarm_dict)
                return results
        except Exception as e:
            logger.error(f"❌ Alarm getirme hatası: {e}")
            return []
            

    async def delete_alarm(self, alarm_id: int) -> bool:
        """Delete specific alarm by ID"""
        db = await self.get_db_connection()
        try:
            async with db.cursor() as cursor:
                result = await cursor.execute("DELETE FROM alarms WHERE id=?", (alarm_id,))
                await db.commit()
                deleted = result.rowcount > 0
                if deleted:
                    logger.info(f"🗑️ Alarm deleted: {alarm_id}")
                return deleted
        except Exception as e:
            await db.rollback()
            logger.error(f"❌ Failed to delete alarm {alarm_id}: {e}")
            return False

    async def cleanup_old_alarms(self, days: int = 30) -> int:
        """Cleanup old alarms and return count of deleted records"""
        db = await self.get_db_connection()
        try:
            async with db.cursor() as cursor:
                result = await cursor.execute(
                    "DELETE FROM alarms WHERE created_at < datetime('now', ?)", 
                    (f'-{days} days',)
                )
                await db.commit()
                deleted_count = result.rowcount
                logger.info(f"🧹 Cleaned up {deleted_count} old alarms (older than {days} days)")
                return deleted_count
        except Exception as e:
            await db.rollback()
            logger.error(f"❌ Failed to cleanup old alarms: {e}")
            return 0

# ============================================================
# TRADE SETTINGS MANAGER  
# ============================================================

class TradeSettingsManager(BaseManager):
    _instance: Optional["TradeSettingsManager"] = None
    _initialized: bool = False
    _cache: Dict[int, Dict[str, Any]] = {}
    _cache_lock: asyncio.Lock = asyncio.Lock()

    def __init__(self):
        super().__init__()

    @classmethod
    async def get_instance(cls) -> "TradeSettingsManager":  # ✅ Async yapı
        if cls._instance is None:
            cls._instance = TradeSettingsManager()
            await cls._instance._async_init()
        return cls._instance

    async def _async_init(self):
        if not self._initialized:
            await self.ensure_db_initialized()
            self._initialized = True
            
    async def cleanup_old_apikeys(self, days: int = 90) -> int:
        """Cleanup old API keys and return count of deleted records"""
        db = await self.get_db_connection()
        try:
            async with db.cursor() as cursor:
                result = await cursor.execute(
                    "DELETE FROM apikeys WHERE updated_at < datetime('now', ?)", 
                    (f'-{days} days',)
                )
                await db.commit()
                deleted_count = result.rowcount
                
                # Cleanup cache for deleted users
                async with self._cache_lock:
                    self._cache.clear()
                
                logger.info(f"🧹 Cleaned up {deleted_count} old API keys (older than {days} days)")
                return deleted_count
        except Exception as e:
            await db.rollback()
            logger.error(f"❌ Failed to cleanup old API keys: {e}")
            return 0

    async def cleanup_cache(self) -> None:
        """Cleanup trade settings cache"""
        async with self._cache_lock:
            self._cache.clear()
            logger.info("🧹 Trade settings cache cleaned up")

# ============================================================

"""
🔹 
- Alarm işlemleri sırasında Telegram kullanıcısının user_id’si kullanılır.
- api_key ve secret_key sadece private (borsa) işlemler gerektiğinde devreye girer
| Durum            | Kullanılan bilgi                  | Açıklama                                   |
| ---------------- | --------------------------------- | ------------------------------------------ |
| Alarm kurarken   | `user_id`                         | Alarm sadece Telegram kimliğine bağlı      |
| Public işlemler  | `user_id`                         | herkes, API anahtarı gerekmez                      |
| Private işlemler | `user_id + (api_key, secret_key)` | kişi, Binance’e bağlantı için anahtarlar gerekir |

🧠 Mantıksal akış:
Kullanıcı Telegram ID → sistemin merkezinde
APIKeyManager → Her kullanıcıya özel anahtar saklar (user_id bazlı)
AlarmManager → Her alarmı user_id ile ilişkilendirir
Binance işlemleri:
user_id → APIKeyManager.get_user_credentials() → (api_key, secret)
Binance’e bağlan, işlemleri o kullanıcı adına yap





api manager şunu yapıyor mu
* .env deki api+secret yapısını kullanarak public ve private analiz sorgu yapar
(BINANCE_API_KEY=m***b  ,BINANCE_API_SE=e***r)

* kişisel api eklenirse analiz işlemleri + kişisel cüzdan işlemleri + trade işlemleri yüklenen bilgiye gmre yapılır


📊 TABLO SÜTUN ANALİZİ
Alarms Tablosu: 8 sütun - tam tanımlı
Trade Settings: 4 sütun - constraint'lerle
Users Tablosu: 6 sütun - kapsamlı
Audit Log: 5 sütun - detaylı logging

Alarms Tablosu - 8 sütun:
1. id (PK) - Alarm ID
2. user_id - Kullanıcı ID
3. symbol - Sembol (BTCUSDT)
4. price - Alarm fiyatı
5. condition - Koşul (above/below)
6. is_active - Aktif/pasif
7. created_at - Oluşturulma
8. triggered_at - Tetiklenme
Trade Settings Tablosu - 4 sütun:
1. user_id - Kullanıcı ID
2. setting_key - Ayar anahtarı
3. setting_value - Ayar değeri
4. updated_at - Güncelleme

"""