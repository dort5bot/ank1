# utils/apikey_manager.py
# v1011

import asyncio
import json
import logging
from pathlib import Path
from typing import Optional, Tuple, List, Dict, Any

import aiosqlite
from cryptography.fernet import Fernet, InvalidToken
from cryptography.exceptions import InvalidKey
from binance import AsyncClient
from binance.exceptions import BinanceAPIException, BinanceRequestException

from config import get_apikey_config

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# ============================================================
# BASE CLASS (DB + Encryption altyapısı)
# ============================================================

class BaseManager:
    _db_path: str = None
    _fernet: Fernet = None
    _init_lock: asyncio.Lock = asyncio.Lock()
    _db_connections: Dict[str, Any] = {}
    _instance = None

    def __init__(self):
        config = get_apikey_config()

        if not BaseManager._db_path:
            BaseManager._db_path = config.DATABASE_URL or "data/apikeys.db"

        if not BaseManager._fernet:
            BaseManager._fernet = self._initialize_encryption(config)

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
        """Ensure database directory and file exist"""
        db_path = Path(self.db_path)
        db_dir = db_path.parent

        if not db_dir.exists():
            db_dir.mkdir(parents=True, exist_ok=True)
            os.chmod(db_dir, 0o777)  # ✅ izin ver db yazma izni
            logger.info(f"📁 Database directory created: {db_dir}")

        if not db_path.exists():
            db_path.touch()
            logger.info(f"🆕 Database file created: {db_path}")

    async def get_db_connection(self) -> aiosqlite.Connection:
        """Get database connection with connection pool"""
        async with BaseManager._init_lock:
            # ✅ Ensure DB directory and file exist
            await self._ensure_db_exists()

            # ✅ Create or reuse connection pool
            if self.db_path not in BaseManager._db_connections:
                BaseManager._db_connections[self.db_path] = await aiosqlite.connect(
                    self.db_path,
                    check_same_thread=False,
                    timeout=30.0
                )
                # Enable WAL mode and foreign keys for better performance and integrity
                db_conn = BaseManager._db_connections[self.db_path]
                await db_conn.execute("PRAGMA journal_mode=WAL")
                await db_conn.execute("PRAGMA foreign_keys = ON")
                await db_conn.commit()
                logger.info(f"✅ Database connection pool created for {self.db_path}")
            
            return BaseManager._db_connections[self.db_path]


    # utils/apikey_manager.py dosyasında init_db metodunu bul ve şu şekilde güncelle:

         

    async def init_db(self) -> bool:
        """Veritabanı tablolarını kesin olarak oluştur - YORUMLAR KALDIRILDI"""
        try:
            logger.info("🔄 Creating database tables...")
            
            async with aiosqlite.connect(self.db_path) as db:
                # API Keys tablosu - YORUMLAR KALDIRILDI
                await db.execute('''
                    CREATE TABLE IF NOT EXISTS apikeys (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        user_id INTEGER NOT NULL,
                        exchange TEXT NOT NULL DEFAULT 'binance',
                        api_key_encrypted TEXT NOT NULL,
                        api_secret_encrypted TEXT NOT NULL,
                        is_active BOOLEAN DEFAULT 1,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        UNIQUE(user_id, exchange)
                    )
                ''')
                
                # Alarms tablosu - YORUMLAR KALDIRILDI
                await db.execute('''
                    CREATE TABLE IF NOT EXISTS alarms (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        user_id INTEGER NOT NULL,
                        symbol TEXT NOT NULL,
                        price REAL NOT NULL,
                        condition TEXT NOT NULL,
                        is_active BOOLEAN DEFAULT 1,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        triggered_at TIMESTAMP NULL
                    )
                ''')
                
                # Trade Settings tablosu - YORUMLAR KALDIRILDI
                await db.execute('''
                    CREATE TABLE IF NOT EXISTS trade_settings (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        user_id INTEGER NOT NULL,
                        symbol TEXT NOT NULL,
                        leverage INTEGER DEFAULT 1,
                        risk_percentage REAL DEFAULT 2.0,
                        is_active BOOLEAN DEFAULT 1,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        UNIQUE(user_id, symbol)
                    )
                ''')
                
                await db.commit()
                logger.info("✅ All database tables created successfully")
                
                # Tabloların oluştuğunu doğrula
                cursor = await db.execute("SELECT name FROM sqlite_master WHERE type='table'")
                tables = await cursor.fetchall()
                table_names = [table[0] for table in tables]
                logger.info(f"📊 Available tables: {table_names}")
                
                return True
                
        except Exception as e:
            logger.error(f"❌ Database table creation failed: {e}")
            return False



    async def close_connections(self) -> None:
        """Close all database connections"""
        for path, conn in BaseManager._db_connections.items():
            try:
                await conn.close()
                logger.info(f"✅ Database connection closed: {path}")
            except Exception as e:
                logger.error(f"❌ Error closing connection {path}: {e}")
        BaseManager._db_connections.clear()

# ============================================================
# API KEY MANAGER
# ============================================================

class APIKeyManager(BaseManager):
    _instance: Optional["APIKeyManager"] = None
    _validation_lock: asyncio.Lock = asyncio.Lock()
    _cache: Dict[int, Tuple[str, str]] = {}
    _cache_lock: asyncio.Lock = asyncio.Lock()

    def __init__(self):
        super().__init__()

    @classmethod
    def get_instance(cls) -> "APIKeyManager":
        if cls._instance is None:
            cls._instance = APIKeyManager()
        return cls._instance


    async def add_or_update_apikey(self, user_id: int, api_key: str, secret_key: str) -> None:
        """Add or update API key with validation and encryption - DÜZELTİLDİ"""
        if not all([user_id, api_key, secret_key]):
            raise ValueError("User ID, API key and secret key are required")
        
        if len(api_key) < 10 or len(secret_key) < 10:
            raise ValueError("API key and secret key are too short")

        # Ensure database is initialized
        await self.init_db()

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
                
                logger.info(f"🔐 API key updated for user {user_id}")
        except Exception as e:
            await db.rollback()
            logger.exception(f"❌ DB update failed for user {user_id}")
            raise

    async def get_apikey(self, user_id: int) -> Optional[Tuple[str, str]]:
        """Get API key from cache or database - DÜZELTİLDİ"""
        # Check cache first
        async with self._cache_lock:
            if user_id in self._cache:
                return self._cache[user_id]
        
        # Query database
        db = await self.get_db_connection()
        try:
            # ✅ DÜZELTME: Sütun isimleri güncellendi
            async with db.execute(
                "SELECT api_key_encrypted, api_secret_encrypted FROM apikeys WHERE user_id=?", 
                (user_id,)
            ) as cursor:
                row = await cursor.fetchone()
                if row:
                    # ✅ DÜZELTME: Ayrı ayrı decrypt et
                    api_key = self._decrypt(row[0])
                    secret_key = self._decrypt(row[1])
                    
                    # Update cache
                    async with self._cache_lock:
                        self._cache[user_id] = (api_key, secret_key)
                    
                    return api_key, secret_key
                return None
        except Exception as e:
            logger.error(f"❌ Failed to get API key for user {user_id}: {e}")
            return None
            


    async def validate_binance_credentials(self, user_id: int) -> bool:
        """Validate Binance credentials with proper error handling"""
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

    async def check_database_status(self) -> Dict[str, Any]:
        """Check database status and return information"""
        try:
            await self._ensure_db_exists()
            db_path = Path(self.db_path)
            
            status = {
                "database_exists": db_path.exists(),
                "database_path": str(db_path),
                "database_size": db_path.stat().st_size if db_path.exists() else 0,
                "cache_size": len(self._cache)
            }
            
            if db_path.exists():
                db = await self.get_db_connection()
                try:
                    # Check table existence and row counts
                    async with db.execute("SELECT COUNT(*) FROM apikeys") as cursor:
                        status["apikeys_count"] = (await cursor.fetchone())[0]
                    
                    async with db.execute("SELECT COUNT(*) FROM alarms") as cursor:
                        status["alarms_count"] = (await cursor.fetchone())[0]
                        
                except Exception as e:
                    logger.warning(f"⚠️ Could not get table counts: {e}")
                    status["apikeys_count"] = 0
                    status["alarms_count"] = 0
            
            return status
        except Exception as e:
            logger.error(f"❌ Database status check failed: {e}")
            return {"error": str(e)}

    async def verify_database_schema(self) -> bool:
        """Database şema uyumluluğunu kontrol et"""
        try:
            async with aiosqlite.connect(self.db_path) as db:
                # Tablo yapılarını kontrol et
                cursor = await db.execute("PRAGMA table_info(apikeys)")
                columns = await cursor.fetchall()
                column_names = [col[1] for col in columns]
                
                logger.info(f"📋 apikeys table columns: {column_names}")
                
                # Gerekli sütunları kontrol et
                required_columns = ['api_key_encrypted', 'api_secret_encrypted']
                missing_columns = [col for col in required_columns if col not in column_names]
                
                if missing_columns:
                    logger.error(f"❌ Missing columns in apikeys table: {missing_columns}")
                    return False
                
                logger.info("✅ Database schema is correct")
                return True
                
        except Exception as e:
            logger.error(f"❌ Schema verification failed: {e}")
            return False


    async def test_connection(self) -> bool:
        """Bağlantıyı ve tabloları test et - DEBUG için"""
        try:
            async with aiosqlite.connect(self.db_path) as db:
                # Tablo var mı kontrol et
                cursor = await db.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='apikeys'")
                result = await cursor.fetchone()
                
                if not result:
                    logger.error("❌ apikeys table does not exist!")
                    return False
                    
                logger.info("✅ Database connection test successful - apikeys table exists")
                return True
                
        except Exception as e:
            logger.error(f"❌ Database connection test failed: {e}")
            return False

    async def get_database_status(self) -> Dict[str, Any]:
        """Database durumunu kontrol et"""
        try:
            import os
            db_status = {}
            
            # Database file existence and size
            if os.path.exists(self.db_path):
                db_status['database_exists'] = True
                db_status['database_path'] = self.db_path
                db_status['database_size'] = os.path.getsize(self.db_path)
            else:
                db_status['database_exists'] = False
                return db_status
            
            # Table counts
            async with aiosqlite.connect(self.db_path) as db:
                # Cache size
                cursor = await db.execute("PRAGMA cache_size")
                cache_result = await cursor.fetchone()
                db_status['cache_size'] = cache_result[0] if cache_result else 0
                
                # Table counts
                tables = ['apikeys', 'alarms', 'trade_settings']
                for table in tables:
                    try:
                        cursor = await db.execute(f"SELECT COUNT(*) FROM {table}")
                        result = await cursor.fetchone()
                        db_status[f'{table}_count'] = result[0] if result else 0
                    except Exception as e:
                        db_status[f'{table}_count'] = f"error: {e}"
            
            return db_status
            
        except Exception as e:
            return {"error": str(e)}

#            

# ============================================================
# ALARM MANAGER
# ============================================================

class AlarmManager(BaseManager):
    _instance: Optional["AlarmManager"] = None
    _cache: Dict[int, Dict[str, Any]] = {}
    _cache_lock: asyncio.Lock = asyncio.Lock()

    def __init__(self):
        super().__init__()

    @classmethod
    def get_instance(cls) -> "AlarmManager":
        if cls._instance is None:
            cls._instance = AlarmManager()
        return cls._instance

    async def set_alarm_settings(self, user_id: int, settings: dict) -> None:
        """Set alarm settings with validation"""
        if not isinstance(settings, dict):
            raise ValueError("Settings must be a dictionary")
            
        # Ensure database is initialized
        await self.init_db()
            
        settings_json = json.dumps(settings, ensure_ascii=False)
        db = await self.get_db_connection()
        
        try:
            await db.execute(
                "UPDATE apikeys SET alarm_settings=?, updated_at=CURRENT_TIMESTAMP WHERE user_id=?", 
                (settings_json, user_id)
            )
            await db.commit()
            
            # Update cache
            async with self._cache_lock:
                self._cache[user_id] = settings
                
            logger.info(f"🔔 Alarm settings updated for user {user_id}")
        except Exception as e:
            await db.rollback()
            logger.error(f"❌ Failed to update alarm settings for user {user_id}: {e}")
            raise

    async def get_alarm_settings(self, user_id: int) -> Optional[dict]:
        """Get alarm settings from cache or database"""
        # Check cache first
        async with self._cache_lock:
            if user_id in self._cache:
                return self._cache[user_id]
        
        # Query database
        db = await self.get_db_connection()
        try:
            async with db.execute("SELECT alarm_settings FROM apikeys WHERE user_id=?", (user_id,)) as cursor:
                row = await cursor.fetchone()
                if row and row[0]:
                    settings = json.loads(row[0])
                    
                    # Update cache
                    async with self._cache_lock:
                        self._cache[user_id] = settings
                    
                    return settings
                return None
        except Exception as e:
            logger.error(f"❌ Failed to get alarm settings for user {user_id}: {e}")
            return None

    async def add_alarm(self, user_id: int, alarm_data: dict) -> int:
        """Add new alarm and return alarm ID"""
        if not isinstance(alarm_data, dict):
            raise ValueError("Alarm data must be a dictionary")
            
        # Ensure database is initialized
        await self.init_db()
            
        alarm_json = json.dumps(alarm_data, ensure_ascii=False)
        db = await self.get_db_connection()
        
        try:
            async with db.cursor() as cursor:
                await cursor.execute(
                    "INSERT INTO alarms (user_id, alarm_data) VALUES (?, ?)", 
                    (user_id, alarm_json)
                )
                await db.commit()
                alarm_id = cursor.lastrowid
                logger.info(f"🔔 Alarm added for user {user_id}, alarm_id: {alarm_id}")
                return alarm_id
        except Exception as e:
            await db.rollback()
            logger.error(f"❌ Failed to add alarm for user {user_id}: {e}")
            raise

    async def get_alarms(self, user_id: int, active_only: bool = True) -> List[dict]:
        """Get alarms for user with optional active filter"""
        # Ensure database is initialized
        await self.init_db()
        
        db = await self.get_db_connection()
        try:
            if active_only:
                query = "SELECT id, alarm_data FROM alarms WHERE user_id=? AND is_active=1 ORDER BY created_at DESC"
            else:
                query = "SELECT id, alarm_data FROM alarms WHERE user_id=? ORDER BY created_at DESC"
                
            async with db.execute(query, (user_id,)) as cursor:
                results = []
                async for row in cursor:
                    try:
                        alarm_data = json.loads(row[1])
                        alarm_data['id'] = row[0]  # Include alarm ID
                        results.append(alarm_data)
                    except json.JSONDecodeError as e:
                        logger.error(f"❌ Invalid alarm data for user {user_id}, alarm_id {row[0]}: {e}")
                return results
        except Exception as e:
            logger.error(f"❌ Failed to get alarms for user {user_id}: {e}")
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
    _cache: Dict[int, Dict[str, Any]] = {}
    _cache_lock: asyncio.Lock = asyncio.Lock()

    def __init__(self):
        super().__init__()

    @classmethod
    def get_instance(cls) -> "TradeSettingsManager":
        if cls._instance is None:
            cls._instance = TradeSettingsManager()
        return cls._instance

    async def set_trade_settings(self, user_id: int, settings: dict) -> None:
        """Set trade settings with validation"""
        if not isinstance(settings, dict):
            raise ValueError("Settings must be a dictionary")
            
        # Basic validation for common trade settings
        if 'max_trade' in settings and settings['max_trade'] <= 0:
            raise ValueError("Max trade must be positive")
            
        # Ensure database is initialized
        await self.init_db()
            
        settings_json = json.dumps(settings, ensure_ascii=False)
        db = await self.get_db_connection()
        
        try:
            await db.execute(
                "UPDATE apikeys SET trade_settings=?, updated_at=CURRENT_TIMESTAMP WHERE user_id=?", 
                (settings_json, user_id)
            )
            await db.commit()
            
            # Update cache
            async with self._cache_lock:
                self._cache[user_id] = settings
                
            logger.info(f"📊 Trade settings updated for user {user_id}")
        except Exception as e:
            await db.rollback()
            logger.error(f"❌ Failed to update trade settings for user {user_id}: {e}")
            raise

    async def get_trade_settings(self, user_id: int) -> Optional[dict]:
        """Get trade settings from cache or database"""
        # Check cache first
        async with self._cache_lock:
            if user_id in self._cache:
                return self._cache[user_id]
        
        # Query database
        db = await self.get_db_connection()
        try:
            async with db.execute("SELECT trade_settings FROM apikeys WHERE user_id=?", (user_id,)) as cursor:
                row = await cursor.fetchone()
                if row and row[0]:
                    settings = json.loads(row[0])
                    
                    # Update cache
                    async with self._cache_lock:
                        self._cache[user_id] = settings
                    
                    return settings
                return None
        except Exception as e:
            logger.error(f"❌ Failed to get trade settings for user {user_id}: {e}")
            return None

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
# INITIALIZATION AND HEALTH CHECK
# ============================================================

async def initialize_managers():
    """Initialize all managers and ensure database is ready - DÜZELTİLDİ"""
    try:
        # Create instances
        api_key_manager = APIKeyManager.get_instance()
        
        # Initialize database
        success = await api_key_manager.init_db()
        if not success:
            logger.error("❌ Database initialization failed")
            return False
        
        # Verify schema
        schema_ok = await api_key_manager.verify_database_schema()
        if not schema_ok:
            logger.error("❌ Database schema verification failed")
            return False
        
        # Check database status
        status = await api_key_manager.check_database_status()
        logger.info(f"📊 Database status: {status}")
        
        # Diğer manager'ları da oluştur (ama init_db çağırma - zaten yapıldı)
        alarm_manager = AlarmManager.get_instance()
        trade_manager = TradeSettingsManager.get_instance()
        
        logger.info("✅ All managers initialized successfully")
        return True
    except Exception as e:
        logger.error(f"❌ Manager initialization failed: {e}")
        return False
# Auto-initialize on import (optional)
# asyncio.create_task(initialize_managers())