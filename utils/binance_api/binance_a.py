# utils/binance_api/binance_a.py
# Sorgu → binance_a.py → YAML config → DirectBinanceClient → BinanceHTTPClient → Cevap

from __future__ import annotations

import os
import asyncio
import logging
import uuid
from typing import Dict, Optional, Any, Set, Tuple
#from importlib import import_module
#from functools import wraps

from utils.apikey_manager import APIKeyManager, GLOBAL_USER
from utils.context_logger import get_context_logger, ContextAwareLogger # BinanceAggregator modülü
from utils.security_auditor import security_auditor
from utils.performance_monitor import monitor_performance

# ✅ LOGGER  - bu satırı buraya ekle
logger = get_context_logger(__name__)



# 🔄 relative import: frm .b_config import ...
# ✅ Kesin(mutlak) absolute import proje büyüyecek, testler, CI/CD, modül dışı kullanımlar

from utils.binance_api.b_config import BinanceConfig
from utils.binance_api.rate_limiter import UserAwareRateLimiter
from utils.binance_api.cache import cached_binance_data, BinanceCacheManager
from utils.binance_api.binance_request import BinanceHTTPClient, UserAwareRateLimiter
from utils.binance_api.binance_circuit_breaker import CircuitBreaker
from utils.binance_api.binance_multi_user import UserSessionManager
from utils.binance_api.binance_metrics import AdvancedMetrics
from utils.binance_api.b_map_validator import BMapValidator, MapValidationError
from utils.binance_api.direct_client import DirectBinanceClient

class MapLoader:
    """YAML map dosyalarını yükler ve doğrular."""

    def __init__(self, base_path: str) -> None:
        self.base_path = base_path
        self.maps: Dict[str, Dict[str, Any]] = {}
        self._validator = BMapValidator()

    def load_all(self) -> None:
        """Tüm YAML map dosyalarını yükle."""
        import os

        files = ("b_map_public.yaml", "b_map_private.yaml")
        for fname in files:
            path = os.path.join(self.base_path, fname)
            if not os.path.exists(path):
                logger.warning(f"Map file missing: {path}")
                continue
            try:
                data = self._validator.load_yaml(path)
                self._validator.validate(data, fname)
                self.maps[fname.replace(".yaml", "")] = data
                logger.info(f"Loaded map: {fname}")
            except MapValidationError as exc:
                logger.error(f"Map validation failed for {fname}: {exc}")

    def get_endpoint(self, name: str) -> Optional[Dict[str, Any]]:
        """Endpoint adını YAML map içinde bul."""
        for map_name, map_data in self.maps.items():
            for section, endpoints in map_data.items():
                if section == "meta":
                    continue
                if name in endpoints:
                    return endpoints[name]
        return None
        

class BinanceDependencyContainer:
    """Dependency injection container for Binance components"""
    
    def __init__(self):
        self._apikey_manager = None
        self._metrics_manager = None
        self._session_manager = None
        self._cache_manager = None
        self._rate_limiter = None
    
    async def get_apikey_manager(self) -> APIKeyManager:
        if not self._apikey_manager:
            self._apikey_manager = APIKeyManager.get_instance()
            await self._apikey_manager.init_db()
        return self._apikey_manager
    
    async def get_metrics_manager(self):
        if not self._metrics_manager:
            self._metrics_manager = AdvancedMetrics.get_instance()
        return self._metrics_manager
    
    async def get_session_manager(self) -> UserSessionManager:
        if not self._session_manager:
            self._session_manager = UserSessionManager(ttl_minutes=60)
        return self._session_manager
    
    async def get_cache_manager(self) -> BinanceCacheManager:
        if not self._cache_manager:
            self._cache_manager = BinanceCacheManager()
        return self._cache_manager
    
    async def get_rate_limiter(self) -> UserAwareRateLimiter:
        if not self._rate_limiter:
            self._rate_limiter = UserAwareRateLimiter()
        return self._rate_limiter

# varsa kişisel yoksa .env den api+secret, 
class BinanceAggregator:
    _instance = None
    _instance_params = None

    def __init__(self, base_path: str, config: Optional["BinanceConfig"] = None):
        if hasattr(self, "_initialized") and self._initialized:
            raise RuntimeError("BinanceAggregator singleton already initialized")
        
        # 📦 Core initialization (mevcut yapı)
        self.map_loader = MapLoader(base_path)
        self.map_loader.load_all()

        self.container = BinanceDependencyContainer()
        self.config = config or BinanceConfig()

        # 🔒 Kullanıcı bazlı lock sistemi
        self._user_locks: Dict[int, asyncio.Lock] = {}
        self._locks_lock = asyncio.Lock()

        self.sessions = UserSessionManager(ttl_minutes=60)
        self.key_manager = APIKeyManager.get_instance()
        self.circuit_breaker = CircuitBreaker()

        self._cleanup_task: Optional[asyncio.Task] = None
        self._stop_event = asyncio.Event()

        # ------------------------------------------------------------
        # 🧩 🔐 Global API key + config (bot) erişimi - DÜZELTİLMİŞ
        # ------------------------------------------------------------
        # ❌ ESKİ: from config import get_config_sync - İPTAL
        # ✅ YENİ: Doğrudan environment variables
        self.global_api_key = os.getenv("BINANCE_API_KEY")
        self.global_api_secret = os.getenv("BINANCE_API_SECRET")
        self.api_manager = self.key_manager  # alias

        # Singleton kontrolü
        self._initialized = True
        logger.info("✅ BinanceAggregator initialized successfully")    

    # ✅ YENİ EKLENECEK METOD - __init__'den SONRA
    async def initialize_managers(self):
        """Manager'ları async olarak initialize et"""
        try:
            # ✅ Database initialization - DOĞRU YÖNTEM
            await self.key_manager.ensure_db_initialized()
            
            # ✅ Global credentials validation (opsiyonel)
            if self.global_api_key and self.global_api_secret:
                is_valid = await self.key_manager.validate_global_credentials()
                if not is_valid:
                    logger.warning("⚠️ Global API credentials validation failed")
            
            logger.info("✅ All managers initialized successfully")
            
        except Exception as e:
            logger.error(f"❌ Manager initialization failed: {e}")
            raise

    # ✅ Singleton pattern düzeltmesi - MEVCUT get_instance YERİNE
    @classmethod
    async def get_instance(cls, base_path: str = None, config: Optional["BinanceConfig"] = None) -> "BinanceAggregator":
        """Async singleton getter - DÜZELTİLMİŞ"""
        if cls._instance is None:
            if base_path is None:
                base_path = os.path.abspath(os.path.join(os.path.dirname(__file__)))
            cls._instance = cls(base_path, config)
            
            # ✅ CRITICAL: Async initialization çağır
            await cls._instance.initialize_managers()
            
        else:
            if config is not None and cls._instance.config != config:
                logger.warning("Singleton already initialized with different config - using existing instance")
                
        return cls._instance
        




    # =======================================================
    # 👇 Kullanıcıya göre API key belirleme
    # =======================================================
                
    async def get_user_credentials(self, user_id: Optional[int] = None) -> Tuple[str, str]:
        # 1. Kişisel API
        if user_id and user_id != GLOBAL_USER:
            try:
                user_creds = await self.api_manager.get_user_credentials(user_id)
                if user_creds:
                    logger.info("Using personal API credentials for request", extra={
                        "user_id": user_id,
                        "source": "personal",
                        "tier": "private",
                        "endpoint": "get_user_credentials"
                    })
                    return user_creds.api_key, user_creds.api_secret
                else:
                    logger.debug("No personal API found, falling back to global", extra={
                        "user_id": user_id,
                        "source": "personal",
                        "tier": "private",
                        "fallback": True
                    })
            except Exception as e:
                logger.warning("Error getting user credentials, falling back to global", extra={
                    "user_id": user_id,
                    "source": "personal", 
                    "tier": "private",
                    "error": str(e),
                    "fallback": True
                })

        # 2. Fallback - Global API
        if not self.global_api_key or not self.global_api_secret:
            logger.error("No valid credentials available", extra={
                "user_id": user_id,
                "source": "none",
                "tier": "system",
                "global_key_available": bool(self.global_api_key),
                "global_secret_available": bool(self.global_api_secret)
            })
            raise RuntimeError("No Binance API credentials available")

        logger.info("Using global API credentials for request", extra={
            "user_id": user_id or 0,
            "source": "global",
            "tier": "system", 
            "endpoint": "get_user_credentials"
        })
        return self.global_api_key, self.global_api_secret
        


    async def validate_global_credentials(self) -> bool:
        """Global API key'leri doğrula - DÜZELTİLDİ"""
        try:
            # ✅ DÜZELTME: _get_global_credentials yerine get_user_credentials kullan
            api_key, secret_key = await self.get_user_credentials(user_id=None)
            
            if not api_key or not secret_key:
                logger.warning("⚠️ Global API keys not found")
                return False
                
            # Binance API ile doğrulama yap
            from binance import AsyncClient
            client = await AsyncClient.create(api_key, secret_key)
            
            try:
                # Basit bir API çağrısı ile doğrula
                await client.get_account()
                logger.info("✅ Global API credentials validated successfully")
                return True
            except Exception as e:
                logger.error(f"❌ Global credentials validation failed: {e}")
                return False
            finally:
                await client.close_connection()
                
        except Exception as e:
            logger.error(f"❌ Global credentials validation error: {e}")
            return False
            

    # -----------------------------------------------------------
    # 🔹 Kullanıcı veya global API anahtarlarını al
    # -----------------------------------------------------------
    async def _get_active_users(self) -> Set[int]:
        """Aktif kullanıcıları al - fallback implementasyon"""
        try:
            if hasattr(self.sessions, 'get_active_users') and callable(getattr(self.sessions, 'get_active_users')):
                users = await self.sessions.get_active_users()
                return set(users)
            
            if hasattr(self.sessions, 'sessions') and isinstance(self.sessions.sessions, dict):
                return set(self.sessions.sessions.keys())
                
            logger.warning("Active users detection fallback - no sessions found")
            return set()
            
        except Exception as e:
            logger.error(f"Error getting active users: {e}")
            return set()


    # ✅ SATIR 260-280: Yeni _get_direct_client metodu
    def _get_direct_client(self, http_client: "BinanceHTTPClient") -> DirectBinanceClient:
        """Create direct client for YAML-based calls"""
        return DirectBinanceClient(http_client)
        


    async def _get_or_create_session(self, user_id: int, api_key: str, secret_key: str):
        """Session'ı al veya oluştur - user_id artık her zaman int"""
        user_lock = await self._get_user_lock(user_id)
        async with user_lock:
            session = await self.sessions.get_session(user_id)
            if not session:
                http_client = BinanceHTTPClient(
                    api_key=api_key, 
                    secret_key=secret_key,
                    user_id=user_id
                )
                await self.sessions.add_session(user_id, http_client, self.circuit_breaker)
                session = await self.sessions.get_session(user_id)
            return session
            

    async def start_background_tasks(self, interval: int = 300) -> None:
        if self._cleanup_task:
            return
        self._cleanup_task = asyncio.create_task(self._cleanup_loop(interval))
        logger.info("🔁 Background cleanup loop started")

    async def stop_background_tasks(self) -> None:
        if not self._cleanup_task:
            return
        self._stop_event.set()
        self._cleanup_task.cancel()
        try:
            await self._cleanup_task
        except asyncio.CancelledError:
            pass
        logger.info("🛑 Background cleanup loop stopped")

    async def _cleanup_loop(self, interval: int) -> None:
        while not self._stop_event.is_set():
            try:
                await self.sessions.cleanup_expired_sessions()
                await self.key_manager.cleanup_cache()
                await self._cleanup_user_locks()
                
                if hasattr(self.circuit_breaker, "cleanup"):
                    cleanup = getattr(self.circuit_breaker, "cleanup")
                    if asyncio.iscoroutinefunction(cleanup):
                        await cleanup()
                    else:
                        cleanup()
            except Exception as e:
                logger.error(f"Cleanup error: {e}")
            await asyncio.sleep(interval)

    async def _get_user_lock(self, user_id: int) -> asyncio.Lock:
        async with self._locks_lock:
            if user_id not in self._user_locks:
                self._user_locks[user_id] = asyncio.Lock()
            return self._user_locks[user_id]

    async def _cleanup_user_locks(self):
        async with self._locks_lock:
            active_users = await self._get_active_users()
            to_remove = [uid for uid in self._user_locks.keys() if uid not in active_users]
            for uid in to_remove:
                del self._user_locks[uid]

    # -----------------------------------------------------------
    # 🔹 Ana endpoint çağrısı (user veya global API key fallback)
    # -----------------------------------------------------------

    @cached_binance_data(ttl=300)
    @monitor_performance("get_data", warning_threshold=2.5)
   
    async def _set_request_context(self, user_id: int, endpoint_name: str):
        """Set context for the current request - user_id artık her zaman int"""
        ContextAwareLogger.set_user_context(user_id)
        request_id = str(uuid.uuid4())[:8]
        ContextAwareLogger.set_request_context(request_id, endpoint_name)
        

    # VERİ SORGULAMA
    # PUBLIC METHOD :  SADECE public data - .env API key ile
    async def get_public_data(self, endpoint_name: str, **params) -> Any:
        logger.info("Public data request initiated", extra={
            "user_id": 0,
            "endpoint": endpoint_name,
            "tier": "public",
            "params": list(params.keys())
        })
        return await self._get_data_internal(endpoint_name, None, **params)    
        
    
    
    # PRIVATE METHOD  : TÜM data - kişisel API key ile
    async def get_private_data(self, user_id: int, endpoint_name: str, **params) -> Any:
        logger.info("Private data request initiated", extra={
            "user_id": user_id,
            "endpoint": endpoint_name, 
            "tier": "private",
            "params": list(params.keys())
        })
        return await self._get_data_internal(endpoint_name, user_id, **params)   


    
    # AKILLI METHOD
    async def get_data(self, endpoint_name: str, user_id: Optional[int] = None, **params) -> Any:
        """AKILLI METHOD: Otomatik public/private seçimi"""
        if user_id is not None:
            return await self.get_private_data(user_id, endpoint_name, **params)
        else:
            return await self.get_public_data(endpoint_name, **params)
            

    # YARDIMCI METHOD: Public- private  Endpoint ayrım Kontrolü 
    def _is_public_endpoint(self, endpoint_name: str) -> bool:
        """Endpoint'in public olup olmadığını kontrol et"""
        endpoint_config = self.map_loader.get_endpoint(endpoint_name)
        if not endpoint_config:
            raise ValueError(f"Endpoint bulunamadı: {endpoint_name}")
        return not endpoint_config.get('signed', False)
        
    
    # ORTAK LOGIC
    async def _get_data_internal(self, endpoint_name: str, user_id: Optional[int], **params) -> Any:
        """ORTAK LOGIC - tüm data method'ları buradan beslenir (YENİ AKIŞ)"""
        # Context ayarla
        effective_user_id = user_id if user_id is not None else 0  # Global için 0
        logger.info("Processing data request", extra={
            "user_id": effective_user_id,
            "endpoint": endpoint_name,
            "params_count": len(params),
            "tier": "private" if user_id else "public"
        })
        
        await self._set_request_context(effective_user_id, endpoint_name)
        
        try:
            # Endpoint kontrolü
            if not endpoint_name:
                raise ValueError("endpoint_name boş olamaz")
            
            endpoint = self.map_loader.get_endpoint(endpoint_name)
            if not endpoint:
                raise ValueError(f"Endpoint bulunamadı: {endpoint_name}")

            # Public endpoint kontrolü (user_id=None ise)
            if user_id is None and not self._is_public_endpoint(endpoint_name):
                raise PermissionError(f"Private endpoint {endpoint_name} requires user authentication")

            # 🔐 Security audit
            audit_result = await security_auditor.audit_request(user_id, endpoint_name, params)
            if not audit_result:
                raise PermissionError(f"Security audit failed: {endpoint_name}")

            # 🔑 API credentials
            api_key, secret_key = await self.get_user_credentials(user_id)

            # 🔒 Session
            session = await self._get_or_create_session(effective_user_id, api_key, secret_key)
            if not session:
                raise RuntimeError("User session oluşturulamadı")

            http_client = session.http_client

            # Endpoint config
            signed = bool(endpoint.get("signed", False))
            estimated_weight = int(endpoint.get("weight", 1))

            # 🚦 Rate limiting (TEK SEFER)
            rate_limiter = await self.container.get_rate_limiter()
            await rate_limiter.acquire(effective_user_id, endpoint_name, estimated_weight)

            try:
                # 🆕 DİREKT YAML TABANLI ÇAĞRI
                result = await self._call_direct_endpoint(http_client, endpoint, **params)
                return result
                
            except Exception as e:
                logger.error(f"API çağrısı başarısız: {e}", exc_info=True)
                raise
            finally:
                await rate_limiter.release(effective_user_id, endpoint_name, estimated_weight)
                
        except Exception as e:
            logger.error("Data request failed", extra={
                "user_id": effective_user_id,
                "endpoint": endpoint_name,
                "error": str(e),
                "error_type": type(e).__name__
            })
            
            raise
        finally:
            ContextAwareLogger.clear_context()
            
    async def _call_direct_endpoint(self, http_client: "BinanceHTTPClient", endpoint_config: Dict[str, Any], **params) -> Any:
        """Call endpoint directly using YAML config"""
        direct_client = self._get_direct_client(http_client)
        return await direct_client.call_endpoint(endpoint_config, **params)
            

    async def _get_data_internal(self, endpoint_name: str, user_id: Optional[int], **params) -> Any:
        """ORTAK LOGIC - tüm data method'ları buradan beslenir (YENİ AKIŞ)"""
        # Context ayarla
        effective_user_id = user_id if user_id is not None else 0  # Global için 0
        logger.info("Processing data request", extra={
            "user_id": effective_user_id,
            "endpoint": endpoint_name,
            "params_count": len(params),
            "tier": "private" if user_id else "public"
        })
        
        await self._set_request_context(effective_user_id, endpoint_name)
        
        # ✅ DÜZELTME: endpoint değişkenini burada tanımla
        endpoint = None
        
        try:
            # Endpoint kontrolü
            if not endpoint_name:
                raise ValueError("endpoint_name boş olamaz")
            
            # ✅ DÜZELTME: endpoint değişkenini burada ata
            endpoint = self.map_loader.get_endpoint(endpoint_name)
            if not endpoint:
                raise ValueError(f"Endpoint bulunamadı: {endpoint_name}")

            # Public endpoint kontrolü (user_id=None ise)
            if user_id is None and not self._is_public_endpoint(endpoint_name):
                raise PermissionError(f"Private endpoint {endpoint_name} requires user authentication")

            # 🔐 Security audit
            audit_result = await security_auditor.audit_request(user_id, endpoint_name, params)
            if not audit_result:
                raise PermissionError(f"Security audit failed: {endpoint_name}")

            # 🔑 API credentials
            api_key, secret_key = await self.get_user_credentials(user_id)

            # 🔒 Session
            session = await self._get_or_create_session(effective_user_id, api_key, secret_key)
            if not session:
                raise RuntimeError("User session oluşturulamadı")

            http_client = session.http_client

            # Endpoint config
            signed = bool(endpoint.get("signed", False))
            estimated_weight = int(endpoint.get("weight", 1))

            # 🚦 Rate limiting (TEK SEFER)
            rate_limiter = await self.container.get_rate_limiter()
            await rate_limiter.acquire(effective_user_id, endpoint_name, estimated_weight)

            try:
                # 🆕 DİREKT YAML TABANLI ÇAĞRI
                result = await self._call_direct_endpoint(http_client, endpoint, **params)
                return result
                
            except Exception as e:
                logger.error(f"API çağrısı başarısız: {e}", exc_info=True)
                raise
            finally:
                await rate_limiter.release(effective_user_id, endpoint_name, estimated_weight)
                
        except Exception as e:
            logger.error("Data request failed", extra={
                "user_id": effective_user_id,
                "endpoint": endpoint_name,
                "error": str(e),
                "error_type": type(e).__name__
            })
            
            raise
        finally:
            ContextAwareLogger.clear_context()
            

#
"""
yapı uygunluğu

utils/binance_api/binance_a.py
- 🔹 API endpoint’lerine ve metodlara doğrudan kod içinden değil, 
“map” (harita/yaml dosyası) üzerinden erişilir.

- 🔹 maps tabanlı agredator akış yapısı
sorgu >> utils/binance_api/binance_a.py >> b_spotclient/ b_futuresclient (clientler içinde public / private ayrımı)>> 
b_map_public.yaml/ b_map_private.yaml >> cevap >> binance_a.py

Yeni
Akış
Sorgu >> binance_a.py >> aggregator (public/private ayrımı burada) >> b_map_public.yaml / b_map_private.yaml >> cevap

Client.py dosyaları iptal, metodlar dinamik çağrılıyor
Tek Aggregator üzerinden base bilgisine göre http_client seçiliyor




binance_a.py üzerindeki sorgularda şu hedefleniyor. 
binance_a.py, kişinin kendi api+secret girme durumuna göre işlem yapacak
eğer kişisel api+secret yok ise .env deki api+secret kullanılacak
- 🔹 Sadece okuma — fiyat, analiz, metrik, public veri
- Sistem genelinde, güvenli read-only sorgular

eğer kişisel api+secret var ise bu api+secret kullanılacak
- 🔹 Okuma + Yazma — trade, pozisyon, cüzdan dahil tüm işlemler
- Kullanıcıya özel, tam yetkili işlemler

 public / private ayrımı hem b_spotclient/ b_futuresclient hem *yaml içinde (signed=true/false)

# 1. PUBLIC (.env API)
# 2. PRIVATE (Kişisel API)
# 3. YASAK DURUM

"""
