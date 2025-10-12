# utils/binance_api/binance_a.py

from __future__ import annotations

import asyncio
from importlib import import_module
from typing import Dict, Optional, Any, Set

from utils.apikey_manager import APIKeyManager
from utils.context_logger import get_context_logger, ContextAwareLogger
from utils.security_auditor import security_auditor
from utils.performance_monitor import monitor_performance

from functools import wraps

# 🔄 TUTARLI IMPORTLAR - hepsi relative
from .config import BinanceConfig
from .rate_limiter import UserAwareRateLimiter
from .cache import cached_binance_data, BinanceCacheManager
from .b_map_validator import BMapValidator, MapValidationError
from .binance_request import BinanceHTTPClient
from .binance_circuit_breaker import CircuitBreaker
from .binance_multi_user import UserSessionManager

logger = get_context_logger(__name__)


class MapLoader:
    """YAML map dosyalarını yükler ve doğrular."""

    def __init__(self, base_path: str) -> None:
        self.base_path = base_path
        self.maps: Dict[str, Dict[str, Any]] = {}
        self._validator = BMapValidator()

    def load_all(self) -> None:
        """Tüm YAML map dosyalarını yükle."""
        import os

        files = ("b-map_public.yaml", "b-map_private.yaml")
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
            from .binance_metrics import AdvancedMetrics
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



class BinanceAggregator:
    
    _instance = None
    _instance_params = None

    @classmethod
    def get_instance(cls, base_path: str = None, config: Optional[BinanceConfig] = None):
        if cls._instance is None:
            if base_path is None:
                import os
                base_path = os.path.dirname(__file__)  # 🎯 YAML'lerin olduğu dizin
            cls._instance = cls(base_path, config)
        return cls._instance
    
    
    def __init__(self, base_path: str, config: Optional[BinanceConfig] = None):
        if hasattr(self, '_initialized') and self._initialized:
            raise RuntimeError("BinanceAggregator singleton already initialized")
            
        self.map_loader = MapLoader(base_path)
        self.map_loader.load_all()
        
        # Dependency container
        self.container = BinanceDependencyContainer()
        
        # Config
        self.config = config or BinanceConfig()
        
        # 🔒 Kullanıcı bazlı lock sistemi
        self._user_locks: Dict[int, asyncio.Lock] = {}
        self._locks_lock = asyncio.Lock()
        
        # Mevcut initialization
        self.sessions = UserSessionManager(ttl_minutes=60)
        self.key_manager = APIKeyManager.get_instance()
        self.circuit_breaker = CircuitBreaker()

        # Cleanup mekanizması
        self._cleanup_task: Optional[asyncio.Task] = None
        self._stop_event = asyncio.Event()

        # Singleton kontrolü
        self._initialized = True
        
        logger.info("✅ BinanceAggregator initialized successfully")

    async def _get_active_users(self) -> Set[int]:
        """Aktif kullanıcıları al - fallback implementasyon"""
        try:
            # UserSessionManager'da bu metod yoksa kendi implementasyonumuzu kullan
            if hasattr(self.sessions, 'get_active_users') and callable(getattr(self.sessions, 'get_active_users')):
                users = await self.sessions.get_active_users()
                return set(users)
            
            # Fallback: sessions dict'inden aktif kullanıcıları al
            if hasattr(self.sessions, 'sessions') and isinstance(self.sessions.sessions, dict):
                return set(self.sessions.sessions.keys())
                
            logger.warning("Active users detection fallback - no sessions found")
            return set()
            
        except Exception as e:
            logger.error(f"Error getting active users: {e}")
            return set()

    async def _get_or_create_session(self, user_id: int, api_key: str, secret_key: str):
        """Session'ı al veya oluştur - user bazlı lock ile"""
        user_lock = await self._get_user_lock(user_id)
        async with user_lock:
            session = await self.sessions.get_session(user_id)
            if not session:
                http_client = BinanceHTTPClient(
                    api_key=api_key, 
                    api_secret=secret_key, 
                    user_id=user_id
                )
                await self.sessions.add_session(user_id, http_client, self.circuit_breaker)
                session = await self.sessions.get_session(user_id)
            return session


    async def start_background_tasks(self, interval: int = 300) -> None:
        """Background task'leri başlat"""
        if self._cleanup_task:
            return
        self._cleanup_task = asyncio.create_task(self._cleanup_loop(interval))
        logger.info("🔁 Background cleanup loop started")

    async def stop_background_tasks(self) -> None:
        """Background task'leri durdur"""
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
        """Periyodik temizlik döngüsü"""
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
        """Kullanıcı için lock al - thread-safe"""
        async with self._locks_lock:
            if user_id not in self._user_locks:
                self._user_locks[user_id] = asyncio.Lock()
            return self._user_locks[user_id]

    async def _cleanup_user_locks(self):
        """Kullanılmayan user lock'larını temizle"""
        async with self._locks_lock:
            active_users = await self._get_active_users()
            to_remove = [uid for uid in self._user_locks.keys() if uid not in active_users]
            for uid in to_remove:
                del self._user_locks[uid]




    # -----------------------------------------------------------
    # 🔹 Ana endpoint çağrısı - DÜZELTİLMİŞ VERSİYON
    # -----------------------------------------------------------
    @cached_binance_data(ttl=300)
    @monitor_performance("get_data", warning_threshold=2.5)
    async def get_data(self, user_id: int, endpoint_name: str, **params) -> Any:
        """YAML map'teki endpoint'e göre çağrı yapar."""
        
        ContextAwareLogger.set_user_context(user_id)

        # Validasyonlar
        if not endpoint_name:
            raise ValueError("endpoint_name boş olamaz")

        endpoint = self.map_loader.get_endpoint(endpoint_name)
        if not endpoint:
            raise ValueError(f"Endpoint bulunamadı: {endpoint_name}")

        # 🔒 Audit - güvenlik kontrolü
        if not await security_auditor.audit_request(user_id, endpoint_name, params):
            raise PermissionError(f"Audit kontrolü başarısız: {endpoint_name}")

        # 🔑 API key doğrulama
        valid = await self.key_manager.validate_binance_credentials(user_id)
        if not valid:
            raise ValueError(f"Kullanıcı {user_id} için API key geçerli değil.")

        creds = await self.key_manager.get_apikey(user_id)
        if not creds:
            raise ValueError("API anahtarları alınamadı.")
        api_key, secret_key = creds

        # 🔒 Session oluştur/al
        session = await self._get_or_create_session(user_id, api_key, secret_key)
        if not session:
            raise RuntimeError("User session oluşturulamadı.")

        http_client: BinanceHTTPClient = session.http_client

        # Endpoint bilgilerini al
        client_class = endpoint.get("client")
        method_name = endpoint.get("method")
        signed = bool(endpoint.get("signed", False))
        estimated_weight = int(endpoint.get("weight", 1))

        # Dinamik client çözümleme
        client = self._resolve_client(client_class, http_client)
        func = getattr(client, method_name, None)
        if not func:
            raise AttributeError(f"{client_class} içinde {method_name} bulunamadı.")

        logger.info(
            f"⚙️  Executing {client_class}.{method_name} (signed={signed}, weight={estimated_weight}) for user {user_id}"
        )

        # Container'dan manager'ları al
        cache_manager = await self.container.get_cache_manager()
        rate_limiter = await self.container.get_rate_limiter()

        # 🚦 Rate limiting
        await rate_limiter.acquire(user_id, endpoint_name, estimated_weight)

        try:
            # 🔄 Circuit breaker ile API çağrısı
            result = await self.circuit_breaker.execute(
                func, **params, estimated_weight=estimated_weight
            )
            
            # 🧹 Signed endpoint'lerde cache invalidation
            if signed:
                await cache_manager.invalidate_user_cache(user_id)
            
            return result
            
        except Exception as e:
            logger.error(f"❌ API çağrısı başarısız: {e}")
            raise
        finally:
            # 🔓 Rate limiter'ı her durumda serbest bırak
            await rate_limiter.release(user_id, endpoint_name, estimated_weight)

    def _resolve_client(self, client_class: str, http_client: BinanceHTTPClient) -> Any:
        """Dinamik modül ve sınıf yükleyici."""
        try:
            module_name = f"utils.binance_api.{client_class.lower()}"
            module = import_module(module_name)
            cls = getattr(module, client_class)
            return cls.get_instance(http_client=http_client)
        except Exception as e:
            logger.error(f"Client çözümleme hatası: {e}")
            raise