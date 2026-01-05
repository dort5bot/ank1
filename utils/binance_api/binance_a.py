# utils/binance_api/binance_a.py
# v2
# Binance API bu metriklerin tamamını besleyebilir
from __future__ import annotations

import os
import asyncio
import logging
import uuid
from dataclasses import dataclass
from typing import Dict, Optional, Any, Tuple, List

from utils.apikey_manager import APIKeyManager, GLOBAL_USER
from utils.context_logger import get_context_logger, ContextAwareLogger
from utils.security_auditor import security_auditor
from utils.performance_monitor import monitor_performance

# ✅ LOGGER
logger = get_context_logger(__name__)

# ✅ Absolute imports
from utils.binance_api.b_config import BinanceConfig
from utils.binance_api.binance_request import BinanceHTTPClient, UserAwareRateLimiter
from utils.binance_api.binance_multi_user import UserSessionManager
from utils.binance_api.binance_metrics import BinanceMetrics, record_request, record_retry
from utils.binance_api.binance_client import DirectBinanceClient

# =============================================================
# 🔹 HARD-CODED ENDPOINT TANIMLARI
# =============================================================

@dataclass
class Endpoint:
    base: str
    http_method: str
    path: str
    signed: bool
    weight: int
    parameters: List[str]
    cache: str
    return_type: str = "auto"

# binance_a.py - ENDPOINTS dict
ENDPOINTS = {
    "klines": Endpoint(
        base="spot",
        http_method="GET",
        path="/api/v3/klines",
        signed=False,
        weight=1,
        parameters=[
            {"name": "symbol", "required": True},
            {"name": "interval", "required": True},
            {"name": "limit", "required": False}
        ],
        cache="15s",
        return_type="list"
    ),
    "agg_trades": Endpoint(
        base="spot",
        http_method="GET",
        path="/api/v3/aggTrades",
        signed=False,
        weight=1,
        parameters=[
            {"name": "symbol", "required": True},
            {"name": "limit", "required": False},
            {"name": "fromId", "required": False},
            {"name": "startTime", "required": False},
            {"name": "endTime", "required": False}
        ],
        cache="5s",
        return_type="list"
    ),
    "get_avg_price": Endpoint(
        base="spot",
        http_method="GET",
        path="/api/v3/avgPrice",
        signed=False,
        weight=1,
        parameters=[
            {"name": "symbol", "required": True}
        ],
        cache="5s",
        return_type="dict"
    ),
    "exchange_info": Endpoint(
        base="spot",
        http_method="GET",
        path="/api/v3/exchangeInfo",
        signed=False,
        weight=10,
        parameters=[],
        cache="3600s",
        return_type="dict"
    ),
    "historical_trades": Endpoint(
        base="spot",
        http_method="GET",
        path="/api/v3/historicalTrades",
        signed=False,
        weight=5,
        parameters=[
            {"name": "symbol", "required": True},
            {"name": "limit", "required": False}
        ],
        cache="30s",
        return_type="list"
    ),
    "ui_klines": Endpoint(
        base="spot",
        http_method="GET",
        path="/api/v3/uiKlines",
        signed=False,
        weight=1,
        parameters=[
            {"name": "symbol", "required": True},
            {"name": "interval", "required": True}
        ],
        cache="15s",
        return_type="list"
    ),
    "depth": Endpoint( # order_book_depth
        base="spot",
        http_method="GET",
        path="/api/v3/depth",
        signed=False,
        weight=5,
        parameters=[
            {"name": "symbol", "required": True},
            {"name": "limit", "required": False}
        ],
        cache="10s",
        return_type="dict"
    ),
    "order_book_ticker": Endpoint(
        base="spot",
        http_method="GET",
        path="/api/v3/ticker/bookTicker",
        signed=False,
        weight=1,
        parameters=[
            {"name": "symbol", "required": False}
        ],
        cache="3s",
        return_type="dict"
    ),
    "recent_trades": Endpoint(
        base="spot",
        http_method="GET",
        path="/api/v3/trades",
        signed=False,
        weight=1,
        parameters=[
            {"name": "symbol", "required": True},
            {"name": "limit", "required": False}
        ],
        cache="5s",
        return_type="list"
    ),
    "server_time": Endpoint(
        base="spot",
        http_method="GET",
        path="/api/v3/time",
        signed=False,
        weight=1,
        parameters=[],
        cache="3600s",
        return_type="dict"
    ),
    "symbol_ticker": Endpoint(
        base="spot",
        http_method="GET",
        path="/api/v3/ticker/price",
        signed=False,
        weight=1,
        parameters=[
            {"name": "symbol", "required": False}
        ],
        cache="5s",
        return_type="dict"
    ),
    "ticker_24hr": Endpoint(
        base="spot",
        http_method="GET",
        path="/api/v3/ticker/24hr",
        signed=False,
        weight=1,
        parameters=[
            {"name": "symbol", "required": False}
        ],
        cache="5s",
        return_type="auto"
    ),
    "account_info": Endpoint(
        base="spot",
        http_method="GET",
        path="/api/v3/account",
        signed=True,
        weight=10,
        parameters=[],
        cache="5s",
        return_type="dict"
    ),
    "create_order": Endpoint(
        base="spot",
        http_method="POST",
        path="/api/v3/order",
        signed=True,
        weight=1,
        parameters=[
            {"name": "symbol", "required": True},
            {"name": "side", "required": True},
            {"name": "type", "required": True},
            {"name": "quantity", "required": False},
            {"name": "price", "required": False}
        ],
        cache="0s",
        return_type="dict"
    ),
    # "premium_index"
    "premiumIndex": Endpoint(
        base="futures",
        http_method="GET",
        path="/fapi/v1/premiumIndex",
        signed=False,
        weight=1,
        parameters=[
            {"name": "symbol", "required": False}
        ],
        cache="5s",
        return_type="list_or_dict"
    ),
    "futures_klines": Endpoint(
        base="futures",
        http_method="GET",
        path="/fapi/v1/klines",
        signed=False,
        weight=1,
        parameters=[
            {"name": "symbol", "required": True},
            {"name": "interval", "required": True},
            {"name": "limit", "required": False}
        ],
        cache="15s",
        return_type="list"
    ),
    "futures_liquidations": Endpoint(
        base="futures",
        http_method="GET",
        path="/fapi/v1/allForceOrders",
        signed=False,
        weight=20,
        parameters=[
            {"name": "symbol", "required": False},
            {"name": "startTime", "required": False},
            {"name": "endTime", "required": False},
            {"name": "limit", "required": False}
        ],
        cache="10s",
        return_type="list"
    ),
    "futures_open_interest": Endpoint(
        base="futures",
        http_method="GET",
        path="/fapi/v1/openInterest",
        signed=False,
        weight=1,
        parameters=[
            {"name": "symbol", "required": True}
        ],
        cache="10s",
        return_type="dict"
    ),

    "open_interest": Endpoint(
        base="fapi",
        http_method="GET",
        path="/fapi/v1/openInterest",  # ✅ Doğru yol /v1/openInterest
        signed=False,
        weight=1,
        parameters=[{"name": "symbol", "required": True}],
        cache="1m",
        return_type="dict"  # Tek bir dict döner
    ),

    "open_interest_statistics": Endpoint( # ❌,zorunlu DB’ye yazdırmak ❌openInterestHist
        base="fapi",
        http_method="GET",
        path="/futures/data/openInterestStatistics",  # ✅ Doğru yol
        signed=False,
        weight=1,
        parameters=[
            {"name": "symbol", "required": True},
            {"name": "period", "required": True},     # 5m,15m,30m,1h,2h,4h,6h,12h,1d
            {"name": "limit", "required": False},
            {"name": "startTime", "required": False},
            {"name": "endTime", "required": False}
        ],
        cache="1m",
        return_type="list"
    ),


    "continuous_klines": Endpoint(
        base="futures",
        http_method="GET",
        path="/fapi/v1/continuousKlines",
        signed=False,
        weight=1,
        parameters=[
            {"name": "pair", "required": True},
            {"name": "contractType", "required": True},
            {"name": "interval", "required": True},
            {"name": "limit", "required": False}
        ],
        cache="15s",
        return_type="list"
    ),
    "fundingRate": Endpoint(
        base="futures",
        http_method="GET",
        path="/fapi/v1/fundingRate",
        signed=False,
        weight=1,
        parameters=[
            {"name": "symbol", "required": False},
            {"name": "limit", "required": False},
            {"name": "startTime", "required": False},
            {"name": "endTime", "required": False}
        ],
        cache="30s",
        return_type="list"
    ),    
    "open_interest": Endpoint(
        base="futures",
        http_method="GET", 
        path="/fapi/v1/openInterest",
        signed=False,
        weight=1,
        parameters=[
            {"name": "symbol", "required": True}
        ],
        cache="10s",
        return_type="dict"
    ),
    "futures_ticker_24hr": Endpoint(
        base="futures",
        http_method="GET",
        path="/fapi/v1/ticker/24hr",
        signed=False,
        weight=1,
        parameters=[],
        cache="10s",
        return_type="auto"
    ),
    "force_orders": Endpoint(
        base="futures",
        http_method="GET",
        path="/fapi/v1/forceOrders",
        signed=False,
        weight=20,
        parameters=[
            {"name": "symbol", "required": False},
            {"name": "limit", "required": False}
        ],
        cache="20s",
        return_type="list"
    ),
    "long_short_ratio": Endpoint(
        base="futures",
        http_method="GET",
        path="/futures/data/globalLongShortAccountRatio",
        signed=False,
        weight=5,
        parameters=[
            {"name": "symbol", "required": True},
            {"name": "period", "required": False},
            {"name": "limit", "required": False}
        ],
        cache="300s",
        return_type="list"
    ),
    "top_long_short_ratio": Endpoint(
        base="futures",
        http_method="GET",
        path="/futures/data/topLongShortPositionRatio",
        signed=False,
        weight=1,
        parameters=[
            {"name": "symbol", "required": True},
            {"name": "period", "required": True},   # 5m, 15m, 30m, 1h, 4h, 1d
            {"name": "limit", "required": False}    # default 30, max 500
        ],
        cache="15s",
        return_type="list"
    ),

    "taker_buy_sell_volume": Endpoint(
        base="futures",
        http_method="GET",
        path="/futures/data/takerlongshortRatio",
        signed=False,
        weight=10,
        parameters=[
            {"name": "symbol", "required": True},
            {"name": "limit", "required": False},
            {"name": "period", "required": True, "default": "1h"}
        ],
        cache="60s",
        return_type="list"
    ),

}


class MapLoader:
    """Hard-coded endpoint tanımlarını yönetir."""

    def __init__(self):
        self.maps = {"hardcoded": ENDPOINTS}
        logger.info("✅ Hard-coded endpoint map loaded")

    def get_endpoint(self, name: str) -> Optional[Dict[str, Any]]:
        """Endpoint bilgilerini dictionary formatında döndür"""
        endpoint = ENDPOINTS.get(name)
        if not endpoint:
            logger.warning(f"Endpoint not found: {name}")
            return None
            
        # Endpoint'i YAML formatına uygun dictionary'ye çevir
        return {
            "key": name,
            "base": endpoint.base,
            "http_method": endpoint.http_method,
            "path": endpoint.path,
            "signed": endpoint.signed,
            "weight": endpoint.weight,
            "parameters": endpoint.parameters,  # ✅ Artık zaten dict formatında
            "cache_ttl": int(endpoint.cache.replace('s', '')) if endpoint.cache and endpoint.cache.endswith('s') else 5,
            "return_type": endpoint.return_type
        }


class BinanceAggregator:
    _instance = None
    _init_lock = asyncio.Lock()

    def __init__(self, config: Optional["BinanceConfig"] = None):
        if hasattr(self, "_initialized") and self._initialized:
            raise RuntimeError("BinanceAggregator singleton already initialized")
        
        # 🔒 Basit kullanıcı lock sistemi
        self._locks_lock = asyncio.Lock()
        self._user_locks: Dict[int, asyncio.Lock] = {}
        
        # 📦 Core initialization
        self.map_loader = MapLoader()
        self.config = config or BinanceConfig()
        self.sessions = UserSessionManager(ttl_minutes=60)

        self._cleanup_task: Optional[asyncio.Task] = None
        self._stop_event = asyncio.Event()

        # Global API key
        self.global_api_key = os.getenv("BINANCE_API_KEY")
        self.global_api_secret = os.getenv("BINANCE_API_SECRET")
        self.api_manager = None

        # ✅ Basit rate limiter
        self.rate_limiter = UserAwareRateLimiter()

        self._initialized = True
        logger.info(" BinanceAggregator initialized with hard-coded endpoints")

    @property
    def metrics_manager(self):
        return BinanceMetrics.get_instance()

    async def initialize_managers(self):
        """Manager'ları async olarak initialize et"""
        try:
            self.key_manager = await APIKeyManager.get_instance()
            self.api_manager = self.key_manager
            await self.key_manager.ensure_db_initialized()
            logger.info(" All managers initialized")
        except Exception as e:
            logger.error(f" Manager initialization failed: {e}")
            raise

    @classmethod
    async def get_instance(cls, config: Optional["BinanceConfig"] = None) -> "BinanceAggregator":
        """Async singleton getter"""
        async with cls._init_lock:
            if cls._instance is None:
                cls._instance = cls(config)
                await cls._instance.initialize_managers()
                    
            return cls._instance

    async def get_user_credentials(self, user_id: Optional[int] = None) -> Tuple[str, str]:
        """Kullanıcı credential'larını al"""
        # 1. Kişisel API (sadece gerçek user ID'ler için)
        if user_id is not None and user_id > 0:
            creds = await self.api_manager.get_apikey(user_id)
            if creds:
                logger.info(f"Using personal API for user {user_id}")
                return creds

        # 2. Global API fallback
        if not self.global_api_key or not self.global_api_secret:
            raise RuntimeError("No Binance API credentials available")

        logger.info("Using global API credentials")
        return self.global_api_key, self.global_api_secret
        
    async def _get_or_create_session(self, user_id: int, api_key: str, secret_key: str):
        """Kullanıcı session'ını oluştur veya getir"""
        async with self._locks_lock:
            if user_id not in self._user_locks:
                self._user_locks[user_id] = asyncio.Lock()
        
        user_lock = self._user_locks[user_id]
        
        async with user_lock:
            session = await self.sessions.get_session(user_id)
            if not session:
                http_client = BinanceHTTPClient(
                    api_key=api_key, 
                    secret_key=secret_key,
                    user_id=user_id
                )
                await self.sessions.add_session(user_id, http_client)
                session = await self.sessions.get_session(user_id)
            return session
            
    async def _set_request_context(self, user_id: int, endpoint_name: str):
        """Set context for the current request"""
        ContextAwareLogger.set_user_context(user_id)
        ContextAwareLogger.set_request_context(str(uuid.uuid4())[:8], endpoint_name)

    async def _execute_with_rate_limit(self, user_id: int, endpoint_name: str, operation, estimated_weight: int = 1):
        """Basit rate limit entegrasyonu"""
        if not await self.rate_limiter.acquire(user_id, endpoint_name, estimated_weight):
            from utils.binance_api.binance_exceptions import BinanceRateLimitError
            raise BinanceRateLimitError(f"Rate limit exceeded for user {user_id}")
        
        return await operation()

    # =======================================================
    # 🔹 METRICS KAYIT FONKSİYONLARI 
    # =======================================================
    async def _record_metrics_success(self, endpoint_name: str, response_time: float, weight_used: int = 1):
        """Başarılı istek için metrics kaydet"""
        try:
            await record_request(
                endpoint=endpoint_name,
                response_time=response_time,
                status_code=200,
                weight_used=weight_used
            )
        except Exception as e:
            logger.warning(f"Metrics recording failed: {e}")

    async def _record_metrics_error(self, endpoint_name: str, response_time: float, error: Exception, weight_used: int = 1):
        """Hatalı istek için metrics kaydet"""
        try:
            await record_request(
                endpoint=endpoint_name,
                response_time=response_time,
                error=error,
                weight_used=weight_used
            )
        except Exception as e:
            logger.warning(f"Error metrics recording failed: {e}")

    async def _record_retry_metrics(self, endpoint_name: str, attempt: int):
        """Retry için metrics kaydet"""
        try:
            await record_retry(endpoint_name, attempt)
        except Exception as e:
            logger.warning(f"Retry metrics recording failed: {e}")

    # =======================================================
    # 🔹 ANA SORGULAMA METODLARI
    # =======================================================
    async def get_public_data(self, endpoint_name: str, **params) -> Any:
        """Public data sorgusu"""
        return await self._get_data_internal(endpoint_name, None, **params)
        
    async def get_private_data(self, user_id: int, endpoint_name: str, **params) -> Any:
        """Private data sorgusu"""
        return await self._get_data_internal(endpoint_name, user_id, **params)

    async def get_data(self, endpoint_name: str, user_id: Optional[int] = None, **params) -> Any:
        """Akıllı metod - public/private otomatik seçim"""
        if user_id is not None:
            return await self.get_private_data(user_id, endpoint_name, **params)
        else:
            return await self.get_public_data(endpoint_name, **params)

    def _is_public_endpoint(self, endpoint_name: str) -> bool:
        """Endpoint'in public olup olmadığını kontrol et"""
        endpoint = ENDPOINTS.get(endpoint_name)
        if not endpoint:
            return False
        return not endpoint.signed
        
    @monitor_performance("get_data", warning_threshold=2.5)
    async def _get_data_internal(self, endpoint_name: str, user_id: Optional[int], **params) -> Any:
        """Ortak logic"""
        effective_user_id = user_id if user_id is not None else 0
        start_time = asyncio.get_event_loop().time()
        
        await self._set_request_context(effective_user_id, endpoint_name)
        
        try:
            # Endpoint kontrolü
            endpoint = self.map_loader.get_endpoint(endpoint_name)
            if not endpoint:
                raise ValueError(f"Endpoint not found: {endpoint_name}")

            # Public endpoint kontrolü
            if user_id is None and not self._is_public_endpoint(endpoint_name):
                raise PermissionError(f"Private endpoint {endpoint_name} requires user authentication")

            # Security audit
            if not await security_auditor.audit_request(user_id, endpoint_name, params):
                raise PermissionError(f"Security audit failed: {endpoint_name}")

            # API credentials
            api_key, secret_key = await self.get_user_credentials(user_id)

            # Session
            session = await self._get_or_create_session(effective_user_id, api_key, secret_key)
            if not session:
                raise RuntimeError("User session creation failed")

            # Rate limit + API call
            async def api_operation():
                return await self._call_direct_endpoint(session.http_client, endpoint, **params)

            result = await self._execute_with_rate_limit(
                user_id=effective_user_id,
                endpoint_name=endpoint_name,
                operation=api_operation,
                estimated_weight=endpoint.get("weight", 1)
            )
            
            # Başarılı metrics kaydı
            response_time = asyncio.get_event_loop().time() - start_time
            await self._record_metrics_success(
                endpoint_name, 
                response_time, 
                endpoint.get("weight", 1)
            )
            
            return result
                
        except Exception as e:
            # Hata metrics kaydı
            response_time = asyncio.get_event_loop().time() - start_time
            await self._record_metrics_error(
                endpoint_name,
                response_time,
                e,
                endpoint.get("weight", 1) if endpoint else 1
            )
            logger.error(f"Data request failed: user={effective_user_id}, endpoint={endpoint_name}, error={e}")
            raise
        finally:
            ContextAwareLogger.clear_context()

    def _get_binance_client(self, http_client):
        """Binance client oluştur"""
        return DirectBinanceClient(http_client)

    async def _call_direct_endpoint(self, http_client: "BinanceHTTPClient", endpoint_config: Dict[str, Any], **params) -> Any:
        binance_client = self._get_binance_client(http_client)
        return await binance_client.call_endpoint(endpoint_config, **params)

    # =======================================================
    # 🔹 TEMİZLİK METODLARI
    # =======================================================   
    async def start_background_tasks(self, interval: int = 300) -> None:
        """Basit background tasks"""
        if not self._cleanup_task:
            self._cleanup_task = asyncio.create_task(self._cleanup_loop(interval))

    async def stop_background_tasks(self) -> None:
        """Stop background tasks"""
        if self._cleanup_task:
            self._stop_event.set()
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass

    async def _cleanup_loop(self, interval: int) -> None:
        """Basit cleanup loop"""
        while not self._stop_event.is_set():
            try:
                await self.sessions.cleanup_expired_sessions()
                if self.key_manager:
                    await self.key_manager.cleanup_cache()
            except Exception as e:
                logger.error(f"Cleanup error: {e}")
            finally:
                ContextAwareLogger.clear_context()
            await asyncio.sleep(interval)
    
    async def cleanup(self):
        """Tüm kaynakları temizle"""
        await self.stop_background_tasks()
        
        # Session'ları temizle - doğru metod ismi
        if hasattr(self, 'sessions'):
            await self.sessions.cleanup_expired_sessions()  # ✅ Doğru metod
        
        # HTTP client'ları temizle
        if hasattr(self, '_user_locks'):
            for user_id in list(self._user_locks.keys()):
                session = await self.sessions.get_session(user_id)
                if session and hasattr(session, 'http_client'):
                    await session.http_client.close()
        
        # Lock'ları temizle
        if hasattr(self, '_user_locks'):
            self._user_locks.clear()
        
        logger.info(" BinanceAggregator cleanup completed")
# end