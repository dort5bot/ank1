# utils/binance_api/binance_circuit_breaker.py
# v104 Geliştirilmiş Circuit Breaker implementation.

from __future__ import annotations

import asyncio
import time
import logging
import inspect
from asyncio import timeout, TimeoutError
from dataclasses import dataclass
from collections import OrderedDict
from typing import Any, Awaitable, Callable, Optional, Dict, Tuple, List

# Kullanıcı tarafı özel exception'larınızdan biri
from .binance_exceptions import BinanceCircuitBreakerError

logger = logging.getLogger(__name__)

# APIKeyManager import'u (opsiyonel - eğer yoksa çalışmaya devam eder)
try:
    from utils.apikey_manager import APIKeyManager
except ImportError:
    APIKeyManager = None
    print("⚠️ APIKeyManager not available - circuit breaker will work without user-specific features")




@dataclass
class CircuitBreakerState:
    failures: int = 0
    last_failure_time: float = 0.0
    state: str = "CLOSED"  # "CLOSED", "OPEN", "HALF_OPEN"


# Tipler
AsyncCallable = Callable[..., Awaitable[Any]]
SyncOrAsyncCallable = Callable[..., Any]
FailurePredicate = Callable[[Exception], bool]
HookCallable = Callable[[Dict[str, Any]], None]


class CircuitBreaker:
    def __init__(
        self,
        failure_threshold: int = 5,
        reset_timeout: float = 60.0,
        half_open_timeout: float = 30.0,
        max_half_open_calls: int = 1,
        name: str = "binance_circuit_breaker",
        failure_predicate: Optional[FailurePredicate] = None,
        on_open: Optional[HookCallable] = None,
        on_half_open: Optional[HookCallable] = None,
        on_close: Optional[HookCallable] = None,
        on_failure: Optional[HookCallable] = None,
        on_success: Optional[HookCallable] = None,
        rate_limit_weight_threshold: int = 1000,  # 1 dakikada max weight
        rate_limit_window: int = 60,  # 60 saniye
    ):
        """
        Args:
            failure_threshold: arka arkaya başarısız çağrı sayısı -> OPEN
            reset_timeout: OPEN durumundan HALF_OPEN'a geçiş için bekleme (saniye)
            half_open_timeout: HALF_OPEN durumunda test süresi (saniye)
            max_half_open_calls: HALF_OPEN durumunda eş zamanlı izin verilen test çağrı sayısı
            name: isim (log için)
            failure_predicate: Exception -> bool. True ise o exception failure olarak sayılır.
            on_open/on_half_open/on_close/on_failure/on_success: opsiyonel hook fonksiyonları.
        """
        self.failure_threshold = int(failure_threshold)
        self.reset_timeout = float(reset_timeout)
        self.half_open_timeout = float(half_open_timeout)
        self.max_half_open_calls = int(max_half_open_calls)
        self.name = name

        self.state = CircuitBreakerState()
        self._lock = asyncio.Lock()
        
        # Rate limiting alanlar
        self.rate_limit_weight_threshold = rate_limit_weight_threshold
        self.rate_limit_window = rate_limit_window
        self._request_weights: List[Tuple[float, int]] = []  # (timestamp, weight)
        self._rate_limit_lock = asyncio.Lock()
        
        
        # half-open test concurrency kontrolü
        self._half_open_semaphore = asyncio.Semaphore(self.max_half_open_calls)

        # Hooks
        self.failure_predicate = failure_predicate or self._default_failure_predicate
        self.on_open = on_open
        self.on_half_open = on_half_open
        self.on_close = on_close
        self.on_failure = on_failure
        self.on_success = on_success
        self._state_history: List[Dict[str, Any]] = []
        self._max_state_history = 100  # Son 100 state değişikliğini tut

        logger.info(f"✅ CircuitBreaker '{self.name}' initialized")


    async def check_rate_limit(self, estimated_weight: int = 1) -> bool:
        """
        Rate limit kontrolü yapar. 
        Eğer limit aşılıyorsa False döner.
        """
        async with self._rate_limit_lock:
            now = time.time()
            # Eski kayıtları temizle
            self._request_weights = [
                (ts, weight) for ts, weight in self._request_weights 
                if now - ts < self.rate_limit_window
            ]
            
            # Toplam weight'i hesapla
            total_weight = sum(weight for _, weight in self._request_weights)
            
            if total_weight + estimated_weight > self.rate_limit_weight_threshold:
                return False
            
            # Yeni isteği kaydet
            self._request_weights.append((now, estimated_weight))
            return True

    async def get_rate_limit_metrics(self) -> Dict[str, Any]:
        """Rate limit metrics'larını döndürür"""
        async with self._rate_limit_lock:
            now = time.time()
            self._request_weights = [
                (ts, weight) for ts, weight in self._request_weights 
                if now - ts < self.rate_limit_window
            ]
            
            total_weight = sum(weight for _, weight in self._request_weights)
            window_usage = (total_weight / self.rate_limit_weight_threshold) * 100
            
            return {
                "current_weight": total_weight,
                "weight_limit": self.rate_limit_weight_threshold,
                "window_usage_percent": window_usage,
                "requests_in_window": len(self._request_weights),
                "window_seconds": self.rate_limit_window
            }




    # ---------- Public API ----------
    
    async def execute(
        self, 
        func: SyncOrAsyncCallable, 
        *args, 
        estimated_weight: int = 1,  # Yeni parametre
        **kwargs
    ) -> Any:
        """
        Execute the callable under circuit breaker protection.

        Args:
            estimated_weight: Binance API weight estimate for rate limiting
        """
        # Rate limit kontrolü
        if not await self.check_rate_limit(estimated_weight):
            raise BinanceCircuitBreakerError(
                f"Rate limit exceeded for circuit breaker '{self.name}'. "
                f"Try again in {self.rate_limit_window} seconds."
            )
            
        
        # İlk state kontrolü (kısa, lock'un gereksiz tutulmasını engellemek için)
        await self._pre_execute_check()

        # Eğer fonksiyon sync ise executor'da çalıştır (non-blocking)
        is_coro = inspect.iscoroutinefunction(func)
        try:
            if is_coro:
                result = await self._invoke_async(func, *args, **kwargs)
            else:
                # sync fonksiyon -> executor
                loop = asyncio.get_running_loop()
                result = await loop.run_in_executor(None, lambda: func(*args, **kwargs))
            # başarı işlemleri
            await self.record_success()
            return result
        except Exception as ex:
            # hata sayılacak mı kontrolü
            is_failure = False
            try:
                is_failure = self.failure_predicate(ex)
            except Exception as pred_ex:
                # predicate kendi içinde hata verirse, güvenlik nedeniyle "failure" say
                logger.exception("Failure predicate raised an exception; counting as failure", exc_info=pred_ex)
                is_failure = True

            if is_failure:
                await self.record_failure(ex)
            # Hook: yine de kullanıcıya hatayı fırlat
            raise



    async def _invoke_async(self, coro_func: AsyncCallable, *args, **kwargs) -> Any:
        """
        İç çağırı: HALF_OPEN durumunda semaphore ile test sayısını sınırla.
        """
        # Eğer HALF_OPEN ise semaphore'ya bağlan (max_half_open_calls)
        if self.is_half_open():
            # try to acquire a slot for testing
            acquired = await self._half_open_semaphore.acquire()  # acquires
            try:
                # çağrıyı yap
                return await coro_func(*args, **kwargs)
            finally:
                # serbest bırak
                self._half_open_semaphore.release()
        else:
            # normal durumda doğrudan çağır
            return await coro_func(*args, **kwargs)

    # ---------- State management ----------
    async def _pre_execute_check(self) -> None:
        """Check current state and possibly transition to HALF_OPEN or raise."""
        async with self._lock:
            now = time.time()
            st = self.state.state
            if st == "OPEN":
                elapsed = now - self.state.last_failure_time
                if elapsed >= self.reset_timeout:
                    # OPEN -> HALF_OPEN
                    self.state.state = "HALF_OPEN"
                    # reset failures for half-open testing but keep last_failure_time for metrics
                    # not resetting failures to 0 allows us to keep historical count if needed
                    logger.warning(f"⚠️ CircuitBreaker '{self.name}' transitioned OPEN -> HALF_OPEN (elapsed {elapsed:.1f}s)")
                    if self.on_half_open:
                        self._safe_call_hook(self.on_half_open, self.get_state())
                else:
                    retry_in = max(0.0, self.reset_timeout - elapsed)
                    raise BinanceCircuitBreakerError(
                        f"Circuit breaker '{self.name}' is OPEN. Retry in {retry_in:.1f}s"
                    )
            
    # if HALF_OPEN we allow execution but concurrency limited in _invoke_async

#
    async def record_failure(self, error: Optional[Exception] = None) -> None:
        """Record a failure and update state if threshold reached."""
        async with self._lock:
            previous_state = self.state.state
            self.state.failures += 1
            self.state.last_failure_time = time.time()

            # Eğer HALF_OPEN ise, bir başarısızlık direkt OPEN yapar
            if self.state.state == "HALF_OPEN":
                self.state.state = "OPEN"
                self._record_state_change(previous_state, "OPEN", f"HALF_OPEN test failed: {error}")
                logger.error(f"❌ CircuitBreaker '{self.name}' moved HALF_OPEN -> OPEN due to failure (error: {error})")
                if self.on_open:
                    self._safe_call_hook(self.on_open, self.get_state())
            elif self.state.state == "CLOSED" and self.state.failures >= self.failure_threshold:
                self.state.state = "OPEN"
                self._record_state_change(previous_state, "OPEN", f"Failure threshold reached: {self.state.failures}")
                logger.error(f"❌ CircuitBreaker '{self.name}' opened (failures={self.state.failures})")
                if self.on_open:
                    self._safe_call_hook(self.on_open, self.get_state())
            
            # çağrıya özel hook
            if self.on_failure:
                self._safe_call_hook(self.on_failure, {"error": error, **self.get_state()})

    async def record_success(self) -> None:
        """Record a success: HALF_OPEN -> CLOSED (reset), or decrement/reset failures in CLOSED."""
        async with self._lock:
            previous_state = self.state.state
            if self.state.state == "HALF_OPEN":
                # Başarı: circuit'i kapat
                self.state.failures = 0
                self.state.state = "CLOSED"
                self.state.last_failure_time = 0.0
                self._record_state_change(previous_state, "CLOSED", "HALF_OPEN test succeeded")
                logger.info(f"✅ CircuitBreaker '{self.name}' HALF_OPEN -> CLOSED (successful test)")
                if self.on_close:
                    self._safe_call_hook(self.on_close, self.get_state())
            else:
                # Normal başarı: failures sıfırlanabilir veya azaltılabilir (burada sıfırlıyoruz)
                self.state.failures = 0
                if self.on_success:
                    self._safe_call_hook(self.on_success, self.get_state())
                    



    async def reset(self) -> None:
        """Manual reset to CLOSED."""
        async with self._lock:
            self.state.failures = 0
            self.state.state = "CLOSED"
            self.state.last_failure_time = 0.0
            logger.info(f"🔄 CircuitBreaker '{self.name}' manually reset to CLOSED")
            if self.on_close:
                self._safe_call_hook(self.on_close, self.get_state())

    async def force_open(self) -> None:
        async with self._lock:
            self.state.state = "OPEN"
            self.state.last_failure_time = time.time()
            logger.warning(f"⚠️ CircuitBreaker '{self.name}' manually forced OPEN")
            if self.on_open:
                self._safe_call_hook(self.on_open, self.get_state())


    # ---------- State History Management ----------
    def _record_state_change(self, from_state: str, to_state: str, reason: str = "") -> None:
        """Record state changes for monitoring and debugging."""
        state_change = {
            "timestamp": time.time(),
            "from_state": from_state,
            "to_state": to_state,
            "reason": reason,
            "failures": self.state.failures,
            "last_failure_time": self.state.last_failure_time,
        }
        
        self._state_history.append(state_change)
        
        # Keep only last N entries
        if len(self._state_history) > self._max_state_history:
            self._state_history.pop(0)

    def get_state_history(self) -> List[Dict[str, Any]]:
        """Get the state change history for monitoring."""
        return self._state_history.copy()

    def clear_state_history(self) -> None:
        """Clear state history."""
        self._state_history.clear()


#0
    # ---------- Queries ----------
    def is_closed(self) -> bool:
        return self.state.state == "CLOSED"

    def is_open(self) -> bool:
        return self.state.state == "OPEN"

    def is_half_open(self) -> bool:
        return self.state.state == "HALF_OPEN"

    def get_state(self) -> Dict[str, Any]:
        return {
            "state": self.state.state,
            "failures": self.state.failures,
            "last_failure_time": self.state.last_failure_time,
            "time_since_last_failure": (time.time() - self.state.last_failure_time) if self.state.last_failure_time else 0.0,
            "name": self.name,
        }

    # ---------- Metrics ---------- 
    def get_metrics(self) -> Dict[str, Any]:
        """Get circuit breaker metrics for monitoring."""
        now = time.time()
        state_info = self.get_state()
        
        # Daha güvenli semaphore value kontrolü
        semaphore_value = None
        try:
            # Semaphore değerine güvenli şekilde erişim
            semaphore_value = self._half_open_semaphore._value
        except (AttributeError, ValueError):
            pass
        
        uptime_start = self._state_history[0]["timestamp"] if self._state_history else now
        uptime_seconds = now - uptime_start
        
        return {
            **state_info,
            "total_state_changes": len(self._state_history),
            "recent_state_changes": self._state_history[-10:] if self._state_history else [],
            "uptime_seconds": uptime_seconds,
            "half_open_semaphore_value": semaphore_value,
            "max_half_open_calls": self.max_half_open_calls,
        }




    # ---------- Utilities ----------  

    @staticmethod
    def _default_failure_predicate(exc: Exception) -> bool:
        """
        Geliştirilmiş failure predicate: Binance-specific hataları işler
        """
        # Binance API exception'ları
        if hasattr(exc, 'code'):
            code = exc.code
            # Rate limit hataları
            if code in [-1003, -1005, -1006, -1007, -1015]:
                return True
            # Server hataları
            if code <= -1000 and code > -2000:
                return True
            # Authentication hataları (circuit breaker'i açma)
            if code in [-2014, -2015, -1016]:
                return False
        
        # HTTP status code kontrolü
        status = getattr(exc, "status_code", None)
        if isinstance(status, int):
            # 5xx ve 429'ı failure say
            if status >= 500 or status == 429:
                return True
            # 401 (unauthorized) -> authentication hatası
            if status == 401:
                return False
            # Diğer 4xx'ler genelde failure sayılmaz
            return False
        
        # Timeout hataları
        if isinstance(exc, (asyncio.TimeoutError, TimeoutError)):
            return True
            
        # Connection hataları
        if isinstance(exc, (ConnectionError, OSError)):
            return True
            
        # Default: say
        return True
        

    def _safe_call_hook(self, hook: HookCallable, payload: Dict[str, Any]) -> None:
        try:
            hook(payload)
        except Exception:
            logger.exception("CircuitBreaker hook raised an exception", exc_info=True)



# TEK BİR MANAGER
# Ana manager olarak kalacak
# İçinde hem API key hem user+endpoint desteği olacak
class CircuitBreakerManager:
    """
    Unified Circuit Breaker Manager - hem API key hem user+endpoint bazlı yönetim.
    """
    
    def __init__(
        self,
        failure_threshold: int = 5,
        reset_timeout: float = 60.0,
        half_open_timeout: float = 30.0,
        max_half_open_calls: int = 1,
        failure_predicate: Optional[FailurePredicate] = None,
        on_open: Optional[HookCallable] = None,
        on_half_open: Optional[HookCallable] = None,
        on_close: Optional[HookCallable] = None,
        on_failure: Optional[HookCallable] = None,
        on_success: Optional[HookCallable] = None,
        max_cache_size: int = 1000,
        ttl_seconds: int = 3600,
        use_api_keys: bool = True,  # 🔑 API key bazlı yönetim aktif mi?
    ):
        """
        Args:
            use_api_keys: True ise API key bazlı, False ise user+endpoint bazlı çalışır
            Diğer parametreler her iki mod için ortak
        """
        self._lock = asyncio.Lock()
        self.use_api_keys = use_api_keys
        
        # User+endpoint bazlı breaker'lar için
        self._user_breaker_map: OrderedDict[Tuple[str, str], Tuple[CircuitBreaker, float]] = OrderedDict()
        
        # API key bazlı breaker'lar için  
        self._api_key_breaker_map: Dict[str, CircuitBreaker] = {}
        self._api_key_access_times: Dict[str, float] = {}

        # Ortak config
        self._config = {
            "failure_threshold": failure_threshold,
            "reset_timeout": reset_timeout,
            "half_open_timeout": half_open_timeout,
            "max_half_open_calls": max_half_open_calls,
            "failure_predicate": failure_predicate,
            "on_open": on_open,
            "on_half_open": on_half_open,
            "on_close": on_close,
            "on_failure": on_failure,
            "on_success": on_success,
        }

        self.max_cache_size = max_cache_size
        self.ttl_seconds = ttl_seconds

    # 🔄 ORTAK CLEANUP METODLARI
    
    async def _cleanup_expired_api_keys(self):
        """API key breaker'ları için TTL temizleme"""
        now = time.time()
        keys_to_delete = []
        
        async with self._lock:
            for api_key, last_access in self._api_key_access_times.items():
                if now - last_access > self.ttl_seconds:
                    keys_to_delete.append(api_key)
            
            for key in keys_to_delete:
                self._api_key_breaker_map.pop(key, None)
                self._api_key_access_times.pop(key, None)
            
            # LRU cleanup
            while len(self._api_key_breaker_map) > self.max_cache_size:
                oldest_key = min(self._api_key_access_times.items(), key=lambda x: x[1])[0]
                self._api_key_breaker_map.pop(oldest_key, None)
                self._api_key_access_times.pop(oldest_key, None)

    async def _cleanup_expired_users(self):
        """User breaker'ları için TTL temizleme"""
        now = time.time()
        keys_to_delete = []

        # TTL kontrolü
        for key, (_, last_access) in self._user_breaker_map.items():
            if now - last_access > self.ttl_seconds:
                keys_to_delete.append(key)

        for key in keys_to_delete:
            del self._user_breaker_map[key]

        # LRU kontrolü
        while len(self._user_breaker_map) > self.max_cache_size:
            self._user_breaker_map.popitem(last=False)

    async def _cleanup_all(self):
        """Tüm cache'leri temizle"""
        await self._cleanup_expired_api_keys()
        await self._cleanup_expired_users()

    # 🔑 API KEY BAZLI METODLAR
    
    async def get_breaker_for_api_key(self, api_key: str) -> CircuitBreaker:
        """
        API key için circuit breaker döndürür.
        Artık ASYNC - thread safe
        """
        await self._cleanup_expired_api_keys()
        
        # Kısa hash kullan (güvenlik için full key saklama)
        key_hash = f"api_{hash(api_key) & 0xFFFFFFFF}"
        
        async with self._lock:
            if key_hash in self._api_key_breaker_map:
                self._api_key_access_times[key_hash] = time.time()
                return self._api_key_breaker_map[key_hash]
            
            # Yeni breaker oluştur
            name = f"apikey_cb:{key_hash}"
            breaker = CircuitBreaker(name=name, **self._config)
            self._api_key_breaker_map[key_hash] = breaker
            self._api_key_access_times[key_hash] = time.time()
            
            return breaker

    async def execute_with_api_key(
        self, 
        api_key: str, 
        func: SyncOrAsyncCallable, 
        *args, **kwargs
    ) -> Any:
        """
        API key ile circuit breaker koruması altında fonksiyon çalıştırır.
        """
        breaker = await self.get_breaker_for_api_key(api_key)
        return await breaker.execute(func, *args, **kwargs)

    # 👤 USER BAZLI METODLAR
    
    async def get_breaker_for_user(self, user_id: str, endpoint: str = "default") -> CircuitBreaker:
        """
        Kullanıcı+endpoint bazında breaker objesini döner.
        """
        key = (user_id, endpoint)
        async with self._lock:
            await self._cleanup_expired_users()

            if key in self._user_breaker_map:
                # LRU için order güncelle
                breaker, _ = self._user_breaker_map.pop(key)
                self._user_breaker_map[key] = (breaker, time.time())
                return breaker

            # Yeni breaker oluştur
            name = f"cb:{user_id}:{endpoint}"
            breaker = CircuitBreaker(name=name, **self._config)
            self._user_breaker_map[key] = (breaker, time.time())
            return breaker

    async def execute_with_user(
        self, 
        user_id: str, 
        endpoint: str, 
        func: Callable[..., Any], 
        *args, **kwargs
    ):
        """
        User+endpoint ile circuit breaker koruması altında fonksiyon çalıştırır.
        """
        breaker = await self.get_breaker_for_user(user_id, endpoint)
        return await breaker.execute(func, *args, **kwargs)

    # 🎯 AKILLI EXECUTE - OTOMATİK MOD SEÇİMİ
    
    async def execute(
        self, 
        identifier: str, 
        endpoint: str = "default",
        func: Optional[SyncOrAsyncCallable] = None,
        *args, **kwargs
    ) -> Any:
        """
        Akıllı execute: use_api_keys ayarına göre otomatik mod seçer.
        
        Args:
            identifier: API key (use_api_keys=True) veya user_id (use_api_keys=False)
            endpoint: Sadece user modunda kullanılır
        """
        if self.use_api_keys:
            return await self.execute_with_api_key(identifier, func, *args, **kwargs)
        else:
            return await self.execute_with_user(identifier, endpoint, func, *args, **kwargs)

    async def execute_with_user_id(
        self, 
        user_id: int, 
        endpoint: str, 
        func: SyncOrAsyncCallable, 
        *args, **kwargs
    ) -> Any:
        """
        User ID ile circuit breaker - APIKeyManager entegrasyonlu.
        Backward compatibility için korunuyor.
        """
        if self.use_api_keys and APIKeyManager is not None:
            try:
                # API key'i al
                api_manager = APIKeyManager.get_instance()
                creds = await api_manager.get_apikey(user_id)
                if not creds:
                    raise ValueError(f"API key not found for user {user_id}")
                    
                api_key, secret_key = creds
                return await self.execute_with_api_key(api_key, func, *args, **kwargs)
            except Exception as e:
                logger.error(f"Error in execute_with_user_id: {e}")
                # Fallback to user mode
                return await self.execute_with_user(str(user_id), endpoint, func, *args, **kwargs)
        else:
            # Direct user mode
            return await self.execute_with_user(str(user_id), endpoint, func, *args, **kwargs)

    # 🎛️ MANAGEMENT METODLARI (Her İki Mod İçin)
    
    async def force_open(self, identifier: str, endpoint: str = "default"):
        """Breaker'ı OPEN state'e zorla"""
        if self.use_api_keys:
            breaker = await self.get_breaker_for_api_key(identifier)
        else:
            breaker = await self.get_breaker_for_user(identifier, endpoint)
        await breaker.force_open()

    async def reset(self, identifier: str, endpoint: str = "default"):
        """Breaker'ı resetle"""
        if self.use_api_keys:
            breaker = await self.get_breaker_for_api_key(identifier)
        else:
            breaker = await self.get_breaker_for_user(identifier, endpoint)
        await breaker.reset()

    async def remove(self, identifier: str, endpoint: str = "default"):
        """Breaker'ı cache'den kaldır"""
        async with self._lock:
            if self.use_api_keys:
                key_hash = f"api_{hash(identifier) & 0xFFFFFFFF}"
                self._api_key_breaker_map.pop(key_hash, None)
                self._api_key_access_times.pop(key_hash, None)
            else:
                key = (identifier, endpoint)
                self._user_breaker_map.pop(key, None)

    async def cleanup(self):
        """Tüm cache'leri temizle"""
        await self._cleanup_all()

    # 📊 METRICS VE MONITORING
    
    async def get_api_key_metrics(self) -> Dict[str, Dict[str, Any]]:
        """API key breaker'larının metrics'larını döndürür"""
        await self._cleanup_expired_api_keys()
        return {
            key: breaker.get_metrics() 
            for key, breaker in self._api_key_breaker_map.items()
        }

    async def get_user_metrics(self) -> Dict[str, Dict[str, Any]]:
        """User breaker'larının metrics'larını döndürür"""
        await self._cleanup_expired_users()
        return {
            f"{user_id}:{endpoint}": breaker.get_metrics()
            for (user_id, endpoint), (breaker, _) in self._user_breaker_map.items()
        }

    async def get_metrics(self) -> Dict[str, Dict[str, Any]]:
        """
        Tüm breaker'ların metrics'larını topla.
        """
        api_metrics = await self.get_api_key_metrics()
        user_metrics = await self.get_user_metrics()
        return {**api_metrics, **user_metrics}

    async def get_breaker_count(self) -> int:
        """Toplam breaker sayısını döner."""
        async with self._lock:
            return len(self._api_key_breaker_map) + len(self._user_breaker_map)

    async def force_close_all(self) -> None:
        """Tüm breaker'ları CLOSED state'e zorla."""
        async with self._lock:
            # API key breaker'ları
            for breaker in self._api_key_breaker_map.values():
                previous_state = breaker.state.state
                if previous_state != "CLOSED":
                    breaker._record_state_change(previous_state, "CLOSED", "Manual force close all")
                await breaker.reset()
            
            # User breaker'ları
            for (user_id, endpoint), (breaker, _) in self._user_breaker_map.items():
                previous_state = breaker.state.state
                if previous_state != "CLOSED":
                    breaker._record_state_change(previous_state, "CLOSED", "Manual force close all")
                await breaker.reset()

    def get_all_states(self) -> Dict[str, Dict]:
        """
        Tüm breaker durumlarının anlık snapshot'u (sync)
        """
        states = {}
        
        # API key states
        for key, breaker in self._api_key_breaker_map.items():
            states[f"api_{key}"] = breaker.get_state()
        
        # User states
        for (user_id, endpoint), (breaker, _) in self._user_breaker_map.items():
            states[f"user_{user_id}:{endpoint}"] = breaker.get_state()
            
        return states


# utils/binance_api/circuit_breaker.py'ye ek
class EnhancedCircuitBreaker(CircuitBreaker):
    async def execute_with_fallback(self, func, fallback_func, *args, **kwargs):
        try:
            return await self.execute(func, *args, **kwargs)
        except Exception as e:
            logger.warning(f"Primary failed, using fallback: {e}")
            return await fallback_func(*args, **kwargs)
            
#

"""

Özellikler:
- Async / sync uyumluluğu (sync fonksiyonlar executor'da çalıştırılır)
- Configurable failure_predicate: hangi hataların 'failure' sayılacağını belirler
- Half-open test concurrency kontrolü (max_half_open_calls)
- Manual record_failure / record_success hook'ları
- Metrics / hooks (on_open, on_half_open, on_close, on_failure, on_success)
- Detailed typing ve docstrings

✅ Çok kullanıcı desteği - Her API key için ayrı circuit breaker
✅ Rate limiting - Binance weight sistemine uyum
✅ APIKeyManager entegrasyonu - Mevcut sistemle uyumlu
✅ Detaylı monitoring - Rate limit metrics'ları
✅ Güvenlik - API key'ler hash'lenerek saklanır
""" 