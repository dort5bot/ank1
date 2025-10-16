# utils/cache_manager.py class TTLCacheManager
# Global TTL Cache Manager
# Çoklu kullanıcı destekli, async, thread-safe, yüksek performanslı
"""
from utils.cache_manager import TTLCacheManager as Cache
from utils.cache_manager import TTLCacheManager as Cache

| Özellik                         | Açıklama                                                                 |
| ------------------------------- | ------------------------------------------------------------------------ |
| 🧠 **TTL Cache (Time-to-Live)** | Veriyi belirli süre saklar (örn. 60 saniye).                             |
| 👥 **User + Endpoint Bazlı**    | Aynı endpoint farklı kullanıcıda ayrı tutulur.                           |
| 🧵 **Async + Thread-Safe**      | `asyncio.Lock` ile yarış koşullarına karşı korumalı.                     |
| 🧹 **Otomatik temizlik**        | Expired kayıtları periyodik olarak temizler.                             |
| ⚙️ **Hafif & bağımsız**         | Her modül, handler veya analiz modülü tarafından kolayca kullanılabilir. |
| 📊 **İstatistik fonksiyonu**    | Cache boyutu ve TTL bilgisini döndürür.                                  |


KULLANIMI>>

from utils.cache_manager import TTLCacheManager

class BinancePriceService:
    _instance = None

    def __init__(self):
        self.aggregator = BinanceAggregator.get_instance()
        self.key_manager = APIKeyManager.get_instance()
        # 🌐 Global cache manager kullanımı
        self.cache = TTLCacheManager(ttl_seconds=60)

    async def fetch_ticker_data(self, user_id: int):
        cache_key = (user_id, "ticker_24h")
        cached = await self.cache.get(cache_key)
        if cached is not None:
            return cached
        result = await self.aggregator.get_data(user_id, "ticker_24h")
        await self.cache.set(cache_key, result)
        return result

"""

import asyncio
import time
import logging
from typing import Any, Dict, Tuple, Optional

logger = logging.getLogger(__name__)


class TTLCacheManager:
    """
    Asenkron TTL (Time-To-Live) cache yöneticisi.
    Kullanım örneği:
        cache = TTLCacheManager(ttl_seconds=60)
        await cache.set(("user1", "ticker"), data)
        result = await cache.get(("user1", "ticker"))
    """

    def __init__(self, ttl_seconds: int = 60, max_items: int = 5000):
        self.ttl = ttl_seconds
        self.max_items = max_items
        self._cache: Dict[Tuple[Any, Any], Tuple[float, Any]] = {}
        self._lock = asyncio.Lock()
        self._cleanup_task: Optional[asyncio.Task] = None
        self._stop_event = asyncio.Event()

    # ---------------------------------------------------------------
    # 🧩 Core Cache Operations
    # ---------------------------------------------------------------
    async def get(self, key: Tuple[Any, Any]) -> Optional[Any]:
        """TTL süresi dolmamışsa cache’den veriyi döndür."""
        async with self._lock:
            entry = self._cache.get(key)
            if not entry:
                return None
            ts, value = entry
            if time.time() - ts < self.ttl:
                return value
            # Süresi dolmuş → sil
            del self._cache[key]
            return None

    async def set(self, key: Tuple[Any, Any], value: Any):
        """Veriyi cache’e ekler (TTL başlatılır)."""
        async with self._lock:
            # Kapasite kontrolü (FIFO temizlik)
            if len(self._cache) >= self.max_items:
                oldest_key = next(iter(self._cache.keys()))
                del self._cache[oldest_key]
            self._cache[key] = (time.time(), value)

    async def delete(self, key: Tuple[Any, Any]):
        """Belirli bir cache kaydını siler."""
        async with self._lock:
            if key in self._cache:
                del self._cache[key]

    async def clear(self):
        """Tüm cache’i temizler."""
        async with self._lock:
            self._cache.clear()
            logger.info("🧹 TTLCacheManager: Cache cleared manually")

    # ---------------------------------------------------------------
    # 🧹 Otomatik Temizlik Döngüsü
    # ---------------------------------------------------------------
    async def _cleanup_loop(self, interval: int = 60):
        """Belirli aralıklarla cache temizlik işlemi yapar."""
        logger.info(f"🌀 TTLCacheManager cleanup loop started (interval={interval}s)")
        while not self._stop_event.is_set():
            await asyncio.sleep(interval)
            await self.cleanup_expired()

    async def cleanup_expired(self):
        """Süresi dolan cache kayıtlarını temizler."""
        async with self._lock:
            now = time.time()
            expired_keys = [k for k, (ts, _) in self._cache.items() if now - ts >= self.ttl]
            for k in expired_keys:
                del self._cache[k]
            if expired_keys:
                logger.debug(f"🧹 TTLCacheManager cleaned {len(expired_keys)} expired items")

    async def start_background_cleanup(self, interval: int = 60):
        """Arka planda temizlik görevini başlatır."""
        if self._cleanup_task:
            return
        self._cleanup_task = asyncio.create_task(self._cleanup_loop(interval))

    async def stop_background_cleanup(self):
        """Arka plan temizlik görevini durdurur."""
        if self._cleanup_task:
            self._stop_event.set()
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass
            logger.info("🛑 TTLCacheManager cleanup loop stopped")

    # ---------------------------------------------------------------
    # 🔎 Yardımcı Fonksiyonlar
    # ---------------------------------------------------------------
    async def stats(self) -> Dict[str, Any]:
        """Cache istatistiklerini döndürür."""
        async with self._lock:
            return {
                "items": len(self._cache),
                "ttl": self.ttl,
                "max_items": self.max_items,
            }
# ---------------------------------------------------------------
# 🎯 Eski modüllerle uyumluluk: cache_result decorator
# ---------------------------------------------------------------
# Bu sayede `from utils.cache import cache_result` yerine
# `from utils.cache_manager import cache_result` kullanılabilir.
# İleride refactor sürecinde geçici çözüm olarak yeterlidir.

_cache = TTLCacheManager(ttl_seconds=60)

def cache_result(ttl_seconds: int = 60, **kwargs):
    """Basit TTL cache decorator (async fonksiyonlar için)."""
    ttl = kwargs.get("ttl", ttl_seconds)
    def decorator(func):
        async def wrapper(*args, **kwargs):
            key = (func.__name__, args, tuple(kwargs.items()))
            cached = await _cache.get(key)
            if cached is not None:
                return cached
            result = await func(*args, **kwargs)
            await _cache.set(key, result)
            return result
        return wrapper
    return decorator

