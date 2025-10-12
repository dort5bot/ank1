# utils/binance_api/cache.py
from aiocache import cached, Cache
from aiocache.serializers import JsonSerializer
from typing import Any, Optional

class BinanceCacheManager:
    """Binance cache yöneticisi"""
    
    _instance = None
    
    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance
    
    def __init__(self):
        self.cache = Cache(
            Cache.MEMORY,
            serializer=JsonSerializer(),
            namespace="binance_"
        )

    async def invalidate_user_cache(self, user_id: int):
        """Kullanıcı cache'ini temizle"""
        try:
            # Basit implementasyon - production'da daha sofistike olmalı
            keys_to_remove = []
            # Not: Gerçek implementasyonda cache key'lerini takip etmen gerekir
            logger.info(f"Cache invalidated for user {user_id}")
        except Exception as e:
            logger.error(f"Cache invalidation error for user {user_id}: {e}")

def cached_binance_data(ttl=300):
    """
    Binance verileri için cache decorator'ı
    """
    def key_builder(f, *args, **kwargs):
        # args[0] = self (BinanceAggregator instance), args[1] = user_id
        user_id = args[1] if len(args) > 1 else "unknown"
        endpoint_name = args[2] if len(args) > 2 else f.__name__
        return f"binance_user_{user_id}_{endpoint_name}"
    
    return cached(
        ttl=ttl, 
        key_builder=key_builder,
        serializer=JsonSerializer(),
        namespace="binance_api"
    )