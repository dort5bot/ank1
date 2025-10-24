# utils/binance_api/b_rate_limiter.py
"""
class UserAwareRateLimiter:
    def __init__(self):
        self.user_limits: Dict[int, Dict] = {}
    
    async def acquire(self, user_id: int, endpoint: str):
        user_limit = self.user_limits.get(user_id, {"count": 0, "window_start": time.time()})
        # User-specific rate limiting logic
        await rate_limiter.acquire(user_id, endpoint_name)

 """       
        

import time
import asyncio
from typing import Dict

class UserAwareRateLimiter:
    def __init__(self, max_requests: int = 10, window_seconds: int = 60):
        """
        Args:
            max_requests: Her kullanıcı için pencere başına maksimum izin verilen toplam ağırlık.
            window_seconds: Rate limit penceresinin süresi (saniye).
        """
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        # Kullanıcı -> {"count": int, "window_start": float, "lock": asyncio.Lock}
        self.user_limits: Dict[int, Dict] = {}
        self._global_lock = asyncio.Lock()

    async def acquire(self, user_id: int, endpoint: str, weight: int = 1):
        """
        Rate limit için izin ister. Eğer limit aşılmışsa bekler.
        
        Args:
            user_id: Kullanıcı ID'si.
            endpoint: Çağrılan endpoint ismi (gerekirse detaylı logging için).
            weight: Bu isteğin ağırlığı.
        """
        async with self._global_lock:
            if user_id not in self.user_limits:
                self.user_limits[user_id] = {
                    "count": 0,
                    "window_start": time.time(),
                    "lock": asyncio.Lock()
                }
            user_data = self.user_limits[user_id]

        async with user_data["lock"]:
            current_time = time.time()
            elapsed = current_time - user_data["window_start"]
            if elapsed > self.window_seconds:
                # Yeni pencere başlat
                user_data["count"] = 0
                user_data["window_start"] = current_time
            
            # Eğer limit aşılırsa bekle (pencere sonuna kadar)
            while user_data["count"] + weight > self.max_requests:
                sleep_time = self.window_seconds - (time.time() - user_data["window_start"])
                if sleep_time > 0:
                    await asyncio.sleep(sleep_time)
                # Pencere yenilenince sayacı sıfırla
                user_data["count"] = 0
                user_data["window_start"] = time.time()

            # İzin verildi, sayaç arttır
            user_data["count"] += weight

    async def release(self, user_id: int, endpoint: str, weight: int = 1):
        """
        İsteğin işlenmesi bittiğinde sayaçtan düşmek için kullanılabilir.
        Eğer senaryonda sadece acquire yeterliyse, release metodunu boş bırakabilirsin.
        """
        # İstersen sayaçtan düşme mantığı ekleyebilirsin.
        pass

    async def cleanup(self):
        """Uzun süre kullanılmayan kullanıcı verilerini temizle (opsiyonel)."""
        async with self._global_lock:
            to_delete = []
            current_time = time.time()
            for user_id, data in self.user_limits.items():
                if current_time - data["window_start"] > 10 * self.window_seconds:
                    to_delete.append(user_id)
            for user_id in to_delete:
                del self.user_limits[user_id]
