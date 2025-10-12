# utils/binance_api/b_rate_limiter.py
class UserAwareRateLimiter:
    def __init__(self):
        self.user_limits: Dict[int, Dict] = {}
    
    async def acquire(self, user_id: int, endpoint: str):
        user_limit = self.user_limits.get(user_id, {"count": 0, "window_start": time.time()})
        # User-specific rate limiting logic