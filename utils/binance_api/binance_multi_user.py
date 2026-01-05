# utils/binance_api/binance_multi_user.py

import asyncio
import logging
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from datetime import datetime

logger = logging.getLogger(__name__)

@dataclass
class UserSession:
    """User session data container."""
    user_id: int
    http_client: Any
    circuit_breaker: Any
    created_at: datetime
    last_used: datetime
    
    def update_usage(self):
        """Update last used timestamp."""
        self.last_used = datetime.now()

# utils/binance_api/binance_multi_user.py

class UserSessionManager:
    """
    Manages user sessions with TTL and cleanup.
    """
    
    def __init__(self, ttl_minutes: int = 60):
        self.sessions: Dict[int, UserSession] = {}
        self.ttl_seconds = ttl_minutes * 60
        self._lock = asyncio.Lock()

    async def get_session(self, user_id: int) -> Optional[UserSession]:
        """Get user session if exists and valid - user_id=0 desteği"""
        async with self._lock:
            if user_id in self.sessions:
                session = self.sessions[user_id]
                if self._is_session_valid(session):
                    session.update_usage()
                    return session
                else:
                    # Session expired, cleanup
                    await self._cleanup_session(session)
                    del self.sessions[user_id]
            return None

    async def add_session(self, user_id: int, http_client: Any):
        """Add new user session - user_id=0 desteği"""
        async with self._lock:
            # Eğer zaten bir session varsa önce temizle
            if user_id in self.sessions:
                await self._cleanup_session(self.sessions[user_id])
                
            session = UserSession(
                user_id=user_id,
                http_client=http_client,
                circuit_breaker=None,
                created_at=datetime.now(),
                last_used=datetime.now()
            )
            self.sessions[user_id] = session
            logger.info(f"Session created for user_id: {user_id}")
    
    async def cleanup_expired_sessions(self):
        """Cleanup all expired sessions."""
        async with self._lock:
            expired_users = []
            for user_id, session in self.sessions.items():
                if not self._is_session_valid(session):
                    expired_users.append(user_id)
                    await self._cleanup_session(session)
            
            for user_id in expired_users:
                del self.sessions[user_id]
                
            if expired_users:
                logger.info(f"Cleaned up {len(expired_users)} expired user sessions")
    
    def _is_session_valid(self, session: UserSession) -> bool:
        """Check if session is still valid."""
        time_since_use = (datetime.now() - session.last_used).total_seconds()
        return time_since_use < self.ttl_seconds
    
    async def _cleanup_session(self, session: UserSession):
        """Cleanup session resources."""
        try:
            if hasattr(session.http_client, 'close'):
                await session.http_client.close()
        except Exception as e:
            logger.warning(f"Error cleaning up HTTP client for user {session.user_id}: {e}")
            
# binance_multi_user.py'ye ek
class EnhancedUserSessionManager(UserSessionManager):
    async def get_or_create_session(self, user_id: int, http_client: Any, circuit_breaker: Any):
        """Get existing session or create new one"""
        session = await self.get_session(user_id)
        if not session:
            await self.add_session(user_id, http_client, circuit_breaker)
            session = await self.get_session(user_id)
        return session