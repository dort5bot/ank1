"""
utils/security_auditor.py
Enhanced security auditing and sanitization
ANA GÜVENLİK MODÜLÜ
"""

import logging
import re
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

class SecurityAuditor:
    """Security auditing and data sanitization"""
    
    def __init__(self):
        self.suspicious_activities = {}
        self.sensitive_patterns = [
            r'api[_-]?key', r'secret', r'token', r'password', 
            r'auth', r'credential', r'private[_-]?key'
        ]
    
    @staticmethod
    def sanitize_log_data(data: Dict[str, Any]) -> Dict[str, Any]:
        """Sensitive data'yı log'lardan temizle"""
        sensitive_keys = [
            'api_key', 'secret_key', 'token', 'password', 
            'auth', 'credential', 'private_key', 'signature',
            'listenKey', 'recvWindow', 'timestamp'  # Binance specific
        ]
        
        sanitized = {}
        for key, value in data.items():
            # Key-based filtering
            if any(sensitive in key.lower() for sensitive in sensitive_keys):
                sanitized[key] = '***'
            # Value pattern matching
            elif isinstance(value, str) and any(
                re.search(pattern, value, re.IGNORECASE) 
                for pattern in SecurityAuditor().sensitive_patterns
            ):
                sanitized[key] = '***'
            else:
                sanitized[key] = value
        
        return sanitized
    
    async def audit_request(self, user_id: int, endpoint: str, params: Dict[str, Any]) -> bool:
        """Şüpheli aktivite tespiti ve audit"""
        
        # Rate limiting check
        if await self._check_rate_abuse(user_id):
            logger.warning(f"🚨 Rate limit abuse detected for user {user_id}")
            return False
        
        # Parameter validation
        if not self._validate_parameters(endpoint, params):
            logger.warning(f"🚨 Invalid parameters from user {user_id}: {endpoint}")
            return False
        
        # Suspicious pattern detection
        if self._detect_suspicious_patterns(params):
            logger.warning(f"🚨 Suspicious patterns from user {user_id}")
            return False
        
        # Log sanitized request
        sanitized_params = self.sanitize_log_data(params)
        logger.info(f"🔒 Audited request - User: {user_id}, Endpoint: {endpoint}, Params: {sanitized_params}")
        
        return True
    
    async def _check_rate_abuse(self, user_id: int) -> bool:
        """Rate limiting abuse detection"""
        now = datetime.now()
        if user_id not in self.suspicious_activities:
            self.suspicious_activities[user_id] = []
        
        # Son 1 dakikadaki istekleri temizle
        self.suspicious_activities[user_id] = [
            timestamp for timestamp in self.suspicious_activities[user_id]
            if now - timestamp < timedelta(minutes=1)
        ]
        
        # 1 dakikada 60'dan fazla istek = abuse
        if len(self.suspicious_activities[user_id]) > 60:
            return True
        
        self.suspicious_activities[user_id].append(now)
        return False
    
    def _validate_parameters(self, endpoint: str, params: Dict[str, Any]) -> bool:
        """Endpoint-specific parameter validation"""
        # Binance API validation rules
        validation_rules = {
            'order': ['symbol', 'side', 'type', 'quantity'],
            'account': [],
            'klines': ['symbol', 'interval']
        }
        
        # Basit validation - genişletilebilir
        for endpoint_pattern, required_params in validation_rules.items():
            if endpoint_pattern in endpoint:
                return all(param in params for param in required_params)
        
        return True
    
    def _detect_suspicious_patterns(self, params: Dict[str, Any]) -> bool:
        """Şüpheli pattern'leri tespit et"""
        suspicious_indicators = [
            # Extremely large quantities
            ('quantity', lambda x: float(x) > 1000000),
            ('price', lambda x: float(x) <= 0),
            # SQL injection patterns
            ('symbol', lambda x: any(char in str(x) for char in [';', '--', '/*'])),
            # XSS patterns
            ('symbol', lambda x: any(pattern in str(x).lower() for pattern in ['<script>', 'javascript:']))
        ]
        
        for param_name, check_function in suspicious_indicators:
            if param_name in params and check_function(params[param_name]):
                return True
        
        return False

# Global instance
security_auditor = SecurityAuditor()