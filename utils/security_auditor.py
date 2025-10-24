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
        # Endpoint configuration cache
        self._endpoint_configs: Dict[str, Dict[str, Any]] = {}
    
    def _get_endpoint_config(self, endpoint: str) -> Dict[str, Any]:
        """
        Get endpoint configuration from BinanceAggregator maps.
        Bu metod BinanceAggregator'dan endpoint konfigürasyonunu alır.
        """
        try:
            # Eğer cache'te varsa kullan
            if endpoint in self._endpoint_configs:
                return self._endpoint_configs[endpoint]
            
            # BinanceAggregator instance'ına eriş
            from utils.binance_api.binance_a import BinanceAggregator
            
            aggregator = BinanceAggregator.get_instance()
            if hasattr(aggregator, 'map_loader'):
                endpoint_config = aggregator.map_loader.get_endpoint(endpoint)
                
                # Cache'e kaydet
                self._endpoint_configs[endpoint] = endpoint_config or {}
                return self._endpoint_configs[endpoint]
            
            # Fallback: Basit endpoint classification
            return self._get_fallback_endpoint_config(endpoint)
            
        except Exception as e:
            logger.warning(f"⚠️ Could not get endpoint config for {endpoint}: {e}")
            return self._get_fallback_endpoint_config(endpoint)
    
    def _get_fallback_endpoint_config(self, endpoint: str) -> Dict[str, Any]:
        """
        Fallback endpoint configuration when aggregator is not available.
        """
        # Public endpoints (signed=False)
        public_endpoints = {
            'ticker_24hr_all': {'signed': False},
            'ticker_24h': {'signed': False},
            'exchange_info': {'signed': False},
            'ping': {'signed': False},
            'time': {'signed': False},
            'depth': {'signed': False},
            'trades': {'signed': False},
            'klines': {'signed': False},
            'uiKlines': {'signed': False},
            'avgPrice': {'signed': False}
        }
        
        # Private endpoints (signed=True)
        private_endpoints = {
            'account': {'signed': True},
            'account_info': {'signed': True},
            'my_trades': {'signed': True},
            'order': {'signed': True},
            'open_orders': {'signed': True},
            'all_orders': {'signed': True},
            'order_test': {'signed': True},
            'user_data_stream': {'signed': True}
        }
        
        # Check both dictionaries
        if endpoint in public_endpoints:
            return public_endpoints[endpoint]
        elif endpoint in private_endpoints:
            return private_endpoints[endpoint]
        else:
            # Default: assume public for safety
            logger.debug(f"🔍 Unknown endpoint '{endpoint}', assuming public")
            return {'signed': False}
    
    def _is_private_endpoint(self, endpoint: str) -> bool:
        """
        Check if endpoint requires authentication.
        """
        config = self._get_endpoint_config(endpoint)
        return config.get('signed', False)
    
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
    
    async def audit_request(self, user_id: Optional[int], endpoint: str, params: Dict[str, Any]) -> bool:
        """Gelişmiş güvenlik audit'i - user_id optional"""
        
        try:
            # Endpoint config al
            is_private = self._is_private_endpoint(endpoint)
            
            # KRİTİK KURAL: Private endpoint + user_id=None → REDDET
            if is_private and user_id is None:
                logger.error(f"🚨 Private endpoint {endpoint} accessed without user_id")
                return False
                
            # Rate limiting check
            if user_id and await self._check_rate_abuse(user_id):
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
            
        except Exception as e:
            logger.error(f"❌ Security audit failed for {endpoint}: {e}")
            # Safety first: audit failure = block request
            return False
    
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
        """Security auditor'dan validator'ı kullan"""
        try:
            from utils.binance_api.b_map_validator import validator
            endpoint_config = self._get_endpoint_config(endpoint)
            return validator.validate_parameters(endpoint, params, endpoint_config or {})
        except Exception as e:
            logger.error(f"❌ Validator integration failed: {e}")
            return self._fallback_parameter_validation(endpoint, params)
            
            

    
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
            if param_name in params:
                try:
                    if check_function(params[param_name]):
                        return True
                except (ValueError, TypeError):
                    # Conversion failed, potentially suspicious
                    return True
        
        return False

    def clear_cache(self):
        """Clear endpoint configuration cache"""
        self._endpoint_configs.clear()

# Global instance
security_auditor = SecurityAuditor()



"""
_get_endpoint_config metodunu ekledim - Endpoint konfigürasyonunu alır

_is_private_endpoint metodunu ekledim - Public/private kontrolü yapar

Fallback mekanizması ekledim - Aggregator yoksa basit sınıflandırma kullanır

Error handling geliştirdim - Audit hataları request'i bloklar (safety first)
"""