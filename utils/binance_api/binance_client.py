# utils/binance_api/binance_client.py
"""
direct_client.py > binance_client YAPILDI
Direct HTTP client for YAML-based endpoint calls
Replaces b_spotclient.py and b_futuresclient.py
"""

import logging
from typing import Any, Dict, Optional
from .binance_request import BinanceHTTPClient

logger = logging.getLogger(__name__)


class DirectBinanceClient:
    """Direct HTTP client that uses YAML config for endpoint calls"""
    
    def __init__(self, http_client: BinanceHTTPClient):
        self.http_client = http_client
        

    # binance_client.py - call_endpoint metodunda şu kısmı değiştir:
    r"""14-1 async def call_endpoint(self, endpoint_config: Dict[str, Any], **params) -> Any:
        try:
            # Parametre validasyonu - SADECE required parametreleri kontrol et
            endpoint_params = endpoint_config.get('parameters', [])
            required_params = [p['name'] for p in endpoint_params 
                              if p.get('required', False)]
            
            for param in required_params:
                if param not in params:
                    raise ValueError(f"Required parameter missing: {param}")
            
            # HTTP method standardizasyonu
            http_method = endpoint_config.get('http_method', 'GET').upper()
            if http_method not in ['GET', 'POST', 'PUT', 'DELETE']:
                raise ValueError(f"Invalid HTTP method: {http_method}")
                
            # Base URL belirleme
            base_type = endpoint_config.get('base', 'spot')
            futures = base_type == 'futures'
            
            logger.info(f" Direct call: {http_method} {endpoint_config['path']}, futures: {futures}")
            
            return await self.http_client.send_request(
                method=http_method,
                endpoint=endpoint_config['path'],
                signed=endpoint_config.get('signed', False),
                params=params,
                futures=futures
            )
            
        except Exception as e:
            logger.error(f" Direct client call failed for {endpoint_config.get('key', 'unknown')}: {e}")
            raise
    """        
 

    # binance_client.py - call_endpoint metoduna DEBUG ekle
    async def call_endpoint(self, endpoint_config: Dict[str, Any], **params) -> Any:
        """DEBUG ile geliştirilmiş versiyon"""
        try:
            print(f" DEBUG call_endpoint: {endpoint_config.get('key')}, params={params}")
            
            # Parametre validasyonu
            endpoint_params = endpoint_config.get('parameters', [])
            required_params = [p['name'] for p in endpoint_params if p.get('required', False)]
                
            for param in required_params:
                if param not in params:
                    error_msg = f"Required parameter missing: {param}"
                    print(f" DEBUG: {error_msg}")
                    raise ValueError(error_msg)
            
            # Özellikle symbol parametresini kontrol et
            if 'symbol' in params:
                symbol = params['symbol']
                print(f" DEBUG symbol param: '{symbol}'")
                
                # Symbol validasyonu
                if not symbol or not isinstance(symbol, str) or len(symbol) < 3:
                    error_msg = f"Invalid symbol parameter: {symbol}"
                    print(f" DEBUG: {error_msg}")
                    raise ValueError(error_msg)
            
            
            # HTTP method standardizasyonu
            http_method = endpoint_config.get('http_method', 'GET').upper()
            if http_method not in ['GET', 'POST', 'PUT', 'DELETE']:
                raise ValueError(f"Invalid HTTP method: {http_method}")
                
            # Base URL belirleme
            base_type = endpoint_config.get('base', 'spot')
            futures = base_type == 'futures'
            
            logger.info(f" Direct call: {http_method} {endpoint_config['path']}, futures: {futures}")
            
            return await self.http_client.send_request(
                method=http_method,
                endpoint=endpoint_config['path'],
                signed=endpoint_config.get('signed', False),
                params=params,
                futures=futures
            )
            
        except Exception as e:
            logger.error(f" Direct client call failed for {endpoint_config.get('key', 'unknown')}: {e}")
            raise
    
    
    def _validate_and_prepare_params(self, endpoint_config: Dict[str, Any], params: Dict[str, Any]) -> Dict[str, Any]:
        """Parametreleri validate et ve hazırla"""
        validated = params.copy()
        
        # Özel durumlar
        endpoint_name = endpoint_config.get('key', '')
        
        # continuous_klines ve index_price_klines için symbol -> pair dönüşümü
        if endpoint_name in ['continuous_klines', 'index_price_klines']:
            if 'symbol' in validated and 'pair' not in validated:
                validated['pair'] = validated.pop('symbol')
                print(f" DEBUG: Converted symbol to pair for {endpoint_name}")
        
        return validated