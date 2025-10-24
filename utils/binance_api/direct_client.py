# utils/binance_api/direct_client.py
"""
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
        
    async def call_endpoint(self, endpoint_config: Dict[str, Any], **params) -> Any:
        """
        Call endpoint directly using YAML configuration
        """
        try:
            method = endpoint_config['http_method']
            path = endpoint_config['path'] 
            signed = endpoint_config['signed']
            futures = endpoint_config.get('base') == 'futures'
            
            logger.debug(f"Direct call: {method} {path}, signed: {signed}, futures: {futures}")
            
            return await self.http_client.send_request(
                method=method,
                endpoint=path,
                signed=signed,
                params=params,
                futures=futures
            )
            
        except Exception as e:
            logger.error(f"Direct client call failed: {e}")
            raise