# utils/binance_api/binance_a.py
"""
BinanceAggregator - Multi-user destekli, map tabanlı Binance API yöneticisi.

MapLoader:
  - b-map_public.yaml & b-map_private.yaml dosyalarını otomatik yükler.
  - Validator ile doğrular.
  - Endpointleri bellek içinde organize eder.

BinanceAggregator:
  - get_data(user_id, endpoint_name, params) ile otomatik çağrı yapar.
"""

import os
import yaml
import asyncio
import logging
from typing import Any, Dict, List, Optional
from .b_map_validator import BMapValidator, MapValidationError
from .binance_client import BinanceHTTPClient
from .binance_circuit_breaker import CircuitBreaker
from .binance_multi_user import MultiUserManager

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------
# Map Loader
# ---------------------------------------------------------------------
class MapLoader:
    """b-map_public.yaml ve b-map_private.yaml dosyalarını otomatik yükleyen sınıf"""

    def __init__(self, base_path: str):
        self.base_path = base_path
        self.maps: Dict[str, Dict[str, Any]] = {}
        self.validator = BMapValidator()

    def load_all(self) -> None:
        """Tüm map dosyalarını yükler ve doğrular."""
        map_files = [
            "b-map_public.yaml",
            "b-map_private.yaml",
        ]
        for fname in map_files:
            path = os.path.join(self.base_path, fname)
            if not os.path.exists(path):
                logger.warning(f"Map file missing: {path}")
                continue

            data = self.validator.load_yaml(path)
            try:
                self.validator.validate(data, fname)
                self.maps[fname.replace(".yaml", "")] = data
                logger.info(f"Loaded and validated {fname}")
            except MapValidationError as e:
                logger.error(str(e))
                continue

    def get_endpoint(self, name: str) -> Optional[Dict[str, Any]]:
        """Endpoint adına göre arama (örnek: public_spot.klines)"""
        for mname, mdata in self.maps.items():
            for section, endpoints in mdata.items():
                if section == "meta":
                    continue
                if name in endpoints:
                    return endpoints[name]
        return None


# ---------------------------------------------------------------------
# Multi-user Aggregator
# ---------------------------------------------------------------------
class BinanceAggregator:
    def __init__(self, base_path: str):
        self.map_loader = MapLoader(base_path)
        self.map_loader.load_all()
        self.users = MultiUserManager()
        self.circuit_breaker = CircuitBreaker()
        logger.info("BinanceAggregator initialized with multi-user support")

    async def get_data(self, user_id: str, endpoint_name: str, **params):
        """
        Tek bir entrypoint üzerinden veri çeker.
        Endpoint tanımı YAML map’ten alınır.
        """
        ep = self.map_loader.get_endpoint(endpoint_name)
        if not ep:
            raise ValueError(f"Unknown endpoint: {endpoint_name}")

        client_class = ep.get("client")
        method_name = ep.get("method")
        path = ep.get("path")
        signed = ep.get("signed", False)
        http_method = ep.get("http_method", "GET")

        # User’a ait HTTP client
        http_client = await self.users.get_client(user_id)
        if not http_client:
            raise ValueError(f"No valid client for user: {user_id}")

        # Dinamik metod çağrısı (örnek: SpotClient.get_account_info)
        instance = self._resolve_client(client_class, http_client)
        func = getattr(instance, method_name, None)
        if not func:
            raise ValueError(f"Client {client_class} has no method {method_name}")

        logger.debug(f"Executing {client_class}.{method_name} ({path}) signed={signed}")

        try:
            async with self.circuit_breaker:
                result = await func(**params)
            return result
        except Exception as e:
            logger.exception(f"Aggregator call failed for {endpoint_name}: {e}")
            raise

    def _resolve_client(self, client_class: str, http_client: BinanceHTTPClient):
        """Dinamik olarak client sınıfını çözer."""
        from importlib import import_module

        module_name = f"utils.binance_api.{client_class.lower()}"
        mod = import_module(module_name)
        cls = getattr(mod, client_class)
        return cls.get_instance(http_client=http_client)
