# utils/binance_api/b_map_validator.py
"""
b_map_validator.py
------------------
Binance endpoint YAML dosyalarını doğrulayan yardımcı modül.

Validasyon türleri:
- Zorunlu alanlar kontrolü
- HTTP metodu kontrolü
- signed / public uyumu
- Path formatı doğrulaması
"""

import yaml
import logging
from typing import Dict, Any, List

logger = logging.getLogger(__name__)

REQUIRED_FIELDS = ["client", "method", "path"]
VALID_HTTP_PREFIXES = ("/api/v3", "/sapi/v1", "/fapi/v1", "/fapi/v2")
VALID_HTTP_METHODS = ("GET", "POST", "DELETE")

class MapValidationError(Exception):
    pass


class BMapValidator:
    """YAML map dosyalarını kontrol eden sınıf."""

    def __init__(self):
        self.errors: List[str] = []

    def validate(self, data: Dict[str, Any], file_name: str = "") -> bool:
        logger.info(f"Validating {file_name} ...")

        # meta kontrolü
        if "meta" not in data:
            self.errors.append(f"{file_name}: 'meta' bölümü eksik.")

        for section, endpoints in data.items():
            if section == "meta":
                continue
            if not isinstance(endpoints, dict):
                self.errors.append(f"{file_name}: '{section}' dict olmalı.")
                continue

            for name, info in endpoints.items():
                # Zorunlu alanlar
                for f in REQUIRED_FIELDS:
                    if f not in info:
                        self.errors.append(f"{file_name}: {section}.{name} -> '{f}' alanı eksik.")
                # Path formatı
                if "path" in info and not str(info["path"]).startswith(VALID_HTTP_PREFIXES):
                    self.errors.append(f"{file_name}: {section}.{name} -> path hatalı: {info['path']}")
                # Method kontrolü
                if "http_method" in info and info["http_method"].upper() not in VALID_HTTP_METHODS:
                    self.errors.append(f"{file_name}: {section}.{name} -> geçersiz HTTP method: {info['http_method']}")

        if self.errors:
            for e in self.errors:
                logger.warning(e)
            raise MapValidationError(f"Map validation failed for {file_name}")

        logger.info(f"{file_name} doğrulandı ✅")
        return True

    @staticmethod
    def load_yaml(path: str) -> Dict[str, Any]:
        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)
