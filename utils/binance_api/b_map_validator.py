# -*- coding: utf-8 -*-
"""
utils/binance_api/b_map_validator.py
(MAPS v2025.11 YAML formatı uyumlu sürüm)

MAPS Framework - Binance Public YAML Validator (v2025.11)
Tam uyumlu: yaml_refactor_public.py çıktısı ile
"""

import yaml
import re
import logging
from typing import Dict, Any, List, Callable

logger = logging.getLogger(__name__)

# ✅ Sabitler
REQUIRED_FIELDS = ["key", "sdk_method", "http_method", "path", "client"]
VALID_HTTP_PREFIXES = ("/api/v3", "/sapi/v1", "/fapi/v1", "/fapi/v2", "/futures/data")
VALID_RETURN_TYPES = ("dict", "list", "auto", "raw")

VALID_HTTP_METHODS = ("GET", "POST", "DELETE")

# ✅ Global pattern kuralları
BINANCE_PARAMETER_RULES = {
    "symbol": {"type": "string", "pattern": r"^[A-Z]{3,20}(USDT|BUSD|BTC|ETH)$"},
    "interval": {"type": "string", "enum": ["1m","3m","5m","15m","30m","1h","2h","4h","6h","12h","1d","1w","1M"]},
    "limit": {"type": "integer", "min": 1, "max": 1000},
    "pair": {"type": "string", "pattern": r"^[A-Z]{3,20}USDT$"},
    "period": {"type": "string", "enum": ["5m","15m","30m","1h","2h","4h","6h","12h","1d"]},
}


class MapValidationError(Exception):
    """YAML validasyonu başarısız olduğunda fırlatılır."""
    pass


class BMapValidator:
    """MAPS v2025.11 için optimize edilmiş YAML validator"""

    def __init__(self):
        self.errors: List[str] = []
        self._type_checks: Dict[str, Callable] = {
            "string": lambda x: isinstance(x, str),
            "integer": lambda x: isinstance(x, int) or (isinstance(x, str) and x.isdigit()),
            "number": lambda x: isinstance(x, (int, float)) or (isinstance(x, str) and self._is_float(x)),
        }

    # =============================================================
    # 🔍 Ana doğrulama metodu
    # =============================================================
    def validate(self, data: Dict[str, Any], file_name: str = "") -> bool:
        self.errors.clear()

        if "meta" not in data:
            self.errors.append(f"{file_name}: 'meta' bölümü eksik.")
        if "public_spot" not in data and "private_futures" not in data:
            self.errors.append(f"{file_name}: 'public_spot' veya 'private_futures' bölümü eksik.")

        for section in ("public_spot", "private_futures"):
            if section not in data:
                continue

            endpoints = data[section]
            if not isinstance(endpoints, list):
                self.errors.append(f"{file_name}: '{section}' list olmalı (dict değil).")
                continue

            for entry in endpoints:
                self._validate_entry(file_name, section, entry)

        if self.errors:
            for e in self.errors:
                logger.warning(e)
            raise MapValidationError(f"{file_name} YAML validation failed ❌")

        logger.info(f"{file_name} YAML validated successfully ✅")
        return True

    # =============================================================
    # 🔧 Endpoint yapısı kontrolü
    # =============================================================
    def _validate_entry(self, file_name: str, section: str, entry: Dict[str, Any]):
        if not isinstance(entry, dict):
            self.errors.append(f"{file_name}: {section} altındaki giriş dict olmalı.")
            return

        key = entry.get("key", "<unknown>")
        # Eksik alanlar kontrolü
        missing = [f for f in REQUIRED_FIELDS if f not in entry]
        if missing:
            self.errors.append(f"{file_name}: {section}::{key} -> Eksik alanlar: {missing}")
            return
            
            
        path = entry.get("path", "")
        if not any(path.startswith(p) for p in VALID_HTTP_PREFIXES):
            self.errors.append(f"{file_name}: {section}::{key} -> Geçersiz path: {path}")

        sdk_method = entry.get("sdk_method")
        if not sdk_method or not isinstance(sdk_method, str):
            self.errors.append(f"{file_name}: {section}::{key} -> 'sdk_method' string olmalı.")
        
        # HTTP method kontrolü
        http_method = entry.get("http_method")
        if not http_method or http_method.upper() not in VALID_HTTP_METHODS:
            self.errors.append(f"{file_name}: {section}::{key} -> Geçersiz veya eksik HTTP method: {http_method}")
            
        ret_type = entry.get("return_type", "auto")
        if ret_type not in VALID_RETURN_TYPES:
            self.errors.append(f"{file_name}: {section}::{key} -> Geçersiz return_type: {ret_type}")

        # Parametre yapısı kontrolü
        params = entry.get("parameters", [])
        if params and not isinstance(params, list):
            self.errors.append(f"{file_name}: {section}::{key} -> 'parameters' list olmalı.")

    # =============================================================
    # 🧪 Parametre validasyonu
    # =============================================================
    def validate_parameters(self, params: Dict[str, Any]) -> bool:
        for name, value in params.items():
            rule = BINANCE_PARAMETER_RULES.get(name)
            if not rule:
                continue

            if not self._check_type(value, rule["type"]):
                logger.warning(f"Parametre tipi hatalı: {name}={value}")
                return False

            if "pattern" in rule and not re.match(rule["pattern"], str(value)):
                logger.warning(f"Pattern uyuşmazlığı: {name}={value}")
                return False

            if "enum" in rule and value not in rule["enum"]:
                logger.warning(f"Geçersiz enum değeri: {name}={value}")
                return False

            if rule.get("type") == "integer":
                try:
                    num = int(value)
                    if "min" in rule and num < rule["min"]:
                        return False
                    if "max" in rule and num > rule["max"]:
                        return False
                except Exception:
                    return False
        return True

    # =============================================================
    # 🧩 Yardımcılar
    # =============================================================
    def _check_type(self, value: Any, expected: str) -> bool:
        check = self._type_checks.get(expected, lambda x: True)
        return check(value)

    def _is_float(self, x: str) -> bool:
        try:
            float(x)
            return True
        except ValueError:
            return False

    # =============================================================
    # 📂 YAML yükleme
    # =============================================================
    @staticmethod
    def load_yaml(path: str) -> Dict[str, Any]:
        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)


# ✅ Global instance
validator = BMapValidator()

if __name__ == "__main__":
    path = "utils/binance_api/b_map_public.yaml"
    path = "utils/binance_api/b_map_private.yaml"
    data = validator.load_yaml(path)
    validator.validate(data, path)
