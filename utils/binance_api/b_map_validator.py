# utils/binance_api/b_map_validator.py
"""
GELİŞMİŞ BINANCE YAML VALIDATOR
- YAML yapısı validasyonu
- Parametre validation rules
- Akıllı endpoint analizi
- Security audit desteği
"""

import yaml
import re
import logging
from typing import Dict, Any, List, Optional

logger = logging.getLogger(__name__)

# ✅ VALIDATION CONSTANTS
REQUIRED_FIELDS = ["client", "method", "path"]
VALID_HTTP_PREFIXES = ("/api/v3", "/sapi/v1", "/fapi/v1", "/fapi/v2", "/futures/data")
VALID_HTTP_METHODS = ("GET", "POST", "DELETE")

# ✅ GLOBAL VALIDATION RULES (Merkezi)
BINANCE_PARAMETER_RULES = {
    "symbol": {
        "type": "string",
        "pattern": r"^[A-Z]{3,20}(USDT|BUSD|BTC|ETH)$",
        "description": "Trading pair symbol"
    },
    "interval": {
        "type": "string", 
        "enum": ["1m", "3m", "5m", "15m", "30m", "1h", "2h", "4h", "6h", "8h", "12h", "1d", "3d", "1w", "1M"],
        "description": "Kline interval"
    },
    "limit": {
        "type": "integer",
        "min": 1,
        "max": 1000,
        "default": 500,
        "description": "Result limit"
    },
    # ... diğer kurallar
}

# ✅ ENDPOINT PATTERN TEMPLATES
ENDPOINT_PATTERNS = {
    # Market Data - Single Symbol
    r".*_24h$": {"required": ["symbol"], "optional": []},
    r"depth$": {"required": ["symbol"], "optional": ["limit"]},
    r"trades$": {"required": ["symbol"], "optional": ["limit"]},
    r"avg_price$": {"required": ["symbol"], "optional": []},
    
    # Market Data - All Symbols  
    r".*_all$": {"required": [], "optional": ["limit"]},
    r".*_limited$": {"required": [], "optional": ["limit"]},
    r"exchange_info$": {"required": [], "optional": []},
    
    # Klines/Candles
    r"klines$": {"required": ["symbol", "interval"], "optional": ["limit", "startTime", "endTime"]},
    r".*klines$": {"required": ["symbol", "interval"], "optional": ["limit", "startTime", "endTime"]},
    
    # Order Management
    r"order$": {"required": ["symbol", "side", "type", "quantity"], "optional": ["price", "timeInForce"]},
    
    # Account Data
    r"account.*": {"required": [], "optional": []},
    r"balance$": {"required": [], "optional": []},
    
    # Default fallback
    "default": {"required": [], "optional": []}
}

class MapValidationError(Exception):
    pass

class BMapValidator:
    """Gelişmiş YAML validator - Validation + Parameter Rules"""
    
    def __init__(self):
        self.errors: List[str] = []
        self._validation_cache: Dict[str, Dict] = {}

    # ✅ MEVCUT YAML VALIDATION (koru)
    def validate_yaml_structure(self, data: Dict[str, Any], file_name: str = "") -> bool:
        """Orijinal YAML yapı validasyonu"""
        logger.info(f"Validating YAML structure for {file_name} ...")

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

        logger.info(f"{file_name} YAML structure validated ✅")
        return True

    # ✅ YENİ: PARAMETER VALIDATION SİSTEMİ
    def get_parameter_rules(self, endpoint: str, endpoint_config: Dict[str, Any]) -> Dict[str, Any]:
        """Endpoint için parametre kurallarını getir"""
        
        # 1. Önce YAML'daki explicit parametreleri kontrol et
        yaml_params = endpoint_config.get('parameters', {})
        if yaml_params:
            return self._parse_yaml_parameters(yaml_params)
        
        # 2. Pattern-based otomatik tespit
        return self._auto_detect_parameters(endpoint)
    
    def _parse_yaml_parameters(self, yaml_params: Dict) -> Dict[str, Any]:
        """YAML'daki basit parametre formatını parse et"""
        rules = {"required": [], "optional": []}
        
        for param_name, param_rule in yaml_params.items():
            if param_rule == "required":
                rules["required"].append(param_name)
            elif param_rule == "optional":
                rules["optional"].append(param_name)
            elif isinstance(param_rule, dict):
                # Detaylı rule - ileride genişletilebilir
                if param_rule.get("required", False):
                    rules["required"].append(param_name)
                else:
                    rules["optional"].append(param_name)
        
        return rules
    
    def _auto_detect_parameters(self, endpoint: str) -> Dict[str, Any]:
        """Endpoint adına göre otomatik parametre kuralları oluştur"""
        for pattern, template in ENDPOINT_PATTERNS.items():
            if pattern == "default":
                continue
            if re.match(pattern, endpoint):
                logger.debug(f"🔍 Auto-detected parameters for {endpoint}: {template}")
                return template
        
        # Fallback
        logger.debug(f"🔍 Using default parameters for {endpoint}")
        return ENDPOINT_PATTERNS["default"]
    
    def validate_parameters(self, endpoint: str, params: Dict[str, Any], endpoint_config: Dict[str, Any]) -> bool:
        """Parametreleri validate et"""
        try:
            rules = self.get_parameter_rules(endpoint, endpoint_config)
            
            # Required parametreleri kontrol et
            missing_required = [p for p in rules["required"] if p not in params]
            if missing_required:
                logger.warning(f"🚨 Missing required parameters for {endpoint}: {missing_required}")
                return False
            
            # İzin verilen parametreleri kontrol et
            allowed_params = set(rules["required"] + rules["optional"])
            extra_params = set(params.keys()) - allowed_params
            if extra_params:
                logger.warning(f"🚨 Extra parameters for {endpoint}: {extra_params}")
                return False
            
            # Parametre değerlerini validate et
            for param_name, param_value in params.items():
                if not self._validate_parameter_value(param_name, param_value):
                    return False
            
            logger.debug(f"✅ Parameter validation passed for {endpoint}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Parameter validation failed for {endpoint}: {e}")
            return False
    
    def _validate_parameter_value(self, param_name: str, value: Any) -> bool:
        """Parametre değerini global kurallara göre validate et"""
        if param_name not in BINANCE_PARAMETER_RULES:
            return True  # Bilinmeyen parametreye izin ver
        
        rule = BINANCE_PARAMETER_RULES[param_name]
        
        # Type validation
        if not self._check_type(value, rule.get("type", "string")):
            logger.warning(f"🚨 Invalid type for {param_name}: {type(value)}")
            return False
        
        # Pattern validation
        if rule.get("type") == "string" and "pattern" in rule:
            if not re.match(rule["pattern"], str(value)):
                logger.warning(f"🚨 Pattern mismatch for {param_name}: {value}")
                return False
        
        # Enum validation
        if "enum" in rule and value not in rule["enum"]:
            logger.warning(f"🚨 Invalid value for {param_name}: {value}")
            return False
        
        # Min/Max validation
        if rule.get("type") == "integer":
            try:
                num_value = int(value)
                if "min" in rule and num_value < rule["min"]:
                    logger.warning(f"🚨 Value too small for {param_name}: {num_value}")
                    return False
                if "max" in rule and num_value > rule["max"]:
                    logger.warning(f"🚨 Value too large for {param_name}: {num_value}")
                    return False
            except (ValueError, TypeError):
                logger.warning(f"🚨 Invalid integer for {param_name}: {value}")
                return False
        
        return True
    
    def _check_type(self, value: Any, expected_type: str) -> bool:
        """Type kontrolü"""
        type_checks = {
            "string": lambda x: isinstance(x, str),
            "integer": lambda x: isinstance(x, int) or (isinstance(x, str) and x.isdigit()),
            "number": lambda x: isinstance(x, (int, float)) or (isinstance(x, str) and self._is_float(x)),
            "boolean": lambda x: isinstance(x, bool) or str(x).lower() in ["true", "false", "1", "0"]
        }
        return type_checks.get(expected_type, lambda x: True)(value)
    
    def _is_float(self, x: str) -> bool:
        """String'in float olup olmadığını kontrol et"""
        try:
            float(x)
            return True
        except ValueError:
            return False

    # ✅ ORJİNAL METHODLARI KORU
    def validate(self, data: Dict[str, Any], file_name: str = "") -> bool:
        """Orjinal validate metodunu koru (backward compatibility)"""
        return self.validate_yaml_structure(data, file_name)
    
    @staticmethod
    def load_yaml(path: str) -> Dict[str, Any]:
        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)

# ✅ GLOBAL INSTANCE
validator = BMapValidator()