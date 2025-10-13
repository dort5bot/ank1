#__init__.py
# utils/binance_api/__init__.py
try:
    from utils.apikey_manager import APIKeyManager
except ImportError:
    APIKeyManager = None
    logging.warning("APIKeyManager not available")

# Diğer dosyalarda:
from . import APIKeyManager