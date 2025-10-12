# utils/binance_api/binance_client.py
"""
(Yani artık hiçbir mantık içermez, sadece alias’dır. Böylece modül içi değişiklikler kırılmaz.)

Tüm handler ve jobs kodları artık BinanceAggregator üzerinden veri alır
"""
from .binance_request import BinanceHTTPClient
__all__ = ["BinanceHTTPClient"]
