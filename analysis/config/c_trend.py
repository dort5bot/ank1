# analysis/config/trend.py
"""
Trend Module Config
-------------------
Config for Trend & Momentum Analysis Module
Kullanım: TrendModule(config=CONFIG)
"""

from pydantic import BaseModel, validator
from typing import Dict

class TrendConfigModel(BaseModel):
    module_name: str = "trend"
    version: str = "1.0.0"

    # Metrik ağırlıkları (toplam 1 olmalı)
    weights: Dict[str, float] = {
        "ema": 0.2,
        "rsi": 0.2,
        "macd": 0.3,
        "kalman_trend": 0.3
    }

    # Threshold değerleri
    thresholds: Dict[str, float] = {
        "bullish_threshold": 0.7,
        "bearish_threshold": 0.3
    }

    # Metrik parametreleri
    ema_period: int = 21
    rsi_period: int = 14
    macd_fast: int = 12
    macd_slow: int = 26
    macd_signal: int = 9

    @validator("weights")
    def validate_weights(cls, v):
        total = sum(v.values())
        if not 0.99 <= total <= 1.01:  # small floating point tolerance
            raise ValueError(f"Weights must sum to 1.0, got {total}")
        return v

    @validator("thresholds")
    def validate_thresholds(cls, v):
        if "bullish_threshold" in v and "bearish_threshold" in v:
            if v["bullish_threshold"] <= v["bearish_threshold"]:
                raise ValueError("Bullish threshold must be greater than bearish threshold")
        return v


# CONFIG nesnesi modül import edildiğinde kullanılacak
#CONFIG = TrendConfigModel().dict()
TREND_CONFIG = TrendConfigModel().dict()
