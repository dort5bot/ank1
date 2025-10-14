# analysis/config/c_deriv.py
"""
Derivatives & Sentiment Analysis Configuration
Futures pozisyon verilerine dayalı sentiment analizi için config
"""

DerivSentimentConfig = {
    "module_name": "derivatives_sentiment",
    "version": "1.0.0",
    "description": "Futures market positioning and sentiment analysis",
    
    # Ağırlıklar (toplam 1.0 olmalı)
    "weights": {
        "funding_rate": 0.15,        # Funding rate sentiment
        "open_interest": 0.12,       # OI seviye ve trend
        "long_short_ratio": 0.13,    # Long/Short balance
        "oi_change_rate": 0.12,      # OI momentum
        "funding_skew": 0.10,        # Funding distribution skew
        "volume_imbalance": 0.10,    # Taker buy/sell imbalance
        "liquidation_heat": 0.15,    # Liquidation intensity
        "oi_delta_divergence": 0.08, # OI-Funding divergence
        "volatility_skew": 0.05      # OI volatility pattern
    },
    
    # Threshold değerleri
    "thresholds": {
        "bullish": 0.6,          # Bullish sinyal eşiği
        "bearish": 0.4,          # Bearish sinyal eşiği  
        "extreme_bull": 0.8,     # Extreme bullish
        "extreme_bear": 0.2,     # Extreme bearish
        "neutral_upper": 0.55,   # Nötr üst sınır
        "neutral_lower": 0.45    # Nötr alt sınır
    },
    
    # Parametreler
    "parameters": {
        "oi_lookback": 24,           # Open Interest lookback (saat)
        "funding_lookback": 8,       # Funding rate lookback
        "liquidation_window": 12,    # Liquidation analiz penceresi
        "volatility_period": 20,     # Volatility hesaplama periyodu
        "min_data_points": 5,        # Minimum veri noktası
        "cache_ttl": 60              # Cache TTL (saniye)
    },
    
    # Normalizasyon ayarları
    "normalization": {
        "method": "tanh",            # tanh normalizasyon
        "scale_factor": 3,           # tanh scaling faktörü
        "rolling_window": 100,       # Rolling normalization window
        "use_robust": True           # Outlier'a dayanıklı normalizasyon
    },
    
    # API ve Data ayarları
    "data_sources": {
        "funding_rate": "/fapi/v1/fundingRate",
        "open_interest": "/fapi/v1/openInterestHist", 
        "long_short_ratio": "/fapi/v1/longShortRatio",
        "liquidation_orders": "/fapi/v1/liquidationOrders",
        "taker_ratio": "/fapi/v1/takerlongshortRatio",
        "timeframe": "5m",           # Varsayılan timeframe
        "limit": 100                 # Varsayılan limit
    },
    
    # Risk ve Sınırlamalar
    "limits": {
        "max_symbols_batch": 10,     # Batch işlem maks sembol
        "request_timeout": 30,       # API timeout (saniye)
        "rate_limit_delay": 0.1,     # Rate limit delay
        "circuit_breaker_failures": 5 # Circuit breaker threshold
    },
    
    # Metrik özellikleri
    "metrics_metadata": {
        "funding_rate": {
            "description": "Funding rate sentiment (-1 to 1)",
            "formula": "tanh((current_funding - avg_funding) * 1000)",
            "interpretation": "Positive = perpetual premium, Negative = discount"
        },
        "open_interest": {
            "description": "Open Interest change sentiment", 
            "formula": "tanh(oi_change * 10)",
            "interpretation": "Positive = new positions, Negative = position closing"
        },
        "liquidation_heat": {
            "description": "Liquidation intensity metric",
            "formula": "tanh(total_liquidation / 1e6)",
            "interpretation": "High values indicate market stress"
        }
    },
    
    # Modül yaşam döngüsü
    "lifecycle": {
        "stage": "development",
        "stability": "beta",
        "deprecation_date": None
    },
    
    # Paralel işlem ayarları
    "parallel_config": {
        "mode": "async",
        "max_concurrent": 5,
        "batch_size": 3,
        "job_type": "io_bound"
    }
}

# Parametre açıklamaları
PARAM_DESCRIPTIONS = {
    "weights": "Her metrik için ağırlık değerleri (toplam 1.0 olmalı)",
    "thresholds": "Sentiment sinyalleri için eşik değerleri",
    "oi_lookback": "Open Interest analizi için lookback periyodu (saat)",
    "funding_lookback": "Funding rate analizi için lookback",
    "liquidation_window": "Likidasyon analizi için pencere boyutu",
    "normalization.method": "Metrik normalizasyon yöntemi (tanh/zscore/minmax)"
}

# Validasyon kuralları
VALIDATION_RULES = {
    "weights_sum": {"rule": "sum == 1.0", "tolerance": 0.01},
    "threshold_ordering": {"rule": "extreme_bear < bearish < bullish < extreme_bull"},
    "positive_parameters": {"params": ["oi_lookback", "funding_lookback", "cache_ttl"]}
}