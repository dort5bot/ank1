# analysis/config/c_corr.py
"""
Korelasyon & Lead-Lag Analiz Modülü Config
"""
CorrelationConfig = {
    "module_name": "correlation_lead_lag",
    "version": "1.0.0",
    "description": "Coin'ler arası korelasyon ve liderlik analizi",
    
    # Hesaplama parametreleri
    "calculation": {
        "default_interval": "1h",
        "default_limit": 100,
        "max_lag_period": 10,
        "rolling_window": 20,
        "granger_max_lags": 5,
        "var_max_lags": 5
    },
    
    # Metrik ağırlıkları
    "weights": {
        "pearson_corr": 0.15,
        "beta": 0.15, 
        "rolling_cov": 0.10,
        "partial_corr": 0.10,
        "lead_lag_delta": 0.20,
        "granger_causality": 0.15,
        "dtw_distance": 0.10,
        "var_impulse": 0.05
    },
    
    # Threshold değerleri
    "thresholds": {
        "high_correlation": 0.7,
        "medium_correlation": 0.3,
        "significant_lead": 0.1,
        "strong_causality": 0.05,
        "high_connectivity": 0.6,
        "medium_connectivity": 0.4
    },
    
    # Paralel işleme
    "parallel_processing": {
        "enabled": True,
        "max_workers": 10,
        "batch_size": 5
    },
    
    # API config
    "api_config": {
        "timeout": 30,
        "retry_attempts": 3,
        "rate_limit_delay": 0.1
    },
    
    # Cache config
    "cache": {
        "enabled": True,
        "ttl": 300,  # 5 dakika
        "max_size": 1000
    },
    
    # Lifecycle
    "lifecycle": {
        "stage": "development",
        "stability": "beta",
        "deprecation_date": None
    },
    
    # Parametre açıklamaları
    "param_descriptions": {
        "default_interval": "Varsayılan zaman aralığı",
        "default_limit": "Varsayılan veri noktası sayısı", 
        "max_lag_period": "Maksimum lead-lag periyodu",
        "rolling_window": "Rolling hesaplamalar için pencere boyutu",
        "granger_max_lags": "Granger testi için maksimum lag",
        "var_max_lags": "VAR modeli için maksimum lag"
    }
}

# Parametre validasyon şeması
PARAM_SCHEMA = {
    "default_interval": {
        "type": "string", 
        "allowed": ["1m", "5m", "15m", "1h", "4h", "1d"],
        "default": "1h"
    },
    "default_limit": {
        "type": "integer",
        "min": 20,
        "max": 1000,
        "default": 100
    },
    "max_lag_period": {
        "type": "integer", 
        "min": 5,
        "max": 50,
        "default": 10
    },
    "rolling_window": {
        "type": "integer",
        "min": 10, 
        "max": 200,
        "default": 20
    }
}