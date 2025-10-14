"""
config/c_micro.py
Micro Alpha Factor Configuration
Tick-level microstructure analysis for high-frequency alpha generation

MICRO_ALPHA_CONFIG> CONFIG
"""

MicroAlphaConfig = {
    "module_name": "micro_alpha",
    "version": "1.0.0",
    "description": "Real-time microstructure alpha factor generator",
    
    # Data sources and endpoints
    "data_sources": {
        "rest_endpoints": [
            "/fapi/v1/depth",
            "/fapi/v1/trades", 
            "/fapi/v1/ticker/price",
            "/fapi/v1/ticker/bookTicker",
            "/fapi/v1/aggTrades"
        ],
        "websocket_streams": [
            "<symbol>@trade",
            "<symbol>@depth@100ms", 
            "<symbol>@aggTrade",
            "<symbol>@depth",
            "<symbol>@kline_1m"
        ]
    },
    
    # Core parameters
    "parameters": {
        "lookback_window": 1000,  # Number of ticks to look back
        "update_frequency_ms": 100,  # WebSocket update frequency
        "min_tick_volume": 10,  # Minimum volume to consider valid tick
        "spread_threshold": 0.0001,  # Minimum spread ratio
    },
    
    # Metric calculation windows
    "windows": {
        "cvd_window": 50,
        "ofi_window": 20,
        "microprice_window": 10,
        "zscore_window": 100
    },
    
    # Weights for final alpha score
    "weights": {
        "cumulative_volume_delta": 0.25,
        "order_flow_imbalance": 0.25,
        "microprice_deviation": 0.20,
        "market_impact": 0.15,
        "latency_flow_ratio": 0.10,
        "hf_zscore": 0.05
    },
    
    # Thresholds for signal generation
    "thresholds": {
        "bullish_threshold": 0.7,
        "bearish_threshold": 0.3,
        "cvd_extreme": 2.0,  # Z-score extreme
        "ofi_extreme": 1.5
    },
    
    # Kalman filter parameters for market impact
    "kalman": {
        "process_variance": 1e-6,
        "observation_variance": 1e-4,
        "initial_state": 0.0,
        "initial_covariance": 1.0
    },
    
    # Performance settings
    "performance": {
        "cache_ttl": 1,  # 1 second cache for real-time data
        "batch_size": 100,
        "max_retries": 3,
        "timeout_seconds": 5
    },
    
    # Module metadata
    "lifecycle": "development",
    "parallel_mode": "async",
    "job_type": "stream",
    "output_type": "micro_alpha_score"
}

# Parameter descriptions for documentation
PARAM_DESCRIPTIONS = {
    "lookback_window": "Number of recent ticks to maintain in memory for calculations",
    "cvd_window": "Window for Cumulative Volume Delta calculation",
    "ofi_window": "Window for Order Flow Imbalance calculation", 
    "weights": "Component weights for final alpha score aggregation",
    "kalman": "Kalman filter parameters for market impact estimation"
}