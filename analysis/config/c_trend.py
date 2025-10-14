"""
Trend & Momentum Analysis Module Configuration
Version: 1.0.0
deepsek
"""

TrendConfig = {
    "module_name": "trend_moment",
    "version": "1.0.0",
    "description": "Trend direction and momentum strength analysis",
    "lifecycle": "development",
    "parallel_mode": "batch",
    
    # Data parameters
    "window": 100,
    "min_data_points": 50,
    
    # Classic TA parameters
    "ema_periods": [20, 50, 200],
    "rsi_period": 14,
    "macd_fast": 12,
    "macd_slow": 26,
    "macd_signal": 9,
    "bollinger_period": 20,
    "bollinger_std": 2,
    "atr_period": 14,
    "adx_period": 14,
    "stoch_rsi_period": 14,
    "stoch_rsi_smooth": 3,
    "momentum_period": 10,
    
    # Advanced metrics parameters
    "kalman": {
        "process_var": 1e-4,
        "obs_var": 1e-3
    },
    "z_score_window": 21,
    "wavelet_family": "db4",
    "wavelet_level": 3,
    "hilbert_window": 10,
    "fdi_window": 10,
    
    # Scoring weights
    "weights": {
        "ema_trend": 0.15,
        "rsi_momentum": 0.12,
        "macd_trend": 0.13,
        "bollinger_trend": 0.10,
        "atr_volatility": 0.08,
        "adx_strength": 0.10,
        "stoch_rsi_momentum": 0.08,
        "momentum_oscillator": 0.07,
        "kalman_trend": 0.05,
        "z_score_normalization": 0.04,
        "wavelet_trend": 0.03,
        "hilbert_slope": 0.03,
        "fdi_complexity": 0.02
    },
    
    # Thresholds for signal classification
    "thresholds": {
        "bullish": 0.7,
        "bearish": 0.3,
        "strong_trend": 0.6,
        "weak_trend": 0.4
    },
    
    # Validation rules
    "validation": {
        "min_price_change": 0.001,
        "max_data_gap": 300  # seconds
    }
}

# Parameter descriptions for documentation
PARAM_DESCRIPTIONS = {
    "ema_periods": "Exponential Moving Average periods for multi-timeframe trend analysis",
    "rsi_period": "Relative Strength Index period for momentum measurement",
    "macd_fast": "MACD fast EMA period",
    "macd_slow": "MACD slow EMA period", 
    "macd_signal": "MACD signal line period",
    "bollinger_period": "Bollinger Bands moving average period",
    "bollinger_std": "Bollinger Bands standard deviation multiplier",
    "weights": "Component weights for final trend score calculation",
    "thresholds": "Score thresholds for trend classification"
}