"""
Config for Volatility & Regime module
File: analysis/config/c_volat.py
"""

CONFIG = {
    # data windowing
    "ohlcv_limit": 500,
    "annualization": 365,            # for HV (daily basis)
    "hv_scale": 0.8,                 # scale used to normalize historical vol
    # ATR / Bollinger
    "atr_period": 14,
    "atr_lookback": 50,
    "atr_scale": 0.01,
    "bb_window": 20,
    "bb_scale": 0.02,
    # Variance ratio
    "var_lag": 2,
    "var_sensitivity": 4.0,
    # Hurst
    "hurst_max_lag": 20,
    # Entropy
    "entropy_bins": 50,
    "entropy_scale": 3.5,
    # GARCH fallback
    "garch_proxy_span": 20,
    "realized_window": 20,
    "premium_scale": 0.05,
    "rei_lookback": 20,
    "rei_scale": 0.5,
    # Parallelism
    "max_workers": 4,
    # Scoring thresholds
    "trend_threshold": 0.6,
    "range_threshold": 0.55,
    # Weights (will be normalized by module if sum != 1)
    "weights": {
        "historical_volatility": 0.15,
        "atr": 0.1,
        "bollinger_width": 0.1,
        "variance_ratio": 0.2,
        "hurst": 0.15,
        "entropy_struct": 0.05,
        "garch_implied_realized_diff": 0.15,
        "premium": 0.05,
        "rei": 0.05,
    }
}
