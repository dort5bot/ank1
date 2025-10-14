# analysis/config/c_risk.py

"""
Configuration for the Risk & Exposure module.

Tune default parameters here. This file should be imported as:
from analysis.config.c_risk import CONFIG
"""

RiskExposureConfig = {
    # OHLCV / ATR
    "ohlcv": {
        "interval": "1h",
        "lookback_bars": 500,   # number of bars to fetch for historical metrics
    },

    # ATR (Average True Range) adaptive stop multiplier
    "atr": {
        "period": 21,
        "multiplier": 3.0
    },

    # VaR / CVaR
    "var": {
        "window": 250,  # days / bars using for VaR historical simulation
        "confidence_levels": [0.95, 0.99]
    },

    # Volatility targeting
    "vol_target": {
        "target_volatility": 0.12,  # annualized target volatility (12%)
        "lookback": 63  # lookback bars used to estimate vol (e.g., 63 ~ 3 months of daily)
    },

    # Sharpe / Sortino dynamic params
    "performance": {
        "rolling_window": 63,
        "risk_free_rate": 0.0  # used in Sharpe calc (per-period; for hourly/daily align accordingly)
    },

    # Leverage & maintenance assumptions if exact values unavailable
    "leverage": {
        "default_maintenance_margin": 0.005,  # 0.5% (approximation)
        "min_leverage": 1,
        "max_leverage": 125
    },

    # Scoring weights (must sum to 1.0)
    "weights": {
        "var": 0.30,
        "cvar": 0.25,
        "leverage": 0.15,
        "vol_targeting": 0.10,
        "max_drawdown": 0.10,
        "atr_stop": 0.10
    },

    # thresholds for interpretability
    "thresholds": {
        "high_risk": 0.75,
        "medium_risk": 0.45
    },

    # execution / runtime
    "parallel": {
        "cpu_bound_pool_workers": 2
    },

    # metadata
    "module_version": "1.0.0",
    "name": "risk_expos"
}
