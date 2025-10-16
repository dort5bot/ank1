"""
Configuration for Order Flow & Microstructure module (c_order.py)

Adjust parameters and weights depending on deployment needs.
"""

OrderFlowConfig = {
    # Sampling / data windows
    "depth_levels": 12,            # how many price levels to sum for orderbook imbalance
    "elasticity_levels": 12,       # levels used to compute depth elasticity
    "trades_limit": 500,           # how many recent trades to fetch
    "liquidity_window_bps": 10.0,  # window around mid to compute liquidity density (in bps)

    # Normalization ranges (min, max) for metrics; tune from historical data / calibration runs
    "normalization": {
        "orderbook_imbalance": (-1.0, 1.0),
        "spread_bps": (0.0, 80.0),            # spreads up to 80 bps
        "market_pressure": (-1.0, 1.0),
        "trade_aggression": (0.0, 1.5),
        "slippage": (0.0, 0.5),
        "depth_elasticity": (0.0, 50.0),
        "cvd": (-1e6, 1e6),
        "ofi": (-1e6, 1e6),
        "taker_dom_ratio": (0.0, 1.0),
        "liquidity_density": (0.0, 1e6)
    },

    # Which metrics to invert after normalization (higher -> worse)
    "invert_metrics": ["spread_bps", "slippage"],

    # Weights (prioritized). Sum will be normalized automatically.
    # Professional metrics prioritized with * (high), ** (medium), *** (low) in original spec.
    "weights": {
        # Classic mandatory metrics
        "orderbook_imbalance": 0.22,
        "spread_bps": 0.15,
        "market_buy_sell_pressure": 0.15,
        "trade_aggression_ratio": 0.08,
        "slippage": 0.08,
        "depth_elasticity": 0.10,
        "cvd": 0.07,
        "ofi": 0.06,
        "taker_dom_ratio": 0.05,
        "liquidity_density": 0.04
    },

    # Thresholds for signal generation
    "imbalance_signal_thresh": 0.12,  # raw imbalance threshold to consider directional signal

    # Meta / other
    "mid_price": 100.0,  # fallback mid price if book missing (for tests)
    "version": "1.0.0",
}

CONFIG = OrderFlowConfig