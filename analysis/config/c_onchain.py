# analysis/config/c_onchain.py
# Module config for onchain.py

OnChainConfig = {
    "version": "1.0.0",
    "windows": {
        "short_days": 7,
        "medium_days": 30,
        "long_days": 90
    },
    "weights": {
        # toplam 1.0 olmalı (eğer değişirse normalizasyon yapılır)
        "etf_net_flow": 0.15,
        "stablecoin_flow": 0.15,
        "exchange_netflow": 0.20,
        "net_realized_pl": 0.15,
        "exchange_whale_ratio": 0.10,
        "mvrv_zscore": 0.10,
        "nupl": 0.05,
        "sopr": 0.10
    },
    "thresholds": {
        "bullish": 0.65,
        "bearish": 0.35
    },
    "normalization": {
        "method": "zscore_clip",  # options: zscore_clip, minmax
        "clip_z": 3.0
    },
    "data_timeout_seconds": 10,
    "explain_components_limit": 5,
    "parallel_mode": "async",  # I/O bound
    "prometheus": {
        "enable": False
    }
}
