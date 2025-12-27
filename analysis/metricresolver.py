"""
analysis/metricresolver.py
version: 1207

a_core.py >  metricresolver.py (metriklerin config +endpoit bilgisi alır) <> metrikdosyaları (config+saf matematik yapı)
MetricResolver – Yeni Versiyon
-----------------------------------
Görevi: Core'a metrik tanımını (function, columns, params) döndürmek
• Kategori → Liste yapısı ile çalışır
• Otomatik modül yükleyici (importlib)
• Metric fonksiyon caching
• Error-safe (never crashes core)
✔ weakref yerine LRU cache + strong function reference
✔ modül load için LRU cache

MetricResolver'ın yaptıkları:
"ema" için kategori bul ("classical")
load_function("classical", "ema") → classical.py'den ema fonksiyonunu getir
Config'den gerekli kolonları al: "required_columns": ["close"]
NEED_ENDPOINT'den endpointleri belirle:
"ema" direkt olarak NEED_ENDPOINT'de yok
"close" kolonu için endpoint: "klines"

örnek Tam ve Doğru Akış Özeti

1. Handler: "trend" skoru iste
2. Core: trend → [ema, macd, rsi, ...] çözümle
3. MetricResolver: ema → classical kategori, ["close"] kolon, ["klines"] endpoint
4. BinanceFetcher: klines endpoint'ini çek
5. Data: OHLCV DataFrame'i oluştur
6. Prepare: sadece "close" kolonunu seç
7. Calculate: classical.ema() çağır
8. Finalize: son değer al, normalizasyon uygula
9. Composite: ema değerini formülde kullan
10. Return: trend skorunu döndür

analysis/a_core.py
core (a_core.py)
 ├─ pipeline / orchestration
 ├─ fetch
 ├─ parallel execution
 └─ composite & macro hesaplama

metricresolver
analysis/metricresolver.py
 ├─ metrik tanımları
 ├─ hangi endpoint?
 ├─ hangi kolon?
 ├─ hangi fonksiyon?
 └─ 🔥 VERİYİ METRİĞE UYARLAMA (ADAPTER’IN İŞİ)

metric dosyaları
analysis/metrics/
 ├	advanced.py
 ├	classical.py
 ├	microstructure.py
 ├	onchain.py
 ├	regime.py
 ├	risk.py
 ├	sentiment.py
 ├	volatility.py



| Bileşen | Rol                      |
| ------- | ------------------------ |
| ROC     | hız                      |
| ADX     | güç                      |
| ATR (−) | aşırı gürültüyü bastırma |



"""
# analysis/metricresolver.py


import importlib
import logging
from functools import lru_cache
from typing import Callable, Dict, Any, Optional

logger = logging.getLogger("MetricResolver")

# -------------------------------------------------------------
# 1) Metric listesi (değişmedi)
# -------------------------------------------------------------
# metrik konumu: analysis/metrics/*.py >> örnek analysis/metrics/classical.py
# metricresolver.py içeriği
METRICS = {
    "classical": [
        "ema", "sma", "macd", "rsi", "adx", "stochastic_oscillator", 
        "roc", "atr", "bollinger_bands", "value_at_risk",
        "conditional_value_at_risk", "max_drawdown", "oi_growth_rate",
        "oi_price_correlation", "spearman_corr", "cross_correlation", "futures_roc"
    ],
    "advanced": [
        "kalman_filter_trend", "wavelet_transform", "hilbert_transform_slope",
        "hilbert_transform_amplitude", "fractal_dimension_index_fdi", 
        "shannon_entropy", "permutation_entropy", "sample_entropy",
        "granger_causality", "phase_shift_index"
    ],
    "volatility": [
        "historical_volatility", "bollinger_width", "garch_1_1", "hurst_exponent",
        "entropy_index", "variance_ratio_test", "range_expansion_index","vol_of_vol"
    ],
    "sentiment": [
        "funding_rate", "funding_premium", "oi_trend"
    ],
    "microstructure": [
        "ofi", "cvd", "microprice_deviation", "market_impact", "depth_elasticity",
        "taker_dominance_ratio", "liquidity_density"
    ],
    "onchain": [
        "etf_net_flow", "exchange_netflow", "stablecoin_flow", "net_realized_pl",
        "realized_cap", "nupl", "exchange_whale_ratio", "mvrv_zscore", "sopr"
    ],
    "regime": [
        "advance_decline_line", "volume_leadership", "performance_dispersion"
    ],
    "risk": [ #8
        "volatility_risk", "liquidity_depth_risk", "spread_risk", "price_impact_risk",
        "taker_pressure_risk","liquidity_gaps", "open_interest_shock_risk", "open_interest_risk","funding_risk", "funding_stress_risk"
    ]
}

# Basit veri = "key": "endpoint"
# Derived veri = "key": ["endpoint1", "endpoint2"]

NEED_ENDPOINT = {
    # --- OHLCV (Basic & Derived) ---
    # Klines üzerinden hesaplanan temel ve türetilmiş fiyat verileri
    "open": "klines",
    "high": "klines",
    "low": "klines",
    "close": "klines",
    "volume": "klines",
    "quote_volume": "klines",
    "trades": "klines",
    "hl2": "klines",
    "hlc3": "klines",
    "ohlc4": "klines",
    "returns": "klines",
    "log_returns": "klines",
    "volatility_risk": "klines",

    # --- ORDER BOOK (Depth) ---
    # Tahta derinliği ve spread tabanlı metrikler
    "bid": "depth",
    "ask": "depth",
    "bid_price": "depth",
    "bid_size": "depth",
    "ask_price": "depth",
    "ask_size": "depth",
    "depth_levels": "depth",
    "mid_price": "depth",
    "spread": "depth",
    "liquidity_gaps": "depth",
    "liquidity_depth_risk": "depth",
    "price_impact_risk": "depth",
    "spread_risk": "depth",
    "best_bid": "ticker_book",
    "best_ask": "ticker_book",

    # --- TRADES & MICROSTRUCTURE ---
    # Anlık işlemler ve emir akışı metrikleri
    "price": "agg_trades",
    "qty": "agg_trades",
    "is_buyer_maker": "agg_trades",
    "isBuyerMaker": "agg_trades",
    "taker_buy_volume": "agg_trades",
    "taker_sell_volume": "agg_trades",
    "cvd": "agg_trades",
    "microprice": "agg_trades",
    "taker_pressure_risk": "agg_trades",
    "ofi": "depth", # Order Flow Imbalance derinlikten hesaplanır

    # --- MULTI-SOURCE (Kombine Metrikler) ---
    "microprice_dev": ["agg_trades", "depth"],
    "microprice_deviation": ["agg_trades", "depth"],
    "funding_stress_risk": ["fundingRate", "klines"],

    # --- FUNDING & MARKET PRICES ---
    "funding_rate": "fundingRate",
    "funding_rate_history": "fundingRate",
    "funding_risk": "fundingRate",
    "mark_price": "markPrice",
    "index_price": "premiumIndex",
    "funding_premium": "premiumIndex",

    # --- OPEN INTEREST ---
    "open_interest": "open_interest",
    "open_interest_history": "open_interest_statistics",
    "open_interest_risk": "open_interest_statistics",
    "open_interest_shock_risk": ["open_interest_statistics", "top_long_short_ratio"],

    # --- SENTIMENT (Long/Short Ratios) ---
    "long_short_ratio": "longShortAccountRatio",
    "top_long_short_ratio": "topLongShortPositionRatio",
    "bull_bear_ratio": "longShortAccountRatio",

    # --- LIQUIDATIONS ---
    "liquidation_qty": "forceOrders",
    "liquidation_side": "forceOrders",
    "liquidation_imbalance": "forceOrders",
}

ENDPOINT_PARAMS = {
    # Zaman serisi verileri (OHLCV)
    "klines": lambda sym, interval, limit: {
        "symbol": sym,
        "interval": interval,
        "limit": limit
    },

    # Anlık OI verisi
    "open_interest": lambda sym, interval, limit: {
        "symbol": sym
    },

    # Tarihsel OI ve İstatistikler (Period formatı dönüşümü ile)
    "open_interest_statistics": lambda sym, interval, limit: {
        "symbol": sym,
        "period": interval.replace("h", "H").replace("m", "min"),
        "limit": min(limit, 500)
    },

    # Order Book Parametreleri
    "depth": lambda sym, interval, limit: {
        "symbol": sym,
        "limit": 100
    },
    "ticker_book": lambda sym, interval, limit: {
        "symbol": sym
    },

    # Finansman ve Fiyat Verileri
    "fundingRate": lambda sym, interval, limit: {
        "symbol": sym,
        "limit": min(limit, 100)
    },
    "premiumIndex": lambda sym, interval, limit: {
        "symbol": sym
    },
    "markPrice": lambda sym, interval, limit: {
        "symbol": sym
    },

    # İşlem Verileri
    "agg_trades": lambda sym, interval, limit: {
        "symbol": sym,
        "limit": min(limit, 1000)
    },
    "trades": lambda sym, interval, limit: {
        "symbol": sym,
        "limit": min(limit, 1000)
    },

    # Likidasyonlar
    "forceOrders": lambda sym, interval, limit: {
        "symbol": sym,
        "limit": min(limit, 100)
    },

    # Duyarlılık (Sentiment) Oranları
    "longShortAccountRatio": lambda sym, interval, limit: {
        "symbol": sym,
        "period": interval,
        "limit": min(limit, 500)
    },
    "topLongShortPositionRatio": lambda sym, interval, limit: {
        "symbol": sym,
        "period": interval,
        "limit": min(limit, 500)
    }
}

# -------------------------------------------------------------
# 2) Category → Metric hızlı lookup (O(1))
# -------------------------------------------------------------
CATEGORY_LOOKUP = {}
for cat, metrics in METRICS.items():
    for m in metrics:
        CATEGORY_LOOKUP[m.lower().replace("_", "")] = cat


# -------------------------------------------------------------
# 3) Module Loader (LRU + strong ref)
# -------------------------------------------------------------
@lru_cache(maxsize=32)
def load_module(category: str):
    module_path = f"analysis.metrics.{category}"
    try:
        return importlib.import_module(module_path)
    except Exception as e:
        logger.error(f"[MetricResolver] Modül yüklenemedi: {module_path} → {e}")
        return None


# -------------------------------------------------------------
# 4) Function Loader (LRU, NO weakref)
# -------------------------------------------------------------
@lru_cache(maxsize=256)
def load_function(category: str, name: str) -> Optional[Callable]:
    module = load_module(category)
    if module is None:
        return None

    try:
        if hasattr(module, "get_function"):
            return module.get_function(name)
        return getattr(module, name)
    except Exception:
        logger.error(f"[MetricResolver] Fonksiyon bulunamadı: {category}.{name}")
        return None


# -------------------------------------------------------------
# 5) Config Loader (LRU cache)
# -------------------------------------------------------------
@lru_cache(maxsize=32)
def load_module_config(category: str) -> Dict[str, Any]:
    module = load_module(category)
    if module is None:
        return {}

    try:
        if hasattr(module, "get_module_config"):
            return module.get_module_config()
        return getattr(module, "_MODULE_CONFIG", {}) or {}
    except Exception:
        return {}


# -------------------------------------------------------------
# 6) MetricResolver Sadeltilmiş Hızlı Versiyon
# -------------------------------------------------------------
class MetricResolver:
    def __init__(self, metric_map: Dict[str, list] = None):
        self.metric_map = metric_map or METRICS

    # ---------------------------------------------------------
    def _find_category(self, metric: str) -> Optional[str]:
        return CATEGORY_LOOKUP.get(metric.lower().replace("_", ""))

    # ---------------------------------------------------------
        
    def resolve_metric_definition(self, metric_name: str) -> Dict[str, Any]:

        category = self._find_category(metric_name)
        if category is None:
            raise ValueError(f"Metric '{metric_name}' not found")

        func = load_function(category, metric_name)
        if func is None:
            raise ValueError(f"Function for metric '{metric_name}' not found")

        config = load_module_config(category)
        
        
        # Adaptasyon bilgisini modülden al
        adaptations = config.get("adaptations", {})
        adaptation_info = adaptations.get(metric_name, {"required": False})
        

        # required_columns
        rc = config.get("required_columns", {})
        if isinstance(rc, dict):
            required_cols = rc.get(metric_name, [])
        elif isinstance(rc, list):
            required_cols = rc
        else:
            required_cols = []

        # --- NEED_ENDPOINT çözümleyici ---
        metric_endpoint = NEED_ENDPOINT.get(metric_name)

        endpoint_factories = set()
        if metric_endpoint:
            if isinstance(metric_endpoint, list):
                endpoint_factories.update(metric_endpoint)
            else:
                endpoint_factories.add(metric_endpoint)

        for col in required_cols:
            col_endpoint = NEED_ENDPOINT.get(col)
            if col_endpoint:
                if isinstance(col_endpoint, list):
                    endpoint_factories.update(col_endpoint)
                else:
                    endpoint_factories.add(col_endpoint)

        required_endpoints = sorted(list(endpoint_factories))

        # --- ENDPOINT PARAM EŞLEME TABLOSU ---
        endpoint_param_factories = {
            ep: ENDPOINT_PARAMS[ep]
            for ep in required_endpoints
            if ep in ENDPOINT_PARAMS
        }

        return {
            "function": func,
            "required_columns": required_cols,
            "required_endpoints": required_endpoints,
            "endpoint_params": endpoint_param_factories,  # ★★★ EKLEDİĞİMİZ KISIM

            "data_model": config.get("data_model", "pandas"),
            "execution_type": config.get("execution_type", "sync"),
            "normalization": config.get("normalization", {}),
            "default_params": config.get("default_params", {}).get(metric_name, {}),

            "metadata": {
                "category": category,
                "module_name": f"analysis.metrics.{category}",
                "metric_name": metric_name,
            },
        }

    # ---------------------------------------------------------
    def resolve_multiple_definitions(self, metric_names: list):
        return {
            m: self.resolve_metric_definition(m)
            for m in metric_names
        }

    # ---------------------------------------------------------
    def get_available_metrics(self) -> list:
        all_metrics = []
        for items in self.metric_map.values():
            all_metrics.extend(items)
        return sorted(set(all_metrics))
        

# -------------------------------------------------------------
# 7) Global Resolver
# -------------------------------------------------------------
_default_resolver = MetricResolver()

def get_default_resolver():
    return _default_resolver




