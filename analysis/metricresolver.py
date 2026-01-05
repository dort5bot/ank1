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
# adaptation = veri gereksinimi
# policy = cache / lifecycle

import importlib
import logging
from functools import lru_cache
from typing import Callable, Dict, Any, Optional

logger = logging.getLogger("MetricResolver")

# -------------------------------------------------------------
# 1) Metric listesi
# -------------------------------------------------------------
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
    "risk": [
        "volatility_risk", "liquidity_depth_risk", "spread_risk", "price_impact_risk",
        "taker_pressure_risk","liquidity_gaps", "open_interest_shock_risk", "open_interest_risk","funding_risk", "funding_stress_risk"
    ]
}

# Standardize edilmiş Policy Keyleri (DÜZELTME: Alt çizgiler kaldırıldı)
METRIC_POLICY = {
    "ema": {"stateful": False, "cache": False, "source": "realtime"},
    "macd": {"stateful": False},
    "rsi": {"stateful": False},
    "oitrend": {"stateful": True, "retention": "rolling", "source": "db"},
    "oigrowthrate": {"stateful": True, "retention": "rolling", "source": "db"},
    "openinterestshockrisk": {"stateful": True, "retention": "ttl_7d", "source": "db", "heavy": True},
    "fundingstressrisk": {"stateful": True, "retention": "ttl_14d", "source": "db"},
}

NEED_ENDPOINT = {
    "open": "klines", "high": "klines", "low": "klines", "close": "klines",
    "volume": "klines", "quote_volume": "klines", "trades": "klines",
    "bid": "depth", "ask": "depth", "spread": "depth",
    "price": "agg_trades", "cvd": "agg_trades",
    "funding_rate": "fundingRate",
    "open_interest": "open_interest",
    "microprice_deviation": ["agg_trades", "depth"],
}

ENDPOINT_PARAMS = {
    "klines": lambda sym, interval, limit: {"symbol": sym, "interval": interval, "limit": limit},
    "open_interest": lambda sym, interval, limit: {"symbol": sym},
    "depth": lambda sym, interval, limit: {"symbol": sym, "limit": 100},
    "agg_trades": lambda sym, interval, limit: {"symbol": sym, "limit": 1000},
}

# -------------------------------------------------------------
# LOOKUP ve LOADER (Cache Destekli)
# -------------------------------------------------------------

def _normalize_key(name: str) -> str:
    return name.lower().replace("_", "")

CATEGORY_LOOKUP = { _normalize_key(m): cat for cat, metrics in METRICS.items() for m in metrics }

@lru_cache(maxsize=32)
def load_module(category: str):
    try:
        return importlib.import_module(f"analysis.metrics.{category}")
    except Exception as e:
        logger.error(f"[MetricResolver] Modül yüklenemedi: {category} -> {e}")
        return None

@lru_cache(maxsize=256)
def load_function(category: str, name: str) -> Optional[Callable]:
    module = load_module(category)
    if not module: return None
    try:
        return getattr(module, "get_function")(name) if hasattr(module, "get_function") else getattr(module, name)
    except:
        return None

@lru_cache(maxsize=32)
def load_module_config(category: str) -> Dict[str, Any]:
    module = load_module(category)
    if not module: return {}
    return getattr(module, "get_module_config")() if hasattr(module, "get_module_config") else getattr(module, "_MODULE_CONFIG", {})

# -------------------------------------------------------------
# MetricResolver Sınıfı
# -------------------------------------------------------------

class MetricResolver:
    def __init__(self, metric_map: Dict[str, list] = None):
        self.metric_map = metric_map or METRICS

    def resolve_metric_definition(self, metric_name: str) -> Dict[str, Any]:
        normalized_name = _normalize_key(metric_name)
        category = CATEGORY_LOOKUP.get(normalized_name)

        if not category:
            raise ValueError(f"Metric '{metric_name}' not found in categories.")

        # 1. Temel Fonksiyon ve Config Yükleme
        func = load_function(category, metric_name)
        if not func:
            raise ValueError(f"Function '{metric_name}' not found in {category}.py")
        
        config = load_module_config(category)

        # 2. Policy ve Adaptation Birleştirme (DÜZELTME)
        # Önce global policy'yi al
        policy = METRIC_POLICY.get(normalized_name, {"stateful": False, "source": "realtime"}).copy()
        
        # Modül seviyesindeki adaptation bilgisini al
        adaptations = config.get("adaptations", {})
        adaptation_info = adaptations.get(metric_name, {"required": False})

        # Eğer adaptation "required" ise policy'yi zorunlu olarak DB moduna çek
        if adaptation_info.get("required"):
            policy["stateful"] = True
            policy["source"] = "db"

        # 3. Endpoint Çözümleme (DÜZELTME: Gerekli Kolonlar ve Metrik Kendi Gereksinimi)
        required_cols = config.get("required_columns", {}).get(metric_name, []) if isinstance(config.get("required_columns"), dict) else config.get("required_columns", [])
        
        endpoint_keys = set()
        # Metriğin doğrudan endpoint gereksinimi (Örn: funding_rate -> fundingRate)
        metric_ep = NEED_ENDPOINT.get(metric_name) or NEED_ENDPOINT.get(normalized_name)
        if metric_ep:
            endpoint_keys.update(metric_ep) if isinstance(metric_ep, list) else endpoint_keys.add(metric_ep)

        # Bağımlı kolonların endpoint gereksinimi (Örn: close -> klines)
        for col in required_cols:
            col_ep = NEED_ENDPOINT.get(col)
            if col_ep:
                endpoint_keys.update(col_ep) if isinstance(col_ep, list) else endpoint_keys.add(col_ep)

        # 4. Parametre Fabrikalarını Eşle
        endpoint_params = {ep: ENDPOINT_PARAMS[ep] for ep in endpoint_keys if ep in ENDPOINT_PARAMS}

        return {
            "policy": policy,
            "adaptation": adaptation_info,  # DÜZELTME: Core'un bilmesi için eklendi
            "function": func,
            "required_columns": required_cols,
            "required_endpoints": sorted(list(endpoint_keys)),
            "endpoint_params": endpoint_params,
            "data_model": config.get("data_model", "pandas"),
            "execution_type": config.get("execution_type", "sync"),
            "normalization": config.get("normalization", {}).get(metric_name, {}),
            "default_params": config.get("default_params", {}).get(metric_name, {}),
            "metadata": {
                "category": category,
                "module_name": f"analysis.metrics.{category}",
                "metric_name": metric_name,
            },
        }

    def resolve_multiple_definitions(self, metric_names: list):
        return {m: self.resolve_metric_definition(m) for m in metric_names}

# -------------------------------------------------------------
# 7) Global Resolver
# -------------------------------------------------------------
_default_resolver = MetricResolver()

def get_default_resolver():
    return _default_resolver




