"""
analysis/metrics/microstructure.py
Universal Metric Template with Data-Model-Aware Normalization
Date: 03/12/2025
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple, Union, Optional
import warnings

# ==================== DATA MODEL DEFINITIONS ====================
class DataModel:
    """Veri modeli enum benzeri sınıf"""
    PANDAS = "pandas"
    NUMPY = "numpy" 
    POLARS = "polars"
    LIST = "list"
    DICT = "dict"


class NormalizationMethod:
    """Normalizasyon metodları"""
    MINMAX = "minmax"
    ZSCORE = "zscore"
    PERCENTILE = "percentile"
    TANH = "tanh"
    CLAMP = "clamp"
    RAW = "raw"  # Normalize etme


# ==================== MODULE CONFIG microstructure.py====================
_MODULE_CONFIG = {
    # ZORUNLU: Veri modeli ve execution tipi
    "data_model": DataModel.PANDAS,
    "execution_type": "sync",
    "category": "microstructure",
    
    # ZORUNLU: Her metrik için gerekli kolonlar
    "required_columns": {
        "ofi": ["bid_price", "bid_size", "ask_price", "ask_size"],
        "cvd": ["buy_volume", "sell_volume"],
        "microprice_deviation": ["best_bid", "best_ask", "bid_size", "ask_size"],
        "market_impact": ["trade_volume", "price_series"], # Binance verisiyle edilemez
        "depth_elasticity": ["depth_price", "depth_volume"], # Binance verisiyle edilemez
        "taker_dominance_ratio": ["taker_buy_volume", "taker_sell_volume"], # Binance verisiyle edilemez
        "liquidity_density": ["bid_size", "ask_size"],  # Binance uyumlu
    },
    
    # ÖNERİLEN: Normalizasyon konfigürasyonu
    "normalization": {
        # Global normalizasyon ayarları
        "global_range": {"min": -1.0, "max": 1.0},
        "default_method": NormalizationMethod.TANH,
        
        # Metrik-specific normalizasyon
        "metric_specific": {
            "ofi": {
                "method": NormalizationMethod.TANH,
                "params": {"scale": 0.01},
                "description": "OFI değerlerini tanh ile normalize et"
            },
            "cvd": {
                "method": NormalizationMethod.PERCENTILE,
                "params": {"lookback": 100, "neutral": 0.5},
                "description": "CVD'nin percentile pozisyonu"
            },
            "microprice_deviation": {
                "method": NormalizationMethod.ZSCORE,
                "params": {"window": 20, "clip_sigma": 2.0},
                "description": "Microprice deviation z-score"
            },
            "market_impact": {
                "method": NormalizationMethod.MINMAX,
                "params": {"data_min": -1.0, "data_max": 1.0},
                "description": "Market impact correlation -1..1 arası"
            },
            "depth_elasticity": {
                "method": NormalizationMethod.TANH,
                "params": {"scale": 0.5},
                "description": "Elasticity tanh normalization"
            },
            "taker_dominance_ratio": {
                "method": NormalizationMethod.MINMAX,
                "params": {"data_min": 0.0, "data_max": 2.0},
                "description": "Taker ratio 0..2 arasından normalize"
            },
            "liquidity_density": {
                "method": NormalizationMethod.PERCENTILE,
                "params": {"lookback": 50, "neutral": 0.5},
                "description": "Liquidity density percentile"
            }
        }
    },
    
    # OPSİYONEL: Metrik parametreleri
    "default_params": {
        "market_impact": {"window": 20},
        "depth_elasticity": {"window": 10},
        "liquidity_density": {"tick_range": 10}
    }
}


# ==================== DATA MODEL AWARE NORMALIZATION ====================
class Normalizer:
    """Veri modeline göre otomatik normalize eden sınıf"""
    
    @staticmethod
    def normalize(
        data: Any, 
        method: str = NormalizationMethod.TANH,
        target_range: Tuple[float, float] = (-1.0, 1.0),
        **params
    ) -> Union[float, np.ndarray, pd.Series, pd.DataFrame]:
        """
        Herhangi bir veri tipini normalize eder.
        
        Args:
            data: Normalize edilecek veri (float, pd.Series, np.ndarray, pl.Series, list, dict)
            method: Normalizasyon metodu
            target_range: Hedef aralık (min, max)
            **params: Metoda özel parametreler
        
        Returns:
            Normalize edilmiş veri (orijinal tip korunur)
        """
        if data is None:
            return 0.0
        
        # Veri tipine göre dispatch
        if isinstance(data, (int, float, np.number)):
            return Normalizer._normalize_scalar(float(data), method, target_range, **params)
        elif isinstance(data, pd.Series):
            return Normalizer._normalize_pandas_series(data, method, target_range, **params)
        elif isinstance(data, pd.DataFrame):
            return Normalizer._normalize_pandas_df(data, method, target_range, **params)
        elif isinstance(data, np.ndarray):
            return Normalizer._normalize_numpy_array(data, method, target_range, **params)
        elif isinstance(data, list):
            return Normalizer._normalize_list(data, method, target_range, **params)
        elif isinstance(data, dict):
            return Normalizer._normalize_dict(data, method, target_range, **params)
        else:
            # Diğer tipler için string representation ile dene
            try:
                scalar = float(data)
                return Normalizer._normalize_scalar(scalar, method, target_range, **params)
            except:
                warnings.warn(f"Cannot normalize type {type(data)}, returning 0.0")
                return 0.0
    
    @staticmethod
    def _normalize_scalar(
        value: float, 
        method: str, 
        target_range: Tuple[float, float],
        **params
    ) -> float:
        """Skalar değeri normalize eder"""
        if np.isnan(value):
            return 0.0
        
        target_min, target_max = target_range
        
        if method == NormalizationMethod.MINMAX:
            # Min-max normalization için range gerekli
            data_min = params.get('data_min', value)
            data_max = params.get('data_max', value)
            if data_max == data_min:
                normalized = 0.0
            else:
                normalized = ((value - data_min) / (data_max - data_min)) * (target_max - target_min) + target_min
        
        elif method == NormalizationMethod.ZSCORE:
            # Z-score normalization
            mean = params.get('mean', value)
            std = params.get('std', 1.0)
            if std == 0:
                zscore = 0.0
            else:
                zscore = (value - mean) / std
            # Tanh ile yumuşat
            normalized = np.tanh(zscore * 0.5)  # 0.5 scaling factor
            # Target range'e map et
            normalized = (normalized + 1) / 2  # -1..1 → 0..1
            normalized = normalized * (target_max - target_min) + target_min
        
        elif method == NormalizationMethod.PERCENTILE:
            # Percentile-based (skalar için anlamlı değil, default)
            normalized = 0.0
        
        elif method == NormalizationMethod.TANH:
            # Tanh ile normalize (extreme değerleri yumuşat)
            scale = params.get('scale', 0.1)
            normalized = np.tanh(value * scale)
            # Target range'e map et
            normalized = (normalized + 1) / 2  # -1..1 → 0..1
            normalized = normalized * (target_max - target_min) + target_min
        
        elif method == NormalizationMethod.CLAMP:
            # Sadece clamp
            normalized = np.clip(value, target_min, target_max)
        
        elif method == NormalizationMethod.RAW:
            # Normalize etme
            normalized = value
        
        else:
            warnings.warn(f"Unknown normalization method: {method}, using tanh")
            normalized = np.tanh(value * 0.1)
            normalized = (normalized + 1) / 2
            normalized = normalized * (target_max - target_min) + target_min
        
        # Final clamp
        return np.clip(normalized, target_min, target_max)
    
    @staticmethod
    def _normalize_pandas_series(
        series: pd.Series,
        method: str,
        target_range: Tuple[float, float],
        **params
    ) -> pd.Series:
        """Pandas Series'i normalize eder"""
        if series.empty:
            return pd.Series([], dtype=float)
        
        target_min, target_max = target_range
        
        if method == NormalizationMethod.MINMAX:
            data_min = series.min()
            data_max = series.max()
            if data_max == data_min:
                normalized = pd.Series(0.0, index=series.index)
            else:
                normalized = ((series - data_min) / (data_max - data_min)) * (target_max - target_min) + target_min
        
        elif method == NormalizationMethod.ZSCORE:
            mean = series.mean()
            std = series.std()
            if std == 0:
                zscore = pd.Series(0.0, index=series.index)
            else:
                zscore = (series - mean) / std
            normalized = np.tanh(zscore * 0.5)
            normalized = (normalized + 1) / 2
            normalized = normalized * (target_max - target_min) + target_min
        
        elif method == NormalizationMethod.PERCENTILE:
            # Rolling percentile
            lookback = params.get('lookback', min(100, len(series)))
            if len(series) >= lookback:
                # Her nokta için percentile hesapla
                percentiles = []
                for i in range(len(series)):
                    start = max(0, i - lookback + 1)
                    window = series.iloc[start:i+1]
                    if len(window) > 1:
                        percentile = (window < series.iloc[i]).sum() / len(window)
                        # 0..1 → target range
                        value = percentile * (target_max - target_min) + target_min
                    else:
                        value = 0.0
                    percentiles.append(value)
                normalized = pd.Series(percentiles, index=series.index)
            else:
                normalized = pd.Series(0.0, index=series.index)
        
        elif method == NormalizationMethod.TANH:
            scale = params.get('scale', 0.1)
            normalized = np.tanh(series * scale)
            normalized = (normalized + 1) / 2
            normalized = normalized * (target_max - target_min) + target_min
        
        elif method == NormalizationMethod.CLAMP:
            normalized = series.clip(lower=target_min, upper=target_max)
        
        elif method == NormalizationMethod.RAW:
            normalized = series
        
        else:
            normalized = series
        
        return normalized
    
    @staticmethod
    def _normalize_numpy_array(
        arr: np.ndarray,
        method: str,
        target_range: Tuple[float, float],
        **params
    ) -> np.ndarray:
        """NumPy array'i normalize eder"""
        if arr.size == 0:
            return np.array([], dtype=float)
        
        # 1D array için
        if arr.ndim == 1:
            series = pd.Series(arr)
            normalized_series = Normalizer._normalize_pandas_series(series, method, target_range, **params)
            return normalized_series.values
        
        # 2D+ array için her kolonu ayrı normalize et
        elif arr.ndim == 2:
            normalized_cols = []
            for col_idx in range(arr.shape[1]):
                col_series = pd.Series(arr[:, col_idx])
                normalized_col = Normalizer._normalize_pandas_series(col_series, method, target_range, **params)
                normalized_cols.append(normalized_col.values)
            return np.column_stack(normalized_cols)
        
        else:
            warnings.warn(f"Cannot normalize {arr.ndim}D array, flattening")
            flattened = arr.flatten()
            series = pd.Series(flattened)
            normalized = Normalizer._normalize_pandas_series(series, method, target_range, **params)
            return normalized.values.reshape(arr.shape)
    
    @staticmethod
    def _normalize_pandas_df(
        df: pd.DataFrame,
        method: str,
        target_range: Tuple[float, float],
        **params
    ) -> pd.DataFrame:
        """Pandas DataFrame'i normalize eder"""
        if df.empty:
            return pd.DataFrame()
        
        normalized_cols = {}
        for col in df.columns:
            if pd.api.types.is_numeric_dtype(df[col]):
                normalized_cols[col] = Normalizer._normalize_pandas_series(df[col], method, target_range, **params)
            else:
                normalized_cols[col] = df[col]
        
        return pd.DataFrame(normalized_cols, index=df.index)
    
    @staticmethod
    def _normalize_list(data: list, method: str, target_range: Tuple[float, float], **params) -> list:
        """List'ı normalize eder"""
        if not data:
            return []
        
        # Listeyi pandas series'e çevir ve normalize et
        series = pd.Series(data)
        normalized_series = Normalizer._normalize_pandas_series(series, method, target_range, **params)
        return normalized_series.tolist()
    
    @staticmethod
    def _normalize_dict(data: dict, method: str, target_range: Tuple[float, float], **params) -> dict:
        """Dict'i normalize eder (sadece numeric values)"""
        normalized = {}
        for key, value in data.items():
            if isinstance(value, (int, float, np.number)):
                normalized[key] = Normalizer._normalize_scalar(float(value), method, target_range, **params)
            else:
                normalized[key] = value
        return normalized


# ==================== METRIC FINALIZATION ====================
def finalize_metric(
    raw_result: Any,
    metric_name: str,
    config: Optional[Dict[str, Any]] = None
) -> float:
    """
    Ham metrik sonucunu final float'a çevirir.
    
    Args:
        raw_result: Metrik fonksiyonunun ham çıktısı
        metric_name: Metrik adı (config'de bulmak için)
        config: Modül config'i (None ise _MODULE_CONFIG kullanır)
    
    Returns:
        -1.0 ile 1.0 arasında normalize edilmiş float
    """
    if config is None:
        config = _MODULE_CONFIG
    
    # 1. Ham sonuçtan skalar değer çıkar
    scalar_value = extract_scalar_from_result(raw_result)
    
    # 2. Config'den normalizasyon ayarlarını al
    norm_config = config.get("normalization", {})
    global_range = norm_config.get("global_range", {"min": -1.0, "max": 1.0})
    target_range = (global_range["min"], global_range["max"])
    
    # Metrik-specific normalizasyon var mı?
    metric_specific = norm_config.get("metric_specific", {}).get(metric_name, {})
    if metric_specific:
        method = metric_specific.get("method", norm_config.get("default_method", NormalizationMethod.TANH))
        params = metric_specific.get("params", {})
    else:
        method = norm_config.get("default_method", NormalizationMethod.TANH)
        params = {}
    
    # 3. Normalize et
    normalized = Normalizer.normalize(
        scalar_value,
        method=method,
        target_range=target_range,
        **params
    )
    
    # 4. Float'a çevir ve clamp
    try:
        final_float = float(normalized)
    except (TypeError, ValueError):
        final_float = 0.0
    
    return np.clip(final_float, -1.0, 1.0)


def extract_scalar_from_result(result: Any) -> float:
    """
    Herhangi bir metrik sonucundan skalar değer çıkarır.
    
    Supports: float, pd.Series, pd.DataFrame, np.ndarray, list, dict
    """
    if result is None:
        return 0.0
    
    # 1. Zaten skalar
    if isinstance(result, (int, float, np.number)):
        return float(result)
    
    # 2. Pandas Series
    elif isinstance(result, pd.Series):
        if result.empty:
            return 0.0
        # NaN olmayan son değer
        non_nan = result[result.notna()]
        if non_nan.empty:
            return 0.0
        return float(non_nan.iloc[-1])
    
    # 3. Pandas DataFrame
    elif isinstance(result, pd.DataFrame):
        if result.empty:
            return 0.0
        # İlk kolonun son değeri
        first_col = result.columns[0]
        non_nan = result[first_col].dropna()
        if non_nan.empty:
            return 0.0
        return float(non_nan.iloc[-1])
    
    # 4. NumPy array
    elif isinstance(result, np.ndarray):
        if result.size == 0:
            return 0.0
        # Flatten ve NaN olmayan son değer
        flattened = result.flatten()
        non_nan = flattened[~np.isnan(flattened)]
        if non_nan.size == 0:
            return 0.0
        return float(non_nan[-1])
    
    # 5. List
    elif isinstance(result, list):
        if not result:
            return 0.0
        # Son eleman
        last_item = result[-1]
        if isinstance(last_item, (int, float, np.number)):
            return float(last_item)
        else:
            return 0.0
    
    # 6. Dict (ilk numeric value)
    elif isinstance(result, dict):
        for value in result.values():
            if isinstance(value, (int, float, np.number)):
                return float(value)
        return 0.0
    
    # 7. Diğer
    else:
        try:
            return float(result)
        except (TypeError, ValueError):
            return 0.0


# ==================== PURE METRIC FUNCTIONS ====================
def ofi(data: pd.DataFrame, **params) -> float:
    """
    Order Flow Imbalance (Cont, Stoikov & Talreja, 2014)
    OFI = ΔBid_Size (if bid_price up) - ΔAsk_Size (if ask_price down)
    
    Args:
        data: DataFrame containing 'bid_price', 'bid_size', 'ask_price', 'ask_size'
    
    Returns:
        -1.0 to 1.0 normalized float
    """
    required_cols = ["bid_price", "bid_size", "ask_price", "ask_size"]
    if not all(col in data.columns for col in required_cols):
        raise ValueError(f"ofi: required columns {required_cols} not found")

    bid_price = data["bid_price"].values
    bid_size = data["bid_size"].values
    ask_price = data["ask_price"].values
    ask_size = data["ask_size"].values

    d_bid = np.diff(bid_size, prepend=bid_size[0])
    d_ask = np.diff(ask_size, prepend=ask_size[0])

    bid_up = np.full_like(bid_price, False, dtype=bool)
    ask_down = np.full_like(ask_price, False, dtype=bool)
    
    if len(bid_price) > 1:
        bid_up[1:] = bid_price[1:] >= bid_price[:-1]
        ask_down[1:] = ask_price[1:] <= ask_price[:-1]

    ofi_values = np.where(bid_up, d_bid, 0.0) - np.where(ask_down, d_ask, 0.0)
    ofi_values = np.where(np.abs(ofi_values) > 1e10, np.nan, ofi_values)
    
    ofi_series = pd.Series(ofi_values, index=data.index)
    
    # Return normalized final value
    return finalize_metric(ofi_series, "ofi")


def cvd(data: pd.DataFrame, **params) -> float:
    """
    Cumulative Volume Delta.
    CVD = cumulative sum of (buy_volume - sell_volume)
    
    Args:
        data: DataFrame containing 'buy_volume', 'sell_volume'
    
    Returns:
        -1.0 to 1.0 normalized float
    """
    required_cols = ["buy_volume", "sell_volume"]
    if not all(col in data.columns for col in required_cols):
        raise ValueError(f"cvd: required columns {required_cols} not found")

    buy_volume = data["buy_volume"].values
    sell_volume = data["sell_volume"].values

    delta = buy_volume - sell_volume
    cvd_values = np.cumsum(delta)
    cvd_values = np.where(np.abs(cvd_values) > 1e15, np.nan, cvd_values)
    
    cvd_series = pd.Series(cvd_values, index=data.index)
    
    # Return normalized final value
    return finalize_metric(cvd_series, "cvd")


def microprice_deviation(data: pd.DataFrame, **params) -> float:
    """
    Microprice deviation from midprice.
    
    Args:
        data: DataFrame containing 'best_bid', 'best_ask', 'bid_size', 'ask_size'
    
    Returns:
        -1.0 to 1.0 normalized float
    """
    required_cols = ["best_bid", "best_ask", "bid_size", "ask_size"]
    if not all(col in data.columns for col in required_cols):
        raise ValueError(f"microprice_deviation: required columns {required_cols} not found")

    best_bid = data["best_bid"].values
    best_ask = data["best_ask"].values
    bid_size = data["bid_size"].values
    ask_size = data["ask_size"].values

    mid = (best_bid + best_ask) / 2.0
    total_size = bid_size + ask_size
    safe_total = np.where(total_size == 0, 1.0, total_size)
    
    micro = (best_ask * bid_size + best_bid * ask_size) / safe_total
    deviation = micro - mid
    deviation = np.where(np.abs(deviation) > 1e10, np.nan, deviation)
    
    deviation_series = pd.Series(deviation, index=data.index)
    
    # Return normalized final value
    return finalize_metric(deviation_series, "microprice_deviation")


def market_impact(data: pd.DataFrame, **params) -> float:
    """
    Rolling correlation between |ΔP| and trade volume.
    Impact = Corr(|ΔP|, Volume)
    
    Args:
        data: DataFrame containing 'trade_volume', 'price_series'
        window: Rolling window size (optional parameter)
    
    Returns:
        -1.0 to 1.0 normalized float
    """
    required_cols = ["trade_volume", "price_series"]
    if not all(col in data.columns for col in required_cols):
        raise ValueError(f"market_impact: required columns {required_cols} not found")

    trade_volume = data["trade_volume"].values
    price_series = data["price_series"].values
    window = params.get('window', 20)

    n = len(price_series)
    price_change = np.abs(np.diff(price_series, prepend=price_series[0]))
    impact = np.full(n, np.nan, dtype=float)

    for i in range(window, n):
        pv = trade_volume[i - window:i]
        pr = price_change[i - window:i]
        
        mask = ~(np.isnan(pv) | np.isnan(pr))
        pv_clean = pv[mask]
        pr_clean = pr[mask]
        
        if len(pv_clean) < 2 or np.std(pv_clean) == 0 or np.std(pr_clean) == 0:
            impact[i] = np.nan
        else:
            corr_matrix = np.corrcoef(pv_clean, pr_clean)
            impact[i] = corr_matrix[0, 1] if not np.isnan(corr_matrix[0, 1]) else np.nan

    impact_series = pd.Series(impact, index=data.index)
    
    # Return normalized final value
    return finalize_metric(impact_series, "market_impact")


def depth_elasticity(data: pd.DataFrame, **params) -> float:
    """
    Elasticity of order book depth:
    E = %ΔVolume / %ΔPrice
    
    Args:
        data: DataFrame containing 'depth_price', 'depth_volume'
        window: Smoothing window (optional parameter)
    
    Returns:
        -1.0 to 1.0 normalized float
    """
    required_cols = ["depth_price", "depth_volume"]
    if not all(col in data.columns for col in required_cols):
        raise ValueError(f"depth_elasticity: required columns {required_cols} not found")

    depth_price = data["depth_price"].values
    depth_volume = data["depth_volume"].values
    window = params.get('window', 10)

    pct_price = np.diff(depth_price, prepend=depth_price[0]) / (depth_price + 1e-12)
    pct_volume = np.diff(depth_volume, prepend=depth_volume[0]) / (depth_volume + 1e-12)
    
    elasticity = np.divide(pct_volume, pct_price, 
                          out=np.full_like(pct_volume, np.nan), 
                          where=np.abs(pct_price) > 1e-12)

    if window > 1 and len(elasticity) >= window:
        elasticity_smooth = np.full_like(elasticity, np.nan)
        for i in range(window - 1, len(elasticity)):
            window_data = elasticity[i - window + 1: i + 1]
            window_clean = window_data[~np.isnan(window_data)]
            if len(window_clean) > 0:
                elasticity_smooth[i] = np.mean(window_clean)
        elasticity = elasticity_smooth

    elasticity_series = pd.Series(elasticity, index=data.index)
    
    # Return normalized final value
    return finalize_metric(elasticity_series, "depth_elasticity")


def taker_dominance_ratio(data: pd.DataFrame, **params) -> float:
    """
    Aggressive taker dominance ratio.
    > 1 → buyer dominance, < 1 → seller dominance
    
    Args:
        data: DataFrame containing 'taker_buy_volume', 'taker_sell_volume'
    
    Returns:
        -1.0 to 1.0 normalized float
    """
    required_cols = ["taker_buy_volume", "taker_sell_volume"]
    if not all(col in data.columns for col in required_cols):
        raise ValueError(f"taker_dominance_ratio: required columns {required_cols} not found")

    # Güvenli float dönüşümü
    taker_buy_volume = data["taker_buy_volume"].astype(float)
    taker_sell_volume = data["taker_sell_volume"].astype(float)
    
    # Sıfıra bölme durumunu kontrol et
    with np.errstate(divide='ignore', invalid='ignore'):
        ratio = taker_buy_volume / taker_sell_volume
    
    # Aşırı değerleri ve sıfıra bölme durumlarını kontrol et
    ratio = ratio.replace([np.inf, -np.inf], np.nan)
    ratio = ratio.clip(lower=-1e6, upper=1e6)  # Çok büyük değerleri sınırla
    
    # Return normalized final value
    return finalize_metric(ratio, "taker_dominance_ratio")







# ----- 1️⃣ Liquidity density fonksiyonu -----
def liquidity_density(data: pd.DataFrame, tick_range: int = 10, normalize: bool = True) -> float:
    """
    Average liquidity per price tick, computed as bid_size + ask_size per level.
    """
    if not all(col in data.columns for col in ["bid_size", "ask_size"]):
        raise ValueError("Data must contain 'bid_size' and 'ask_size' columns")
    
    depth_volume = (data["bid_size"] + data["ask_size"]).values
    
    if len(depth_volume) < tick_range:
        avg_density = np.nan
    else:
        density_values = np.array([
            np.mean(depth_volume[max(i - tick_range + 1, 0): i + 1])
            for i in range(len(depth_volume))
        ])
        avg_density = np.nanmean(density_values)
    
    if normalize and not np.isnan(avg_density):
        return np.tanh(avg_density)
    
    return avg_density

# ----- 2️⃣ Binance depth verisini çek ve DataFrame oluştur -----
def get_order_book(symbol: str, limit: int = 500) -> pd.DataFrame:
    """
    Fetch order book from Binance API and return as DataFrame with bid_size and ask_size
    """
    url = f"https://api.binance.com/api/v3/depth?symbol={symbol}&limit={limit}"
    resp = requests.get(url).json()
    
    bids = pd.DataFrame(resp['bids'], columns=['price', 'bid_size']).astype(float)
    asks = pd.DataFrame(resp['asks'], columns=['price', 'ask_size']).astype(float)
    
    # Bid ve ask'leri tek DataFrame'de birleştir
    df = pd.DataFrame({
        'bid_size': bids['bid_size'],
        'ask_size': asks['ask_size']
    })
    return df





# ==================== REGISTRY ====================
_METRICS = {
    "ofi": ofi,
    "cvd": cvd,
    "microprice_deviation": microprice_deviation,
    "market_impact": market_impact,
    "depth_elasticity": depth_elasticity,
    "taker_dominance_ratio": taker_dominance_ratio,
    "liquidity_density": liquidity_density
}


def get_metrics() -> List[str]:
    """Kullanılabilir metriklerin listesini döndürür"""
    return list(_METRICS.keys())


def get_function(metric_name: str):
    """Metrik fonksiyonunu döndürür"""
    return _METRICS.get(metric_name)


def get_module_config() -> Dict[str, Any]:
    """Modül konfigürasyonunu döndürür"""
    return _MODULE_CONFIG.copy()


# ==================== SELF-TEST ====================
def self_test() -> Dict[str, bool]:
    """Modülün kendi kendini test etmesi"""
    results = {}
    
    # Test verisi - microstructure için uygun veri yapısı
    test_data = pd.DataFrame({
        'bid_price': [100, 101, 102, 101, 100, 99, 100, 101, 102, 103],
        'bid_size': [10, 12, 15, 14, 13, 11, 12, 14, 16, 15],
        'ask_price': [101, 102, 103, 102, 101, 100, 101, 102, 103, 104],
        'ask_size': [8, 9, 10, 11, 10, 9, 8, 9, 10, 11],
        'buy_volume': [100, 150, 200, 120, 180, 90, 110, 130, 160, 140],
        'sell_volume': [80, 120, 180, 100, 160, 70, 90, 110, 140, 120],
        'best_bid': [100, 101, 102, 101, 100, 99, 100, 101, 102, 103],
        'best_ask': [101, 102, 103, 102, 101, 100, 101, 102, 103, 104],
        'trade_volume': [50, 60, 70, 55, 65, 45, 50, 60, 70, 65],
        'price_series': [100.5, 101.5, 102.5, 101.5, 100.5, 99.5, 100.5, 101.5, 102.5, 103.5],
        'depth_price': [100, 101, 102, 101, 100, 99, 100, 101, 102, 103],
        'depth_volume': [1000, 1100, 1200, 1150, 1050, 950, 1000, 1100, 1200, 1150],
        'taker_buy_volume': [30, 40, 50, 35, 45, 25, 30, 40, 50, 45],
        'taker_sell_volume': [20, 30, 40, 25, 35, 15, 20, 30, 40, 35]
    })
    
    print(f"🧪 {__name__} Self-Test")
    print("=" * 60)
    
    # Her metriği test et
    for metric_name, metric_func in _METRICS.items():
        try:
            result = metric_func(test_data)
            
            # Kontroller
            is_float = isinstance(result, float)
            in_range = -1.0 <= result <= 1.0 if is_float else False
            not_nan = not np.isnan(result) if is_float else False
            
            passed = is_float and in_range and not_nan
            
            if passed:
                print(f"✅ {metric_name}: {result:.4f}")
            else:
                print(f"❌ {metric_name}: {result} (float:{is_float}, range:{in_range}, !nan:{not_nan})")
            
            results[metric_name] = passed
            
        except Exception as e:
            print(f"❌ {metric_name}: ERROR - {e}")
            results[metric_name] = False
    
    # Config test
    try:
        config = get_module_config()
        has_data_model = "data_model" in config
        has_required_columns = "required_columns" in config
        print(f"\n📋 Config: data_model={has_data_model}, required_columns={has_required_columns}")
        results["config_valid"] = has_data_model and has_required_columns
    except Exception as e:
        print(f"❌ Config ERROR: {e}")
        results["config_valid"] = False
    
    print(f"\n📊 Results: {sum(results.values())}/{len(results)} passed")
    return results


if __name__ == "__main__":
    self_test()