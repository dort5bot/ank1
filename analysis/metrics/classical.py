"""
analysis/metrics/classical.py
Universal Metric Template with Data-Model-Aware Normalization
date: 03/12/2025
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


# ==================== MODULE CONFIG classical.py====================
_MODULE_CONFIG = {
    # ZORUNLU: Veri modeli ve execution tipi
    "data_model": DataModel.PANDAS,
    "execution_type": "sync",
    "category": "technical",
    
    # ZORUNLU: Her metrik için gerekli kolonlar
    "required_columns": {
        "adx": ["high", "low", "close"],
        "atr": ["high", "low", "close"],
        "bollinger_bands": ["close"],
        "cross_correlation": ["close", "close"],  # İki farklı close serisi
        "conditional_value_at_risk": ["close"],
        "ema": ["close"],
        "futures_roc": ["close"],
        "macd": ["close"],
        "max_drawdown": ["close"],
        "oi_growth_rate": ["open_interest"],
        "oi_price_correlation": ["price", "open_interest"],
        "roc": ["close"],
        "rsi": ["close"],
        "sma": ["close"],
        "spearman_corr": ["close", "close"],  # İki farklı close serisi
        "stochastic_oscillator": ["high", "low", "close"],
        "value_at_risk": ["close"]
    },
    
    # ÖNERİLEN: Normalizasyon konfigürasyonu
    "normalization": {
        # Global normalizasyon ayarları
        "global_range": {"min": -1.0, "max": 1.0},
        "default_method": NormalizationMethod.TANH,
        
        # Metrik-specific normalizasyon
        "metric_specific": {
            "adx": {
                "method": NormalizationMethod.MINMAX,
                "params": {"data_min": 0, "data_max": 100},
                "description": "0-100 aralığı, 25 altı zayıf trend, 50 üstü güçlü trend"
            },
            "atr": {
                "method": NormalizationMethod.ZSCORE,
                "params": {"window": 20, "clip_sigma": 3.0, "scale": 50.0},
                "description": "Z-score with 3 sigma clipping, negative direction"
            },
            "bollinger_bands": {
                "method": NormalizationMethod.MINMAX,
                "params": {"data_min": -1, "data_max": 1},
                "description": "-1 (lower band) to +1 (upper band)"
            },
            "ema": {
                "method": NormalizationMethod.ZSCORE,
                "params": {"window": 20},
                "description": "Price deviation from EMA"
            },
            "macd": {
                "method": NormalizationMethod.MINMAX,
                "params": {"data_min": -1, "data_max": 1},
                "description": "MACD histogram normalized"
            },
            "rsi": {
                "method": NormalizationMethod.MINMAX,
                "params": {"data_min": 0, "data_max": 100},
                "description": "0-100 aralığı, 30 altı oversold, 70 üstü overbought"
            },
            "stochastic_oscillator": {
                "method": NormalizationMethod.MINMAX,
                "params": {"data_min": 0, "data_max": 100},
                "description": "0-100 aralığı, 20 altı oversold, 80 üstü overbought"
            },
            "roc": {
                "method": NormalizationMethod.ZSCORE,
                "params": {"window": 20, "clip_sigma": 3.0},
                "description": "Z-score normalized rate of change"
            },
            "oi_growth_rate": {
                "method": NormalizationMethod.ZSCORE,
                "params": {"window": 20},
                "description": "Open interest growth z-score"
            },
            "oi_price_correlation": {
                "method": NormalizationMethod.MINMAX,
                "params": {"data_min": -1, "data_max": 1},
                "description": "-1 to +1 correlation coefficient"
            }
        }
    },
    
    # OPSİYONEL: Metrik parametreleri
    "default_params": {
        "adx": {"period": 14},
        "atr": {"period": 14},
        "bollinger_bands": {"period": 20, "std_factor": 2.0},
        "ema": {"period": 14},
        "macd": {"fast": 12, "slow": 26, "signal": 9},
        "rsi": {"period": 14},
        "stochastic_oscillator": {"period": 14},
        "roc": {"period": 1},
        "oi_growth_rate": {"period": 7},
        "oi_price_correlation": {"window": 14},
        "value_at_risk": {"confidence": 0.95},
        "conditional_value_at_risk": {"confidence": 0.95}
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
            # Eğer data_min/data_max verilmişse onları kullan
            data_min = params.get('data_min', series.min())
            data_max = params.get('data_max', series.max())
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
            # Sigma clipping
            clip_sigma = params.get('clip_sigma', None)
            if clip_sigma is not None:
                zscore = zscore.clip(lower=-clip_sigma, upper=clip_sigma)
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
    def _normalize_pandas_df(
        df: pd.DataFrame,
        method: str,
        target_range: Tuple[float, float],
        **params
    ) -> pd.DataFrame:
        """Pandas DataFrame'i normalize eder (her kolon için)"""
        normalized_cols = {}
        for col in df.columns:
            normalized_cols[col] = Normalizer._normalize_pandas_series(df[col], method, target_range, **params)
        return pd.DataFrame(normalized_cols, index=df.index)
    
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

    # # Ham metrik sonucunu final float'a çevirir.
    
    # # Args:
        # # raw_result: Metrik fonksiyonunun ham çıktısı
        # # metric_name: Metrik adı (config'de bulmak için)
        # # config: Modül config'i (None ise _MODULE_CONFIG kullanır)
    
    # # Returns:
        # # -1.0 ile 1.0 arasında normalize edilmiş float

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


# classical.py'de finalize_metric fonksiyonuna debug ekle
"""def finalize_metric(
    raw_result: Any,
    metric_name: str,
    config: Optional[Dict[str, Any]] = None
) -> float:
    print(f"\n🔧 DEBUG finalize_metric START")
    print(f"   Metric: {metric_name}")
    print(f"   Raw result type: {type(raw_result)}")
    print(f"   Raw result value (first 3 if series): {raw_result.head(3) if isinstance(raw_result, pd.Series) else raw_result}")
    
    if config is None:
        config = _MODULE_CONFIG
    
    # 1. Ham sonuçtan skalar değer çıkar
    scalar_value = extract_scalar_from_result(raw_result)
    print(f"   Scalar value: {scalar_value}")
    
    # 2. Config'den normalizasyon ayarlarını al
    norm_config = config.get("normalization", {})
    global_range = norm_config.get("global_range", {"min": -1.0, "max": 1.0})
    target_range = (global_range["min"], global_range["max"])
    print(f"   Target range: {target_range}")
    
    # Metrik-specific normalizasyon var mı?
    metric_specific = norm_config.get("metric_specific", {}).get(metric_name, {})
    print(f"   Metric specific config: {metric_specific}")
    
    if metric_specific:
        method = metric_specific.get("method", norm_config.get("default_method", NormalizationMethod.TANH))
        params = metric_specific.get("params", {})
    else:
        method = norm_config.get("default_method", NormalizationMethod.TANH)
        params = {}
    
    print(f"   Normalization method: {method}")
    print(f"   Normalization params: {params}")
    
    # 3. Normalize et
    print(f"   Calling Normalizer.normalize...")
    normalized = Normalizer.normalize(
        scalar_value,
        method=method,
        target_range=target_range,
        **params
    )
    print(f"   Normalized value: {normalized}")
    
    # 4. Float'a çevir ve clamp
    try:
        final_float = float(normalized)
    except (TypeError, ValueError):
        final_float = 0.0
    
    final_result = np.clip(final_float, -1.0, 1.0)
    print(f"   Final result (clamped): {final_result}")
    print(f"🔧 DEBUG finalize_metric END\n")
    
    return final_result
"""
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
# ==========================================================
# === Trend & Moving averages ==============================
# ==========================================================

def ema(data: pd.DataFrame, **params) -> float:

    period = params.get("period", 14)
    if "close" not in data.columns:
        raise ValueError("ema: data must contain 'close' column")
    
    close = data["close"]
    ema_value = close.ewm(span=period, adjust=False).mean()
    
    # Price deviation from EMA (as percentage)
    deviation = (close - ema_value) / (ema_value + 1e-10)
    
    return finalize_metric(deviation, "ema")


def sma(data: pd.DataFrame, **params) -> float:
    """Simple Moving Average - Normalized deviation from price"""
    period = params.get("period", 14)
    if "close" not in data.columns:
        raise ValueError("sma: data must contain 'close' column")
    
    close = data["close"]
    sma_value = close.rolling(window=period, min_periods=1).mean()
    
    # Price deviation from SMA (as percentage)
    deviation = (close - sma_value) / (sma_value + 1e-10)
    
    return finalize_metric(deviation, "sma")


def macd(data: pd.DataFrame, **params) -> float:
    """Moving Average Convergence Divergence - Histogram normalized"""
    fast = params.get("fast", 12)
    slow = params.get("slow", 26)
    signal = params.get("signal", 9)
    
    if "close" not in data.columns:
        raise ValueError("macd: data must contain 'close' column")
    
    close = data["close"]
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    histogram = macd_line - signal_line
    
    return finalize_metric(histogram, "macd")


def rsi(data: pd.DataFrame, **params) -> float:
    """Relative Strength Index - 0-100 normalized to -1..1"""
    period = params.get("period", 14)
    if "close" not in data.columns:
        raise ValueError("rsi: data must contain 'close' column")
    
    close = data["close"]
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window=period, min_periods=1).mean()
    avg_loss = loss.rolling(window=period, min_periods=1).mean()
    rs = avg_gain / (avg_loss + 1e-10)
    rsi_value = 100 - (100 / (1 + rs))
    
    return finalize_metric(rsi_value, "rsi")


def adx(data: pd.DataFrame, **params) -> float:
    """Average Directional Index - 0-100 normalized to -1..1"""
    period = params.get("period", 14)
    required_cols = ["high", "low", "close"]
    if not all(c in data.columns for c in required_cols):
        raise ValueError(f"adx: required columns {required_cols} not found")
    
    high = data["high"]
    low = data["low"]
    close = data["close"]
    
    plus_dm = high.diff().clip(lower=0)
    minus_dm = (-low.diff()).clip(lower=0)
    
    tr = pd.concat([
        high - low,
        (high - close.shift()).abs(),
        (low - close.shift()).abs()
    ], axis=1).max(axis=1)

    atr_val = tr.rolling(period, min_periods=1).mean()
    plus_di = 100 * (plus_dm.rolling(period, min_periods=1).mean() / (atr_val + 1e-10))
    minus_di = 100 * (minus_dm.rolling(period, min_periods=1).mean() / (atr_val + 1e-10))
    dx = (abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)) * 100
    adx_value = dx.rolling(period, min_periods=1).mean()
    
    return finalize_metric(adx_value, "adx")


def stochastic_oscillator(data: pd.DataFrame, **params) -> float:
    """Stochastic Oscillator - 0-100 normalized to -1..1"""
    period = params.get("period", 14)
    required_cols = ["high", "low", "close"]
    if not all(c in data.columns for c in required_cols):
        raise ValueError(f"stochastic_oscillator: required columns {required_cols} not found")
    
    high = data["high"]
    low = data["low"]
    close = data["close"]
    
    lowest_low = low.rolling(window=period).min()
    highest_high = high.rolling(window=period).max()
    stoch_value = ((close - lowest_low) / (highest_high - lowest_low + 1e-10)) * 100
    
    return finalize_metric(stoch_value, "stochastic_oscillator")


def roc(data: pd.DataFrame, **params) -> float:
    """Rate of Change - Price change percentage z-score normalized"""
    period = params.get("period", 1)
    if "close" not in data.columns:
        raise ValueError("roc: data must contain 'close' column")
    
    close = data["close"]
    roc_value = close.pct_change(periods=period, fill_method=None) * 100

    return finalize_metric(roc_value, "roc")


# ==========================================================
# === Volatility metrics ===================================
# ==========================================================

def atr(data: pd.DataFrame, **params) -> float:
    """Average True Range - Z-score normalized"""
    period = params.get("period", 14)
    required_cols = ["high", "low", "close"]
    if not all(c in data.columns for c in required_cols):
        raise ValueError(f"atr: required columns {required_cols} not found")
    
    high = data["high"]
    low = data["low"]
    close = data["close"]
    
    tr = pd.concat([
        (high - low),
        (high - close.shift()).abs(),
        (low - close.shift()).abs()
    ], axis=1).max(axis=1)
    
    atr_value = tr.rolling(window=period, min_periods=1).mean()
    
    return finalize_metric(atr_value, "atr")


def bollinger_bands(data: pd.DataFrame, **params) -> float:
    """Bollinger Bands - Position within bands (-1 to +1)"""
    period = params.get("period", 20)
    std_factor = params.get("std_factor", 2.0)
    
    if "close" not in data.columns:
        raise ValueError("bollinger_bands: data must contain 'close' column")
    
    close = data["close"]
    sma_val = close.rolling(window=period, min_periods=1).mean()
    std = close.rolling(window=period, min_periods=1).std()
    upper = sma_val + std_factor * std
    lower = sma_val - std_factor * std
    
    # Position within bands: -1 (at lower) to +1 (at upper)
    band_position = (2 * (close - lower) / (upper - lower + 1e-10)) - 1
    
    return finalize_metric(band_position, "bollinger_bands")

# ==========================================================
# === Risk metrics =========================================
# ==========================================================

def value_at_risk(data: pd.DataFrame, **params) -> float:
    """Value at Risk (percentile-based) - Negative values normalized"""
    confidence = params.get("confidence", 0.95)
    
    if "close" not in data.columns:
        raise ValueError("value_at_risk: data must contain 'close' column")
    
    close = data["close"]
    returns = close.pct_change().dropna()
    if len(returns) == 0:
        return finalize_metric(0.0, "value_at_risk")
    
    var_val = np.percentile(returns, (1 - confidence) * 100)
    
    return finalize_metric(var_val, "value_at_risk")


def conditional_value_at_risk(data: pd.DataFrame, **params) -> float:
    """Conditional Value at Risk (expected shortfall) - Negative values normalized"""
    confidence = params.get("confidence", 0.95)
    
    if "close" not in data.columns:
        raise ValueError("conditional_value_at_risk: data must contain 'close' column")
    
    close = data["close"]
    returns = close.pct_change().dropna()
    if len(returns) == 0:
        return finalize_metric(0.0, "conditional_value_at_risk")
    
    var_val = np.percentile(returns, (1 - confidence) * 100)
    cvar = returns[returns <= var_val].mean()
    
    return finalize_metric(cvar, "conditional_value_at_risk")


def max_drawdown(data: pd.DataFrame, **params) -> float:
    """Maximum Drawdown - Negative values normalized"""
    if "close" not in data.columns:
        raise ValueError("max_drawdown: data must contain 'close' column")
    
    close = data["close"]
    roll_max = close.cummax()
    drawdown = (close - roll_max) / (roll_max + 1e-10)
    max_dd = drawdown.min()
    
    return finalize_metric(max_dd, "max_drawdown")


# ==========================================================
# === Open Interest & Market structure =====================
# ==========================================================

def oi_growth_rate(data: pd.DataFrame, **params) -> float:
    """Open Interest Growth Rate - Z-score normalized"""
    period = params.get("period", 7)
    
    if "open_interest" not in data.columns:
        raise ValueError("oi_growth_rate: data must contain 'open_interest' column")
    
    oi_series = data["open_interest"]
    growth_rate = oi_series.pct_change(periods=period).fillna(0)
    
    return finalize_metric(growth_rate, "oi_growth_rate")


def oi_price_correlation(data: pd.DataFrame, **params) -> float:
    """Rolling correlation between Open Interest and Price - -1 to +1"""
    window = params.get("window", 14)
    required_cols = ["price", "open_interest"]
    if not all(c in data.columns for c in required_cols):
        raise ValueError(f"oi_price_correlation: required columns {required_cols} not found")
    
    price_series = data["price"]
    oi_series = data["open_interest"]
    correlation = oi_series.rolling(window=window, min_periods=1).corr(price_series)
    
    return finalize_metric(correlation, "oi_price_correlation")


# ==========================================================
# === Correlation metrics ==================================
# ==========================================================

def spearman_corr(data: pd.DataFrame, **params) -> float:
    """Spearman rank correlation coefficient - -1 to +1"""
    if len(data.columns) < 2:
        raise ValueError("spearman_corr: data must contain at least two columns")
    
    series_x = data.iloc[:, 0]
    series_y = data.iloc[:, 1]
    
    aligned_x, aligned_y = series_x.align(series_y, join='inner')
    
    if len(aligned_x) < 2:
        return finalize_metric(0.0, "spearman_corr")
    
    # Daha verimli Spearman korelasyonu
    # 1. Rank hesapla
    x_rank = aligned_x.rank()
    y_rank = aligned_y.rank()
    
    # 2. Rolling Pearson korelasyonu rank'lar üzerinde (bu Spearman'dır)
    window = min(20, len(x_rank))
    rolling_corr = x_rank.rolling(window=window).corr(y_rank)
    
    # Son değeri al
    if rolling_corr.notna().any():
        last_corr = rolling_corr.dropna().iloc[-1]
        return finalize_metric(last_corr, "spearman_corr")
    else:
        # Tüm seri için Spearman
        correlation = x_rank.corr(y_rank)
        return finalize_metric(correlation, "spearman_corr")

def cross_correlation(data: pd.DataFrame, **params) -> float:
    """Cross-correlation between two series - -1 to +1"""
    max_lag = params.get("max_lag", 10)
    
    if len(data.columns) < 2:
        raise ValueError("cross_correlation: data must contain at least two columns")
    
    series_x = data.iloc[:, 0]
    series_y = data.iloc[:, 1]
    
    aligned_x, aligned_y = series_x.align(series_y, join='inner')
    if len(aligned_x) == 0:
        return finalize_metric(0.0, "cross_correlation")
    
    # Basit korelasyon hesapla (gecikmesiz)
    corr = aligned_x.corr(aligned_y)
    
    return finalize_metric(corr, "cross_correlation")


# ==========================================================
# === Futures metrics ======================================
# ==========================================================

def futures_roc(data: pd.DataFrame, **params) -> float:
    """Futures Price Change - Same as ROC but named differently"""
    period = params.get("period", 1)
    
    if "close" not in data.columns:
        raise ValueError("futures_roc: data must contain 'close' column")
    
    futures_series = data["close"]
    roc_value = futures_series.pct_change(periods=period) * 100
    
    return finalize_metric(roc_value, "futures_roc")


# ==================== REGISTRY ====================
_METRICS = {
    "adx": adx,
    "atr": atr,
    "bollinger_bands": bollinger_bands,
    "cross_correlation": cross_correlation,
    "conditional_value_at_risk": conditional_value_at_risk,
    "ema": ema,
    "futures_roc": futures_roc,
    "macd": macd,
    "max_drawdown": max_drawdown,
    "oi_growth_rate": oi_growth_rate,
    "oi_price_correlation": oi_price_correlation,
    "roc": roc,
    "rsi": rsi,
    "sma": sma,
    "spearman_corr": spearman_corr,
    "stochastic_oscillator": stochastic_oscillator,
    "value_at_risk": value_at_risk
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
    
    # Test verisi (OHLC + open_interest)
    test_data = pd.DataFrame({
        'open': [100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110],
        'high': [105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115],
        'low': [95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105],
        'close': [100, 102, 101, 104, 103, 106, 105, 108, 107, 110, 112],
        'price': [100, 102, 101, 104, 103, 106, 105, 108, 107, 110, 112],
        'open_interest': [1000, 1050, 1100, 1150, 1200, 1250, 1300, 1350, 1400, 1450, 1500]
    })
    
    print(f"🧪 {__name__} Self-Test")
    print("=" * 60)
    
    # Her metriği test et
    for metric_name, metric_func in _METRICS.items():
        try:
            # Özel durum: cross_correlation ve spearman_corr için iki kolonlu data gerek
            if metric_name in ["cross_correlation", "spearman_corr"]:
                # İki farklı seri için
                test_data_special = test_data[['close', 'price']].copy()
                result = metric_func(test_data_special)
            else:
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