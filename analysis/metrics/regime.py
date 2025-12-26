"""
analysis/metrics/regime.py
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


# ==================== MODULE CONFIG regime.py====================
_MODULE_CONFIG = {
    # ZORUNLU: Veri modeli ve execution tipi
    "data_model": DataModel.PANDAS,
    "execution_type": "sync",
    "category": "regime",
    
    # ZORUNLU: Her metrik için gerekli kolonlar
    "required_columns": {
        "advance_decline_line": ["close"],
        "volume_leadership": ["close"],
        "performance_dispersion": ["close"],
    },
    
    # ÖNERİLEN: Normalizasyon konfigürasyonu
    "normalization": {
        # Global normalizasyon ayarları
        "global_range": {"min": -1.0, "max": 1.0},
        "default_method": NormalizationMethod.TANH,
        
        # Metrik-specific normalizasyon
        "metric_specific": {
            "advance_decline_line": {
                "method": NormalizationMethod.TANH,
                "params": {"scale": 0.01},
                "description": "Tanh normalization for cumulative AD line"
            },
            "volume_leadership": {
                "method": NormalizationMethod.MINMAX,
                "params": {},
                "description": "Already 0-1 range, scale to -1..1"
            },
            "performance_dispersion": {
                "method": NormalizationMethod.TANH,
                "params": {"scale": 10.0},
                "description": "Tanh for volatility dispersion"
            }
        }
    },
    
    # OPSİYONEL: Metrik parametreleri
    "default_params": {
        "advance_decline_line": {"threshold": 0.001},
        "volume_leadership": {"window": 10},
        "performance_dispersion": {"window": 15}
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
    def _normalize_pandas_df(
        df: pd.DataFrame,
        method: str,
        target_range: Tuple[float, float],
        **params
    ) -> pd.DataFrame:
        """Pandas DataFrame'i normalize eder (her kolon ayrı)"""
        if df.empty:
            return pd.DataFrame()
        
        normalized_cols = {}
        for col in df.columns:
            if pd.api.types.is_numeric_dtype(df[col]):
                normalized_cols[col] = Normalizer._normalize_pandas_series(
                    df[col], method, target_range, **params
                )
            else:
                normalized_cols[col] = df[col]
        
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
def advance_decline_line(data: pd.DataFrame, **params) -> float:
    """
    Basitleştirilmiş Advance-Decline Line
    Price series'den advances/declines hesaplar
    
    Args:
        data: Pandas DataFrame, 'close' kolonu gereklidir
        **params: threshold (float) vb. ek parametreler
    
    Returns:
        -1.0 ile 1.0 arasında normalize edilmiş float
    """
    # 1. Input validation (saf matematik - veri kontrolü minimal)
    if "close" not in data.columns:
        raise ValueError("advance_decline_line: 'close' column required")
    
    # 2. Saf hesaplama
    price_series = data["close"]
    threshold = params.get("threshold", 0.001)
    
    if len(price_series) < 20:
        # Yeterli veri yoksa nötr değer
        return 0.0
    
    returns = price_series.pct_change()
    advances = (returns > threshold).astype(int)
    declines = (returns < -threshold).astype(int)
    ad_line = (advances - declines).cumsum()
    
    # 3. Finalize metodu ile normalize edilmiş float döndür
    return finalize_metric(ad_line, "advance_decline_line")


def volume_leadership(data: pd.DataFrame, **params) -> float:
    """
    Basitleştirilmiş Volume Leadership
    Volatility'yi volume proxy'si olarak kullanır
    
    Args:
        data: Pandas DataFrame, 'close' kolonu gereklidir
        **params: window (int) vb. ek parametreler
    
    Returns:
        -1.0 ile 1.0 arasında normalize edilmiş float
    """
    # 1. Input validation
    if "close" not in data.columns:
        raise ValueError("volume_leadership: 'close' column required")
    
    # 2. Saf hesaplama
    price_series = data["close"]
    window = params.get("window", 10)
    
    if len(price_series) < window:
        return 0.0
    
    returns = np.log(price_series).diff()
    volatility = returns.rolling(window=window, min_periods=1).std()
    
    # Normalize et (0-1 arası - bu zaten orijinal fonksiyonun çıktısı)
    min_val = volatility.min()
    max_val = volatility.max()
    
    if pd.isna(min_val) or pd.isna(max_val) or abs(max_val - min_val) < 1e-10:
        leadership = pd.Series(0.0, index=volatility.index)
    else:
        leadership = (volatility - min_val) / (max_val - min_val + 1e-6)
    
    leadership = leadership.fillna(0.0)
    
    # 3. Finalize metodu ile normalize edilmiş float döndür
    # Not: Orijinal 0-1 aralığı -1..1'e dönüştürülecek
    return finalize_metric(leadership, "volume_leadership")


def performance_dispersion(data: pd.DataFrame, **params) -> float:
    """
    Basitleştirilmiş Performance Dispersion  
    Rolling window'daki return varyansı
    
    Args:
        data: Pandas DataFrame, 'close' kolonu gereklidir
        **params: window (int) vb. ek parametreler
    
    Returns:
        -1.0 ile 1.0 arasında normalize edilmiş float
    """
    # 1. Input validation
    if "close" not in data.columns:
        raise ValueError("performance_dispersion: 'close' column required")
    
    # 2. Saf hesaplama
    price_series = data["close"]
    window = params.get("window", 15)
    
    if len(price_series) < window:
        return 0.0
    
    returns = np.log(price_series).diff()
    dispersion = returns.rolling(window=window, min_periods=1).std()
    dispersion = dispersion.fillna(0.0)
    
    # 3. Finalize metodu ile normalize edilmiş float döndür
    return finalize_metric(dispersion, "performance_dispersion")


# ==================== REGISTRY ====================
_METRICS = {
    "advance_decline_line": advance_decline_line,
    "volume_leadership": volume_leadership,
    "performance_dispersion": performance_dispersion,
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
    
    # Test verisi (gerçekçi fiyat serisi)
    np.random.seed(42)
    n_periods = 100
    base_price = 100.0
    returns = np.random.normal(0.001, 0.02, n_periods)
    prices = base_price * np.exp(np.cumsum(returns))
    
    test_data = pd.DataFrame({
        'close': prices,
        'open': prices * (1 + np.random.normal(0, 0.005, n_periods)),
        'high': prices * (1 + np.abs(np.random.normal(0, 0.01, n_periods))),
        'low': prices * (1 - np.abs(np.random.normal(0, 0.01, n_periods)))
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