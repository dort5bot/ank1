"""
analysis/metrics/sentiment.py
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


# ==================== MODULE CONFIG sentiment.py====================
_MODULE_CONFIG = {
    # ZORUNLU: Veri modeli ve execution tipi
    "data_model": DataModel.PANDAS,
    "execution_type": "sync",
    "category": "sentiment",
    
    # ZORUNLU: Her metrik için gerekli kolonlar
    "required_columns": {
        "funding_rate": ["funding_rate"],
        "funding_premium": ["futures_price", "spot_price"],
        "oi_trend": ["open_interest"],
    },
    
    # ÖNERİLEN: Normalizasyon konfigürasyonu
    "normalization": {
        # Global normalizasyon ayarları
        "global_range": {"min": -1.0, "max": 1.0},
        "default_method": NormalizationMethod.TANH,
        
        # Metrik-specific normalizasyon
        "metric_specific": {
            "funding_rate": {
                "method": NormalizationMethod.MINMAX,
                # "params": {"lookback": 100, "data_min": -0.1, "data_max": 0.1},
                "params": {"lookback": 100, "data_min": -0.01, "data_max": 0.01},
                "description": "Min-max normalization in typical funding rate range [-0.1, 0.1]"
            },
            "funding_premium": {
                "method": NormalizationMethod.ZSCORE,
                # "params": {"window": 20, "clip_sigma": 5.0},
                "params": {"window": 20, "clip_sigma": 3.0},
                "description": "Z-score with 5 sigma clipping for percent range"
            },
            "oi_trend": {
                "method": NormalizationMethod.ZSCORE,
                # "params": {"window": 50, "clip_sigma": 3.0},
                "params": {"window": 20, "clip_sigma": 3.0},
                "description": "Z-score for trend direction"
            }
        }
    },
    
    # OPSİYONEL: Metrik parametreleri
    "default_params": {
        "funding_rate": {"window": 8},
        "funding_premium": {"window": 1},
        "oi_trend": {"window": 20}
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
            # Sigma clipping
            clip_sigma = params.get('clip_sigma', 3.0)
            zscore = np.clip(zscore, -clip_sigma, clip_sigma)
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
            # Lookback window kullan
            lookback = params.get('lookback', min(100, len(series)))
            data_min = series.rolling(window=lookback, min_periods=1).min()
            data_max = series.rolling(window=lookback, min_periods=1).max()
            
            # Fallback için global min/max
            if params.get('data_min') is not None:
                data_min = data_min.clip(lower=params['data_min'])
            if params.get('data_max') is not None:
                data_max = data_max.clip(upper=params['data_max'])
            
            mask = data_max != data_min
            normalized = pd.Series(0.0, index=series.index)
            normalized[mask] = ((series[mask] - data_min[mask]) / (data_max[mask] - data_min[mask])) * (target_max - target_min) + target_min
        
        elif method == NormalizationMethod.ZSCORE:
            # Rolling z-score
            window = params.get('window', min(20, len(series)))
            rolling_mean = series.rolling(window=window, min_periods=2).mean()
            rolling_std = series.rolling(window=window, min_periods=2).std().replace(0, 1.0)
            zscore = (series - rolling_mean) / rolling_std
            
            # Sigma clipping
            clip_sigma = params.get('clip_sigma', 3.0)
            zscore = zscore.clip(lower=-clip_sigma, upper=clip_sigma)
            
            # Tanh ile yumuşat
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
            normalized_cols[col] = Normalizer._normalize_pandas_series(df[col], method, target_range, **params)
        
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
def funding_rate(data: pd.DataFrame, **params) -> float:
    """
    Pure rolling mean funding rate.
    Returns: -1.0 to 1.0 normalized float
    """
    # 1. Input validation (saf matematik)
    if "funding_rate" not in data.columns:
        raise ValueError("funding_rate: data must contain 'funding_rate' column")
    
    # 2. Pure calculation
    window = params.get("window", 8)
    series = data["funding_rate"]
    
    if len(series) < window:
        return 0.0
    
    # 3. Rolling mean
    rolling_mean = series.rolling(window=window, min_periods=1).mean()
    
    # 4. Return normalized final value
    return finalize_metric(rolling_mean, "funding_rate")


def funding_premium(data: pd.DataFrame, **params) -> float:
    """
    Pure funding premium calculation.
    Returns: -1.0 to 1.0 normalized float
    """
    # 1. Input validation (saf matematik)
    futures_col = params.get("futures_column", "futures_price")
    spot_col = params.get("spot_column", "spot_price")
    
    if futures_col not in data.columns:
        raise ValueError(f"funding_premium: data must contain '{futures_col}' column")
    if spot_col not in data.columns:
        raise ValueError(f"funding_premium: data must contain '{spot_col}' column")
    
    # 2. Pure calculation
    futures_price = data[futures_col]
    spot_price = data[spot_col]
    
    with np.errstate(divide='ignore', invalid='ignore'):
        # Futures premium as percentage
        result = (futures_price / spot_price.replace(0, np.nan) - 1.0) * 100
    
    # 3. Optional rolling window
    window = params.get("window", 1)
    if window > 1:
        result = result.rolling(window=window, min_periods=1).mean()
    
    # 4. Return normalized final value
    return finalize_metric(result, "funding_premium")


def oi_trend(data: pd.DataFrame, **params) -> float:
    """
    Linear trend direction of Open Interest.
    Returns: -1.0 to 1.0 normalized float
    """
    # 1. Input validation (saf matematik)
    oi_col = params.get("oi_column", "open_interest")
    
    if oi_col not in data.columns:
        raise ValueError(f"oi_trend: data must contain '{oi_col}' column")
    
    # 2. Pure calculation
    window = params.get("window", 20)
    series = data[oi_col]
    
    if len(series) < window:
        return 0.0
    
    def _slope(x):
        if len(x) < 2:
            return np.nan
        x_axis = np.arange(len(x))
        return np.polyfit(x_axis, x, 1)[0]
    
    # 3. Rolling linear regression slope
    trend = series.rolling(window=window, min_periods=2).apply(_slope, raw=True)
    
    # 4. Return normalized final value
    return finalize_metric(trend, "oi_trend")


# ==================== REGISTRY ====================
_METRICS = {
    "funding_rate": funding_rate,
    "funding_premium": funding_premium,
    "oi_trend": oi_trend,
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
    
    # Test verisi (sentiment için özel veri)
    test_data = pd.DataFrame({
        'funding_rate': [0.0001, 0.0002, 0.0003, 0.0004, 0.0005, 0.0006, 0.0007, 0.0008, 0.0009, 0.0010],
        'futures_price': [10100, 10150, 10200, 10250, 10300, 10350, 10400, 10450, 10500, 10550],
        'spot_price': [10000, 10050, 10100, 10150, 10200, 10250, 10300, 10350, 10400, 10450],
        'open_interest': [1000, 1100, 1200, 1300, 1400, 1500, 1600, 1700, 1800, 1900]
    })
    
    print(f"🧪 {__name__} Self-Test")
    print("=" * 60)
    print(f"📊 Test Data Shape: {test_data.shape}")
    print(f"📋 Available Metrics: {get_metrics()}")
    print("=" * 60)
    
    # Her metriği test et
    for metric_name, metric_func in _METRICS.items():
        try:
            result = metric_func(test_data)
            
            # Kontroller
            is_float = isinstance(result, float)
            in_range = -1.0 <= result <= 1.0 if is_float else False
            not_nan = not np.isnan(result) if is_float else False
            not_inf = not np.isinf(result) if is_float else False
            
            passed = is_float and in_range and not_nan and not_inf
            
            if passed:
                print(f"✅ {metric_name}: {result:.6f}")
            else:
                print(f"❌ {metric_name}: {result} (float:{is_float}, range:{in_range}, !nan:{not_nan}, !inf:{not_inf})")
            
            results[metric_name] = passed
            
        except Exception as e:
            print(f"❌ {metric_name}: ERROR - {e}")
            results[metric_name] = False
    
    # Config test
    try:
        config = get_module_config()
        has_data_model = config.get("data_model") == DataModel.PANDAS
        has_required_columns = "required_columns" in config and len(config["required_columns"]) > 0
        has_normalization = "normalization" in config
        
        print(f"\n📋 Config Test:")
        print(f"  data_model={config.get('data_model')} (valid:{has_data_model})")
        print(f"  required_columns={len(config.get('required_columns', {}))} metrics (valid:{has_required_columns})")
        print(f"  normalization config present: {has_normalization}")
        
        results["config_valid"] = has_data_model and has_required_columns and has_normalization
    except Exception as e:
        print(f"❌ Config ERROR: {e}")
        results["config_valid"] = False
    
    print(f"\n📊 Results: {sum(results.values())}/{len(results)} passed")
    return results


if __name__ == "__main__":
    self_test()