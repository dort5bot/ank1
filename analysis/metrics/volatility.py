"""
analysis/metrics/volatility.py
Universal Metric Template with Data-Model-Aware Normalization
Date: 03/12/2025
Volatility Metrics Module
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


# ==================== MODULE CONFIG volatility.py====================
_MODULE_CONFIG = {
    # ZORUNLU: Veri modeli ve execution tipi
    "data_model": DataModel.PANDAS,
    "execution_type": "sync",
    "category": "volatility",
    
    # ZORUNLU: Her metrik için gerekli kolonlar
    "required_columns": {
        "historical_volatility": ["close"],
        "bollinger_width": ["close"],
        "garch_1_1": ["close"],
        "vol_of_vol": ["close"],
        "hurst_exponent": ["close"],
        "entropy_index": ["close"],
        "variance_ratio_test": ["close"],
        "range_expansion_index": ["high", "low", "close"],
    },
    
    # ÖNERİLEN: Normalizasyon konfigürasyonu
    "normalization": {
        # Global normalizasyon ayarları
        "global_range": {"min": -1.0, "max": 1.0},
        "default_method": NormalizationMethod.TANH,
        
        # Metrik-specific normalizasyon
        "metric_specific": {
            "historical_volatility": {
                "method": NormalizationMethod.TANH,
                "params": {"scale": 20.0}, # 0.5 çok küçüktü, 20.0 veya 50.0 yaparak küçük değişimleri belirginleştirin
                "description": "Tanh normalization for volatility"
            },
            "bollinger_width": {
                "method": NormalizationMethod.TANH,
                "params": {"scale": 2.0},
                "description": "Tanh normalization for relative width"
            },
            "garch_1_1": {
                "method": NormalizationMethod.TANH,
                "params": {"scale": 5.0},
                "description": "Tanh normalization for GARCH volatility"
            },
            "vol_of_vol": {
                    "method": NormalizationMethod.TANH,
                    "params": {"scale": 1.0},
                    "description": "Volatility of volatility (vol spike detector)"
                },            
            "hurst_exponent": {
                "method": NormalizationMethod.MINMAX,
                "params": {"data_min": 0.0, "data_max": 1.0},
                "description": "Hurst exponent normalized to -1..1 (0.5 = random)"
            },
            "entropy_index": {
                "method": NormalizationMethod.TANH,
                "params": {"scale": 0.5},
                "description": "Tanh normalization for entropy"
            },
            "variance_ratio_test": {
                "method": NormalizationMethod.TANH,
                "params": {"scale": 2.0},
                "description": "Tanh normalization for variance ratio"
            },
            "range_expansion_index": {
                "method": NormalizationMethod.CLAMP,
                "params": {},
                "description": "REI already in -1..1 range"
            }
        }
    },
    
    # OPSİYONEL: Metrik parametreleri
    "default_params": {
        "historical_volatility": {"window": 30, "annualize": True},
        "bollinger_width": {"window": 20, "num_std": 2.0},
        "garch_1_1": {"omega": 1e-6, "alpha": 0.05, "beta": 0.9, "min_periods": 10},
        "vol_of_vol": {"returns_window": 1,"vol_window": 20,"vov_window": 20,"annualize": False},
        
        "hurst_exponent": {"max_lag": 100, "min_periods": 50, "window": 200},
        "entropy_index": {"window": 100, "bins": 20},
        "variance_ratio_test": {"lag": 10, "window": 100},
        "range_expansion_index": {"window": 14}
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
            normalized = np.tanh(zscore * 0.5)
            # Target range'e map et
            normalized = (normalized + 1) / 2
            normalized = normalized * (target_max - target_min) + target_min
        
        elif method == NormalizationMethod.PERCENTILE:
            # Percentile-based (skalar için anlamlı değil, default)
            normalized = 0.0
        
        elif method == NormalizationMethod.TANH:
            # Tanh ile normalize (extreme değerleri yumuşat)
            scale = params.get('scale', 0.1)
            normalized = np.tanh(value * scale)
            # Target range'e map et
            normalized = (normalized + 1) / 2
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
            data_min = params.get('data_min', series.min())
            data_max = params.get('data_max', series.max())
            if data_max == data_min:
                normalized = pd.Series(0.0, index=series.index)
            else:
                normalized = ((series - data_min) / (data_max - data_min)) * (target_max - target_min) + target_min
        
        elif method == NormalizationMethod.ZSCORE:
            mean = params.get('mean', series.mean())
            std = params.get('std', series.std())
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
# Historical volatility using close prices.
# Returns: -1.0 to 1.0 normalized float
    
"""def historical_volatility(data: pd.DataFrame, **params) -> float:
    if "close" not in data.columns:
        raise ValueError("historical_volatility: data must contain 'close' column")

    close = data["close"]
    window = params.get("window", 30)
    annualize = params.get("annualize", True)

    # Log returns
    returns = np.log(close / close.shift(1))
    
    # Rolling standard deviation
    rolling_std = returns.rolling(window=window).std()
    
    # Annualization factor
    if annualize:
        rolling_std = rolling_std * np.sqrt(252)
    
    # return finalize_metric(rolling_std, "historical_volatility")
    # Veri çok küçükse (0.0001 gibi), bunu standardize etmek zordur. 
    # finalize_metric'e gitmeden önce bir çarpan ekleyebilirsiniz:
    return finalize_metric(rolling_std * 100, "historical_volatility")
"""

def historical_volatility(data: pd.DataFrame, **params) -> float:
    if "close" not in data.columns:
        raise ValueError("historical_volatility: data must contain 'close' column")

    close = data["close"]
    window = params.get("window", 30)
    annualize = params.get("annualize", True)

    # Log returns
    returns = np.log(close / close.shift(1))
    
    # Rolling standard deviation
    rolling_std = returns.rolling(window=window).std()
    
    # Annualization factor
    if annualize:
        rolling_std = rolling_std * np.sqrt(252)
    
    # DEBUG: Değerleri kontrol et
    last_value = rolling_std.iloc[-1] if not rolling_std.empty and not pd.isna(rolling_std.iloc[-1]) else 0
    logger.debug(f"DEBUG historical_volatility: window={window}, annualize={annualize}, "
                 f"last_value={last_value:.6f}, mean={rolling_std.mean():.6f}")
    
    return finalize_metric(rolling_std, "historical_volatility")


def bollinger_width(data: pd.DataFrame, **params) -> float:
    """
    Bollinger Band width using close prices.
    Returns: -1.0 to 1.0 normalized float
    """
    if "close" not in data.columns:
        raise ValueError("bollinger_width: data must contain 'close' column")

    close = data["close"]
    window = params.get("window", 20)
    num_std = params.get("num_std", 2.0)

    # Rolling mean and standard deviation
    rolling_mean = close.rolling(window=window).mean()
    rolling_std = close.rolling(window=window).std()

    # Bollinger width calculation
    with np.errstate(divide='ignore', invalid='ignore'):
        width = (2 * num_std * rolling_std) / rolling_mean.replace(0, np.nan)

    return finalize_metric(width, "bollinger_width")


"""def garch_1_1(data: pd.DataFrame, **params) -> float:

    # GARCH(1,1) conditional volatility model using close prices.
    # Returns: -1.0 to 1.0 normalized float

    if "close" not in data.columns:
        raise ValueError("garch_1_1: data must contain 'close' column")

    close = data["close"]
    omega = params.get("omega", 1e-6)
    alpha = params.get("alpha", 0.05)
    beta = params.get("beta", 0.9)
    min_periods = params.get("min_periods", 10)

    # Log returns
    returns = np.log(close / close.shift(1))
    returns = returns.fillna(0)

    n = len(returns)
    var = np.zeros(n)
    
    # Initialize with simple variance
    if n > min_periods:
        init_window = min(20, n)
        var[0] = np.var(returns[:init_window])
    else:
        var[0] = 1e-6

    # GARCH(1,1) recursion
    for t in range(1, n):
        # var[t] = omega + alpha * (returns[t-1] ** 2) + beta * var[t-1]
        var[t] = omega + alpha * (returns.iloc[t-1] ** 2) + beta * var[t-1]

        
        # Stability check
        if not np.isfinite(var[t]) or var[t] <= 0:
            var[t] = var[t-1]

    # Volatility (standard deviation)
    vol = np.sqrt(np.maximum(var, 1e-12))
    
    return finalize_metric(vol, "garch_1_1")
"""

# GARCH(1,1) conditional volatility model using close prices.
# Returns: -1.0 to 1.0 normalized float
    
def garch_1_1(data: pd.DataFrame, **params) -> float:
    """
    GARCH(1,1) conditional volatility model using close prices.
    Returns: -1.0 to 1.0 normalized float
    """
    try:
        print(f"DEBUG GARCH: Starting calculation, data shape: {data.shape}")
        
        if "close" not in data.columns:
            print(f"DEBUG GARCH: Columns available: {list(data.columns)}")
            raise ValueError("garch_1_1: data must contain 'close' column")

        close = data["close"]
        print(f"DEBUG GARCH: Close prices sample (first 5): {close.head().tolist()}")
        print(f"DEBUG GARCH: Close prices stats - min: {close.min():.2f}, max: {close.max():.2f}, mean: {close.mean():.2f}")
        
        omega = params.get("omega", 1e-6)
        alpha = params.get("alpha", 0.05)
        beta = params.get("beta", 0.9)
        min_periods = params.get("min_periods", 10)
        
        print(f"DEBUG GARCH: Parameters - omega: {omega}, alpha: {alpha}, beta: {beta}")

        # Log returns
        returns = np.log(close / close.shift(1))
        returns = returns.fillna(0)
        
        print(f"DEBUG GARCH: Returns stats - min: {returns.min():.6f}, max: {returns.max():.6f}, mean: {returns.mean():.6f}")
        print(f"DEBUG GARCH: Returns sample: {returns.head().tolist()}")

        n = len(returns)
        print(f"DEBUG GARCH: Number of returns: {n}")
        
        if n < min_periods:
            print(f"DEBUG GARCH: Not enough data: {n} < {min_periods}")
            return 0.0

        var = np.zeros(n)
        
        # Initialize with simple variance
        init_window = min(20, n)
        initial_var = np.var(returns[:init_window]) if init_window > 1 else 1e-6
        var[0] = initial_var
        
        print(f"DEBUG GARCH: Initial variance: {initial_var}")

        # GARCH(1,1) recursion
        for t in range(1, n):
            prev_return = returns.iloc[t-1]
            var[t] = omega + alpha * (prev_return ** 2) + beta * var[t-1]
            
            # Stability check
            if not np.isfinite(var[t]) or var[t] <= 0:
                var[t] = var[t-1]
                
            # Debug her 20 adımda bir
            if t % 20 == 0 or t == n-1:
                print(f"DEBUG GARCH: t={t}, return={prev_return:.6f}, var={var[t]:.6f}")

        # Volatility (standard deviation)
        vol = np.sqrt(np.maximum(var, 1e-12))
        
        # Son volatilite değeri
        final_vol = vol[-1]
        print(f"DEBUG GARCH: Final volatility: {final_vol:.6f}")
        
        # Basit normalize
        # Tarihsel volatiliteye göre normalize
        hist_vol = np.std(returns) * 100  # Yüzde olarak
        
        if hist_vol > 0:
            # GARCH'ın tarihsel vol'a oranı
            ratio = final_vol / hist_vol
            
            # -1 ile 1 arasında normalize et
            # ratio = 1 -> GARCH = tarihsel vol -> normalized = 0
            # ratio > 1 -> GARCH > tarihsel vol -> positive
            # ratio < 1 -> GARCH < tarihsel vol -> negative
            normalized = np.tanh(ratio - 1)  # -1 ile 1 arası
            
            print(f"DEBUG GARCH: hist_vol={hist_vol:.6f}, ratio={ratio:.6f}, normalized={normalized:.6f}")
            
            return float(normalized)
        else:
            print(f"DEBUG GARCH: hist_vol is zero or negative: {hist_vol}")
            return 0.0
        
    except Exception as e:
        print(f"DEBUG GARCH: ERROR - {type(e).__name__}: {str(e)}")
        import traceback
        traceback.print_exc()
        return 0.0



def vol_of_vol(data: pd.DataFrame, **params) -> float:
    """
    Volatility of Volatility (VoV)
    Measures instability / explosiveness of volatility.
    Returns: -1.0 to 1.0 normalized float
    """
    if "close" not in data.columns:
        raise ValueError("vol_of_vol: data must contain 'close' column")

    close = data["close"]

    returns_window = params.get("returns_window", 1)
    vol_window = params.get("vol_window", 20)
    vov_window = params.get("vov_window", 20)
    annualize = params.get("annualize", False)

    # Log returns
    returns = np.log(close / close.shift(returns_window)).fillna(0)

    # First-order volatility (rolling std of returns)
    rolling_vol = returns.rolling(window=vol_window).std()

    # Second-order volatility (vol of vol)
    vol_of_vol_series = rolling_vol.rolling(window=vov_window).std()

    if annualize:
        vol_of_vol_series *= np.sqrt(252)

    return finalize_metric(vol_of_vol_series, "vol_of_vol")



def hurst_exponent(data: pd.DataFrame, **params) -> float:
    """
    Hurst exponent for mean reversion or trend strength using close prices.
    Returns: -1.0 to 1.0 normalized float
    """
    if "close" not in data.columns:
        raise ValueError("hurst_exponent: data must contain 'close' column")

    close = data["close"]
    max_lag = params.get("max_lag", 100)
    min_periods = params.get("min_periods", 50)
    window = params.get("window", 200)

    n = len(close)
    hurst_values = np.full(n, np.nan)

    for i in range(window, n+1):
        if i < min_periods:
            continue
            
        segment = close.iloc[max(0, i-window):i].values
        segment_n = len(segment)
        
        if segment_n < min_periods:
            continue
            
        current_max_lag = min(max_lag, segment_n // 4)
        if current_max_lag < 5:
            continue
            
        lags = np.arange(2, current_max_lag + 1)
        tau = np.zeros(len(lags))
        
        # R/S calculation for each lag
        for lag_idx, lag in enumerate(lags):
            if lag >= segment_n:
                break
                
            segments = segment_n // lag
            if segments < 2:
                continue
                
            rs_values = np.zeros(segments)
            
            for j in range(segments):
                start = j * lag
                end = min(start + lag, segment_n)
                sub_segment = segment[start:end]
                
                if len(sub_segment) < 2:
                    continue
                    
                mean_val = np.mean(sub_segment)
                deviations = sub_segment - mean_val
                cumulative = np.cumsum(deviations)
                std_val = np.std(sub_segment, ddof=1)
                
                if std_val > 1e-12:
                    rs_values[j] = (np.max(cumulative) - np.min(cumulative)) / std_val
            
            valid_rs = rs_values[rs_values > 0]
            if len(valid_rs) > 0:
                tau[lag_idx] = np.mean(valid_rs)
            else:
                tau[lag_idx] = np.nan
        
        # Linear regression for Hurst exponent
        valid_mask = np.isfinite(tau) & (tau > 1e-12)
        if np.sum(valid_mask) < 3:
            hurst_values[i-1] = np.nan
            continue
            
        try:
            hurst = np.polyfit(np.log(lags[valid_mask]), np.log(tau[valid_mask]), 1)[0]
            hurst_values[i-1] = hurst if np.isfinite(hurst) else np.nan
        except:
            hurst_values[i-1] = np.nan

    return finalize_metric(hurst_values, "hurst_exponent")


def entropy_index(data: pd.DataFrame, **params) -> float:
    """
    Entropy-based volatility and randomness measure using close prices.
    Returns: -1.0 to 1.0 normalized float
    """
    if "close" not in data.columns:
        raise ValueError("entropy_index: data must contain 'close' column")

    close = data["close"]
    window = params.get("window", 100)
    bins = params.get("bins", 20)

    # Log returns
    returns = np.log(close / close.shift(1)).fillna(0)
    
    # Rolling entropy calculation
    entropy_values = np.full(len(close), np.nan)
    
    for i in range(window, len(returns)):
        segment = returns.iloc[i-window:i].values
        
        if np.std(segment) < 1e-12:
            entropy_values[i] = 0
            continue
            
        # Histogram and entropy
        hist, _ = np.histogram(segment, bins=bins, density=True)
        hist = hist / (np.sum(hist) + 1e-12)
        hist = np.clip(hist, 1e-12, 1.0)
        
        entropy = -np.sum(hist * np.log2(hist))
        entropy_values[i] = entropy if np.isfinite(entropy) else np.nan

    return finalize_metric(entropy_values, "entropy_index")


def variance_ratio_test(data: pd.DataFrame, **params) -> float:
    """
    Variance ratio test for random walk detection using close prices.
    Returns: -1.0 to 1.0 normalized float
    """
    if "close" not in data.columns:
        raise ValueError("variance_ratio_test: data must contain 'close' column")

    close = data["close"]
    lag = params.get("lag", 10)
    window = params.get("window", 100)

    # Price differences (simple returns)
    returns = close.diff().fillna(0)
    
    # Rolling variance ratio
    vr_values = np.full(len(close), np.nan)
    
    for i in range(window, len(returns)):
        segment = returns.iloc[i-window:i].values
        n = len(segment)
        
        if n < lag or lag <= 1:
            continue
            
        # Variance of 1-period returns
        var_1 = np.var(segment, ddof=1)
        if var_1 < 1e-12:
            continue
            
        # Lagged returns
        lag_returns = np.zeros(n // lag)
        for k in range(len(lag_returns)):
            start = k * lag
            end = min(start + lag, n)
            lag_returns[k] = np.sum(segment[start:end])
        
        if len(lag_returns) < 2:
            continue
            
        # Variance of lag-period returns
        var_k = np.var(lag_returns, ddof=1)
        vr = var_k / (lag * var_1)
        
        vr_values[i] = vr if np.isfinite(vr) else np.nan

    return finalize_metric(vr_values, "variance_ratio_test")


def range_expansion_index(data: pd.DataFrame, **params) -> float:
    """
    Range Expansion Index measuring price acceleration.
    Returns: -1.0 to 1.0 normalized float
    """
    required_cols = ["high", "low", "close"]
    if not all(col in data.columns for col in required_cols):
        raise ValueError(f"range_expansion_index: required columns {required_cols} not found")

    high = data["high"]
    low = data["low"]
    close = data["close"]
    window = params.get("window", 14)

    # Calculate REI
    mean_close = close.rolling(window=window).mean()
    price_range = high - low
    
    with np.errstate(divide='ignore', invalid='ignore'):
        rei = (close - mean_close) / price_range.replace(0, np.nan)

    return finalize_metric(rei, "range_expansion_index")


# ==================== REGISTRY ====================
_METRICS = {
    "historical_volatility": historical_volatility,
    "bollinger_width": bollinger_width,
    "garch_1_1": garch_1_1,
    "vol_of_vol": vol_of_vol,
    "hurst_exponent": hurst_exponent,
    "entropy_index": entropy_index,
    "variance_ratio_test": variance_ratio_test,
    "range_expansion_index": range_expansion_index,
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
    
    # Test verisi
    test_data = pd.DataFrame({
        'open': [100, 101, 102, 103, 104, 105, 106, 107, 108, 109],
        'high': [105, 106, 107, 108, 109, 110, 111, 112, 113, 114],
        'low': [95, 96, 97, 98, 99, 100, 101, 102, 103, 104],
        'close': [100, 102, 101, 104, 103, 106, 105, 108, 107, 110]
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


# ==================== EXPORT ====================
__all__ = [
    "historical_volatility",
    "bollinger_width",
    "garch_1_1",
    "hurst_exponent",
    "entropy_index",
    "variance_ratio_test",
    "range_expansion_index",
    "get_metrics",
    "get_function",
    "get_module_config",
    "self_test",
    "Normalizer",
    "finalize_metric",
    "DataModel",
    "NormalizationMethod"
]


if __name__ == "__main__":
    self_test()