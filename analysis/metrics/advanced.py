"""
analysis/metrics/advanced.py
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


# ==================== MODULE CONFIG  advanced.py====================
_MODULE_CONFIG = {
    # ZORUNLU: Veri modeli ve execution tipi
    "data_model": DataModel.PANDAS,
    "execution_type": "sync",
    "category": "technical",
    
    # ZORUNLU: Her metrik için gerekli kolonlar
    "required_columns": {
        "kalman_filter_trend": ["close"],
        "wavelet_transform": ["close"],
        "hilbert_transform_amplitude": ["close"],
        "hilbert_transform_slope": ["close"],
        "fractal_dimension_index_fdi": ["close"],
        "shannon_entropy": ["close"],
        "permutation_entropy": ["close"],
        "sample_entropy": ["close"],
        "granger_causality": ["close", "close_secondary"],
        "phase_shift_index": ["close", "close_secondary"]
    },
    
    # ÖNERİLEN: Normalizasyon konfigürasyonu
    "normalization": {
        # Global normalizasyon ayarları
        "global_range": {"min": -1.0, "max": 1.0},
        "default_method": NormalizationMethod.TANH,
        
        # Metrik-specific normalizasyon
        "metric_specific": {
            "kalman_filter_trend": {
                "method": NormalizationMethod.MINMAX,
                "params": {"lookback": 100},
                "description": "Kalman filter trend değerleri"
            },
            "wavelet_transform": {
                "method": NormalizationMethod.ZSCORE,
                "params": {"window": 50, "clip_sigma": 2.5},
                "description": "Wavelet transform z-score"
            },
            "hilbert_transform_amplitude": {
                "method": NormalizationMethod.MINMAX,
                "params": {"lookback": 100},
                "description": "Hilbert amplitude normalization"
            },
            "hilbert_transform_slope": {
                "method": NormalizationMethod.ZSCORE,
                "params": {"window": 50, "clip_sigma": 3.0},
                "description": "Hilbert slope z-score"
            },
            "fractal_dimension_index_fdi": {
                "method": NormalizationMethod.MINMAX,
                "params": {"data_min": 1.0, "data_max": 2.0},
                "description": "Fractal dimension 1-2 aralığı"
            },
            "shannon_entropy": {
                "method": NormalizationMethod.MINMAX,
                "params": {"data_min": 0.0, "data_max": 10.0},
                "description": "Shannon entropy 0-10 aralığı"
            },
            "permutation_entropy": {
                "method": NormalizationMethod.MINMAX,
                "params": {"data_min": 0.0, "data_max": 2.0},
                "description": "Permutation entropy 0-2 aralığı"
            },
            "sample_entropy": {
                "method": NormalizationMethod.MINMAX,
                "params": {"data_min": 0.0, "data_max": 2.0},
                "description": "Sample entropy 0-2 aralığı"
            },
            "granger_causality": {
                "method": NormalizationMethod.TANH,
                "params": {"scale": 0.5},
                "description": "Granger causality tanh normalization"
            },
            "phase_shift_index": {
                "method": NormalizationMethod.MINMAX,
                "params": {"data_min": -np.pi, "data_max": np.pi},
                "description": "Phase shift -π to π"
            }
        }
    },
    
    # OPSİYONEL: Metrik parametreleri
    "default_params": {
        "kalman_filter_trend": {
            "process_variance": 1e-5,
            "measurement_variance": 1e-2
        },
        "wavelet_transform": {
            "level": 1
        },
        "hilbert_transform_amplitude": {},
        "hilbert_transform_slope": {},
        "fractal_dimension_index_fdi": {
            "k_max": 100
        },
        "shannon_entropy": {
            "bins": 30
        },
        "permutation_entropy": {
            "order": 3,
            "delay": 1
        },
        "sample_entropy": {
            "m": 2,
            "r": None
        },
        "granger_causality": {
            "max_lag": 5,
            "significance_level": 0.05
        },
        "phase_shift_index": {
            "method": "hilbert"
        }
    }
}


# ==================== UTILITY FUNCTIONS ====================
_eps = 1e-12

def _to_numpy(x) -> np.ndarray:
    """Convert input to numpy array, handling various data types."""
    if isinstance(x, pd.Series):
        arr = x.values
    elif isinstance(x, pd.DataFrame):
        arr = x.iloc[:, 0].values if len(x.columns) > 0 else np.array([])
    else:
        arr = np.asarray(x, dtype=float)
    
    if arr.ndim == 0:
        arr = arr.reshape(1)
    elif arr.ndim > 1:
        arr = arr.ravel()
    return arr

def _mask_valid(arr: np.ndarray) -> np.ndarray:
    """Return boolean mask of finite values."""
    return np.isfinite(arr)

def _create_lag_matrix(series: np.ndarray, lag: int) -> np.ndarray:
    """Create lag matrix for time series."""
    n = series.size
    if n <= lag:
        return np.array([]).reshape(0, lag)
    
    matrix = np.full((n - lag, lag), np.nan)
    for i in range(lag):
        matrix[:, i] = series[i:n - lag + i]
    
    return matrix

def _analytic_signal_via_fft(x: np.ndarray) -> np.ndarray:
    """Compute analytic signal using FFT (no scipy)."""
    x = _to_numpy(x)
    n = x.size
    if n == 0:
        return np.array([], dtype=complex)
    X = np.fft.fft(x)
    h = np.zeros(n, dtype=float)
    if n % 2 == 0:
        # even
        h[0] = 1.0
        h[n//2] = 1.0
        h[1:n//2] = 2.0
    else:
        h[0] = 1.0
        h[1:(n+1)//2] = 2.0
    analytic = np.fft.ifft(X * h)
    return analytic

def _dwt_haar(signal: np.ndarray):
    """Single-level Haar DWT: returns (approx, detail). If odd length, last sample dropped."""
    n = signal.size
    if n < 2:
        return signal.copy(), np.array([], dtype=float)
    even = signal[0:(n // 2) * 2:2]
    odd = signal[1:(n // 2) * 2:2]
    approx = (even + odd) / np.sqrt(2.0)
    detail = (even - odd) / np.sqrt(2.0)
    return approx, detail

def _idwt_haar(approx: np.ndarray, detail: np.ndarray) -> np.ndarray:
    """Inverse single-level Haar DWT."""
    n = approx.size + detail.size
    if detail.size == 0:
        return approx.copy()
    out = np.empty(approx.size * 2, dtype=float)
    out[0::2] = (approx + detail) / np.sqrt(2.0)
    out[1::2] = (approx - detail) / np.sqrt(2.0)
    return out

def _extract_series(data: Union[pd.DataFrame, pd.Series, np.ndarray], column: str = None) -> np.ndarray:
    """Extract series from data based on column name or position."""
    if isinstance(data, pd.DataFrame):
        if column and column in data.columns:
            return data[column].values
        elif len(data.columns) > 0:
            return data.iloc[:, 0].values
        else:
            return np.array([])
    elif isinstance(data, pd.Series):
        return data.values
    else:
        return _to_numpy(data)


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
            # Min-max normalization
            data_min = params.get('data_min', value)
            data_max = params.get('data_max', value)
            if data_max == data_min:
                normalized = 0.0
            else:
                # Map from [data_min, data_max] to [target_min, target_max]
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
            scale = params.get('tanh_scale', 0.5)
            normalized = np.tanh(zscore * scale)
            
            # -1..1 aralığında zaten, target range'e map et
            if target_min != -1.0 or target_max != 1.0:
                normalized = (normalized + 1) / 2  # -1..1 → 0..1
                normalized = normalized * (target_max - target_min) + target_min
        
        elif method == NormalizationMethod.PERCENTILE:
            # Percentile-based (skalar için anlamlı değil, default)
            normalized = 0.0
        
        elif method == NormalizationMethod.TANH:
            # Tanh ile normalize
            scale = params.get('scale', 0.1)
            normalized = np.tanh(value * scale)
            
            # -1..1 aralığında, target range'e map et
            if target_min != -1.0 or target_max != 1.0:
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
            if target_min != -1.0 or target_max != 1.0:
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
            # Lookback parametresi varsa rolling min/max kullan
            lookback = params.get('lookback', None)
            if lookback and len(series) > lookback:
                data_min = series.rolling(window=lookback, min_periods=1).min()
                data_max = series.rolling(window=lookback, min_periods=1).max()
                normalized = ((series - data_min) / (data_max - data_min + _eps)) * (target_max - target_min) + target_min
            else:
                data_min = series.min()
                data_max = series.max()
                if data_max == data_min:
                    normalized = pd.Series(0.0, index=series.index)
                else:
                    normalized = ((series - data_min) / (data_max - data_min)) * (target_max - target_min) + target_min
        
        elif method == NormalizationMethod.ZSCORE:
            # Rolling z-score
            window = params.get('window', min(50, len(series)))
            clip_sigma = params.get('clip_sigma', 3.0)
            
            rolling_mean = series.rolling(window=window, min_periods=1).mean()
            rolling_std = series.rolling(window=window, min_periods=1).std().replace(0, 1.0)
            
            zscore = (series - rolling_mean) / rolling_std
            zscore_clipped = zscore.clip(lower=-clip_sigma, upper=clip_sigma)
            
            # Tanh ile normalize
            normalized = np.tanh(zscore_clipped * 0.5)
            
            if target_min != -1.0 or target_max != 1.0:
                normalized = (normalized + 1) / 2
                normalized = normalized * (target_max - target_min) + target_min
        
        elif method == NormalizationMethod.TANH:
            scale = params.get('scale', 0.1)
            normalized = np.tanh(series * scale)
            
            if target_min != -1.0 or target_max != 1.0:
                normalized = (normalized + 1) / 2
                normalized = normalized * (target_max - target_min) + target_min
        
        elif method == NormalizationMethod.CLAMP:
            normalized = series.clip(lower=target_min, upper=target_max)
        
        elif method == NormalizationMethod.RAW:
            normalized = series
        
        else:
            normalized = series
        
        return normalized.fillna(0.0)
    
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
    
    @staticmethod
    def _normalize_pandas_df(
        df: pd.DataFrame,
        method: str,
        target_range: Tuple[float, float],
        **params
    ) -> pd.DataFrame:
        """Pandas DataFrame'i normalize eder"""
        if df.empty:
            return df
        
        normalized_cols = {}
        for col in df.columns:
            if pd.api.types.is_numeric_dtype(df[col]):
                normalized_cols[col] = Normalizer._normalize_pandas_series(df[col], method, target_range, **params)
            else:
                normalized_cols[col] = df[col]
        
        return pd.DataFrame(normalized_cols, index=df.index)


# ==================== METRIC FINALIZATION ====================
def finalize_metric(
    raw_result: Any,
    metric_name: str,
    config: Optional[Dict[str, Any]] = None
) -> float:
    """
    Ham metrik sonucunu final float'a çevirir.
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
        # Öncelikle 'value' anahtarı varsa onu kullan
        if 'value' in result and isinstance(result['value'], (int, float, np.number)):
            return float(result['value'])
        # Sonra numeric olan ilk değer
        for key, value in result.items():
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
def kalman_filter_trend(data: pd.DataFrame, **params) -> float:
    """
    1D Kalman filter (simple, robust).
    Returns: -1.0 to 1.0 normalized float
    """
    if "close" not in data.columns:
        raise ValueError("kalman_filter_trend: data must contain 'close' column")
    
    series = data["close"].values
    process_variance = params.get("process_variance", 1e-5)
    measurement_variance = params.get("measurement_variance", 1e-2)
    
    x = series
    n = x.size
    if n == 0:
        return 0.0

    # initialize
    xhat = np.full(n, np.nan, dtype=float)
    P = np.zeros(n, dtype=float)

    # handle initial valid value
    mask = _mask_valid(x)
    if not np.any(mask):
        return 0.0
    first_idx = np.argmax(mask)
    x0 = x[first_idx]

    xhat[first_idx] = x0
    P[first_idx] = 1.0

    # forward pass
    for t in range(first_idx + 1, n):
        if not mask[t]:
            xhat[t] = xhat[t - 1]
            P[t] = P[t - 1] + process_variance
            continue

        # predict
        Pminus = P[t - 1] + process_variance
        # update
        K = Pminus / (Pminus + measurement_variance + _eps)
        xhat[t] = xhat[t - 1] + K * (x[t] - xhat[t - 1])
        P[t] = (1.0 - K) * Pminus

    # backward fill for leading NaNs (if any before first_idx)
    if first_idx > 0:
        xhat[:first_idx] = xhat[first_idx]

    # Extract the smoothed series
    smoothed_series = pd.Series(xhat, index=data.index)
    
    # Calculate returns from smoothed series
    smoothed_returns = smoothed_series.pct_change().fillna(0.0)
    
    # Return normalized value
    return finalize_metric(smoothed_returns, "kalman_filter_trend")

def wavelet_transform(data: pd.DataFrame, **params) -> float:
    """
    Simple Haar wavelet denoise-like transform using multilevel DWT + full reconstruction.
    Returns: -1.0 to 1.0 normalized float
    """
    if "close" not in data.columns:
        raise ValueError("wavelet_transform: data must contain 'close' column")
    
    x = data["close"].values
    level = params.get("level", 1)
    
    n0 = x.size
    if n0 == 0:
        return 0.0

    # copy and pad to allow dyadic decomposition
    data_arr = x.copy()
    # if odd, drop last sample for stable haar decomposition; we'll pad back later
    drop_last = False
    if data_arr.size % 2 == 1:
        drop_last = True
        data_arr = data_arr[:-1]

    approx = data_arr
    details = []
    lvl = max(1, int(level))
    for _ in range(lvl):
        a, d = _dwt_haar(approx)
        details.append(d)
        approx = a
        if approx.size < 2:
            break

    # naive thresholding: zero out smallest-detail coefficients (light denoising)
    if details:
        all_details = np.concatenate([d for d in details if d.size > 0]) if any(d.size>0 for d in details) else np.array([])
        if all_details.size > 0:
            mad = np.median(np.abs(all_details - np.median(all_details)))
            thr = 3.0 * (mad + _eps)
            details = [np.where(np.abs(d) < thr, 0.0, d) for d in details]

    # reconstruct backward
    recon = approx
    for d in reversed(details):
        recon = _idwt_haar(recon, d)

    # if we dropped last sample, append it back (simple copy)
    if drop_last:
        recon = np.concatenate([recon, x[-1:]])

    # ensure same length
    if recon.size > n0:
        recon = recon[:n0]
    elif recon.size < n0:
        recon = np.concatenate([recon, np.full(n0 - recon.size, recon[-1] if recon.size>0 else 0.0)])

    # Calculate reconstruction error (original vs reconstructed)
    if len(x) > 0 and len(recon) > 0:
        min_len = min(len(x), len(recon))
        reconstruction_error = x[:min_len] - recon[:min_len]
        metric_value = np.std(reconstruction_error) if len(reconstruction_error) > 1 else 0.0
    else:
        metric_value = 0.0
    
    # Return normalized value
    return finalize_metric(metric_value, "wavelet_transform")

def hilbert_transform_amplitude(data: pd.DataFrame, **params) -> float:
    """
    Instantaneous amplitude (envelope) from analytic signal.
    Returns: -1.0 to 1.0 normalized float
    """
    if "close" not in data.columns:
        raise ValueError("hilbert_transform_amplitude: data must contain 'close' column")
    
    x = data["close"].values
    if x.size == 0:
        return 0.0
    
    analytic = _analytic_signal_via_fft(x)
    amplitude = np.abs(analytic)
    
    # Calculate amplitude change rate
    if len(amplitude) > 1:
        amplitude_change = np.diff(amplitude) / (amplitude[:-1] + _eps)
        metric_value = np.mean(amplitude_change) if len(amplitude_change) > 0 else 0.0
    else:
        metric_value = 0.0
    
    return finalize_metric(metric_value, "hilbert_transform_amplitude")

def hilbert_transform_slope(data: pd.DataFrame, **params) -> float:
    """
    Instantaneous phase slope (derivative of unwrapped angle).
    Returns: -1.0 to 1.0 normalized float
    """
    if "close" not in data.columns:
        raise ValueError("hilbert_transform_slope: data must contain 'close' column")
    
    x = data["close"].values
    if x.size == 0:
        return 0.0
    
    analytic = _analytic_signal_via_fft(x)
    phase = np.unwrap(np.angle(analytic))
    # slope: gradient of phase; preserve length by using np.gradient
    slope = np.gradient(phase)
    
    # Return mean slope
    metric_value = np.mean(slope) if len(slope) > 0 else 0.0
    return finalize_metric(metric_value, "hilbert_transform_slope")

def fractal_dimension_index_fdi(data: pd.DataFrame, **params) -> float:
    """
    Higuchi-like fractal dimension estimate.
    Returns: -1.0 to 1.0 normalized float
    """
    if "close" not in data.columns:
        raise ValueError("fractal_dimension_index_fdi: data must contain 'close' column")
    
    x = data["close"].values
    k_max = params.get("k_max", 100)
    
    N = x.size
    if N < 10:
        return 0.0

    k_max = min(int(k_max), N // 2)
    Lk = np.zeros(k_max, dtype=float)

    for k in range(1, k_max + 1):
        Lm_sum = 0.0
        m_count = 0
        for m in range(k):
            idx = np.arange(m, N, k)
            if idx.size < 2:
                continue
            diffs = np.abs(np.diff(x[idx]))
            if diffs.size == 0:
                continue
            norm = (N - 1) / ( (idx.size) * k )
            Lm = (np.sum(diffs) * norm) / k
            Lm_sum += Lm
            m_count += 1
        if m_count > 0:
            Lk[k - 1] = Lm_sum / m_count
        else:
            Lk[k - 1] = np.nan

    valid = np.isfinite(Lk) & (Lk > 0)
    ks = np.arange(1, k_max + 1)[valid]
    Lk_valid = Lk[valid]
    if ks.size < 2:
        return 0.0

    coeffs = np.polyfit(np.log(ks), np.log(Lk_valid), 1)
    # fractal dimension ~ -slope
    fractal_dim = -coeffs[0]
    
    # Map fractal dimension (typically 1-2) to -1..1 range via config
    return finalize_metric(fractal_dim, "fractal_dimension_index_fdi")

def shannon_entropy(data: pd.DataFrame, **params) -> float:
    """
    Shannon entropy of distribution (bits).
    Returns: -1.0 to 1.0 normalized float
    """
    if "close" not in data.columns:
        raise ValueError("shannon_entropy: data must contain 'close' column")
    
    x = data["close"].values
    bins = params.get("bins", 30)
    
    if x.size == 0:
        return 0.0
    
    # Remove NaN values
    x_clean = x[~np.isnan(x)]
    if x_clean.size == 0:
        return 0.0
    
    hist, _ = np.histogram(x_clean, bins=bins, density=True)
    hist = hist[hist > 0]
    if hist.size == 0:
        entropy = 0.0
    else:
        probs = hist / (np.sum(hist) + _eps)
        entropy = -np.sum(probs * np.log2(probs + _eps))
    
    return finalize_metric(entropy, "shannon_entropy")

def permutation_entropy(data: pd.DataFrame, **params) -> float:
    """
    Permutation entropy (Bandt & Pompe).
    Returns: -1.0 to 1.0 normalized float
    """
    if "close" not in data.columns:
        raise ValueError("permutation_entropy: data must contain 'close' column")
    
    x = data["close"].values
    order = params.get("order", 3)
    delay = params.get("delay", 1)
    
    n = x.size
    if n < order * delay + 1:
        return 0.0

    patterns = {}
    m = order
    for i in range(n - (m - 1) * delay):
        window = x[i : i + m * delay : delay]
        ranks = tuple(np.argsort(window).tolist())
        patterns[ranks] = patterns.get(ranks, 0) + 1

    counts = np.array(list(patterns.values()), dtype=float)
    probs = counts / (np.sum(counts) + _eps)
    entropy = -np.sum(probs * np.log2(probs + _eps))
    
    # Normalize by log2(factorial(order)) for 0-1 range
    # np.math yerine math kullanıyoruz
    import math
    max_entropy = np.log2(math.factorial(order))
    if max_entropy > 0:
        normalized_entropy = entropy / max_entropy
    else:
        normalized_entropy = 0.0
    
    return finalize_metric(normalized_entropy, "permutation_entropy")
    

def sample_entropy(data: pd.DataFrame, **params) -> float:
    """
    Sample entropy approximation (unbiased-ish).
    Returns: -1.0 to 1.0 normalized float
    """
    if "close" not in data.columns:
        raise ValueError("sample_entropy: data must contain 'close' column")
    
    x = data["close"].values
    m = params.get("m", 2)
    r = params.get("r", None)
    
    N = x.size
    if N < m + 2:
        return 0.0
    
    if r is None:
        r = 0.2 * np.nanstd(x) + _eps
    
    # Build m-length templates
    def _count_similar(m_len: int) -> float:
        count = 0
        templates = N - m_len + 1
        for i in range(templates):
            xi = x[i:i + m_len]
            # compare to subsequent templates
            for j in range(i + 1, templates):
                xj = x[j:j + m_len]
                if np.max(np.abs(xi - xj)) <= r:
                    count += 1
        return float(count)

    try:
        B = _count_similar(m)
        A = _count_similar(m + 1)
        if B == 0:
            sample_ent = 0.0
        else:
            sample_ent = -np.log((A + _eps) / (B + _eps))
    except Exception:
        sample_ent = 0.0
    
    return finalize_metric(sample_ent, "sample_entropy")

def granger_causality(data: pd.DataFrame, **params) -> float:
    """
    Granger causality test between two time series.
    Tests if series_y Granger-causes series_x.
    
    Returns: -1.0 to 1.0 normalized float
    """
    # Extract primary and secondary series
    if "close_secondary" in data.columns:
        series_x = data["close"].values
        series_y = data["close_secondary"].values
    elif 'secondary_series' in params:
        series_x = data["close"].values
        series_y = _extract_series(params['secondary_series'], params.get('secondary_column', 'close'))
    else:
        return 0.0
    
    max_lag = params.get("max_lag", 5)
    significance_level = params.get("significance_level", 0.05)
    
    x = series_x
    y = series_y
    
    # Ensure same length and remove NaNs
    mask = _mask_valid(x) & _mask_valid(y)
    x_clean = x[mask]
    y_clean = y[mask]
    
    n = x_clean.size
    if n < max_lag + 10:  # Minimum sample size requirement
        return 0.0
    
    # Find optimal lag using AIC
    best_lag = 1
    best_aic = np.inf
    
    for lag in range(1, max_lag + 1):
        if n <= lag * 3:
            continue
            
        # Restricted model (only x's own lags)
        X_r = _create_lag_matrix(x_clean, lag)
        if X_r.shape[0] == 0:
            continue
            
        # Unrestricted model (x's lags + y's lags)
        X_ur = np.column_stack([
            _create_lag_matrix(x_clean, lag),
            _create_lag_matrix(y_clean, lag)
        ])
        
        # Remove rows with NaN
        valid_mask = ~(np.any(np.isnan(X_ur), axis=1) | np.isnan(x_clean[lag:]))
        if np.sum(valid_mask) < lag + 5:
            continue
            
        X_r_valid = X_r[valid_mask]
        X_ur_valid = X_ur[valid_mask]
        y_target = x_clean[lag:][valid_mask]
        
        try:
            # Fit models
            beta_r = np.linalg.lstsq(X_r_valid, y_target, rcond=None)[0]
            beta_ur = np.linalg.lstsq(X_ur_valid, y_target, rcond=None)[0]
            
            # Calculate residuals
            resid_r = y_target - X_r_valid @ beta_r
            resid_ur = y_target - X_ur_valid @ beta_ur
            
            # Calculate AIC
            k_r = lag + 1
            k_ur = 2 * lag + 1
            T = len(y_target)
            
            aic_r = T * np.log(np.var(resid_r)) + 2 * k_r
            aic_ur = T * np.log(np.var(resid_ur)) + 2 * k_ur
            
            if aic_ur < best_aic:
                best_aic = aic_ur
                best_lag = lag
                
        except np.linalg.LinAlgError:
            continue
    
    # Perform Granger test with best lag
    if best_lag == 0:
        return 0.0
    
    # Final test with best lag
    X_r = _create_lag_matrix(x_clean, best_lag)
    X_ur = np.column_stack([
        _create_lag_matrix(x_clean, best_lag),
        _create_lag_matrix(y_clean, best_lag)
    ])
    
    valid_mask = ~(np.any(np.isnan(X_ur), axis=1) | np.isnan(x_clean[best_lag:]))
    X_r_valid = X_r[valid_mask]
    X_ur_valid = X_ur[valid_mask]
    y_target = x_clean[best_lag:][valid_mask]
    
    try:
        # Fit final models
        beta_r = np.linalg.lstsq(X_r_valid, y_target, rcond=None)[0]
        beta_ur = np.linalg.lstsq(X_ur_valid, y_target, rcond=None)[0]
        
        # Calculate residuals
        resid_r = y_target - X_r_valid @ beta_r
        resid_ur = y_target - X_ur_valid @ beta_ur
        
        # Calculate F-statistic
        RSS_r = np.sum(resid_r ** 2)
        RSS_ur = np.sum(resid_ur ** 2)
        T = len(y_target)
        
        if RSS_ur < _eps or T <= 2 * best_lag + 1:
            f_statistic = 0.0
            p_value = 1.0
        else:
            f_statistic = ((RSS_r - RSS_ur) / best_lag) / (RSS_ur / (T - 2 * best_lag - 1))
            
            # Calculate p-value using F-distribution
            from scipy.stats import f
            p_value = 1 - f.cdf(f_statistic, best_lag, T - 2 * best_lag - 1)
        
        # Convert to metric value: 1 if significant (p < 0.05), -1 if not
        if p_value < significance_level:
            metric_value = 1.0  # Significant Granger causality
        else:
            metric_value = -1.0  # Not significant
        
        return finalize_metric(metric_value, "granger_causality")
        
    except (np.linalg.LinAlgError, ValueError):
        return 0.0

def phase_shift_index(data: pd.DataFrame, **params) -> float:
    """
    Calculate phase shift between two signals.
    
    Returns: -1.0 to 1.0 normalized float
    """
    # Extract primary and secondary series
    if "close_secondary" in data.columns:
        series1 = data["close"].values
        series2 = data["close_secondary"].values
    elif 'secondary_series' in params:
        series1 = data["close"].values
        series2 = _extract_series(params['secondary_series'], params.get('secondary_column', 'close'))
    else:
        return 0.0
    
    method = params.get("method", "hilbert")
    
    x1 = series1
    x2 = series2
    
    # Ensure same length and remove NaNs
    min_len = min(len(x1), len(x2))
    x1 = x1[:min_len]
    x2 = x2[:min_len]
    
    mask = _mask_valid(x1) & _mask_valid(x2)
    x1_clean = x1[mask]
    x2_clean = x2[mask]
    
    n = x1_clean.size
    if n < 10:
        return 0.0
    
    if method.lower() == "hilbert":
        # Using analytic signal method
        analytic1 = _analytic_signal_via_fft(x1_clean)
        analytic2 = _analytic_signal_via_fft(x2_clean)
        
        phase1 = np.angle(analytic1)
        phase2 = np.angle(analytic2)
        
    elif method.lower() == "fft":
        # Using FFT phase method
        fft1 = np.fft.fft(x1_clean)
        fft2 = np.fft.fft(x2_clean)
        
        # Get phase at dominant frequency
        dominant_idx = np.argmax(np.abs(fft1))
        phase1 = np.angle(fft1[dominant_idx])
        phase2 = np.angle(fft2[dominant_idx])
        
    else:
        return 0.0
    
    # Calculate phase difference
    phase_diff = phase1 - phase2
    
    # Unwrap phase differences to avoid 2π jumps
    phase_diff_unwrapped = np.unwrap(phase_diff)
    
    # Calculate mean phase shift
    mean_phase_shift = np.mean(phase_diff_unwrapped) if len(phase_diff_unwrapped) > 0 else 0.0
    
    # Normalize phase shift to -1..1 range via config
    return finalize_metric(mean_phase_shift, "phase_shift_index")


# ==================== REGISTRY ====================
_METRICS = {
    "kalman_filter_trend": kalman_filter_trend,
    "wavelet_transform": wavelet_transform,
    "hilbert_transform_amplitude": hilbert_transform_amplitude,
    "hilbert_transform_slope": hilbert_transform_slope,
    "fractal_dimension_index_fdi": fractal_dimension_index_fdi,
    "shannon_entropy": shannon_entropy,
    "permutation_entropy": permutation_entropy,
    "sample_entropy": sample_entropy,
    "granger_causality": granger_causality,
    "phase_shift_index": phase_shift_index
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
        'close': [100, 102, 101, 104, 103, 106, 105, 108, 107, 110],
        'close_secondary': [99, 103, 100, 105, 102, 107, 104, 109, 106, 111]
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