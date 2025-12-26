"""
metrics/onchain.py
Universal Onchain Metrics with Data-Model-Aware Normalization
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


# ==================== MODULE CONFIG onchain.py====================
_MODULE_CONFIG = {
    # ZORUNLU: Veri modeli ve execution tipi
    "data_model": DataModel.PANDAS,
    "execution_type": "sync",
    "category": "onchain",
    
    # ZORUNLU: Her metrik için gerekli kolonlar
    "required_columns": {
        "etf_net_flow": ["etf_inflow", "etf_outflow"],
        "exchange_netflow": ["exchange_deposits", "exchange_withdrawals"],
        "stablecoin_flow": ["stablecoin_in", "stablecoin_out"],
        "net_realized_pl": ["realized_profit", "realized_loss"],
        "realized_cap": ["price", "realized_price"],
        "nupl": ["market_cap", "realized_cap"],
        "exchange_whale_ratio": ["whale_deposits", "total_deposits"],
        "mvrv_zscore": ["market_cap", "realized_cap", "market_cap_std"],
        "sopr": ["realized_value", "spent_value"],
        "etf_flow_composite": [
            "etf_inflow", "etf_outflow", 
            "stablecoin_in", "stablecoin_out",
            "exchange_deposits", "exchange_withdrawals"
        ]
    },
    
    # ÖNERİLEN: Normalizasyon konfigürasyonu
    "normalization": {
        # Global normalizasyon ayarları
        "global_range": {"min": -1.0, "max": 1.0},
        "default_method": NormalizationMethod.TANH,
        
        # Metrik-specific normalizasyon
        "metric_specific": {
            "etf_net_flow": {
                "method": NormalizationMethod.MINMAX,
                "params": {"data_min": -1e9, "data_max": 1e9},
                "description": "Min-max normalized ETF flow"
            },
            "exchange_netflow": {
                "method": NormalizationMethod.MINMAX,
                "params": {"data_min": -1e9, "data_max": 1e9},
                "description": "Min-max normalized exchange netflow"
            },
            "stablecoin_flow": {
                "method": NormalizationMethod.ZSCORE,
                "params": {"clip_sigma": 3.0},
                "description": "Z-score normalized stablecoin flow"
            },
            "net_realized_pl": {
                "method": NormalizationMethod.MINMAX,
                "params": {"data_min": -1e9, "data_max": 1e9},
                "description": "Min-max normalized realized P/L"
            },
            "nupl": {
                "method": NormalizationMethod.MINMAX,
                "params": {"data_min": -1.0, "data_max": 1.0},
                "description": "NUPL already in -1..1 range"
            },
            "exchange_whale_ratio": {
                "method": NormalizationMethod.MINMAX,
                "params": {"data_min": 0.0, "data_max": 1.0},
                "description": "Whale ratio in 0..1 range"
            },
            "mvrv_zscore": {
                "method": NormalizationMethod.ZSCORE,
                "params": {"clip_sigma": 3.0},
                "description": "Z-score normalized MVRV"
            },
            "sopr": {
                "method": NormalizationMethod.MINMAX,
                "params": {"data_min": 0.0, "data_max": 2.0},
                "description": "SOPR in 0..2 range"
            },
            "etf_flow_composite": {
                "method": NormalizationMethod.MINMAX,
                "params": {"data_min": -1e9, "data_max": 1e9},
                "description": "Composite ETF flow"
            }
        }
    },
    
    # OPSİYONEL: Metrik parametreleri
    "default_params": {
        "realized_cap": {"window": 1},
        "nupl": {"window": 1},
        "mvrv_zscore": {"window": 1},
        "sopr": {"window": 1}
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
            data_min = params.get('data_min', value)
            data_max = params.get('data_max', value)
            if data_max == data_min:
                normalized = 0.0
            else:
                normalized = ((value - data_min) / (data_max - data_min)) * (target_max - target_min) + target_min
        
        elif method == NormalizationMethod.ZSCORE:
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
            normalized = np.tanh(zscore * 0.5)
            # Target range'e map et
            normalized = (normalized + 1) / 2
            normalized = normalized * (target_max - target_min) + target_min
        
        elif method == NormalizationMethod.PERCENTILE:
            normalized = 0.0
        
        elif method == NormalizationMethod.TANH:
            scale = params.get('scale', 0.1)
            normalized = np.tanh(value * scale)
            normalized = (normalized + 1) / 2
            normalized = normalized * (target_max - target_min) + target_min
        
        elif method == NormalizationMethod.CLAMP:
            normalized = np.clip(value, target_min, target_max)
        
        elif method == NormalizationMethod.RAW:
            normalized = value
        
        else:
            warnings.warn(f"Unknown normalization method: {method}, using tanh")
            normalized = np.tanh(value * 0.1)
            normalized = (normalized + 1) / 2
            normalized = normalized * (target_max - target_min) + target_min
        
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
            mean = series.mean()
            std = series.std()
            if std == 0:
                zscore = pd.Series(0.0, index=series.index)
            else:
                zscore = (series - mean) / std
            # Sigma clipping
            clip_sigma = params.get('clip_sigma', 3.0)
            zscore = zscore.clip(lower=-clip_sigma, upper=clip_sigma)
            normalized = np.tanh(zscore * 0.5)
            normalized = (normalized + 1) / 2
            normalized = normalized * (target_max - target_min) + target_min
        
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
        """Pandas DataFrame'i normalize eder"""
        if df.empty:
            return pd.DataFrame()
        
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
        
        if arr.ndim == 1:
            series = pd.Series(arr)
            normalized_series = Normalizer._normalize_pandas_series(series, method, target_range, **params)
            return normalized_series.values
        
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
        """Dict'i normalize eder"""
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
    """
    if config is None:
        config = _MODULE_CONFIG
    
    scalar_value = extract_scalar_from_result(raw_result)
    
    norm_config = config.get("normalization", {})
    global_range = norm_config.get("global_range", {"min": -1.0, "max": 1.0})
    target_range = (global_range["min"], global_range["max"])
    
    metric_specific = norm_config.get("metric_specific", {}).get(metric_name, {})
    if metric_specific:
        method = metric_specific.get("method", norm_config.get("default_method", NormalizationMethod.TANH))
        params = metric_specific.get("params", {})
    else:
        method = norm_config.get("default_method", NormalizationMethod.TANH)
        params = {}
    
    normalized = Normalizer.normalize(
        scalar_value,
        method=method,
        target_range=target_range,
        **params
    )
    
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
    
    if isinstance(result, (int, float, np.number)):
        return float(result)
    
    elif isinstance(result, pd.Series):
        if result.empty:
            return 0.0
        non_nan = result[result.notna()]
        if non_nan.empty:
            return 0.0
        return float(non_nan.iloc[-1])
    
    elif isinstance(result, pd.DataFrame):
        if result.empty:
            return 0.0
        first_col = result.columns[0]
        non_nan = result[first_col].dropna()
        if non_nan.empty:
            return 0.0
        return float(non_nan.iloc[-1])
    
    elif isinstance(result, np.ndarray):
        if result.size == 0:
            return 0.0
        flattened = result.flatten()
        non_nan = flattened[~np.isnan(flattened)]
        if non_nan.size == 0:
            return 0.0
        return float(non_nan[-1])
    
    elif isinstance(result, list):
        if not result:
            return 0.0
        last_item = result[-1]
        if isinstance(last_item, (int, float, np.number)):
            return float(last_item)
        else:
            return 0.0
    
    elif isinstance(result, dict):
        for value in result.values():
            if isinstance(value, (int, float, np.number)):
                return float(value)
        return 0.0
    
    else:
        try:
            return float(result)
        except (TypeError, ValueError):
            return 0.0


# ==================== PURE METRIC FUNCTIONS ====================
def etf_net_flow(data: pd.DataFrame, **params) -> float:
    """
    ETF Net Flow = Inflow - Outflow
    Returns: -1.0 to 1.0 normalized float
    """
    required_cols = ["etf_inflow", "etf_outflow"]
    missing = [c for c in required_cols if c not in data.columns]
    if missing:
        raise ValueError(f"etf_net_flow: missing columns {missing}")
    
    inflow = data["etf_inflow"]
    outflow = data["etf_outflow"]
    result = inflow - outflow
    
    return finalize_metric(result, "etf_net_flow")


def exchange_netflow(data: pd.DataFrame, **params) -> float:
    """
    Exchange Netflow = Deposits - Withdrawals
    Returns: -1.0 to 1.0 normalized float
    """
    required_cols = ["exchange_deposits", "exchange_withdrawals"]
    missing = [c for c in required_cols if c not in data.columns]
    if missing:
        raise ValueError(f"exchange_netflow: missing columns {missing}")
    
    deposits = data["exchange_deposits"]
    withdrawals = data["exchange_withdrawals"]
    result = deposits - withdrawals
    
    return finalize_metric(result, "exchange_netflow")


def stablecoin_flow(data: pd.DataFrame, **params) -> float:
    """
    Stablecoin Flow = Stable In - Stable Out
    Returns: -1.0 to 1.0 normalized float
    """
    required_cols = ["stablecoin_in", "stablecoin_out"]
    missing = [c for c in required_cols if c not in data.columns]
    if missing:
        raise ValueError(f"stablecoin_flow: missing columns {missing}")
    
    stable_in = data["stablecoin_in"]
    stable_out = data["stablecoin_out"]
    result = stable_in - stable_out
    
    return finalize_metric(result, "stablecoin_flow")


def net_realized_pl(data: pd.DataFrame, **params) -> float:
    """
    Net Realized Profit/Loss = Profit - Loss
    Returns: -1.0 to 1.0 normalized float
    """
    required_cols = ["realized_profit", "realized_loss"]
    missing = [c for c in required_cols if c not in data.columns]
    if missing:
        raise ValueError(f"net_realized_pl: missing columns {missing}")
    
    profit = data["realized_profit"]
    loss = data["realized_loss"]
    result = profit - loss
    
    return finalize_metric(result, "net_realized_pl")


def realized_cap(data: pd.DataFrame, **params) -> float:
    """
    Realized Cap = mean(price * realized_price)
    Returns: -1.0 to 1.0 normalized float
    """
    required_cols = ["price", "realized_price"]
    missing = [c for c in required_cols if c not in data.columns]
    if missing:
        raise ValueError(f"realized_cap: missing columns {missing}")
    
    price = data["price"]
    realized_price = data["realized_price"]
    product = price * realized_price
    
    window = params.get("window", 1)
    if window > 1:
        result = product.rolling(window=window, min_periods=1).mean()
    else:
        mean_val = product.mean() if len(product) > 0 else 0.0
        result = pd.Series([mean_val] * len(product), index=data.index)
    
    return finalize_metric(result, "realized_cap")


def nupl(data: pd.DataFrame, **params) -> float:
    """
    NUPL = (Market Cap - Realized Cap) / Market Cap
    Returns: -1.0 to 1.0 normalized float
    """
    required_cols = ["market_cap", "realized_cap"]
    missing = [c for c in required_cols if c not in data.columns]
    if missing:
        raise ValueError(f"nupl: missing columns {missing}")
    
    market_cap = data["market_cap"]
    realized_cap = data["realized_cap"]
    
    with np.errstate(divide='ignore', invalid='ignore'):
        denominator = market_cap.replace(0, np.nan)
        result = (market_cap - realized_cap) / denominator
    
    return finalize_metric(result, "nupl")


def exchange_whale_ratio(data: pd.DataFrame, **params) -> float:
    """
    Whale Ratio = Top 10 inflow wallets / Total inflow
    Returns: -1.0 to 1.0 normalized float
    """
    required_cols = ["whale_deposits", "total_deposits"]
    missing = [c for c in required_cols if c not in data.columns]
    if missing:
        raise ValueError(f"exchange_whale_ratio: missing columns {missing}")
    
    whale_deposits = data["whale_deposits"]
    total_deposits = data["total_deposits"]
    
    with np.errstate(divide='ignore', invalid='ignore'):
        denominator = total_deposits.replace(0, np.nan)
        result = whale_deposits / denominator
    
    return finalize_metric(result, "exchange_whale_ratio")


def mvrv_zscore(data: pd.DataFrame, **params) -> float:
    """
    MVRV Z-Score = (Market Cap - Realized Cap) / Std(Market Cap)
    Returns: -1.0 to 1.0 normalized float
    """
    required_cols = ["market_cap", "realized_cap", "market_cap_std"]
    missing = [c for c in required_cols if c not in data.columns]
    if missing:
        raise ValueError(f"mvrv_zscore: missing columns {missing}")
    
    market_cap = data["market_cap"]
    realized_cap = data["realized_cap"]
    std_dev = data["market_cap_std"]
    
    with np.errstate(divide='ignore', invalid='ignore'):
        denominator = std_dev.replace(0, np.nan)
        result = (market_cap - realized_cap) / denominator
    
    return finalize_metric(result, "mvrv_zscore")


def sopr(data: pd.DataFrame, **params) -> float:
    """
    SOPR = Realized Value / Spent Value
    Returns: -1.0 to 1.0 normalized float
    """
    required_cols = ["realized_value", "spent_value"]
    missing = [c for c in required_cols if c not in data.columns]
    if missing:
        raise ValueError(f"sopr: missing columns {missing}")
    
    realized_value = data["realized_value"]
    spent_value = data["spent_value"]
    
    with np.errstate(divide='ignore', invalid='ignore'):
        denominator = spent_value.replace(0, np.nan)
        result = realized_value / denominator
    
    return finalize_metric(result, "sopr")


def etf_flow_composite(data: pd.DataFrame, **params) -> float:
    """
    ETF Flow Composite = 0.5*NetETF + 0.3*NetStable + 0.2*(-NetExchange)
    Returns: -1.0 to 1.0 normalized float
    """
    required_cols = [
        "etf_inflow", "etf_outflow",
        "stablecoin_in", "stablecoin_out",
        "exchange_deposits", "exchange_withdrawals"
    ]
    
    missing = [c for c in required_cols if c not in data.columns]
    if missing:
        raise ValueError(f"etf_flow_composite: missing columns {missing}")
    
    net_etf = data["etf_inflow"] - data["etf_outflow"]
    net_stable = data["stablecoin_in"] - data["stablecoin_out"]
    net_exchange = data["exchange_deposits"] - data["exchange_withdrawals"]
    
    result = (0.5 * net_etf) + (0.3 * net_stable) + (0.2 * (-net_exchange))
    
    return finalize_metric(result, "etf_flow_composite")


# ==================== REGISTRY ====================
_METRICS = {
    "etf_net_flow": etf_net_flow,
    "exchange_netflow": exchange_netflow,
    "stablecoin_flow": stablecoin_flow,
    "net_realized_pl": net_realized_pl,
    "realized_cap": realized_cap,
    "nupl": nupl,
    "exchange_whale_ratio": exchange_whale_ratio,
    "mvrv_zscore": mvrv_zscore,
    "sopr": sopr,
    "etf_flow_composite": etf_flow_composite,
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
        'etf_inflow': [100, 150, 200, 180, 220, 250, 300, 280, 320, 350],
        'etf_outflow': [80, 120, 180, 160, 200, 220, 280, 260, 300, 320],
        'exchange_deposits': [500, 550, 600, 580, 620, 650, 700, 680, 720, 750],
        'exchange_withdrawals': [450, 500, 550, 530, 570, 600, 650, 630, 670, 700],
        'stablecoin_in': [200, 220, 240, 230, 250, 270, 290, 280, 300, 320],
        'stablecoin_out': [180, 200, 220, 210, 230, 250, 270, 260, 280, 300],
        'realized_profit': [1000, 1100, 1200, 1150, 1250, 1300, 1400, 1350, 1450, 1500],
        'realized_loss': [800, 900, 1000, 950, 1050, 1100, 1200, 1150, 1250, 1300],
        'price': [50000, 51000, 52000, 51500, 52500, 53000, 54000, 53500, 54500, 55000],
        'realized_price': [48000, 49000, 50000, 49500, 50500, 51000, 52000, 51500, 52500, 53000],
        'market_cap': [1e12, 1.02e12, 1.04e12, 1.03e12, 1.05e12, 1.06e12, 1.08e12, 1.07e12, 1.09e12, 1.1e12],
        'realized_cap': [9.6e11, 9.8e11, 1e12, 9.9e11, 1.01e12, 1.02e12, 1.04e12, 1.03e12, 1.05e12, 1.06e12],
        'whale_deposits': [50, 55, 60, 58, 62, 65, 70, 68, 72, 75],
        'total_deposits': [500, 550, 600, 580, 620, 650, 700, 680, 720, 750],
        'market_cap_std': [1e10, 1.1e10, 1.2e10, 1.15e10, 1.25e10, 1.3e10, 1.4e10, 1.35e10, 1.45e10, 1.5e10],
        'realized_value': [1000, 1100, 1200, 1150, 1250, 1300, 1400, 1350, 1450, 1500],
        'spent_value': [800, 900, 1000, 950, 1050, 1100, 1200, 1150, 1250, 1300]
    })
    
    print(f"🧪 {__name__} Self-Test")
    print("=" * 60)
    
    # Her metriği test et
    for metric_name, metric_func in _METRICS.items():
        try:
            result = metric_func(test_data)
            
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