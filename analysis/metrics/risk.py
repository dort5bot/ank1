"""
analysis/metrics/risk.py
Universal Risk Metrics with Data-Model-Aware Normalization
Date: 03/12/2025
Pure mathematical risk calculations with automatic normalization.

funding_risk: Funding rate’in normalden ne kadar uzaklaştığını ölçer
funding_stress_risk: Funding rate’in oynaklığını (volatility) ölç
"""
# analysis/metrics/risk.py
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


# ==================== MODULE CONFIG risk.py====================
_MODULE_CONFIG = {
    # ZORUNLU: Veri modeli ve execution tipi
    "data_model": DataModel.PANDAS,  # "pandas", "numpy", "polars"
    "execution_type": "sync",  # "sync" veya "async"
    "category": "risk", 
    
    "adaptations": {
        "liquidity_depth_risk": {
            "required": True,
            "type": "orderbook"
        },
        "price_impact_risk": {
            "required": True,
            "type": "orderbook"
        },
        "liquidity_gaps": {
            "required": True,
            "type": "orderbook"
        }
    },
    
    # ZORUNLU: Her metrik için gerekli kolonlar
    "required_columns": {
        "volatility_risk": ["close"],
        "spread_risk": ["bid", "ask"],
        "liquidity_depth_risk": ["side", "price", "size"],
        "price_impact_risk": ["side", "price", "size"],
        "taker_pressure_risk": ["volume", "is_buyer_maker"],
        "open_interest_risk": ["open_interest"],
        "open_interest_shock_risk": ["open_interest"],
        "liquidity_gaps": ["side", "price", "size"],
        "funding_risk": ["funding_rate"],
        "funding_stress_risk": ["funding_rate"],
        
    },
    
    # Normalizasyon konfigürasyonu
    "normalization": {
        # Global normalizasyon ayarları
        "global_range": {"min": -1.0, "max": 1.0},
        "default_method": NormalizationMethod.TANH,
        
        # Metrik-specific normalizasyon
        "metric_specific": {
            "volatility_risk": {
                "method": NormalizationMethod.MINMAX,
                "params": {"lookback": 100, "neutral": 0.5},
                "description": "Volatility risk normalized to -1..1"
            },
            "spread_risk": {
                "method": NormalizationMethod.MINMAX,
                "params": {"scale": 0.5},
                "description": "Spread risk normalized to -1..1"
            },
            "liquidity_depth_risk": {
                "method": NormalizationMethod.TANH,
                "params": {"scale": 2.0},
                "description": "Liquidity depth risk using tanh normalization"
            },
            "price_impact_risk": {
                "method": NormalizationMethod.TANH,
                "params": {"scale": 3.0},
                "description": "Price impact risk using tanh normalization"
            },
            "taker_pressure_risk": {
                "method": NormalizationMethod.MINMAX,
                "params": {"lookback": 50},
                "description": "Taker pressure risk normalized to -1..1"
            },
            "funding_risk": {
                "method": NormalizationMethod.ZSCORE,
                "params": {"clip_sigma": 2.5},
                "description": "Funding risk using z-score with clipping"
            },
            "open_interest_risk": {
                "method": NormalizationMethod.TANH,
                "params": {"scale": 5.0},
                "description": "Open interest risk using tanh normalization"
            },
            "liquidity_gaps": {
                "method": NormalizationMethod.TANH,
                "params": {"scale": 2.0},
                "description": "Liquidity gaps using tanh normalization"
            },
            "funding_stress_risk": {
                "method": NormalizationMethod.TANH,
                "params": {"scale": 3.0},
                "description": "Funding stress risk using tanh normalization"
            },
        
        }
    },
    
    # Metrik parametreleri
    "default_params": {
        "volatility_risk": {"window": 20},
        "spread_risk": {},
        "liquidity_depth_risk": {"baseline": 500000, "top_n": 20},
        "price_impact_risk": {"target_amount": 10000, "top_n": 20},
        "taker_pressure_risk": {"window": 10},
        "funding_risk": {"window": 20},
        "open_interest_risk": {},
        "liquidity_gaps": {"threshold": 0.5, "top_n": 20},
        "funding_stress_risk": {"window": 20},
        "open_interest_shock_risk": {"window": 20},
    },
    "data_schema": {
        "order_book": {
            "columns": ["level", "side", "price", "size"],
            "side_values": ["bid", "ask"],
            "sorted_by": {"bid": "price_desc", "ask": "price_asc"}
        }
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
                normalized_cols[col] = Normalizer._normalize_pandas_series(df[col], method, target_range, **params)
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


# ==================== ADAPTATION FUNCTIONS ====================
# TEK adaptasyon fonksiyonu.
# Raw depth data -> [side, price, size] DataFrame
    
def _adapt_orderbook(data: Any, top_n: int = 20) -> pd.DataFrame:
    """
    Core'dan gelen formatları kabul eden tek adaptasyon.
    """
    # 1. Zaten doğru formatta
    if isinstance(data, pd.DataFrame) and 'side' in data.columns:
        return data.head(top_n * 2)  # İlk top_n bid + top_n ask
    
    # 2. RAW Binance depth (bids/asks listeleri) - artık a_core.py bunu göndermeli
    if isinstance(data, dict) and 'bids' in data:
        return _adapt_raw_depth(data, top_n)
    
    # 3. Geçersiz format
    warnings.warn(f"Cannot adapt orderbook data. Type: {type(data)}")
    return pd.DataFrame(columns=['level', 'side', 'price', 'size'])

def _adapt_raw_depth(data: dict, top_n: int = 20) -> pd.DataFrame:
    """
    Raw Binance depth dict'ini adapte eder.
    """
    rows = []
    
    # Bids
    bids = data.get('bids', [])[:top_n]
    for i, (price_str, size_str) in enumerate(bids):
        try:
            rows.append({
                'level': i,
                'side': 'bid',
                'price': float(price_str),
                'size': float(size_str)
            })
        except (ValueError, TypeError):
            continue
    
    # Asks
    asks = data.get('asks', [])[:top_n]
    for i, (price_str, size_str) in enumerate(asks):
        try:
            rows.append({
                'level': i + len(bids),
                'side': 'ask',
                'price': float(price_str),
                'size': float(size_str)
            })
        except (ValueError, TypeError):
            continue
    
    if not rows:
        return pd.DataFrame(columns=['level', 'side', 'price', 'size'])
    
    df = pd.DataFrame(rows)
    
    # Sort bids descending, asks ascending
    df_bids = df[df['side'] == 'bid'].sort_values('price', ascending=False)
    df_asks = df[df['side'] == 'ask'].sort_values('price', ascending=True)
    
    return pd.concat([df_bids, df_asks], ignore_index=True)

# ==================== HELPER DECORATOR ====================
def adaptive_metric(func):
    """
    Akıllı decorator: Config'e bak, gerekirse adapte et.
    """
    metric_name = func.__name__
    adaptation_config = _MODULE_CONFIG.get('adaptations', {}).get(metric_name, {})
    
    def wrapper(data, **kwargs):
        # Adaptasyon gerekli mi?
        if not adaptation_config.get('required', False):
            return func(data, **kwargs)
        
        # Adaptasyon gerekli - orderbook type
        if adaptation_config.get('type') == 'orderbook':
            top_n = kwargs.get('top_n', 20)
            try:
                adapted_data = _adapt_orderbook(data, top_n)
                return func(adapted_data, **kwargs)
            except Exception as e:
                # CLEAR error, no silent failure
                raise ValueError(f"{metric_name} adaptation failed: {str(e)}")
        
        # Diğer adaptasyon tipleri buraya eklenebilir
        return func(data, **kwargs)
    
    # Preserve metadata
    wrapper.__name__ = func.__name__
    wrapper.__doc__ = func.__doc__
    
    return wrapper

# ==================== METRIC FUNCTIONS ====================
@adaptive_metric
def liquidity_depth_risk(data: pd.DataFrame, **params) -> float:
    """
    TOP-N order book liquidity risk.
    Automatically adapts raw depth data if needed.
    """
    top_n = params.get('top_n', 20)
    baseline = params.get('baseline', 500_000)
    
    # Data artık adapte edilmiş olarak geliyor
    bids = data[data['side'] == 'bid'].head(top_n)
    asks = data[data['side'] == 'ask'].head(top_n)
    
    if bids.empty or asks.empty:
        # Maximum risk if no liquidity
        return finalize_metric(1.0, 'liquidity_depth_risk')
    
    bid_liquidity = (bids['price'] * bids['size']).sum()
    ask_liquidity = (asks['price'] * asks['size']).sum()
    total_liquidity = bid_liquidity + ask_liquidity
    
    if total_liquidity <= 0:
        return finalize_metric(1.0, 'liquidity_depth_risk')
    
    liquidity_ratio = total_liquidity / baseline
    risk = 1.0 / (1.0 + liquidity_ratio)
    
    return finalize_metric(risk, 'liquidity_depth_risk')

@adaptive_metric
def price_impact_risk(data: pd.DataFrame, **params) -> float:
    """
    TOP-N price impact risk using depth consumption.
    Automatically adapts raw depth data if needed.
    """
    top_n = params.get('top_n', 20)
    target_amount = params.get('target_amount', 10_000)
    
    # Data artık adapte edilmiş olarak geliyor
    asks = data[data['side'] == 'ask'].head(top_n)
    
    if asks.empty:
        # Maximum impact if no asks
        return finalize_metric(1.0, 'price_impact_risk')
    
    remaining = target_amount
    cost = 0.0
    filled = 0.0
    
    for _, row in asks.iterrows():
        if remaining <= 0:
            break
        
        level_liquidity = row['price'] * row['size']
        take = min(level_liquidity, remaining)
        
        cost += take
        filled += take / row['price']
        remaining -= take
    
    if filled == 0:
        return finalize_metric(1.0, 'price_impact_risk')
    
    vwap_price = cost / filled
    best_price = asks.iloc[0]['price']
    impact = abs(vwap_price - best_price) / best_price
    
    return finalize_metric(impact, 'price_impact_risk')

def volatility_risk(data: pd.DataFrame, **params) -> float:
    """
    Calculate volatility risk based on price returns.
    No adaptation needed.
    """
    if 'close' not in data.columns:
        raise ValueError("volatility_risk: 'close' column required")
    
    close = data['close']
    window = params.get('window', 20)
    
    returns = np.log(close / close.shift(1))
    rolling_vol = returns.rolling(window=window, min_periods=1).std()
    
    return finalize_metric(rolling_vol, 'volatility_risk')

def spread_risk(data: pd.DataFrame, **params) -> float:
    """
    Calculate bid-ask spread risk.
    No adaptation needed.
    """
    required = ['bid', 'ask']
    missing = [c for c in required if c not in data.columns]
    if missing:
        raise ValueError(f"spread_risk: '{missing[0]}' column required" 
                        if len(missing) == 1 else 
                        f"spread_risk: '{missing[0]}' and '{missing[1]}' columns required")
                        
    
    bid = data['bid']
    ask = data['ask']
    mid_price = (bid + ask) / 2
    
    spread_pct = (ask - bid) / mid_price.replace(0, np.nan)
    
    return finalize_metric(spread_pct, 'spread_risk')

def taker_pressure_risk(data: pd.DataFrame, **params) -> float:
    """
    Calculate taker pressure risk based on trade imbalance.
    Returns: -1.0 to 1.0 normalized float
    """
    # 1. Input validation
    required_cols = ["volume", "is_buyer_maker"]
    missing = [c for c in required_cols if c not in data.columns]
    if missing:
        raise ValueError(f"taker_pressure_risk: missing columns {missing}")
    
    # 2. Pure calculation
    window = params.get("window", 10)
    
    # Taker buy volume (is_buyer_maker=False means taker is buying)
    taker_buy_volume = data["volume"] * (~data["is_buyer_maker"].astype(bool))
    
    # Taker sell volume (is_buyer_maker=True means taker is selling)
    taker_sell_volume = data["volume"] * data["is_buyer_maker"].astype(bool)
    
    # Rolling sums
    rolling_buy = taker_buy_volume.rolling(window=window, min_periods=1).sum()
    rolling_sell = taker_sell_volume.rolling(window=window, min_periods=1).sum()
    
    # Imbalance = (buy - sell) / (buy + sell), range -1..1
    total_volume = rolling_buy + rolling_sell
    imbalance = (rolling_buy - rolling_sell) / total_volume.replace(0, np.nan)
    imbalance = imbalance.fillna(0.0)
    
    # 3. Return normalized final value
    return finalize_metric(imbalance, "taker_pressure_risk")


def open_interest_risk(data: pd.DataFrame, **params) -> float:
    """
    Calculate open interest change risk.
    Returns: -1.0 to 1.0 normalized float
    """
    # 1. Input validation
    if "open_interest" not in data.columns:
        raise ValueError("open_interest_risk: 'open_interest' column required")
    
    # 2. Pure calculation
    open_interest = data["open_interest"]
    
    # Percentage change (log for better normalization)
    pct_change = open_interest.pct_change()
    log_change = np.log1p(pct_change)  # log(1 + x)
    
    # 3. Return normalized final value
    return finalize_metric(log_change, "open_interest_risk")

def open_interest_shock_risk(data: pd.DataFrame, **params) -> float:
    """Calculate open interest shock risk"""
    if "open_interest" not in data.columns:
        raise ValueError("open_interest_shock_risk: 'open_interest' column required")
    
    oi = data["open_interest"]
    window = params.get("window", 20)
    
    # Örnek: Large percentage changes
    pct_change = oi.pct_change()
    shock = pct_change.abs().rolling(window).mean()
    
    return finalize_metric(shock, "open_interest_shock_risk")


@adaptive_metric
def liquidity_gaps(data: pd.DataFrame, **params) -> float:
    """
    Calculates liquidity gaps from order book data.
    Uses adapted [side, price, size] format via @adaptive_metric.
    
    Args:
        data: Adapted order book DataFrame with ['level', 'side', 'price', 'size'] columns
        **params: 
            threshold: gap threshold price difference (default: 0.5)
            top_n: number of price levels to consider per side (default: 20)
    
    Returns:
        Normalized float in range [-1.0, 1.0] according to _MODULE_CONFIG
    """
    threshold = params.get("threshold", 0.5)
    top_n = params.get("top_n", 20)
    
    try:
        # Data artık @adaptive_metric tarafından adapte edilmiş olarak geliyor
        # Format: ['level', 'side', 'price', 'size']
        
        # 1. İlk top_n bid ve ask seviyelerini al
        bids = data[data['side'] == 'bid'].head(top_n)
        asks = data[data['side'] == 'ask'].head(top_n)
        
        if bids.empty or asks.empty:
            # Minimum liquidity, maximum gaps risk
            return finalize_metric(1.0, "liquidity_gaps")
        
        # 2. Bid'leri fiyata göre sırala (yüksekten düşüğe)
        #    Ask'leri fiyata göre sırala (düşükten yükseğe)
        bids_sorted = bids.sort_values('price', ascending=False)
        asks_sorted = asks.sort_values('price', ascending=True)
        
        # 3. Fiyatları al
        bid_prices = bids_sorted['price'].values
        ask_prices = asks_sorted['price'].values
        
        # 4. Gap hesapla (ardışık fiyatlar arası fark)
        #    Bids: price_i - price_{i+1} (descending order, so positive gaps)
        #    Asks: price_{i+1} - price_i (ascending order, so positive gaps)
        if len(bid_prices) > 1:
            bid_gaps = np.diff(bid_prices)
        else:
            bid_gaps = np.array([0.0])
        
        if len(ask_prices) > 1:
            ask_gaps = np.diff(ask_prices)
        else:
            ask_gaps = np.array([0.0])
        
        # 5. Gap analizi
        #    a) Gap oranı: threshold üstü gap sayısı / toplam gap sayısı
        #    b) Ortalama gap büyüklüğü
        #    c) Maksimum gap
        
        if len(bid_gaps) > 0:
            # Threshold'u aşan gap'lerin oranı
            bid_gap_ratio = np.sum(bid_gaps > threshold) / len(bid_gaps)
            # Normalize edilmiş ortalama gap (0-1 arası)
            if np.max(bid_gaps) > 0:
                bid_avg_gap_norm = np.mean(bid_gaps) / np.max(bid_gaps)
            else:
                bid_avg_gap_norm = 0.0
        else:
            bid_gap_ratio = 0.0
            bid_avg_gap_norm = 0.0
        
        if len(ask_gaps) > 0:
            ask_gap_ratio = np.sum(ask_gaps > threshold) / len(ask_gaps)
            if np.max(ask_gaps) > 0:
                ask_avg_gap_norm = np.mean(ask_gaps) / np.max(ask_gaps)
            else:
                ask_avg_gap_norm = 0.0
        else:
            ask_gap_ratio = 0.0
            ask_avg_gap_norm = 0.0
        
        # 6. Composite gap metric
        #    %60 gap oranı + %40 ortalama gap büyüklüğü
        bid_composite = 0.6 * bid_gap_ratio + 0.4 * bid_avg_gap_norm
        ask_composite = 0.6 * ask_gap_ratio + 0.4 * ask_avg_gap_norm
        
        # 7. İki tarafın ortalaması
        gap_metric = (bid_composite + ask_composite) / 2
        
        # 8. Additional check: large single gap
        max_bid_gap = np.max(bid_gaps) if len(bid_gaps) > 0 else 0.0
        max_ask_gap = np.max(ask_gaps) if len(ask_gaps) > 0 else 0.0
        max_gap = max(max_bid_gap, max_ask_gap)
        
        # 9. Penalize very large gaps
        if max_gap > threshold * 5:  # 5x threshold'dan büyükse
            gap_metric = min(gap_metric + 0.3, 1.0)  # Risk'i artır
        
        # 10. Convert to Series for consistent normalization
        gap_series = pd.Series([gap_metric])
        
        return finalize_metric(gap_series, "liquidity_gaps")
    
    except Exception as e:
        warnings.warn(f"liquidity_gaps calculation error: {e}")
        # Return neutral risk on error
        return finalize_metric(0.5, "liquidity_gaps")
        
def funding_risk(data: pd.DataFrame, **params) -> float:
    """
    Calculate funding rate risk using z-score.
    Returns: -1.0 to 1.0 normalized float
    """
    # 1. Input validation
    if "funding_rate" not in data.columns:
        raise ValueError("funding_risk: 'funding_rate' column required")
    
    # 2. Pure calculation
    funding_rate = data["funding_rate"]
    window = params.get("window", 20)
    
    # Rolling statistics
    rolling_mean = funding_rate.rolling(window=window, min_periods=1).mean()
    rolling_std = funding_rate.rolling(window=window, min_periods=1).std().replace(0, 1.0)
    
    # Z-score
    z_score = (funding_rate - rolling_mean) / rolling_std
    
    # 3. Return normalized final value
    return finalize_metric(z_score, "funding_risk")

def funding_stress_risk(data: pd.DataFrame, **params) -> float:
    """Calculate funding stress risk"""
    if "funding_rate" not in data.columns:
        raise ValueError("funding_stress_risk: 'funding_rate' column required")
    
    funding_rate = data["funding_rate"]
    window = params.get("window", 20)
    
    # Örnek: funding rate volatility
    rolling_std = funding_rate.rolling(window=window).std()
    
    return finalize_metric(rolling_std, "funding_stress_risk")


    
# ==================== DATA MODEL SPECIFIC EXAMPLES ====================
# Pandas based implementations
def pandas_volatility_risk(data: pd.DataFrame, **params) -> float:
    """Pandas tabanlı volatilite riski (volatility_risk ile aynı)"""
    return volatility_risk(data, **params)


# NumPy based implementation (sync)  
def numpy_spread_risk(data: np.ndarray, **params) -> float:
    """
    NumPy tabanlı spread risk hesaplaması.
    Assumes data structure: [bid, ask, ...]
    """
    if data.ndim == 2 and data.shape[1] >= 2:
        bid = data[:, 0]
        ask = data[:, 1]
    elif data.ndim == 1 and len(data) >= 2:
        bid = data[0]
        ask = data[1]
    else:
        return 0.0
    
    # Calculate spread percentage
    mid_price = (bid + ask) / 2
    mid_price = np.where(mid_price == 0, np.nan, mid_price)
    spread_pct = (ask - bid) / mid_price
    spread_pct = np.nan_to_num(spread_pct, nan=0.0)
    
    # Convert to pandas Series for consistent normalization
    spread_series = pd.Series(spread_pct)
    return finalize_metric(spread_series, "spread_risk")



# ==================== REGISTRY ====================
_METRICS = {
    "volatility_risk": volatility_risk,
    "spread_risk": spread_risk,
    "liquidity_depth_risk": liquidity_depth_risk,
    "price_impact_risk": price_impact_risk,
    "taker_pressure_risk": taker_pressure_risk,
    "funding_risk": funding_risk,
    "open_interest_risk": open_interest_risk,
    "liquidity_gaps": liquidity_gaps,
    "funding_stress_risk": funding_stress_risk,
    "open_interest_shock_risk": open_interest_shock_risk,
    
    "pandas_volatility_risk": pandas_volatility_risk,
    "numpy_spread_risk": numpy_spread_risk,
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
        'bid': [99.5, 101.5, 100.5, 103.5, 102.5, 105.5, 104.5, 107.5, 106.5, 109.5],
        'ask': [100.5, 102.5, 101.5, 104.5, 103.5, 106.5, 105.5, 108.5, 107.5, 110.5],
        'bid_price': [99.5, 101.5, 100.5, 103.5, 102.5, 105.5, 104.5, 107.5, 106.5, 109.5],
        'bid_size': [1000, 1200, 1100, 1300, 1250, 1400, 1350, 1500, 1450, 1600],
        'ask_price': [100.5, 102.5, 101.5, 104.5, 103.5, 106.5, 105.5, 108.5, 107.5, 110.5],
        'ask_size': [800, 900, 850, 950, 925, 1000, 975, 1050, 1025, 1100],
        'volume': [10000, 12000, 11000, 13000, 12500, 14000, 13500, 15000, 14500, 16000],
        'is_buyer_maker': [True, False, True, False, True, False, True, False, True, False],
        'funding_rate': [0.0001, 0.0002, -0.0001, 0.0003, -0.0002, 0.0004, -0.0003, 0.0005, -0.0004, 0.0006],
        'open_interest': [100000, 102000, 101000, 104000, 103000, 106000, 105000, 108000, 107000, 110000]
    })
    
    print(f"🧪 {__name__} Self-Test")
    print("=" * 60)
    
    # Her metriği test et
    for metric_name, metric_func in _METRICS.items():
        try:
            # NumPy based metrics need different input
            if metric_name == "numpy_spread_risk":
                # Create numpy array from test data
                numpy_data = test_data[['bid', 'ask']].values
                result = metric_func(numpy_data)
            else:
                result = metric_func(test_data)
            
            # Kontroller
            is_float = isinstance(result, float)
            in_range = -1.0 <= result <= 1.0 if is_float else False
            not_nan = not np.isnan(result) if is_float else False
            not_inf = not np.isinf(result) if is_float else False
            
            passed = is_float and in_range and not_nan and not_inf
            
            if passed:
                print(f"✅ {metric_name}: {result:.4f}")
            else:
                status = []
                if not is_float: status.append("not float")
                if not in_range: status.append("not in range")
                if not not_nan: status.append("nan")
                if not not_inf: status.append("inf")
                print(f"❌ {metric_name}: {result} ({', '.join(status)})")
            
            results[metric_name] = passed
            
        except Exception as e:
            print(f"❌ {metric_name}: ERROR - {e}")
            results[metric_name] = False
    
    # Config test
    try:
        config = get_module_config()
        has_data_model = "data_model" in config
        has_required_columns = "required_columns" in config
        has_normalization = "normalization" in config
        print(f"\n📋 Config: data_model={has_data_model}, required_columns={has_required_columns}, normalization={has_normalization}")
        results["config_valid"] = has_data_model and has_required_columns and has_normalization
    except Exception as e:
        print(f"❌ Config ERROR: {e}")
        results["config_valid"] = False
    
    print(f"\n📊 Results: {sum(results.values())}/{len(results)} passed")
    return results

if __name__ == "__main__":
    self_test()
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    # risk.py'ye ekleyin (Normalizer sınıfından sonra)
