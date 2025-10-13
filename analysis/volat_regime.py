"""
Volatility & Regime Module
File: analysis/volat_regime.py

Beklenen çıktılar:
- score: 0..1 normalize edilmiş overall volatility/regime score
- signal: "trend" / "range" / "neutral"
- components: bileşen skorları (hv, atr, bw, var_ratio, premium)
- explain: kısa metin açıklama
- regime_label: Trend / Range

Özellikler:
- Batch ve async destekli (multi-symbol)
- CPU-bound hesaplamalar için ThreadPoolExecutor kullanımı
- Config üzerinden parametreler (analysis/config/c_volat.py)
- Basit Prometheus wrapper yer tutucu (opsiyonel entegrasyon)
"""
from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor
import logging
import math
from typing import Dict, List, Any, Optional, Tuple

import numpy as np
import pandas as pd

# Try importing optional libs; fallback yapacağız
try:
    from arch import arch_model  # for GARCH(1,1)
    _HAS_ARCH = True
except Exception:
    _HAS_ARCH = False

from scipy.stats import kurtosis
from scipy.signal import detrend

# Local imports within project
try:
    from analysis.analysis_base_module import BaseAnalysisModule
except Exception:
    # Fallback base if your project's base class not available during isolated testing
    class BaseAnalysisModule:
        def __init__(self, config: Dict[str, Any] = None):
            self.config = config or {}

logger = logging.getLogger(__name__)

# Helper functions
def ewma_std(x: np.ndarray, span: int) -> np.ndarray:
    """
    Exponentially weighted moving std (vectorized approximate)
    """
    series = pd.Series(x)
    return series.ewm(span=span, adjust=False).std().to_numpy()

def hurst_exponent(ts: np.ndarray, max_lag: int = 20) -> float:
    """
    Estimate Hurst exponent via rescaled range (R/S) method (simple)
    """
    N = len(ts)
    lags = np.floor(np.logspace(0.0, np.log10(max_lag), num=20)).astype(int)
    lags = np.unique(lags[lags > 1])
    if len(lags) < 2:
        return 0.5
    rs = []
    for lag in lags:
        n_segments = N // lag
        if n_segments < 2:
            continue
        vals = []
        for i in range(n_segments):
            seg = ts[i * lag:(i + 1) * lag]
            if len(seg) < 2:
                continue
            mean = np.mean(seg)
            Y = np.cumsum(seg - mean)
            R = np.max(Y) - np.min(Y)
            S = np.std(seg, ddof=1)
            if S > 0:
                vals.append(R / S)
        if vals:
            rs.append(np.mean(vals))
    if len(rs) < 2:
        return 0.5
    import scipy.stats as sps
    slope, _, _, _, _ = sps.linregress(np.log(lags[:len(rs)]), np.log(rs))
    return slope

def shannon_entropy(x: np.ndarray, bins: int = 50) -> float:
    p, _ = np.histogram(x, bins=bins, density=True)
    p = p[p > 0]
    return -np.sum(p * np.log(p))

def variance_ratio_test(returns: np.ndarray, lag: int = 2) -> float:
    """
    Lo-MacKinlay variance ratio statistic simplified: returns ratio of variances
    """
    if len(returns) < lag + 1:
        return 1.0
    var_q = np.var(np.sum(returns.reshape(-1, lag), axis=1), ddof=1)
    var_1 = np.var(returns, ddof=1)
    if var_1 == 0:
        return 1.0
    return var_q / (lag * var_1)

def atr(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int = 14) -> np.ndarray:
    # True range and rolling mean
    tr1 = high - low
    tr2 = np.abs(high - np.concatenate(([close[0]], close[:-1])))
    tr3 = np.abs(low - np.concatenate(([close[0]], close[:-1])))
    tr = np.maximum.reduce([tr1, tr2, tr3])
    tr_s = pd.Series(tr).rolling(period, min_periods=1).mean().to_numpy()
    return tr_s

# Module class
class VolatRegimeModule(BaseAnalysisModule):
    """
    Volatility & Regime analysis module.
    Inherits BaseAnalysisModule of the project.
    """

    version = "1.0.0"
    module_name = "volat_regime"

    def __init__(self, config: Optional[Dict[str, Any]] = None, executor: Optional[ThreadPoolExecutor] = None):
        super().__init__(config=config or {})
        # load default config if not provided
        from analysis.config.c_volat import CONFIG as DEFAULT
        cfg = DEFAULT.copy()
        if config:
            cfg.update(config)
        self.config = cfg
        self._executor = executor or ThreadPoolExecutor(max_workers=self.config.get("max_workers", 4))
        self._prometheus = None  # placeholder for metrics integration

    async def run_batch(self, symbols: List[str], data_provider, interval: str = "1h") -> Dict[str, Any]:
        """
        Entry point for batch processing multiple symbols.
        data_provider is expected to implement async get_ohlcv(symbol, interval, limit)
        """
        tasks = []
        for s in symbols:
            tasks.append(self.run_symbol(s, data_provider, interval))
        results = await asyncio.gather(*tasks, return_exceptions=True)
        out = {}
        for sym, res in zip(symbols, results):
            if isinstance(res, Exception):
                logger.exception("Error computing symbol %s", sym)
                out[sym] = {"error": str(res)}
            else:
                out[sym] = res
        return out

    async def run_symbol(self, symbol: str, data_provider, interval: str = "1h") -> Dict[str, Any]:
        """
        Compute metrics & regime for a single symbol.
        """
        # Fetch OHLCV async from provided data provider
        ohlcv: pd.DataFrame = await data_provider.get_ohlcv(symbol=symbol, interval=interval, limit=self.config.get("ohlcv_limit", 500))
        if ohlcv is None or ohlcv.empty:
            raise ValueError(f"No OHLCV data for {symbol}")

        # Ensure columns: ['open','high','low','close','volume','timestamp']
        df = ohlcv.copy()
        # convert to numpy arrays for heavy compute
        close = df["close"].to_numpy(dtype=float)
        high = df["high"].to_numpy(dtype=float)
        low = df["low"].to_numpy(dtype=float)
        volume = df["volume"].to_numpy(dtype=float)

        # Offload CPU-bound heavy compute to threadpool
        loop = asyncio.get_running_loop()
        components = await loop.run_in_executor(self._executor, self._compute_components, close, high, low, volume, df)

        # Combine into score & signal
        score, signal, explain = self._aggregate_components(components)

        result = {
            "symbol": symbol,
            "timestamp": int(df["timestamp"].iloc[-1]) if "timestamp" in df.columns else None,
            "score": float(score),
            "signal": signal,
            "components": components,
            "explain": explain,
            "regime_label": "Trend" if signal == "trend" else ("Range" if signal == "range" else "Neutral"),
        }
        return result

    def _compute_components(self, close: np.ndarray, high: np.ndarray, low: np.ndarray, volume: np.ndarray, df: pd.DataFrame) -> Dict[str, float]:
        """
        CPU-bound calculations executed in threadpool.
        Returns normalized component scores between 0 and 1 (higher => more volatile / trending depending)
        """
        cfg = self.config

        # 1) Historical Volatility (annualized) - use log returns
        logret = np.diff(np.log(close + 1e-12))
        hv_daily = np.std(logret, ddof=1) * math.sqrt(cfg.get("annualization", 365))
        # Normalize hv by config hv_band
        hv_norm = min(1.0, hv_daily / (cfg.get("hv_scale", 0.8)))

        # 2) ATR (last value relative to price)
        atr_arr = atr(high, low, close, period=cfg.get("atr_period", 14))
        last_atr = atr_arr[-1] if len(atr_arr) > 0 else 0.0
        atr_norm = min(1.0, (last_atr / (np.mean(close[-cfg.get("atr_lookback", 50):]) + 1e-12)) / cfg.get("atr_scale", 0.01))

        # 3) Bollinger Width (std/ma)
        ma = pd.Series(close).rolling(cfg.get("bb_window", 20), min_periods=1).mean().to_numpy()
        std = pd.Series(close).rolling(cfg.get("bb_window", 20), min_periods=1).std(ddof=1).to_numpy()
        last_bw = (std[-1] * 2) / (ma[-1] + 1e-12)
        bw_norm = min(1.0, last_bw / cfg.get("bb_scale", 0.02))

        # 4) Variance Ratio test (momentum vs mean-reverting)
        vr = variance_ratio_test(logret, lag=cfg.get("var_lag", 2))
        # VR > 1 => trending; <1 => mean reverting. Normalize to [0,1] with 1 => trending strong
        vr_norm = 1 / (1 + math.exp(- (vr - 1) * cfg.get("var_sensitivity", 4)))  # sigmoid around 1

        # 5) Hurst exponent (persistence)
        try:
            hurst = hurst_exponent(logret, max_lag=cfg.get("hurst_max_lag", 20))
        except Exception:
            hurst = 0.5
        # Map hurst: >0.5 trending => closer to 1, <0.5 mean-revert => 0
        hurst_norm = min(1.0, max(0.0, (hurst - 0.2) / 0.8))

        # 6) Entropy (lower entropy => more structured/trending)
        ent = shannon_entropy(logret, bins=cfg.get("entropy_bins", 50))
        # We invert entropy so lower entropy -> higher score of trend-structure (normalize empirically)
        ent_norm = 1 - min(1.0, ent / cfg.get("entropy_scale", 3.5))

        # 7) GARCH(1,1) conditional volatility vs realized
        garch_score = 0.0
        try:
            if _HAS_ARCH:
                am = arch_model(logret * 100, vol="Garch", p=1, q=1, rescale=False)
                res = am.fit(disp="off", last_obs=len(logret) - 1)
                cond_vol = res.conditional_volatility / 100.0
                garch_pred = np.mean(cond_vol[-5:]) if len(cond_vol) > 0 else np.std(logret)
                realized = np.std(logret[-cfg.get("realized_window", 20):]) if len(logret) >= cfg.get("realized_window", 20) else np.std(logret)
                if realized > 0:
                    garch_score = min(1.0, abs(garch_pred - realized) / (realized + 1e-12))
                else:
                    garch_score = 0.0
            else:
                # fallback: use ewma std as proxy
                ew = ewma_std(logret, span=cfg.get("garch_proxy_span", 20))
                garch_score = min(1.0, abs(ew[-1] - np.std(logret[-cfg.get("realized_window", 20):])) / (np.std(logret[-cfg.get("realized_window", 20):]) + 1e-12))
        except Exception:
            garch_score = 0.0

        # 8) Premium / implied-realized proxy (using futures premium if available in df)
        # The data_provider should include a 'premium' column if available (optional)
        premium_norm = 0.0
        if "premium" in df.columns:
            prem = float(df["premium"].iloc[-1])
            premium_norm = min(1.0, abs(prem) / cfg.get("premium_scale", 0.05))

        # 9) Range Expansion Index (REI) approximate: counts of expanding range relative to rolling avg
        ranges = high - low
        rei = (ranges[-1] / (np.mean(ranges[-cfg.get("rei_lookback", 20):]) + 1e-12)) if len(ranges) >= 1 else 1.0
        rei_norm = min(1.0, (rei - 1) / cfg.get("rei_scale", 0.5) if rei > 1 else 0.0)

        components = {
            "historical_volatility": float(hv_norm),
            "atr": float(atr_norm),
            "bollinger_width": float(bw_norm),
            "variance_ratio": float(vr_norm),
            "hurst": float(hurst_norm),
            "entropy_struct": float(ent_norm),
            "garch_implied_realized_diff": float(garch_score),
            "premium": float(premium_norm),
            "rei": float(rei_norm),
        }

        return components

    def _aggregate_components(self, components: Dict[str, float]) -> Tuple[float, str, str]:
        """
        Weight and combine component scores into final score and regime signal.
        Signal logic:
          - If trend-related components (variance_ratio, hurst, rei) high => "trend"
          - If volatility components high but VR < 0.5 and hurst < 0.45 => "range"
          - else "neutral"
        """
        cfg = self.config
        weights = cfg.get("weights", {
            "historical_volatility": 0.15,
            "atr": 0.1,
            "bollinger_width": 0.1,
            "variance_ratio": 0.2,
            "hurst": 0.15,
            "entropy_struct": 0.05,
            "garch_implied_realized_diff": 0.15,
            "premium": 0.05,
            "rei": 0.05,
        })
        # normalize weights if they don't sum to 1
        s = sum(weights.values())
        if s == 0:
            s = 1.0
        weights = {k: v / s for k, v in weights.items()}

        # score: weighted sum
        score = 0.0
        for k, v in components.items():
            score += weights.get(k, 0.0) * v
        score = max(0.0, min(1.0, score))

        # Signal rules
        trend_strength = (components.get("variance_ratio", 0) * 0.6 +
                          components.get("hurst", 0) * 0.3 +
                          components.get("rei", 0) * 0.1)

        volatility_strength = (components.get("historical_volatility", 0) * 0.5 +
                               components.get("atr", 0) * 0.2 +
                               components.get("bollinger_width", 0) * 0.2 +
                               components.get("garch_implied_realized_diff", 0) * 0.1)

        # thresholds configurable
        t_trend = cfg.get("trend_threshold", 0.6)
        t_range = cfg.get("range_threshold", 0.55)

        if trend_strength >= t_trend and components.get("variance_ratio", 0) > 0.6:
            signal = "trend"
        elif volatility_strength >= t_range and components.get("variance_ratio", 0) < 0.45:
            signal = "range"
        else:
            signal = "neutral"

        # Explain short text
        explain = f"trend_strength={trend_strength:.3f}, vol_strength={volatility_strength:.3f}, score={score:.3f}"

        return score, signal, explain

# Compatibility helper for old run(symbol, priority) signature
async def run(symbol: str, priority: int = 5, config: Optional[Dict[str, Any]] = None, data_provider=None):
    """
    Backward-compatible run function expected by analysis_core.
    If data_provider is None, raises.
    """
    if data_provider is None:
        raise ValueError("data_provider must be provided to run()")
    module = VolatRegimeModule(config=config)
    return await module.run_symbol(symbol=symbol, data_provider=data_provider)
