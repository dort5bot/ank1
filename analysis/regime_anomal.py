# analysis/regime_anomal.py
"""
Regime Change Detection & Anomaly Module
File: regime_anomal.py

Purpose:
- Detect sudden regime changes and anomalies using spot + futures data:
  /api/v3/klines, /fapi/v1/openInterestHist, /fapi/v1/fundingRate,
  /fapi/v1/markPriceKlines, /fapi/v1/openInterest and optionally markPrice websocket.
- Computes rolling z-score, rolling skewness/kurtosis, cumulative return deviation.
- Uses CUSUM (changepoint), IsolationForest (anomaly), and a lightweight spectral-residual placeholder.
- Async, vectorized with pandas/numpy and designed to fit into the project's BaseAnalysisModule interface.

Outputs:
{
  "score": float (0-1),
  "signal": "anomaly"|"regime_change"|"neutral",
  "components": {"cusum": 0-1, "iso_forest": 0-1, "zscore": 0-1, "cumret_dev": 0-1},
  "explain": { ... detailed fields ... },
  "raw": { optionally include small summary of computed series / meta }
}

Compatibility:
- Inherits BaseAnalysisModule if available; otherwise provides a backward-compatible run(symbol, priority) function.
- Accepts a data_provider with async methods:
    get_klines(symbol, interval, limit) -> pd.DataFrame with columns ['open_time','open','high','low','close','volume',...]
    get_open_interest_hist(symbol, period, limit) -> pd.DataFrame (timestamp, openInterest)
    get_funding_rate(symbol, limit) -> pd.DataFrame (time, fundingRate)
    get_markprice_klines(symbol, interval, limit) -> pd.DataFrame
    get_open_interest(symbol) -> float
  This allows DI for testing / mocking.

Notes:
- This module is self-contained in terms of algorithm logic but expects the environment's data_provider or utils.binance_api client.
- Optional external deps:
    scikit-learn (IsolationForest) - used if available, otherwise a fallback simple median-deviation anomaly detector is used.
    scipy (for skew/kurtosis); if not available, uses pandas' skew/kurt.
"""

from __future__ import annotations

import asyncio
import logging
import math
from dataclasses import dataclass
from typing import Dict, Any, Optional, List, Sequence, Callable

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

# Try optional imports
try:
    from sklearn.ensemble import IsolationForest
    _HAVE_SKLEARN = True
except Exception:
    _HAVE_SKLEARN = False

try:
    from scipy import signal  # placeholder for spectral-residual step if we want to implement
    _HAVE_SCIPY = True
except Exception:
    _HAVE_SCIPY = False

# Try to import BaseAnalysisModule from project
try:
    from .analysis_base_module import BaseAnalysisModule
except Exception:
    BaseAnalysisModule = object  # graceful fallback; we still expose run()

# ---------------------------
# Default configuration
# ---------------------------
DEFAULT_CONFIG = {
    "klines_interval": "1h",
    "klines_limit": 500,
    "oi_hist_period": "1h",
    "oi_hist_limit": 500,
    "funding_limit": 500,
    "rolling_window": 50,
    "zscore_threshold": 3.0,
    "cusum_h": 5.0,  # CUSUM threshold (tunable)
    "cusum_k": 0.5,  # slack
    "iso_n_estimators": 100,
    "iso_contamination": 0.01,
    "combine_weights": {"cusum": 0.4, "iso": 0.3, "zscore": 0.2, "cumret": 0.1},
    "min_points": 100,
    "normalize_bounds": (0.0, 1.0),
}


# ---------------------------
# Helper algorithms
# ---------------------------
def rolling_zscore(series: pd.Series, window: int) -> pd.Series:
    """
    Rolling z-score: (x - rolling_mean) / rolling_std
    If std == 0 -> 0
    """
    rm = series.rolling(window=window, min_periods=1).mean()
    rs = series.rolling(window=window, min_periods=1).std(ddof=0)
    zs = (series - rm) / rs.replace(0, np.nan)
    zs = zs.fillna(0.0)
    return zs.abs()  # we care about magnitude


def cumulative_return_deviation(close: pd.Series, window: int) -> pd.Series:
    """
    Compute cumulative return deviation from rolling mean cumulative return.
    Steps:
    - compute log returns
    - compute rolling cumulative sum over window
    - compute rolling mean and deviation
    """
    r = np.log(close).diff().fillna(0.0)
    cr = r.rolling(window=window, min_periods=1).sum()
    cr_mean = cr.rolling(window=window, min_periods=1).mean()
    cr_std = cr.rolling(window=window, min_periods=1).std(ddof=0).replace(0, np.nan)
    dev = ((cr - cr_mean).abs() / cr_std).fillna(0.0)
    return dev


def simple_cusum(series: Sequence[float], h: float = 5.0, k: float = 0.5) -> Dict[str, Any]:
    """
    Basic two-sided CUSUM change detection.
    Returns:
      { "pos": [indices], "neg": [indices], "cusum_score": normalized_score }
    Score is normalized between 0-1 by clipping number of detections vs length.
    """
    s_pos = 0.0
    s_neg = 0.0
    pos_idx = []
    neg_idx = []

    # Use demeaned series
    arr = np.asarray(series, dtype=float)
    if arr.size == 0:
        return {"pos": pos_idx, "neg": neg_idx, "cusum_score": 0.0}

    mu = np.nanmedian(arr)
    for i, x in enumerate(arr):
        diff = x - mu - k
        s_pos = max(0.0, s_pos + diff)
        diffn = - (x - mu) - k
        s_neg = max(0.0, s_neg + diffn)
        if s_pos > h:
            pos_idx.append(i)
            s_pos = 0.0
        if s_neg > h:
            neg_idx.append(i)
            s_neg = 0.0

    total = len(pos_idx) + len(neg_idx)
    # Normalize: expect at most length/10 changes -> map to 0-1
    norm = min(1.0, (total / max(1, arr.size / 10.0)))
    return {"pos": pos_idx, "neg": neg_idx, "cusum_score": float(norm)}


def spectral_residual_score(series: pd.Series) -> pd.Series:
    """
    Lightweight spectral-residual placeholder:
    If scipy.signal available, compute periodogram and subtract smooth spectral envelope.
    Otherwise fallback to simple high-pass filter via differencing.
    Returns anomaly magnitude series (normalized-ish).
    """
    x = np.asarray(series.fillna(method="ffill").fillna(0.0), dtype=float)
    if x.size < 3:
        return pd.Series(np.zeros_like(x), index=series.index)

    if _HAVE_SCIPY:
        # Very small SR-like heuristic: high-pass via detrending + absolute residual
        detr = signal.detrend(x)
        res = np.abs(detr)
        # normalize
        res = (res - res.min()) / (res.max() - res.min() + 1e-12)
        return pd.Series(res, index=series.index)
    else:
        # fallback: second-order difference magnitude
        d2 = np.abs(np.diff(x, n=2))
        # pad to original length
        pad = np.zeros(x.size)
        if d2.size > 0:
            pad[2:] = d2
        # normalize
        pad = (pad - pad.min()) / (pad.max() - pad.min() + 1e-12)
        return pd.Series(pad, index=series.index)


def isolation_forest_score(series: pd.Series, n_estimators=100, contamination=0.01) -> pd.Series:
    """
    Return anomaly score 0..1 (1 = most anomalous) per sample.
    If sklearn not available, fallback to using deviation from rolling median.
    """
    x = np.asarray(series.fillna(method="ffill").fillna(0.0), dtype=float).reshape(-1, 1)
    if x.shape[0] == 0:
        return pd.Series([], dtype=float)

    if _HAVE_SKLEARN:
        try:
            iso = IsolationForest(n_estimators=n_estimators, contamination=contamination, random_state=42)
            iso.fit(x)
            raw_scores = -iso.decision_function(x)  # higher -> more anomalous
            # normalize to 0-1
            scaled = (raw_scores - raw_scores.min()) / (raw_scores.max() - raw_scores.min() + 1e-12)
            return pd.Series(scaled, index=series.index)
        except Exception as e:
            logger.warning("IsolationForest failed: %s. Falling back to median-deviation.", e)

    # fallback: rolling median absolute deviation
    med = pd.Series(x.flatten()).rolling(window=50, min_periods=1).median().values
    mad = np.abs(x.flatten() - med)
    scaled = (mad - mad.min()) / (mad.max() - mad.min() + 1e-12)
    return pd.Series(scaled, index=series.index)


def normalize_score(val: float, minv=0.0, maxv=1.0) -> float:
    if math.isnan(val):
        return 0.0
    return float(min(maxv, max(minv, val)))


# ---------------------------
# Module Implementation
# ---------------------------
class RegimeAnomalyModule(BaseAnalysisModule if BaseAnalysisModule is not object else object):
    """
    Regime & Anomaly detection module.
    """

    version = "2025.1"

    def __init__(self, config: Optional[Dict[str, Any]] = None, data_provider: Optional[Any] = None, metrics_client: Optional[Any] = None):
        """
        config: module configuration dict
        data_provider: must implement the async fetch methods described in module docstring.
        metrics_client: optional prometheus/metrics wrapper - this module will call metrics_client.observe(name, value) if provided.
        """
        self.config = {**DEFAULT_CONFIG, **(config or {})}
        self.data_provider = data_provider
        self.metrics = metrics_client
        self._validate_config()

    def _validate_config(self):
        # simple validation
        w = self.config.get("combine_weights", {})
        if not math.isclose(sum(w.values()), 1.0, rel_tol=1e-4):
            # normalize automatically but warn
            s = sum(w.values()) or 1.0
            for k in w:
                w[k] = w[k] / s
            self.config["combine_weights"] = w
            logger.warning("combine_weights did not sum to 1. Normalized automatically to: %s", w)

    async def _fetch_all(self, symbol: str) -> Dict[str, pd.DataFrame]:
        """
        Orchestrates required data fetches concurrently.
        Uses data_provider when available. Otherwise attempts to import a default client from utils.binance_api.
        """
        # helpers to call
        async def _safe_call(func: Callable, *args, **kwargs):
            try:
                return await func(*args, **kwargs)
            except Exception as e:
                logger.exception("Data provider call failed: %s", e)
                return None

        dp = self.data_provider
        if dp is None:
            # lazy import of project's binance client if available
            try:
                from ..utils.binance_api.futuresclient import FuturesClient as _FClient  # type: ignore
                dp = _FClient()  # default constructor
                self.data_provider = dp
            except Exception:
                logger.warning("No data_provider provided and default FuturesClient not available. Module will fail if fetch attempted.")
                dp = None

        tasks = {}
        results = {}

        if dp is None:
            raise RuntimeError("No data provider available for regime_anomal module.")

        # Kick off parallel fetch tasks (IO-bound)
        tasks["klines"] = asyncio.create_task(_safe_call(dp.get_klines, symbol, self.config["klines_interval"], self.config["klines_limit"]))
        tasks["oi_hist"] = asyncio.create_task(_safe_call(dp.get_open_interest_hist, symbol, self.config["oi_hist_period"], self.config["oi_hist_limit"]))
        tasks["funding"] = asyncio.create_task(_safe_call(dp.get_funding_rate, symbol, self.config["funding_limit"]))
        tasks["mark_klines"] = asyncio.create_task(_safe_call(dp.get_markprice_klines, symbol, self.config["klines_interval"], self.config["klines_limit"]))
        tasks["open_interest"] = asyncio.create_task(_safe_call(dp.get_open_interest, symbol))

        # wait for tasks
        done = await asyncio.gather(*tasks.values(), return_exceptions=True)
        for k, v in zip(tasks.keys(), done):
            results[k] = v

        return results

    async def compute_metrics(self, symbol: str) -> Dict[str, Any]:
        """
        Main entry: fetch data, compute metrics and detection, produce structured output.
        """
        raw = await self._fetch_all(symbol)

        # Validate returned frames
        klines = raw.get("klines")
        mark_klines = raw.get("mark_klines")
        oi_hist = raw.get("oi_hist")
        funding = raw.get("funding")
        open_interest = raw.get("open_interest")

        # Convert to pandas and ensure 'close' column exists
        if klines is None or (not hasattr(klines, "close") and "close" not in getattr(klines, "columns", [])):
            raise ValueError(f"klines data missing or malformed for {symbol}")

        # For safety convert to DataFrame if list-like
        if not isinstance(klines, pd.DataFrame):
            klines = pd.DataFrame(klines)

        close = pd.to_numeric(klines["close"].astype(float), errors="coerce").fillna(method="ffill").fillna(0.0)

        window = int(self.config["rolling_window"])
        min_points = int(self.config["min_points"])
        if close.size < min_points:
            logger.warning("Not enough data points (%d) for robust analysis, min_points=%d", close.size, min_points)

        # Compute features
        zscore_series = rolling_zscore(close, window)
        cumret_dev_series = cumulative_return_deviation(close, window)

        # Use mark price or oi_hist as complementary series for change detection
        # Prefer open interest changes if available
        oi_series = None
        if isinstance(oi_hist, pd.DataFrame):
            # assume oi_hist has 'openInterest' or similar
            if "openInterest" in oi_hist.columns:
                oi_series = pd.to_numeric(oi_hist["openInterest"].astype(float), errors="coerce").fillna(method="ffill").fillna(0.0)
            elif "value" in oi_hist.columns:
                oi_series = pd.to_numeric(oi_hist["value"].astype(float), errors="coerce").fillna(method="ffill").fillna(0.0)

        # fallback to mark price klines close if available
        mark_series = None
        if isinstance(mark_klines, pd.DataFrame) and "close" in mark_klines.columns:
            mark_series = pd.to_numeric(mark_klines["close"].astype(float), errors="coerce").fillna(method="ffill").fillna(0.0)

        # Prepare series for CUSUM: price returns + oi changes combined if possible
        price_ret = np.log(close).diff().fillna(0.0)
        combined_series = price_ret.copy()
        if oi_series is not None and len(oi_series) >= len(price_ret):
            # align to the price_ret index length: take last len(price_ret)
            oi_aligned = oi_series.iloc[-len(price_ret):].astype(float)
            # scale oi change to comparable range (zscore)
            oi_change = pd.Series(oi_aligned).pct_change().fillna(0.0).values
            # combine by simple addition with small weight
            combined_series = price_ret + pd.Series(oi_change, index=price_ret.index) * 0.1
        elif mark_series is not None:
            # incorporate mark price returns
            mark_ret = np.log(mark_series).diff().fillna(0.0)
            if len(mark_ret) >= len(price_ret):
                mr = mark_ret.iloc[-len(price_ret):].values
                combined_series = price_ret + pd.Series(mr, index=price_ret.index) * 0.2

        # CUSUM
        cusum_res = simple_cusum(combined_series.values, h=float(self.config["cusum_h"]), k=float(self.config["cusum_k"]))
        cusum_score = normalize_score(cusum_res.get("cusum_score", 0.0))

        # Isolation forest score on price close
        iso_series = isolation_forest_score(close, n_estimators=int(self.config["iso_n_estimators"]), contamination=float(self.config["iso_contamination"]))
        # aggregated iso score: max of last window
        iso_score_val = float(iso_series.rolling(window=window, min_periods=1).max().iloc[-1]) if not iso_series.empty else 0.0
        iso_score_val = normalize_score(iso_score_val)

        # zscore aggregate: max recent zscore magnitude
        zscore_val = float(zscore_series.rolling(window=window, min_periods=1).max().iloc[-1]) if not zscore_series.empty else 0.0
        # map zscore magnitude to 0..1 using threshold
        z_thr = float(self.config["zscore_threshold"])
        zscore_norm = normalize_score(min(1.0, zscore_val / (z_thr + 1e-12)))

        # cumulative return deviation aggregated
        cumret_val = float(cumret_dev_series.rolling(window=window, min_periods=1).max().iloc[-1]) if not cumret_dev_series.empty else 0.0
        cumret_norm = normalize_score(min(1.0, cumret_val))  # already scaled somewhat

        # spectral residual signal (unused in ensemble but included in explain)
        spectral_series = spectral_residual_score(close)
        spectral_val = float(spectral_series.rolling(window=window, min_periods=1).max().iloc[-1]) if not spectral_series.empty else 0.0

        # Combine scores by configured weights
        weights = self.config.get("combine_weights", {})
        combined = (
            cusum_score * weights.get("cusum", 0.0)
            + iso_score_val * weights.get("iso", 0.0)
            + zscore_norm * weights.get("zscore", 0.0)
            + cumret_norm * weights.get("cumret", 0.0)
        )
        combined = normalize_score(combined)

        # Derive signal label
        signal_label = "neutral"
        if combined >= 0.75:
            # strong anomaly/regime change - decide based on which component dominates
            dominant = max(
                ("cusum", cusum_score),
                ("iso", iso_score_val),
                ("zscore", zscore_norm),
                ("cumret", cumret_norm),
                key=lambda x: x[1],
            )
            if dominant[0] in ("cusum", "cumret"):
                signal_label = "regime_change"
            else:
                signal_label = "anomaly"
        elif combined >= 0.45:
            signal_label = "anomaly"

        # Build explain and components
        components = {
            "cusum": round(float(cusum_score), 6),
            "iso_forest": round(float(iso_score_val), 6),
            "zscore": round(float(zscore_norm), 6),
            "cumret_dev": round(float(cumret_norm), 6),
            "spectral": round(float(spectral_val), 6),
        }

        explain = {
            "window": window,
            "zscore_threshold": z_thr,
            "cusum_hits": {"pos": cusum_res.get("pos", []), "neg": cusum_res.get("neg", [])},
            "iso_summary": {
                "last_max_window": float(iso_score_val),
                "n_points": int(len(iso_series)) if hasattr(iso_series, "__len__") else 0,
                "method": "IsolationForest" if _HAVE_SKLEARN else "median-deviation-fallback",
            },
            "zscore_recent": float(zscore_val),
            "cumret_recent": float(cumret_val),
            "spectral_recent": float(spectral_val),
        }

        output = {
            "score": float(round(combined, 6)),
            "signal": signal_label,
            "components": components,
            "explain": explain,
            "raw": {
                "close_last": float(close.iloc[-1]) if len(close) > 0 else None,
                "open_interest": float(open_interest) if open_interest is not None else None,
            },
        }

        # metrics observe if client provided (non-blocking)
        try:
            if self.metrics is not None:
                # metrics client may expose observe(name, value) or gauge/set
                try:
                    self.metrics.observe("regime_anomaly.score", output["score"])
                except Exception:
                    try:
                        self.metrics.set("regime_anomaly.score", output["score"])
                    except Exception:
                        logger.debug("metrics client has no observe/set interface")
        except Exception:
            logger.exception("Metrics observation failed")

        return output

    async def generate_report(self, symbol: str) -> Dict[str, Any]:
        """
        Report wrapper that returns a human-friendly summary plus the structured metrics.
        """
        metrics = await self.compute_metrics(symbol)
        score = metrics["score"]
        signal = metrics["signal"]
        friendly = {
            "title": f"Regime/Anomaly Report - {symbol}",
            "summary": "",
            "score": score,
            "signal": signal,
            "short_explain": "",
        }
        if score >= 0.75:
            friendly["summary"] = f"High-confidence {signal} detected for {symbol} (score={score:.3f})."
        elif score >= 0.45:
            friendly["summary"] = f"Medium confidence anomaly signals for {symbol} (score={score:.3f})."
        else:
            friendly["summary"] = f"No strong anomaly/regime change detected for {symbol} (score={score:.3f})."

        friendly["short_explain"] = f"Top components: " + ", ".join(
            f"{k}={v:.3f}" for k, v in sorted(metrics["components"].items(), key=lambda x: -x[1])[:3]
        )

        return {"report": friendly, "metrics": metrics}

    # Backwards compatible run function (some aggregator calls run(symbol, priority))
    async def run(self, symbol: str, priority: Optional[int] = None) -> Dict[str, Any]:
        """
        Backward-compatible entry. Returns generate_report output.
        """
        return await self.generate_report(symbol)


# Expose top-level helper run() for modules that are loaded dynamically (legacy)
async def run(symbol: str, priority: Optional[int] = None, config: Optional[Dict[str, Any]] = None, data_provider: Optional[Any] = None) -> Dict[str, Any]:
    """
    Convenience function expected by some aggregators: run(symbol, priority)
    """
    module = RegimeAnomalyModule(config=config, data_provider=data_provider)
    return await module.run(symbol, priority)


# If executed as script for quick local test (not hitting real Binance), provide a small smoke test.
if __name__ == "__main__":
    import asyncio

    async def _smoke():
        # Create dummy data provider that returns synthetic series for quick debugging
        class DummyDP:
            async def get_klines(self, symbol, interval, limit):
                # create synthetic close price walking series with occasional jumps
                idx = pd.date_range(end=pd.Timestamp.utcnow(), periods=limit, freq="H")
                base = np.cumsum(np.random.randn(limit) * 0.01) + 100.0
                # add a jump
                if limit > 200:
                    base[-150:] += np.linspace(0, 5.0, 150)
                df = pd.DataFrame({"close": base}, index=idx)
                return df

            async def get_open_interest_hist(self, symbol, period, limit):
                idx = pd.date_range(end=pd.Timestamp.utcnow(), periods=limit, freq="H")
                oi = np.abs(1000 + np.cumsum(np.random.randn(limit) * 10))
                return pd.DataFrame({"openInterest": oi}, index=idx)

            async def get_funding_rate(self, symbol, limit):
                idx = pd.date_range(end=pd.Timestamp.utcnow(), periods=limit, freq="H")
                fr = np.random.randn(limit) * 0.0001
                return pd.DataFrame({"fundingRate": fr}, index=idx)

            async def get_markprice_klines(self, symbol, interval, limit):
                return await self.get_klines(symbol, interval, limit)

            async def get_open_interest(self, symbol):
                return float(1000.0 + np.random.randn() * 10)

        dp = DummyDP()
        mod = RegimeAnomalyModule(data_provider=dp)
        out = await mod.run("BTCUSDT")
        import json, pprint
        pprint.pprint(out)

    asyncio.run(_smoke())
