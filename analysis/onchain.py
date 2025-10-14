# analysis/onchain.py
"""
On-Chain & Macro Analysis Module
- File: onchain.py
- Config: analysis/config/c_onchain.py
- Purpose: Zincir üstü likidite & makro eğilim hesapları -> Macro Score (0-1)
- Exposes: OnChainModule class (BaseAnalysisModule-compatible) and run(symbol, priority) function
"""

from __future__ import annotations
import asyncio
import logging
from typing import Dict, Any, Optional, List, Union
import time

import numpy as np
import pandas as pd

# Local imports (projede var olduğu varsayılır)
from analysis.config.c_onchain import CONFIG
from .analysis_base_module import BaseAnalysisModule  # base class in repo
from utils.data_sources.data_provider import DataProvider

_LOG = logging.getLogger(__name__)

# Simple circuit breaker for external calls resilience
class CircuitBreaker:
    def __init__(self, failure_threshold: int = 3, recovery_time: int = 60):
        self.failure_threshold = failure_threshold
        self.recovery_time = recovery_time
        self.failures = 0
        self.last_failure_ts = 0

    def record_failure(self):
        self.failures += 1
        self.last_failure_ts = time.time()

    def record_success(self):
        self.failures = 0
        self.last_failure_ts = 0

    def allow(self) -> bool:
        if self.failures < self.failure_threshold:
            return True
        # if enough time elapsed, allow retry
        if time.time() - self.last_failure_ts > self.recovery_time:
            self.failures = 0
            return True
        return False

# Utility normalization functions
def zscore_clip(series: pd.Series, clip=3.0) -> pd.Series:
    if series.isnull().all():
        return series
    s = (series - series.mean()) / (series.std(ddof=0) if series.std(ddof=0) != 0 else 1.0)
    s = s.clip(-clip, clip)
    # map to 0-1
    s = (s - s.min()) / (s.max() - s.min()) if (s.max() - s.min()) != 0 else (s * 0) + 0.5
    return s

def minmax_norm(series: pd.Series) -> pd.Series:
    if series.isnull().all():
        return series
    mn, mx = series.min(), series.max()
    if mx - mn == 0:
        return pd.Series(0.5, index=series.index)
    return (series - mn) / (mx - mn)

def normalize(series: pd.Series, method: str = "zscore_clip", clip: float = 3.0) -> pd.Series:
    if method == "minmax":
        return minmax_norm(series)
    return zscore_clip(series, clip=clip)

def safe_last(series: pd.Series, default: float = 0.0) -> float:
    try:
        v = series.dropna()
        return float(v.iloc[-1]) if len(v) > 0 else default
    except Exception:
        return default

class OnChainModule(BaseAnalysisModule):
    """
    OnChain analysis module.
    Implements:
      - async compute_metrics(symbol)
      - aggregate_output(metrics_dict)
      - generate_report(symbol)
      - run(symbol, priority)  -> backwards compat
    """

    VERSION = CONFIG.get("version", "1.0.0")

    def __init__(self, config: Optional[Dict[str, Any]] = None, data_provider: Optional[DataProvider] = None):
        self.config = config or CONFIG
        self.dp = data_provider or DataProvider()
        self.cb = CircuitBreaker()
        self.weights = self._normalize_weights(self.config.get("weights", {}))
        self.windows = self.config.get("windows", {"short_days": 7, "medium_days": 30, "long_days": 90})

    def _normalize_weights(self, w: Dict[str, float]) -> Dict[str, float]:
        total = sum(w.values()) if w else 0.0
        if total == 0:
            return {k: 1.0 / max(len(w), 1) for k in w}
        return {k: float(v) / float(total) for k, v in w.items()}

    async def _safe_fetch(self, fetch_coro, *args, **kwargs):
        # wrapper to call external data with circuit-breaker and timeout
        if not self.cb.allow():
            _LOG.warning("Circuit breaker open - skipping data fetch")
            return None
        try:
            return await asyncio.wait_for(fetch_coro(*args, **kwargs), timeout=self.config.get("data_timeout_seconds", 10))
        except Exception as e:
            _LOG.exception("Data fetch failed: %s", e)
            self.cb.record_failure()
            return None
        finally:
            # if success will be recorded by callers when data present
            pass

    # -------------------------
    # Metric fetch helpers
    # -------------------------
    async def _fetch_etf_net_flow(self, symbol: str) -> Optional[pd.Series]:
        # Farside provides ETF flows; data_provider.get_etf_flows returns time series or None
        data = await self._safe_fetch(self.dp.get_etf_flows, symbol)
        if not data:
            return None
        # Expect data to be list/dict-like time series -> convert to pandas Series
        try:
            ser = pd.Series(data).sort_index()
            self.cb.record_success()
            return ser
        except Exception:
            return None

    async def _fetch_exchange_netflow(self, symbol: str) -> Optional[pd.Series]:
        data = await self._safe_fetch(self.dp.get_exchange_netflow, symbol)
        if not data:
            return None
        try:
            ser = pd.Series(data).sort_index()
            self.cb.record_success()
            return ser
        except Exception:
            return None

    async def _fetch_metric_generic(self, source: str, metric: str, symbol: str) -> Optional[pd.Series]:
        # generic call to dp.get_metric(source, coin, metric)
        data = await self._safe_fetch(self.dp.get_metric, source, symbol, metric)
        if not data:
            return None
        try:
            ser = pd.Series(data).sort_index()
            self.cb.record_success()
            return ser
        except Exception:
            return None

    # -------------------------
    # Metric computations
    # -------------------------
    def _compute_rolling_sum(self, series: pd.Series, window_days: int) -> pd.Series:
        if series is None or len(series) == 0:
            return pd.Series(dtype=float)
        # assume series index is date-like or integer steps; compute rolling sum over last N samples
        try:
            return series.rolling(window=window_days, min_periods=1).sum()
        except Exception:
            # fallback: simple rolling via convolution
            arr = np.asarray(series.fillna(0.0))
            if len(arr) == 0:
                return pd.Series(dtype=float)
            kernel = np.ones(min(window_days, len(arr)))
            conv = np.convolve(arr, kernel, mode="same")
            return pd.Series(conv, index=series.index)

    def _score_component_from_series(self, series: pd.Series, method: str) -> float:
        # For a given series, return normalized score (0-1) using config normalization
        if series is None or series.empty:
            return 0.5  # neutral fallback
        norm = normalize(series, method=self.config.get("normalization", {}).get("method", "zscore_clip"),
                         clip=self.config.get("normalization", {}).get("clip_z", 3.0))
        return safe_last(norm, default=0.5)

    async def compute_metrics(self, symbol: str) -> Dict[str, Any]:
        """
        Main metric orchestrator for a symbol.
        Returns dict of raw series (where relevant) and scalar component scores.
        """
        out = {
            "symbol": symbol,
            "raw": {},
            "components": {},
            "version": self.VERSION
        }

        # fetch tasks in parallel
        tasks = {
            "etf_net_flow": asyncio.create_task(self._fetch_etf_net_flow(symbol)),
            "exchange_netflow": asyncio.create_task(self._fetch_exchange_netflow(symbol)),
            # additional metrics via generic endpoints (Glassnode/CryptoQuant)
            "stablecoin_flow": asyncio.create_task(self._fetch_metric_generic("glassnode", "stablecoin_netflow", symbol)),
            "realized_cap": asyncio.create_task(self._fetch_metric_generic("glassnode", "realized_cap", symbol)),
            "nupl": asyncio.create_task(self._fetch_metric_generic("cryptoquant", "nupl", symbol)),
            "net_realized_pl": asyncio.create_task(self._fetch_metric_generic("cryptoquant", "net_realized_profit_loss", symbol)),
            "exchange_whale_ratio": asyncio.create_task(self._fetch_metric_generic("cryptoquant", "exchange_whale_ratio", symbol)),
            "mvrv_zscore": asyncio.create_task(self._fetch_metric_generic("glassnode", "mvrv_zscore", symbol)),
            "sopr": asyncio.create_task(self._fetch_metric_generic("glassnode", "sopr", symbol)),
        }

        # await all tasks robustly
        results = {}
        for key, t in tasks.items():
            try:
                results[key] = await t
            except Exception as e:
                _LOG.exception("Task %s failed: %s", key, e)
                results[key] = None

        # store raw
        out["raw"].update(results)

        # compute component scores (use rolling windows where appropriate)
        # For flows we compute short and medium rolling sums then normalize
        try:
            short = self.windows.get("short_days", 7)
            medium = self.windows.get("medium_days", 30)

            # ETF net flow: more positive is bullish -> normalize
            etf_series = results.get("etf_net_flow")
            if etf_series is not None:
                etf_roll = self._compute_rolling_sum(etf_series.fillna(0), medium)
                out["components"]["etf_net_flow"] = self._score_component_from_series(etf_roll, method=self.config.get("normalization", {}).get("method"))
            else:
                out["components"]["etf_net_flow"] = 0.5

            # Stablecoin flow: inflow to exchanges -> bearish (so invert)
            sc_series = results.get("stablecoin_flow")
            if sc_series is not None:
                sc_roll = self._compute_rolling_sum(sc_series.fillna(0), medium)
                # invert so that more stablecoin inflow (to exchanges) reduces score
                sc_score = 1.0 - self._score_component_from_series(sc_roll, method=self.config.get("normalization", {}).get("method"))
                out["components"]["stablecoin_flow"] = float(np.clip(sc_score, 0.0, 1.0))
            else:
                out["components"]["stablecoin_flow"] = 0.5

            # Exchange netflow (exchange in - out): positive inflow -> bearish (invert)
            ex_series = results.get("exchange_netflow")
            if ex_series is not None:
                ex_roll = self._compute_rolling_sum(ex_series.fillna(0), medium)
                ex_score = 1.0 - self._score_component_from_series(ex_roll, method=self.config.get("normalization", {}).get("method"))
                out["components"]["exchange_netflow"] = float(np.clip(ex_score, 0.0, 1.0))
            else:
                out["components"]["exchange_netflow"] = 0.5

            # Net Realized P/L: large realized profit near cycle top -> bearish -> invert
            nrp = results.get("net_realized_pl")
            if nrp is not None:
                nrp_roll = self._compute_rolling_sum(nrp.fillna(0), medium)
                nrp_score = 1.0 - self._score_component_from_series(nrp_roll, method=self.config.get("normalization", {}).get("method"))
                out["components"]["net_realized_pl"] = float(np.clip(nrp_score, 0.0, 1.0))
            else:
                out["components"]["net_realized_pl"] = 0.5

            # Exchange whale ratio: higher whale concentration on exchanges -> bearish -> invert
            ewr = results.get("exchange_whale_ratio")
            if ewr is not None:
                ewr_score = 1.0 - self._score_component_from_series(ewr.fillna(0), method=self.config.get("normalization", {}).get("method"))
                out["components"]["exchange_whale_ratio"] = float(np.clip(ewr_score, 0.0, 1.0))
            else:
                out["components"]["exchange_whale_ratio"] = 0.5

            # MVRV Z-Score: higher -> overvalued -> bearish -> invert
            mvrv = results.get("mvrv_zscore")
            if mvrv is not None:
                mvrv_score = 1.0 - self._score_component_from_series(mvrv.fillna(0), method=self.config.get("normalization", {}).get("method"))
                out["components"]["mvrv_zscore"] = float(np.clip(mvrv_score, 0.0, 1.0))
            else:
                out["components"]["mvrv_zscore"] = 0.5

            # NUPL: high positive -> euphoria -> bearish -> invert slightly
            nupl = results.get("nupl")
            if nupl is not None:
                nupl_score = 1.0 - self._score_component_from_series(nupl.fillna(0), method=self.config.get("normalization", {}).get("method"))
                out["components"]["nupl"] = float(np.clip(nupl_score, 0.0, 1.0))
            else:
                out["components"]["nupl"] = 0.5

            # SOPR: >1 profit being realized -> bearish -> invert
            sopr = results.get("sopr")
            if sopr is not None:
                sopr_score = 1.0 - self._score_component_from_series(sopr.fillna(0), method=self.config.get("normalization", {}).get("method"))
                out["components"]["sopr"] = float(np.clip(sopr_score, 0.0, 1.0))
            else:
                out["components"]["sopr"] = 0.5

        except Exception as e:
            _LOG.exception("Error computing components: %s", e)

        return out

    def aggregate_output(self, metrics: Dict[str, Any]) -> Dict[str, Any]:
        """
        Combine component scores into final Macro Score (0-1), signal, and explain.
        """
        comps: Dict[str, float] = metrics.get("components", {})
        # Ensure we use the configured weight keys (some keys may be missing)
        weighted_sum = 0.0
        total_weight = 0.0
        for k, w in self.weights.items():
            comp_val = float(comps.get(k, 0.5))
            weighted_sum += comp_val * float(w)
            total_weight += float(w)
        score = weighted_sum / total_weight if total_weight > 0 else 0.5
        # Determine signal
        th = self.config.get("thresholds", {"bullish": 0.65, "bearish": 0.35})
        if score >= th.get("bullish", 0.65):
            signal = "bullish"
        elif score <= th.get("bearish", 0.35):
            signal = "bearish"
        else:
            signal = "neutral"

        # Prepare explain field - top contributing components
        components_sorted = sorted(self.weights.keys(), key=lambda x: -self.weights.get(x, 0))
        explain_list = []
        for k in components_sorted[: self.config.get("explain_components_limit", 5)]:
            explain_list.append({
                "name": k,
                "weight": self.weights.get(k, 0.0),
                "value": float(metrics.get("components", {}).get(k, 0.5))
            })

        return {
            "symbol": metrics.get("symbol"),
            "macro_score": float(np.round(score, 4)),
            "signal": signal,
            "components": metrics.get("components", {}),
            "explain": explain_list,
            "version": self.VERSION,
            "raw": metrics.get("raw", {})
        }

    async def generate_report(self, symbol: str) -> Dict[str, Any]:
        """
        Convenience method: compute_metrics + aggregate_output -> full report
        """
        metrics = await self.compute_metrics(symbol)
        report = self.aggregate_output(metrics)
        return report

    # backward-compatible run
    async def run(self, symbol: Union[str, List[str]], priority: Optional[int] = None) -> Dict[str, Any]:
        """
        Backward-compatible entrypoint. Accepts single symbol or list.
        Returns a mapping symbol -> report
        """
        if isinstance(symbol, str):
            rep = await self.generate_report(symbol)
            return {symbol: rep}
        elif isinstance(symbol, list):
            # run tasks in parallel
            tasks = {s: asyncio.create_task(self.generate_report(s)) for s in symbol}
            out = {}
            for s, t in tasks.items():
                try:
                    out[s] = await t
                except Exception as e:
                    _LOG.exception("Report generation failed for %s: %s", s, e)
                    out[s] = {"error": str(e)}
            return out
        else:
            raise ValueError("symbol must be str or list of str")

# module-level helper for backwards compatibility if called as function
async def run(symbol: Union[str, List[str]], priority: Optional[int] = None, config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    mod = OnChainModule(config=config)
    return await mod.run(symbol, priority=priority)
