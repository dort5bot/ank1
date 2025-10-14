# analysis/risk_expos.py

"""
Risk & Exposure Management Module

Exports:
    - class RiskExposureModule(BaseAnalysisModule)
    - run(symbols, priority) function for backward compatibility

Purpose:
    Calculate risk metrics for spot + futures (public + private data if available)
    Metrics computed:
      - ATR-based adaptive stop
      - Liquidation zone estimation
      - Max Drawdown
      - Volatility targeting (position sizing guidance)
      - Position Leverage Ratio
      - VaR (historical simulation) at configurable confidence
      - Expected Shortfall / CVaR
      - Dynamic Sharpe & Sortino
      - Final normalized Risk Score (0..1) with components & explainability

Design notes:
    - The module expects to be given a data provider with methods for fetching:
        - get_klines(symbol, interval, limit) -> DataFrame-like (ts,indexed)
        - get_futures_position_risk(optional params) -> dict/list
      Your existing Binance aggregator (binance_a.py) should be adapted to match.
    - If BaseAnalysisModule cannot be imported from analysis.analysis_base_module,
      a minimal fallback base class is used (to allow standalone testing).
"""

from __future__ import annotations

import asyncio
import math
import time
from dataclasses import dataclass
from typing import Dict, Any, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

#from analysis.config.c_risk import CONFIG as CONFIG_LOCAL
# Try to import project base class; fallback to minimal base for standalone.
try:
    from analysis.analysis_base_module import BaseAnalysisModule
except Exception:
    class BaseAnalysisModule:
        """
        Minimal fallback base class so module can be tested standalone.
        The real project BaseAnalysisModule likely provides more hooks.
        """
        def __init__(self, config: Dict[str, Any]):
            self.config = config

        async def init(self):
            return None

# Pydantic-like validation (lightweight)
try:
    from pydantic import BaseModel, Field, validator
except Exception:
    BaseModel = object
    Field = lambda *a, **k: None
    def validator(*a, **k):
        def _f(f): return f
        return _f

# Optional typing convenience
ArrayLike = np.ndarray

# --- Helpers & numeric functions ---


def ensure_df_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure DataFrame has columns: ['open','high','low','close','volume'] and index is datetime.
    Accepts dict-like or DataFrame and returns DataFrame copy.
    """
    if df is None or len(df) == 0:
        return pd.DataFrame(columns=['open', 'high', 'low', 'close', 'volume'])
    df = df.copy()
    cols = [c.lower() for c in df.columns]
    mapping = {}
    if 'open' not in cols and 'Open' in df.columns:
        mapping['Open'] = 'open'
    # best-effort mapping
    candidates = {'open': ['open', 'o'], 'high': ['high', 'h'], 'low': ['low', 'l'], 'close': ['close', 'c'], 'volume': ['volume', 'v']}
    colmap = {}
    for std, names in candidates.items():
        for n in names:
            if n in df.columns:
                colmap[n] = std
                break
    df = df.rename(columns=colmap)
    # ensure required columns exist
    for c in ['open', 'high', 'low', 'close', 'volume']:
        if c not in df.columns:
            df[c] = np.nan
    # index to datetime if possible
    if not isinstance(df.index, pd.DatetimeIndex):
        if 'timestamp' in df.columns:
            df.index = pd.to_datetime(df['timestamp'], unit='ms', utc=True)
        else:
            try:
                df.index = pd.to_datetime(df.index, unit='ms', utc=True)
            except Exception:
                df.index = pd.to_datetime(df.index, utc=True, errors='coerce')
    return df[['open', 'high', 'low', 'close', 'volume']]


def true_range(df: pd.DataFrame) -> pd.Series:
    """
    Compute True Range series
    """
    high = df['high']
    low = df['low']
    prev_close = df['close'].shift(1)
    tr = pd.concat([high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1).max(axis=1)
    return tr


def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    tr = true_range(df)
    return tr.ewm(span=period, adjust=False).mean()


def max_drawdown(series: pd.Series) -> float:
    """
    Returns maximum drawdown (positive number e.g. 0.25 for 25%)
    """
    if series is None or len(series) == 0:
        return 0.0
    roll_max = series.cummax()
    drawdown = (roll_max - series) / roll_max
    return float(drawdown.max())


def historical_var(returns: np.ndarray, confidence: float = 0.95) -> float:
    """
    Historical VaR (percentile). returns is negative for losses if standard convention,
    but we assume returns as pct returns; VaR returned as positive loss magnitude.
    """
    if len(returns) == 0:
        return 0.0
    # we want the percentile of losses; sort ascending and pick (1 - confidence) quantile
    q = np.quantile(returns, 1 - confidence)
    # VaR as positive number = -q if q is negative loss
    return float(max(0.0, -q))


def historical_cvar(returns: np.ndarray, confidence: float = 0.95) -> float:
    if len(returns) == 0:
        return 0.0
    threshold = np.quantile(returns, 1 - confidence)
    tail = returns[returns <= threshold]
    if len(tail) == 0:
        return 0.0
    return float(max(0.0, -tail.mean()))


def annualize_vol(std_per_period: float, periods_per_year: int) -> float:
    return std_per_period * math.sqrt(periods_per_year)


def rolling_sharpe(returns: pd.Series, window: int, rf_per_period: float = 0.0) -> pd.Series:
    excess = returns - rf_per_period
    roll_mean = excess.rolling(window).mean()
    roll_std = excess.rolling(window).std()
    return roll_mean / (roll_std.replace(0, np.nan))


def rolling_sortino(returns: pd.Series, window: int, rf_per_period: float = 0.0) -> pd.Series:
    # downside deviation uses negative returns only
    excess = returns - rf_per_period
    def sortino_s(arr):
        arr = np.asarray(arr)
        if len(arr) == 0:
            return np.nan
        downside = arr[arr < 0]
        if len(downside) == 0:
            return np.nan
        mean = arr.mean()
        dd = np.sqrt((downside ** 2).mean())
        return (mean) / dd if dd > 0 else np.nan
    return returns.rolling(window).apply(lambda x: sortino_s(x - rf_per_period), raw=False)


# --- Module Implementation ---


class RiskExposureModule(BaseAnalysisModule):
    """
    Risk & Exposure Module implementing required metrics and unified output.

    Constructor:
        RiskExposureModule(config: dict, data_provider: Optional[object] = None, metrics_client: Optional[object] = None)

    - data_provider: object with methods to fetch kline/positionRisk/account data:
        - async get_klines(symbol, interval, limit) -> pd.DataFrame-like
        - async get_futures_position_risk(symbol=None) -> list/dict
        - async get_account() -> dict (optional)

    Public Methods:
        - async compute_for_symbol(symbol: str, **kwargs) -> dict
        - async run(symbols: Sequence[str], priority: int = 0) -> Dict[str, Any]  (backwards compatibility)
    """

    def __init__(self, config: Dict[str, Any], data_provider: Optional[Any] = None, metrics_client: Optional[Any] = None):
        super().__init__(config)
        self.config = config
        self.data_provider = data_provider
        self.metrics_client = metrics_client
        self.module_name = config.get("name", "risk_expos")
        self.weights = config.get("weights", {})
        self._loop = asyncio.get_event_loop()

    async def _safe_fetch_klines(self, symbol: str, interval: str, limit: int) -> pd.DataFrame:
        """
        Wrapper to call the data provider's get_klines. Graceful fallback to empty DataFrame.
        """
        if self.data_provider is None:
            return pd.DataFrame()
        # expect the provider to have async method; if not, try sync wrapper
        try:
            if asyncio.iscoroutinefunction(self.data_provider.get_klines):
                res = await self.data_provider.get_klines(symbol=symbol, interval=interval, limit=limit)
            else:
                # run in thread executor
                res = await self._loop.run_in_executor(None, lambda: self.data_provider.get_klines(symbol=symbol, interval=interval, limit=limit))
            # convert to DataFrame if needed
            if isinstance(res, pd.DataFrame):
                df = res
            else:
                df = pd.DataFrame(res)
            df = ensure_df_ohlcv(df)
            return df
        except Exception as e:
            # log if metrics client available
            if self.metrics_client:
                try:
                    self.metrics_client.increment("risk_expos.fetch_klines_error")
                except Exception:
                    pass
            return pd.DataFrame()

    async def _safe_fetch_positions(self) -> List[Dict[str, Any]]:
        """
        Fetch futures position risk or account positions if available. Return list of positions or empty list.
        """
        if self.data_provider is None:
            return []
        try:
            if hasattr(self.data_provider, "get_futures_position_risk"):
                fn = self.data_provider.get_futures_position_risk
                if asyncio.iscoroutinefunction(fn):
                    res = await fn()
                else:
                    res = await self._loop.run_in_executor(None, fn)
                # normalize to list
                if isinstance(res, dict):
                    return [res]
                return list(res or [])
            # fallback: get_account and parse positions
            if hasattr(self.data_provider, "get_account"):
                fn = self.data_provider.get_account
                if asyncio.iscoroutinefunction(fn):
                    acc = await fn()
                else:
                    acc = await self._loop.run_in_executor(None, fn)
                positions = acc.get("positions", []) if acc else []
                return positions
            return []
        except Exception:
            if self.metrics_client:
                try:
                    self.metrics_client.increment("risk_expos.fetch_positions_error")
                except Exception:
                    pass
            return []

    def _estimate_liquidation_price(self, entry_price: float, leverage: float, side: str = "LONG", maintenance: float = 0.005) -> float:
        """
        Conservative approximation of liquidation price for isolated margin:
        For long: liq_price ≈ entry_price * (1 - 1/leverage + maintenance_adjustment)
        For short: liq_price ≈ entry_price * (1 + 1/leverage - maintenance_adjustment)

        Note: real futures exchanges have precise formula that depends on margin balance, unrealized PNL, etc.
        We document this is an approximation.
        """
        if leverage is None or leverage <= 1:
            return float(entry_price)
        if side.upper() == "LONG":
            liq = entry_price * (1 - (1.0 / max(leverage, 1.0)) - maintenance)
        else:
            liq = entry_price * (1 + (1.0 / max(leverage, 1.0)) + maintenance)
        return float(max(0.0, liq))

    def _score_from_component(self, value: float, higher_is_riskier: bool = True, clip: Tuple[float, float] = (0.0, 1.0)) -> float:
        """
        Map a raw metric value into [0..1] risk score for that component.
        - higher_is_riskier indicates whether larger metric means more risk.
        - clip defines domain for linear mapping; outside is clipped.
        Default linear mapping within clip.
        """
        lo, hi = clip
        if hi == lo:
            return 0.0
        v = float(value)
        normalized = (v - lo) / (hi - lo)
        if not higher_is_riskier:
            normalized = 1.0 - normalized
        return float(max(0.0, min(1.0, normalized)))

    async def compute_for_symbol(self, symbol: str, interval: Optional[str] = None, lookback: Optional[int] = None) -> Dict[str, Any]:
        """
        Compute risk metrics for a single symbol.

        Returns a dictionary:
        {
            "symbol": symbol,
            "score": 0..1,
            "signal": "low|medium|high",
            "components": { "var": x, "cvar": y, ... },
            "explain": {... detailed components and intermediate stats ...}
        }
        """
        cfg = self.config
        interval = interval or cfg.get("ohlcv", {}).get("interval", "1h")
        lookback = lookback or cfg.get("ohlcv", {}).get("lookback_bars", 500)

        # fetch historical price data
        df = await self._safe_fetch_klines(symbol=symbol, interval=interval, limit=lookback)
        if df is None or df.empty:
            # graceful return with neutral risk if data missing
            return {
                "symbol": symbol,
                "score": 0.5,
                "signal": "unknown",
                "components": {},
                "explain": {"message": "no price data available; returned neutral score"}
            }

        # compute returns (periodic pct returns)
        close = df['close'].astype(float)
        period_returns = close.pct_change().dropna()
        # basic stats
        periods = len(period_returns)
        # assume interval is hourly for annualization if unknown: We'll infer periods_per_year from interval string
        # crude mapping: 1m,5m,1h,4h,1d
        if 'm' in interval and interval.endswith('m'):
            minutes = int(interval[:-1])
            periods_per_year = int((60 / minutes) * 24 * 365)
        elif 'h' in interval:
            hours = int(interval[:-1])
            periods_per_year = int((24 / hours) * 365)
        elif 'd' in interval:
            days = int(interval[:-1])
            periods_per_year = int(365 / days)
        else:
            periods_per_year = 24 * 365  # fallback assume hourly

        # volatility (std)
        std_per_period = float(period_returns.std(ddof=1) or 0.0)
        ann_vol = annualize_vol(std_per_period, periods_per_year)

        # Max drawdown (on close series)
        md = max_drawdown(close)

        # ATR stop
        atr_period = cfg.get("atr", {}).get("period", 21)
        atr_multiplier = cfg.get("atr", {}).get("multiplier", 3.0)
        atr_series = atr(df, period=atr_period)
        latest_atr = float(atr_series.dropna().iloc[-1]) if len(atr_series.dropna()) > 0 else 0.0
        latest_price = float(close.iloc[-1])
        atr_stop = max(0.0, latest_price - atr_multiplier * latest_atr)

        # VaR / CVaR (using returns)
        var_cfg = cfg.get("var", {})
        var_conf_levels = var_cfg.get("confidence_levels", [0.95, 0.99])
        var_results = {}
        cvar_results = {}
        for c in var_conf_levels:
            v = historical_var(period_returns.values, confidence=c)
            cv = historical_cvar(period_returns.values, confidence=c)
            var_results[f"VaR_{int(c*100)}"] = v
            cvar_results[f"CVaR_{int(c*100)}"] = cv

        # Volatility targeting: compute suggested leverage adj factor = target_vol / ann_vol
        vt_cfg = cfg.get("vol_target", {})
        target_vol = vt_cfg.get("target_volatility", 0.12)
        vol_targeting_factor = 1.0
        if ann_vol > 0:
            vol_targeting_factor = float(min(3.0, target_vol / ann_vol))  # cap leverage change suggestion
        else:
            vol_targeting_factor = 1.0

        # positions & leverage
        positions = await self._safe_fetch_positions()
        # compute aggregated leverage ratio across positions if available
        leverage_ratios = []
        liquidation_infos = []
        maintenance_default = cfg.get("leverage", {}).get("default_maintenance_margin", 0.005)
        for pos in positions:
            try:
                # try several possible keys found in Binance responses
                entry_price = float(pos.get("entryPrice", pos.get("entry_price", pos.get("price", latest_price))))
                leverage = float(pos.get("leverage", pos.get("leverage", 1))) if pos.get("leverage", None) is not None else float(pos.get("positionAmt", 0)) and 1.0
                side = "LONG" if float(pos.get("positionAmt", 0)) > 0 else "SHORT"
                liq = pos.get("liquidationPrice") or pos.get("liquidation_price") or None
                if liq is None:
                    liq = self._estimate_liquidation_price(entry_price=entry_price, leverage=leverage, side=side, maintenance=maintenance_default)
                liquidation_infos.append({
                    "symbol": pos.get("symbol", symbol),
                    "entry_price": entry_price,
                    "leverage": leverage,
                    "side": side,
                    "estimated_liq_price": float(liq)
                })
                leverage_ratios.append(float(leverage))
            except Exception:
                continue

        avg_leverage = float(np.mean(leverage_ratios)) if len(leverage_ratios) > 0 else 1.0
        max_leverage = float(np.max(leverage_ratios)) if len(leverage_ratios) > 0 else 1.0

        # dynamic Sharpe/Sortino
        perf_cfg = cfg.get("performance", {})
        rolling_window = perf_cfg.get("rolling_window", 63)
        rf = perf_cfg.get("risk_free_rate", 0.0)
        sharpe_series = rolling_sharpe(period_returns, window=rolling_window, rf_per_period=rf)
        sortino_series = rolling_sortino(period_returns, window=rolling_window, rf_per_period=rf)
        latest_sharpe = float(sharpe_series.dropna().iloc[-1]) if len(sharpe_series.dropna()) > 0 else 0.0
        latest_sortino = float(sortino_series.dropna().iloc[-1]) if len(sortino_series.dropna()) > 0 else 0.0

        # Compose component scores in 0..1 risk (higher = riskier)
        # Define heuristics and clip ranges for mapping
        # VaR_95: map 0..0.10 -> 0..1 (10% loss as extreme)
        v95 = var_results.get("VaR_95", 0.0)
        cv95 = cvar_results.get("CVaR_95", 0.0)
        weights = self.weights or cfg.get("weights", {})

        comp_var = self._score_from_component(v95, higher_is_riskier=True, clip=(0.0, 0.10))
        comp_cvar = self._score_from_component(cv95, higher_is_riskier=True, clip=(0.0, 0.15))
        # leverage: map 1..max_leverage to 0..1 (cap at a configured reasonable max)
        cap_lev = cfg.get("leverage", {}).get("max_leverage", 125)
        comp_leverage = self._score_from_component(avg_leverage, higher_is_riskier=True, clip=(1.0, min(cap_lev, 50.0)))
        # vol_targeting: if vol_targeting_factor < 1 => portfolio is too volatile (needs de-risk)
        # we interpret vol_targeting_factor < 1 => riskier (need reduce exposure), map range 0..1.5
        comp_vol_target = self._score_from_component(1.0 / (vol_targeting_factor + 1e-9), higher_is_riskier=True, clip=(0.0, 2.0))
        # max drawdown: map 0..0.5
        comp_mdd = self._score_from_component(md, higher_is_riskier=True, clip=(0.0, 0.5))
        # atr_stop proximity: distance between current price and atr stop relative to price -> small distance = riskier
        if latest_price > 0:
            atr_dist = (latest_price - atr_stop) / latest_price
        else:
            atr_dist = 1.0
        # smaller distance is riskier -> invert: higher component for smaller distance
        comp_atr = self._score_from_component(atr_dist, higher_is_riskier=False, clip=(0.0, 0.5))

        # weighted sum -> raw risk score in 0..1
        # ensure weights sum ~1; if not, normalize
        total_w = sum(weights.values()) if isinstance(weights, dict) and len(weights) > 0 else None
        if not total_w or abs(total_w - 1.0) > 1e-6:
            # normalize default or provided
            if isinstance(weights, dict) and len(weights) > 0:
                total = sum(weights.values())
                weights = {k: float(v / total) for k, v in weights.items()}
            else:
                # fallback equal weights for the 6 comps
                weights = {
                    "var": 0.30,
                    "cvar": 0.25,
                    "leverage": 0.15,
                    "vol_targeting": 0.10,
                    "max_drawdown": 0.10,
                    "atr_stop": 0.10
                }

        # compute final weighted risk score
        component_scores = {
            "var": comp_var,
            "cvar": comp_cvar,
            "leverage": comp_leverage,
            "vol_targeting": comp_vol_target,
            "max_drawdown": comp_mdd,
            "atr_stop": comp_atr
        }
        score = 0.0
        for k, v in component_scores.items():
            w = float(weights.get(k, 0.0))
            score += w * v
        score = float(max(0.0, min(1.0, score)))

        # signal interpretation
        thresholds = cfg.get("thresholds", {})
        t_high = thresholds.get("high_risk", 0.75)
        t_med = thresholds.get("medium_risk", 0.45)
        if score >= t_high:
            signal = "high"
        elif score >= t_med:
            signal = "medium"
        else:
            signal = "low"

        explain = {
            "latest_price": latest_price,
            "ann_vol": ann_vol,
            "periods_analyzed": periods,
            "atr": latest_atr,
            "atr_stop": atr_stop,
            "max_drawdown": md,
            "var": var_results,
            "cvar": cvar_results,
            "vol_targeting_factor": vol_targeting_factor,
            "avg_leverage": avg_leverage,
            "max_leverage": max_leverage,
            "positions_count": len(positions),
            "liquidation_info": liquidation_infos,
            "sharpe": latest_sharpe,
            "sortino": latest_sortino,
            "component_raw_scores": component_scores,
            "weights_used": weights
        }

        output = {
            "symbol": symbol,
            "score": score,
            "signal": signal,
            "components": component_scores,
            "explain": explain
        }
        return output

    async def run(self, symbols: Sequence[str], priority: int = 0) -> Dict[str, Any]:
        """
        Backwards compatible run method: compute concurrently for a list of symbols and return mapping.
        """
        if isinstance(symbols, str):
            symbols = [symbols]
        tasks = [self.compute_for_symbol(s) for s in symbols]
        results = await asyncio.gather(*tasks, return_exceptions=False)
        return {r["symbol"]: r for r in results}

# Backwards-compatible function
async def run(symbols: Sequence[str], priority: int = 0, config: Optional[Dict[str, Any]] = None, data_provider: Optional[Any] = None):
    """
    Convenience top-level run when module used standalone:
        await run(["BTCUSDT","ETHUSDT"], priority=0, config=CONFIG, data_provider=my_provider)
    """
    if config is None:
        # try to import config
        try:
            from analysis.config.c_risk import CONFIG as CONFIG_LOCAL
            config = CONFIG_LOCAL
        except Exception:
            config = {
                "name": "risk_expos",
                "weights": {
                    "var": 0.30,
                    "cvar": 0.25,
                    "leverage": 0.15,
                    "vol_targeting": 0.10,
                    "max_drawdown": 0.10,
                    "atr_stop": 0.10
                },
                "thresholds": {"high_risk": 0.75, "medium_risk": 0.45},
                "ohlcv": {"interval": "1h", "lookback_bars": 500},
                "atr": {"period": 21, "multiplier": 3.0},
                "var": {"confidence_levels": [0.95, 0.99]},
                "leverage": {"default_maintenance_margin": 0.005, "max_leverage": 125},
                "vol_target": {"target_volatility": 0.12}
            }

    mod = RiskExposureModule(config=config, data_provider=data_provider)
    return await mod.run(symbols, priority=priority)
