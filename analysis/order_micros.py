"""
Order Flow & Microstructure module
File: order_micros.py
Config file: c_order.py

Purpose:
 - Compute real-time and batch microstructure metrics from orderbook + trades + bookTicker.
 - Produce a normalized Liquidity Pressure Score (0-1) and explainable components.

Design notes:
 - Class-based, inherits a light BaseAnalysisModule interface assumed by the analysis framework.
 - Async-friendly and batch-capable (methods support both real-time stream input and batch snapshots).
 - Vectorized computations with numpy/pandas.
 - Config-driven weights/thresholds in c_order.py.
 - Includes fallback/mock data provider interface; in production inject your Binance aggregator / websocket manager.
 - Exposes run(symbol, priority) for backward compatibility.
 - Output schema:
    {
      "score": float (0-1),
      "signal": str ("buy_pressure"|"sell_pressure"|"neutral"),
      "components": {metric: normalized_value, ...},
      "explain": {metric: {"raw": x, "normalized": y, "weight": w, "contribution": c}, ...},
      "meta": {"symbol": symbol, "timestamp": iso8601}
    }
"""

from __future__ import annotations
import asyncio
import math
import time
from dataclasses import dataclass
from typing import Dict, Any, Optional, Tuple, List

import numpy as np
import pandas as pd

# If your project provides BaseAnalysisModule, import it. Otherwise, we emulate a minimal interface.
try:
    from analysis.analysis_base_module import BaseAnalysisModule
except Exception:
    class BaseAnalysisModule:
        """Minimal fallback stub -- production should use real BaseAnalysisModule."""
        def __init__(self, config: Dict[str, Any]):
            self.config = config

# Import config
from .config.c_order import CONFIG  # adjust path depending on project layout


# ---- Data provider interface -------------------------------------------------
class DataProviderInterface:
    """
    Provide snapshot and small-history data to the module.
    In production, implement using utils.binance_api.binance_a or WebSocket stream handlers.

    Required methods:
      async def get_order_book(self, symbol) -> dict  # bids/asks top N: [{'price':p,'qty':q}, ...]
      async def get_recent_trades(self, symbol, limit=500) -> List[dict]  # [{price,qty,isBuyerMaker,timestamp}, ...]
      async def get_book_ticker(self, symbol) -> dict  # bestBid, bestAsk
    """
    async def get_order_book(self, symbol: str) -> dict:
        raise NotImplementedError

    async def get_recent_trades(self, symbol: str, limit: int = 500) -> List[dict]:
        raise NotImplementedError

    async def get_book_ticker(self, symbol: str) -> dict:
        raise NotImplementedError


# ---- Simple Mock Provider for testing/demo ----------------------------------
class MockDataProvider(DataProviderInterface):
    """Generate synthetic orderbook/trade snapshots for local testing."""

    def __init__(self, mid_price: float = 100.0, depth_levels: int = 20):
        self.mid = mid_price
        self.depth = depth_levels
        np.random.seed(42)

    async def get_order_book(self, symbol: str) -> dict:
        # create symmetric orderbook around mid price with random quantities
        bids = []
        asks = []
        step = 0.01 * self.mid
        for i in range(self.depth):
            price_bid = round(self.mid - (i + 0.5) * step, 8)
            price_ask = round(self.mid + (i + 0.5) * step, 8)
            bids.append({"price": price_bid, "qty": float(np.abs(np.random.exponential(scale=2.0)))})
            asks.append({"price": price_ask, "qty": float(np.abs(np.random.exponential(scale=2.0)))})
        return {"bids": bids, "asks": asks, "timestamp": int(time.time() * 1000)}

    async def get_recent_trades(self, symbol: str, limit: int = 200) -> List[dict]:
        # generate trades around mid price with buyer/seller flags
        trades = []
        for i in range(limit):
            price = float(self.mid + np.random.normal(scale=0.02 * self.mid))
            qty = float(abs(np.random.exponential(scale=1.0)))
            isBuyerMaker = bool(np.random.rand() > 0.5)  # random aggression
            trades.append({"price": price, "qty": qty, "isBuyerMaker": isBuyerMaker, "timestamp": int(time.time() * 1000) - i * 100})
        # most recent first
        trades = sorted(trades, key=lambda x: x["timestamp"], reverse=False)
        return trades

    async def get_book_ticker(self, symbol: str) -> dict:
        # return top-of-book
        ob = await self.get_order_book(symbol)
        return {"bidPrice": ob["bids"][0]["price"], "askPrice": ob["asks"][0]["price"], "timestamp": ob["timestamp"]}


# ---- Utility functions ------------------------------------------------------
def _safe_div(a, b):
    return a / b if (b is not None and b != 0) else 0.0


def _min_max_scale(x: float, lo: float, hi: float):
    if math.isfinite(x):
        if hi == lo:
            return 0.0
        return float(max(0.0, min(1.0, (x - lo) / (hi - lo))))
    return 0.0


# ---- Core Module ------------------------------------------------------------
class OrderMicroModule(BaseAnalysisModule):
    """
    Order Flow & Microstructure analysis module.

    Core public methods:
      - compute_metrics_from_snapshot(order_book, trades, book_ticker)
      - aggregate_output(component_scores) -> final score and explain
      - run(symbol, priority) -> backward-compatible entry point (async)
    """

    VERSION = "1.0.0"

    def __init__(self, config: Dict[str, Any], data_provider: Optional[DataProviderInterface] = None):
        super().__init__(config)
        self.cfg = config
        self.data_provider = data_provider or MockDataProvider(mid_price=config.get("mid_price", 100.0))
        # define normalization windows / defaults used for MinMax scaling of raw metrics
        self.norm = self.cfg.get("normalization", {
            "orderbook_imbalance": (-1.0, 1.0),
            "spread_bps": (0.0, 50.0),
            "market_pressure": (-1.0, 1.0),
            "trade_aggression": (0.0, 1.5),
            "slippage": (0.0, 0.5),
            "depth_elasticity": (0.0, 10.0),
            "cvd": (-1e6, 1e6),
            "ofi": (-1e6, 1e6),
            "taker_dom": (0.0, 1.0),
            "liquidity_density": (0.0, 1e6),
        })

    async def compute_metrics_from_snapshot(self, order_book: dict, trades: List[dict], book_ticker: dict) -> Dict[str, Any]:
        """
        Compute raw microstructure metrics from a single snapshot of order book + trades + ticker.

        Returns:
          dict of raw metric values (not normalized).
        """

        # Convert order book to DataFrame
        bids = pd.DataFrame(order_book.get("bids", []))
        asks = pd.DataFrame(order_book.get("asks", []))

        # Ensure sorted
        bids = bids.sort_values("price", ascending=False).reset_index(drop=True)
        asks = asks.sort_values("price", ascending=True).reset_index(drop=True)

        # Top-of-book
        best_bid = float(bids.loc[0, "price"]) if not bids.empty else float(book_ticker.get("bidPrice", np.nan))
        best_ask = float(asks.loc[0, "price"]) if not asks.empty else float(book_ticker.get("askPrice", np.nan))
        mid_price = (best_bid + best_ask) / 2.0 if (np.isfinite(best_bid) and np.isfinite(best_ask)) else float(self.cfg.get("mid_price", 0.0))
        spread_bps = 10000.0 * _safe_div((best_ask - best_bid), mid_price)  # in basis points

        # Orderbook Imbalance: (bid_qty - ask_qty) / (bid_qty + ask_qty) over a configurable depth
        depth_levels = int(self.cfg.get("depth_levels", 10))
        bid_qty = bids.head(depth_levels)["qty"].sum() if not bids.empty else 0.0
        ask_qty = asks.head(depth_levels)["qty"].sum() if not asks.empty else 0.0
        orderbook_imbalance = _safe_div((bid_qty - ask_qty), (bid_qty + ask_qty))

        # Depth Elasticity: how quickly cumulative volume decays with price: approximate slope of cumulative qty vs price distance
        def depth_elasticity_side(df_side, direction="bid"):
            if df_side.empty:
                return 0.0
            df = df_side.head(self.cfg.get("elasticity_levels", 12)).copy()
            df["dist"] = np.abs(df["price"] - mid_price) / mid_price
            df["cum_qty"] = df["qty"].cumsum()
            # fit simple linear slope cum_qty ~ dist
            if df["dist"].nunique() < 2:
                return 0.0
            coef = np.polyfit(df["dist"].values, df["cum_qty"].values, 1)[0]
            return float(np.abs(coef))
        depth_el_bid = depth_elasticity_side(bids, "bid")
        depth_el_ask = depth_elasticity_side(asks, "ask")
        depth_elasticity = (depth_el_bid + depth_el_ask) / 2.0

        # Liquidity Density Map (summary): total qty within near book window
        near_window = float(self.cfg.get("liquidity_window_bps", 10))  # bps relative to mid
        price_tol = (near_window / 10000.0) * mid_price
        liq_bid = bids[bids["price"] >= (mid_price - price_tol)]["qty"].sum() if not bids.empty else 0.0
        liq_ask = asks[asks["price"] <= (mid_price + price_tol)]["qty"].sum() if not asks.empty else 0.0
        liquidity_density = liq_bid + liq_ask

        # Process trades DataFrame
        trades_df = pd.DataFrame(trades)
        if trades_df.empty:
            # fallback
            trades_df = pd.DataFrame([{"price": mid_price, "qty": 0.0, "isBuyerMaker": False, "timestamp": int(time.time() * 1000)}])

        # CVD - cumulative buy minus sell volume over trades window
        # isBuyerMaker==True indicates that the maker was buyer => taker was seller (trade initiated by seller)
        # We'll define buy_taker = not isBuyerMaker (true if taker bought)
        trades_df["taker_side"] = trades_df["isBuyerMaker"].apply(lambda x: "sell" if x else "buy")
        trades_df["signed_qty"] = trades_df.apply(lambda r: r["qty"] if r["taker_side"] == "buy" else -r["qty"], axis=1)
        cvd = float(trades_df["signed_qty"].sum())

        # Order Flow Imbalance (OFI) -- simple form: sum of signed_size * sign(price change) over trades
        # We'll use price diff between consecutive trades and sign it
        trades_df["price_diff"] = trades_df["price"].diff().fillna(0.0)
        trades_df["price_sign"] = trades_df["price_diff"].apply(lambda x: 1 if x > 0 else (-1 if x < 0 else 0))
        trades_df["ofi_component"] = trades_df["signed_qty"] * trades_df["price_sign"]
        ofi = float(trades_df["ofi_component"].sum())

        # Taker Dominance Ratio: taker buy volume / (taker buy + taker sell)
        taker_buy_vol = trades_df[trades_df["taker_side"] == "buy"]["qty"].sum()
        taker_sell_vol = trades_df[trades_df["taker_side"] == "sell"]["qty"].sum()
        taker_dom_ratio = _safe_div(taker_buy_vol, (taker_buy_vol + taker_sell_vol))

        # Market Buy/Sell Pressure: signed trade qty normalized by total volume
        total_trade_vol = trades_df["qty"].sum()
        market_pressure = _safe_div((taker_buy_vol - taker_sell_vol), total_trade_vol) if total_trade_vol > 0 else 0.0

        # Trade Aggression Ratio: ratio of aggressive trades (taker trades that consume top-of-book)
        # We approximate: trades where price >= best_ask => aggressive buy, price <= best_bid => aggressive sell
        aggressive_buys = trades_df[trades_df["price"] >= best_ask]["qty"].sum()
        aggressive_sells = trades_df[trades_df["price"] <= best_bid]["qty"].sum()
        trade_aggression_ratio = _safe_div((aggressive_buys + aggressive_sells), total_trade_vol) if total_trade_vol > 0 else 0.0

        # Slippage estimate: average execution slippage relative to mid (abs)
        trades_df["slippage_abs_bps"] = 10000.0 * np.abs(trades_df["price"] - mid_price) / mid_price
        slippage = float(trades_df["slippage_abs_bps"].mean()) / 10000.0  # convert back to price fraction

        # Market Buy/Sell Pressure Ratio (market_buy_sell_pressure): normalized between -1..1
        market_buy_sell_pressure = market_pressure  # already -1..1-ish

        # Taker Dominance Ratio is between 0..1

        raw_metrics = {
            "best_bid": best_bid,
            "best_ask": best_ask,
            "mid_price": mid_price,
            "spread_bps": spread_bps,
            "orderbook_imbalance": orderbook_imbalance,
            "depth_elasticity": depth_elasticity,
            "liquidity_density": liquidity_density,
            "cvd": cvd,
            "ofi": ofi,
            "taker_dom_ratio": taker_dom_ratio,
            "market_buy_sell_pressure": market_buy_sell_pressure,
            "trade_aggression_ratio": trade_aggression_ratio,
            "slippage": slippage,
            "bid_qty_top": bid_qty,
            "ask_qty_top": ask_qty,
            "timestamp": int(time.time() * 1000),
        }

        return raw_metrics

    def _normalize_metrics(self, raw: Dict[str, Any]) -> Dict[str, float]:
        """Normalize raw metrics to [0,1] according to normalization config and classical mapping."""
        normed = {}
        # Mapping - for metrics naturally between -1..1, map (x - lo) / (hi - lo)
        for k, (lo, hi) in self.norm.items():
            val = float(raw.get(k, 0.0))
            normed[k] = _min_max_scale(val, lo, hi)
        # Some metrics require flipping: orderbook_imbalance positive means buy-side heavy -> good (higher liquidity pressure score)
        # For metrics where higher is worse (spread, slippage), invert after scaling if config indicates.
        invert = self.cfg.get("invert_metrics", ["spread_bps", "slippage"])
        for m in invert:
            if m in normed:
                normed[m] = 1.0 - normed[m]
        return normed

    def aggregate_output(self, raw_metrics: Dict[str, Any]) -> Dict[str, Any]:
        """
        Combine normalized components into a Liquidity Pressure Score (0-1) and explainable breakdown.
        Uses weights from config.
        """
        weights: Dict[str, float] = self.cfg.get("weights", {
            "orderbook_imbalance": 0.2,
            "spread_bps": 0.15,
            "market_buy_sell_pressure": 0.15,
            "trade_aggression_ratio": 0.1,
            "slippage": 0.1,
            "depth_elasticity": 0.1,
            "cvd": 0.05,
            "ofi": 0.05,
            "taker_dom_ratio": 0.05,
            "liquidity_density": 0.05,
        })

        # Normalize weights to sum 1
        total_w = sum(weights.values())
        if total_w <= 0:
            raise ValueError("Sum of weights must be positive")
        weights = {k: float(v) / total_w for k, v in weights.items()}

        # Ensure raw dict has all keys used for normalization
        # Prepare a mapping from raw metric keys to normalization keys if they differ
        raw_to_norm_key = {
            "orderbook_imbalance": "orderbook_imbalance",
            "spread_bps": "spread_bps",
            "market_buy_sell_pressure": "market_pressure",
            # we store market_buy_sell_pressure raw as 'market_buy_sell_pressure' but normalization defined as 'market_pressure' in defaults.
            # To be robust, set both possibilities below.
        }
        # For simplicity, add keys expected by normalization with values from raw_metrics when possible
        # Build combined raw for normalization
        to_norm_raw = {}
        for k in self.norm.keys():
            # prefer raw_metrics[k], else try common synonyms
            if k in raw_metrics:
                to_norm_raw[k] = raw_metrics[k]
            elif k == "market_pressure" and "market_buy_sell_pressure" in raw_metrics:
                to_norm_raw[k] = raw_metrics["market_buy_sell_pressure"]
            elif k == "trade_aggression" and "trade_aggression_ratio" in raw_metrics:
                to_norm_raw[k] = raw_metrics["trade_aggression_ratio"]
            else:
                to_norm_raw[k] = raw_metrics.get(k, 0.0)

        normalized = self._normalize_metrics(to_norm_raw)

        # Build explain & component contributions
        components = {}
        explain = {}
        for metric, w in weights.items():
            # get normalized value for metric name
            norm_key = metric
            # some weights use keys that map to normalization keys:
            if metric == "market_buy_sell_pressure":
                norm_key = "market_pressure"
            if metric == "trade_aggression_ratio":
                norm_key = "trade_aggression"
            # fallback
            raw_val = raw_metrics.get(metric, to_norm_raw.get(norm_key, 0.0))
            norm_val = normalized.get(norm_key, 0.0)
            contribution = norm_val * w
            components[metric] = norm_val
            explain[metric] = {"raw": raw_val, "normalized": norm_val, "weight": w, "contribution": contribution}

        # Liquidity Pressure Score: sum of contributions, clipped [0,1]
        score_raw = sum(e["contribution"] for e in explain.values())
        score = float(max(0.0, min(1.0, score_raw)))

        # Signal: buy_pressure if orderbook_imbalance positive and taker_dom_ratio high and market_pressure positive
        signal = "neutral"
        if raw_metrics.get("orderbook_imbalance", 0.0) > self.cfg.get("imbalance_signal_thresh", 0.1) and raw_metrics.get("market_buy_sell_pressure", 0.0) > 0.05:
            signal = "buy_pressure"
        elif raw_metrics.get("orderbook_imbalance", 0.0) < -self.cfg.get("imbalance_signal_thresh", 0.1) and raw_metrics.get("market_buy_sell_pressure", 0.0) < -0.05:
            signal = "sell_pressure"

        output = {
            "score": score,
            "signal": signal,
            "components": components,
            "explain": explain,
            "meta": {
                "timestamp": pd.Timestamp.utcnow().isoformat(),
            }
        }
        return output

    # ---- Public run interface ------------------------------------------------
    async def run(self, symbol: str, priority: Optional[int] = None) -> Dict[str, Any]:
        """
        Main entry to compute metrics for `symbol`.
        - Fetches order book, trades, bookTicker from injected data provider.
        - Computes raw metrics, aggregates into score + explain.
        Returns the standardized output dict.
        """
        # Fetch
        order_book = await self.data_provider.get_order_book(symbol)
        trades = await self.data_provider.get_recent_trades(symbol, limit=self.cfg.get("trades_limit", 500))
        book_ticker = await self.data_provider.get_book_ticker(symbol)

        raw = await self.compute_metrics_from_snapshot(order_book, trades, book_ticker)
        out = self.aggregate_output(raw)
        out["meta"].update({"symbol": symbol, "version": self.VERSION})
        return out

    # Backward-compat wrapper for synchronous contexts (keeps interface run(symbol, priority))
    def run_sync(self, symbol: str, priority: Optional[int] = None, timeout: float = 5.0) -> Dict[str, Any]:
        """Synchronous wrapper (for older parts of the system)."""
        return asyncio.get_event_loop().run_until_complete(self.run(symbol, priority))


# If used as standalone script, demonstrate on synthetic data
if __name__ == "__main__":
    import asyncio
    from .config.c_order import CONFIG as default_cfg

    async def demo():
        provider = MockDataProvider(mid_price=123.45, depth_levels=30)
        module = OrderMicroModule(default_cfg, data_provider=provider)
        result = await module.run("BTCUSDT")
        print("=== Liquidity Pressure Module Demo ===")
        print(f"Symbol: {result['meta']['symbol']}")
        print(f"Score: {result['score']:.4f}")
        print(f"Signal: {result['signal']}")
        print("Components:")
        for k, v in result["components"].items():
            print(f"  {k}: {v:.4f} (weight {result['explain'][k]['weight']})")

    asyncio.run(demo())
