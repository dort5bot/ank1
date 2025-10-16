"""
microalpha.py
Micro Alpha Factor Module
Real-time tick-level microstructure analysis for high-frequency alpha generation

Key Metrics:
- Cumulative Volume Delta (CVD)
- Order Flow Imbalance (OFI) 
- Microprice Deviation
- Market Impact Model (Kyle's λ)
- Latency Adjusted Flow Ratio
- High-Frequency Z-score

Output: Micro Alpha Score (0-1)
"""

import numpy as np
import pandas as pd
import asyncio
from typing import Dict, List, Optional, Tuple, Any
import logging
from dataclasses import dataclass
from collections import deque
import time

# Local imports
from analysis.analysis_base_module import BaseAnalysisModule
from utils.binance_api.binance_a import BinanceAggregator
from utils.cache_manager import cache_result
#yok> from analysis.config.c_micro import CONFIG

logger = logging.getLogger(__name__)

@dataclass
class TickData:
    """Individual tick data structure"""
    symbol: str
    timestamp: float
    price: float
    quantity: float
    side: str  # 'buy' or 'sell'
    is_maker: bool


@dataclass
class OrderBookSnapshot:
    """Order book snapshot"""
    symbol: str
    timestamp: float
    bids: List[Tuple[float, float]]  # [(price, quantity)]
    asks: List[Tuple[float, float]]
    spread: float


class MicroAlphaModule(BaseAnalysisModule):
    """
    Microstructure Alpha Factor Generator
    Processes tick-level data to generate real-time alpha signals
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.config = config
        self.module_name = "micro_alpha"
        self.version = "1.0.0"
        
        # Data storage
        self.tick_buffer = deque(maxlen=config["parameters"]["lookback_window"])
        self.order_book_buffer = deque(maxlen=100)
        self.metric_history = deque(maxlen=500)
        
        # State variables
        self.last_microprice = 0.0
        self.cumulative_delta = 0.0
        self.last_trade_side = None
        
        # Initialize Binance client
        self.binance_client = BinanceAggregator()
        
        # Kalman filter state for market impact
        self.kalman_state = config["kalman"]["initial_state"]
        self.kalman_covariance = config["kalman"]["initial_covariance"]
        
        logger.info(f"MicroAlphaModule initialized with config: {config['module_name']}")

    async def initialize(self):
        """Initialize module resources"""
        await self.binance_client.initialize()
        logger.info("MicroAlphaModule initialized successfully")

    async def compute_metrics(self, symbol: str, priority: str = "normal") -> Dict[str, Any]:
        """
        Compute all micro-structure metrics for given symbol
        """
        try:
            # Fetch real-time data
            tick_data, order_book = await self._fetch_realtime_data(symbol)
            
            if not tick_data or not order_book:
                return self._get_default_output(symbol)
            
            # Update buffers
            self._update_buffers(tick_data, order_book)
            
            # Calculate individual metrics
            metrics = await self._calculate_all_metrics(symbol, tick_data, order_book)
            
            # Generate alpha score
            alpha_score, components = self._aggregate_alpha_score(metrics)
            
            # Generate explanation
            explanation = self._generate_explanation(alpha_score, components, metrics)
            
            return {
                "symbol": symbol,
                "score": alpha_score,
                "signal": self._get_signal(alpha_score),
                "components": components,
                "explain": explanation,
                "metrics": metrics,
                "timestamp": time.time(),
                "module": self.module_name,
                "version": self.version
            }
            
        except Exception as e:
            logger.error(f"Error computing micro alpha metrics for {symbol}: {e}")
            return self._get_error_output(symbol, str(e))

    async def _fetch_realtime_data(self, symbol: str) -> Tuple[Optional[TickData], Optional[OrderBookSnapshot]]:
        """
        Fetch real-time tick and order book data
        """
        try:
            # Get recent trades
            trades_data = await self.binance_client.get_recent_trades(symbol=symbol, limit=10)
            # Get order book
            order_book_data = await self.binance_client.get_order_book(symbol=symbol, limit=20)
            
            if not trades_data or not order_book_data:
                return None, None
            
            # Convert to internal format
            latest_trade = trades_data[-1] if trades_data else None
            if latest_trade:
                tick_data = TickData(
                    symbol=symbol,
                    timestamp=latest_trade.get('time', time.time()),
                    price=float(latest_trade['price']),
                    quantity=float(latest_trade['qty']),
                    side='buy' if latest_trade['isBuyerMaker'] else 'sell',
                    is_maker=latest_trade['isBuyerMaker']
                )
            else:
                tick_data = None
            
            # Create order book snapshot
            order_book_snapshot = OrderBookSnapshot(
                symbol=symbol,
                timestamp=time.time(),
                bids=[(float(bid[0]), float(bid[1])) for bid in order_book_data.get('bids', [])],
                asks=[(float(ask[0]), float(ask[1])) for ask in order_book_data.get('asks', [])],
                spread=0.0
            )
            
            # Calculate spread
            if order_book_snapshot.bids and order_book_snapshot.asks:
                best_bid = order_book_snapshot.bids[0][0]
                best_ask = order_book_snapshot.asks[0][0]
                order_book_snapshot.spread = best_ask - best_bid
            
            return tick_data, order_book_snapshot
            
        except Exception as e:
            logger.error(f"Error fetching real-time data for {symbol}: {e}")
            return None, None

    def _update_buffers(self, tick_data: TickData, order_book: OrderBookSnapshot):
        """Update internal data buffers"""
        if tick_data:
            self.tick_buffer.append(tick_data)
        if order_book:
            self.order_book_buffer.append(order_book)

    async def _calculate_all_metrics(self, symbol: str, tick_data: TickData, order_book: OrderBookSnapshot) -> Dict[str, float]:
        """
        Calculate all micro-structure metrics
        """
        metrics = {}
        
        # 1. Cumulative Volume Delta (CVD)
        metrics['cvd'] = self._calculate_cvd()
        
        # 2. Order Flow Imbalance (OFI)
        metrics['ofi'] = self._calculate_order_flow_imbalance(order_book)
        
        # 3. Microprice and Deviation
        microprice, microprice_dev = self._calculate_microprice_deviation(order_book)
        metrics['microprice'] = microprice
        metrics['microprice_deviation'] = microprice_dev
        
        # 4. Market Impact (Kyle's Lambda)
        metrics['market_impact'] = self._calculate_market_impact(tick_data, order_book)
        
        # 5. Latency Adjusted Flow Ratio
        metrics['latency_flow_ratio'] = self._calculate_latency_flow_ratio()
        
        # 6. High-Frequency Z-score
        metrics['hf_zscore'] = self._calculate_hf_zscore(metrics)
        
        return metrics

    def _calculate_cvd(self) -> float:
        """
        Calculate Cumulative Volume Delta
        CVD = Σ(Buy Volume) - Σ(Sell Volume)
        """
        if len(self.tick_buffer) < 2:
            return 0.0
        
        cvd = 0.0
        for tick in list(self.tick_buffer)[-self.config["windows"]["cvd_window"]:]:
            if tick.side == 'buy':
                cvd += tick.quantity
            else:
                cvd -= tick.quantity
        
        # Normalize by recent volume
        total_volume = sum(tick.quantity for tick in list(self.tick_buffer)[-self.config["windows"]["cvd_window"]:])
        if total_volume > 0:
            cvd_normalized = cvd / total_volume
        else:
            cvd_normalized = 0.0
            
        return cvd_normalized

    def _calculate_order_flow_imbalance(self, order_book: OrderBookSnapshot) -> float:
        """
        Calculate Order Flow Imbalance (OFI)
        OFI = (Bid Size - Ask Size) / (Bid Size + Ask Size)
        """
        if not order_book.bids or not order_book.asks:
            return 0.0
        
        # Calculate total sizes at best levels
        bid_size = sum(qty for _, qty in order_book.bids[:5])  # Top 5 levels
        ask_size = sum(qty for _, qty in order_book.asks[:5])
        
        if bid_size + ask_size == 0:
            return 0.0
            
        ofi = (bid_size - ask_size) / (bid_size + ask_size)
        return ofi

    def _calculate_microprice_deviation(self, order_book: OrderBookSnapshot) -> Tuple[float, float]:
        """
        Calculate Microprice and its deviation from current price
        Microprice = (BidSize * AskPrice + AskSize * BidPrice) / (BidSize + AskSize)
        """
        if not order_book.bids or not order_book.asks:
            return 0.0, 0.0
        
        best_bid_price, best_bid_size = order_book.bids[0]
        best_ask_price, best_ask_size = order_book.asks[0]
        
        if best_bid_size + best_ask_size == 0:
            return 0.0, 0.0
        
        # Calculate microprice
        microprice = (best_bid_size * best_ask_price + best_ask_size * best_bid_price) / (best_bid_size + best_ask_size)
        
        # Calculate mid price
        mid_price = (best_bid_price + best_ask_price) / 2
        
        # Calculate deviation from mid price
        if mid_price > 0:
            deviation = (microprice - mid_price) / mid_price
        else:
            deviation = 0.0
            
        self.last_microprice = microprice
        return microprice, deviation

    def _calculate_market_impact(self, tick_data: TickData, order_book: OrderBookSnapshot) -> float:
        """
        Calculate Market Impact using Kalman filter (Kyle's Lambda approximation)
        """
        if not order_book.bids or not order_book.asks:
            return 0.0
        
        # Price change from previous tick
        price_change = 0.0
        if len(self.tick_buffer) >= 2:
            recent_ticks = list(self.tick_buffer)[-2:]
            if len(recent_ticks) == 2:
                price_change = recent_ticks[1].price - recent_ticks[0].price
        
        # Order flow (signed volume)
        order_flow = tick_data.quantity if tick_data.side == 'buy' else -tick_data.quantity
        
        # Kalman filter update for market impact coefficient
        if order_flow != 0:
            # Prediction step
            process_var = self.config["kalman"]["process_variance"]
            self.kalman_covariance += process_var
            
            # Update step
            obs_var = self.config["kalman"]["observation_variance"]
            kalman_gain = self.kalman_covariance / (self.kalman_covariance + obs_var)
            
            # Observation: price_change = lambda * order_flow + noise
            predicted_change = self.kalman_state * order_flow
            innovation = price_change - predicted_change
            
            self.kalman_state += kalman_gain * innovation / order_flow if order_flow != 0 else 0
            self.kalman_covariance *= (1 - kalman_gain)
        
        return abs(self.kalman_state)

    def _calculate_latency_flow_ratio(self) -> float:
        """
        Calculate Latency Adjusted Flow Ratio
        Measures the efficiency of order flow
        """
        if len(self.tick_buffer) < 10:
            return 0.5  # Neutral
        
        recent_ticks = list(self.tick_buffer)[-10:]
        
        # Calculate time between ticks
        timestamps = [tick.timestamp for tick in recent_ticks]
        time_diffs = [timestamps[i] - timestamps[i-1] for i in range(1, len(timestamps))]
        
        if not time_diffs:
            return 0.5
            
        avg_time_diff = np.mean(time_diffs)
        if avg_time_diff == 0:
            return 0.5
            
        # Calculate volume flow
        volumes = [tick.quantity for tick in recent_ticks]
        avg_volume = np.mean(volumes)
        
        # Flow ratio: higher = more efficient flow
        flow_ratio = avg_volume / avg_time_diff if avg_time_diff > 0 else 0
        
        # Normalize to 0-1 range (empirical normalization)
        normalized_ratio = min(flow_ratio / 1000, 1.0)  # Adjust divisor based on typical values
        
        return normalized_ratio

    def _calculate_hf_zscore(self, current_metrics: Dict[str, float]) -> float:
        """
        Calculate High-Frequency Z-score for anomaly detection
        """
        if len(self.metric_history) < self.config["windows"]["zscore_window"]:
            return 0.0
        
        # Use CVD for z-score calculation
        cvd_values = [metric.get('cvd', 0) for metric in list(self.metric_history)[-self.config["windows"]["zscore_window"]:]]
        
        if len(cvd_values) < 2:
            return 0.0
        
        current_cvd = current_metrics.get('cvd', 0)
        mean_cvd = np.mean(cvd_values)
        std_cvd = np.std(cvd_values)
        
        if std_cvd > 0:
            zscore = (current_cvd - mean_cvd) / std_cvd
            # Normalize to 0-1 range using sigmoid
            normalized_zscore = 1 / (1 + np.exp(-zscore))
        else:
            normalized_zscore = 0.5
            
        return normalized_zscore

    def _aggregate_alpha_score(self, metrics: Dict[str, float]) -> Tuple[float, Dict[str, float]]:
        """
        Aggregate individual metrics into final alpha score (0-1)
        """
        weights = self.config["weights"]
        
        # Normalize and weight components
        components = {}
        weighted_sum = 0.0
        total_weight = 0.0
        
        for metric_name, weight in weights.items():
            if metric_name in metrics:
                # Normalize metric value to 0-1 range
                normalized_value = self._normalize_metric(metric_name, metrics[metric_name])
                components[metric_name] = normalized_value
                weighted_sum += normalized_value * weight
                total_weight += weight
        
        # Store metrics for history
        self.metric_history.append(metrics)
        
        if total_weight > 0:
            alpha_score = weighted_sum / total_weight
        else:
            alpha_score = 0.5  # Neutral
            
        return alpha_score, components

    def _normalize_metric(self, metric_name: str, value: float) -> float:
        """
        Normalize metric values to 0-1 range based on metric characteristics
        """
        if metric_name == 'cvd':
            # CVD: -1 to 1 -> 0 to 1
            return (value + 1) / 2
            
        elif metric_name == 'ofi':
            # OFI: -1 to 1 -> 0 to 1  
            return (value + 1) / 2
            
        elif metric_name == 'microprice_deviation':
            # Deviation: typically -0.01 to 0.01 -> 0 to 1
            return min(max((value * 100) + 0.5, 0), 1)
            
        elif metric_name == 'market_impact':
            # Impact: 0 to infinity -> 0 to 1 (saturating)
            return min(value * 1000, 1.0)
            
        elif metric_name == 'latency_flow_ratio':
            # Already normalized
            return value
            
        elif metric_name == 'hf_zscore':
            # Already normalized via sigmoid
            return value
            
        else:
            return min(max(value, 0), 1)

    def _generate_explanation(self, alpha_score: float, components: Dict[str, float], 
                            metrics: Dict[str, float]) -> Dict[str, Any]:
        """
        Generate detailed explanation for the alpha score
        """
        explanation = {
            "summary": "",
            "key_drivers": [],
            "market_regime": "",
            "confidence": 0.0
        }
        
        # Identify key drivers
        sorted_components = sorted(components.items(), key=lambda x: abs(x[1] - 0.5), reverse=True)
        key_drivers = [f"{driver}: {value:.3f}" for driver, value in sorted_components[:3]]
        
        explanation["key_drivers"] = key_drivers
        
        # Determine market regime
        if alpha_score > self.config["thresholds"]["bullish_threshold"]:
            regime = "bullish_microstructure"
            summary = "Strong buying pressure with positive order flow"
        elif alpha_score < self.config["thresholds"]["bearish_threshold"]:
            regime = "bearish_microstructure" 
            summary = "Strong selling pressure with negative order flow"
        else:
            regime = "neutral_microstructure"
            summary = "Balanced order flow with neutral microstructure"
            
        explanation["market_regime"] = regime
        explanation["summary"] = summary
        
        # Calculate confidence based on metric consistency
        confidence = 1.0 - (np.std(list(components.values())) / 0.5)  # Normalize
        explanation["confidence"] = max(0.0, min(1.0, confidence))
        
        return explanation

    def _get_signal(self, alpha_score: float) -> str:
        """Convert alpha score to trading signal"""
        if alpha_score >= self.config["thresholds"]["bullish_threshold"]:
            return "bullish"
        elif alpha_score <= self.config["thresholds"]["bearish_threshold"]:
            return "bearish"
        else:
            return "neutral"

    def _get_default_output(self, symbol: str) -> Dict[str, Any]:
        """Return default output when data is insufficient"""
        return {
            "symbol": symbol,
            "score": 0.5,
            "signal": "neutral",
            "components": {},
            "explain": {"summary": "Insufficient data", "key_drivers": [], "confidence": 0.0},
            "metrics": {},
            "timestamp": time.time(),
            "module": self.module_name,
            "version": self.version
        }

    def _get_error_output(self, symbol: str, error_msg: str) -> Dict[str, Any]:
        """Return error output"""
        return {
            "symbol": symbol,
            "score": 0.5,
            "signal": "neutral",
            "components": {},
            "explain": {"summary": f"Error: {error_msg}", "key_drivers": [], "confidence": 0.0},
            "metrics": {},
            "timestamp": time.time(),
            "module": self.module_name,
            "version": self.version,
            "error": error_msg
        }

    async def cleanup(self):
        """Cleanup resources"""
        await self.binance_client.close()
        logger.info("MicroAlphaModule cleanup completed")


# Factory function for module creation
def create_module(config: Dict[str, Any]) -> MicroAlphaModule:
    """Factory function for creating MicroAlphaModule instances"""
    return MicroAlphaModule(config)