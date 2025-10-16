"""
Trend & Momentum Analysis Module
Version: 1.0.0
deepsek

Purpose: Analyze price direction and momentum strength using classical and advanced technical indicators
Output: Trend Score (0-1) with detailed component breakdown

Metrics:
- Classic: EMA, RSI, MACD, Bollinger Bands, ATR, ADX, Stochastic RSI, Momentum Oscillator
- Advanced: Kalman Filter, Z-Score Normalization, Wavelet Transform, Hilbert Transform, Fractal Dimension

Dependencies: numpy, pandas, scipy, pykalman, pywt
"""

import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Optional, Tuple, Any
import asyncio
from scipy import stats
from scipy.signal import hilbert, detrend
import pywt
from pykalman import KalmanFilter

#from utils.analysis.analysis_base_module import BaseAnalysisModule
from analysis.analysis_base_module import BaseAnalysisModule

from utils.binance_api.binance_a import BinanceAggregator
from utils.cache_manager import cache_result

logger = logging.getLogger(__name__)


class TrendModule(BaseAnalysisModule):
    """
    Trend and Momentum Analysis Module
    
    Computes comprehensive trend score using multiple technical indicators
    and advanced signal processing techniques.
    """
    
    def __init__(self, config: Dict = None):
        super().__init__()
        self.module_name = "trend_moment"
        self.version = "1.0.0"
        
        # Load configuration
        if config is None:
            from config.loader import load_module_config
            self.config = load_module_config("trend_moment")
        else:
            self.config = config
            
        self.weights = self.config.get("weights", {})
        self.thresholds = self.config.get("thresholds", {})
        
        # Initialize state
        self._kalman_filters = {}
        self._cache_ttl = 60  # 1 minute cache
        
        logger.info(f"TrendModule initialized with {len(self.weights)} components")

    async def compute_metrics(self, symbol: str, interval: str = "1h", 
                            lookback: int = None) -> Dict[str, Any]:
        """
        Compute all trend and momentum metrics for given symbol
        
        Args:
            symbol: Trading symbol (e.g., BTCUSDT)
            interval: Kline interval
            lookback: Number of periods to analyze
            
        Returns:
            Dictionary containing all computed metrics and scores
        """
        try:
            # Get OHLCV data
            ohlcv_data = await self._get_ohlcv_data(symbol, interval, lookback)
            if ohlcv_data.empty:
                raise ValueError(f"No data available for {symbol}")
                
            # Compute all metrics
            metrics = await self._compute_all_metrics(ohlcv_data)
            
            # Aggregate scores
            aggregated = self.aggregate_output(metrics)
            
            # Generate explanation
            explanation = self.generate_report(aggregated, metrics)
            
            return {
                **aggregated,
                "explain": explanation,
                "components": metrics,
                "timestamp": pd.Timestamp.now().isoformat(),
                "symbol": symbol,
                "interval": interval
            }
            
        except Exception as e:
            logger.error(f"Error computing metrics for {symbol}: {str(e)}")
            return await self._fallback_computation(symbol)

    @cache_result(ttl=60)
    async def _get_ohlcv_data(self, symbol: str, interval: str, lookback: int = None) -> pd.DataFrame:
        """Get OHLCV data with caching"""
        if lookback is None:
            lookback = self.config.get("window", 100)
            
        binance = BinanceAggregator.get_instance()
        data = await binance.get_klines(symbol, interval, limit=lookback)
        
        if data.empty:
            raise ValueError(f"No OHLCV data for {symbol}")
            
        return data

    async def _compute_all_metrics(self, data: pd.DataFrame) -> Dict[str, float]:
        """Compute all trend and momentum metrics"""
        close_prices = data['close'].values
        high_prices = data['high'].values
        low_prices = data['low'].values
        volumes = data['volume'].values
        
        metrics = {}
        
        # Classic TA Metrics
        metrics.update(self._compute_ema_trend(close_prices))
        metrics.update(self._compute_rsi_momentum(close_prices))
        metrics.update(self._compute_macd_trend(close_prices))
        metrics.update(self._compute_bollinger_trend(close_prices))
        metrics.update(self._compute_atr_volatility(high_prices, low_prices, close_prices))
        metrics.update(self._compute_adx_strength(high_prices, low_prices, close_prices))
        metrics.update(self._compute_stoch_rsi_momentum(close_prices))
        metrics.update(self._compute_momentum_oscillator(close_prices))
        
        # Advanced Metrics
        metrics.update(self._compute_kalman_trend(close_prices))
        metrics.update(self._compute_z_score_normalization(close_prices))
        metrics.update(self._compute_wavelet_trend(close_prices))
        metrics.update(self._compute_hilbert_slope(close_prices))
        metrics.update(self._compute_fdi_complexity(close_prices))
        
        return metrics

    def _compute_ema_trend(self, prices: np.ndarray) -> Dict[str, float]:
        """Compute EMA trend strength"""
        periods = self.config.get("ema_periods", [20, 50, 200])
        ema_values = []
        
        for period in periods:
            if len(prices) >= period:
                ema = self._exponential_moving_average(prices, period)
                # Normalize recent EMA trend (last 5 periods)
                if len(ema) > 5:
                    recent_trend = (ema[-1] - ema[-6]) / ema[-6]
                    ema_values.append(recent_trend)
        
        if ema_values:
            # Score based on EMA alignment and slope
            alignment_score = np.mean(np.sign(ema_values))  # Direction alignment
            slope_score = np.mean(ema_values) / np.std(prices) if np.std(prices) > 0 else 0
            final_score = 0.5 + 0.5 * np.tanh(alignment_score + slope_score)
            return {"ema_trend": float(np.clip(final_score, 0, 1))}
        
        return {"ema_trend": 0.5}

    def _compute_rsi_momentum(self, prices: np.ndarray) -> Dict[str, float]:
        """Compute RSI momentum score"""
        period = self.config.get("rsi_period", 14)
        
        if len(prices) < period + 1:
            return {"rsi_momentum": 0.5}
            
        rsi = self._relative_strength_index(prices, period)
        current_rsi = rsi[-1] if len(rsi) > 0 else 50
        
        # Normalize RSI to 0-1 scale with non-linear mapping
        if current_rsi > 70:
            score = 0.9 + 0.1 * ((current_rsi - 70) / 30)
        elif current_rsi < 30:
            score = 0.1 * (current_rsi / 30)
        else:
            score = 0.5 + (current_rsi - 50) / 40
            
        return {"rsi_momentum": float(np.clip(score, 0, 1))}

    def _compute_macd_trend(self, prices: np.ndarray) -> Dict[str, float]:
        """Compute MACD trend strength"""
        fast = self.config.get("macd_fast", 12)
        slow = self.config.get("macd_slow", 26)
        signal = self.config.get("macd_signal", 9)
        
        if len(prices) < slow + signal:
            return {"macd_trend": 0.5}
            
        macd_line = self._exponential_moving_average(prices, fast) - self._exponential_moving_average(prices, slow)
        signal_line = self._exponential_moving_average(macd_line, signal)
        histogram = macd_line - signal_line
        
        if len(histogram) > 1:
            # Score based on histogram direction and magnitude
            hist_trend = np.mean(np.diff(histogram[-5:])) if len(histogram) >= 5 else 0
            hist_strength = np.abs(histogram[-1]) / (np.std(prices) + 1e-8)
            
            score = 0.5 + 0.5 * np.tanh(hist_trend * 10 + hist_strength * 2)
            return {"macd_trend": float(np.clip(score, 0, 1))}
        
        return {"macd_trend": 0.5}

    def _compute_bollinger_trend(self, prices: np.ndarray) -> Dict[str, float]:
        """Compute Bollinger Bands trend position"""
        period = self.config.get("bollinger_period", 20)
        std_dev = self.config.get("bollinger_std", 2)
        
        if len(prices) < period:
            return {"bollinger_trend": 0.5}
            
        sma = self._simple_moving_average(prices, period)
        rolling_std = pd.Series(prices).rolling(period).std()
        
        upper_band = sma + (rolling_std * std_dev)
        lower_band = sma - (rolling_std * std_dev)
        
        current_price = prices[-1]
        current_sma = sma[-1]
        band_width = upper_band[-1] - lower_band[-1]
        
        if band_width > 0:
            # Position within bands (-1 to 1)
            position = (current_price - current_sma) / (band_width / 2)
            # Squeeze indicator (low volatility -> potential breakout)
            squeeze = np.clip(1 - (band_width / np.mean(band_width[-10:])), 0, 1) if len(band_width) >= 10 else 0
            
            score = 0.5 + 0.3 * position + 0.2 * squeeze
            return {"bollinger_trend": float(np.clip(score, 0, 1))}
        
        return {"bollinger_trend": 0.5}

    def _compute_atr_volatility(self, high: np.ndarray, low: np.ndarray, close: np.ndarray) -> Dict[str, float]:
        """Compute Average True Range volatility adjustment"""
        period = self.config.get("atr_period", 14)
        
        if len(high) < period + 1:
            return {"atr_volatility": 0.5}
            
        tr = np.maximum(high[1:] - low[1:], 
                       np.maximum(np.abs(high[1:] - close[:-1]), 
                                 np.abs(low[1:] - close[:-1])))
        atr = pd.Series(tr).rolling(period).mean().values
        
        if len(atr) > 0 and not np.isnan(atr[-1]):
            # Normalize ATR relative to price
            current_atr = atr[-1]
            price_level = close[-1]
            normalized_atr = current_atr / price_level if price_level > 0 else 0
            
            # High volatility can reduce confidence in trend
            volatility_score = 1 - np.tanh(normalized_atr * 100)
            return {"atr_volatility": float(np.clip(volatility_score, 0, 1))}
        
        return {"atr_volatility": 0.5}

    def _compute_adx_strength(self, high: np.ndarray, low: np.ndarray, close: np.ndarray) -> Dict[str, float]:
        """Compute ADX trend strength"""
        period = self.config.get("adx_period", 14)
        
        if len(high) < period * 2:
            return {"adx_strength": 0.5}
            
        # Simplified ADX calculation
        plus_dm = high[1:] - high[:-1]
        minus_dm = low[:-1] - low[1:]
        
        plus_dm = np.where((plus_dm > minus_dm) & (plus_dm > 0), plus_dm, 0)
        minus_dm = np.where((minus_dm > plus_dm) & (minus_dm > 0), minus_dm, 0)
        
        tr = np.maximum(high[1:] - low[1:], 
                       np.maximum(np.abs(high[1:] - close[:-1]), 
                                 np.abs(low[1:] - close[:-1])))
        
        plus_di = 100 * (pd.Series(plus_dm).rolling(period).mean() / 
                         pd.Series(tr).rolling(period).mean())
        minus_di = 100 * (pd.Series(minus_dm).rolling(period).mean() / 
                          pd.Series(tr).rolling(period).mean())
        
        dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
        adx = dx.rolling(period).mean()
        
        if len(adx) > 0 and not np.isnan(adx.iloc[-1]):
            current_adx = adx.iloc[-1]
            # ADX > 25 indicates strong trend
            score = np.clip(current_adx / 50, 0, 1)
            return {"adx_strength": float(score)}
        
        return {"adx_strength": 0.5}

    def _compute_stoch_rsi_momentum(self, prices: np.ndarray) -> Dict[str, float]:
        """Compute Stochastic RSI momentum"""
        period = self.config.get("stoch_rsi_period", 14)
        smooth = self.config.get("stoch_rsi_smooth", 3)
        
        if len(prices) < period + smooth:
            return {"stoch_rsi_momentum": 0.5}
            
        rsi = self._relative_strength_index(prices, period)
        stoch_rsi = (rsi - pd.Series(rsi).rolling(period).min()) / \
                   (pd.Series(rsi).rolling(period).max() - pd.Series(rsi).rolling(period).min() + 1e-8)
        stoch_rsi_smooth = pd.Series(stoch_rsi).rolling(smooth).mean()
        
        if len(stoch_rsi_smooth) > 0 and not np.isnan(stoch_rsi_smooth.iloc[-1]):
            current_stoch = stoch_rsi_smooth.iloc[-1]
            # Map to momentum score (oversold/overbought regions)
            if current_stoch > 0.8:
                score = 0.9
            elif current_stoch < 0.2:
                score = 0.1
            else:
                score = 0.5 + (current_stoch - 0.5)
                
            return {"stoch_rsi_momentum": float(np.clip(score, 0, 1))}
        
        return {"stoch_rsi_momentum": 0.5}

    def _compute_momentum_oscillator(self, prices: np.ndarray) -> Dict[str, float]:
        """Compute momentum oscillator"""
        period = self.config.get("momentum_period", 10)
        
        if len(prices) < period:
            return {"momentum_oscillator": 0.5}
            
        momentum = (prices[-1] / prices[-period] - 1) * 100
        # Normalize momentum score
        score = 0.5 + 0.5 * np.tanh(momentum / 10)
        return {"momentum_oscillator": float(np.clip(score, 0, 1))}

    def _compute_kalman_trend(self, prices: np.ndarray) -> Dict[str, float]:
        """Compute Kalman Filter trend estimation"""
        try:
            if len(prices) < 10:
                return {"kalman_trend": 0.5}
                
            # Initialize Kalman Filter
            kf = KalmanFilter(
                initial_state_mean=prices[0],
                n_dim_obs=1,
                transition_matrices=[1],
                observation_matrices=[1],
                initial_state_covariance=1,
                observation_covariance=self.config.get("kalman", {}).get("obs_var", 1e-3),
                transition_covariance=self.config.get("kalman", {}).get("process_var", 1e-4)
            )
            
            # Fit and smooth
            state_means, _ = kf.filter(prices)
            state_means_smooth, _ = kf.smooth(prices)
            
            # Calculate trend from smoothed states
            if len(state_means_smooth) > 5:
                recent_trend = np.mean(np.diff(state_means_smooth.flatten()[-5:]))
                price_std = np.std(prices) + 1e-8
                normalized_trend = recent_trend / price_std
                
                score = 0.5 + 0.5 * np.tanh(normalized_trend * 10)
                return {"kalman_trend": float(np.clip(score, 0, 1))}
                
        except Exception as e:
            logger.warning(f"Kalman filter failed: {str(e)}")
            
        return {"kalman_trend": 0.5}

    def _compute_z_score_normalization(self, prices: np.ndarray) -> Dict[str, float]:
        """Compute Z-Score based trend normalization"""
        window = self.config.get("z_score_window", 21)
        
        if len(prices) < window:
            return {"z_score_normalization": 0.5}
            
        recent_prices = prices[-window:]
        z_scores = stats.zscore(recent_prices)
        current_z = z_scores[-1] if len(z_scores) > 0 else 0
        
        # Convert Z-score to trend probability
        score = stats.norm.cdf(current_z)
        return {"z_score_normalization": float(score)}

    def _compute_wavelet_trend(self, prices: np.ndarray) -> Dict[str, float]:
        """Compute Wavelet transform trend analysis"""
        try:
            if len(prices) < 32:  # Minimum length for meaningful wavelet analysis
                return {"wavelet_trend": 0.5}
                
            # Detrend the data first
            detrended = detrend(prices)
            
            # Perform wavelet decomposition
            wavelet = self.config.get("wavelet_family", "db4")
            level = self.config.get("wavelet_level", 3)
            
            coeffs = pywt.wavedec(detrended, wavelet, level=level)
            
            # Use approximation coefficients for trend
            approx_coeffs = coeffs[0]
            
            if len(approx_coeffs) > 1:
                trend_slope = np.polyfit(range(len(approx_coeffs)), approx_coeffs, 1)[0]
                normalized_slope = trend_slope / (np.std(approx_coeffs) + 1e-8)
                
                score = 0.5 + 0.5 * np.tanh(normalized_slope)
                return {"wavelet_trend": float(np.clip(score, 0, 1))}
                
        except Exception as e:
            logger.warning(f"Wavelet transform failed: {str(e)}")
            
        return {"wavelet_trend": 0.5}

    def _compute_hilbert_slope(self, prices: np.ndarray) -> Dict[str, float]:
        """Compute Hilbert transform instantaneous trend"""
        try:
            window = self.config.get("hilbert_window", 10)
            
            if len(prices) < window * 2:
                return {"hilbert_slope": 0.5}
                
            # Apply Hilbert transform
            analytic_signal = hilbert(prices)
            instantaneous_phase = np.unwrap(np.angle(analytic_signal))
            instantaneous_freq = np.diff(instantaneous_phase) / (2 * np.pi)
            
            if len(instantaneous_freq) >= window:
                recent_freq = instantaneous_freq[-window:]
                freq_trend = np.polyfit(range(len(recent_freq)), recent_freq, 1)[0]
                
                score = 0.5 + 0.5 * np.tanh(freq_trend * 100)
                return {"hilbert_slope": float(np.clip(score, 0, 1))}
                
        except Exception as e:
            logger.warning(f"Hilbert transform failed: {str(e)}")
            
        return {"hilbert_slope": 0.5}

    def _compute_fdi_complexity(self, prices: np.ndarray) -> Dict[str, float]:
        """Compute Fractal Dimension Index for market complexity"""
        try:
            window = self.config.get("fdi_window", 10)
            
            if len(prices) < window * 2:
                return {"fdi_complexity": 0.5}
                
            # Simplified FDI calculation using Hurst exponent approximation
            n = len(prices)
            max_lag = min(20, n // 4)
            
            lags = range(2, max_lag)
            tau = [np.std(np.subtract(prices[lag:], prices[:-lag])) for lag in lags]
            lag = np.log(lags)
            tau = np.log(tau)
            
            if len(lag) > 1 and len(tau) > 1:
                hurst = np.polyfit(lag, tau, 1)[0]
                # H ~ 0.5: random walk, H > 0.5: trending, H < 0.5: mean-reverting
                fdi_score = hurst  # Direct use as complexity measure
                score = 0.5 + (fdi_score - 0.5)  # Center around 0.5
                return {"fdi_complexity": float(np.clip(score, 0, 1))}
                
        except Exception as e:
            logger.warning(f"FDI calculation failed: {str(e)}")
            
        return {"fdi_complexity": 0.5}

    def aggregate_output(self, metrics: Dict[str, float]) -> Dict[str, Any]:
        """
        Aggregate individual metrics into final trend score
        
        Args:
            metrics: Dictionary of computed metric scores
            
        Returns:
            Aggregated output with final score and signal
        """
        # Apply weights to each component
        weighted_scores = []
        valid_components = {}
        
        for metric_name, weight in self.weights.items():
            if metric_name in metrics and weight > 0:
                component_score = metrics[metric_name]
                weighted_scores.append(component_score * weight)
                valid_components[metric_name] = {
                    "score": component_score,
                    "weight": weight,
                    "contribution": component_score * weight
                }
        
        if weighted_scores:
            final_score = np.sum(weighted_scores)
        else:
            final_score = 0.5
            
        # Determine trend signal
        bullish_threshold = self.thresholds.get("bullish", 0.7)
        bearish_threshold = self.thresholds.get("bearish", 0.3)
        
        if final_score >= bullish_threshold:
            signal = "bullish"
        elif final_score <= bearish_threshold:
            signal = "bearish"
        else:
            signal = "neutral"
            
        # Trend strength classification
        strong_threshold = self.thresholds.get("strong_trend", 0.6)
        weak_threshold = self.thresholds.get("weak_trend", 0.4)
        
        if final_score >= strong_threshold or final_score <= (1 - strong_threshold):
            strength = "strong"
        elif final_score >= weak_threshold or final_score <= (1 - weak_threshold):
            strength = "moderate"
        else:
            strength = "weak"
            
        return {
            "score": float(np.clip(final_score, 0, 1)),
            "signal": signal,
            "strength": strength,
            "components": valid_components,
            "timestamp": pd.Timestamp.now().isoformat()
        }

    def generate_report(self, aggregated: Dict, metrics: Dict) -> Dict[str, Any]:
        """
        Generate detailed explanation for the trend analysis
        
        Args:
            aggregated: Aggregated output from aggregate_output
            metrics: Raw metric scores
            
        Returns:
            Detailed explanation dictionary
        """
        score = aggregated["score"]
        signal = aggregated["signal"]
        strength = aggregated["strength"]
        
        # Key contributors
        components = aggregated.get("components", {})
        top_contributors = sorted(
            [(name, data["contribution"]) for name, data in components.items()],
            key=lambda x: abs(x[1]),
            reverse=True
        )[:3]
        
        explanation = {
            "summary": f"Trend analysis indicates {strength} {signal} bias",
            "confidence": min(score, 1 - score) * 2,  # Distance from 0.5
            "key_metrics": {
                name: {
                    "score": components[name]["score"],
                    "contribution": contrib
                }
                for name, contrib in top_contributors
            },
            "interpretation": self._interpret_trend_score(score, strength),
            "recommendation": self._generate_recommendation(score, signal, strength)
        }
        
        return explanation

    def _interpret_trend_score(self, score: float, strength: str) -> str:
        """Generate human-readable interpretation of trend score"""
        if score >= 0.8:
            return "Very strong upward momentum with clear bullish trend"
        elif score >= 0.7:
            return "Strong upward trend with positive momentum"
        elif score >= 0.6:
            return "Moderate upward bias with developing trend"
        elif score >= 0.4:
            return "Neutral to slightly directional, awaiting clearer trend"
        elif score >= 0.3:
            return "Moderate downward bias with developing trend"
        elif score >= 0.2:
            return "Strong downward trend with negative momentum"
        else:
            return "Very strong downward momentum with clear bearish trend"

    def _generate_recommendation(self, score: float, signal: str, strength: str) -> str:
        """Generate trading recommendation based on trend analysis"""
        if signal == "bullish" and strength == "strong":
            return "Consider long positions with tight stop-loss"
        elif signal == "bullish" and strength == "moderate":
            return "Potential long opportunities, monitor for confirmation"
        elif signal == "bearish" and strength == "strong":
            return "Consider short positions or reducing long exposure"
        elif signal == "bearish" and strength == "moderate":
            return "Potential short opportunities, await confirmation"
        else:
            return "Wait for clearer trend direction before taking positions"

    async def _fallback_computation(self, symbol: str) -> Dict[str, Any]:
        """Fallback computation when primary method fails"""
        logger.warning(f"Using fallback computation for {symbol}")
        
        return {
            "score": 0.5,
            "signal": "neutral",
            "strength": "weak",
            "components": {},
            "explain": {
                "summary": "Fallback analysis: insufficient data or computation error",
                "confidence": 0.1,
                "key_metrics": {},
                "interpretation": "Unable to determine clear trend direction",
                "recommendation": "Wait for more data or verify symbol"
            },
            "timestamp": pd.Timestamp.now().isoformat(),
            "symbol": symbol
        }

    # Technical indicator helper methods
    def _exponential_moving_average(self, prices: np.ndarray, period: int) -> np.ndarray:
        """Calculate Exponential Moving Average"""
        return pd.Series(prices).ewm(span=period, adjust=False).mean().values

    def _simple_moving_average(self, prices: np.ndarray, period: int) -> np.ndarray:
        """Calculate Simple Moving Average"""
        return pd.Series(prices).rolling(period).mean().values

    def _relative_strength_index(self, prices: np.ndarray, period: int) -> np.ndarray:
        """Calculate Relative Strength Index"""
        delta = pd.Series(prices).diff()
        gain = (delta.where(delta > 0, 0)).rolling(period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
        rs = gain / (loss + 1e-8)
        rsi = 100 - (100 / (1 + rs))
        return rsi.values

    async def run(self, symbol: str, priority: str = "normal") -> Dict[str, Any]:
        """
        Main execution method for backward compatibility
        
        Args:
            symbol: Trading symbol to analyze
            priority: Execution priority
            
        Returns:
            Analysis results
        """
        return await self.compute_metrics(symbol)

    def get_metadata(self) -> Dict[str, Any]:
        """Return module metadata"""
        return {
            "module_name": self.module_name,
            "version": self.version,
            "description": "Trend direction and momentum strength analysis",
            "metrics": list(self.weights.keys()),
            "parallel_mode": "batch",
            "lifecycle": "development"
        }


# Factory function for module creation
def create_trend_module(config: Dict = None) -> TrendModule:
    """
    Factory function to create TrendModule instance
    
    Args:
        config: Optional configuration dictionary
        
    Returns:
        TrendModule instance
    """
    return TrendModule(config)