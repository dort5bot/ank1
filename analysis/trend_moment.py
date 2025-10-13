# analysis/trend_moment.py
"""
Trend & Momentum Analysis Module
--------------------------------
Metrikler: EMA, RSI, MACD, Bollinger Bands, ATR, ADX, Stochastic RSI, Momentum Oscillator
Profesyonel: Kalman Filter Trend, Z-Score Normalization, Wavelet Transform, Hilbert Transform, FDI
Çıktı: Trend Score (0-1), explainable
Komut: /trend
Paralel İşlem: Batch (CPU-bound)
"""

import asyncio
import logging
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional
from pykalman import KalmanFilter
import pywt
from scipy.signal import hilbert

from analysis.analysis_base_module import BaseAnalysisModule
from analysis.config.trend import CONFIG as trend_config

logger = logging.getLogger(__name__)

class TrendModule(BaseAnalysisModule):
    version = "1.0.0"
    parallel_mode = "batch"

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        config = config or trend_config
        super().__init__(config)
        self.weights = config.get("weights", {
            "ema": 0.2,
            "rsi": 0.2,
            "macd": 0.3,
            "kalman_trend": 0.3
        })
        self.thresholds = config.get("thresholds", {
            "bullish_threshold": 0.7,
            "bearish_threshold": 0.3
        })

    async def compute_metrics(self, symbol: str, priority: Optional[str] = None) -> Dict[str, Any]:
        """
        Ana metrik hesaplama
        """
        df = await self._fetch_ohlcv_data(symbol, interval="1h", limit=100)
        metrics = {}

        # ✅ EMA
        ema_period = self.config.get("ema_period", 21)
        metrics["ema"] = df["close"].ewm(span=ema_period, adjust=False).mean().iloc[-1]

        # ✅ RSI
        delta = df["close"].diff()
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)
        avg_gain = pd.Series(gain).rolling(14).mean().iloc[-1]
        avg_loss = pd.Series(loss).rolling(14).mean().iloc[-1]
        rs = avg_gain / avg_loss if avg_loss != 0 else 0
        metrics["rsi"] = 100 - 100 / (1 + rs)

        # ✅ MACD
        ema12 = df["close"].ewm(span=12, adjust=False).mean()
        ema26 = df["close"].ewm(span=26, adjust=False).mean()
        macd_line = ema12 - ema26
        signal_line = macd_line.ewm(span=9, adjust=False).mean()
        metrics["macd_hist"] = (macd_line - signal_line).iloc[-1]

        # ✅ Kalman Filter Trend
        kf = KalmanFilter(initial_state_mean=df["close"].iloc[0], n_dim_obs=1)
        state_means, _ = kf.filter(df["close"].values)
        metrics["kalman_trend"] = state_means[-1, 0]

        # ✅ Z-Score Normalization (son kapanış)
        metrics["zscore"] = (df["close"].iloc[-1] - df["close"].mean()) / df["close"].std()

        # ✅ Wavelet Transform (son trend component)
        coeffs = pywt.wavedec(df["close"], 'db1', level=3)
        metrics["wavelet_trend"] = coeffs[0][-1]

        # ✅ Hilbert Transform (instantaneous phase)
        analytic_signal = hilbert(df["close"])
        metrics["hilbert_phase"] = np.angle(analytic_signal)[-1]

        # Trend Score Hesaplama (weights ile aggregate)
        score = (
            self.weights.get("ema", 0) * metrics["ema"] / df["close"].iloc[-1] +
            self.weights.get("rsi", 0) * metrics["rsi"] / 100 +
            self.weights.get("macd", 0) * metrics["macd_hist"] / (df["close"].iloc[-1] or 1) +
            self.weights.get("kalman_trend", 0) * metrics["kalman_trend"] / df["close"].iloc[-1]
        )
        score = max(0, min(score, 1))  # 0-1 normalize
        metrics["trend_score"] = score

        # ✅ Signal threshold
        if score > self.thresholds["bullish_threshold"]:
            signal = "bullish"
        elif score < self.thresholds["bearish_threshold"]:
            signal = "bearish"
        else:
            signal = "neutral"
        metrics["signal"] = signal

        # Explainable components
        metrics["explain"] = {
            "components": {
                "ema": metrics["ema"],
                "rsi": metrics["rsi"],
                "macd": metrics["macd_hist"],
                "kalman_trend": metrics["kalman_trend"]
            },
            "signal": signal,
            "score": score
        }

        return metrics

    async def aggregate_output(self, metrics: Dict[str, float], symbol: str) -> Dict[str, Any]:
        """
        Final aggregate output
        """
        return {
            "symbol": symbol,
            "trend_score": metrics.get("trend_score"),
            "signal": metrics.get("signal"),
            "components": metrics.get("explain", {}).get("components", {}),
            "explain": metrics.get("explain")
        }

    def generate_report(self) -> Dict[str, Any]:
        """
        Basit health/report
        """
        return {
            "module": self.module_name,
            "version": self.version,
            "weights": self.weights,
            "thresholds": self.thresholds
        }


# Backward compatibility
async def run(symbol: str, priority: Optional[str] = None):
    module = TrendModule()
    return await module.compute_metrics(symbol, priority)
