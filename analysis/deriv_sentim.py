# analysis/deriv_sentim.py
"""
Derivatives & Sentiment Analysis Module
Futures piyasası pozisyon verilerine dayalı trader sentiment analizi

Metrikler:
- Temel: Funding Rate, Open Interest, Long/Short Ratio
- Gelişmiş: OI Change Rate, Funding Rate Skew, Volume Imbalance Index
- Profesyonel: Liquidation Heatmap, OI Delta Divergence, Volatility Skew

Amaç: Trader positioning & sentiment eğilimi
Çıktı: Sentiment Score (-1 → 1)
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
import asyncio
from dataclasses import dataclass
import logging

from analysis.analysis_base_module import BaseAnalysisModule
from utils.binance_api.binance_a import BinanceAggregator
#from analysis.config.c_deriv import CONFIG

logger = logging.getLogger(__name__)

@dataclass
class SentimentComponents:
    """Sentiment skor bileşenleri"""
    funding_rate: float
    open_interest: float
    long_short_ratio: float
    oi_change_rate: float
    funding_skew: float
    volume_imbalance: float
    liquidation_heat: float
    oi_delta_divergence: float
    volatility_skew: float

class DerivativesSentimentModule(BaseAnalysisModule):
    """
    Futures pozisyon verilerine dayalı sentiment analiz modülü
    """
    
    def __init__(self, config: Optional[Dict] = None):
        super().__init__()
        self.module_name = "derivatives_sentiment"
        self.version = "1.0.0"
        
        # Varsayılan config
        self.default_config = {
            "weights": {
                "funding_rate": 0.15,
                "open_interest": 0.12,
                "long_short_ratio": 0.13,
                "oi_change_rate": 0.12,
                "funding_skew": 0.10,
                "volume_imbalance": 0.10,
                "liquidation_heat": 0.15,
                "oi_delta_divergence": 0.08,
                "volatility_skew": 0.05
            },
            "thresholds": {
                "bullish": 0.6,
                "bearish": 0.4,
                "extreme_bull": 0.8,
                "extreme_bear": 0.2
            },
            "parameters": {
                "oi_lookback": 24,  # saat
                "funding_lookback": 8,
                "liquidation_window": 12,
                "volatility_period": 20
            },
            "normalization": {
                "method": "zscore",
                "rolling_window": 100
            }
        }
        
        # Config merge
        self.config = {**self.default_config, **(config or {})}
        self.binance = BinanceAggregator()
        
        # Cache için
        self._cache = {}
        self._cache_ttl = 60  # saniye
        
    async def compute_metrics(self, symbol: str, **kwargs) -> Dict:
        """
        Tüm sentiment metriklerini hesapla
        
        Args:
            symbol: İşlem çifti (Ör: BTCUSDT)
            
        Returns:
            Dict: Tüm metrikler ve sentiment skoru
        """
        try:
            # Verileri paralel olarak getir
            tasks = [
                self._get_funding_data(symbol),
                self._get_open_interest_data(symbol),
                self._get_long_short_data(symbol),
                self._get_liquidation_data(symbol),
                self._get_taker_ratio_data(symbol)
            ]
            
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Hata kontrolü
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    logger.error(f"Error in data fetch task {i}: {result}")
                    return await self._fallback_sentiment(symbol)
            
            funding_data, oi_data, ls_data, liq_data, taker_data = results
            
            # Metrikleri hesapla
            components = await self._calculate_components(
                symbol, funding_data, oi_data, ls_data, liq_data, taker_data
            )
            
            # Sentiment skoru oluştur
            score_result = await self._compute_sentiment_score(components)
            
            return {
                **score_result,
                "symbol": symbol,
                "module": self.module_name,
                "timestamp": pd.Timestamp.now().isoformat(),
                "metadata": {
                    "oi_lookback": self.config["parameters"]["oi_lookback"],
                    "funding_lookback": self.config["parameters"]["funding_lookback"]
                }
            }
            
        except Exception as e:
            logger.error(f"Error computing sentiment metrics for {symbol}: {e}")
            return await self._fallback_sentiment(symbol)
    
    async def _calculate_components(self, symbol: str, funding_data: pd.DataFrame, 
                                  oi_data: pd.DataFrame, ls_data: pd.DataFrame,
                                  liq_data: pd.DataFrame, taker_data: pd.DataFrame) -> SentimentComponents:
        """Tüm sentiment bileşenlerini hesapla"""
        
        # Temel metrikler
        funding_rate = await self._calculate_funding_sentiment(funding_data)
        open_interest = await self._calculate_oi_sentiment(oi_data)
        long_short_ratio = await self._calculate_ls_sentiment(ls_data)
        
        # Gelişmiş metrikler
        oi_change_rate = await self._calculate_oi_change_rate(oi_data)
        funding_skew = await self._calculate_funding_skew(funding_data)
        volume_imbalance = await self._calculate_volume_imbalance(taker_data)
        
        # Profesyonel metrikler
        liquidation_heat = await self._calculate_liquidation_heat(liq_data)
        oi_delta_divergence = await self._calculate_oi_delta_divergence(oi_data, funding_data)
        volatility_skew = await self._calculate_volatility_skew(symbol, oi_data)
        
        return SentimentComponents(
            funding_rate=funding_rate,
            open_interest=open_interest,
            long_short_ratio=long_short_ratio,
            oi_change_rate=oi_change_rate,
            funding_skew=funding_skew,
            volume_imbalance=volume_imbalance,
            liquidation_heat=liquidation_heat,
            oi_delta_divergence=oi_delta_divergence,
            volatility_skew=volatility_skew
        )
    
    async def _compute_sentiment_score(self, components: SentimentComponents) -> Dict:
        """
        Bileşenleri ağırlıklandırarak sentiment skoru oluştur
        
        Returns:
            Dict: Skor, sinyal ve açıklamalar
        """
        weights = self.config["weights"]
        
        # Ağırlıklı ortalama
        weighted_sum = (
            components.funding_rate * weights["funding_rate"] +
            components.open_interest * weights["open_interest"] +
            components.long_short_ratio * weights["long_short_ratio"] +
            components.oi_change_rate * weights["oi_change_rate"] +
            components.funding_skew * weights["funding_skew"] +
            components.volume_imbalance * weights["volume_imbalance"] +
            components.liquidation_heat * weights["liquidation_heat"] +
            components.oi_delta_divergence * weights["oi_delta_divergence"] +
            components.volatility_skew * weights["volatility_skew"]
        )
        
        # Normalizasyon (-1 ile 1 arası)
        sentiment_score = np.tanh(weighted_sum * 3)  # tanh ile sınırlama
        
        # Sinyal belirleme
        signal = self._get_sentiment_signal(sentiment_score)
        
        return {
            "score": float(sentiment_score),
            "signal": signal,
            "components": {
                "funding_rate": float(components.funding_rate),
                "open_interest": float(components.open_interest),
                "long_short_ratio": float(components.long_short_ratio),
                "oi_change_rate": float(components.oi_change_rate),
                "funding_skew": float(components.funding_skew),
                "volume_imbalance": float(components.volume_imbalance),
                "liquidation_heat": float(components.liquidation_heat),
                "oi_delta_divergence": float(components.oi_delta_divergence),
                "volatility_skew": float(components.volatility_skew)
            },
            "explain": self._generate_explanation(sentiment_score, components, signal)
        }
    
    def _get_sentiment_signal(self, score: float) -> str:
        """Skora göre sinyal belirle"""
        thresholds = self.config["thresholds"]
        
        if score >= thresholds["extreme_bull"]:
            return "extreme_bullish"
        elif score >= thresholds["bullish"]:
            return "bullish"
        elif score <= thresholds["extreme_bear"]:
            return "extreme_bearish"
        elif score <= thresholds["bearish"]:
            return "bearish"
        else:
            return "neutral"
    
    def _generate_explanation(self, score: float, components: SentimentComponents, signal: str) -> str:
        """Sentiment skoru için açıklama oluştur"""
        
        explanations = []
        
        # Funding rate analizi
        if abs(components.funding_rate) > 0.3:
            direction = "pozitif" if components.funding_rate > 0 else "negatif"
            explanations.append(f"Funding rate {direction} bölgede")
        
        # Open Interest
        if components.open_interest > 0.2:
            explanations.append("Open Interest artışı")
        elif components.open_interest < -0.2:
            explanations.append("Open Interest düşüşü")
        
        # Long/Short Ratio
        if components.long_short_ratio > 0.25:
            explanations.append("Long pozisyonlar hakim")
        elif components.long_short_ratio < -0.25:
            explanations.append("Short pozisyonlar hakim")
        
        # Liquidation heat
        if components.liquidation_heat > 0.3:
            explanations.append("Yüksek likidasyon riski")
        
        if not explanations:
            explanations.append("Piyasa dengeli seyirde")
        
        return f"{signal.upper()} - " + ". ".join(explanations)
    
    # METRİK HESAPLAMA FONKSİYONLARI
    
    async def _calculate_funding_sentiment(self, funding_data: pd.DataFrame) -> float:
        """Funding rate sentiment hesapla"""
        if funding_data.empty:
            return 0.0
        
        current_funding = funding_data['fundingRate'].iloc[-1]
        avg_funding = funding_data['fundingRate'].mean()
        
        # Normalize edilmiş funding sentiment
        funding_sentiment = np.tanh((current_funding - avg_funding) * 1000)
        return float(funding_sentiment)
    
    async def _calculate_oi_sentiment(self, oi_data: pd.DataFrame) -> float:
        """Open Interest sentiment hesapla"""
        if len(oi_data) < 2:
            return 0.0
        
        current_oi = oi_data['sumOpenInterest'].iloc[-1]
        prev_oi = oi_data['sumOpenInterest'].iloc[-2]
        
        # OI değişim oranı
        oi_change = (current_oi - prev_oi) / prev_oi if prev_oi != 0 else 0
        return float(np.tanh(oi_change * 10))
    
    async def _calculate_ls_sentiment(self, ls_data: pd.DataFrame) -> float:
        """Long/Short Ratio sentiment hesapla"""
        if ls_data.empty:
            return 0.0
        
        current_ratio = ls_data['longShortRatio'].iloc[-1]
        
        # 1.0 nötr seviye, üstü long hakimiyeti
        ls_sentiment = np.tanh((current_ratio - 1.0) * 2)
        return float(ls_sentiment)
    
    async def _calculate_oi_change_rate(self, oi_data: pd.DataFrame) -> float:
        """OI değişim hızı (momentum)"""
        if len(oi_data) < 5:
            return 0.0
        
        oi_series = oi_data['sumOpenInterest']
        returns = oi_series.pct_change().dropna()
        
        if len(returns) < 2:
            return 0.0
        
        # Son 5 periyodun momentumu
        momentum = returns.rolling(5).mean().iloc[-1]
        return float(np.tanh(momentum * 100))
    
    async def _calculate_funding_skew(self, funding_data: pd.DataFrame) -> float:
        """Funding rate skew (dağılım çarpıklığı)"""
        if len(funding_data) < 10:
            return 0.0
        
        funding_rates = funding_data['fundingRate']
        skew = funding_rates.skew()
        return float(np.tanh(skew))
    
    async def _calculate_volume_imbalance(self, taker_data: pd.DataFrame) -> float:
        """Volume imbalance index"""
        if taker_data.empty:
            return 0.0
        
        buy_vol = taker_data['buyVol'].iloc[-1]
        sell_vol = taker_data['sellVol'].iloc[-1]
        total_vol = buy_vol + sell_vol
        
        if total_vol == 0:
            return 0.0
        
        imbalance = (buy_vol - sell_vol) / total_vol
        return float(imbalance)
    
    async def _calculate_liquidation_heat(self, liq_data: pd.DataFrame) -> float:
        """Liquidation heatmap metriği"""
        if liq_data.empty:
            return 0.0
        
        # Son 12 saatlik likidasyon toplamı
        recent_liq = liq_data.head(12)  # En son 12 kayıt
        total_liq = recent_liq['executedQty'].sum()
        
        # Normalize edilmiş liquidation heat
        heat = np.tanh(total_liq / 1e6)  # 1M USDT için normalize
        return float(heat)
    
    async def _calculate_oi_delta_divergence(self, oi_data: pd.DataFrame, 
                                           funding_data: pd.DataFrame) -> float:
        """OI delta divergence (OI vs Funding divergence)"""
        if len(oi_data) < 5 or len(funding_data) < 5:
            return 0.0
        
        # OI momentum
        oi_momentum = oi_data['sumOpenInterest'].pct_change(3).iloc[-1]
        
        # Funding momentum
        funding_momentum = funding_data['fundingRate'].diff(3).iloc[-1]
        
        # Divergence (zıt yönlü hareket)
        divergence = oi_momentum * funding_momentum * -100  # Negatif correlation beklenir
        
        return float(np.tanh(divergence))
    
    async def _calculate_volatility_skew(self, symbol: str, oi_data: pd.DataFrame) -> float:
        """Volatility skew (OI dağılım volatilitesi)"""
        if len(oi_data) < 20:
            return 0.0
        
        oi_returns = oi_data['sumOpenInterest'].pct_change().dropna()
        
        if len(oi_returns) < 10:
            return 0.0
        
        # OI volatilitesi (realized vol)
        volatility = oi_returns.std() * np.sqrt(365 * 24)  # Yıllıklaştırılmış
        
        # Normalize edilmiş skew
        skew = np.tanh(volatility * 10)
        return float(skew)
    
    # DATA FETCH FONKSİYONLARI
    
    async def _get_funding_data(self, symbol: str) -> pd.DataFrame:
        """Funding rate verilerini getir"""
        cache_key = f"funding_{symbol}"
        if cache_key in self._cache:
            return self._cache[cache_key]
        
        try:
            data = await self.binance.get_funding_rate(symbol=symbol, limit=50)
            df = pd.DataFrame(data)
            df['fundingTime'] = pd.to_datetime(df['fundingTime'], unit='ms')
            df.set_index('fundingTime', inplace=True)
            
            self._cache[cache_key] = df
            return df
        except Exception as e:
            logger.warning(f"Failed to fetch funding data for {symbol}: {e}")
            return pd.DataFrame()
    
    async def _get_open_interest_data(self, symbol: str) -> pd.DataFrame:
        """Open Interest verilerini getir"""
        cache_key = f"oi_{symbol}"
        if cache_key in self._cache:
            return self._cache[cache_key]
        
        try:
            data = await self.binance.get_open_interest_hist(
                symbol=symbol, 
                period='5m', 
                limit=100
            )
            df = pd.DataFrame(data)
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            
            self._cache[cache_key] = df
            return df
        except Exception as e:
            logger.warning(f"Failed to fetch OI data for {symbol}: {e}")
            return pd.DataFrame()
    
    async def _get_long_short_data(self, symbol: str) -> pd.DataFrame:
        """Long/Short Ratio verilerini getir"""
        cache_key = f"ls_{symbol}"
        if cache_key in self._cache:
            return self._cache[cache_key]
        
        try:
            data = await self.binance.get_long_short_ratio(
                symbol=symbol,
                period='5m',
                limit=100
            )
            df = pd.DataFrame(data)
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            
            self._cache[cache_key] = df
            return df
        except Exception as e:
            logger.warning(f"Failed to fetch LS ratio for {symbol}: {e}")
            return pd.DataFrame()
    
    async def _get_liquidation_data(self, symbol: str) -> pd.DataFrame:
        """Liquidation verilerini getir"""
        cache_key = f"liq_{symbol}"
        if cache_key in self._cache:
            return self._cache[cache_key]
        
        try:
            data = await self.binance.get_liquidation_orders(
                symbol=symbol,
                limit=100
            )
            df = pd.DataFrame(data)
            if not df.empty:
                df['time'] = pd.to_datetime(df['time'], unit='ms')
                df.set_index('time', inplace=True)
                df.sort_index(ascending=False, inplace=True)  # En yeni başta
            
            self._cache[cache_key] = df
            return df
        except Exception as e:
            logger.warning(f"Failed to fetch liquidation data for {symbol}: {e}")
            return pd.DataFrame()
    
    async def _get_taker_ratio_data(self, symbol: str) -> pd.DataFrame:
        """Taker buy/sell volume verilerini getir"""
        cache_key = f"taker_{symbol}"
        if cache_key in self._cache:
            return self._cache[cache_key]
        
        try:
            data = await self.binance.get_taker_long_short_ratio(
                symbol=symbol,
                period='5m',
                limit=100
            )
            df = pd.DataFrame(data)
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            
            self._cache[cache_key] = df
            return df
        except Exception as e:
            logger.warning(f"Failed to fetch taker ratio for {symbol}: {e}")
            return pd.DataFrame()
    
    async def _fallback_sentiment(self, symbol: str) -> Dict:
        """Fallback sentiment response"""
        return {
            "score": 0.0,
            "signal": "neutral",
            "components": {},
            "explain": "Veri alınamadı - nötr sentiment uygulandı",
            "symbol": symbol,
            "module": self.module_name,
            "timestamp": pd.Timestamp.now().isoformat(),
            "metadata": {"fallback": True}
        }
    
    async def aggregate_output(self, results: List[Dict]) -> Dict:
        """Çoklu sembol çıktılarını aggregate et"""
        if not results:
            return {"overall_sentiment": 0.0, "market_bias": "neutral"}
        
        scores = [r.get("score", 0) for r in results]
        avg_score = np.mean(scores)
        
        # Piyasa geneli bias
        if avg_score > 0.1:
            bias = "bullish"
        elif avg_score < -0.1:
            bias = "bearish"
        else:
            bias = "neutral"
        
        return {
            "overall_sentiment": float(avg_score),
            "market_bias": bias,
            "symbol_count": len(results),
            "symbol_scores": {r["symbol"]: r["score"] for r in results}
        }
    
    async def generate_report(self, results: Dict) -> str:
        """Sentiment raporu oluştur"""
        score = results.get("score", 0)
        signal = results.get("signal", "neutral")
        explain = results.get("explain", "")
        
        return f"""
DERIVATIVES SENTIMENT REPORT
============================
Score: {score:.3f} ({signal})
Signal: {signal.upper()}
Explanation: {explain}

Components:
{self._format_components(results.get('components', {}))}
        """
    
    def _format_components(self, components: Dict) -> str:
        """Bileşenleri formatla"""
        if not components:
            return "  No component data available"
        
        lines = []
        for key, value in components.items():
            lines.append(f"  {key.replace('_', ' ').title():<25}: {value:>7.3f}")
        
        return "\n".join(lines)
    
    def cleanup(self):
        """Kaynakları temizle"""
        self._cache.clear()
    
    @property
    def parallel_mode(self) -> str:
        return "async"
    
    @property
    def job_type(self) -> str:
        return "io_bound"